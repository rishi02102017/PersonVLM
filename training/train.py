"""
Training Pipeline for Person VLM
================================
Supports both single-GPU and multi-GPU (DDP) training.

Multi-GPU Training:
    Uses DistributedDataParallel (DDP) for efficient multi-GPU training.
    Launch with: torchrun --nproc_per_node=4 scripts/train.py ...

Single-GPU Training:
    Falls back automatically when only one GPU is available or DDP not initialized.
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

from .utils import (
    set_seed,
    get_optimizer,
    get_scheduler,
    AverageMeter,
    EarlyStopping,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
)


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize only if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    train_file: str = ""
    val_file: Optional[str] = None
    
    # Model
    vision_backbone: str = "mobilevit_xs"
    decoder_size: str = "large"  # Updated default to large
    vision_freeze_ratio: float = 0.9
    
    # Training (Linear Scaling Rule: LR scales with batch size)
    epochs: int = 75  # Larger models need more epochs to converge
    batch_size: int = 64  # Per-GPU batch size (total = batch_size * num_gpus)
    learning_rate: float = 2e-4  # Scaled up per Linear Scaling Rule
    weight_decay: float = 0.005  # Reduced for larger model
    warmup_ratio: float = 0.05  # Shorter warmup to reach peak LR faster
    
    # Optimizer & Scheduler
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Mixed precision (Tensor Cores on V100)
    use_amp: bool = True
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # Data loading
    num_workers: int = 8  # More workers for faster data loading
    image_size: int = 224
    max_seq_length: int = 256  # Accommodate MSP60k detailed captions
    
    # Checkpointing
    output_dir: str = "./checkpoints_scaled"
    save_every: int = 1  # Save every N epochs
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15  # Much more patience - larger models converge slowly
    
    # Logging
    log_every: int = 50  # Log every N steps
    eval_every: int = 500  # Eval every N steps
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from: Optional[str] = None
    
    # Multi-GPU settings
    distributed: bool = False  # Auto-detected at runtime
    local_rank: int = 0
    world_size: int = 1


class Trainer:
    """
    Trainer for Person VLM.
    
    Handles:
    - Training loop with mixed precision
    - Multi-GPU training with DistributedDataParallel (DDP)
    - Validation
    - Checkpointing
    - Early stopping
    - Logging
    
    Hardware Support:
    - CUDA (NVIDIA GPUs): Full support with Tensor Cores, DDP
    - MPS (Apple Silicon): Single GPU, limited AMP support
    - CPU: Fallback option
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        vocab,
        config: TrainingConfig,
    ):
        self.config = config
        self.vocab = vocab
        
        # Setup distributed training
        self.distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main = is_main_process()
        
        if self.distributed:
            config.distributed = True
            config.local_rank = self.local_rank
            config.world_size = self.world_size
        
        # Setup output directory (only on main process)
        self.output_dir = Path(config.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Save config
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(asdict(config), f, indent=2)
        
        # Set seed (with rank offset for different shuffling per GPU)
        set_seed(config.seed + self.rank)
        
        # Device selection
        if self.distributed:
            # DDP: Use specific GPU assigned to this process
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        if self.is_main:
            print(f"\n{'='*60}")
            print("GPU/Hardware Configuration")
            print("="*60)
            if self.distributed:
                print(f"Mode: Distributed Data Parallel (DDP)")
                print(f"World Size: {self.world_size} GPUs")
                print(f"Backend: NCCL (optimized for NVIDIA GPUs)")
                for i in range(self.world_size):
                    props = torch.cuda.get_device_properties(i)
                    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
                print(f"Effective Batch Size: {config.batch_size} × {self.world_size} = {config.batch_size * self.world_size}")
            else:
                print(f"Mode: Single GPU")
                print(f"Device: {self.device}")
                if self.device.type == "cuda":
                    props = torch.cuda.get_device_properties(0)
                    print(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
                elif self.device.type == "mps":
                    print(f"  Apple Silicon (MPS backend)")
        
        # Model to device
        self.model = model.to(self.device)
        
        # Wrap model with DDP if distributed
        if self.distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            if self.is_main:
                print(f"Model wrapped with DistributedDataParallel")
        
        # Print parameter counts (only on main)
        if self.is_main:
            # Get base model for parameter count
            base_model = self.model.module if self.distributed else self.model
            param_counts = count_parameters(base_model)
            print(f"\nModel parameters:")
            print(f"  Total: {param_counts['total']:,}")
            print(f"  Trainable: {param_counts['trainable']:,}")
            print(f"  Frozen: {param_counts['frozen']:,}")
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
        
        # Calculate training steps
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * config.epochs
        if self.is_main:
            print(f"Training for {config.epochs} epochs ({self.total_steps} steps)")
        
        # Optimizer (use base model for DDP)
        base_model = self.model.module if self.distributed else self.model
        self.optimizer = get_optimizer(
            base_model,
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=config.scheduler,
            num_training_steps=self.total_steps,
            warmup_ratio=config.warmup_ratio,
        )
        
        # Mixed precision
        # CUDA: Full support with Tensor Cores (FP16 → up to 125 TFLOPS on V100)
        # MPS: Limited AMP support, some ops fallback to FP32
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        if self.is_main:
            if self.use_amp:
                print(f"\nMixed Precision: ENABLED (FP16 with Tensor Cores)")
                print(f"  → Faster training with reduced memory footprint")
            elif config.use_amp:
                print(f"\nMixed Precision: DISABLED (only supported on CUDA)")
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode="min",
        ) if config.early_stopping else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Resume if specified
        if config.resume_from:
            self._resume_training(config.resume_from)
        
        # Metrics
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Synchronize all processes before training
        if self.distributed:
            dist.barrier()
    
    def _resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        state = load_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            checkpoint_path,
            device=str(self.device),
        )
        self.epoch = state["epoch"]
        self.global_step = state["step"]
        self.best_val_loss = state.get("loss", float("inf"))
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Supports both single-GPU and multi-GPU (DDP) training.
        
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "gpu_config": {
                "distributed": self.distributed,
                "world_size": self.world_size,
                "device": str(self.device),
            }
        }
        
        if self.is_main:
            print("\n" + "=" * 60)
            print("Starting Training")
            print("=" * 60)
            if self.distributed:
                print(f"Training with {self.world_size} GPUs (DDP)")
                print(f"Effective batch size: {self.config.batch_size * self.world_size}")
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training epoch
            train_loss = self._train_epoch()
            
            # Gather loss across all processes
            if self.distributed:
                train_loss_tensor = torch.tensor([train_loss], device=self.device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = train_loss_tensor.item()
            
            history["train_loss"].append(train_loss)
            
            # Validation
            val_loss = None
            if self.val_loader:
                val_loss = self._validate()
                
                # Gather val loss across all processes
                if self.distributed:
                    val_loss_tensor = torch.tensor([val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                
                history["val_loss"].append(val_loss)
                
                # Check for improvement (only on main process)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.is_main:
                        self._save_best_model()
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)
            
            # Logging (only on main process)
            if self.is_main:
                print(f"\nEpoch {epoch + 1}/{self.config.epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                if val_loss:
                    print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint (only on main process)
            if self.is_main and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self.early_stopping and val_loss:
                if self.early_stopping(val_loss):
                    if self.is_main:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Synchronize all processes
            if self.distributed:
                dist.barrier()
        
        # Save final model (only on main process)
        if self.is_main:
            self._save_final_model()
            
            # Save training history
            with open(self.output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            
            print("\n" + "=" * 60)
            print("Training Complete")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Checkpoints saved to: {self.output_dir}")
            print("=" * 60)
        
        # Clean up distributed training
        if self.distributed:
            cleanup_distributed()
        
        return history
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter("Loss")
        
        # Only show progress bar on main process
        if self.is_main:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        else:
            pbar = self.train_loader
        
        for batch in pbar:
            # Move to device
            images = batch["image"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision (Tensor Cores on V100)
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, input_ids, labels)
                loss = outputs["loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            self.global_step += 1
            
            # Update progress bar (only on main process)
            if self.is_main and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
            
            # Mid-epoch validation
            if self.val_loader and self.global_step % self.config.eval_every == 0:
                val_loss = self._validate()
                self.model.train()  # Back to training mode
        
        return loss_meter.avg
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        
        loss_meter = AverageMeter("Val Loss")
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, input_ids, labels)
                loss = outputs["loss"]
            
            loss_meter.update(loss.item(), images.size(0))
        
        return loss_meter.avg
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        # Get base model (unwrap DDP if necessary)
        model_to_save = self.model.module if self.distributed else self.model
        save_checkpoint(
            model_to_save,
            self.optimizer,
            self.scheduler,
            epoch,
            self.global_step,
            self.best_val_loss,
            str(path),
        )
    
    def _save_best_model(self):
        """Save best model based on validation loss."""
        path = self.output_dir / "best_model.pt"
        # Get base model (unwrap DDP if necessary)
        model_to_save = self.model.module if self.distributed else self.model
        model_to_save.save_pretrained(str(path))
        print(f"  New best model saved!")
    
    def _save_final_model(self):
        """Save final model."""
        path = self.output_dir / "final_model.pt"
        # Get base model (unwrap DDP if necessary)
        model_to_save = self.model.module if self.distributed else self.model
        model_to_save.save_pretrained(str(path))


def train_person_vlm(config: TrainingConfig):
    """
    Convenience function to train Person VLM from config.
    
    Args:
        config: Training configuration
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models import PersonVLM, PersonVLMConfig
    from data import PersonVocabulary, create_dataloaders
    
    # Create vocabulary
    print("Loading vocabulary...")
    vocab = PersonVocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_file=config.train_file,
        val_file=config.val_file,
        vocab=vocab,
        batch_size=config.batch_size,
        image_size=config.image_size,
        max_seq_length=config.max_seq_length,
        num_workers=config.num_workers,
        augment_train=True,
    )
    
    # Create model
    print("\nCreating model...")
    model_config = PersonVLMConfig(
        vision_backbone=config.vision_backbone,
        vision_freeze_ratio=config.vision_freeze_ratio,
        decoder_size=config.decoder_size,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    )
    model = PersonVLM(model_config, tokenizer=vocab)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config,
    )
    
    # Train
    history = trainer.train()
    
    return history


if __name__ == "__main__":
    # Example training script
    print("Person VLM Training Pipeline")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from training import TrainingConfig, train_person_vlm
    
    config = TrainingConfig(
        train_file="data/train.json",
        val_file="data/val.json",
        epochs=20,
        batch_size=32,
        learning_rate=1e-4,
        output_dir="./checkpoints",
    )
    
    train_person_vlm(config)
    """)
