#!/bin/bash
# ==============================================================================
# Multi-GPU Training Script for Person VLM (Scaled Version)
# ==============================================================================
# 
# This script launches distributed training using PyTorch's torchrun.
# Uses all available NVIDIA GPUs with NCCL backend.
#
# Hardware: 4x Tesla V100-DGXS-32GB (128GB total VRAM)
# Model: PersonVLM Scaled (~33.8M parameters)
#
# Key Differences from M4 MacBook:
# --------------------------------
# | Feature          | Apple M4 (MPS)      | Tesla V100 (CUDA)           |
# |------------------|---------------------|------------------------------|
# | GPUs             | 1 (unified)         | 4 (dedicated)               |
# | Memory           | 16-24GB (shared)    | 32GB × 4 = 128GB (dedicated)|
# | Compute          | Neural Engine       | Tensor Cores (FP16/FP32)    |
# | Mixed Precision  | Limited             | Full (125 TFLOPS FP16)      |
# | Multi-GPU        | Not supported       | DDP with NVLink             |
# | Backend          | MPS                 | CUDA + NCCL                 |
#
# Usage:
#   ./scripts/train_multigpu.sh              # Use default config
#   ./scripts/train_multigpu.sh --epochs 50  # Override epochs
# ==============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Number of GPUs (auto-detect)
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "============================================================"
echo "PersonVLM Scaled - Multi-GPU Training"
echo "============================================================"
echo "Detected GPUs: $NUM_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "============================================================"

# Default training parameters (can be overridden via command line)
# 
# OPTIMIZED based on research (Linear Scaling Rule):
# - LR scaled up from 5e-5 to 2e-4 (batch size 256 vs 32)
# - More epochs (75) for proper convergence
# - Reduced weight decay for larger model
#
CONFIG_FILE="configs/config.yaml"
TRAIN_FILE="PERSON_DATA/caption_with_attribute_labels/train.jsonl"
VAL_FILE="PERSON_DATA/caption_with_attribute_labels/val.jsonl"
IMAGE_DIR="PERSON_DATA/images"
VOCAB_FILE="data/vocabulary.json"
OUTPUT_DIR="./checkpoints_scaled"  # Overwrite previous (suboptimal) training
BATCH_SIZE=64         # Per-GPU batch size (effective = 64 × 4 = 256)
EPOCHS=75             # Increased from 30 - larger models need more epochs
LEARNING_RATE=2e-4    # Increased from 5e-5 - Linear Scaling Rule
DECODER_SIZE="large"  # 6 layers, 512 dim, 2048 FFN

# Parse any additional arguments
EXTRA_ARGS="$@"

# Additional training parameters
WEIGHT_DECAY=0.005    # Reduced for larger model (was 0.01)
WARMUP_RATIO=0.05     # Shorter warmup to reach peak LR faster

echo ""
echo "Training Configuration:"
echo "  Config: $CONFIG_FILE"
echo "  Decoder: $DECODER_SIZE (~27M params)"
echo "  Per-GPU Batch Size: $BATCH_SIZE"
echo "  Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Warmup Ratio: $WARMUP_RATIO"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "CHANGES FROM PREVIOUS RUN:"
echo "  - LR: 5e-5 → 2e-4 (4× increase, Linear Scaling Rule)"
echo "  - Epochs: 30 → 75 (larger models need more training)"
echo "  - Weight Decay: 0.01 → 0.005 (less regularization)"
echo "  - Warmup: 10% → 5% (reach peak LR faster)"
echo "============================================================"
echo ""

# Launch distributed training with torchrun
# - nproc_per_node: Number of processes (GPUs) per node
# - NCCL backend: Optimized for NVIDIA GPUs with NVLink
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/train.py \
    --config "$CONFIG_FILE" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --image_dir "$IMAGE_DIR" \
    --vocab_file "$VOCAB_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --decoder_size $DECODER_SIZE \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "============================================================"
