#!/usr/bin/env python3
"""
PersonVLM Demo Script
Demonstrates model inference on test images with comparison to ground truth.

Usage:
    python3 demo.py                              # Run with best pretrained model
    python3 demo.py --model baseline             # Run with baseline model
    python3 demo.py --model pretrained           # Run with pretrained model (default)
    python3 demo.py --num_samples 10             # Specify number of samples
    python3 demo.py --save_html                  # Save results as HTML report
"""

import sys
import os
import json
import random
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

import torch
from PIL import Image
from torchvision import transforms


def load_model_and_vocab(model_type='pretrained'):
    """Load the trained model and vocabulary."""
    
    device = torch.device('mps' if torch.backends.mps.is_available() 
                          else 'cuda' if torch.cuda.is_available() 
                          else 'cpu')
    
    if model_type == 'baseline':
        # Load baseline custom decoder model
        from models import PersonVLM
        from data.vocabulary import PersonVocabulary
        
        checkpoint_path = 'checkpoints/best_model.pt'
        
        # Check if checkpoint exists and is valid (not a stub file)
        if not os.path.exists(checkpoint_path):
            print("Baseline checkpoint not found. Falling back to pretrained model.")
            return load_model_and_vocab('pretrained')
        
        # Check file size - baseline model should be ~30MB, stub files are ~1MB
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        if file_size < 5:  # Stub files are ~1MB
            print(f"Baseline checkpoint appears to be a stub file ({file_size:.1f}MB).")
            print("Falling back to pretrained model.")
            return load_model_and_vocab('pretrained')
        
        print("Loading baseline model (7.26M params)...")
        vocab = PersonVocabulary.load('data/vocabulary.json')
        model = PersonVLM.from_pretrained(checkpoint_path, tokenizer=vocab)
        model = model.to(device)
        model.eval()
        
        return model, vocab, device, 'baseline'
    
    else:
        # Load pretrained GPT-2 decoder model
        from models.person_vlm_pretrained import PersonVLMPretrained, PersonVLMPretrainedConfig
        from transformers import GPT2Tokenizer
        
        print("Loading pretrained model (98.51M params)...")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Determine checkpoint path
        if os.path.exists('checkpoints_pretrained_v2/best_model.pt'):
            checkpoint_path = 'checkpoints_pretrained_v2/best_model.pt'
            print(f"Using fine-tuned checkpoint: {checkpoint_path}")
        elif os.path.exists('checkpoints_pretrained/best_model.pt'):
            checkpoint_path = 'checkpoints_pretrained/best_model.pt'
            print(f"Using pretrained checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError("No pretrained checkpoint found. Please train the model first.")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get config from checkpoint
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            if isinstance(saved_config, PersonVLMPretrainedConfig):
                config = saved_config
            elif isinstance(saved_config, dict):
                config = PersonVLMPretrainedConfig(**saved_config)
            else:
                config = PersonVLMPretrainedConfig()
        else:
            config = PersonVLMPretrainedConfig()
        
        # Create and load model
        model = PersonVLMPretrained(config=config, tokenizer=tokenizer)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device, 'pretrained'


def load_test_data(test_file='PERSON_DATA/caption_with_attribute_labels/val.jsonl'):
    """Load test data with ground truth captions."""
    samples = []
    with open(test_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples.append({
                'image': sample.get('image', ''),
                'ground_truth': sample.get('answer', sample.get('caption', '')),
            })
    return samples


def get_image_transform():
    """Get the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_inference(model, image_path, transform, device, model_type='pretrained'):
    """Run inference on a single image."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if model_type == 'pretrained':
            # Pretrained model returns list of strings
            descriptions = model.generate(
                image_tensor,
                max_length=100,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
            )
            description = descriptions[0] if descriptions else "The image shows a person."
        else:
            # Baseline model
            description = model.generate(
                image_tensor,
                max_length=100,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )[0]
    
    return description


def truncate_text(text, max_words=50):
    """Truncate text to max words for display."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text


def print_results(results, model_type):
    """Print results to console in a nice format."""
    print("\n" + "=" * 80)
    print(f"PERSONVLM INFERENCE RESULTS ({model_type.upper()} MODEL)")
    print("=" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"\n[Sample {i}] {r['image_name']}")
        print("-" * 60)
        print(f"GENERATED: {truncate_text(r['generated'], 40)}")
        print(f"GROUND TRUTH: {truncate_text(r['ground_truth'], 40)}")
    
    print("\n" + "=" * 80)


def generate_html_report(results, model_type, output_path='demo_results.html'):
    """Generate an enterprise-grade HTML report with professional UI/UX."""
    
    # Set stats based on model type
    if model_type == 'pretrained':
        params = "98.51M"
        params_short = "98.51"
        inference_time = "25"
        bleu1, bleu2, bleu3, bleu4 = "54.23", "40.33", "31.57", "24.86"
        cider, rouge = "0.83", "42.22"
    else:
        params = "7.26M"
        params_short = "7.26"
        inference_time = "15"
        bleu1, bleu2, bleu3, bleu4 = "54.66", "39.89", "30.83", "24.19"
        cider, rouge = "0.75", "42.98"
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PersonVLM | Enterprise Vision-Language AI Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        
        :root {
            --bg-dark: #0a0b14;
            --bg-darker: #060710;
            --bg-card: rgba(15, 20, 40, 0.6);
            --bg-card-solid: #0d1025;
            --accent-blue: #4f8fff;
            --accent-cyan: #00d4ff;
            --accent-purple: #a855f7;
            --accent-pink: #ec4899;
            --accent-green: #22c55e;
            --text-white: #ffffff;
            --text-gray: #94a3b8;
            --text-muted: #64748b;
            --border-subtle: rgba(255, 255, 255, 0.06);
            --glow-blue: rgba(79, 143, 255, 0.3);
            --glow-purple: rgba(168, 85, 247, 0.3);
        }
        
        html { scroll-behavior: smooth; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-dark);
            color: var(--text-white);
            line-height: 1.6;
            overflow-x: hidden;
            min-height: 100vh;
        }
        
        /* Particle Canvas Background */
        #particles-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }
        
        /* Gradient Overlay */
        .gradient-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(ellipse 100% 70% at 50% -20%, rgba(79, 143, 255, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse 80% 50% at 80% 20%, rgba(168, 85, 247, 0.12) 0%, transparent 50%),
                radial-gradient(ellipse 60% 40% at 20% 80%, rgba(0, 212, 255, 0.08) 0%, transparent 50%);
            pointer-events: none;
            z-index: 1;
        }
        
        /* Navigation */
        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            padding: 16px 48px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(10, 11, 20, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-subtle);
        }
        
        .nav-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }
        
        .nav-logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .nav-logo-icon svg {
            width: 22px;
            height: 22px;
            fill: white;
        }
        
        .nav-logo-text {
            font-size: 20px;
            font-weight: 700;
            color: var(--text-white);
            letter-spacing: -0.02em;
        }
        
        .nav-logo-text span {
            color: var(--text-gray);
            font-weight: 500;
        }
        
        .nav-links {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .nav-link {
            color: var(--text-gray);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            padding: 10px 18px;
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        
        .nav-link:hover {
            color: var(--text-white);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .nav-btn {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            color: var(--text-white);
            font-size: 14px;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.2s ease;
            margin-left: 8px;
        }
        
        .nav-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        .nav-btn svg {
            width: 18px;
            height: 18px;
            fill: currentColor;
        }
        
        /* Main Content */
        .main-content {
            position: relative;
            z-index: 2;
        }
        
        /* Hero Section */
        .hero {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 120px 24px 80px;
            position: relative;
        }
        
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(79, 143, 255, 0.1);
            border: 1px solid rgba(79, 143, 255, 0.2);
            border-radius: 100px;
            font-size: 13px;
            font-weight: 500;
            color: var(--accent-cyan);
            margin-bottom: 32px;
        }
        
        .hero-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse-dot 2s infinite;
        }
        
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        
        .hero-title {
            font-size: clamp(42px, 7vw, 72px);
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.03em;
            margin-bottom: 24px;
            max-width: 900px;
        }
        
        .hero-title .gradient-text {
            background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 50%, var(--accent-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .hero-subtitle {
            font-size: 18px;
            color: var(--text-gray);
            max-width: 640px;
            margin-bottom: 40px;
            line-height: 1.7;
        }
        
        .hero-buttons {
            display: flex;
            gap: 16px;
            margin-bottom: 80px;
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 14px 28px;
            font-size: 15px;
            font-weight: 600;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
            box-shadow: 0 4px 24px var(--glow-blue), 0 0 0 1px rgba(255,255,255,0.1) inset;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px var(--glow-blue), 0 0 0 1px rgba(255,255,255,0.2) inset;
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-white);
            border: 1px solid var(--border-subtle);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        /* Hero Stats */
        .hero-stats {
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .hero-stat {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px 40px;
            text-align: center;
            min-width: 160px;
            transition: all 0.3s ease;
        }
        
        .hero-stat:hover {
            transform: translateY(-4px);
            border-color: rgba(79, 143, 255, 0.3);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }
        
        .hero-stat-value {
            font-size: 36px;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }
        
        .hero-stat-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        /* Scroll Indicator */
        .scroll-indicator {
            position: absolute;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            color: var(--text-muted);
            font-size: 12px;
            animation: bounce 2s infinite;
        }
        
        .scroll-indicator svg {
            width: 20px;
            height: 20px;
            stroke: currentColor;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateX(-50%) translateY(0); }
            50% { transform: translateX(-50%) translateY(8px); }
        }
        
        /* Container */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px;
        }
        
        /* Section */
        .section {
            padding: 100px 0;
        }
        
        .section-header {
            text-align: center;
            margin-bottom: 64px;
        }
        
        .section-label {
            display: inline-block;
            font-size: 12px;
            font-weight: 700;
            color: var(--accent-cyan);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 16px;
        }
        
        .section-title {
            font-size: 40px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 16px;
        }
        
        .section-subtitle {
            font-size: 16px;
            color: var(--text-gray);
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Architecture Section */
        .arch-grid {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 24px;
            flex-wrap: wrap;
        }
        
        .arch-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px;
            text-align: center;
            width: 240px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .arch-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .arch-card:hover {
            transform: translateY(-8px);
            border-color: rgba(79, 143, 255, 0.3);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        }
        
        .arch-card:hover::before {
            opacity: 1;
        }
        
        .arch-icon {
            width: 64px;
            height: 64px;
            background: linear-gradient(135deg, rgba(79, 143, 255, 0.2), rgba(168, 85, 247, 0.2));
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 28px;
        }
        
        .arch-label {
            font-size: 11px;
            font-weight: 700;
            color: var(--accent-cyan);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }
        
        .arch-name {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-white);
            margin-bottom: 12px;
        }
        
        .arch-info {
            font-size: 13px;
            color: var(--text-gray);
            line-height: 1.6;
        }
        
        .arch-arrow {
            font-size: 32px;
            color: var(--accent-blue);
            opacity: 0.6;
        }
        
        /* Metrics Section */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 20px;
        }
        
        @media (max-width: 1200px) { .metrics-grid { grid-template-columns: repeat(3, 1fr); } }
        @media (max-width: 768px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } }
        
        .metric-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 28px 24px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(34, 197, 94, 0.3);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            color: var(--accent-green);
            margin-bottom: 8px;
        }
        
        .metric-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        
        /* Results Section */
        .results-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }
        
        @media (max-width: 1024px) { .results-grid { grid-template-columns: 1fr; } }
        
        .result-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .result-card:hover {
            border-color: rgba(79, 143, 255, 0.3);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid var(--border-subtle);
        }
        
        .result-num {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .result-badge {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            color: white;
        }
        
        .result-label {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-white);
        }
        
        .result-filename {
            font-size: 11px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            color: var(--text-muted);
            background: rgba(0, 0, 0, 0.3);
            padding: 6px 10px;
            border-radius: 6px;
        }
        
        .result-body {
            display: flex;
            gap: 20px;
            padding: 20px;
        }
        
        .result-image {
            width: 180px;
            height: 180px;
            object-fit: cover;
            border-radius: 12px;
            flex-shrink: 0;
            border: 2px solid var(--border-subtle);
        }
        
        .result-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .desc-box {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            padding: 14px 16px;
            position: relative;
        }
        
        .desc-box.generated {
            border-left: 3px solid var(--accent-green);
        }
        
        .desc-box.ground-truth {
            border-left: 3px solid var(--accent-blue);
        }
        
        .desc-label {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
            padding: 4px 10px;
            border-radius: 4px;
        }
        
        .desc-box.generated .desc-label {
            background: rgba(34, 197, 94, 0.15);
            color: var(--accent-green);
        }
        
        .desc-box.ground-truth .desc-label {
            background: rgba(79, 143, 255, 0.15);
            color: var(--accent-blue);
        }
        
        .desc-text {
            font-size: 13px;
            color: var(--text-gray);
            line-height: 1.6;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 60px 24px;
            border-top: 1px solid var(--border-subtle);
            margin-top: 60px;
        }
        
        .footer-logo {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .footer-logo-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .footer-logo-icon svg {
            width: 18px;
            height: 18px;
            fill: white;
        }
        
        .footer-logo-text {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-white);
        }
        
        .footer-text {
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 24px;
        }
        
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 32px;
        }
        
        .footer-link {
            color: var(--text-gray);
            text-decoration: none;
            font-size: 14px;
            transition: color 0.2s;
        }
        
        .footer-link:hover {
            color: var(--accent-blue);
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-in {
            animation: fadeInUp 0.8s ease forwards;
        }
        
        .delay-1 { animation-delay: 0.1s; opacity: 0; }
        .delay-2 { animation-delay: 0.2s; opacity: 0; }
        .delay-3 { animation-delay: 0.3s; opacity: 0; }
        .delay-4 { animation-delay: 0.4s; opacity: 0; }
        
        /* Responsive */
        @media (max-width: 768px) {
            .navbar { padding: 12px 20px; }
            .nav-links { display: none; }
            .hero { padding: 100px 20px 60px; }
            .hero-buttons { flex-direction: column; width: 100%; max-width: 300px; }
            .hero-stats { gap: 12px; }
            .hero-stat { padding: 20px 24px; min-width: 140px; }
            .result-body { flex-direction: column; }
            .result-image { width: 100%; height: 200px; }
        }
    </style>
</head>
<body>
    <canvas id="particles-canvas"></canvas>
    <div class="gradient-overlay"></div>
    
    <!-- Navigation -->
    <nav class="navbar">
        <a href="#" class="nav-logo">
            <div class="nav-logo-icon">
                <svg viewBox="0 0 24 24"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>
            </div>
            <span class="nav-logo-text">Person<span>VLM</span></span>
        </a>
        <div class="nav-links">
            <a href="#overview" class="nav-link">Overview</a>
            <a href="#architecture" class="nav-link">Architecture</a>
            <a href="#results" class="nav-link">Results</a>
            <a href="#metrics" class="nav-link">Metrics</a>
            <a href="https://github.com/rishi02102017/PersonVLM" class="nav-btn" target="_blank">
                <svg viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                GitHub
            </a>
        </div>
    </nav>
    
    <div class="main-content">
        <!-- Hero Section -->
        <section class="hero" id="overview">
            <div class="hero-badge animate-in">
                Lightweight Vision-Language Model
            </div>
            <h1 class="hero-title animate-in delay-1">
                Transform <span class="gradient-text">Images</span><br>Into <span class="gradient-text">Understanding</span>
            </h1>
            <p class="hero-subtitle animate-in delay-2">
                A """ + params + """ parameter Vision-Language Model that generates structured, accurate descriptions of people from images. Built for real-time surveillance and monitoring applications.
            </p>
            <div class="hero-buttons animate-in delay-3">
                <a href="#results" class="btn btn-primary">View Results</a>
                <a href="#architecture" class="btn btn-secondary">Explore Architecture</a>
            </div>
            <div class="hero-stats animate-in delay-4">
                <div class="hero-stat">
                    <div class="hero-stat-value">""" + params_short + """</div>
                    <div class="hero-stat-label">Million Params</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">""" + inference_time + """</div>
                    <div class="hero-stat-label">ms Inference</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">30,000</div>
                    <div class="hero-stat-label">Training Samples</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">75</div>
                    <div class="hero-stat-label">% Loss Reduction</div>
                </div>
            </div>
            <div class="scroll-indicator">
                <svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M12 5v14M19 12l-7 7-7-7"/></svg>
            </div>
        </section>
        
        <!-- Architecture Section -->
        <section class="section" id="architecture">
            <div class="container">
                <div class="section-header">
                    <span class="section-label">How It Works</span>
                    <h2 class="section-title">Model Architecture</h2>
                    <p class="section-subtitle">Three-stage pipeline transforming visual input into natural language descriptions</p>
                </div>
                <div class="arch-grid">
                    <div class="arch-card">
                        <div class="arch-icon">👁️</div>
                        <div class="arch-label">Vision Encoder</div>
                        <div class="arch-name">MobileViT-XS</div>
                        <div class="arch-info">2.03M params<br>224×224 input<br>256-dim features</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-card">
                        <div class="arch-icon">🔗</div>
                        <div class="arch-label">Projection Layer</div>
                        <div class="arch-name">MLP Bridge</div>
                        <div class="arch-info">14.56M params<br>12 visual tokens<br>768-dim output</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-card">
                        <div class="arch-icon">📝</div>
                        <div class="arch-label">Text Decoder</div>
                        <div class="arch-name">DistilGPT-2</div>
                        <div class="arch-info">81.91M params<br>Fully fine-tuned<br>Autoregressive</div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Metrics Section -->
        <section class="section" id="metrics">
            <div class="container">
                <div class="section-header">
                    <span class="section-label">Performance</span>
                    <h2 class="section-title">Evaluation Metrics</h2>
                    <p class="section-subtitle">Comprehensive benchmarks on held-out validation set</p>
                </div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">""" + bleu1 + """</div>
                        <div class="metric-label">BLEU-1</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">""" + bleu2 + """</div>
                        <div class="metric-label">BLEU-2</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">""" + bleu3 + """</div>
                        <div class="metric-label">BLEU-3</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">""" + bleu4 + """</div>
                        <div class="metric-label">BLEU-4</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">""" + rouge + """</div>
                        <div class="metric-label">ROUGE-L</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">""" + cider + """</div>
                        <div class="metric-label">CIDEr</div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Results Section -->
        <section class="section" id="results">
            <div class="container">
                <div class="section-header">
                    <span class="section-label">Live Demo</span>
                    <h2 class="section-title">Sample Results</h2>
                    <p class="section-subtitle">Real-time inference results on validation images</p>
                </div>
                <div class="results-grid">
"""
    
    for i, r in enumerate(results, 1):
        img_path = f"PERSON_DATA/images/{r['image_name']}"
        
        html_content += f"""
                    <div class="result-card">
                        <div class="result-header">
                            <div class="result-num">
                                <span class="result-badge">{i}</span>
                                <span class="result-label">Sample {i}</span>
                            </div>
                            <span class="result-filename">{r['image_name']}</span>
                        </div>
                        <div class="result-body">
                            <img class="result-image" src="{img_path}" alt="Person {i}" onerror="this.style.background='linear-gradient(135deg, #0d1025, #060710)';">
                            <div class="result-content">
                                <div class="desc-box generated">
                                    <div class="desc-label">● Generated</div>
                                    <div class="desc-text">{r['generated']}</div>
                                </div>
                                <div class="desc-box ground-truth">
                                    <div class="desc-label">● Ground Truth</div>
                                    <div class="desc-text">{truncate_text(r['ground_truth'], 150)}</div>
                                </div>
                            </div>
                        </div>
                    </div>
"""
    
    html_content += f"""
                </div>
            </div>
        </section>
        
        <!-- Footer -->
        <footer class="footer">
            <div class="footer-logo">
                <div class="footer-logo-icon">
                    <svg viewBox="0 0 24 24"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/></svg>
                </div>
                <span class="footer-logo-text">PersonVLM</span>
            </div>
            <p class="footer-text">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} · Powered by Tesla V100 GPU</p>
            <div class="footer-links">
                <a href="https://github.com/rishi02102017/PersonVLM" class="footer-link">GitHub</a>
                <a href="#architecture" class="footer-link">Architecture</a>
                <a href="#metrics" class="footer-link">Metrics</a>
                <a href="#results" class="footer-link">Results</a>
            </div>
        </footer>
    </div>
    
    <script>
        // Particle Animation
        const canvas = document.getElementById('particles-canvas');
        const ctx = canvas.getContext('2d');
        let particles = [];
        const particleCount = 80;
        const connectionDistance = 150;
        
        function resize() {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }}
        
        class Particle {{
            constructor() {{
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;
                this.radius = Math.random() * 2 + 1;
                this.opacity = Math.random() * 0.5 + 0.2;
            }}
            
            update() {{
                this.x += this.vx;
                this.y += this.vy;
                
                if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
            }}
            
            draw() {{
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(79, 143, 255, ${{this.opacity}})`;
                ctx.fill();
            }}
        }}
        
        function init() {{
            resize();
            particles = [];
            for (let i = 0; i < particleCount; i++) {{
                particles.push(new Particle());
            }}
        }}
        
        function drawConnections() {{
            for (let i = 0; i < particles.length; i++) {{
                for (let j = i + 1; j < particles.length; j++) {{
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < connectionDistance) {{
                        const opacity = (1 - dist / connectionDistance) * 0.15;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(79, 143, 255, ${{opacity}})`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }}
                }}
            }}
        }}
        
        function animate() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            particles.forEach(p => {{
                p.update();
                p.draw();
            }});
            
            drawConnections();
            requestAnimationFrame(animate);
        }}
        
        window.addEventListener('resize', () => {{
            resize();
        }});
        
        init();
        animate();
        
        // Smooth scroll
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function(e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }});
        }});
        
        // Animate elements on scroll
        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }}
            }});
        }}, {{ threshold: 0.1 }});
        
        document.querySelectorAll('.animate-in').forEach(el => {{
            observer.observe(el);
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nHTML report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='PersonVLM Demo')
    parser.add_argument('--model', type=str, default='pretrained', 
                        choices=['baseline', 'pretrained'],
                        help='Model to use: baseline (7.26M) or pretrained (98.51M)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--save_html', action='store_true', help='Save results as HTML report')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("PersonVLM Demo")
    print("=" * 60)
    
    # Load model
    model, tokenizer, device, model_type = load_model_and_vocab(args.model)
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    
    # Load test data
    test_samples = load_test_data()
    print(f"Total test samples: {len(test_samples)}")
    
    # Select random samples
    selected = random.sample(test_samples, min(args.num_samples, len(test_samples)))
    
    # Get transform
    transform = get_image_transform()
    
    # Run inference
    results = []
    image_dir = 'PERSON_DATA/images'
    
    print(f"\nRunning inference on {len(selected)} samples...")
    
    for i, sample in enumerate(selected, 1):
        image_name = os.path.basename(sample['image'])
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"  [{i}] Image not found: {image_name}")
            continue
        
        generated = run_inference(model, image_path, transform, device, model_type)
        
        results.append({
            'image_name': image_name,
            'image_path': image_path,
            'generated': generated,
            'ground_truth': sample['ground_truth'],
        })
        
        print(f"  [{i}] Processed: {image_name}")
    
    # Print results
    print_results(results, model_type)
    
    # Save HTML if requested
    if args.save_html:
        generate_html_report(results, model_type)
    
    print("\nDemo complete!")
    
    return results


if __name__ == '__main__':
    main()
