# Overview

Prithvi is a geospatial foundation model developed for Earth Observation tasks such as:
- Flood Mapping
- Crop Classification
- Disaster Monitoring

The original Prithvi model is based on Vision Transformer (ViT) architecture and is trained on multi-band satellite imagery.


This project implements a simplified “Tiny Prithvi-style” inference pipeline to demonstrate:
- Local transformer execution
- Vision Transformer inference
- Satellite-style tensor processing
- Feature extraction workflow

# Architecture
Satellite Image ->
Patch Embedding ->
Vision Transformer Encoder ->
Feature Extraction ->
Earth Observation Tasks

# Installation
- git clone (https://github.com/Dhruv-Bhola/tiny_prithvi_demo)
- cd tiny-prithvi-demo
- python -m venv prithvi_env
- source prithvi_env/bin/activate
- pip install -r requirements.txt

# Execute:
python tiny_prithvi_demo.py

## How It Works
Step 1 — Device Detection

The script detects available hardware acceleration:

Apple Silicon GPU (MPS)
CUDA GPU
CPU fallback
Step 2 — Vision Transformer Initialization

A Vision Transformer backbone is created using TIMM:

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=0
)
Step 3 — Satellite-style Tensor Creation

A random tensor simulates satellite image data:

x = torch.randn(1, 3, 224, 224)
Step 4 — Transformer Inference

The tensor is passed through the transformer encoder for feature extraction.

Learning Outcomes

This project demonstrates:

Transformer-based image processing
Vision Transformer inference
Geospatial AI pipeline concepts
Local deployment of foundation-model-inspired architectures
Deep learning workflow implementation
Limitations

This is a lightweight educational implementation inspired by the Prithvi architecture.

It does not include:

Original Prithvi pretrained weights
Multi-band satellite datasets
Large-scale geospatial training
Fine-tuning pipelines

The goal is to demonstrate the core transformer-based Earth Observation workflow locally.
