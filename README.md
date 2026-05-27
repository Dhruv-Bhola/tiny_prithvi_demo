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
Satellite Image
       ↓
Patch Embedding
       ↓
Vision Transformer Encoder
       ↓
Feature Extraction
       ↓
Earth Observation Tasks

# Installation
- git clone (https://github.com/Dhruv-Bhola/tiny_prithvi_demo)
- cd tiny-prithvi-demo
- python -m venv prithvi_env
- source prithvi_env/bin/activate
- pip install -r requirements.txt

# Requirements

Create a requirements.txt file wi

torch
timm
Running the Project

Execute:

python tiny_prithvi_demo.py
