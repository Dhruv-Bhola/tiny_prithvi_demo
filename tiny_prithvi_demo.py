import torch
import timm

print("\n🌍 Tiny Prithvi Foundation Model Demo")
print("-" * 50)

# Device detection
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# Create Vision Transformer backbone
print("\nLoading Vision Transformer backbone...")

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=0
)

model = model.to(device)
model.eval()

print("✅ Transformer backbone loaded!")

# Simulated satellite image tensor
# (Prithvi uses multi-band satellite imagery)
print("\nCreating satellite-style input tensor...")

x = torch.randn(1, 3, 224, 224).to(device)

print(f"Input shape: {x.shape}")

# Inference
print("\nRunning inference...")

with torch.no_grad():
    features = model(x)

print("✅ Inference successful!")

print(f"Feature vector shape: {features.shape}")

print("\n🚀 Tiny Prithvi-style pipeline executed successfully!")
