import torch
checkpoint = torch.load("checkpoints/checkpoint_last.pt", map_location="cpu")
print(checkpoint.keys())  # Check what keys exist
