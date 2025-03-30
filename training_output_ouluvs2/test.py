import torch

checkpoint_path = "checkpoints/checkpoint_last.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu",  weights_only=False)

# Check what keys are inside the checkpoint
print(checkpoint.keys())

# If the model state_dict exists, check the architecture info
if "args" in checkpoint:
    print(checkpoint["args"])

print(checkpoint["cfg"])
