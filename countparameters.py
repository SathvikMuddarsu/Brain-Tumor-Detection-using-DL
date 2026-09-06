from models.cscan import CSCAN

model = CSCAN()

print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")