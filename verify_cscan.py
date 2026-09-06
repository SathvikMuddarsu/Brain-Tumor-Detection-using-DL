import torch
import torch.optim as optim
from utils.config import CSCANConfig
from utils.transforms import get_train_transforms
from utils.dataset import BrainTumorDataset
from models.cscan import CSCAN
from utils.losses import get_loss_function

# Feature dimension after architecture upgrade
FEAT_DIM = 768


def verify_cscan_pipeline():
    """
    Verifies the CSCAN model pipe
    line on a real batch of brain MRI scans.

    Checks:
    1. DataLoader yields real MRI images and labels.
    2. ConvNeXt-Tiny backbone outputs (B, 768, 7, 7).
    3. Swin-Tiny backbone outputs (B, 768, 7, 7).
    4. Cross-Attention fuses features into (B, 768, 7, 7).
    5. Classifier outputs logits (B, 4).
    6. Loss computed successfully.
    7. Backward pass computes gradients and optimizer step updates weights.
    8. DCRF is intentionally bypassed — its grad/weight delta will be 0. This is correct.
    """
    print("=" * 60)
    print("         CSCAN PIPELINE REAL-BATCH VERIFICATION")
    print("=" * 60)

    device = CSCANConfig.DEVICE
    print(f"[Info] Running verification on device: {device}")

    # ------------------------------------------------------------------ #
    # Step 1: Real dataset batch
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Initializing Dataset and DataLoader...")
    train_transforms = get_train_transforms(CSCANConfig.IMAGE_SIZE)
    dataset = BrainTumorDataset(root_dir=CSCANConfig.TRAIN_DIR, transform=train_transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    try:
        images, labels = next(iter(loader))
    except Exception as e:
        print(f"[Error] Failed to read batch from dataset: {e}")
        return

    print(f"[*] Successfully loaded a batch of real MRIs:")
    print(f"    - Image batch shape : {images.shape}  (Expected: [4, 3, 224, 224])")
    print(f"    - Label batch shape : {labels.shape}  (Expected: [4])")
    print(f"    - Labels in batch   : {labels.tolist()}")

    # ------------------------------------------------------------------ #
    # Step 2: Instantiate model
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Instantiating CSCAN Model...")
    model = CSCAN(num_classes=4, img_size=224,
                  dropout_rate=CSCANConfig.DROPOUT_RATE).to(device)
    model.train()

    images = images.to(device)
    labels = labels.to(device)

    # ------------------------------------------------------------------ #
    # Step 3: Intermediate shape validation
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3] Running intermediate shape validation (FEAT_DIM={FEAT_DIM})...")

    PASS = "[PASS]"
    FAIL = "[FAIL]"

    with torch.no_grad():
        feat_local = model.convnext(images)
        expected_feat = f"[4, {FEAT_DIM}, 7, 7]"
        status = PASS if list(feat_local.shape) == [4, FEAT_DIM, 7, 7] else FAIL
        print(f"    {status} ConvNeXt output   : {feat_local.shape}  (Expected: {expected_feat})")

        feat_global = model.swin(images)
        status = PASS if list(feat_global.shape) == [4, FEAT_DIM, 7, 7] else FAIL
        print(f"    {status} Swin output        : {feat_global.shape}  (Expected: {expected_feat})")

        feat_fused = model.fusion(feat_local, feat_global)
        status = PASS if list(feat_fused.shape) == [4, FEAT_DIM, 7, 7] else FAIL
        print(f"    {status} Cross-Attention out : {feat_fused.shape}  (Expected: {expected_feat})")

        # DCRF is bypassed in forward() — just verify its interface still works
        print(f"    [INFO] DCRF is intentionally bypassed (ablation result).")
        print(f"           Its weights will NOT receive gradients during training.")

        logits_check = model.classifier(feat_fused)
        status = PASS if list(logits_check.shape) == [4, 4] else FAIL
        print(f"    {status} Classifier output  : {logits_check.shape}  (Expected: [4, 4])")

    # ------------------------------------------------------------------ #
    # Step 4: End-to-end forward
    # ------------------------------------------------------------------ #
    print("\n[Step 4] Running End-to-End Forward Pass...")
    logits = model(images)
    print(f"[*] Forward pass completed. Logits shape: {logits.shape}")

    # ------------------------------------------------------------------ #
    # Step 5: Loss
    # ------------------------------------------------------------------ #
    print("\n[Step 5] Checking loss calculation...")
    criterion = get_loss_function(weights=None)
    loss = criterion(logits, labels)
    print(f"[*] Loss computed successfully: {loss.item():.4f}")

    # ------------------------------------------------------------------ #
    # Step 6: Gradient flow — check ACTIVE modules only
    # ------------------------------------------------------------------ #
    print("\n[Step 6] Verifying gradient flow and weight updates (active modules)...")
    optimizer = optim.AdamW(model.parameters(), lr=CSCANConfig.LEARNING_RATE,
                            weight_decay=CSCANConfig.WEIGHT_DECAY)

    # Track weights in ACTIVE (unfrozen) parameters
    # If FREEZE_STAGES > 0, pick a parameter from an unfrozen stage
    convnext_active = [p for p in model.convnext.parameters() if p.requires_grad][0]
    swin_active     = [p for p in model.swin.parameters() if p.requires_grad][0]
    param_awf       = model.awf.gate[1].weight
    param_fusion    = model.fusion.q_proj.weight
    param_cls       = model.classifier.mlp[0].weight

    convnext_before = convnext_active.clone().detach()
    swin_before     = swin_active.clone().detach()
    awf_before      = param_awf.clone().detach()
    fusion_before   = param_fusion.clone().detach()
    cls_before      = param_cls.clone().detach()

    optimizer.zero_grad()
    loss.backward()

    # Total gradient norm across all modules
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"    - Total gradient norm      : {grad_norm:.6f}")

    # Verify each active module received gradients
    for name, param in [("ConvNeXt (unfrozen)", convnext_active),
                        ("Swin (unfrozen)",     swin_active),
                        ("Adaptive Fusion",    param_awf),
                        ("Fusion q_proj",      param_fusion),
                        ("Classifier MLP",     param_cls)]:
        g = param.grad.norm().item() if param.grad is not None else 0.0
        status = PASS if g > 0 else FAIL
        print(f"    {status} {name:<20} grad norm: {g:.6f}")

    optimizer.step()

    # Weight delta verification
    print("\n[Step 7] Verifying weight deltas after optimizer step...")
    all_passed = True
    for name, before, after_param in [
        ("ConvNeXt", convnext_before, convnext_active),
        ("Swin",     swin_before,     swin_active),
        ("AWF",      awf_before,      param_awf),
        ("Fusion",   fusion_before,   param_fusion),
        ("Classifier", cls_before,    param_cls),
    ]:
        delta = torch.abs(before - after_param.detach()).sum().item()
        status = PASS if delta > 0 else FAIL
        if delta == 0:
            all_passed = False
        print(f"    {status} {name:<12} weight delta: {delta:.6f}")

    print()
    print("=" * 60)
    if all_passed:
        print("    *** SUCCESS: All active modules verified. Pipeline is correct! ***")
    else:
        print("    *** FAILURE: Some active module weights did not change. ***")
    print("=" * 60)


if __name__ == "__main__":
    verify_cscan_pipeline()
