import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from utils.config import CSCANConfig
from utils.transforms import get_train_transforms, get_val_test_transforms
from utils.dataset import BrainTumorDataset
from models.cscan import CSCAN
from utils.losses import get_loss_function, SoftTargetCrossEntropy
from utils.regularization import mixup_cutmix, mixup_criterion, ModelEMA
from utils.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_classification_report
)


def set_backbone_frozen(model, frozen: bool):
    """(Un)freezes the top `CSCANConfig.FREEZE_STAGES` stages of both
    backbones. Called once at start (frozen=True) and again at
    CSCANConfig.UNFREEZE_AT_EPOCH (frozen=False). No-op for scratch backbones.
    """
    if not model.pretrained or CSCANConfig.FREEZE_STAGES <= 0:
        return
    for backbone in (model.convnext, model.swin):
        for idx, child in enumerate(backbone.features.children()):
            if idx < CSCANConfig.FREEZE_STAGES:
                for p in child.parameters():
                    p.requires_grad = not frozen


def build_param_groups(model):
    """Discriminative learning rates: pretrained backbone params get a much
    lower LR than newly-initialized fusion/DCRF/classifier params, so
    fine-tuning nudges ImageNet features toward MRI instead of overwriting
    them within the first few noisy steps.
    """
    if not model.pretrained:
        return [{"params": model.parameters(), "lr": CSCANConfig.LEARNING_RATE}]

    backbone_param_ids = {id(p) for m in (model.convnext, model.swin) for p in m.parameters()}
    backbone_params, head_params = [], []
    for p in model.parameters():
        (backbone_params if id(p) in backbone_param_ids else head_params).append(p)

    return [
        {"params": backbone_params, "lr": CSCANConfig.BACKBONE_LEARNING_RATE},
        {"params": head_params, "lr": CSCANConfig.LEARNING_RATE},
    ]


def run_eval(model, loader, criterion, device, dataset_len):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    avg_loss = total_loss / dataset_len
    metrics = compute_classification_metrics(targets, preds)
    return avg_loss, metrics, preds, targets


def train_model():
    """
    Orchestrates the entire training pipeline for CSCAN-DCRF.

    Pipeline Steps:
    1. Loads configuration from CSCANConfig.
    2. Builds train and validation dataloaders (85% train, 15% validation split).
    3. Initializes CSCAN (optionally ImageNet-pretrained), discriminative-LR
       AdamW optimizer, soft/hard loss functions, and EMA.
    4. Runs training epochs with MixUp/CutMix, a backbone freeze->unfreeze
       schedule, and early stopping on validation accuracy.
    5. Saves the best raw-weight AND best EMA-weight checkpoints, whichever
       scores higher.
    6. Plots and exports loss/accuracy curves, confusion matrices, and metrics.
    """
    CSCANConfig.print_config()
    device = CSCANConfig.DEVICE

    # -------------------------------------------------------------------------
    # Step 1: Datasets & Dataloaders Setup
    # -------------------------------------------------------------------------
    print("\n[Stage 1] Loading Dataset & Splitting Train/Validation...")
    train_transforms = get_train_transforms(CSCANConfig.IMAGE_SIZE)
    val_transforms = get_val_test_transforms(CSCANConfig.IMAGE_SIZE)

    full_dataset_train = BrainTumorDataset(root_dir=CSCANConfig.TRAIN_DIR, transform=train_transforms)
    full_dataset_val = BrainTumorDataset(root_dir=CSCANConfig.TRAIN_DIR, transform=val_transforms)

    from sklearn.model_selection import train_test_split

    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset_train)),
        test_size=0.15,
        stratify=full_dataset_train.labels,
        random_state=42,
    )

    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)

    print(f"Dataset Split Details:")
    print(f"  - Training Subset: {len(train_dataset)} images (augmentations enabled)")
    print(f"  - Validation Subset: {len(val_dataset)} images (augmentations disabled)")

    train_generator = torch.Generator()
    train_generator.manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CSCANConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=CSCANConfig.NUM_WORKERS,
        pin_memory=CSCANConfig.PIN_MEMORY,
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CSCANConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=CSCANConfig.NUM_WORKERS,
        pin_memory=CSCANConfig.PIN_MEMORY
    )

    # -------------------------------------------------------------------------
    # Step 2: Model, Optimizer, and Loss Initialization
    # -------------------------------------------------------------------------
    print("\n[Stage 2] Initializing CSCAN Model, Optimizer, and Loss...")
    model = CSCAN(
        num_classes=CSCANConfig.NUM_CLASSES,
        img_size=CSCANConfig.IMAGE_SIZE,
        dropout_rate=CSCANConfig.DROPOUT_RATE,
        pretrained=CSCANConfig.PRETRAINED,
        freeze_stages=CSCANConfig.FREEZE_STAGES,
        use_dcrf=CSCANConfig.USE_DCRF,
    ).to(device)

    set_backbone_frozen(model, frozen=True)

    all_train_labels = [full_dataset_train.labels[idx] for idx in train_indices]
    class_counts = np.bincount(all_train_labels, minlength=CSCANConfig.NUM_CLASSES)
    total_samples = len(all_train_labels)
    class_weights = total_samples / (CSCANConfig.NUM_CLASSES * class_counts)

    criterion_hard = get_loss_function(weights=class_weights)
    criterion_soft = SoftTargetCrossEntropy(label_smoothing=0.1, num_classes=CSCANConfig.NUM_CLASSES)
    eval_criterion = get_loss_function(weights=None)  # unweighted, for a fair val/comparison signal

    optimizer = torch.optim.AdamW(
        build_param_groups(model),
        weight_decay=CSCANConfig.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(CSCANConfig.EPOCHS - 5, 1),
        eta_min=1e-6
    )

    ema = ModelEMA(model, decay=CSCANConfig.EMA_DECAY) if CSCANConfig.USE_EMA else None

    # -------------------------------------------------------------------------
    # Step 3: Epoch Loop
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Beginning CSCAN Training Loop...")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_epoch_train_targets, best_epoch_train_preds = [], []

    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    for epoch in range(CSCANConfig.EPOCHS):
        # Unfreeze backbone stages once past the warmup window.
        if epoch == CSCANConfig.UNFREEZE_AT_EPOCH:
            set_backbone_frozen(model, frozen=False)
            print(f"[*] Unfroze backbone stage(s) at epoch {epoch + 1}.")

        # Linear LR warmup for the first 5 epochs (applies per-param-group).
        if epoch < 5:
            warmup_scale = (epoch + 1) / 5
            base_lrs = [CSCANConfig.BACKBONE_LEARNING_RATE, CSCANConfig.LEARNING_RATE] \
                if model.pretrained else [CSCANConfig.LEARNING_RATE]
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base_lr * warmup_scale

        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CSCANConfig.EPOCHS}", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            do_mix = CSCANConfig.MIXUP_ALPHA > 0 or CSCANConfig.CUTMIX_ALPHA > 0
            mix_labels = None
            if do_mix:
                images, mix_labels = mixup_cutmix(
                    images, labels, CSCANConfig.NUM_CLASSES,
                    mixup_alpha=CSCANConfig.MIXUP_ALPHA,
                    cutmix_alpha=CSCANConfig.CUTMIX_ALPHA,
                    prob=CSCANConfig.MIX_PROB,
                )

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if use_cuda_amp else "cpu"):
                logits = model(images)
                if do_mix:
                    loss = mixup_criterion(criterion_soft, logits, mix_labels)
                else:
                    loss = criterion_hard(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            train_loss += loss.item() * images.size(0)
            # NOTE: with MixUp/CutMix active, "train accuracy" below is computed
            # against the *original* (pre-mix) labels for a readable diagnostic
            # signal — it is not the exact optimization target for mixed batches.
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        epoch_train_loss = train_loss / len(train_dataset)
        train_metrics = compute_classification_metrics(train_targets, train_preds)

        # --- Validation Phase (raw weights) ---
        epoch_val_loss, val_metrics, val_preds, val_targets = run_eval(
            model, val_loader, eval_criterion, device, len(val_dataset)
        )

        # --- Validation Phase (EMA weights), if enabled ---
        ema_val_acc = -1.0
        if ema is not None:
            eval_model = CSCAN(
                num_classes=CSCANConfig.NUM_CLASSES, img_size=CSCANConfig.IMAGE_SIZE,
                dropout_rate=CSCANConfig.DROPOUT_RATE, pretrained=CSCANConfig.PRETRAINED,
                freeze_stages=CSCANConfig.FREEZE_STAGES, use_dcrf=CSCANConfig.USE_DCRF,
            ).to(device)
            eval_model.load_state_dict(ema.state_dict())
            _, ema_metrics, _, _ = run_eval(eval_model, val_loader, eval_criterion, device, len(val_dataset))
            ema_val_acc = ema_metrics["accuracy"]
            del eval_model

        current_lr = optimizer.param_groups[-1]["lr"]

        print("\n" + "=" * 90)
        print(f"Epoch [{epoch+1}/{CSCANConfig.EPOCHS}] Performance Summary")
        print("-" * 90)
        print(
            f"Train - Loss: {epoch_train_loss:.4f} | Acc: {train_metrics['accuracy']*100:.2f}% | "
            f"Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1_score']:.4f}"
        )
        print(
            f"Val   - Loss: {epoch_val_loss:.4f} | Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1_score']:.4f}"
        )
        if ema is not None:
            print(f"Val (EMA weights) - Acc: {ema_val_acc*100:.2f}%")
        print(f"Head LR: {current_lr:.8f}")
        print("=" * 90 + "\n")

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_metrics["accuracy"])

        if epoch >= 5:
            scheduler.step()

        # Compare raw vs EMA validation accuracy; checkpoint whichever wins.
        candidate_acc = max(val_metrics["accuracy"], ema_val_acc)
        if candidate_acc > best_val_acc:
            best_val_acc = candidate_acc
            epochs_without_improvement = 0
            if ema is not None and ema_val_acc >= val_metrics["accuracy"]:
                torch.save(ema.state_dict(), CSCANConfig.MODEL_SAVE_PATH)
                print(f"[*] New best validation accuracy: {best_val_acc*100:.2f}% (EMA weights). Checkpoint saved.")
            else:
                torch.save(model.state_dict(), CSCANConfig.MODEL_SAVE_PATH)
                print(f"[*] New best validation accuracy: {best_val_acc*100:.2f}% (raw weights). Checkpoint saved.")
            best_epoch_train_targets = train_targets.copy()
            best_epoch_train_preds = train_preds.copy()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CSCANConfig.EARLY_STOPPING_PATIENCE:
                print(
                    f"[*] Early stopping: no val accuracy improvement for "
                    f"{CSCANConfig.EARLY_STOPPING_PATIENCE} epochs."
                )
                break

    # -------------------------------------------------------------------------
    # Step 4: Plots & Outputs Generation
    # -------------------------------------------------------------------------
    print("\n[Stage 4] Generating Final Results and Curves...")
    plot_training_curves(history["train_loss"], history["val_loss"], "Loss", CSCANConfig.LOSS_PLOT_PATH)
    plot_training_curves(history["train_acc"], history["val_acc"], "Accuracy", CSCANConfig.ACCURACY_PLOT_PATH)

    if len(best_epoch_train_targets) > 0:
        plot_confusion_matrix(
            best_epoch_train_targets, best_epoch_train_preds,
            CSCANConfig.TRAIN_CM_PATH, title="Training Confusion Matrix"
        )

    print(f"\n[Done] Training completed successfully. Best Validation Accuracy: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    train_model()
