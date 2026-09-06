import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.config import CSCANConfig
from utils.transforms import get_val_test_transforms
from utils.dataset import BrainTumorDataset
from models.cscan import CSCAN
from utils.losses import get_loss_function
from utils.metrics import (
    compute_classification_metrics, 
    plot_confusion_matrix, 
    save_classification_report,
    save_metrics_to_csv
)

def test_model():
    """
    Evaluates the trained CSCAN model on the Test dataset.
    
    Steps:
    1. Loads val/test transforms (resizing, CLAHE, normalization; no augmentations).
    2. Builds the Test DataLoader.
    3. Instantiates CSCAN and loads the best saved weights (best_cscan_model.pth).
    4. Evaluates the model on the test dataset:
       - Computes test loss, accuracy, precision, recall, and F1-score.
    5. Saves classification report text file.
    6. Saves test confusion matrix heatmap.
    7. Exports final metrics to CSV for paper table inclusion.
    """
    device = CSCANConfig.DEVICE
    print("=" * 60)
    print("               CSCAN MODEL TESTING RUN")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Test Dataset
    # -------------------------------------------------------------------------
    print("\n[Stage 1] Loading Test Dataset...")
    test_transforms = get_val_test_transforms(CSCANConfig.IMAGE_SIZE)
    test_dataset = BrainTumorDataset(root_dir=CSCANConfig.TEST_DIR, transform=test_transforms)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CSCANConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=CSCANConfig.NUM_WORKERS,
        pin_memory=CSCANConfig.PIN_MEMORY
    )
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize Model & Load Best Checkpoint
    # -------------------------------------------------------------------------
    print("\n[Stage 2] Loading Trained CSCAN Model Weights...")
    model = CSCAN(
        num_classes=CSCANConfig.NUM_CLASSES,
        img_size=CSCANConfig.IMAGE_SIZE,
        dropout_rate=CSCANConfig.DROPOUT_RATE,
        pretrained=CSCANConfig.PRETRAINED,
        freeze_stages=CSCANConfig.FREEZE_STAGES,
        use_dcrf=CSCANConfig.USE_DCRF,
    ).to(device)
    
    if not os.path.exists(CSCANConfig.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {CSCANConfig.MODEL_SAVE_PATH}. "
            "Please run train.py first to train the model."
        )
        
    # Load state dict
    checkpoint = torch.load(CSCANConfig.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"[*] Successfully loaded model weights from: {CSCANConfig.MODEL_SAVE_PATH}")
    
    # Initialize loss function (no class weights applied for testing evaluation)
    criterion = get_loss_function(weights=None)
    
    # -------------------------------------------------------------------------
    # Step 3: Run Inference Evaluation
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Running Inference on Test Set...")
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            # Test-Time Augmentation: average softmax probabilities over the
            # original image and its horizontal flip. Free accuracy gain,
            # no retraining required. Brain MRI is left-right-symmetric-ish
            # (the same assumption the RandomHorizontalFlip training
            # augmentation already relies on), so this is a safe TTA choice.
            if CSCANConfig.USE_TTA:
                flipped_logits = model(torch.flip(images, dims=[3]))
                probs = 0.5 * (torch.softmax(logits, dim=1) + torch.softmax(flipped_logits, dim=1))
                # Reported loss uses NLL directly on the averaged log-probabilities
                # (CrossEntropyLoss can't be reused here since it expects raw
                # logits and would re-apply softmax). Doesn't affect accuracy.
                loss = torch.nn.functional.nll_loss(torch.log(probs.clamp_min(1e-8)), labels)
                preds = torch.argmax(probs, dim=1)
            else:
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)

            # Collect Stats
            test_loss += loss.item() * images.size(0)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    # Calculate overall metrics
    avg_test_loss = test_loss / len(test_dataset)
    metrics = compute_classification_metrics(test_targets, test_preds)
    metrics["test_loss"] = avg_test_loss
    
    print("\n" + "=" * 50)
    print("               TESTING RESULTS SUMMARY")
    print("=" * 50)
    print(f"Test Loss:      {avg_test_loss:.4f}")
    print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall:    {metrics['recall']:.4f}")
    print(f"Test F1-Score:  {metrics['f1_score']:.4f}")
    print("=" * 50)
    
    # -------------------------------------------------------------------------
    # Step 4: Export Reports & Plots
    # -------------------------------------------------------------------------
    print("\n[Stage 4] Exporting Reports & Visualizations...")
    
    # 1. Save detailed classification report (precision/recall per class)
    save_classification_report(test_targets, test_preds, CSCANConfig.REPORT_PATH)
    
    # 2. Plot and save confusion matrix image
    plot_confusion_matrix(
        test_targets, 
        test_preds, 
        CSCANConfig.TEST_CM_PATH, 
        title="Testing Confusion Matrix (CSCAN)"
    )
    
    # 3. Export final results to CSV
    save_metrics_to_csv(metrics, CSCANConfig.METRICS_CSV_PATH)
    
    print("\n[Done] Evaluation completed successfully. Results saved in 'results/' directory.")

if __name__ == "__main__":
    test_model()
