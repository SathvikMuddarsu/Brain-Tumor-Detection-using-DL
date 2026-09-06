import os
import torch

class CSCANConfig:
    """
    Configuration settings for the ConvNeXt-Swin Cross Attention Network (CSCAN).
    This centralizes all hyperparameters, directory paths, and training variables
    to prevent hardcoding and ensure reproducibility across training and testing.
    """
    
    # =========================================================================
    # 1. Dataset & Directory Paths
    # =========================================================================
    # Root directory of the project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dataset locations
    TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "Train")
    TEST_DIR = os.path.join(BASE_DIR, "dataset", "Test")
    
    # Directory to store training plots, confusion matrices, and model weights
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Target tumor classes in the Masoud Nickparvar dataset
    CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
    NUM_CLASSES = len(CLASSES)
    
    # =========================================================================
    # 2. Image Preprocessing & Dimensions
    # =========================================================================
    # Dimensions to resize MRI images. 224x224 is the standard resolution for
    # ImageNet models and offers a balance between fine details (tumor borders)
    # and computational workload on mid-range GPUs.
    IMAGE_SIZE = 224
    
    # =========================================================================
    # 3. Hardware & Optimization Settings (RTX 2050 VRAM Budgeting)
    # =========================================================================
    # Use GPU (CUDA) if available, otherwise fallback to CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Batch size: A size of 16 is chosen to avoid Out-Of-Memory (OOM) errors on 
    # the RTX 2050 GPU (4GB VRAM) while still providing stable gradients.
    BATCH_SIZE = 16
    
    # Number of workers for data loading
    NUM_WORKERS = 2
    
    # Pin memory for faster CPU-to-GPU data transfer
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # =========================================================================
    # 4. Hyperparameters
    # =========================================================================
    # Whether to initialize both backbones from ImageNet-pretrained weights
    # (models/pretrained_backbones.py) instead of training from scratch.
    # On a ~5,600-image dataset this is the single biggest lever for closing
    # the gap toward high accuracy. Default: True.
    PRETRAINED = True

    # Number of leading backbone stages to freeze when PRETRAINED=True.
    # 0 = fine-tune everything. 1-2 = freeze the stem/early stages (which
    # encode generic low-level texture/edge filters that transfer well as-is)
    # and only adapt mid/late stages + fusion + head. Recommended: 1.
    FREEZE_STAGES = 1

    # Number of epochs the frozen stages stay frozen before being unfrozen for
    # gradual/full fine-tuning. Set to 0 to keep FREEZE_STAGES frozen for the
    # whole run.
    UNFREEZE_AT_EPOCH = 5

    # Whether to run the Discriminative Confidence Refinement Fusion module.
    # The original ablation disabled it on the from-scratch backbones; it is
    # re-enabled by default here because pretrained features are already more
    # discriminative and are more likely to benefit from the refinement step.
    # Re-run the ablation on your data and flip this off if it doesn't help.
    USE_DCRF = True

    # Number of training epochs. Lower than the from-scratch config (100)
    # because pretrained backbones converge much faster and 100 epochs of
    # fine-tuning on 5.6k images will overfit; EARLY_STOPPING_PATIENCE will
    # usually stop well before this anyway.
    EPOCHS = 40

    # Stop training if val accuracy hasn't improved for this many epochs.
    EARLY_STOPPING_PATIENCE = 8

    # Learning rate for the classifier head / fusion / DCRF (newly-initialized
    # layers, trained from scratch on top of pretrained features).
    LEARNING_RATE = 3e-4

    # Learning rate for the pretrained backbone layers. Kept ~10x lower than
    # the head LR (discriminative learning rates) so fine-tuning nudges the
    # ImageNet features toward MRI without destroying them in a few noisy
    # steps on a small dataset.
    BACKBONE_LEARNING_RATE = 3e-5

    # Weight decay: 5e-2 pairs well with AdamW + cosine schedule for ViT/ConvNeXt
    # fine-tuning (this is what both architectures' original recipes use).
    WEIGHT_DECAY = 5e-2

    # Dropout rate in the classifier head.
    DROPOUT_RATE = 0.3

    # Label smoothing is already applied inside utils/losses.py (0.1).

    # MixUp / CutMix alpha. Set to 0 to disable. These synthesize new
    # training examples by blending image/label pairs, which meaningfully
    # reduces overfitting on small datasets. Applied only during training.
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    # Probability of applying MixUp/CutMix to a given batch (vs. plain CE).
    MIX_PROB = 0.5

    # Exponential Moving Average of model weights. EMA weights are evaluated
    # alongside raw weights each epoch and the better of the two is what gets
    # checkpointed — EMA usually gives a small but consistent accuracy/stability
    # bump at no extra training cost.
    USE_EMA = True
    EMA_DECAY = 0.999

    # Test-Time Augmentation: average predictions over the original image and
    # its horizontal flip at inference time (test.py). Free accuracy, no
    # retraining required.
    USE_TTA = True
    
    # =========================================================================
    # 5. Output File Names
    # =========================================================================
    MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "best_cscan_model.pth")
    ACCURACY_PLOT_PATH = os.path.join(RESULTS_DIR, "accuracy_curve.png")
    LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "loss_curve.png")
    TRAIN_CM_PATH = os.path.join(RESULTS_DIR, "train_confusion_matrix.png")
    TEST_CM_PATH = os.path.join(RESULTS_DIR, "test_confusion_matrix.png")
    REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
    METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "final_metrics.csv")
    
    @classmethod
    def print_config(cls):
        """Prints configuration settings to the console for tracking."""
        print("=" * 60)
        print("               CSCAN CONFIGURATION SETTINGS")
        print("=" * 60)
        print(f"Device:             {cls.DEVICE}")
        print(f"Train Directory:    {cls.TRAIN_DIR}")
        print(f"Test Directory:     {cls.TEST_DIR}")
        print(f"Results Directory:  {cls.RESULTS_DIR}")
        print(f"Classes:            {cls.CLASSES}")
        print(f"Image Size:         {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Batch Size:         {cls.BATCH_SIZE}")
        print(f"Epochs:             {cls.EPOCHS} (patience={cls.EARLY_STOPPING_PATIENCE})")
        print(f"Head LR / Backbone LR: {cls.LEARNING_RATE} / {cls.BACKBONE_LEARNING_RATE}")
        print(f"Weight Decay:       {cls.WEIGHT_DECAY}")
        print(f"Pretrained:         {cls.PRETRAINED} (freeze_stages={cls.FREEZE_STAGES}, unfreeze@{cls.UNFREEZE_AT_EPOCH})")
        print(f"DCRF Enabled:       {cls.USE_DCRF}")
        print(f"MixUp/CutMix:       alpha={cls.MIXUP_ALPHA}/{cls.CUTMIX_ALPHA}, prob={cls.MIX_PROB}")
        print(f"EMA:                {cls.USE_EMA} (decay={cls.EMA_DECAY})")
        print(f"TTA (test.py):      {cls.USE_TTA}")
        print("=" * 60)
