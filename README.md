
# CSCAN-DCRF: Brain Tumor MRI Classification

A deep learning pipeline that classifies brain MRI scans into four categories — **glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using a dual-backbone hybrid network that fuses **ConvNeXt** (local/texture features) and a **Swin Transformer** (global/contextual features) through a custom cross-attention and confidence-refinement pipeline.

The model achieves **95.75% test accuracy** on the Nickparvar Brain Tumor MRI dataset.Grad-CAM visualizations are included to improve model interpretability and visualize the regions influencing predictions.

<h1 align="center">CSCAN-DCRF: Brain Tumor MRI Classification</h1>

<p align="center">
  <img 
    width="100%" 
    alt="CSCAN-DCRF Architecture" 
    src="https://github.com/user-attachments/assets/78f2dbda-c69d-4dce-8668-4a8dffd74276" 
  />
</p>

## Overview

Brain tumor diagnosis from MRI usually depends on a radiologist visually distinguishing tumor types by both fine-grained texture (edges, boundaries) and broader spatial/anatomical context. This project mirrors that dual perspective architecturally:

- **ConvNeXt-Tiny** extracts local, texture-level features (tumor borders, tissue detail).
- **Swin-Tiny** extracts global, context-level features (spatial relationships across the whole scan).
- The two feature streams are fused, refined, and classified through several purpose-built modules described below.

## Architecture

```
Input MRI (3 x 224 x 224)
        │
        ├── ConvNeXt-Tiny (local features)  ──────┐
        │                                          │
        └── Swin-Tiny (global features)  ──────────┤
                                                    ▼
                              Adaptive Weighted Fusion (AWF)
                        per-channel gated blend of both feature maps
                                                    │
                                                    ▼
                          Cross-Attention Fusion (Pre-LayerNorm)
                     ConvNeXt = Query, Swin = Key/Value
                                                    │
                                                    ▼
                Discriminative Confidence Refinement Fusion (DCRF)
             margin-confidence-guided spatial refinement of ambiguous regions
                                                    │
                                                    ▼
                      GeM Pooling (p = 3.0, learnable)
                                                    │
                                                    ▼
                     LayerNorm → MLP Head (GELU, Dropout)
                                                    │
                                                    ▼
                        Logits (glioma / meningioma / pituitary / notumor)
```

**Key components**

| Module | File | Purpose |
|---|---|---|
| `ConvNeXtTinyPretrained` / `SwinTinyPretrained` | `models/pretrained_backbones.py` | ImageNet-pretrained backbones (drop-in replacements for the from-scratch versions in `models/convnext.py` / `models/swin_transformer.py`) |
| `AdaptiveWeightedFusion` | `models/cscan.py` | Learns a per-channel gate `α` to blend ConvNeXt and Swin features: `F = α·F_conv + (1-α)·F_swin` |
| `CrossAttentionFusion` | `models/cross_attention.py` | Multi-head cross-attention where ConvNeXt features query the Swin feature map for global context |
| `DiscriminativeConfidenceRefinementFusion` (DCRF) | `models/dcrf.py` | Estimates per-pixel class-margin confidence and refines low-confidence (ambiguous) regions with a depthwise 3×3 convolution over local context |
| `CSCANClassifier` | `models/classifier.py` | GeM pooling (learnable exponent) → LayerNorm → 2-layer MLP head |

## Results

Evaluated on the held-out test set (1,600 images, 400 per class), with Test-Time Augmentation (horizontal-flip averaging) enabled:

| Metric | Score |
|---|---|
| Accuracy | **95.75%** |
| Precision (weighted) | 96.13% |
| Recall (weighted) | 95.75% |
| F1-score (weighted) | 95.68% |
| Test loss | 0.2205 |

**Per-class performance**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Glioma | 1.00 | 0.84 | 0.91 |
| Meningioma | 0.88 | 0.99 | 0.94 |
| Pituitary | 0.99 | 1.00 | 1.00 |
| No tumor | 0.97 | 1.00 | 0.98 |

Glioma is the hardest class to recognize (recall 0.84) and is most often confused with meningioma — visible in the confusion matrix below. This is a known, Glioma is the hardest class to recognize (recall 0.84) and is most often confused with meningioma, as visible in the confusion matrix below.
<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/4c85d52a-83a7-4b2c-a7c1-635e86bc9e06"
    width="48%"
    alt="Training and Validation Accuracy Curve"
  />
  <img 
    src="https://github.com/user-attachments/assets/580e7f96-5df2-4af8-97d4-2cf0d8766217"
    width="48%"
    alt="Training and Validation Loss Curve"
  />
</p>


> **A note on this number.** The dataset used here is known to contain near-duplicate slices leaking between its official Train/Test split (adjacent slices from the same patient scan can look almost identical), which is why some public notebooks on this dataset report 99%+ accuracy. Before citing this result in a report or paper, consider running a perceptual-hash duplicate check between Train/Test, or reporting stratified k-fold cross-validation accuracy as a more defensible number. See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for the full discussion.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle), combining images from three source datasets (figshare, SARTAJ, Br35H).

- **Classes:** glioma, meningioma, pituitary, notumor
- **Split used:** ~4,000 training images / 1,600 test images (400 per class in the test set)
- The dataset is **not included** in this repository. Download it from Kaggle and place it as:

```
dataset/
├── Train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

## Preprocessing & Augmentation

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the LAB lightness channel — enhances tumor boundary contrast without amplifying background noise (`utils/transforms.py`)
- Resize to 224×224, RGB conversion (grayscale MRIs are replicated to 3 channels)
- **Training only:** random horizontal flip, random affine (±5% translation/zoom)
- **MixUp / CutMix** during training (`utils/regularization.py`) to reduce overfitting on a relatively small dataset
- Normalization to `[-1, 1]`

## Project Structure

```
.
├── models/
│   ├── convnext.py               # ConvNeXt-Tiny, trained from scratch
│   ├── swin_transformer.py       # Swin-Tiny, trained from scratch
│   ├── pretrained_backbones.py   # ImageNet-pretrained ConvNeXt/Swin wrappers
│   ├── cross_attention.py        # Cross-attention fusion module
│   ├── dcrf.py                   # Discriminative Confidence Refinement Fusion
│   ├── classifier.py             # GeM pooling + MLP classification head
│   └── cscan.py                  # Full CSCAN model assembly
├── utils/
│   ├── config.py                 # All hyperparameters and paths
│   ├── dataset.py                # PyTorch Dataset for the MRI folders
│   ├── transforms.py             # CLAHE + augmentation pipelines
│   ├── losses.py                 # Loss functions (label smoothing, soft-target CE)
│   ├── regularization.py         # MixUp/CutMix, EMA
│   └── metrics.py                # Metrics, confusion matrix, report generation
├── results/                      # Training curves, confusion matrix, reports, Grad-CAM outputs
├── train.py                      # Training entry point
├── test.py                       # Evaluation entry point (with TTA)
├── grad_cam.py                   # Grad-CAM visualization generator
├── verify_cscan.py               # Sanity-checks the full forward/backward pipeline on a real batch
├── countparameters.py            # Prints total/trainable parameter counts
├── requirements.txt
└── IMPROVEMENTS.md               # Design notes and rationale for the current configuration
```

## Installation

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.5.1 (CUDA optional but recommended — see [Hardware notes](#hardware-notes)).

## Usage

**1. Train the model**

```bash
python train.py
```

On first run this downloads ImageNet-pretrained weights for ConvNeXt-Tiny and Swin-Tiny (internet access required). The best checkpoint (by validation accuracy) is saved to `results/best_cscan_model.pth`.

**2. Evaluate on the test set**

```bash
python test.py
```

Outputs a classification report, confusion matrix, and a metrics CSV to `results/`.

**3. Generate Grad-CAM visualizations**

```bash
python grad_cam.py
```

**4. Sanity-check the pipeline** (verifies shapes, gradients, and weight updates on a real batch)

```bash
python verify_cscan.py
```

**5. Count model parameters**

```bash
python countparameters.py
```

## Configuration

All hyperparameters live in `utils/config.py`. Notable defaults:

| Setting | Default | Notes |
|---|---|---|
| `PRETRAINED` | `True` | Use ImageNet-pretrained ConvNeXt/Swin instead of training from scratch |
| `FREEZE_STAGES` | `1` | Freeze the first backbone stage for the first few epochs |
| `UNFREEZE_AT_EPOCH` | `5` | Epoch at which frozen stages are unfrozen |
| `USE_DCRF` | `True` | Enable the confidence-refinement fusion module |
| `EPOCHS` | `40` | With early stopping (`patience=8`) |
| `LEARNING_RATE` / `BACKBONE_LEARNING_RATE` | `3e-4` / `3e-5` | Discriminative learning rates for head vs. backbone |
| `MIXUP_ALPHA` / `CUTMIX_ALPHA` | `0.2` / `1.0` | Set both to `0` to disable |
| `USE_EMA` | `True` | Exponential moving average of weights, checkpointed if it beats raw weights |
| `USE_TTA` | `True` | Horizontal-flip test-time augmentation (`test.py` only) |

To reproduce the original from-scratch (non-pretrained) configuration for comparison, set `PRETRAINED = False`, `USE_DCRF = False`, `MIXUP_ALPHA = CUTMIX_ALPHA = 0`, `USE_EMA = False`, `EPOCHS = 100`, `LEARNING_RATE = 1e-4`, `WEIGHT_DECAY = 1e-4`, `DROPOUT_RATE = 0.2`.

## Hardware Notes

Default `BATCH_SIZE = 16` was chosen to fit a 4GB VRAM GPU (developed/tested on an RTX 2050). Increase it if you have more VRAM available. Training will fall back to CPU automatically if no CUDA device is found, but this will be considerably slower.

## Explainability

`grad_cam.py` produces Grad-CAM overlays using the final ConvNeXt feature map as the target layer (kept consistent across all four classes for methodological fairness), with anatomical erosion to suppress skull/background artifacts. Sample outputs for all four classes are in `results/gradcam/`.
##Results of GradCAM
<h3 align="center">Grad-CAM Visualization Results</h3>

<table align="center">
  <tr>
    <th>Glioma</th>
    <th>Meningioma</th>
    <th>Pituitary</th>
    <th>No Tumor</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/ccbafec3-753d-4eb5-b69c-70b2b56eea31" width="200" alt="Glioma Grad-CAM" />
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/3aa17caa-85ea-499f-965d-9518d0f077c1" width="200" alt="Meningioma Grad-CAM" />
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/189076d9-4533-4fc4-b35f-b87ccd01af58" width="200" alt="Pituitary Grad-CAM" />
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/3cea67e1-d8ff-4cb5-8c1c-38e03061b0ff" width="200" alt="No Tumor Grad-CAM" />
    </td>
  </tr>
</table>


## Limitations & Honest Notes

- The Nickparvar dataset's Train/Test split has documented near-duplicate leakage between slices of the same patient scan; treat very high accuracy numbers (99%+) on this dataset with appropriate skepticism, including this project's own result.
- Both backbones can optionally run from-scratch (no ImageNet pretraining) — this is a much harder training regime on ~5,600 images and yields meaningfully lower accuracy (~93% in earlier experiments). Pretraining is the single largest lever in this pipeline.
- Glioma vs. meningioma remains the primary confusion the model makes; see the confusion matrix above.
- This project is for research/educational purposes and is **not** a clinical diagnostic tool.

## Acknowledgments

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle), aggregating data from figshare, SARTAJ, and Br35H sources.
- Backbone architectures: [ConvNeXt](https://arxiv.org/abs/2201.03545) (Liu et al., 2022) and [Swin Transformer](https://arxiv.org/abs/2103.14030) (Liu et al., 2021).
