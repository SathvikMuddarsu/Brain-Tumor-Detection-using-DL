# CSCAN-DCRF: Changes made and honest expectations

## First, two corrections on the ask

**LIME doesn't affect accuracy.** LIME (Local Interpretable Model-agnostic
Explanations) is a post-hoc *explainability* tool — it highlights which
pixels influenced a prediction, for interpretability/trust, not a
preprocessing or training technique. It cannot raise test accuracy. If you
want it for the "explainability" section of a paper/report, it's easy to add
on top of the trained model, but it's a separate concern from the 93%→higher
question.

**CLAHE is already implemented** in `utils/transforms.py` (`ApplyCLAHE`,
applied to both train and test pipelines). Nothing to add there.

**99% is not something I can guarantee**, and you should be skeptical of any
tool or notebook that promises it outright — see the dataset-leakage note
below. I did not retrain the model in this environment (no GPU here, and this
run needs several hours on real hardware), so none of the numbers below are
verified; they're realistic based on how this class of change behaves on
this class of problem.

## Why 93% is actually a reasonable result for this codebase

The most important fact about the current pipeline: **both backbones
(ConvNeXt-Tiny and Swin-Tiny) are trained entirely from random
initialization** — no ImageNet pretraining anywhere (I checked; there's no
`pretrained=`, `state_dict` loading, or `timm`/`torchvision.models` weights
call in the original `models/convnext.py` or `models/swin_transformer.py`).

A ConvNeXt+Swin hybrid with cross-attention has ~60M+ parameters. Training
that from scratch on 5,600 images (1,400/class) is a genuinely hard regime,
especially for the Swin half — self-attention has very little built-in
inductive bias and normally needs either a huge dataset or a pretrained
initialization to generalize well. 93% from scratch on this data is a
believable, solid result, not a sign that anything is broken.

**This is the actual lever, more than any of the architectural additions
you listed** (yet another attention variant, another preprocessing filter,
etc.). Adding more novel modules to a model that's already undertrained for
its capacity tends to add parameters and instability, not accuracy.

## What I changed

All changes are toggleable via `utils/config.py` so you can ablate each one.

1. **ImageNet-pretrained backbones** (`models/pretrained_backbones.py`,
   `CSCANConfig.PRETRAINED = True`). Wraps torchvision's `convnext_tiny` and
   `swin_t` (both ImageNet-1k weights) so they output the same `(B, 768, 7, 7)`
   feature map contract as the original scratch backbones — drop-in
   replacement, cross-attention/DCRF/classifier code is untouched. This is
   the highest-leverage change here by a wide margin.
2. **Discriminative learning rates** (`train.py: build_param_groups`).
   Pretrained backbone params get `BACKBONE_LEARNING_RATE=3e-5`; the
   newly-initialized fusion/DCRF/classifier get `LEARNING_RATE=3e-4`. Prevents
   the first few noisy gradient steps from wrecking the pretrained features.
3. **Freeze → unfreeze schedule** (`FREEZE_STAGES=1`, `UNFREEZE_AT_EPOCH=5`).
   Stem/stage-1 (generic edge/texture filters) stays frozen for the first 5
   epochs while the head "warms up," then everything fine-tunes together.
4. **MixUp / CutMix** (`utils/regularization.py`, `MIXUP_ALPHA`,
   `CUTMIX_ALPHA`, `MIX_PROB=0.5`). Blends random pairs of training images
   and soft-blends their labels. Meaningfully reduces overfitting when you
   only have ~1,400 images per class — this is one of the best-evidenced
   regularizers for exactly this data regime.
5. **EMA of model weights** (`ModelEMA`, `USE_EMA=True`). Each epoch, both
   the raw and EMA-averaged weights are validated and the better one is
   checkpointed. Free, small, consistent accuracy/stability improvement.
6. **DCRF re-enabled** (`USE_DCRF=True`). The prior ablation disabling it ran
   on the from-scratch, weaker backbones; it's worth re-testing now that the
   features going into it are pretrained and more discriminative. It's a
   config flag — flip it off and retrain if your own ablation shows it still
   hurts.
7. **Early stopping** (`EARLY_STOPPING_PATIENCE=10`) + fewer max epochs
   (`EPOCHS=40` vs. 100) — pretrained models converge much faster and 100
   epochs of fine-tuning on 5.6k images is asking to overfit.
8. **Test-Time Augmentation** (`test.py`, `USE_TTA=True`) — averages
   softmax probabilities over each test image and its horizontal flip.
   Zero-cost accuracy at inference; no retraining needed. (Consistent with
   the existing `RandomHorizontalFlip` training augmentation's assumption
   that MRI is roughly left-right symmetric.)
9. Higher weight decay (`5e-2`, standard for ConvNeXt/ViT-style fine-tuning
   recipes) and slightly higher head dropout (`0.3`) since pretrained
   features are richer and can overfit faster on a small dataset.

## What I did *not* add, and why

- **Another architecture bolted on** (e.g. a third backbone, another
  attention block): more parameters on an already-undertrained model tends
  to hurt, not help, especially on 5.6k images. If pretrained fine-tuning
  plateaus below your target, the next thing to try is a *stronger single
  pretrained backbone* (e.g. ConvNeXt-Small/Base, or Swin-S) rather than
  more architecture on top of Tiny models.
- **LIME**: explainability tool, not an accuracy technique (see above).
  Happy to add a LIME/Grad-CAM visualization script separately if useful for
  a report.

## Please read this before you report a 99% number

The dataset here (4,000 train / 1,311 or 1,600 test brain MRI slices,
Nickparvar's Kaggle "Brain Tumor MRI Dataset") is well known in the Kaggle
community for near-duplicate or highly similar slices leaking between the
official Train/Test split (adjacent slices from the same patient scan look
almost identical). This is why some public notebooks report 99%+ accuracy
even with fairly simple models — it can reflect memorized near-duplicates
more than genuine generalization. Two honest ways to guard against this if
you'll be defending this number (thesis, paper, viva):
- Do a quick manual/perceptual-hash duplicate check between your Train and
  Test folders before trusting a very high test number.
- If possible, report a stratified k-fold cross-validation accuracy on the
  combined data in addition to the fixed Train/Test split — it's a more
  defensible number and reviewers/examiners increasingly ask for it on this
  exact dataset.

## How to run

```bash
pip install -r requirements.txt
python train.py    # will download ImageNet weights for ConvNeXt-Tiny/Swin-T
                    # on first run (needs internet access)
python test.py
```

Everything is driven by `utils/config.py`. To reproduce the *original*
from-scratch behavior for comparison, set `PRETRAINED = False`,
`USE_DCRF = False`, `MIXUP_ALPHA = CUTMIX_ALPHA = 0`, `USE_EMA = False`,
`EPOCHS = 100`, `LEARNING_RATE = 1e-4`, `WEIGHT_DECAY = 1e-4`,
`DROPOUT_RATE = 0.2`.

## Dataset folder

This zip does **not** include `dataset/` (thousands of MRI images, unchanged
from what you uploaded) to keep the download small. Drop your existing
`dataset/Train` and `dataset/Test` folders back into the project root before
running.
