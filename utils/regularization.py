"""
MixUp / CutMix batch augmentation and Exponential Moving Average (EMA) of
model weights.

Both are standard, well-tested techniques for squeezing extra generalization
out of a fixed, small dataset without touching the architecture:

- MixUp/CutMix synthesize new (image, soft-label) training pairs by blending
  two real samples. This smooths the decision boundary and is one of the
  most reliable regularizers when you only have a few thousand images per
  class, which is exactly this dataset's regime (1,400/class).
- EMA keeps a slow-moving average of the weights seen during training. The
  averaged weights are typically less noisy than the final raw weights and
  often generalize slightly better, especially with cosine LR schedules.
"""

import copy
import numpy as np
import torch


def rand_bbox(size, lam):
    """Sample a random bounding box for CutMix given the image size and mix ratio."""
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)

    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    return y1, y2, x1, x2


def mixup_cutmix(images, labels, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    """
    Randomly applies MixUp or CutMix to a batch.

    Returns:
        mixed_images, (labels_a_onehot, labels_b_onehot, lam)
        Use `lam * loss(out, labels_a) + (1-lam) * loss(out, labels_b)` with a
        soft-label-capable loss (see utils.losses.soft_cross_entropy), or the
        provided helper `mixup_criterion` below.
    """
    B = images.size(0)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes).float()

    if np.random.rand() > prob:
        # No mixing this batch — return as-is (lam=1 -> pure labels_a)
        return images, (labels_onehot, labels_onehot, 1.0)

    use_cutmix = np.random.rand() < 0.5 and cutmix_alpha > 0
    perm = torch.randperm(B, device=images.device)

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        y1, y2, x1, x2 = rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        # Adjust lam to the actual pixel ratio replaced
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        images = lam * images + (1 - lam) * images[perm]

    labels_b = labels_onehot[perm]
    return images, (labels_onehot, labels_b, lam)


def mixup_criterion(criterion_soft, logits, mix_labels):
    """Applies a soft-label criterion to a MixUp/CutMix-mixed batch."""
    labels_a, labels_b, lam = mix_labels
    target = lam * labels_a + (1 - lam) * labels_b
    return criterion_soft(logits, target)


class ModelEMA:
    """Maintains an exponential moving average of a model's parameters.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        ...
        loss.backward(); optimizer.step()
        ema.update(model)          # call once per training step
        ...
        ema.apply_to(eval_model)   # load EMA weights into a model for eval
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            model_v = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(model_v, alpha=1 - self.decay)
            else:
                v.copy_(model_v)

    def state_dict(self):
        return self.shadow.state_dict()
