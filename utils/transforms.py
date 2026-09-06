import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class ApplyCLAHE:
    """
    A custom PyTorch-compatible transform that applies Contrast Limited Adaptive
    Histogram Equalization (CLAHE) to MRI images.

    Why it is used:
    Standard brain MRI scans often exhibit variable contrast and bias fields.
    Global histogram equalization over-amplifies background noise (such as the black
    space around the skull). CLAHE enhances local details and tumor boundaries by
    processing the image in small tiles (default 8x8) and clipping high contrast peaks.

    How it works:
    1. If the image is RGB, converts to LAB color space.
    2. Applies CLAHE to the L (Lightness) channel.
    3. Merges back to RGB.
    4. If grayscale, applies CLAHE directly.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)

        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_enhanced = clahe.apply(img_np)

        return Image.fromarray(img_enhanced)


# Normalization constants shared by BOTH train and val pipelines.
# Using 0.5/0.5 maps pixel values to [-1, +1], appropriate for scratch training
# without ImageNet pretrained weights. CRITICAL: train and val MUST use the same
# values, otherwise validation is measured on an out-of-distribution input.
_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD  = [0.5, 0.5, 0.5]


def get_train_transforms(image_size: int) -> transforms.Compose:
    """
    Returns the training transformation pipeline.

    Preprocessing flow:
    Input -> CLAHE -> Augmentation -> Resize -> Z-score Normalization
    """
    return transforms.Compose([
        ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),   # ±5% translation
            scale=(0.95, 1.05)        # ±5% zoom
        ),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])


def get_val_test_transforms(image_size: int) -> transforms.Compose:
    """
    Returns the validation/test transformation pipeline.
    
    Preprocessing flow:
    Input -> CLAHE -> Resize -> Z-score Normalization
    """
    return transforms.Compose([
        ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])
