"""
ImageNet-pretrained backbone wrappers for CSCAN.

WHY THIS FILE EXISTS
---------------------
The original ConvNeXtTinyScratch / SwinTransformerTinyScratch backbones are
trained entirely from random initialization on ~5,600 MRI images. A hybrid
CNN+Transformer with cross-attention has tens of millions of parameters;
Swin-style self-attention in particular has very little inductive bias and
needs either a huge dataset or a pretrained initialization to generalize well.
Training it from scratch on 5.6k images is the single biggest reason accuracy
plateaus well below what the architecture is capable of.

These wrappers load the standard torchvision ImageNet-1k weights for
ConvNeXt-Tiny and Swin-T, then expose a `forward(x) -> (B, 768, 7, 7)` feature
map, identical in shape/contract to the scratch backbones, so they are a
drop-in replacement inside CSCAN. Only the stem needs no change since MRI is
resized to 224x224, 3-channel, which is exactly what these nets expect.
"""

import torch
import torch.nn as nn
import torchvision


class ConvNeXtTinyPretrained(nn.Module):
    """torchvision convnext_tiny (IMAGENET1K_V1), truncated before global pooling.
    Output: (B, 768, 7, 7) for a 224x224 input.
    """

    def __init__(self, drop_path_rate: float = 0.1, freeze_stages: int = 0):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        net = torchvision.models.convnext_tiny(weights=weights)

        # `features` is the conv trunk; torchvision's classifier/avgpool are dropped.
        self.features = net.features  # -> (B, 768, 7, 7) on 224x224 input

        # Re-apply the requested stochastic-depth rate on the pretrained trunk.
        # torchvision builds each CNBlock with its own DropPath; we simply patch
        # the probabilities so fine-tuning on a small dataset isn't over/under
        # regularized relative to the from-scratch counterpart.
        for m in self.features.modules():
            if m.__class__.__name__ == "StochasticDepth":
                m.p = drop_path_rate

        if freeze_stages > 0:
            # features is a Sequential of 8 items: stem + 3x(downsample+stage) pairs.
            # freeze_stages=N freezes the first N top-level children (stem counts as 1).
            for idx, child in enumerate(self.features.children()):
                if idx < freeze_stages:
                    for p in child.parameters():
                        p.requires_grad = False

    def forward(self, x):
        return self.features(x)


class SwinTinyPretrained(nn.Module):
    """torchvision swin_t (IMAGENET1K_V1), truncated before pooling/head.
    Re-projects to (B, 768, 7, 7) so it matches ConvNeXt-Tiny spatially and
    channel-wise for cross-attention fusion.
    """

    def __init__(self, drop_path_rate: float = 0.1, freeze_stages: int = 0):
        super().__init__()
        weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
        net = torchvision.models.swin_t(weights=weights)

        self.features = net.features  # sequential stages, channels-last internally
        self.norm = net.norm          # final LayerNorm(768)

        for m in self.features.modules():
            if m.__class__.__name__ == "StochasticDepth":
                m.p = drop_path_rate

        if freeze_stages > 0:
            for idx, child in enumerate(self.features.children()):
                if idx < freeze_stages:
                    for p in child.parameters():
                        p.requires_grad = False

    def forward(self, x):
        # torchvision's swin_t.features returns (B, H, W, C) = (B, 7, 7, 768)
        feat = self.features(x)
        feat = self.norm(feat)
        # Convert to (B, C, H, W) so downstream fusion code (written for
        # channels-first conv feature maps) doesn't need to change.
        feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat
