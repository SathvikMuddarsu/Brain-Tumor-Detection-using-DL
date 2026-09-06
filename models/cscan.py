import torch
import torch.nn as nn
from models.convnext import ConvNeXtTinyScratch
from models.swin_transformer import SwinTransformerTinyScratch
from models.pretrained_backbones import ConvNeXtTinyPretrained, SwinTinyPretrained
from models.cross_attention import CrossAttentionFusion
from models.dcrf import DiscriminativeConfidenceRefinementFusion
from models.classifier import CSCANClassifier
from utils.config import CSCANConfig


class AdaptiveWeightedFusion(nn.Module):
    """
    Adaptive Weighted Fusion (AWF) module:
    F_AWF = alpha * F_ConvNeXt + (1 - alpha) * F_Swin
    where alpha in (0, 1)^C is a dynamic per-channel gating factor.
    """
    def __init__(self, dim=768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_conv, feat_swin):
        concat_feat = torch.cat([feat_conv, feat_swin], dim=1)  # (B, 2*C, 1, 1)
        alpha = self.gate(concat_feat)                         # (B, C, 1, 1)
        fused = alpha * feat_conv + (1.0 - alpha) * feat_swin
        return fused, alpha


class CSCAN(nn.Module):
    """
    CSCAN-DCRF v2: ConvNeXt–Swin Cross Attention Network with
    Adaptive Weighted Fusion, Discriminative Confidence Refinement Fusion,
    GeM Pooling (p=3.0), and LayerNorm MLP Classifier Head.

    Flow:
        Input (B, 3, 224, 224)
            ├── ConvNeXt-Tiny → (B, 768, 7, 7)
            └── Swin-Tiny     → (B, 768, 7, 7)
                        ↓
              Adaptive Weighted Fusion → (B, 768, 7, 7)
                        ↓
              Pre-LayerNorm Cross Attention Fusion → (B, 768, 7, 7)
                        ↓
              DCRF (Spatial Margin Confidence Refinement) → (B, 768, 7, 7)
                        ↓
              GeM Pooling (p=3.0) → (B, 768)
                        ↓
              LayerNorm + MLP Classifier Head → (B, 4)
    """

    def __init__(self, num_classes=CSCANConfig.NUM_CLASSES,
                 img_size=CSCANConfig.IMAGE_SIZE,
                 dropout_rate=CSCANConfig.DROPOUT_RATE,
                 pretrained=CSCANConfig.PRETRAINED,
                 freeze_stages=CSCANConfig.FREEZE_STAGES,
                 use_dcrf=CSCANConfig.USE_DCRF):
        super().__init__()

        FEAT_DIM = 768
        self.use_dcrf = use_dcrf
        self.pretrained = pretrained

        if pretrained:
            self.convnext = ConvNeXtTinyPretrained(
                drop_path_rate=0.1,
                freeze_stages=freeze_stages,
            )
            self.swin = SwinTinyPretrained(
                drop_path_rate=0.1,
                freeze_stages=freeze_stages,
            )
        else:
            self.convnext = ConvNeXtTinyScratch(
                in_chans=3,
                depths=[2, 2, 6, 2],
                dims=[96, 192, 384, 768],
                drop_path_rate=0.05,
            )
            self.swin = SwinTransformerTinyScratch(
                img_size=img_size,
                patch_size=4,
                in_chans=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.05,
            )

        # 3. Adaptive Weighted Fusion
        self.awf = AdaptiveWeightedFusion(dim=FEAT_DIM)

        # 4. Cross Attention Fusion Module (with Pre-LayerNorm stabilization)
        self.fusion = CrossAttentionFusion(
            dim=FEAT_DIM,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.05,
        )

        # 5. DCRF module
        if self.use_dcrf:
            self.dcrf = DiscriminativeConfidenceRefinementFusion(
                dim=FEAT_DIM,
                num_classes=num_classes,
                dropout_rate=0.1,
            )
        else:
            self.dcrf = None

        # 6. Classification Head (GeM p=3.0 -> LayerNorm -> MLP Head)
        self.classifier = CSCANClassifier(
            in_features=FEAT_DIM,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_gem=True,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input MRI image, shape (B, 3, 224, 224)
        Returns:
            logits (Tensor): Class logits, shape (B, 4)
        """
        # 1. Parallel backbone feature extraction
        feat_local  = self.convnext(x)   # (B, 768, 7, 7)
        feat_global = self.swin(x)        # (B, 768, 7, 7)

        # 2. Adaptive Weighted Fusion
        feat_awf, _ = self.awf(feat_local, feat_global)  # (B, 768, 7, 7)

        # 3. Cross-attention fusion (ConvNeXt=Query, Swin=Key/Value) with Pre-LN
        feat_fused = self.fusion(feat_awf, feat_global)  # (B, 768, 7, 7)

        # 4. DCRF spatial margin refinement
        feat_refined = self.dcrf(feat_fused) if self.dcrf is not None else feat_fused

        # 5. GeM Pooling + LayerNorm + MLP Classifier
        logits = self.classifier(feat_refined)   # (B, 4)

        return logits
