import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import CSCANConfig

class DiscriminativeConfidenceRefinementFusion(nn.Module):
    """
    Discriminative Confidence Refinement Fusion (DCRF) module.
    
    Why it is used:
    Brain tumor classification (specifically Glioma vs Meningioma) suffers from 
    class boundary ambiguity and low-confidence visual features. DCRF acts as a 
    lightweight, confidence-guided spatial refinement module that identifies 
    ambiguous zones and uses neighborhood context to resolve them.
    
    Internal Block Layout:
    1. Discriminative Confidence Estimation: Projects features to local class logits 
       and computes Margin Confidence (top-1 probability - top-2 probability).
    2. Adaptive Feature Reweighting: Uses confidence maps to divide features into 
       confident and ambiguous spatial channels.
    3. Context Refinement: Refines ambiguous boundary features using a $3\times3$ 
       depthwise convolution to capture neighborhood tissue structural relations.
    4. Residual Fusion: Recombines refined features with original inputs via a 
       learnable residual scaling factor.
    """
    def __init__(self, dim=512, num_classes=CSCANConfig.NUM_CLASSES, dropout_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        
        # --- 1. Discriminative Confidence Estimation Layers ---
        # Projects 512 channels down to class logit scores at each spatial coordinate
        self.score_proj = nn.Conv2d(dim, num_classes, kernel_size=1)
        
        # --- 2. Adaptive Feature Reweighting Layers ---
        # Maps the 1-channel margin confidence value to a gate value
        # Maps the 1-channel margin confidence value to a gate value
        self.gate_proj = nn.Conv2d(1, 1, kernel_size=1)

        # Normalize the confidence map before gating
        self.gate_norm = nn.GroupNorm(num_groups=1, num_channels=1)
        
        # --- 3. Context Refinement Layers ---
        # A 3x3 depthwise convolution that aggregates local spatial context in ambiguous areas
        # Groups = dim (512) makes it depthwise (very lightweight, 512 * 9 parameters)
        self.context_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        
        # --- 4. Residual Fusion Layer ---
        # Learnable scaling factor initialized to a small positive value (0.01) 
        # to allow gradient flow from day one while keeping the identity path dominant.
        self.gamma = nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def estimate_confidence(self, x):
        """
        Step 1: Discriminative Confidence Estimation.
        Projects features to pseudo-class scores and calculates Margin Confidence.
        
        Formula:
        Margin Confidence = top-1 probability - top-2 probability
        """
        # Project channel dimensions: (B, 512, 7, 7) -> (B, 4, 7, 7)
        local_scores = self.score_proj(x.detach())
        
        # Compute local pseudo-probabilities across classes
        local_probs = F.softmax(local_scores, dim=1)
        
        # Sort probabilities to extract top-1 and top-2 predictions
        top_probs, _ = torch.topk(local_probs, k=2, dim=1)
        
        # Margin confidence (B, 1, 7, 7)
        margin_confidence = top_probs[:, 0:1] - top_probs[:, 1:2]
        
        return margin_confidence

    def adaptive_reweighting(self, x, margin_confidence):
        """
        Step 2: Adaptive Feature Reweighting.
        Calculates gating masks and splits features into confident and ambiguous paths.
        """
        # Pass margin confidence through a Conv1x1 and Sigmoid to get Confident Gate: (B, 1, 7, 7)
        margin_confidence = self.gate_norm(margin_confidence)
        gate_conf = torch.sigmoid(self.gate_proj(margin_confidence))
        
        # Confident Path: Emphasizes highly discriminative tumor areas
        x_conf = gate_conf * x
        
        # Ambiguous Path: Pinpoints areas with low confidence (e.g. tumor boundaries)
        x_ambig = (1.0 - gate_conf) * x
        
        return x_conf, x_ambig

    def refine_context(self, x_ambig):
        """
        Step 3: Context Refinement.
        Applies a local 3x3 depthwise convolution to gather neighboring boundary structure
        to resolve classification confusion.
        """
        # 1. Apply depthwise context convolution
        refined = self.context_conv(x_ambig)
        refined = self.act(refined)
        
        # 2. Normalize features for stable gradient flow
        # Permute to channels-last for LayerNorm, then permute back
        B, C, H, W = refined.shape
        refined = refined.permute(0, 2, 3, 1).contiguous()
        refined = self.norm(refined)
        refined = refined.permute(0, 3, 1, 2).contiguous()
        
        return refined

    def residual_fusion(self, x, x_refined):
        """
        Step 4: Residual Fusion.
        Fuses the refined representation back with the input using a learnable residual gate.
        """
        # Out = X + gamma * X_refined
        out = x + self.gamma * x_refined
        return out

    def forward(self, x):
        """
        Input: (B, 512, 7, 7) feature maps.
        Output: (B, 512, 7, 7) refined feature maps.
        """
        # 1. Estimate prediction confidence
        margin_confidence = self.estimate_confidence(x)
        
        # 2. Split features via adaptive gating reweighting
        x_conf, x_ambig = self.adaptive_reweighting(x, margin_confidence)
        
        # 3. Refine spatial boundary context for ambiguous zones
        x_refined_ambig = self.refine_context(x_ambig)
        
        # 4. Fuse paths together
        # We combine the direct confident features with the refined ambiguous boundary features
        x_refined_total = x_conf + x_refined_ambig
        
        # 5. Residual connection recombination
        out = self.residual_fusion(x, x_refined_total)
        
        return out
