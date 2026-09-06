import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPooling2d(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer with learnable exponent p.
    Formula: f_c = (1/HW * sum(x_{c,i,j}^p))^(1/p)
    Initialized to p = 3.0 (empirically optimal for visual recognition tasks).
    When p=1 -> Global Average Pooling (GAP)
    When p->inf -> Global Max Pooling (GMP)
    """
    def __init__(self, p=3.0, eps=1e-6, clamp_p=True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([float(p)]), requires_grad=True)
        self.eps = eps
        self.clamp_p = clamp_p

    def forward(self, x):
        p = torch.clamp(self.p, min=1.0, max=10.0) if self.clamp_p else self.p
        x_clamped = x.clamp(min=self.eps)
        pooled = F.avg_pool2d(x_clamped.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
        return pooled


class CSCANClassifier(nn.Module):
    """
    Publication-Quality Classification Head for CSCAN.
    
    Structure:
    1. GeM Pooling (p=3.0) -> Output shape: (B, C, 1, 1) -> Flatten to (B, C)
    2. LayerNorm(C) -> Normalizes features prior to classifier projection
    3. MLP Head: Linear(C -> C//2) -> GELU -> Dropout -> Linear(C//2 -> num_classes)
    """
    def __init__(self, in_features=768, num_classes=4, dropout_rate=0.2, use_gem=True):
        super().__init__()
        self.use_gem = use_gem
        if use_gem:
            self.pooling = GeMPooling2d(p=3.0)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.gmp = nn.AdaptiveMaxPool2d(1)
            self.pooling = None

        # Pre-MLP LayerNorm for feature stabilization
        self.norm = nn.LayerNorm(in_features)
        
        # 2-Layer MLP Head with GELU nonlinearity and Dropout
        hidden_dim = in_features // 2
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Feature map tensor, shape (B, C, H, W) e.g. (B, 768, 7, 7)
        Returns:
            logits (Tensor): Class logits, shape (B, num_classes)
        """
        if self.pooling is not None:
            x = self.pooling(x)  # (B, C, 1, 1)
        else:
            x = 0.5 * (self.gap(x) + self.gmp(x))
            
        x = torch.flatten(x, 1)   # (B, C)
        x = self.norm(x)          # LayerNorm
        logits = self.mlp(x)      # MLP Head -> (B, num_classes)
        return logits

    def get_probabilities(self, x):
        """
        Runs forward pass and applies Softmax to yield normalized probabilities.
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
