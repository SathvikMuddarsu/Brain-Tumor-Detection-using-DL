import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross Attention Fusion Module for CSCAN.

    Q-K-V Routing Logic:
    - ConvNeXt Feature Map (Local) -> Query (Q)
      Local texture/boundary details drive the attention search.
    - Swin Feature Map (Global) -> Key (K) and Value (V)
      Global anatomical context serves as the reference dictionary.

    Residual Connection (FIXED):
    The correct cross-attention residual adds only the QUERY input back to
    the attended output:
        fused = x_Q + Attention(Q, K, V)
    The previous formula  `0.5*(x_c + x_s) + out`  was incorrect: it injected
    raw, unweighted Swin features through the residual path, bypassing the
    attention mechanism and creating a shortcut that prevents the module from
    learning meaningful feature alignment.

    Dropout (FIXED):
    Reduced from 0.1/0.1 to 0.0/0.05 to avoid over-regularizing the only
    feature fusion module in the network.
    """

    def __init__(self, dim=512, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.05):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Linear projection layers
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Pre-attention LayerNorm for both input streams
        self.norm_conv = nn.LayerNorm(dim)
        self.norm_swin = nn.LayerNorm(dim)

        # Post-attention LayerNorm (applied before MLP refinement)
        self.norm_after_attn = nn.LayerNorm(dim)

        # Feed-Forward MLP refinement block (standard Transformer FFN design)
        self.norm_refine = nn.LayerNorm(dim)
        self.refine_mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(4 * dim, dim),
            nn.Dropout(proj_drop),
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

    def forward(self, x_conv, x_swin):
        """
        Args:
            x_conv (Tensor): Local CNN features, shape (B, C, H, W)
            x_swin (Tensor): Global Transformer features, shape (B, C, H, W)
        Returns:
            fused_features (Tensor): Fused output, shape (B, C, H, W)
        """
        B, C, H, W = x_conv.shape
        N = H * W

        # Flatten spatial dimensions to token sequences
        x_c = x_conv.view(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)
        x_s = x_swin.view(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)

        # Pre-attention LayerNorm (standard Transformer pre-norm design)
        x_c_norm = self.norm_conv(x_c)
        x_s_norm = self.norm_swin(x_s)

        # QKV projections
        q = self.q_proj(x_c_norm).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x_s_norm).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x_s_norm).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)           # (B, nH, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # CORRECTED residual: add only the query (ConvNeXt) stream back.
        # This preserves the attention's learned alignment signal without injecting
        # unweighted Swin features through a bypass shortcut.
        fused = x_c + out  # (B, N, C)

        # Post-attention LayerNorm
        fused = self.norm_after_attn(fused)

        # MLP Feed-Forward refinement with residual
        fused = fused + self.refine_mlp(self.norm_refine(fused))

        # Reshape back to spatial feature map format
        fused = fused.transpose(1, 2).view(B, C, H, W).contiguous()

        return fused