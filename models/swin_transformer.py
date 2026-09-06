import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.convnext import DropPath

def window_partition(x, window_size: int):
    """
    Partitions the input feature map into non-overlapping local windows.
    Args:
        x: (B, H, W, C)
        window_size (int): Size of the local window (e.g., 7).
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Permute to (B, H/M, W/M, M, M, C) and reshape to (num_windows * B, M, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Reconstructs the original feature map shape from partitioned windows.
    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size (int): Window size.
        H, W (int): Original height and width of the feature map.
    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    # Permute to (B, H/M, M, W/M, M, C) and reshape to (B, H, W, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block for Swin block projection.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA) with Relative Position Bias.
    Supports shifted window attention (SW-MSA) when supplied with a masking tensor.
    
    Why it is used:
    In standard self-attention, computation scales quadratically. Restricting attention 
    to local windows of size MxM (e.g. 7x7) bounds complexity. The Relative Position Bias
    table enforces spatial grid relationships, ensuring the model remembers pixels'
    relative distances (highly critical for anatomical brain layouts).
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define relative position bias parameter table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor of shape (num_windows * B, N, C) where N = window_size * window_size
            mask: attention mask of shape (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        # QKV projection: (B_, N, 3*C) -> (B_, N, 3, nH, d) -> permute to (3, B_, nH, N, d)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Query, Key, Value shapes: (B_, nH, N, d)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, nH, N, N)

        # Retrieve relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )  # (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)  # Broaden batch dimension

        # If mask is active (Shifted Window attention), apply mask to ignore out-of-boundary tokens
        if mask is not None:
            nW = mask.shape[0]
            # Reshape attn to add window dimension: (B, nW, nH, N, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Compute output: (B_, nH, N, d) -> permute to (B_, N, nH*d)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Alternates between Window Multi-head Self-Attention (W-MSA) and 
    Shifted Window Multi-head Self-Attention (SW-MSA).
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Shift size sanity check
        if min(self.input_resolution) <= self.window_size:
            # If resolution is smaller than window size, do not shift window
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 1. Cyclic shift if using Shifted Window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 2. Partition feature map into windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, window_size*window_size, C)

        # 3. Apply Window Self-Attention (only apply mask if shift_size > 0)
        attn_windows = self.attn(x_windows, mask=attn_mask if self.shift_size > 0 else None)

        # 4. Reverse partition back to grid shape
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        # 5. Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 6. Reshape back to 1D sequence and residual connection
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # 7. Feed-Forward MLP layer
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    Combines 2x2 neighboring patches to downsample resolution by 2x while 
    expanding channels by 2x.
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) is not even"

        x = x.view(B, H, W, C)

        # Slice to group 2x2 grid features
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        
        # Concatenate along channel dimension to get shape: (B, H/2, W/2, 4*C)
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4*C)

        x = self.norm(x)
        x = self.reduction(x)  # Project from 4C to 2C
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer.
    Divides the input 2D image (B, 3, H, W) into patch grids and maps to channel dimension C.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Input shape: (B, 3, 224, 224)
        x = self.proj(x)  # (B, C, 56, 56)
        x = x.flatten(2).transpose(1, 2)  # (B, 56*56, C)
        x = self.norm(x)
        return x


class SwinTransformerTinyScratch(nn.Module):
    """
    Swin Transformer-Tiny architecture implemented completely from scratch.
    
    Structure:
    - Stage 1: Resolution 56x56, 2 blocks, 64 channels
    - Stage 2: Resolution 28x28, 2 blocks, 128 channels
    - Stage 3: Resolution 14x14, 6 blocks, 256 channels
    - Stage 4: Resolution 7x7, 2 blocks, 512 channels
    
    This lightweight default setup (embed_dim=64, depths=[2, 2, 6, 2]) runs
    efficiently on an RTX 2050 (4GB VRAM) and perfectly outputs a (B, 512, 7, 7) 
    feature representation, directly aligning with ConvNeXt-Tiny's output dimensions.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64,
                 depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16], window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.05):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # final stage dimension (512)
        self.window_size = window_size

        # Patch Embedding Stem (224x224x3 -> 56x56x64)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rate rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Swin Stages
        self.layers = nn.ModuleList()
        grid_resolution = self.patch_embed.grid_size  # (56, 56)

        for i_layer in range(self.num_layers):
            stage_dim = int(embed_dim * 2 ** i_layer)
            stage_resolution = (grid_resolution[0] // (2 ** i_layer), grid_resolution[1] // (2 ** i_layer))
            
            # Swin Blocks for current stage
            blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=stage_dim,
                    input_resolution=stage_resolution,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else (window_size // 2),  # Alternate shift
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) + j]
                ) for j in range(depths[i_layer])
            ])
            
            # Downsampling layer between stages (except for final stage)
            downsample = None
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(stage_resolution, dim=stage_dim)
                
            self.layers.append(nn.ModuleDict({
                "blocks": blocks,
                "downsample": downsample
            }))

        self.norm = nn.LayerNorm(self.num_features)
        
        # Precompute attention masks for shifted windows in each stage
        self.register_buffer("attn_mask_stage1", self._get_mask((56, 56), window_size, window_size // 2))
        self.register_buffer("attn_mask_stage2", self._get_mask((28, 28), window_size, window_size // 2))
        self.register_buffer("attn_mask_stage3", self._get_mask((14, 14), window_size, window_size // 2))
        self.register_buffer("attn_mask_stage4", self._get_mask((7, 7), window_size, window_size // 2))

        self.apply(self._init_weights)

    def _get_mask(self, resolution, window_size, shift_size):
        """Helper to precalculate the attention masks for shifted window self-attention."""
        device = torch.device("cpu")  # Calculate on CPU and transfer
        H, W = resolution
        # Calculate matching padded size
        Hp = int(np.ceil(H / window_size)) * window_size
        Wp = int(np.ceil(W / window_size)) * window_size
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

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
        Input: (B, 3, 224, 224)
        Output: (B, 512, 7, 7) feature maps.
        """
        # Patch embedding
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # Cache masks list corresponding to each stage
        masks = [
            self.attn_mask_stage1,
            self.attn_mask_stage2,
            self.attn_mask_stage3,
            self.attn_mask_stage4
        ]

        # Execute stages
        for i_layer, layer in enumerate(self.layers):
            blocks = layer["blocks"]
            downsample = layer["downsample"]
            
            # Select correct resolution mask for shifts
            mask = masks[i_layer]
            
            for j, block in enumerate(blocks):
                # Only alternate blocks (j % 2 != 0) use Shifted Window with masking
                if j % 2 != 0:
                    x = block(x, attn_mask=mask)
                else:
                    x = block(x, attn_mask=None)
                    
            if downsample is not None:
                x = downsample(x)

        x = self.norm(x)  # (B, 49, 512)
        
        # Reshape sequence of tokens back to (B, C, H, W) format for cross attention fusion
        B = x.shape[0]
        # output resolution at stage 4 is 7x7
        x = x.transpose(1, 2).view(B, self.num_features, 7, 7).contiguous()
        return x
