import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    Custom Layer Normalization that supports two data formats: 
    'channels_last' (N, H, W, C) or 'channels_first' (N, C, H, W).
    
    Why it is used:
    PyTorch's native nn.LayerNorm only supports normalized_shape on the last dimensions.
    ConvNeXt mixed operation flow alternates between channels_first (for convolutions)
    and channels_last (for MLP blocks). This custom layer avoids constant permute calls,
    speeding up training.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError("Data format must be 'channels_last' or 'channels_first'")
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Stochastic Depth per sample.
    Randomly drops residual connections during training to regularize deep networks.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize to 0 or 1
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) layer module.
    
    Why it is used:
    In clinical MRI training, overparameterized networks easily overfit. DropPath
    regularizes training by randomly disabling entire paths/blocks, encouraging
    robust feature redundancy.
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(nn.Module):
    """
    The core ConvNeXt Block design.
    Preserves the original ConvNeXt structural choices:
    1. Depthwise Convolution with 7x7 kernel (captures local texture and edges over a larger area).
    2. LayerNorm (replaces BatchNorm for training stability).
    3. 1x1 linear layer expansion (4x dimension - inverted bottleneck).
    4. GELU Activation (more continuous gradient flow than ReLU).
    5. 1x1 linear layer contraction.
    6. LayerScale (residual scaling) & Stochastic Depth (DropPath).
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise convolution: groups = dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm applied in channels_last format
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # 1x1 projection layers implemented as Linear layers for faster execution in channels_last
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # LayerScale parameter: stabilizes deep residual network convergence
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        
        # 1. Depthwise 7x7 Conv (captures spatial boundaries)
        x = self.dwconv(x)
        
        # 2. Convert (N, C, H, W) to (N, H, W, C) for normalization and linear projections
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        # 3. Expansion (Inverted Bottleneck)
        x = self.pwconv1(x)
        x = self.act(x)
        
        # 4. Contraction
        x = self.pwconv2(x)
        
        # 5. LayerScale application if active
        if self.gamma is not None:
            x = self.gamma * x
            
        # 6. Convert back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 7. Residual addition with Stochastic Depth
        x = residual + self.drop_path(x)
        return x


class ConvNeXtTinyScratch(nn.Module):
    """
    ConvNeXt-Tiny architecture implemented completely from scratch.
    
    Structure:
    - Stem: 4x4 non-overlapping strided convolution to downsample 224x224 input to 56x56.
    - 4 Sequential Stages:
      - Stage 1: depths[0] blocks, dims[0] channels (H/4, W/4)
      - Stage 2: depths[1] blocks, dims[1] channels (H/8, W/8)
      - Stage 3: depths[2] blocks, dims[2] channels (H/16, W/16)
      - Stage 4: depths[3] blocks, dims[3] channels (H/32, W/32)
    - Downsampling: LayerNorm + 2x2 Conv with stride 2 between stages.
    
    GPU Optimization:
    We allow custom configurations of `depths` and `dims`.
    - Original ConvNeXt-Tiny: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
    - Lightweight RTX 2050 default: depths=[2, 2, 4, 2], dims=[32, 64, 128, 256] (Highly efficient)
    """
    def __init__(self, in_chans=3, depths=[2, 2, 4, 2], dims=[32,64,128,256], 
                 drop_path_rate=0.1, layer_scale_init_value=1e-6):
        super().__init__()
        
        self.dims = dims
        self.depths = depths
        
        # Stem: Downsamples image 4x (224x224 -> 56x56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Downsampling layers between stages (stride=2 convolutions)
        self.downsample_layers = nn.ModuleList()
        # Stem is the first downsampler
        self.downsample_layers.append(self.stem)
        
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)
            
        # Architecture Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            stage_blocks = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value
                ) for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur += depths[i]
            
        # Final Norm layer (we normalize feature maps before feeding to cross attention)
        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        
        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Executes the network stages sequentially.
        Input: (B, 3, 224, 224)
        Output: (B, dims[-1], 7, 7) feature map representation.
        """
        for i in range(4):
            # Downsample (except first stage where self.downsample_layers[0] is the stem)
            x = self.downsample_layers[i](x)
            # Process block stack for current stage
            x = self.stages[i](x)
            
        x = self.norm(x)
        return x
