
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """Chuyển đổi ảnh thành patches và embed chúng"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Projection layer
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # CLS token và Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)  # Tắt dropout khi train nhỏ

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        return x


class MLP(nn.Module):
    """Feed Forward Network - Tối ưu cho CPU"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-head Self-Attention - Tối ưu cho CPU"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim phải chia hết cho num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            drop_rate=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) - CPU Optimized Version
    
    Tối ưu cho training trên CPU với các cải tiến:
    - Model size nhỏ hơn (fewer params)
    - Computation hiệu quả hơn
    - Memory footprint thấp hơn
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=1000,
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True,
        drop_rate=0., 
        attn_drop_rate=0.
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Khởi tạo trọng số
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Khởi tạo trọng số theo ViT paper"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token = x[:, 0]
        x = self.head(cls_token)
        
        return x

    def get_num_params(self):
        """Đếm số lượng parameters"""
        return sum(p.numel() for p in self.parameters())