"""PyTorch CNN -> Vision Transformer hybrid.

Pipeline:

1. ``ConvNet.forward_features`` produces a (B, 1024, H', W') feature map.
2. :class:`PatchEmbed` (a 1x1 convolution) projects each spatial location to an
   ``embed_dim`` token, giving a (B, L, embed_dim) sequence.
3. A learnable class token is prepended and a learnable positional embedding
   (sized ``max_tokens``) is added.
4. ``depth`` pre-norm transformer encoder blocks with multi-head self-attention
   and a GELU MLP refine the sequence.
5. The class token is layer-normed and linearly projected to the class logits.

Defaults come from :class:`src.config.ViTConfig` (the PyTorch hybrid lab):
``embed_dim=768``, ``depth=3``, ``num_heads=6``, ``mlp_ratio=4``, ``dropout=0.1``.

``torch`` is imported lazily via PEP 562 module ``__getattr__``.
"""

from __future__ import annotations

from .. import config
from .pytorch_cnn import build_pytorch_cnn

_CLASSES: dict[str, type] = {}


def _build_classes() -> dict[str, type]:
    import torch
    import torch.nn as nn

    class PatchEmbed(nn.Module):
        """Project a CNN feature map to a token sequence with a 1x1 conv."""

        def __init__(self, in_channels: int, embed_dim: int) -> None:
            super().__init__()
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # (B, C, H, W) -> (B, H*W, embed_dim)
            return self.proj(x).flatten(2).transpose(1, 2)

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(f"embed_dim {dim} must be divisible by num_heads {num_heads}")
            self.num_heads = num_heads
            self.scale = (dim // num_heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.attn_drop = nn.Dropout(dropout)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            b, n, d = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.reshape(b, n, self.num_heads, -1).transpose(1, 2)
            k = k.reshape(b, n, self.num_heads, -1).transpose(1, 2)
            v = v.reshape(b, n, self.num_heads, -1).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            out = (attn @ v).transpose(1, 2).reshape(b, n, d)
            return self.proj_drop(self.proj(out))

    class TransformerEncoderBlock(nn.Module):
        """Pre-norm transformer block: x + MHSA(norm(x)); x + MLP(norm(x))."""

        def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
            self.norm2 = nn.LayerNorm(dim)
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class VisionTransformer(nn.Module):
        """Transformer encoder that classifies a CNN feature map."""

        def __init__(
            self,
            in_channels: int = 1024,
            num_classes: int = 2,
            vit_cfg: config.ViTConfig = config.VIT,
        ) -> None:
            super().__init__()
            dim = vit_cfg.embed_dim
            self.patch = PatchEmbed(in_channels, dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.randn(1, vit_cfg.max_tokens, dim) * 0.02)
            self.blocks = nn.ModuleList(
                TransformerEncoderBlock(
                    dim, vit_cfg.num_heads, vit_cfg.mlp_ratio, vit_cfg.dropout
                )
                for _ in range(vit_cfg.depth)
            )
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, num_classes)

        def forward(self, feature_map: "torch.Tensor") -> "torch.Tensor":
            x = self.patch(feature_map)                      # (B, L, D)
            b, seq_len, _ = x.shape
            cls = self.cls_token.expand(b, -1, -1)           # (B, 1, D)
            x = torch.cat((cls, x), dim=1)                   # (B, L+1, D)
            if seq_len + 1 > self.pos_embed.shape[1]:
                raise ValueError(
                    f"sequence length {seq_len + 1} exceeds max_tokens "
                    f"{self.pos_embed.shape[1]}; raise ViTConfig.max_tokens"
                )
            x = x + self.pos_embed[:, : seq_len + 1]
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x)[:, 0])             # classify CLS token

    class CNNViTHybrid(nn.Module):
        """Frozen (by default) CNN backbone feeding a Vision Transformer."""

        def __init__(
            self,
            num_classes: int = 2,
            vit_cfg: config.ViTConfig = config.VIT,
            *,
            freeze_backbone: bool = True,
        ) -> None:
            super().__init__()
            self.backbone = build_pytorch_cnn(num_classes=num_classes)
            self.vit = VisionTransformer(
                in_channels=config.CNN.conv_channels[-1],
                num_classes=num_classes,
                vit_cfg=vit_cfg,
            )
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad_(False)

        def load_backbone_state_dict(self, state_dict, strict: bool = False):
            """Load pretrained CNN weights (ignores classifier-head mismatches)."""
            return self.backbone.load_state_dict(state_dict, strict=strict)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.vit(self.backbone.forward_features(x))

    return {
        "PatchEmbed": PatchEmbed,
        "MultiHeadSelfAttention": MultiHeadSelfAttention,
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "VisionTransformer": VisionTransformer,
        "CNNViTHybrid": CNNViTHybrid,
    }


def __getattr__(name: str):  # PEP 562
    if not _CLASSES:
        _CLASSES.update(_build_classes())
    if name in _CLASSES:
        return _CLASSES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_pytorch_cnn_vit_hybrid(
    num_classes: int = 2,
    vit_cfg: config.ViTConfig = config.VIT,
    *,
    freeze_backbone: bool = True,
):
    """Convenience builder for :class:`CNNViTHybrid`."""
    return __getattr__("CNNViTHybrid")(
        num_classes=num_classes, vit_cfg=vit_cfg, freeze_backbone=freeze_backbone
    )
