"""
src/model.py
PRISM full network - ViT backbone + noise encoder + cross-modal fusion + decoder.

VRAM budget (RTX 3050, 6 GB):
  ViT-B/16 weights   ~330 MB
  Noise encoder      ~50 MB
  Activations (B=2)  ~300 MB
  Total forward      < 2.5 GB with AMP / no amp on small batch
"""

import torch
import torch.nn as nn
import timm

try:
    from dora_layer import inject_dora
except ImportError:
    from src.dora_layer import inject_dora

_EMBED   = 768   # ViT-B/16 token dim
_PATCHES = 196   # 14x14 spatial patches


# ---------------------------------------------------------------------------
# 1. Sensor Noise Encoder
# ---------------------------------------------------------------------------

class SensorNoiseEncoder(nn.Module):
    """
    4-stage conv network: [B,1,224,224] -> [B,196,768]

    Stride-2 convs halve spatial dims each stage:
      224 -> 112 -> 56 -> 28 -> 14   (4x stride-2)
    Final: [B, 512, 14, 14] -> flatten spatial -> project to embed dim.
    """

    def __init__(self) -> None:
        super().__init__()
        # ponytail: BatchNorm after each conv keeps training stable; no dropout needed here
        self.encoder = nn.Sequential(
            # stage 1: 224->112
            nn.Conv2d(1,   64,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # stage 2: 112->56
            nn.Conv2d(64,  128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # stage 3: 56->28
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # stage 4: 28->14
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        # Project from 512 channels to ViT embed dim
        self.proj = nn.Linear(512, _EMBED)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        # noise: [B, 1, 224, 224]
        x = self.encoder(noise)          # [B, 512, 14, 14]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()       # [B, 14, 14, 512]
        x = x.reshape(B, H * W, C)      # [B, 196, 512]  - no clone, reshape is a view
        return self.proj(x)              # [B, 196, 768]


# ---------------------------------------------------------------------------
# 2. Cross-Modal Attention
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Visual tokens query the noise tokens.
    Residual + LayerNorm around the attention block.

    Input / Output: [B, 196, 768]
    """

    def __init__(self, embed_dim: int = _EMBED, num_heads: int = 8) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, vis: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # vis, noise: [B, 196, 768]
        q  = self.norm_q(vis)
        kv = self.norm_kv(noise)
        attended, _ = self.attn(q, kv, kv, need_weights=False)  # no attn map saved -> saves VRAM
        return self.norm_out(vis + attended)  # residual  [B, 196, 768]


# ---------------------------------------------------------------------------
# 3. Spatial Mask Decoder
# ---------------------------------------------------------------------------

class SpatialMaskDecoder(nn.Module):
    """
    5-stage transposed-conv upsampling: [B, 768, 14, 14] -> [B, 1, 224, 224]

    14 -> 28 -> 56 -> 112 -> 224 (4 stride-2 stages).
    Final conv collapses channels to 1. Sigmoid for mask in [0,1].
    """

    def __init__(self) -> None:
        super().__init__()
        # Input projection: 768 -> 256 to cut memory before upsampling
        self.in_proj = nn.Conv2d(768, 256, 1)

        self.decoder = nn.Sequential(
            # 14 -> 28
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 28 -> 56
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 56 -> 112
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 112 -> 224
            nn.ConvTranspose2d(32,  16,  4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            # channel collapse + sigmoid
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, 196, 768]  - 196 = 14*14
        B, N, C = tokens.shape
        x = tokens.permute(0, 2, 1).contiguous().reshape(B, C, 14, 14)  # [B, 768, 14, 14]
        x = self.in_proj(x)       # [B, 256, 14, 14]  - reduce channels early
        return self.decoder(x)    # [B, 1, 224, 224]


# ---------------------------------------------------------------------------
# 4. PRISMNetwork - full assembly
# ---------------------------------------------------------------------------

class PRISMNetwork(nn.Module):
    """
    Full PRISM model.

    Forward inputs:
        rgb:   [B, 3, 224, 224]  - ImageNet-normalised RGB
        noise: [B, 1, 224, 224]  - PRNU Laplacian residual

    Forward outputs (dict):
        logit: [B, 1]            - binary deepfake logit (real=0 / fake=1)
        mask:  [B, 1, 224, 224]  - pixel-level forgery probability map
    """

    def __init__(self, dora_rank: int = 8, lora_alpha: float = 16.0, pretrained: bool = True) -> None:
        super().__init__()

        # --- ViT backbone: num_classes=0 returns patch tokens ---
        vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,  # False for shape tests, True for production
            num_classes=0,           # returns [B, 197, 768] (CLS + 196 patches)
        )

        # Freeze all ViT params before DoRA injection
        for p in vit.parameters():
            p.requires_grad_(False)

        # Replace every Linear inside ViT with DoRALinear (only adapters are trainable)
        self.vit = inject_dora(vit, rank=dora_rank, lora_alpha=lora_alpha)

        # Remaining modules - all trainable
        self.noise_enc  = SensorNoiseEncoder()
        self.fusion     = CrossModalAttention()
        self.cls_head   = nn.Linear(_EMBED, 1)       # CLS token -> binary logit
        self.mask_dec   = SpatialMaskDecoder()

    def forward(self, rgb: torch.Tensor, noise: torch.Tensor) -> dict:
        # --- ViT: returns [B, 197, 768] (CLS at index 0) ---
        vit_out   = self.vit.forward_features(rgb)   # [B, 197, 768]
        cls_token = vit_out[:, :1, :]                # [B, 1, 768]
        vis_tokens = vit_out[:, 1:, :]               # [B, 196, 768]

        # --- Noise encoder ---
        noise_tokens = self.noise_enc(noise)         # [B, 196, 768]

        # --- Cross-modal fusion: visual queries noise ---
        fused = self.fusion(vis_tokens, noise_tokens)  # [B, 196, 768]

        # --- Classification: pool fused tokens + CLS ---
        pooled = fused.mean(dim=1) + cls_token.squeeze(1)  # [B, 768]
        logit  = self.cls_head(pooled)                      # [B, 1]

        # --- Spatial mask ---
        mask = self.mask_dec(fused)                  # [B, 1, 224, 224]

        return {"logit": logit, "mask": mask}


# ---------------------------------------------------------------------------
# Smoke test - shape + VRAM check
# NOTE: uses pretrained=False to avoid downloading 330MB during CI/shape checks.
#       Production training always instantiates PRISMNetwork(pretrained=True).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ponytail: pretrained=False - shape/VRAM check only, no 330MB download
    model = PRISMNetwork(dora_rank=8, lora_alpha=16.0, pretrained=False).to(device)

    # Count trainable vs frozen params
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # batch=2 - matches VRAM budget spec
    rgb   = torch.randn(2, 3, 224, 224, device=device)
    noise = torch.randn(2, 1, 224, 224, device=device)

    with torch.no_grad():
        out = model(rgb, noise)

    logit = out["logit"]
    mask  = out["mask"]

    print(f"logit shape: {tuple(logit.shape)}")
    print(f"mask shape:  {tuple(mask.shape)}")

    assert logit.shape == (2, 1),           f"Bad logit shape: {logit.shape}"
    assert mask.shape  == (2, 1, 224, 224), f"Bad mask shape: {mask.shape}"
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0, "Mask out of [0,1]"

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"Peak VRAM: {peak_gb:.3f} GB")
        assert peak_gb < 2.5, f"VRAM budget exceeded: {peak_gb:.3f} GB"
        print("VRAM budget OK (< 2.5 GB).")
    else:
        print("Running on CPU - skipping VRAM check.")

    print("All model shape assertions passed.")
