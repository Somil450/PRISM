"""
src/dora_layer.py
DoRA (Directional Low-Rank Adaptation) linear layer.

Formula:
    V'  = W0 + lora_B @ lora_A              # adapted weight matrix
    W_dora = magnitude * (V' / ||V'||_col)  # rescale columns to learned magnitude

W0 is frozen throughout training. Only lora_A, lora_B, magnitude are learned.
No bias handled here - callers pass bias separately (standard timm pattern).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with DoRA adaptation.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        rank:         LoRA rank. Default 8.
        lora_alpha:   LoRA scaling factor. Default 16.0.
        bias:         Pass the original layer's bias tensor (or None).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.scaling      = lora_alpha / rank  # ponytail: fold scale into forward, avoids extra multiply

        # Frozen base weight - registered as buffer so it moves with .to(device) but never trained
        self.register_buffer("W0", torch.empty(out_features, in_features))

        # Low-rank matrices  (kaiming for A, zeros for B → zero init delta)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Per-column magnitude: shape [1, out_features] for broadcast over columns
        # Initialized after W0 is populated via init_from_linear()
        self.magnitude = nn.Parameter(torch.empty(1, out_features))

        # Carry bias as buffer; None-safe
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def init_from_linear(self, linear: nn.Linear) -> "DoRALinear":
        """Populate W0 and magnitude from an existing nn.Linear weight."""
        with torch.no_grad():
            self.W0.copy_(linear.weight)
            # magnitude = column-wise L2 norm of W0; shape [out_features]
            col_norms = linear.weight.norm(p=2, dim=1)  # [out_features]
            self.magnitude.copy_(col_norms.unsqueeze(0))  # [1, out_features]
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapted weight: base + scaled low-rank delta
        V_prime = self.W0 + (self.lora_B @ self.lora_A) * self.scaling  # [out, in]

        # Column-wise unit direction of V'  (dim=1 = input axis = "columns")
        V_norm = V_prime.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)  # [out, 1]
        W_dora = self.magnitude.T * (V_prime / V_norm)  # [out, in]  - no extra clone needed

        return F.linear(x, W_dora, self.bias)


def inject_dora(module: nn.Module, rank: int = 8, lora_alpha: float = 16.0) -> nn.Module:
    """
    Walk `module`, replace every nn.Linear with DoRALinear in-place.
    Returns the same module (mutated).

    Use after loading pretrained weights. Freezes the original W0 automatically.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            dora = DoRALinear(
                child.in_features,
                child.out_features,
                rank=rank,
                lora_alpha=lora_alpha,
                bias=child.bias,
            ).init_from_linear(child)
            setattr(module, name, dora)
        else:
            inject_dora(child, rank=rank, lora_alpha=lora_alpha)  # recurse
    return module


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    lin = nn.Linear(768, 768)
    dora = DoRALinear(768, 768, rank=8, lora_alpha=16.0, bias=lin.bias).init_from_linear(lin)

    x = torch.randn(2, 196, 768)
    out = dora(x)
    assert out.shape == (2, 196, 768), f"Bad shape: {out.shape}"

    # Only lora_A, lora_B, magnitude should be trainable
    trainable = [n for n, p in dora.named_parameters() if p.requires_grad]
    assert set(trainable) == {"lora_A", "lora_B", "magnitude"}, f"Wrong trainables: {trainable}"
    print(f"DoRALinear output: {out.shape}  trainable params: {trainable}")
    print("DoRALinear assertions passed.")
