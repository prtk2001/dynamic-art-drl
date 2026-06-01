"""
Gated Residual Network (GRN)
============================

Building block of the Temporal Fusion Transformer (TFT).

Mathematical formulation
------------------------
Given primary input **x** ∈ ℝ^{input_size} and an optional context
vector **c** ∈ ℝ^{context_size}:

    η₁  = W₁ · x + b₁                          (if c is None)
    η₁  = W₁ · x + W_c · c + b₁                (if c is provided)
    η₂  = W₂ · ELU(η₁) + b₂                    ∈ ℝ^{2 · output_size}
    GLU(η₂) = σ(η₂[:, :d]) ⊙ η₂[:, d:]         ∈ ℝ^{output_size}
    GRN(x, c) = LayerNorm(skip(x) + GLU(η₂))

where *skip(x)* is a linear projection when input_size ≠ output_size,
otherwise the identity.

Reference
---------
Lim, B. et al. "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting," *Int. J. Forecasting*, 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with optional context input.

    Parameters
    ----------
    input_size : int
        Dimensionality of the primary input *x*.
    hidden_size : int
        Width of the intermediate ELU layer.
    output_size : int
        Dimensionality of the output (and the skip-connection target).
    dropout : float, optional
        Dropout probability applied after the ELU activation (default 0.1).
    context_size : int | None, optional
        Dimensionality of the optional static context vector *c*.
        When ``None``, no context pathway is created.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # ── Primary linear: x → hidden ──────────────────────────────────
        self.fc1 = nn.Linear(input_size, hidden_size)       # (*, input) → (*, hidden)

        # ── Optional context linear: c → hidden ─────────────────────────
        self.fc_context: nn.Linear | None = None
        if context_size is not None:
            self.fc_context = nn.Linear(context_size, hidden_size, bias=False)
            # bias=False because fc1 already supplies a bias term

        # ── Second linear: hidden → 2*output (pre-GLU) ──────────────────
        # GLU halves the last dimension, so we project to 2·output_size
        self.fc2 = nn.Linear(hidden_size, output_size * 2)  # (*, hidden) → (*, 2·output)

        # ── Dropout applied after ELU activation ─────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── GLU gate (splits along last dim and applies sigmoid gate) ────
        self.glu = nn.GLU(dim=-1)                           # (*, 2·d) → (*, d)

        # ── Skip-connection projection (when dimensions differ) ──────────
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size)  # (*, input) → (*, output)
        else:
            self.skip_proj = None

        # ── Layer normalisation ──────────────────────────────────────────
        self.layer_norm = nn.LayerNorm(output_size)

    # --------------------------------------------------------------------- #
    #  Forward pass                                                           #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the GRN.

        Parameters
        ----------
        x : Tensor, shape ``(*, input_size)``
            Primary input.
        context : Tensor | None, shape ``(*, context_size)``
            Optional static context vector.

        Returns
        -------
        Tensor, shape ``(*, output_size)``
        """
        # ── Skip connection ──────────────────────────────────────────────
        if self.skip_proj is not None:
            skip = self.skip_proj(x)                         # (*, output_size)
        else:
            skip = x                                         # (*, output_size)

        # ── Hidden layer ─────────────────────────────────────────────────
        hidden = self.fc1(x)                                 # (*, hidden_size)

        if context is not None and self.fc_context is not None:
            hidden = hidden + self.fc_context(context)       # (*, hidden_size)

        hidden = F.elu(hidden)                               # (*, hidden_size)
        hidden = self.dropout(hidden)                        # (*, hidden_size)

        # ── Gated output ─────────────────────────────────────────────────
        gate_input = self.fc2(hidden)                        # (*, 2·output_size)
        gated = self.glu(gate_input)                         # (*, output_size)

        # ── Residual + LayerNorm ─────────────────────────────────────────
        out = self.layer_norm(skip + gated)                  # (*, output_size)
        return out

    def extra_repr(self) -> str:
        ctx = f", context_size={self.fc_context.in_features}" if self.fc_context else ""
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}{ctx}"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Quick self-test                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("GRN Self-Test")
    print("=" * 60)

    # ── Test 1: Basic GRN (no context, same dims) ────────────────────────
    grn_basic = GatedResidualNetwork(
        input_size=64, hidden_size=64, output_size=64, dropout=0.1
    )
    x = torch.randn(8, 60, 64)                              # (batch, seq, features)
    y = grn_basic(x)                                         # (batch, seq, 64)
    assert y.shape == (8, 60, 64), f"Expected (8,60,64), got {y.shape}"
    print(f"[PASS] Basic GRN  : input {x.shape} → output {y.shape}")

    # ── Test 2: GRN with dimension change ────────────────────────────────
    grn_proj = GatedResidualNetwork(
        input_size=32, hidden_size=64, output_size=64, dropout=0.1
    )
    x2 = torch.randn(4, 32)                                 # (batch, input)
    y2 = grn_proj(x2)                                        # (batch, 64)
    assert y2.shape == (4, 64), f"Expected (4,64), got {y2.shape}"
    print(f"[PASS] Projected   : input {x2.shape} → output {y2.shape}")

    # ── Test 3: GRN with context ─────────────────────────────────────────
    grn_ctx = GatedResidualNetwork(
        input_size=64, hidden_size=64, output_size=64,
        dropout=0.1, context_size=16,
    )
    x3 = torch.randn(4, 60, 64)                             # (batch, seq, input)
    c3 = torch.randn(4, 60, 16)                              # (batch, seq, context)
    y3 = grn_ctx(x3, context=c3)                             # (batch, seq, 64)
    assert y3.shape == (4, 60, 64), f"Expected (4,60,64), got {y3.shape}"
    print(f"[PASS] With context: input {x3.shape} + ctx {c3.shape} → {y3.shape}")

    # ── Test 4: Gradient flow ────────────────────────────────────────────
    y3_sum = y3.sum()
    y3_sum.backward()
    grads_ok = all(p.grad is not None for p in grn_ctx.parameters() if p.requires_grad)
    print(f"[PASS] Gradients   : all parameter gradients computed = {grads_ok}")

    # ── Parameter count ──────────────────────────────────────────────────
    n_params = sum(p.numel() for p in grn_ctx.parameters())
    print(f"\nTotal parameters (context GRN): {n_params:,}")
    print("\n✅  All GRN tests passed!")
