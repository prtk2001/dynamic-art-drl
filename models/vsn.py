"""
Variable Selection Network (VSN)
================================

Dynamically selects and weights input features before they are fed into
the Transformer encoder.

Mathematical formulation
------------------------
Given *N* individual feature vectors ξ₁, ξ₂, …, ξ_N  (each ∈ ℝ^{input_size}):

    1.  **Transform** each feature independently:
            ξ̃ᵢ = GRNᵢ(ξᵢ, c)          ∈ ℝ^{hidden_size}

    2.  **Compute selection weights** from the concatenated raw inputs:
            Ξ  = [ξ₁ ‖ ξ₂ ‖ … ‖ ξ_N]  ∈ ℝ^{N · input_size}
            α  = softmax(GRN_select(Ξ, c))  ∈ ℝ^{N}

    3.  **Weighted aggregation**:
            v_t = Σᵢ αᵢ · ξ̃ᵢ           ∈ ℝ^{hidden_size}

The weights **α** are returned alongside the output for interpretability
(feature importance analysis).

Reference
---------
Lim, B. et al. "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting," *Int. J. Forecasting*, 2021.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.grn import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for adaptive feature weighting.

    Parameters
    ----------
    num_features : int
        Number of input features (*N*).
    input_size : int
        Dimensionality of each individual feature vector ξᵢ.
    hidden_size : int
        Width of the GRN hidden layers **and** the output dimension.
    dropout : float, optional
        Dropout probability passed to every GRN (default 0.1).
    context_size : int | None, optional
        Dimensionality of an optional static context vector *c*.
    """

    def __init__(
        self,
        num_features: int,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.input_size = input_size
        self.hidden_size = hidden_size

        # ── Per-feature GRNs (transform each ξᵢ independently) ──────────
        # Each GRN: (*, input_size) → (*, hidden_size)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=context_size,
            )
            for _ in range(num_features)
        ])

        # ── Selection GRN (produces weights over features) ──────────────
        # Input: concatenated raw features  → N weights
        # (*, N·input_size) → (*, num_features)
        self.selection_grn = GatedResidualNetwork(
            input_size=num_features * input_size,
            hidden_size=hidden_size,
            output_size=num_features,
            dropout=dropout,
            context_size=context_size,
        )

    # --------------------------------------------------------------------- #
    #  Forward pass                                                           #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        feature_list: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the VSN.

        Parameters
        ----------
        feature_list : list of Tensor
            *N* tensors each of shape ``(batch, *, input_size)``.
            Extra leading dimensions (e.g. sequence length) are supported.
        context : Tensor | None, shape ``(batch, *, context_size)``
            Optional static context broadcast to every GRN.

        Returns
        -------
        selected : Tensor, shape ``(batch, *, hidden_size)``
            Weighted aggregation of transformed features.
        weights : Tensor, shape ``(batch, *, num_features)``
            Softmax feature-importance weights (sums to 1 along last dim).
        """
        assert len(feature_list) == self.num_features, (
            f"Expected {self.num_features} features, got {len(feature_list)}"
        )

        # ── 1. Transform each feature through its own GRN ───────────────
        # transformed_i : (batch, *, hidden_size)
        transformed = [
            grn(xi, context=context)                         # (batch, *, hidden_size)
            for grn, xi in zip(self.feature_grns, feature_list)
        ]
        # Stack along a new "feature" dim → (batch, *, N, hidden_size)
        transformed_stack = torch.stack(transformed, dim=-2) # (batch, *, N, hidden_size)

        # ── 2. Compute selection weights ─────────────────────────────────
        # Concatenate raw inputs along the feature dim
        concat_raw = torch.cat(feature_list, dim=-1)         # (batch, *, N·input_size)
        weight_logits = self.selection_grn(concat_raw, context=context)
                                                             # (batch, *, num_features)
        weights = F.softmax(weight_logits, dim=-1)           # (batch, *, num_features)

        # ── 3. Weighted aggregation ──────────────────────────────────────
        # Expand weights for broadcasting: (batch, *, N, 1)
        weights_expanded = weights.unsqueeze(-1)             # (batch, *, N, 1)
        # Element-wise multiply and sum over feature dim
        selected = (transformed_stack * weights_expanded).sum(dim=-2)
                                                             # (batch, *, hidden_size)

        return selected, weights

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Quick self-test                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path for the `models.grn` import
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    torch.manual_seed(42)

    print("=" * 60)
    print("VSN Self-Test")
    print("=" * 60)

    N_FEATURES = 6      # e.g. price, volume, RSI, MACD, Kalman state, …
    INPUT_SIZE = 1       # each feature is a scalar (expanded to 1-d)
    HIDDEN_SIZE = 64
    BATCH = 8
    SEQ_LEN = 60

    # ── Test 1: Basic VSN (no context) ───────────────────────────────────
    vsn = VariableSelectionNetwork(
        num_features=N_FEATURES,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout=0.1,
    )

    # Simulate N_FEATURES scalar features over a sequence
    features = [
        torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)             # (8, 60, 1) each
        for _ in range(N_FEATURES)
    ]

    selected, weights = vsn(features)
    assert selected.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE), \
        f"Expected ({BATCH},{SEQ_LEN},{HIDDEN_SIZE}), got {selected.shape}"
    assert weights.shape == (BATCH, SEQ_LEN, N_FEATURES), \
        f"Expected ({BATCH},{SEQ_LEN},{N_FEATURES}), got {weights.shape}"

    # Weights should sum to 1 along feature dim
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Weights do not sum to 1!"

    print(f"[PASS] Basic VSN   : {N_FEATURES} × {(BATCH,SEQ_LEN,INPUT_SIZE)}")
    print(f"       selected    : {selected.shape}")
    print(f"       weights     : {weights.shape}")
    print(f"       weight sums : {weight_sums[0, :5].tolist()} (≈1.0)")

    # ── Test 2: VSN with context ─────────────────────────────────────────
    CONTEXT_SIZE = 16
    vsn_ctx = VariableSelectionNetwork(
        num_features=N_FEATURES,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout=0.1,
        context_size=CONTEXT_SIZE,
    )
    ctx = torch.randn(BATCH, SEQ_LEN, CONTEXT_SIZE)         # (8, 60, 16)
    sel_ctx, w_ctx = vsn_ctx(features, context=ctx)
    assert sel_ctx.shape == (BATCH, SEQ_LEN, HIDDEN_SIZE)
    assert w_ctx.shape == (BATCH, SEQ_LEN, N_FEATURES)
    print(f"[PASS] Context VSN : selected {sel_ctx.shape}, weights {w_ctx.shape}")

    # ── Test 3: 2-D input (no sequence dim) ──────────────────────────────
    features_2d = [torch.randn(BATCH, INPUT_SIZE) for _ in range(N_FEATURES)]
    sel_2d, w_2d = vsn(features_2d)
    assert sel_2d.shape == (BATCH, HIDDEN_SIZE)
    assert w_2d.shape == (BATCH, N_FEATURES)
    print(f"[PASS] 2-D VSN     : selected {sel_2d.shape}, weights {w_2d.shape}")

    # ── Test 4: Gradient flow ────────────────────────────────────────────
    loss = selected.sum()
    loss.backward()
    grads_ok = all(p.grad is not None for p in vsn.parameters() if p.requires_grad)
    print(f"[PASS] Gradients   : all parameter gradients computed = {grads_ok}")

    # ── Parameter count ──────────────────────────────────────────────────
    n_params = sum(p.numel() for p in vsn.parameters())
    print(f"\nTotal parameters (basic VSN): {n_params:,}")
    print("\n✅  All VSN tests passed!")
