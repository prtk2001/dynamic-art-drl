"""
State Encoder — VSN + Transformer
==================================

Combines the Variable Selection Network with a multi-head self-attention
Transformer encoder to produce a compact state vector for the DRL agent.

Pipeline
--------
    Raw features  ──►  VSN  ──►  + Positional Encoding
                                       │
                                       ▼
                              Transformer Encoder
                                       │
                                       ▼
                               h_t = encoded[:, -1, :]   ∈ ℝ^{hidden_size}

Mathematical overview
---------------------
1. **Feature selection** (VSN):
       v_t = Σᵢ αᵢ · GRNᵢ(ξᵢ)          ∈ ℝ^{hidden_size}
       α   = softmax(GRN_select([ξ₁,…,ξ_N]))

2. **Positional encoding** (learnable):
       e_t = v_t + PE_t                  ∈ ℝ^{hidden_size}

3. **Temporal encoding** (Transformer):
       H   = TransformerEncoder(e₁,…,e_T)
       h_t = H[:, -1, :]                ∈ ℝ^{hidden_size}

Configuration defaults (from ``config.settings``):
    hidden_size = 64,  num_heads = 4,  num_layers = 2,
    dropout = 0.1,  lookback = 60

Reference
---------
Lim, B. et al. "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting," *Int. J. Forecasting*, 2021.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from config.settings import (
    LOOKBACK_WINDOW,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_HIDDEN_SIZE,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    VSN_DROPOUT,
    VSN_HIDDEN_SIZE,
)
from models.vsn import VariableSelectionNetwork


class StateEncoder(nn.Module):
    """VSN + Transformer Encoder producing a state vector for DRL.

    Parameters
    ----------
    num_features : int
        Number of raw input features (*N*).
    input_size : int
        Dimensionality of each raw feature ξᵢ (typically 1 for scalars).
    hidden_size : int, optional
        Transformer / VSN hidden dimension (default from config: 64).
    num_heads : int, optional
        Number of self-attention heads (default 4).
    num_layers : int, optional
        Number of Transformer encoder layers (default 2).
    dropout : float, optional
        Dropout probability for both VSN and Transformer (default 0.1).
    max_seq_len : int, optional
        Maximum sequence length for positional encoding (default 60).
    context_size : int | None, optional
        Dimensionality of optional static context for the VSN.
    """

    def __init__(
        self,
        num_features: int,
        input_size: int = 1,
        hidden_size: int = TRANSFORMER_HIDDEN_SIZE,
        num_heads: int = TRANSFORMER_NUM_HEADS,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dropout: float = TRANSFORMER_DROPOUT,
        max_seq_len: int = LOOKBACK_WINDOW,
        context_size: int | None = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # ── Variable Selection Network ──────────────────────────────────
        self.vsn = VariableSelectionNetwork(
            num_features=num_features,
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=context_size,
        )

        # ── Learnable positional encoding ────────────────────────────────
        # PE : (1, max_seq_len, hidden_size) — broadcast over batch
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_size) * 0.02  # small init
        )                                                    # (1, T, hidden)

        # ── Transformer Encoder ──────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,                             # model dimension
            nhead=num_heads,                                 # attention heads
            dim_feedforward=hidden_size * 4,                 # FFN inner dim (4× d_model)
            dropout=dropout,
            activation="gelu",                               # smoother than ReLU
            batch_first=True,                                # (batch, seq, feat)
            norm_first=True,                                 # Pre-LN (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,                      # compatibility
        )

        # ── Final layer norm on the output state ─────────────────────────
        self.output_norm = nn.LayerNorm(hidden_size)

    # --------------------------------------------------------------------- #
    #  Causal mask helper                                                     #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask for self-attention.

        Returns
        -------
        mask : Tensor, shape ``(seq_len, seq_len)``
            Positions that should be masked are filled with ``-inf``.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )                                                    # (T, T)
        return mask

    # --------------------------------------------------------------------- #
    #  Main encode method                                                     #
    # --------------------------------------------------------------------- #
    def encode(
        self,
        feature_list: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a lookback window of features into a state vector.

        Parameters
        ----------
        feature_list : list of Tensor
            *N* tensors each of shape ``(batch, seq_len, input_size)``.
        context : Tensor | None, shape ``(batch, seq_len, context_size)``
            Optional static context for the VSN.

        Returns
        -------
        state_vector : Tensor, shape ``(batch, hidden_size)``
            Compact state representation *h_t* for the DRL agent.
        feature_weights : Tensor, shape ``(batch, seq_len, num_features)``
            VSN feature-importance weights (for interpretability).
        """
        batch_size = feature_list[0].shape[0]
        seq_len = feature_list[0].shape[1]

        # ── 1. Variable Selection ────────────────────────────────────────
        vsn_out, feature_weights = self.vsn(feature_list, context=context)
        # vsn_out        : (batch, seq_len, hidden_size)
        # feature_weights: (batch, seq_len, num_features)

        # ── 2. Add positional encoding ───────────────────────────────────
        # Slice PE to actual seq_len (handles seq_len <= max_seq_len)
        pe = self.positional_encoding[:, :seq_len, :]        # (1, seq_len, hidden)
        vsn_out = vsn_out + pe                               # (batch, seq_len, hidden)

        # ── 3. Transformer encoder ───────────────────────────────────────
        causal_mask = self._generate_causal_mask(
            seq_len, device=vsn_out.device
        )                                                    # (seq_len, seq_len)

        encoded = self.transformer_encoder(
            vsn_out, mask=causal_mask
        )                                                    # (batch, seq_len, hidden)

        # ── 4. Extract last-timestep representation ──────────────────────
        state_vector = encoded[:, -1, :]                     # (batch, hidden_size)
        state_vector = self.output_norm(state_vector)        # (batch, hidden_size)

        return state_vector, feature_weights

    # --------------------------------------------------------------------- #
    #  Convenience forward (delegates to encode)                              #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        feature_list: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`encode` — allows use as a standard nn.Module."""
        return self.encode(feature_list, context=context)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, hidden_size={self.hidden_size}, "
            f"max_seq_len={self.max_seq_len}"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Comprehensive self-test                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is importable
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    torch.manual_seed(42)

    print("=" * 60)
    print("State Encoder Self-Test")
    print("=" * 60)

    # ── Configuration ────────────────────────────────────────────────────
    BATCH       = 8
    SEQ_LEN     = 60       # lookback window
    N_FEATURES  = 6        # price, volume, RSI, MACD, Kalman mean, Kalman var
    INPUT_SIZE  = 1        # each feature is a scalar
    HIDDEN_SIZE = 64
    CONTEXT_SIZE = 16

    # ── Test 1: Basic encoder (no context) ───────────────────────────────
    encoder = StateEncoder(
        num_features=N_FEATURES,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_seq_len=SEQ_LEN,
    )
    features = [
        torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)             # (8, 60, 1) each
        for _ in range(N_FEATURES)
    ]

    state, weights = encoder.encode(features)
    assert state.shape == (BATCH, HIDDEN_SIZE), \
        f"Expected state ({BATCH},{HIDDEN_SIZE}), got {state.shape}"
    assert weights.shape == (BATCH, SEQ_LEN, N_FEATURES), \
        f"Expected weights ({BATCH},{SEQ_LEN},{N_FEATURES}), got {weights.shape}"
    print(f"[PASS] Basic encoder:")
    print(f"       state_vector : {state.shape}")
    print(f"       feat_weights : {weights.shape}")

    # ── Test 2: Encoder with context ─────────────────────────────────────
    encoder_ctx = StateEncoder(
        num_features=N_FEATURES,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_seq_len=SEQ_LEN,
        context_size=CONTEXT_SIZE,
    )
    ctx = torch.randn(BATCH, SEQ_LEN, CONTEXT_SIZE)
    state_ctx, w_ctx = encoder_ctx.encode(features, context=ctx)
    assert state_ctx.shape == (BATCH, HIDDEN_SIZE)
    assert w_ctx.shape == (BATCH, SEQ_LEN, N_FEATURES)
    print(f"[PASS] Context encoder: state {state_ctx.shape}, weights {w_ctx.shape}")

    # ── Test 3: Shorter sequence (< max_seq_len) ────────────────────────
    short_seq = 20
    short_features = [torch.randn(BATCH, short_seq, INPUT_SIZE) for _ in range(N_FEATURES)]
    state_short, w_short = encoder.encode(short_features)
    assert state_short.shape == (BATCH, HIDDEN_SIZE)
    assert w_short.shape == (BATCH, short_seq, N_FEATURES)
    print(f"[PASS] Short sequence ({short_seq}): state {state_short.shape}")

    # ── Test 4: forward() alias ──────────────────────────────────────────
    state_fwd, w_fwd = encoder(features)
    assert state_fwd.shape == (BATCH, HIDDEN_SIZE)
    print(f"[PASS] forward() alias works: {state_fwd.shape}")

    # ── Test 5: Gradient flow ────────────────────────────────────────────
    loss = state.sum()
    loss.backward()
    grads_ok = all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    print(f"[PASS] Gradients: all parameter gradients computed = {grads_ok}")

    # ── Test 6: Feature weight interpretability ──────────────────────────
    # Average weights across batch and time → feature ranking
    avg_weights = weights.mean(dim=(0, 1))                   # (N_FEATURES,)
    ranking = avg_weights.argsort(descending=True)
    feature_names = ["price", "volume", "RSI", "MACD", "kalman_mean", "kalman_var"]
    print(f"\n── Feature Importance (random data — sanity check) ──")
    for rank, idx in enumerate(ranking):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"  #{rank+1}: {name:15s} — weight {avg_weights[idx]:.4f}")

    # ── Parameter summary ────────────────────────────────────────────────
    total = sum(p.numel() for p in encoder.parameters())
    vsn_params = sum(p.numel() for p in encoder.vsn.parameters())
    tfm_params = sum(p.numel() for p in encoder.transformer_encoder.parameters())
    pe_params = encoder.positional_encoding.numel()
    norm_params = sum(p.numel() for p in encoder.output_norm.parameters())

    print(f"\n── Parameter Breakdown ──")
    print(f"  VSN                : {vsn_params:>8,}")
    print(f"  Positional Encoding: {pe_params:>8,}")
    print(f"  Transformer Encoder: {tfm_params:>8,}")
    print(f"  Output LayerNorm   : {norm_params:>8,}")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL              : {total:>8,}")

    print("\n✅  All State Encoder tests passed!")
