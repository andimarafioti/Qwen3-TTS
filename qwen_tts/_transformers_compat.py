# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for Qwen3-TTS on Transformers 5.x."""

from __future__ import annotations

import torch


def _default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    """Initialize the unscaled RoPE variant removed from the 5.x registry."""
    del seq_len, layer_type
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


def rope_init_function(rope_type: str):
    """Return the RoPE initializer for a Qwen3-TTS module.

    Transformers 5 removed "default" from ``ROPE_INIT_FUNCTIONS``. Resolve it
    locally instead of registering it globally: since Transformers 5.17 a
    registry entry overrides each model's own ``compute_default_rope_parameters``
    (for example Mimi's, which reads ``config.rope_parameters``).
    """
    if rope_type == "default":
        return _default_rope_parameters

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    return ROPE_INIT_FUNCTIONS[rope_type]


@torch.no_grad()
def restore_rope_buffers(model) -> None:
    """Rebuild non-persistent RoPE buffers after low-memory model loading."""
    for module in model.modules():
        if not all(hasattr(module, name) for name in ("rope_init_fn", "config", "inv_freq")):
            continue
        inv_freq, attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = module.inv_freq
        module.attention_scaling = attention_scaling


def restore_mimi_full_attention(model) -> None:
    """Preserve the full causal Mimi attention used by Transformers 4.57.3."""
    for module in model.modules():
        if not module.__class__.__module__.startswith("transformers.models.mimi"):
            continue
        config = getattr(module, "config", None)
        if config is None or not hasattr(config, "sliding_window"):
            continue
        full_window = config.max_position_embeddings
        config.sliding_window = full_window
        if hasattr(module, "sliding_window"):
            module.sliding_window = full_window
