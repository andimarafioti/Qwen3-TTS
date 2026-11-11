import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from qwen_tts._transformers_compat import (
    _default_rope_parameters,
    restore_mimi_full_attention,
    restore_rope_buffers,
    rope_init_function,
)


class _RopeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = object()
        self.register_buffer("inv_freq", torch.zeros(2), persistent=False)
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 0.0

    @staticmethod
    def rope_init_fn(config, device):
        del config
        return torch.tensor([1.0, 0.5], device=device), 1.0


class _MimiModule(torch.nn.Module):
    __module__ = "transformers.models.mimi.modeling_mimi"

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"max_position_embeddings": 8000, "sliding_window": 250})()
        self.sliding_window = 250


def test_restore_rope_buffers():
    model = torch.nn.Sequential(_RopeModule())

    restore_rope_buffers(model)

    torch.testing.assert_close(model[0].inv_freq, torch.tensor([1.0, 0.5]))
    assert model[0].original_inv_freq is model[0].inv_freq
    assert model[0].attention_scaling == 1.0


def test_default_rope_initializer_is_local():
    import qwen_tts  # noqa: F401

    assert "default" not in ROPE_INIT_FUNCTIONS
    assert rope_init_function("default") is _default_rope_parameters
    assert rope_init_function("linear") is ROPE_INIT_FUNCTIONS["linear"]


def test_mimi_rope_initializes_after_import():
    import qwen_tts  # noqa: F401
    from transformers import MimiConfig
    from transformers.models.mimi.modeling_mimi import MimiPreTrainedModel, MimiRotaryEmbedding

    config = MimiConfig()
    rope = MimiRotaryEmbedding(config)
    expected = rope.inv_freq.clone()
    rope.inv_freq.zero_()

    MimiPreTrainedModel(config)._init_weights(rope)

    torch.testing.assert_close(rope.inv_freq, expected)


def test_restore_mimi_full_attention():
    model = torch.nn.Sequential(_MimiModule())

    restore_mimi_full_attention(model)

    assert model[0].config.sliding_window == 8000
    assert model[0].sliding_window == 8000
