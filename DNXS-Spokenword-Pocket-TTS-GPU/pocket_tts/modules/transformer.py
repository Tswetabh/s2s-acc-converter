import torch
import torch.nn as nn
from torch.nn import functional as F

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


def complete_kv(
    cache: torch.Tensor, end_offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write new keys/values into a fixed-capacity cache at end_offset.

    Graph-safe: uses index_copy_ with tensor indices (no Python length from
    tensor.shape[0], no reallocation of end markers).

    Args:
        cache: KV cache shaped [2, B, S, H, D].
        end_offset: 0-dim long tensor — next write index (not advanced here).
        k: Keys [B, T, H, D].
        v: Values [B, T, H, D].

    Returns:
        Full-capacity key and value caches [B, S, H, D] each (invalid tail masked later).
    """
    _, t, _, _ = k.shape
    idx = torch.arange(t, device=k.device, dtype=torch.long) + end_offset
    cache[0].index_copy_(1, idx, k)
    cache[1].index_copy_(1, idx, v)
    return cache[0], cache[1]


def _build_streaming_attn_mask(
    query_len: int,
    cache_len: int,
    end_offset: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fixed-shape causal+padding mask for streaming decode (CUDA-graph friendly).

    Query positions are end_offset .. end_offset+query_len-1.
    Key positions 0 .. cache_len-1; only written keys < end_offset+query_len are valid.
    """
    q_pos = end_offset.to(dtype=torch.long) + torch.arange(
        query_len, device=device, dtype=torch.long
    )
    k_pos = torch.arange(cache_len, device=device, dtype=torch.long)
    # [T, S]
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    written = k_pos.unsqueeze(0) < (end_offset.to(dtype=torch.long) + query_len)
    allow = causal & written
    # Additive mask: 0 = keep, large negative = block
    mask = torch.zeros(query_len, cache_len, device=device, dtype=dtype)
    neg = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    return mask.masked_fill(~allow, neg)


class StreamingMultiheadAttention(StatefulModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
    """

    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        """Initializes a transformer layer with rotary positional encoding."""
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim
        mult = 1
        self.in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, mult * embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        """Initialize fixed-capacity KV cache and scalar end_offset (graph-friendly)."""
        dim_per_head = self.embed_dim // self.num_heads
        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype
        return dict(
            # 0-dim long: write index; advanced in-place by increment_step
            end_offset=torch.zeros((), dtype=torch.long, device=device),
            # Legacy alias kept in sync for any external readers (same storage pattern)
            current_end=torch.zeros((), dtype=torch.long, device=device),
            # Zeros (not NaN): SDPA can produce NaN if masked keys are NaN even
            # when the additive mask blocks them. Padding is handled by attn_mask.
            cache=torch.zeros(
                (2, batch_size, sequence_length, self.num_heads, dim_per_head),
                device=device,
                dtype=dtype,
            ),
        )

    def increment_step(self, state: dict, increment: int = 1):
        """Advance end_offset in-place (no tensor reallocation — CUDA-graph safe)."""
        # Prefer end_offset; migrate legacy current_end-as-length tensors.
        if "end_offset" in state:
            state["end_offset"].add_(int(increment))
            if "current_end" in state and state["current_end"].shape == ():
                state["current_end"].copy_(state["end_offset"])
            return
        # Legacy path: length encoded as current_end.shape[0]
        if "current_end" in state:
            legacy = state["current_end"]
            if legacy.dim() == 0:
                legacy.add_(int(increment))
            else:
                new_size = legacy.shape[0] + increment
                state["current_end"] = torch.zeros(
                    (new_size,), device=legacy.device, dtype=legacy.dtype
                )

    def _complete_kv(self, k, v, state: dict | None):
        """Write keys/values into the streaming cache at the current end_offset."""
        end = state["end_offset"] if "end_offset" in state else state["current_end"]
        if end.dim() != 0:
            # Legacy length-via-shape: convert write index to python length once
            end_idx = end.shape[0]
            end = torch.tensor(end_idx, device=k.device, dtype=torch.long)
        return complete_kv(state["cache"], end, k, v)

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor, state: dict | None):
        """Apply RoPE using streaming end_offset as the position base."""
        streaming_offset = self._streaming_offset(state)
        return self.rope(query, key, offset=streaming_offset)

    def _streaming_offset(self, state: dict | None) -> torch.Tensor | int:
        """Return current stream length / next write index."""
        if "end_offset" in state:
            return state["end_offset"]
        cur = state["current_end"]
        if cur.dim() == 0:
            return cur
        return cur.shape[0]

    def check_model_state(self, model_state: dict):
        """Require and return this module's slice of model_state."""
        if model_state is None:
            raise ValueError("model_state must be provided")
        return self.get_state(model_state)

    def forward(self, query: torch.Tensor, model_state: dict | None):
        """Streaming attention over a fixed-capacity cache with causal padding mask."""
        state = self.check_model_state(model_state)

        projected = self.in_proj(query)
        b, t, _ = projected.shape
        d = self.embed_dim // self.num_heads
        packed = projected.view(b, t, 3, self.num_heads, d)
        q, k, v = torch.unbind(packed, dim=2)
        q, k = self._apply_rope(q, k, state)
        k, v = self._complete_kv(k, v, state)

        # Fixed shapes: full cache capacity S (not dynamic prefix slice)
        s = state["cache"].shape[2]
        end = state["end_offset"] if "end_offset" in state else self._streaming_offset(state)
        if not isinstance(end, torch.Tensor):
            end = torch.tensor(end, device=q.device, dtype=torch.long)
        elif end.dim() != 0:
            end = torch.tensor(end.shape[0], device=q.device, dtype=torch.long)

        attn_mask = _build_streaming_attn_mask(t, s, end, q.device, q.dtype)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask)
        x = x.transpose(1, 2)
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)

        return x
