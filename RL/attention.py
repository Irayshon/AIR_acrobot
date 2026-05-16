
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.0

    def setup(self):
        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = nn.Dense(self.d_model)
        self.k_proj = nn.Dense(self.d_model)
        self.v_proj = nn.Dense(self.d_model)
        self.out_proj = nn.Dense(self.d_model)

        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def split_heads(self, x):
        """
        x: [B, T, d_model]
        return: [B, num_heads, T, head_dim]
        """
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        x = x.transpose(0, 2, 1, 3)
        return x

    def merge_heads(self, x):
        """
        x: [B, num_heads, T, head_dim]
        return: [B, T, d_model]
        """
        B, H, T, D = x.shape
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(B, T, H * D)
        return x

    def __call__(
        self,
        x,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ):
        """
        x: [B, T, d_model]
        mask: 可选，形状通常是 [B, 1, 1, T] 或 [B, 1, T, T]
              mask=True 表示可见，False 表示屏蔽
        """

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # q, k, v: [B, H, T, D]

        scale = self.head_dim ** -0.5

        attn_logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        # attn_logits: [B, H, T, T]

        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -1e9)

        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        out = jnp.einsum("bhts,bhsd->bhtd", attn_weights, v)
        # out: [B, H, T, D]

        out = self.merge_heads(out)
        out = self.out_proj(out)

        return out