import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from RL.attention import *
key = jax.random.PRNGKey(0)

B = 2
T = 8
d_model = 64
num_heads = 4

x = jax.random.normal(key, (B, T, d_model))

attention = MultiHeadSelfAttention(
    d_model=d_model,
    num_heads=num_heads,
    dropout_rate=0.1,
)

params_key, dropout_key = jax.random.split(key)

variables = attention.init(
    {"params": params_key, "dropout": dropout_key},
    x,
    deterministic=False,
)

y = attention.apply(
    variables,
    x,
    deterministic=True,
)

print(y.shape)