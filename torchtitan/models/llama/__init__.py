# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama3_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "ado": ModelArgs(dim=768, n_layers=12, n_heads=12, max_seq_len=1024),
    "ado1b": ModelArgs(dim=2048, n_layers=24, n_heads=16, max_seq_len=1024, multiple_of=8, ffn_dim_multiplier=None),
    "doremi500": ModelArgs(dim=1024, n_layers=12, n_heads=16, max_seq_len=1024, ffn_dim_multiplier=8192 / 2730), # enforces hidden dim 8192
    "3b": ModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        multiple_of=1024,
        rope_theta=500000, # no ffn_dim_multiplier gives intermediate size 8192
    ),
    }

llama3_configs["smollm162m"] = llama3_configs["ado"]