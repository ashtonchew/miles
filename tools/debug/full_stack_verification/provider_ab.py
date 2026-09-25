import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from megatron.bridge import AutoBridge
from megatron.training.arguments import parse_args
from transformers import Qwen3Config


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


base = module("baseline_provider", "/tmp/base_provider.py")
fixed = module("fixed_provider", "/tmp/combined_provider.py")
with patch.object(sys, "argv", ["probe", "--no-rope-fusion", "--no-bias-swiglu-fusion"]):
    args = parse_args()
args.variable_seq_lengths = True
c = Qwen3Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
)
c.architectures = ["Qwen3ForCausalLM"]
rows = []
for name, m in [("base", base), ("fixed", fixed)]:
    p = AutoBridge.from_hf_config(c).to_megatron_provider(load_weights=False)
    p.apply_rope_fusion = True
    p.bias_activation_fusion = True
    m._apply_bridge_runtime_config(p, args)
    assert p.activation_func is torch.nn.functional.silu and p.gated_linear_unit
    rows.append(dict(source=name, rope=p.apply_rope_fusion, swiglu=p.bias_activation_fusion, args_swiglu=args.swiglu))
assert rows[0]["rope"] and rows[0]["swiglu"]
assert not rows[1]["rope"] and not rows[1]["swiglu"]
Path("/tmp/results/provider-ab.json").write_text(json.dumps(rows, indent=2))
