"""Real-import Bridge/Qwen3 construction and CUDA RoPE smoke; no weights downloaded."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import mlp as mlp_module
from megatron.training.arguments import parse_args
from transformers import Qwen3Config

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


def build_model(enabled: bool, kind="rope", dtype=torch.float32):
    flag = "--no-rope-fusion" if kind == "rope" else "--no-bias-swiglu-fusion"
    with patch.object(sys, "argv", ["bridge-fusion", *([] if enabled else [flag])]):
        args = parse_args()
    args.variable_seq_lengths = True
    args.gradient_accumulation_fusion = False
    assert getattr(args, "apply_rope_fusion" if kind == "rope" else "bias_swiglu_fusion") is enabled
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
    field = "apply_rope_fusion" if kind == "rope" else "bias_activation_fusion"
    setattr(provider, field, not enabled)
    _apply_bridge_runtime_config(provider, args)
    assert getattr(provider, field) is enabled
    provider.params_dtype = dtype
    provider.bf16 = dtype == torch.bfloat16
    provider.fp16 = False
    provider.attention_dropout = 0.0
    provider.hidden_dropout = 0.0
    provider.finalize()
    model = provider.provide(pre_process=True, post_process=True).cuda()
    assert getattr(model.config, field) is enabled
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    torch.cuda.set_device(0)
    receipts = []
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group("nccl", init_method=f"file://{directory}/store", rank=0, world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        try:
            for kind in ("rope", "swiglu"):
                for dtype in (torch.float32, torch.bfloat16):
                    torch.manual_seed(17)
                    model_parallel_cuda_manual_seed(17)
                    models = [build_model(enabled, kind, dtype) for enabled in (False, True)]
                    models[1].load_state_dict(models[0].state_dict())
                    outputs = []
                    grads = []
                    calls = []
                    module = rope_utils if kind == "rope" else mlp_module
                    name = "fused_apply_rotary_pos_emb" if kind == "rope" else "bias_swiglu_impl"
                    original = getattr(module, name)
                    assert original is not None
                    tokens = torch.arange(128, device="cuda").reshape(2, 64) % 256
                    positions = torch.arange(64, device="cuda").expand(2, -1)
                    for enabled, model in zip((False, True), models, strict=True):
                        with patch.object(module, name, wraps=original) as kernel:
                            output = model(input_ids=tokens, position_ids=positions, attention_mask=None)
                            output = output[0] if isinstance(output, tuple) else output
                            output.float().square().mean().backward()
                            torch.cuda.synchronize()
                            assert torch.isfinite(output).all()
                            assert (kernel.call_count > 0) is enabled
                            outputs.append(output.detach())
                            calls.append(kernel.call_count)
                            grads.append(
                                {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
                            )
                    assert grads[0].keys() == grads[1].keys()
                    atol = 1e-4 if dtype == torch.float32 else 0.02
                    row = dict(
                        kind=kind,
                        dtype=str(dtype),
                        kernel_calls=calls,
                        max_output_error=(outputs[0] - outputs[1]).abs().max().item(),
                        max_gradient_error=max((grads[0][n] - grads[1][n]).abs().max().item() for n in grads[0]),
                        atol=atol,
                        rtol=atol,
                    )
                    receipts.append(row)
                    options.output.write_text(json.dumps(receipts, indent=2))
                    torch.testing.assert_close(outputs[0], outputs[1], atol=atol, rtol=atol)
                    for n in grads[0]:
                        torch.testing.assert_close(grads[0][n], grads[1][n], atol=atol, rtol=atol)
                    row["passed"] = True
                    options.output.write_text(json.dumps(receipts, indent=2))
                    del models, outputs, grads
                    torch.cuda.empty_cache()
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
