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
from megatron.training.arguments import parse_args
from transformers import Qwen3Config

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


def build_model(enabled: bool):
    with patch.object(sys, "argv", ["bridge-rope", *([] if enabled else ["--no-rope-fusion"])]):
        args = parse_args()
    assert args.apply_rope_fusion is enabled
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
    provider.apply_rope_fusion = not enabled
    _apply_bridge_runtime_config(provider, args)
    assert provider.apply_rope_fusion is enabled
    provider.params_dtype = torch.float32
    provider.bf16 = False
    provider.fp16 = False
    provider.attention_dropout = 0.0
    provider.hidden_dropout = 0.0
    provider.finalize()
    model = provider.provide(pre_process=True, post_process=True).cuda()
    assert model.config.apply_rope_fusion is enabled
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group("nccl", init_method=f"file://{directory}/store", rank=0, world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        torch.manual_seed(17)
        model_parallel_cuda_manual_seed(17)
        unfused = build_model(False)
        fused = build_model(True)
        fused.load_state_dict(unfused.state_dict())
        tokens = torch.arange(128, device="cuda").reshape(2, 64) % 256
        positions = torch.arange(64, device="cuda").expand(2, -1)
        results = []
        outputs = []
        gradients = []
        original_kernel = rope_utils.fused_apply_rotary_pos_emb
        assert original_kernel is not None, "fused CUDA RoPE implementation is unavailable"
        for enabled, model in ((False, unfused), (True, fused)):
            with patch.object(rope_utils, "fused_apply_rotary_pos_emb", wraps=original_kernel) as kernel:
                output = model(input_ids=tokens, position_ids=positions, attention_mask=None)
                output = output[0] if isinstance(output, tuple) else output
                loss = output.float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                assert torch.isfinite(output).all()
                assert (kernel.call_count > 0) is enabled
                outputs.append(output.detach())
                grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
                assert grads and all(torch.isfinite(g).all() for g in grads.values())
                gradients.append(grads)
                results.append({"fusion": enabled, "fused_kernel_calls": kernel.call_count, "loss": loss.item()})
        assert gradients[0].keys() == gradients[1].keys()
        output_error = (outputs[0] - outputs[1]).abs().max().item()
        grad_error = max((gradients[0][n] - gradients[1][n]).abs().max().item() for n in gradients[0])
        torch.testing.assert_close(outputs[0], outputs[1], atol=1e-4, rtol=1e-4)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-4, rtol=1e-4)
        options.output.write_text(
            json.dumps(
                {
                    "cases": results,
                    "max_output_error": output_error,
                    "max_gradient_error": grad_error,
                    "torch": torch.__version__,
                },
                indent=2,
            )
        )
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
