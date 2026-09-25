import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig

rows = []
with tempfile.TemporaryDirectory() as tmp:
    dist.init_process_group("nccl", init_method="file://" + tmp + "/store", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(41)
            value = torch.randn(128, 4, 32, device="cuda", dtype=dtype)
            freqs = RotaryEmbedding(32, rotary_percent=1.0)(64)
            lengths = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
            results = []
            grads = []
            calls = []
            for fused in (False, True):
                cfg = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4, apply_rope_fusion=fused)
                x = value.clone().requires_grad_(True)
                kernel = rope_utils.fused_apply_rotary_pos_emb_thd
                assert kernel is not None
                with patch.object(rope_utils, "fused_apply_rotary_pos_emb_thd", wraps=kernel) as observer:
                    y = rope_utils.apply_rotary_pos_emb(
                        x,
                        freqs,
                        cfg,
                        cu_seqlens=lengths,
                        cp_group=parallel_state.get_context_parallel_group(),
                        max_seqlen=64,
                    )
                    y.float().square().mean().backward()
                    torch.cuda.synchronize()
                    assert (observer.call_count > 0) is fused
                    results.append(y.detach())
                    grads.append(x.grad)
                    calls.append(observer.call_count)
            tol = 1e-5 if dtype == torch.float32 else 0.02
            row = dict(
                dtype=str(dtype),
                calls=calls,
                max_output_error=(results[0] - results[1]).abs().max().item(),
                max_gradient_error=(grads[0] - grads[1]).abs().max().item(),
                atol=tol,
                rtol=tol,
            )
            rows.append(row)
            Path("/tmp/results/rope-thd.json").write_text(json.dumps(rows, indent=2))
            torch.testing.assert_close(results[0], results[1], atol=tol, rtol=tol)
            torch.testing.assert_close(grads[0], grads[1], atol=tol, rtol=tol)
            row["passed"] = True
            Path("/tmp/results/rope-thd.json").write_text(json.dumps(rows, indent=2))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
