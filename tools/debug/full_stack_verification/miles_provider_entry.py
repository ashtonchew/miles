import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.arguments import parse_args

from miles.backends.megatron_utils.model_provider import get_model_provider_func

rows = []
with tempfile.TemporaryDirectory() as tmp:
    dist.init_process_group("nccl", init_method="file://" + tmp + "/store", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    torch.manual_seed(17)
    model_parallel_cuda_manual_seed(17)
    try:
        for enabled in (False, True):
            with patch.object(sys, "argv", ["entry", *([] if enabled else ["--no-rope-fusion"])]):
                args = parse_args()
            args.variable_seq_lengths = True
            args.gradient_accumulation_fusion = False
            args.megatron_to_hf_mode = "bridge"
            args.hf_checkpoint = "/tmp/tiny-qwen3"
            args.enable_witness = False
            model = get_model_provider_func(args)().cuda()
            assert model.config.apply_rope_fusion is enabled
            tokens = torch.arange(32, device="cuda").reshape(2, 16)
            pos = torch.arange(16, device="cuda").expand(2, -1)
            with patch.object(
                rope_utils, "fused_apply_rotary_pos_emb", wraps=rope_utils.fused_apply_rotary_pos_emb
            ) as k:
                result = model(input_ids=tokens, position_ids=pos, attention_mask=None)
                result.float().square().mean().backward()
                torch.cuda.synchronize()
                assert torch.isfinite(result).all()
                assert (k.call_count > 0) is enabled
                rows.append(
                    {
                        "fusion": enabled,
                        "kernel_calls": k.call_count,
                        "finite": True,
                        "dtype": str(next(model.parameters()).dtype),
                    }
                )
                Path("/tmp/results/miles-provider-entry.json").write_text(json.dumps(rows, indent=2))
            del model
            torch.cuda.empty_cache()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
