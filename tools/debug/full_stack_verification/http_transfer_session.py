import os

os.environ["NCCL_MIN_NCHANNELS"] = os.environ["NCCL_MAX_NCHANNELS"] = "8"
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
import faulthandler
import json
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    pause_engines,
    resume_engines,
    set_weight_version,
)
from miles.utils import async_utils

rank = int(os.environ["RANK"])
print("imports complete", rank, flush=True)
faulthandler.dump_traceback_later(45, repeat=True)
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=180))
print("trainer group ready", rank, flush=True)
x = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(x)
assert x.item() == 3
print("trainer reduce passed", rank, flush=True)
if rank == 0:
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)  # Miles actors are not torchrun workers.
    client = SGLangApiClient("http://127.0.0.1:30002")
    args = Namespace(rollout_num_gpus_per_engine=2, pause_generation_mode="retract")
    print("connecting weight group", flush=True)
    group = connect_rollout_engines_from_distributed(args, "miles-session-final", [client])
    print("weight group ready", flush=True)
    checks = []
    for step in range(3):
        pause_engines(args, [client])
        begin_weight_update([client])
        source = (torch.arange(512, device="cuda") % 127 + step).to(torch.bfloat16)[::2]
        assert not source.is_contiguous()
        print("sending step", step, flush=True)
        responses = async_utils.wait_futures(
            update_weights_from_distributed("miles-session-final", group, [client], [("model.norm.weight", source)])
        )
        assert all(r.get("success", False) for r in responses), responses
        end_weight_update([client])
        set_weight_version([client], step + 1)
        resume_engines([client])
        checks.append(
            {
                "step": step,
                "acknowledgements": responses,
                "readback_verified": False,
                "readback_limit": "Pinned Qwen3ForCausalLM has no get_weights_by_name method",
                "elements": 256,
                "noncontiguous_source": True,
            }
        )
        Path("/tmp/results/http-transfer.json").write_text(json.dumps(checks, indent=2))
    disconnect_rollout_engines_from_distributed(args, "miles-session-final", group, [client])
dist.barrier()
dist.destroy_process_group()
faulthandler.cancel_dump_traceback_later()
