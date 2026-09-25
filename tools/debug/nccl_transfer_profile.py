"""Three-rank weight-transfer diagnostic; no model, training, or Ray required."""

import argparse
import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

PROFILES = {
    "baseline": (("Ring", None), ("Ring", None)),
    "algorithm-only": (("Ring", "8"), ("allreduce:tree", "8")),
    "channels-only": (("allreduce:tree", None), ("allreduce:tree", "8")),
    "combined": (("Ring", None), ("allreduce:tree", "8")),
    "matched": (("allreduce:tree", "8"), ("allreduce:tree", "8")),
}
KEYS = ("NCCL_ALGO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_group_helper(path: Path, name: str, torch):
    """Execute the actual helper without importing unrelated model dependencies."""
    from packaging.version import parse
    from torch.distributed import distributed_c10d

    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    scope = dict(vars(distributed_c10d))
    scope.update(torch=torch, parse=parse, torch_release=tuple(parse(torch.__version__).release[:2]))
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


def worker(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    algorithm, channels = PROFILES[args.case][rank != 0]
    for key, value in zip(KEYS, (algorithm, channels, channels), strict=True):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,COLL,NET,TUNING"
    os.environ["NCCL_DEBUG_FILE"] = str(args.output / f"nccl-rank{rank}.log")

    # Import only after setting this process's profile, before any CUDA/NCCL use.
    import torch
    import torch.distributed as dist

    source = (
        args.miles / "miles/utils/distributed_utils.py"
        if rank == 0
        else args.sglang / "python/sglang/srt/utils/common.py"
    )
    helper_name = "init_process_group" if rank == 0 else "init_custom_process_group"
    device = torch.device(f"cuda:{rank}" if args.backend == "nccl" else "cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(device)
    receipt = {
        "rank": rank,
        "role": "sender" if rank == 0 else "receiver",
        "case": args.case,
        "backend": args.backend,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version() if args.backend == "nccl" else None,
        "gpu": torch.cuda.get_device_name(device) if args.backend == "nccl" else None,
        "environment": {key: value for key, value in os.environ.items() if key.startswith("NCCL_")},
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "helper": helper_name,
        "status": "before_group_init",
        "checks": [],
    }
    receipt_path = args.output / f"rank{rank}.json"
    write_json(receipt_path, receipt)
    try:
        # The real serving helper requires an existing default group. Keep its
        # rendezvous distinct from the custom weight-update group's PrefixStore.
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
        store = dist.distributed_c10d._get_default_store()
        store = dist.PrefixStore("weight-transfer-diagnostic", store)
        helper = load_group_helper(source, helper_name, torch)
        group = helper(
            backend=args.backend,
            store=store,
            world_size=3,
            rank=rank,
            group_name="miles-pp_0",
            timeout=timedelta(seconds=60),
        )
        receipt["status"] = "group_initialized"
        write_json(receipt_path, receipt)
        for dtype in (torch.uint8, torch.bfloat16, torch.float32):
            for count in (32, 1024, 3072, 131072):
                for iteration in range(3):
                    expected = ((torch.arange(count, device=device) + iteration) % 127).to(dtype)
                    tensor = expected.clone() if rank == 0 else torch.full_like(expected, 255)
                    handle = dist.broadcast(tensor, 0, group=group, async_op=True)
                    handle.wait()
                    equal = torch.equal(tensor, expected)
                    receipt["checks"].append(
                        {"dtype": str(dtype), "count": count, "iteration": iteration, "exact": equal}
                    )
                    write_json(receipt_path, receipt)
                    if not equal:
                        raise AssertionError(f"corrupt transfer: {dtype=} {count=} {iteration=}")
        dist.destroy_process_group(group)
        dist.destroy_process_group()
        receipt["status"] = "passed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = repr(exc)
        raise
    finally:
        write_json(receipt_path, receipt)


def launch(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=False)
    if args.backend == "nccl":
        for name, command in (
            ("gpu.csv", ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"]),
            ("topology.txt", ["nvidia-smi", "topo", "-m"]),
        ):
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            (args.output / name).write_text(result.stdout + result.stderr)
    results = []
    cases = tuple(PROFILES) if args.case == "all" else (args.case,)
    for case in cases:
        directory = args.output / case
        directory.mkdir()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=3",
            str(Path(__file__).resolve()),
            "--worker",
            "--backend",
            args.backend,
            "--case",
            case,
            "--miles",
            str(args.miles),
            "--sglang",
            str(args.sglang),
            "--output",
            str(directory),
        ]
        with (directory / "workers.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = 124
        results.append({"case": case, "returncode": returncode, "command": command})
        print(json.dumps(results[-1]), flush=True)
        write_json(args.output / "summary.json", {"backend": args.backend, "cases": results})
    return int(any(result["returncode"] != 0 for result in results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--case", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.miles = args.miles.resolve()
    args.sglang = args.sglang.resolve()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
