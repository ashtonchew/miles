"""Prepare the NCCL diagnostic; paid execution requires an explicit CLI flag."""

import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

IMAGE = "radixark/miles@sha256:9d01662b9bf3361ac68b57211a2fcb3b6fef63ba9e4f2da01a2758bbe08d2279"


def run_matrix() -> bytes:
    root = Path("/tmp/nccl-evidence")
    root.mkdir()
    for repeat in range(2):
        command = [
            sys.executable,
            "/opt/repro/nccl_transfer_profile.py",
            "--miles",
            "/opt/repro/sources/miles",
            "--sglang",
            "/opt/repro/sources/sglang",
            "--output",
            str(root / f"repeat-{repeat}"),
        ]
        process = subprocess.Popen(command, start_new_session=True)
        try:
            returncode = process.wait(timeout=920)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            (root / "timeout.txt").write_text(f"repeat {repeat} exceeded 920 seconds\n")
            break
        (root / f"exit-{repeat}.txt").write_text(str(returncode) + "\n")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(root, arcname="nccl-evidence")
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-paid", action="store_true", help="Requires prior explicit $15 phase approval")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    manifest = json.loads((repository / "docs/developer/audits/nccl-transfer-profile/sources.json").read_text())
    for name, entry in manifest.items():
        actual = hashlib.sha256((args.sources / name).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"source hash mismatch: {name}")
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = dict(
        image=IMAGE,
        gpu="H200:4",
        cpu=8,
        memory_mib=32768,
        timeout_seconds=1920,
        startup_timeout_seconds=180,
        retries=0,
        repeats=2,
        phase_ceiling_usd=15,
        models=0,
        optimizer_updates=0,
        paid_execution=args.approve_paid,
    )
    print(json.dumps(plan, indent=2))
    if not args.approve_paid:
        return

    # Modal is optional and must not be contacted during the default dry run.
    import modal

    app = modal.App("miles-nccl-transfer-profile")
    image = (
        modal.Image.from_registry(IMAGE)
        .add_local_file(Path(__file__).with_name("nccl_transfer_profile.py"), "/opt/repro/nccl_transfer_profile.py")
        .add_local_dir(args.sources, "/opt/repro/sources")
    )
    remote = app.function(
        image=image,
        gpu="H200:4",
        cpu=(8, 8),
        memory=(32768, 32768),
        timeout=1920,
        startup_timeout=180,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        serialized=True,
    )(run_matrix)
    with app.run():
        evidence = remote.remote()
    with args.output.open("xb") as output:
        output.write(evidence)


if __name__ == "__main__":
    main()
