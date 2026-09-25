# Deterministic TP2 weight-transfer investigation

## Finding

Miles joins one trainer sender and two TP2 serving workers in a three-rank
weight-update process group. SGLang's deterministic CUDA path sets
`NCCL_ALGO=allreduce:tree` and pins `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=8`.
Miles does not currently coordinate those channel settings with its trainer.

Our full-model run failed during initial transfer with `received 6144 bytes
instead of 2048`; a later run with a shared profile completed transfer. That
comparison changed several settings together. It demonstrates a workaround,
not an isolated cause. The byte counts do not establish which tensor was wrong.
Successful transfer also does not establish logprob parity.

## Precedent and interpretation

[SGLang #34159](https://github.com/sgl-project/sglang/pull/34159) introduced the
channel pinning to stabilize serving all-reduce across prefill and decode.
That is a serving arithmetic requirement, not a test of mixed training/serving
weight transfer.

[NCCL's function-qualified syntax](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-algo)
leaves unlisted collectives at their defaults. `allreduce:tree` therefore does
not mean Tree broadcast. [NCCL v2.27.7 tuning](https://github.com/NVIDIA/nccl/blob/v2.27.7-1/src/graph/tuning.cc#L236)
permits Ring for broadcast. A raw Ring/tree string comparison is insufficient
reason to reject the transfer configuration.

A stronger hypothesis is inconsistent channel limits. NCCL first reconciles
[topology across ranks](https://github.com/NVIDIA/nccl/blob/v2.27.7-1/src/init.cc#L914),
then applies [local channel limits](https://github.com/NVIDIA/nccl/blob/v2.27.7-1/src/graph/connect.cc#L440).
Those source links explain a possible mechanism; they do not identify the
runtime library version or prove that mechanism caused our failure.

Miles already coordinates `NCCL_CUMEM_ENABLE` in its trainer environment builder.
A shared initialization policy has local precedent. Copying the entire serving
NCCL environment into training would also affect unrelated training collectives.

## Reproduction

`tools/debug/nccl_transfer_profile.py` runs five cases, each with fresh processes:

| Case | Sender / receiver algorithm | Sender / receiver channels |
| --- | --- | --- |
| baseline | Ring / Ring | default / default |
| algorithm-only | Ring / allreduce:tree | 8 / 8 |
| channels-only | allreduce:tree / allreduce:tree | default / 8 |
| combined | Ring / allreduce:tree | default / 8 |
| matched | allreduce:tree / allreduce:tree | 8 / 8 |

The script executes the actual Miles and SGLang process-group helpers extracted
from the supplied source trees. This avoids importing model packages while
preserving the helper bodies. It uses a separate Gloo rendezvous store, then the
custom NCCL group and the asynchronous broadcast used by Miles. Each rank gets
its own GPU. The full Ray/HTTP/LoRA transfer pipeline is deliberately absent.

Each case checks 36 transfers per rank: four sizes, three dtypes, three patterns.
Receipts include exact equality, software versions, source hashes and each
rank's environment before group creation. GPU runs include device/topology data
and NCCL logs. Each process set has a 180-second deadline; the group timeout is
60 seconds. Output directories must be new to prevent mixing runs.

Use source revisions from the failing experiment before testing newer versions:
Miles `366f614b8ba3904984d431291dd396be48ae1aed` and SGLang
`5a8da8cc3fc1ebb5a308f1d1283e2fdf687ee220`. `sources.json` pins the required files
and hashes. No weights or model downloads are needed.

```sh
python tools/debug/nccl_transfer_profile.py \
  --miles /path/to/pinned-miles --sglang /path/to/pinned-sglang \
  --output /tmp/nccl-profile-first
# Repeat with a new output directory to test independent communicator creation.
```

Add `--backend gloo` for the CPU smoke check. Passing Gloo checks does not verify
NCCL behavior. Two local CPU runs passed all five cases and all 540 rank-level checks each
with PyTorch 2.8.0: first using current Miles and the supplied SGLang snapshot,
then the exact source revisions from the failing experiment. `cpu-results.json`
records per-rank status and source hashes.

Interpretation requires logs, not only exit codes. Compare negotiated channel
counts and the first failed operation. Check inherited CTA settings, NCCL config
files, plugins and the linked NCCL library. An unset channel environment variable
can still have a system configuration value. Do not interpret an unrelated
startup failure as a reproduced transport defect.

If this minimal test passes, recreate the four-process topology: both trainer
ranks first form their TP2 group, both serving ranks form theirs, then ranks
0, 2 and 3 form the weight-update group. This tests initialization order and
cached NCCL parameters. A passing three-rank test does not clear the production
configuration.

## Proposed fix policy

First publish diagnostic evidence. Do not reject all unequal NCCL environment
strings. If channel-only divergence reproduces, choose one channel policy before
Ray actors or communicators initialize, scoped to deterministic CUDA TP>1 with
distributed weight transfer. Preserve unrelated user settings and report explicit
conflicting overrides. Ask maintainers whether actor construction or a
communicator-specific API should own it. Environment changes after NCCL
initialization are too late to rely on.

An upstream fix should include the failing minimal case, the passing corrected
case, repeated fresh-process results, TP1/nondeterministic controls, and a real
Miles weight-transfer smoke. Until then, keep the reproduction on the fork and
avoid claiming that a source-only environment test proves a transport fix.

## Proposed paid phase (not approved)

Use the experiment's base image pinned to
`radixark/miles@sha256:9d01662b9bf3361ac68b57211a2fcb3b6fef63ba9e4f2da01a2758bbe08d2279`.
Run two five-case matrices on one H200:4 container, using three devices. Cap
execution at 1,920 seconds, startup at 180 seconds, retries at zero, and containers
at one. No models, checkpoints, dataset access or training steps. Retain logs
and receipts locally.

At the published H200 rate of $0.001261/GPU-second, 2,100 seconds reserves $10.59
for GPUs. Eight CPU cores and 32 GiB reserve about $0.37 more. Request a $15
phase ceiling including build/control overhead. This is a resource estimate,
not a billing cap; stop rather than retry if the phase cannot fit the ceiling.
[Modal pricing](https://modal.com/pricing).

The optional Modal launcher defaults to a local dry run and verifies the source
hashes before any cloud contact:

```sh
python tools/debug/modal_nccl_transfer_profile.py \
  --sources /path/to/source-files --output /tmp/nccl-evidence.tar.gz
```

Only after explicit phase approval, add `--approve-paid`. It starts one container,
runs the matrix twice, and saves the receipts and NCCL logs as a local archive.
It does not mount experiment volumes or credentials. The launcher has been
checked locally; cloud execution remains unverified.
