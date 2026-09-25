# Reproduce NCCL channel mismatch during weight transfer

A three-rank synthetic broadcast reproduces `Message truncated : received 6144
bytes instead of 2048` when the sender leaves channel limits unset and both
receivers set `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=8`. The sender negotiates
24 collective channels; each receiver negotiates eight. Matching all three
ranks to eight channels passes.

This is relevant to deterministic CUDA TP2 serving: SGLang pins its channel
count, while Miles' trainer environment does not currently mirror that setting.
The standalone test models Miles' weight-update group: one trainer sender plus
two serving ranks. It runs the actual Miles and SGLang group-helper bodies,
extracted from pinned source files without importing the model stack.

## Measured results

Each row ran twice in fresh processes on the same host. All other inherited
settings were unchanged between cases.

| Case | Sender / receiver algorithm | Sender / receiver channel limit | Result, both runs |
| --- | --- | --- | --- |
| baseline | Ring / Ring | unset / unset | Pass |
| algorithm-only | Ring / allreduce:tree | 8 / 8 | Pass |
| channels-only | allreduce:tree / allreduce:tree | unset / 8 | Truncated message |
| combined | Ring / allreduce:tree | unset / 8 | Truncated message |
| matched | allreduce:tree / allreduce:tree | 8 / 8 | Pass |

Each passing case verifies exact tensor equality on all three ranks for 36
broadcasts: uint8, BF16 and FP32; 32, 1024, 3072 and 131072 elements; three
patterns per shape. Across both repetitions, 648 rank-level checks pass.
Every failing case stops before the first tensor check, on the initial 32-byte
uint8 broadcast. The reported 6144/2048 sizes therefore are not the user tensor's
payload size.

Selected NCCL output from the channel-only case:

```text
rank 0: 24 coll channels, 24 collnet channels, 16 nvls channels, 32 p2p channels
rank 1: 8 coll channels, 8 collnet channels, 16 nvls channels, 8 p2p channels
rank 2: 8 coll channels, 8 collnet channels, 16 nvls channels, 8 p2p channels
rank 1: bootstrap.cc:225 NCCL WARN Message truncated : received 6144 bytes instead of 2048
rank 2: bootstrap.cc:225 NCCL WARN Message truncated : received 6144 bytes instead of 2048
```

`results.json` records all ten cases, per-rank status, channel counts and errors.
A sender receipt left at `group_initialized` is incomplete: torchrun terminates
its peers after a receiver raises. It is not a successful sender result. NCCL
initialization is lazy; creating the Python group does not certify a usable
communicator.

## Runtime and source pins

- NVIDIA H200, three devices used from a four-GPU allocation; driver 580.95.05.
- PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, linked NCCL `2.29.7`.
- Image: `radixark/miles@sha256:9d01662b9bf3361ac68b57211a2fcb3b6fef63ba9e4f2da01a2758bbe08d2279`.
- Miles helper: `366f614b8ba3904984d431291dd396be48ae1aed`.
- SGLang helper: `5a8da8cc3fc1ebb5a308f1d1283e2fdf687ee220`.

The image's `NCCL_VERSION` environment label said `2.28.3-1`; both PyTorch's
runtime query and the NCCL error identify `2.29.7`. The linked version is the
relevant one. `nvidia-smi topo -m` failed inside the container, so no topology
matrix is claimed.

## Reproduce

Download the two files listed in `sources.json` into a directory preserving the
manifest's relative paths, and verify their SHA-256 hashes. No model is needed.
For example, from this document's directory:

```sh
python - <<'PY'
import hashlib, json, pathlib, urllib.request
root = pathlib.Path('/tmp/nccl-repro-sources')
for name, entry in json.loads(pathlib.Path('sources.json').read_text()).items():
    data = urllib.request.urlopen(entry['url']).read()
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
PY
```

From the Miles repository root, inside the pinned GPU runtime:

```sh
python tools/debug/nccl_transfer_profile.py \
  --miles /tmp/nccl-repro-sources/miles \
  --sglang /tmp/nccl-repro-sources/sglang \
  --output /tmp/nccl-first
# Repeat with --output /tmp/nccl-second for fresh communicator creation.
```

Use `--case channels-only` to isolate that case. The full matrix intentionally
returns nonzero when a case fails; inspect `summary.json`, `rank*.json` and the
NCCL logs. Each case has a 180-second process deadline and a 60-second group
timeout. `--backend gloo` checks the harness on CPU and does not test NCCL.

## Scope and proposed follow-up

This diagnostic establishes the channel-limit failure on the tested runtime.
It does not exercise Ray launch, HTTP metadata, LoRA loading, existing trainer
TP communicators, multi-node transfer or model numerical parity.

[SGLang #34159](https://github.com/sgl-project/sglang/pull/34159) introduced channel
pinning for deterministic serving all-reduce. [NCCL's qualified algorithm syntax](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-algo)
leaves unlisted collectives at their defaults; the passing algorithm-only case
is consistent with that behavior.

A production fix should coordinate channel limits before workers initialize,
scoped to deterministic multi-GPU serving with distributed weight transfer.
Explicit conflicting overrides need a clear error. Copying every serving NCCL
setting into training would change unrelated reductions. This draft supplies the
reproduction for reviewing that policy; it does not change runtime defaults.

## Launch policy

Miles resolves a shared channel count before constructing worker launch specs for managed CUDA broadcast transfers. If a weight-receiving engine uses deterministic inference with TP greater than one, the actor trainer and every participating CUDA engine receive matching `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS`. This includes TP1 receivers in a mixed deployment. The count comes from SGLang's `SGLANG_DETERMINISTIC_NCCL_NCHANNELS` setting, including its installed default.

Set `SGLANG_DETERMINISTIC_NCCL_NCHANNELS` in the job environment to choose another count. Explicit incompatible NCCL channel overrides raise a startup error with the expected value. Trainer-specific environment values retain their existing precedence over inherited values. Worker-specific overrides are checked again when the launch environment is built. The resolved policy stays in the spec for worker restarts.

The policy preserves `NCCL_ALGO`. It affects the actor's other NCCL collectives as well as weight transfer, so training collective performance should be measured before deployment. Frozen models, placeholder groups, and critics receive no policy override. Colocated IPC, disk-delta, P2P, ROCm, debug-only runs, and external serving retain their existing behavior. External servers require channel settings to be coordinated by their operator. Older SGLang versions without the deterministic channel setting retain their existing behavior.

## Launch-policy regression checks

The focused suite passes 32 tests locally, covering mixed TP sizes, per-group overrides, configured counts, explicit conflicts, unaffected modes, environment preservation, restart reuse, and the complete launch entrypoint with serving factories isolated. Replacing only `entrypoint.py` with the pre-fix version makes the launch regression fail with `KeyError: 'NCCL_MIN_NCHANNELS'`; restoring the patch passes.

```sh
PYTHONPATH=. python -m pytest --confcutdir=tests/fast/ray/specs -o addopts='' \
  tests/fast/ray/specs/test_weight_update_env.py -q
```

A second entrypoint regression uses the full repository fixtures in `tests/fast/ray/specs/test_entrypoint.py`. That test requires the serving dependencies and was added for CI; it was not run locally. The earlier GPU matrix and HTTP sessions above establish the behavior of matching channel limits. The new automatic launch policy still needs a Ray-launched GPU run, exact server readback, and a training-collective performance measurement.
