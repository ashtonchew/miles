# Numerical-alignment upstream assessment — 2026-09-25

## Scope and development workflow

Audit base: Miles `41c5e38b94` (freshly fetched upstream/main). User fork:
`https://github.com/ashtonchew/miles`; upstream: `https://github.com/radixark/miles`.
The original checkout was clean and left on its existing branch. Worktrees:

- `test/numerical-alignment-audit`: this report and executable source-level evidence.
- `fix/bridge-runtime-fusion-flags`: narrow RoPE forwarding fix and CPU regression tests.

Read `AGENTS.md`, `.claude/rules/general-code-style.md`,
`docs/developer/contributor-guide.md` and the actual `pyproject.toml`/pre-commit settings.
The config specifies Black/isort line length 119 despite the contributor prose saying
100. Follow the executable config. The repo already tests Bridge helpers by executing
their AST without importing GPU-only dependencies; the regression follows that seam.
Do not infer end-to-end CUDA support from these tests. No GPU jobs, model downloads,
optimizer steps, upstream issues or upstream PRs were initiated by this audit.

## Recommendations

| Finding | Evidence on current source | Recommendation / owner |
| --- | --- | --- |
| Bridge drops `--no-rope-fusion` | Production helper leaves an opposite provider value unchanged; enabled and disabled cases fail before the patch | Submit the narrow Miles fix after maintainer review of supported Bridge configurations. CPU regression is ready; GPU integration remains untested. |
| Bridge drops SwiGLU fusion selection | Actual CLI destination is `bias_swiglu_fusion`, while the provider field is `bias_activation_fusion`; helper omits mapping | Miles issue or follow-up patch. Mapping must respect the model activation selected by Bridge; blindly forwarding a same-name field is wrong. No broad activation patch included. |
| Batch-invariant flag / true-on-policy readiness | Miles initializer does not call `enable_batch_invariant_mode`; current Megatron initializer does. Current Miles explicitly rejects true-on-policy and its tests remain skipped | Clarify capability/docs and test rejection in Miles. Do not simply activate kernels or remove the guard. Current CLI reference describes the mode as sample staleness rejection, which does not explain this backend limitation. |
| Deterministic TP2 NCCL profile diverges | Executed SGLang dependency handler changes Ring to allreduce:tree and pins channels to 8; actual Miles trainer env builder does not mirror the channel settings. TP1 and nondeterministic controls leave env unchanged; prealigned environment remains stable | Miles/SGLang integration report. Recommend validating or resolving a shared profile before process-group creation. No unconditional global NCCL setting patch: it affects other models/topologies. CPU evidence proves configuration divergence, not distributed failure causality. |
| Batch-invariant RMSNorm input gradient is wrong | Current radixark/Megatron-LM backward produces 0.700768 max error against autograd for nonunit weights; unit-weight control passes. Corrected equation gives ~1.2e-7 | Strongest correctness report: radixark/Megatron-LM, with Miles linked. Candidate one-line patch and 12 CPU backward cases included. CUDA forward/backward must be tested before declaring production readiness. |
| Residual rounding differs between engines | Current Miles SGLang dependency's two native norm modes differ at 400/2048 BF16 elements, max 0.015625; each matches its respective arithmetic reference | Alignment configuration/documentation, not an unconditional default change. Current SGLang main has different semantics: both modes normalize the FP32 sum. A portable patch needs version-aware tests. |

No successful full-model parity result or corrected optimizer update is claimed.
The RMSNorm backward defect is independent of the forward logprob discrepancy.
`BatchInvariantRMSNormFn.forward` also deserves a separate review of zero-centered
weights: its effective-weight local appears unused. The supplied backward patch
makes no claim to repair that forward path.

## Tests and limitations

The initial exploratory provider tests used a same-name activation field. Inspecting
Megatron's parser exposed that as insufficient; the final production regression is
narrowed to the real RoPE argument. SwiGLU stays in the source audit with its actual
CLI destination. This prevents a passing synthetic test from certifying a useless fix.

Final RoPE regression: two enabled/disabled failures before the patch; absent-argument
control passes. After patch: 3 RoPE tests and 6 existing MTP tests pass. The production
patch preserves provider settings when the argument is absent, and does not activate
batch-invariant mode or change model dimensions/activation selection.

Audit uses actual AST-extracted functions with external configuration helpers mocked.
It does not import full CUDA packages. NCCL's collectives, GPU norm kernels, Ray worker
inheritance, performance and end-to-end training are not exercised. Root pytest fixtures
are excluded for the focused provider test because they import the complete serving stack.
The full Miles suite and GPU CLI initialization are not certified by these results.

Current dependency revisions:

- Megatron `miles-main`: `f148a32b4385b758b66a77c9c3ad1641f1295d4b`.
- SGLang `sglang-miles`: `880e3d2453eb7ef1738350e8c35ba2b956cc93a9`.
- SGLang `main` comparison: `31b931edc358c0ec01119a3a55952c1ea334a00e`.

These are branch heads inspected, not proof that an already-published container embeds
those exact revisions. `sources.json` records immutable URLs and SHA-256 checksums.
GitHub searches found no RMSNorm-backward issue in radixark/Megatron-LM and no clearly
matching Bridge-fusion fix in the returned Miles PR results; searches are not exhaustive.

## Reproduce the CPU audit

From this directory, with Python and CPU PyTorch available:

```python
import hashlib, json, pathlib, urllib.request
root = pathlib.Path('/tmp/miles-upstream-audit-sources')
root.mkdir(exist_ok=True)
for name, entry in json.loads(pathlib.Path('sources.json').read_text()).items():
    data = urllib.request.urlopen(entry['url']).read()
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    (root / name).write_bytes(data)
```

```sh
python audit.py --miles /path/to/miles-audit-worktree --sources /tmp/miles-upstream-audit-sources
# Exits 1 while the confirmed defects remain; normal mode saves diagnostic results.
python audit.py --miles /path/to/miles-audit-worktree --sources /tmp/miles-upstream-audit-sources --require-clean
python check_rmsnorm_patch.py /tmp/miles-upstream-audit-sources/megatron_batch_invariant.py
```

`check_rmsnorm_patch.py` applies the reviewed single-expression change only in a temporary
source copy, executes the actual patched backward, and compares 12 cases with autograd.
It checks nonunit/unit weights and centered/noncentered backward conventions. It does
not execute the CUDA forward, so centered forward correctness remains separate.

On the fix worktree:

```sh
python -m pytest --confcutdir=tests/fast/backends/megatron_utils -o addopts='' -p no:cacheprovider tests/fast/backends/megatron_utils/test_bridge_mtp_detachment.py -q
```

## Narrow upstream submission drafts

**Miles: `fix(megatron): honor RoPE fusion flags in Bridge mode`**
Bridge bypasses Megatron's ordinary argument-to-config translation. As a result,
`--no-rope-fusion` can leave the provider's fused path active. Forward the parsed
RoPE setting, preserving providers when that argument is absent. Include the failing
CPU reproducer and green regression; state that no complete GPU parity repair is claimed.

**Megatron: `fix: correct batch-invariant RMSNorm input gradient`**
For nonunit normalization weights, the backward subtracts a correction multiplied by
an extra weight. Remove that factor; compare against autograd using nonunit weights
so the defect cannot be hidden by all-ones initialization. Include the independent
CPU evidence and request CUDA parity/gradient coverage before merge.

**Miles/SGLang: deterministic TP2 weight-sync environment mismatch**
Report the actual handler and trainer-builder divergence, with immutable revisions and
the separate experiment's before/after runtime evidence. Ask maintainers where shared
communicator validation should live. Avoid advertising the profile as a universal fix.

Final candidate validation: all repository pre-commit hooks passed on the changed
production and test files; the 9-test Bridge suite passed after those hooks.
