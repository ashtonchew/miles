# Full-stack verification plan

Prepared after the isolated regressions. No additional paid execution is approved
or started. The existing $15 approval covered the completed NCCL diagnostic only.

## Local findings

`PYTHONPATH=. python -m pytest tests/fast -q --collect-only` fails while importing
root fixtures: `ModuleNotFoundError: No module named 'ray'`. No tests ran.
The local Docker CLI has no running daemon. The upstream CPU workflow uses Linux,
Python 3.11, requirements.txt plus tests/ci/requirements-ci-cpu.txt, and SGLang and
Megatron source checkouts. It also installs Helm. Do not certify a smaller macOS
environment as equivalent.

## RoPE PR #3708

1. Run the complete `tests/fast` suite with root fixtures in the supported Linux
   dependency environment. Preserve JUnit output, skips and collection failures.
   Compare unrelated failures against base 41c5e38b94 rather than editing them away.
2. Run `bridge_rope.py` against the base and patched modules. The script imports
   real Miles, Megatron, Bridge and Transformer Engine. It parses the real flag,
   constructs a small Qwen3 provider from an in-memory HF config, finalizes it,
   constructs the model, and runs forward/backward on CUDA with identical weights.
   It wraps the real fused kernel to count calls without replacing its math.
3. Base must fail flag propagation. Patched must select the requested kernel,
   produce finite outputs/gradients, and pass the documented FP32 comparison.
4. Extend model integration to BF16 and packed-sequence THD; record separate
   numerical tolerances. Do not claim FP32 tiny-model coverage certifies BF16,
   YaRN64K or Qwen3-8B training parity.

The prepared script is syntax-checked only. Runtime dependency/API issues must
be fixed and documented during the controlled run. Missing fused kernels are a
failure, not a passing skip. No model weights are downloaded.

## NCCL PR #3709

This is currently a reproduction PR, not a production fix.

1. Repeat using direct imports of both helpers in the full dependency environment.
2. Use four real processes: trainer TP group (0,1), serving TP group (2,3), then
   the mixed weight group (0,2,3). Initialize each role's TP communicator before
   weight transfer, exercising NCCL parameter caching and initialization order.
3. Exercise the real Miles transfer protocol and SGLang weight-update endpoint,
   with a tiny locally generated model, for initial and repeated updates. Check
   exact tensor equality, update acknowledgements and clean shutdown. Include
   dtype/shape metadata and noncontiguous-source controls.
4. Only then implement a shared channel policy before actor creation and verify
   default deterministic TP2, explicit matching/conflicting overrides, TP1,
   nondeterministic mode and non-distributed transfer modes. Do not globally copy
   the algorithm setting. Test the same policy under the actual Ray launch path.
5. A production-fix draft needs these integration results. Passing our existing
   three-rank broadcast is necessary but not sufficient.

## Remaining upstream work

- Bridge SwiGLU fusion: separately map the actual CLI fields while preserving
  model activation semantics; test SwiGLU and GELU controls before submitting.
- Batch invariance / true-on-policy: initialization wiring and capability docs;
  do not remove the current unsupported guard as a substitute for verification.
- RMSNorm: NVIDIA/Megatron-LM #6859 already contains forward/backward corrections;
  add nonunit-weight CUDA regression coverage and track adoption in the Miles fork.
- Residual precision: version-specific alignment behavior; no unconditional
  default change until forward/backward tests justify it.
- Full Qwen3 numerical parity: separate acceptance gate covering actual 8B BF16
  LoRA/YaRN configuration, optimizer update, export and serving reload.

## Proposed next paid phase

Request $25 for Linux CPU-suite verification plus bounded GPU integration work:
CPU worker (8 cores, 32 GiB, at most 60 min); one H200:4 GPU worker (8 cores,
32 GiB, at most 35 min plus 3 min startup); no automatic retries, no downloaded
model weights, no production training. Resource reservations are about $12.55
at the previously checked rates; retain the remainder for build/control overhead
and explicitly bounded corrective attempts. Stop before exceeding the phase cap.

This phase targets PR verification and a production NCCL integration test. It
cannot honestly guarantee every test passes or establish full 8B training parity.
Record unresolved failures rather than expanding the run without authorization.
