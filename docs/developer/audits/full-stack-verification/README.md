# Bridge and transfer verification, 2026-09-25

These are measured integration results from a pinned Linux CUDA runtime. They
verify the tested configuration and do not establish Qwen3-8B training parity.

## Runtime

- Image `radixark/miles@sha256:9d01662b9bf3361ac68b57211a2fcb3b6fef63ba9e4f2da01a2758bbe08d2279`.
- H200, PyTorch 2.13.0+cu130, Transformer Engine 2.17.0, Transformers 5.12.1,
  Megatron Bridge 0.5.0+40b93089, linked NCCL 2.29.7.
- Megatron source 73b54618f7e58e0f25f619bcfecbe2640765475a;
  SGLang source aea7fb92c047c9c096eae66460acb25dec9ae5a9.
- RoPE patch 0cd31076b7; activation-fusion patch d29e58a10d.

## Real provider A/B

Both source modules are imported normally. The real Megatron parser consumes
`--no-rope-fusion --no-bias-swiglu-fusion`; AutoBridge constructs a Qwen3 provider
from an in-memory HF config. On base, both provider fusion values remain true.
With the patches, both become false. The provider remains gated SiLU even
though the CLI's `args.swiglu` default is false. See `provider-ab.json`.

## Constructed model and CUDA kernels

The model has two layers, hidden size 128, four attention heads, two KV heads,
head dimension 32 and vocabulary 256. Inputs are two 64-token sequences. Each
pair uses identical weights, zero dropout and the same input. Both paths run
forward and backward. Wrappers count calls to the real kernels; no kernel math
is replaced. Standalone gradient-accumulation fusion is disabled because no
optimizer-owned main-grad buffers are allocated.

| Test | Dtype | Fused calls off/on | Max output difference | Max gradient difference |
| --- | --- | --- | --- | --- |
| RoPE | FP32 | 0 / 4 | 9.99123e-5 | 1.09477e-6 |
| RoPE | BF16 | 0 / 4 | 0.005859375 | 6.10352e-5 |
| SwiGLU, independent patch | FP32 | 0 / 2 | 1.37836e-5 | 7.36618e-7 |
| SwiGLU, independent patch | BF16 | 0 / 2 | 0.0048828125 | 6.10352e-5 |

FP32 comparisons used atol=rtol=1e-4; BF16 used atol=rtol=0.02. These are smoke
comparison thresholds, not RL logprob gates or a claim of bitwise equality.
`swiglu-only.json` was obtained with only the activation-fusion provider patch.
`fusion-sweep.json` tests both patches together.

The separate packed-sequence RoPE kernel test uses two sequences of length 64,
four heads and head dimension 32. Fused calls are 0 / 1. FP32 output/gradient
max differences are 2.38419e-7 / 5.82077e-11; BF16 differences are 0.015625 /
3.81470e-6. Tolerances are recorded in `rope-thd.json`. This is kernel coverage,
not an end-to-end packed model test.

## CPU checks

Activation regression: base 6 failed / 14 passed; patched 20 passed locally.
The exact patched file also passes all 20 tests in a fresh Linux Python 3.11
venv with normal root fixtures and no `--confcutdir` override. Dependencies were
installed from requirements.txt and tests/ci/requirements-ci-cpu.txt. This root-
fixture run took 2.77 seconds, with dependency warnings retained.

The broader `tests/fast` run is still in progress. It has failures outside the
changed helper; no full-suite pass is claimed yet. The image's source pins differ
from moving dependency branch heads, which must be considered when interpreting
those failures.

## Reproduction scripts

`tools/debug/full_stack_verification/fusion_sweep.py` imports the real packages,
constructs the models and records forward/backward comparisons. Run in the
pinned GPU environment with both patches:

```sh
PYTHONPATH=/path/to/miles:/root/Megatron-LM python tools/debug/full_stack_verification/fusion_sweep.py --output /tmp/fusion-results.json
```

Run `swiglu_only.py` the same way with only the activation patch. `rope_thd.py`
uses `/tmp/results/rope-thd.json`; create `/tmp/results` first. `provider_ab.py`
expects base and combined provider modules at the temporary paths named in its
source. All model parameters are generated locally; no checkpoint is downloaded.

Remaining gaps: full-suite completion and baseline classification, complete
Miles Bridge model-provider entry point, packed-model integration, Ray-launched
weight updates, full 8B/LoRA/YaRN configuration, optimizer/export/reload, and
production NCCL configuration policy. Keep these distinct from the checks above.
