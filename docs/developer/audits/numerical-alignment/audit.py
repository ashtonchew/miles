"""CPU source-level reproductions; no CUDA execution or training claims.

Usage: python audit.py --miles PATH --sources PATH [--require-clean]
Sources are downloaded separately at the revisions recorded in the report.
"""

import argparse
import ast
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


def load_function(path, name, namespace=None, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = dict(namespace or {})
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns[name]


def norm_backward(source):
    fn = load_function(source, "backward", {"torch": torch}, "BatchInvariantRMSNormFn")
    results = []
    for unit in (False, True):
        torch.manual_seed(0)
        x = torch.randn(3, 8, requires_grad=True)
        w = (torch.ones(8) if unit else torch.randn(8)).requires_grad_()
        g = torch.randn_like(x)
        r = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        ref = torch.autograd.grad(x * r * w, (x, w), g)
        actual = fn(SimpleNamespace(saved_tensors=(x.detach(), w.detach(), r.detach()), zero_centered_gamma=False), g)
        corrected = g * w * r - x * r.pow(3) * (g * x * w).sum(-1, keepdim=True) / x.shape[-1]
        error = (actual[0] - ref[0]).abs().max().item()
        fixed_error = (corrected - ref[0]).abs().max().item()
        assert fixed_error < 2e-6
        assert (actual[1] - ref[1]).abs().max().item() < 2e-6
        assert error < 2e-6 if unit else error > 0.1
        results.append(dict(unit_weight=unit, input_gradient_error=error, corrected_error=fixed_error))
    return results


def bridge(miles):
    path = miles / "miles/backends/megatron_utils/model_provider.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_bridge_runtime_config")
    names = {
        n.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args"
    }
    args = argparse.Namespace(**dict.fromkeys(names, False))
    fields = ("apply_rope_fusion", "bias_activation_fusion", "batch_invariant_mode", "true_on_policy_contract")
    for f in fields:
        setattr(args, f, True)
    args.bias_swiglu_fusion = False
    provider = SimpleNamespace(**dict.fromkeys(fields, False))
    provider.bias_activation_fusion = True
    fn = load_function(path, node.name, {"argparse": argparse})
    fn(provider, args)
    guard = load_function(miles / "miles/backends/megatron_utils/arguments.py", "set_default_megatron_args")
    try:
        guard(SimpleNamespace(true_on_policy_mode=True))
    except NotImplementedError as e:
        rejected = str(e)
    else:
        raise AssertionError("Megatron true-on-policy no longer rejected; reassess audit")
    omitted = [f for f in fields if f != "bias_activation_fusion" and not getattr(provider, f)]
    assert provider.bias_activation_fusion is True, "Explicit no-bias-swiglu-fusion began propagating"
    omitted.append("bias_swiglu_fusion -> bias_activation_fusion")
    assert len(omitted) == 4
    init = (miles / "miles/backends/megatron_utils/initialize.py").read_text()
    assert "enable_batch_invariant_mode" not in init
    return dict(omitted_fields=omitted, guard=rejected, initializer_activation_present=False)


def serving_environment(source, tp, deterministic, inherited):
    cfg = SimpleNamespace(
        enable_prefill_only_deterministic_inference=False,
        rl_on_policy_target=None,
        true_on_policy_contract=None,
        enable_deterministic_inference=deterministic,
        enable_aiter_allreduce_fusion=False,
        model_path="unused",
        attention_backend="fa3",
        tp_size=tp,
    )
    mock_server = SimpleNamespace(RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND=["fa3"])
    ns = dict(
        os=os,
        logger=logging.getLogger("audit"),
        resolving_view=lambda a: a,
        resolved_view=lambda a: a,
        validate_true_on_policy_contract=lambda a: None,
        run_post_process_pass=lambda *a: None,
        declare_resolution=lambda obj, reason, **kw: vars(obj).update(kw),
        parse_connector_type=lambda p: "instance",
        ConnectorType=SimpleNamespace(INSTANCE="instance"),
        get_platform=lambda: SimpleNamespace(is_hip=False),
        envs=SimpleNamespace(
            SGLANG_ENABLE_DETERMINISTIC_INFERENCE=Mock(),
            SGLANG_DETERMINISTIC_NCCL_NCHANNELS=SimpleNamespace(get=lambda: 8),
        ),
        _deterministic_allreduce_fusion_disable=None,
        _deterministic_sampling_backend=None,
        _deterministic_attention_backend=None,
    )
    fn = load_function(source, "handle_deterministic_inference", ns)
    with patch.dict(sys.modules, {"sglang.srt.server_args": mock_server}), patch.dict(
        os.environ, inherited, clear=True
    ):
        fn(cfg)
        return {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}


def nccl(sources, miles):
    source = sources / "sglang_miles_attention_hook.py"
    trainer = {"NCCL_ALGO": "Ring"}
    serving = serving_environment(source, 2, True, trainer)
    assert serving != trainer
    assert serving["NCCL_MIN_NCHANNELS"] == serving["NCCL_MAX_NCHANNELS"] == "8"
    assert serving_environment(source, 1, True, trainer) == trainer
    assert serving_environment(source, 2, False, trainer) == trainer
    assert serving_environment(source, 2, True, serving) == serving
    # Execute Miles' real trainer environment builder with non-offload settings.
    fn = load_function(
        miles / "miles/ray/specs/train.py",
        "compute_trainer_env_vars",
        {"os": os, "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST": []},
    )
    args = SimpleNamespace(train_env_vars={}, dumper_source_patcher_config_train=None, offload_train=False)
    with patch.dict(os.environ, trainer, clear=True):
        worker_env = fn(args, None, fp8_scales="0")
    assert "NCCL_MIN_NCHANNELS" not in worker_env
    return dict(
        trainer_inherited=trainer,
        serving_after_handler=serving,
        aligned_profile_stable=True,
        scope="Actual handler/environment-builder execution with external config helpers mocked; no NCCL collective executed",
    )


def residual(source):
    fn = load_function(source, "forward_native", {"torch": torch}, "RMSNorm")
    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.bfloat16)
    r = torch.randn_like(x)
    obj = SimpleNamespace(
        override_orig_dtype=None,
        hidden_size=128,
        variance_size_override=None,
        variance_epsilon=1e-6,
        weight=torch.ones(128),
        cast_x_before_out_mul=False,
    )
    outs = []
    for flag in (False, True):
        obj.fp32_residual = flag
        outs.append(fn(obj, x.clone(), r.clone())[0])
    # Independently reproduce rounding before normalization versus after.
    refs = []
    for flag in (False, True):
        z = x.float() + r.float() if flag else (x + r).float()
        refs.append((z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype))
    matches = [torch.equal(a, b) for a, b in zip(outs, refs, strict=True)]
    return dict(
        different_elements=int((outs[0] != outs[1]).sum()),
        max_output_difference=(outs[0] - outs[1]).abs().max().item(),
        matches_bf16_then_fp32_references=matches,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.miles))
    results = dict(
        scope="CPU source audit, not distributed or model parity verification",
        bridge=bridge(args.miles),
        rmsnorm_backward=norm_backward(args.sources / "megatron_batch_invariant.py"),
        nccl=nccl(args.sources, args.miles),
        residual_miles_dependency=residual(args.sources / "sglang_miles_layernorm.py"),
        residual_sglang_main=residual(args.sources / "sglang_layernorm.py"),
    )
    results["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.sources.glob("*.py"))
    }
    print(json.dumps(results, indent=2))
    if args.require_clean:
        raise SystemExit("Confirmed defects remain: Bridge flags and Megatron RMSNorm backward")


if __name__ == "__main__":
    main()
