"""Check the candidate backward patch on CPU; does not execute CUDA forward."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from audit import load_function

source = Path(sys.argv[1]).read_text()
old = "dx = go_fp32 * (w_fp32 * r) - (w_fp32 * r3) * (s * x_fp32) / D"
new = "dx = go_fp32 * (w_fp32 * r) - r3 * (s * x_fp32) / D"
assert source.count(old) == 1, "Source changed; reassess the candidate patch"
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "patched.py"
    path.write_text(source.replace(old, new))
    backward = load_function(path, "backward", {"torch": torch}, "BatchInvariantRMSNormFn")
results = []
for shape in [(3, 8), (2, 3, 16), (4, 128)]:
    for centered in [False, True]:
        for unit in [False, True]:
            torch.manual_seed(7)
            x = torch.randn(shape, requires_grad=True)
            w = (torch.ones(shape[-1]) if unit else torch.randn(shape[-1])).requires_grad_()
            g = torch.randn_like(x)
            r = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            reference = torch.autograd.grad(x * r * (w + 1 if centered else w), (x, w), g)
            ctx = SimpleNamespace(saved_tensors=(x.detach(), w.detach(), r.detach()), zero_centered_gamma=centered)
            actual = backward(ctx, g)
            for a, b in zip(actual[:2], reference, strict=True):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            results.append(
                {
                    "shape": shape,
                    "zero_centered": centered,
                    "unit": unit,
                    "dx_max_abs_error": (actual[0] - reference[0]).abs().max().item(),
                }
            )
print(json.dumps({"passed": len(results), "cases": results}, indent=2))
