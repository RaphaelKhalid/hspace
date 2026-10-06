"""bf16-stored / fp32-computed layers give the same exact HVP as the fully-fp32 model (0.8B)."""
import torch

from bf16w import patch_bf16_compute
from common import Ctx, load_model, patch_deltanet, run_from, wikitext_contexts


def _hvp(m, c, u, v):
    h = c.h.clone().requires_grad_(True)
    with torch.enable_grad():
        out, _, _ = run_from(m, c, h, c.l + 1)
        F = (out[:, c.valid] @ u).sum()
        (g,) = torch.autograd.grad(F, h, create_graph=True)
        (hv,) = torch.autograd.grad((g[:, 40] * v).sum(), h)
    return g.detach(), hv.detach()


def test_bf16_compute_matches_fp32(model):
    m32 = model                      # conftest: fp32 layers
    patch_deltanet(m32)
    ids = wikitext_contexts(m32.tok, 1, T=64)[0]
    c32 = Ctx(m32, ids, 12)
    torch.manual_seed(0)
    u = torch.randn(m32.d, device="cuda")
    v = torch.randn(m32.d, device="cuda")
    g32, hv32 = _hvp(m32, c32, u, v)
    # same model, layers back to bf16 storage, then bf16-weight / fp32-compute patch
    for blk in m32.layers:
        blk.to(torch.bfloat16)
    n = patch_bf16_compute(m32.layers)
    assert n > 0
    alloc_before = torch.cuda.memory_allocated()
    gb, hvb = _hvp(m32, c32, u, v)
    eg = ((gb - g32).norm() / g32.norm()).item()
    eh = ((hvb - hv32).norm() / hv32.norm()).item()
    print(f"patched {n} linears; grad rel err {eg:.2e}, HVP rel err {eh:.2e}")
    assert eg < 1e-5 and eh < 1e-5
