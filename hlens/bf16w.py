"""bf16-stored, fp32-computed linear layers that are differentiable to any order (Oct 5 2026).

Qwen checkpoints are natively bf16, so upcasting a weight to fp32 is EXACT. Storing the model in
fp32 doubles memory (27B: 108 GB), and the naive alternative F.linear(x, W.float()) makes autograd
save the fp32 copy of every weight (the same 108 GB). Here the autograd Functions save the bf16
weight and recast it on use, and each Function's backward is the other Function, so double backward
(HVPs) and higher work. Memory = bf16 weights + one transient fp32 copy of the current matrix.

    patch_bf16_compute(model_layers)   # in place; layers then run in fp32 activations
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MODE = "fp32"   # "bf16": forward-only stages run bf16 matmuls (fast); never use for HVPs


class _Lin(torch.autograd.Function):
    """y = x W^T with W bf16 (constant), x fp32."""

    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(W)
        if MODE == "bf16":
            return F.linear(x.to(torch.bfloat16), W).float()
        return F.linear(x, W.float())

    @staticmethod
    def backward(ctx, gy):
        (W,) = ctx.saved_tensors
        return _LinT.apply(gy, W), None


class _LinT(torch.autograd.Function):
    """gx = gy W (the transpose map), itself differentiable via _Lin."""

    @staticmethod
    def forward(ctx, gy, W):
        ctx.save_for_backward(W)
        return gy @ W.float()

    @staticmethod
    def backward(ctx, gx):
        (W,) = ctx.saved_tensors
        return _Lin.apply(gx, W), None


class BFLinear(torch.nn.Module):
    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.weight = lin.weight
        self.weight.requires_grad_(False)
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().float(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        y = _Lin.apply(x.float(), self.weight)
        return y if self.bias is None else y + self.bias


def patch_bf16_compute(layers) -> int:
    """Replace every nn.Linear inside `layers` by BFLinear (weights stay bf16); cast all other
    parameters and buffers (norm gains, conv kernels, A_log, dt_bias) to fp32. Returns #linears."""
    n = 0
    for blk in layers:
        for name, mod in list(blk.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(mod, cname, BFLinear(child))
                    n += 1
        for mod in blk.modules():
            if isinstance(mod, BFLinear):
                continue
            for pn, p in list(mod.named_parameters(recurse=False)):
                if p.dtype != torch.float32:
                    setattr(mod, pn, torch.nn.Parameter(p.detach().float(), requires_grad=False))
            for bn, b in list(mod.named_buffers(recurse=False)):
                if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                    mod._buffers[bn] = b.float()
    return n
