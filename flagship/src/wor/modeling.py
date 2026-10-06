"""Model + lens loading for Qwen3.5 (and other HF decoders) on top of vendored jlens.

Design notes (verified on the laptop with Qwen/Qwen3.5-0.8B, transformers 5.5.4):
- ``AutoModelForCausalLM`` maps the multimodal ``qwen3_5`` config to the text-only
  ``Qwen3_5ForCausalLM`` (vision weights are skipped on load). jlens's own layout
  auto-detection then finds ``Layout(path="model")`` - no custom adapter needed.
  If you instead load ``Qwen3_5ForConditionalGeneration`` (e.g. through PEFT on a
  VLM checkpoint), the ``model.language_model`` layout is also in jlens's table.
- Qwen3.5's final norm is ``Qwen3_5RMSNorm``: ``x / rms(x) * (1 + w)`` computed in
  fp32 (weight initialised at zero). :func:`norm_gain` recovers the effective gain
  empirically so it is correct for both ``w`` and ``1 + w`` conventions.
- The Qwen3.5 tokenizer has no BOS token (``bos_token_id is None``), so jlens's
  ``force_bos`` is a no-op either way; we pass ``force_bos=False`` and feed exact
  chat-template ids (which start with ``<|im_start|>``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from wor import _paths  # noqa: F401

import jlens  # noqa: E402  (vendored)

#: Known pre-fitted neuronpedia lenses (repo, revision, path in repo). The 0.8B
#: file is byte-identical on ``main`` and ``qwen-n1000`` (same LFS sha256
#: aa26b68e...). ``qwen3.5-2b`` exists only on ``main`` as of 2026-09-28.
LENS_REGISTRY: dict[str, tuple[str, str, str]] = {
    "Qwen/Qwen3.5-0.8B": (
        "neuronpedia/jacobian-lens",
        "main",
        "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    ),
    "Qwen/Qwen3.5-2B": (
        "neuronpedia/jacobian-lens",
        "main",
        "qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt",
    ),
    # Pinned to the qwen-n1000 branch commit read on 2026-09-28 (integrator). The file is
    # LFS sha256 1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e, 406,332,644 B,
    # identical on main. main's CREDIT.md says the _n1000.pt file was trained by @mntss
    # (Anthropic); the directory's config.yaml (prompts_fitted: 417) most likely describes the
    # sibling Neuronpedia fit Qwen3.5-4B_jacobian_lens.pt instead. NOT yet downloaded or run.
    "Qwen/Qwen3.5-4B": (
        "neuronpedia/jacobian-lens",
        "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a",
        "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    ),
}


def linear_attention_status() -> dict[str, Any]:
    """Whether Qwen3.5's Gated-DeltaNet layers use the fast (fla + causal-conv1d)
    kernels or the slow pure-torch fallback."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": repr(exc)}
    return {
        "is_fast_path_available": bool(getattr(q, "is_fast_path_available", False)),
        "fla_chunk_gated_delta_rule": getattr(q, "chunk_gated_delta_rule", None)
        is not None,
        "causal_conv1d_fn": getattr(q, "causal_conv1d_fn", None) is not None,
    }


def load_model(
    name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    revision: str | None = None,
) -> tuple[Any, Any]:
    """Load ``(hf_model, tokenizer)`` straight onto ``device`` (no CPU staging
    copy: ``device_map`` + ``low_cpu_mem_usage``)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    hf = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    hf.eval()
    return hf, tok


def wrap(hf_model: Any, tokenizer: Any, layout: jlens.Layout | None = None):
    """jlens ``HFLensModel`` over an already-loaded model (sets requires_grad
    False on all params; does not copy). ``force_bos=False``: we always pass
    exact ids; see module docstring.

    A PEFT-wrapped model (``PeftModel``, e.g. from Montoya's
    ``apply_peft_adapter_stack``) is unwrapped with ``get_base_model()``: the
    LoRA layers stay injected inside the blocks, but jlens's layout search
    needs the plain ``*ForCausalLM`` (``PeftModel.model`` resolves to the
    CausalLM, which has no ``layers`` attribute, so auto-detection fails -
    verified on the laptop: raw ``jlens.from_hf(PeftModel)`` raises ValueError;
    this wrapper works, and a zero-init LoRA gives a bit-identical readout).
    Not yet tested with the real 4-adapter organism stack."""
    try:
        from peft import PeftModel

        if isinstance(hf_model, PeftModel):
            hf_model = hf_model.get_base_model()
    except ImportError:
        pass
    return jlens.from_hf(hf_model, tokenizer, layout=layout, force_bos=False)


def lens_file(model_name: str) -> str:
    """Download (if needed) and return the local path of the registered lens.
    Only the single ``.pt`` file is fetched into the default HF cache."""
    from huggingface_hub import hf_hub_download

    repo, rev, path = LENS_REGISTRY[model_name]
    return hf_hub_download(repo, path, revision=rev)


def load_lens(model_name_or_path: str) -> jlens.JacobianLens:
    if os.path.isfile(model_name_or_path):
        return jlens.JacobianLens.load(model_name_or_path)
    return jlens.JacobianLens.load(lens_file(model_name_or_path))


@torch.no_grad()
def norm_gain(norm: torch.nn.Module, d_model: int) -> tuple[torch.Tensor, str]:
    """Effective per-dimension gain ``g`` of an RMSNorm-style final norm, so that
    ``norm(x) = g * x / rms(x)``. Determined by probing with ``x = 1`` (rms 1) and
    cross-checked against the two conventions ``w`` (Llama) and ``1 + w``
    (Qwen3.5 / Gemma). Raises for norms that are not pure RMS scalings (e.g.
    LayerNorm with mean subtraction / bias).

    Returns ``(gain [d_model] fp32, convention)`` with convention in
    ``{"1+w", "w", "probed"}``."""
    weight = getattr(norm, "weight", None)
    if weight is None:
        raise ValueError(f"{type(norm).__name__} has no weight")
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
    probe = torch.ones(1, d_model, device=weight.device, dtype=torch.float32)
    measured = norm(probe).float()[0] * math.sqrt(1.0 + eps)
    w = weight.detach().float()
    for candidate, label in ((1.0 + w, "1+w"), (w, "w")):
        if torch.allclose(measured, candidate, rtol=1e-2, atol=1e-2):
            return candidate.clone(), label
    # A norm that is an RMS scaling but with an unknown parametrisation: trust
    # the probe (still exact for RMS norms), but a non-RMS norm lands here too.
    probe2 = 2.0 * probe
    if not torch.allclose(norm(probe2).float()[0] * math.sqrt(1.0 + eps / 4), measured, rtol=1e-2, atol=1e-2):
        raise ValueError(f"{type(norm).__name__} is not scale-invariant; not an RMSNorm")
    return measured.clone(), "probed"


@dataclass
class Handles:
    """Everything the other modules need, bundled."""

    hf: Any
    tok: Any
    lens_model: Any  # jlens.HFLensModel
    lens: jlens.JacobianLens | None
    name: str

    @property
    def n_layers(self) -> int:
        return self.lens_model.n_layers

    @property
    def d_model(self) -> int:
        return self.lens_model.d_model

    @property
    def unembedding(self) -> torch.Tensor:
        """``W_U`` as stored: ``[vocab, d_model]`` (tied to embeddings for 0.8B)."""
        return self.lens_model._lm_head.weight

    @property
    def final_norm(self) -> torch.nn.Module:
        return self.lens_model._final_norm

    def final_norm_gain(self) -> torch.Tensor:
        return norm_gain(self.final_norm, self.d_model)[0]


def load_all(name: str, *, with_lens: bool = True, **kw) -> Handles:
    hf, tok = load_model(name, **kw)
    lm = wrap(hf, tok)
    lens = load_lens(name) if with_lens else None
    return Handles(hf=hf, tok=tok, lens_model=lm, lens=lens, name=name)


def free(handles: Handles | None = None) -> None:
    """Drop references and release CUDA memory."""
    import gc

    if handles is not None:
        handles.hf = handles.lens_model = handles.lens = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
