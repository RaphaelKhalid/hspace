"""wor: "Workspace or Reflex?" J-lens tooling (scorer, token families, subspace
ablation, teacher forcing). Importing ``wor`` puts the vendored ``jlens`` on
``sys.path``."""

from wor import _paths  # noqa: F401  (side effect: sys.path for jlens)

__all__ = ["modeling", "scorer", "families", "ablate", "teacher_force"]
