# Provenance of vendored `jlens`

- Upstream: https://github.com/anthropics/jacobian-lens
- Commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e` ("Initial release", Mateusz Piotrowski, 2026-07-02)
- Licence: Apache-2.0 (copied verbatim as `LICENSE` in this directory). Upstream `pyproject.toml` declares `license = "Apache-2.0"`; source files carry `Copyright 2026 Anthropic PBC`, `SPDX-License-Identifier: Apache-2.0`.
- What was copied: the `jlens/` package directory (all `.py` files plus `data/`), unmodified. Not copied: `tests/`, `walkthrough.ipynb`, `assets/`, top-level `data/`, `uv.lock`, `pyproject.toml`.
- Vendored on 2026-09-28 from a local clone (no network fetch at vendoring time).
- Modifications: none. Any Qwen3.5-specific adaptation lives in `flagship/src/wor/`, not here.
- How it is imported: not pip-installed. `wor._paths` (imported by `wor/__init__.py`) prepends `flagship/third_party` to `sys.path`, so `import jlens` resolves to this directory.
