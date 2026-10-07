# H-space explainer

Jacobians and Hessians from scratch, in sixteen levels: from rates and units to the H-lens of Qwen3.6-27B.

Static site. Plain HTML, no build step at serve time. Open `index.html` through any static server:

```
python -m http.server 8765 --directory explainer
```

## Layout

- `index.html`, `glossary.html`, `00.html` ... `15.html`: the pages.
- `assets/site.js`: top bar, prev/next, KaTeX rendering, glossary hover pop-ups (switchable), exercises.
- `assets/sim.js`: tiny canvas helpers used by the figures.
- `assets/glossary/*.json`: glossary sources. `tools/build_glossary.py` merges them into `assets/glossary.js`.
- `assets/hspace_numbers.js`: every run number shown on Level 15. Generated from `../results/*.json` by `tools/extract_numbers.py`.
- `assets/katex/`: KaTeX 0.16.11, bundled so the site needs no external hosts.
- `tools/check_page.js`: static checks (KaTeX renders, inline JS parses, page structure, no em dashes).
- `AUTHORING.md`: the rules every page follows.

## Rebuild generated files

```
cd explainer
python tools/build_glossary.py
python tools/extract_numbers.py
node tools/check_page.js
```

## Deploy

`vercel.json` pins the `@vercel/static` builder so the repo's Python files are never auto-detected. Deploy from this folder with `vercel deploy --prod`.
