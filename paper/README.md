# Paper: RG-TTA — Regime-Guided Meta-Control for Test-Time Adaptation in Streaming Time Series

**Repository:** <https://github.com/IndarKarhana/RGTTA-Regime-Guided-Test-Time-Adaptation>

## Files

| File | Description |
|------|-------------|
| `main.tex` | Full paper (~12 pages): abstract, introduction, related work (31 refs), method (Algorithm 1 + 2 theorems + 1 corollary + 4 propositions), experiments (6 policies × 4 models × 14 datasets × 4 horizons × 3 seeds = 672 experiments), ablation studies, analysis, limitations, conclusion. 10 figures. |
| `references.bib` | 31 BibTeX entries — all cited, all verified. |
| `generate_all_figures.py` | Generates all 10 paper figures (PDF + PNG) + statistical tests + LaTeX snippets from `benchmarks/results/unified_v2_8pol/`. |
| `figures/` | Output directory for generated figures. |

## How to compile

```bash
# Generate figures first
cd paper && python generate_all_figures.py

# Then compile LaTeX
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or use **Overleaf**: upload `main.tex`, `references.bib`, and `figures/*.pdf`.

## Before arXiv submission

1. Add institutional affiliation in `\author{...}`.
2. Verify all 31 bib entries have volume/pages where applicable.
3. Final proof-read of figure captions and cross-references.
