# arXiv submission checklist

Steps to get the project ready for **arXiv** first; then you can adapt for journal submission.

---

## 1. Decide paper format and template

- **Format:** arXiv accepts PDF. Most CS/ML submissions use **LaTeX** (easier for journals later).
- **Templates:** Use a standard class so the same source can go to a journal later.
  - **Option A:** `article` + `biblatex` or `natbib` (simple, journal-friendly).
  - **Option B:** NeurIPS/ICML/ICLR style from [Overleaf](https://www.overleaf.com/gallery/tagged/neurips) or [ctan](https://ctan.org/topics/conference) — good if you might submit to a conference first.
- **Suggestion:** Create a `paper/` folder in the repo with:
  - `main.tex` (or `paper.tex`)
  - `references.bib`
  - `figures/` (for plots)
  - Optional: `paper/supplementary.pdf` or appendix in the same PDF.

---

## 2. Paper structure (sections to write)

| Section | What to include | Where your material lives |
|--------|-----------------|---------------------------|
| **Title & abstract** | Clear title; ~150–250 word abstract (problem, method, main result, impact). | New; use README + WHY_WE_BEAT_FULL_DATA for wording. |
| **1. Introduction** | Motivation (incremental learning, cost of retraining), contribution (update policy, same base model), high-level result (time savings + when we win). | README “Contribution”, docs/WHY_WE_BEAT_FULL_DATA.md. |
| **2. Related work** | Test-time adaptation, continual learning, regime switching / non-stationary time series, incremental forecasting. | Your knowledge + 10–15 key refs. |
| **3. Method** | (a) Problem setup (streaming batches, same base model). (b) Regime = distribution features; matching rule. (c) Policy: match → load checkpoint; no match → retrain and save. (d) Optional: adaptive full vs partial, dynamic threshold. | README “Overview”, forecaster logic, BASELINE_COMPARISON_PROGRESS “Baseline definitions”. |
| **4. Experiments** | Datasets (ETT, 33-dataset mix, multi_regime / covid_shock / recurring). Baselines: always retrain, TTA (same model). Metrics: MAPE, training time, match rate. Tables + 1–2 key figures. | README tables, three_way_results.json, run_publication_benchmark, run_baseline_comparison. |
| **5. Analysis / Why we win** | When we beat full retrain: mixture vs specialist; recurring regimes; numerical/empirical evidence. When we lose: novel shock, wrong match. | docs/WHY_WE_BEAT_FULL_DATA.md, notebooks/why_regime_wins_analysis.ipynb. |
| **6. Conclusion** | Summary, limitations, future work (e.g. more baselines, real-world deployments). | Short; from intro + experiments. |
| **References** | BibTeX for all cited work. | Create `references.bib` and cite in LaTeX. |

---

## 3. Content you already have (to paste or adapt)

- **Problem & contribution:** README “Contribution: Update Policy, Not a New Model”.
- **Method:** README “Overview” (4 steps), “Methodology” (adaptive selection, similarity).
- **Why we beat full data:** Copy/adapt from `docs/WHY_WE_BEAT_FULL_DATA.md` (mixture vs specialist, minimal math).
- **Tables:** README (ETT table, 33-dataset summary, three-way comparison); optionally export from `notebooks/why_regime_wins_analysis.ipynb` (e.g. regime wins by scenario).
- **Reproducibility:** Commands in README (`make benchmark`, `run_baseline_comparison.py`, `run_publication_benchmark.py`).

---

## 4. What to add or polish for arXiv

- [ ] **Full draft** in LaTeX (or Word → PDF) with all sections above.
- [ ] **Abstract** (150–250 words): no formulas; state problem, method, main result (time + accuracy), one sentence on impact.
- [ ] **Figures:** At least 1–2 (e.g. workflow/architecture from README Mermaid → export as image; or plot: time saved vs MAPE by scenario).
- [ ] **Consistent numbers:** Pick one set of runs (e.g. ETT 4 datasets + 33-dataset three-way) and report same numbers in abstract/intro/experiments.
- [ ] **Code & data statement:** In paper or abstract: “Code and data: [GitHub URL]” or “Code available at [URL].” arXiv allows linking to GitHub.
- [ ] **Limitations:** 1 short paragraph (e.g. single base model, synthetic + ETT so far, threshold sensitivity).

---

## 5. arXiv submission steps (when the draft is ready)

1. **Create account:** [arxiv.org](https://arxiv.org) → “Login” → register (free).
2. **Submit:** “Submit” → “New submission.”
3. **Choose category:** e.g. **cs.LG** (Machine Learning) and/or **stat.ML**; optionally **cs.AI** or **eess.SP** (signal processing) if you emphasize forecasting.
4. **Upload:**  
   - **Source:** Upload a single ZIP containing your `.tex` files, `.bib`, and `figures/` (or use “Upload abstract and PDF only” and upload just the PDF).  
   - **PDF:** If you upload source, arXiv will compile it; you can also upload a precompiled PDF.
5. **Abstract:** Paste your abstract (same 150–250 words).
6. **Comments:** Optional “10 pages, 4 figures” or “To appear in …” if you already have a journal target.
7. **Submit for moderation.** You get an arXiv ID (e.g. 2402.xxxxx) after approval (usually 1–2 business days).

---

## 6. Technical: generating the PDF

- **LaTeX:** From `paper/` run `pdflatex main && bibtex main && pdflatex main && pdflatex main` (or use `latexmk -pdf main.tex`). Upload the `.tex`, `.bib`, and figures in a ZIP, or the final PDF.
- **Overleaf:** Create a project, paste your LaTeX, compile, then download “Source” (ZIP) and/or PDF for arXiv.
- **Check:** PDF must be readable, all references resolved, no placeholder “[?]”.

---

## 7. After arXiv (for journal later)

- Cite the arXiv version in the journal submission (e.g. “Preliminary version on arXiv: 24xx.xxxxx”).
- Journal may want an extended version: more related work, extra experiments, or appendix. Your same LaTeX can be extended.
- Some journals accept “arXiv-style” LaTeX; others require their template (you can adapt from this draft).

---

## 8. Minimal “paper/” layout (suggestion)

```
paper/
  main.tex          # Title, abstract, sections 1–6
  references.bib   # BibTeX
  figures/         # workflow.pdf, results_plot.pdf, etc.
  (optional) appendix.tex
```

Start with `main.tex` and `references.bib`; add figures as you draft. Once this is in place, you can iterate on wording and then submit to arXiv.

---

**Summary:** Write the full paper (sections 1–6) in LaTeX, reusing README and `docs/WHY_WE_BEAT_FULL_DATA.md`. Add abstract, 1–2 figures, and a code/data statement. Then create an arXiv account, choose cs.LG (and/or related), upload source or PDF, paste abstract, and submit.
