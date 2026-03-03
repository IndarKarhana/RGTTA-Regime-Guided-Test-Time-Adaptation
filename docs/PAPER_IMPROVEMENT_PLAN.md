# Paper Improvement Plan — Comprehensive Roadmap to Publication

**Single source of truth** for making RG-TTA publishable. Read at session start.

**Paper file:** `paper/main.tex` | **References:** `paper/references.bib`
**Target venues (in priority order):**
1. **arXiv preprint** — post after Phase A is complete
2. **Neurocomputing** / **Knowledge-Based Systems** (IF ~8-9) — strong fit for empirical ML
3. **IEEE TNNLS** — if we complete all phases including Phase C

---

## Self-Review Summary (Honest Assessment)

### What's Strong
- Clear, useful framing ("when-to-adapt and how-aggressively")
- 672 experiments with proper controls (same model, seed, data; only strategy differs)
- Model-agnostic design tested across 4 architectures in 3 families
- Composability demonstrated, not just claimed (RG improves TTA 67%, EWC 75%, DynaTTA 62%)
- 9 honest limitations including where it fails (Weather, synth_volatility)

### Critical Weaknesses to Address
| # | Weakness | Impact | Fix Phase |
|---|----------|--------|-----------|
| 1 | **No retrain baseline** — every reviewer will ask | Blocking for journals | Phase C |
| 2 | **Zero figures** — extremely unusual | High visual impact | Phase A |
| 3 | **No statistical significance** — p-values mentioned but not computed | Required for any venue | Phase A |
| 4 | **DynaTTA evaluation fairness** — protocol favors our method | Serious concern | Phase B |
| 5 | **8/14 datasets are synthetic** — inflates win counts | Moderate | Phase A |
| 6 | **Thin ablations** — only 2 datasets, 1 model for metric ablation | Moderate | Phase B |
| 7 | **3 seeds is borderline** — top venues expect 5 | Low (acceptable for mid-tier) | Phase D |
| 8 | **Proposition 1 is trivial** — basic bias-variance decomposition | Low | Phase B |
| 9 | **Abstract overloaded** with specific numbers | Minor polish | Phase A |

---

## Phase A: CRITICAL — Required for Any Submission (1-2 days)

**Goal:** Make the paper submission-ready for arXiv and second-tier journals.

### A1. Figures (5 figures minimum)

| # | Figure | Description | Data Source | Priority |
|---|--------|-------------|-------------|----------|
| A1.1 | **System diagram** (Figure 1) | Architecture: batch arrives → feature extract → similarity → {checkpoint reuse, LR modulation, early stopping} → adapt → predict → store. v2 continuous flow, not 3-tier. | `generate_all_figures.py` fig_workflow() | ✅ Done |
| A1.2 | **Critical difference diagram** | Demšar-style CD diagram (6 policies ranked across 224 experiments). Shows statistical significance of ranking differences. | `unified_results_full.csv` + `scipy.stats` | ✅ Done |
| A1.3 | **Pair-wise improvement waterfall** | Grouped bar chart with p-values. | Table 3 data | ✅ Done |
| A1.4 | **Per-dataset win-rate heatmap** | 14 datasets × 6 policies heatmap (win count per cell). Shows Weather/synth_volatility weakness. | Per-dataset analysis | ✅ Done |
| A1.5 | **Adaptation behavior example** (3 panels) | MSE + similarity + LR over batches for ETTh2/GRU-Small/H=96. | `unified_batch_detail.csv` | ✅ Done |
| A1.6 | **MSE vs Time scatter** | Shows RG-TTA is Pareto-optimal (lower MSE + faster). | Tables 4 + 7 data | ✅ Done |
| A1.7 | **Horizon analysis** | MSE scaling with forecast horizon (all + real-world). | Results CSV | ✅ Done |
| A1.8 | **Model comparison** | Real-world MSE by model architecture. | Results CSV | ✅ Done |
| A1.9 | **Rank distribution** | Box plots of ranks across 224 experiments. | Results CSV | ✅ Done |

### A2. Statistical Significance

| # | Task | Description | Status |
|---|------|-------------|--------|
| A2.1 | **Wilcoxon signed-rank tests** | All 3 pairs significant: TTA→RG-TTA (p=1.0e-5***), EWC→RG-EWC (p=2.4e-11***), DynaTTA→RG-DynaTTA (p=6.8e-3**). Bonferroni-corrected. Added to Table 3. | ✅ Done |
| A2.2 | **Friedman test + Nemenyi post-hoc** | Friedman χ²=301.95, p=3.81e-63. CD=0.50. RG-TTA (2.46) and RG-EWC (2.51) statistically tied, both sig. better than all baselines. | ✅ Done |
| A2.3 | **Standard deviations in key tables** | Computed and stored in `paper/statistical_tests.txt`. Available for paper tables. | ✅ Done |
| A2.4 | **Separate real-world vs synthetic win rates** | Real-world: 67/96 (69.8%), Synthetic: 89/128 (69.5%). Nearly identical — demolishes "inflated by synthetic" critique. Added to paper. | ✅ Done |

### A3. Paper Text Polish

| # | Task | Description | Status |
|---|------|-------------|--------|
| A3.1 | **Trim abstract** | Remove some specific numbers (keep 69.6%, 672 experiments). Move details to results. | 🟡 Minor polish left |
| A3.2 | **Add "Real-World vs Synthetic" analysis paragraph** | Added with explicit numbers (69.8% vs 69.5%). | ✅ Done |
| A3.3 | **Strengthen streaming justification** | Production deployment argument added. | ✅ Done |
| A3.4 | **Fix DynaTTA reimplementation rationale** | Explains library unavailability. | ✅ Done |
| A3.5 | **LaTeX compilation test** | No pdflatex installed locally. Balance check passed (59 begin/end pairs). All refs resolve. All citations found. | ⚠️ Needs compile |
| A3.6 | **Audit references.bib** | 26 citations, all found. 13 unused entries (can be cleaned). | ✅ Done |

---

## Phase B: IMPORTANT — Required for Peer-Reviewed Journals (2-3 days)

**Goal:** Address reviewers' likely objections before they raise them.

### B1. Deeper Ablations

| # | Task | Description | Datasets | Status |
|---|------|-------------|----------|--------|
| B1.1 | **Loss gate sweep** | Sweep `ckpt_loss_gate` ∈ {0.50, 0.60, 0.70, 0.80, 0.90} | ETTh1, ETTm1, synth_recurring, synth_shock | ☐ Script ready (`paper/run_ablation_sweeps.py`) |
| B1.2 | **LR sim_scale sweep** | Sweep `lr_sim_scale` ∈ {0.0, 0.33, 0.67, 1.0, 1.5} | Same 4 datasets | ☐ Script ready |
| B1.3 | **Memory capacity sweep** | Sweep memory cap ∈ {1, 3, 5, 10, 20} | Same 4 datasets | ☐ Script ready |
| B1.4 | **Expand similarity metric ablation** | Current: 2 datasets, 1 model. Need: 4+ datasets, 2+ models. | ETTh1, ETTm1, synth_recurring, Weather | ☐ Needs custom script |
| B1.5 | **Early stopping ablation** | Compare fixed K=20 vs loss-driven early stopping (both within RG-TTA) | Same 4 datasets | ☐ Script ready |

### B2. Protocol Fairness

| # | Task | Description | Status |
|---|------|-------------|--------|
| B2.1 | **DynaTTA with streaming-tuned η** | Run DynaTTA with η=0.7 (our streaming-mode ablation from Run #65 showed it helps). Report as "DynaTTA-S" in an ablation table. Shows we tried to be fair. | ☐ Not started |
| B2.2 | **Acknowledge protocol choice explicitly** | "Protocol Selection and Fairness" subsection added to Ablation Studies section. Explains 3 steps taken for fairness. | ✅ Done |

### B3. Strengthen Theory

| # | Task | Description | Status |
|---|------|-------------|--------|
| B3.1 | **Improve Proposition 1** | Added Proposition 2 (convergence advantage): checkpoint reuse reduces starting distance, converges in O(log(1/sim)) fewer steps. | ✅ Done |
| B3.2 | **Add computational complexity analysis** | Added full per-batch complexity breakdown: O(M·n log n + M·|θ|) overhead, dominated by gradient steps. | ✅ Done |

---

## Phase C: RETRAIN BASELINE — The Biggest Gap (3-5 days compute)

**Goal:** Address the #1 reviewer objection: "Why not just retrain?"

| # | Task | Description | Status |
|---|------|-------------|--------|
| C1 | **Run retrain on VM** | Currently running. 4 models × 14 datasets × 4 horizons × 3 seeds = 672 experiments. ~40× slower than TTA. | 🟢 Running on VM |
| C2 | **Add retrain to all paper tables** | When results arrive: add a "Retrain" row to Tables 3-8. | ☐ Blocked on C1 |
| C3 | **Add retrain to statistical tests** | Include retrain in Friedman test and CD diagram. | ☐ Blocked on C1 |
| C4 | **Add timing comparison** | Retrain is ~40× slower. This is RG-TTA's strongest argument: comparable accuracy at 40× lower cost. | ☐ Blocked on C1 |
| C5 | **Update conclusion and abstract** | If retrain ≈ RG-TTA accuracy: "RG-TTA matches retrain at 40× lower cost." If retrain > RG-TTA: "Retrain trades 40× compute for X% accuracy." | ☐ Blocked on C1 |

---

## Phase D: NICE-TO-HAVE — Would Strengthen for Top Venues (1 week)

| # | Task | Description | Priority |
|---|------|-------------|----------|
| D1 | **5 seeds instead of 3** | ~30 hours additional compute. Strengthens statistical claims. | Low |
| D2 | **Additional real-world datasets** | Electricity, Traffic, ILI (used by DynaTTA). Would strengthen real-world claim. | Medium |
| D3 | **Adapter module experiment** | For PatchTST/iTransformer: unfreeze adapter layers instead of full backbone freeze. Could improve attention-model results. | Low |
| D4 | **Partial unfreezing ablation** | Compare freeze-all vs freeze-backbone-only vs unfreeze-all across architectures. | Low |
| D5 | **Sensitivity analysis for all 10 hyperparameters** | Comprehensive grid over all manually-tuned params. For IEEE TNNLS. | Medium |

---

## Execution Order

```
Phase A (1-2 days)  ──→  arXiv submission
         │
         ▼
Phase B (2-3 days)  ──→  Neurocomputing / KBS submission
         │
Phase C (parallel)  ──→  Add retrain when VM finishes
         │
         ▼
Phase D (optional)  ──→  IEEE TNNLS if desired
```

**Immediate next actions:**
1. Generate figures (A1.1-A1.6) — `paper/generate_all_figures.py`
2. Run Wilcoxon tests (A2.1-A2.2) — quick Python script
3. Add std to tables (A2.3) — update `main.tex`
4. LaTeX compile test (A3.5)
5. Post to arXiv

---

## Key Data Sources

All results from Run #72:
- **Full CSV:** `benchmarks/results/unified_v2_8pol/unified_results_full.csv` (5376 rows, 8 policies)
- **6-policy subset:** Filter to `policy in [tta, ewc, dynatta, rgtta, rgtta_ewc, rgtta_dynatta]` (4032 rows)
- **Batch detail:** `benchmarks/results/unified_v2_8pol/unified_batch_detail.csv`
- **Report:** `benchmarks/results/unified_v2_8pol/unified_report.md`

Retrain (when complete):
- **Results dir:** `benchmarks/results/unified_retrain_v5/` (VM: `rgtta-benchmark`)

---

*Last updated: 2026-03-02 — Comprehensive plan based on self-review. DynaTTA reimpl rationale + streaming argument fixed in paper.*
