# Research Plan — RGTTA: Gaps Analysis & Path to Submission

**Created:** 2026-02-18
**Last updated:** 2026-02-18
**Purpose:** Single document tracking what we HAVE vs what we NEED for a credible research submission.

---

## 1. What We Have

### 1.1 Method
- [x] 3-tier regime-guided TTA with distributional similarity (RGTTA)
- [x] 3-method similarity ensemble (KS + Wasserstein + feature distance)
- [x] RGTTA+EWC variant with tier-modulated regularisation
- [x] RGTTA+DynaTTA hybrid variant
- [x] 4 model architectures (GRU-Small, LSTM, GRU-Large, Transformer)
- [x] Memory module with checkpoint storage + retrieval

### 1.2 Baselines
- [x] Retrain (full retraining on accumulated data)
- [x] TTA (fixed 20-step adaptation)
- [x] EWC (Elastic Weight Consolidation)
- [x] DynaTTA (dynamic LR via shift metrics — Grover & Etemad, ICML 2025)

### 1.3 Infrastructure
- [x] Unified benchmark runner with fairness guarantees (same seed, data, model)
- [x] 14 datasets (6 real-world + 8 synthetic)
- [x] 6 metrics (MSE, MAE, wMAPE, sMAPE, direction accuracy, wall-clock time)
- [x] Results JSON format for systematic analysis

### 1.4 Results
- [x] v3 benchmark completed (older policy version) — used for internal diagnostics only, not for publication
- [ ] New definitive benchmark pending — will run after smoke test review with latest policy code

### 1.5 Paper Draft
- [x] ~631 lines LaTeX, 4 figures, 6 tables (main) + 4 (appendix), 32 references
- [x] Honest framing — conditional recommendation, not overclaim
- [x] Appendices: per-dataset results, hyperparameters, dataset descriptions, reproducibility

### 1.6 Documentation
- [x] Comprehensive README.md (614 lines)
- [x] Design decisions log (docs/RGTTA_DESIGN_DECISIONS.md)
- [x] Copilot instructions (.github/copilot-instructions.md)
- [x] Paper improvement plan (docs/PAPER_IMPROVEMENT_PLAN.md)

---

## 2. Critical Gaps (Must Fix Before Submission)

### GAP 1: Statistical Significance — Only 1 Seed ⚠️ HIGH PRIORITY

**Problem:** All current results use seed=42 only. Reviewers will immediately reject without variance estimates. A single seed result tells us nothing about whether DynaTTA's 10 wins vs RGTTA+EWC's 7 wins is statistically meaningful.

**Required:**
- Re-run full benchmark with at least 3 seeds (preferably 5)
- Compute mean ± std for all metrics
- Wilcoxon signed-rank test (pairwise policy comparison)
- Demšar critical difference diagram (multi-policy comparison)

**Effort:** ~30-50 hours of compute for 5 seeds × 48 experiments. Can parallelise across models.

**Priority:** BLOCKING — no submission without this.

---

### GAP 2: Missing Transformer Results (12 experiments) ⚠️ MEDIUM PRIORITY

**Problem:** Transformer model experiments are still running in the v3 benchmark. Need these for the model-agnostic claim.

**Required:**
- Wait for current benchmark to complete (PID 75474)
- Verify all 12 Transformer experiments produced valid results
- Add to results analysis

**Effort:** In progress, just needs to finish.

**Priority:** BLOCKING — currently running, should complete within hours.

---

### GAP 3: Synthetic Dataset Experiments Not Run ⚠️ HIGH PRIORITY

**Problem:** 8 synthetic datasets designed specifically to validate regime detection — none have been benchmarked yet. These are the most compelling evidence for RGTTA's thesis:
- `synth_recurring`: Should demonstrate checkpoint reuse advantage (RGTTA's core value proposition)
- `synth_shock_recovery`: Should show aggressive adaptation + EWC benefit
- `synth_stable`: Control — all methods should perform similarly
- Others test edge cases of the similarity metric

**Required:**
- Run all 7 policies × 4 models × 8 synthetic datasets × 3 horizons × N seeds
- Analyse regime tier assignments vs ground-truth regime labels
- Correlation between similarity scores and known regime transitions

**Effort:** ~20-40 hours compute (shorter series = faster). Plus analysis.

**Priority:** HIGH — this is the strongest evidence for the paper's thesis.

---

### GAP 4: Weather & Exchange Not Benchmarked ⚠️ MEDIUM PRIORITY

**Problem:** These 2 real-world datasets are in the codebase but haven't been run in the v3 benchmark (current run is ETT-only).

**Required:**
- Run all 7 policies × 4 models × 2 datasets × 3 horizons × N seeds
- Weather has 21 meteorological indicators — tests multivariate handling
- Exchange has only 7,588 points — tests data-scarce scenarios

**Effort:** ~10-20 hours compute.

**Priority:** MEDIUM — adds diversity but ETT is the standard benchmark.

---

### GAP 5: RGTTA+DynaTTA Not in Current Benchmark ⚠️ MEDIUM PRIORITY

**Problem:** Policy 7 (RGTTA+DynaTTA hybrid) is implemented but was NOT included in the v3 benchmark run. We have only 6 of 7 policies.

**Required:**
- Add rgtta_dynatta to the unified benchmark runner
- Re-run benchmark with all 7 policies
- Or run it separately on same experiments for comparison

**Effort:** ~8-15 hours compute.

**Priority:** MEDIUM — nice to have but RGTTA+EWC may be sufficient for the paper.

---

### GAP 6: Paper Tables Need v3 Results ⚠️ HIGH PRIORITY

**Problem:** Paper tables still have old v1 results (before RGTTA). Need complete rewrite of:
- Abstract numbers
- Results section tables
- Win count analysis
- Speed-accuracy discussion

**Required:**
- Wait for benchmark completion + multi-seed runs
- Generate new tables and figures from unified_results.json
- Rewrite results section with honest v3 numbers

**Effort:** ~4-8 hours text editing.

**Priority:** HIGH — can only start after GAPs 1-2 are resolved.

---

### GAP 7: Ablation Studies Not Run ⚠️ MEDIUM PRIORITY

**Problem:** No systematic ablation to justify design choices:
- Similarity ensemble weights (why 0.4/0.4/0.2?)
- Threshold values (why τ_high=0.80, τ_low=0.55?)
- Steps per tier (why 5/20/50?)
- EWC scale per tier (why 0.5/1.0/1.5?)

**Required:**
- τ_high sweep: {0.70, 0.75, 0.80, 0.85, 0.90} on representative dataset(s)
- Similarity weight sweep: vary KS/Wass/Feat ratios
- Steps sweep: test {3,5,10}/{15,20,30}/{30,50,80} per tier
- At minimum: τ_high ablation (most impactful design choice)

**Effort:** ~10-15 hours compute for τ_high sweep alone.

**Priority:** MEDIUM — reviewers will ask about this but a single ablation (τ_high) may suffice.

---

### GAP 8: No Statistical Tests in Paper ⚠️ HIGH PRIORITY

**Problem:** Even with multiple seeds, need formal statistical comparison:
- Pairwise: Wilcoxon signed-rank test between each policy pair
- Multi-way: Friedman test + Nemenyi post-hoc (Demšar 2006)
- Critical difference diagram showing which policies are statistically indistinguishable

**Required:**
- Implement statistical test pipeline (scipy.stats + scikit-posthocs)
- Generate critical difference diagram
- Report p-values in paper

**Effort:** ~2-3 hours code + analysis (once multi-seed data is available).

**Priority:** HIGH — required for any venue. Depends on GAP 1 (multi-seed).

---

## 3. Nice-to-Have Improvements (Not Blocking)

### NICE 1: Larger Model Validation

Test on a production-scale model (e.g., PatchTST, Informer) to validate that RGTTA works beyond small models. Would strengthen the "model-agnostic" claim significantly.

**Effort:** ~20-40 hours (implement wrapper + benchmark).

### NICE 2: Online Similarity Threshold Tuning

Instead of fixed τ_high/τ_low, learn thresholds adaptively from running similarity statistics. Would address "how to set thresholds without validation data?" criticism.

### NICE 3: Multivariate Extension

Current implementation forecasts only `OT` (oil temperature) from ETT. Extend to multi-target forecasting.

### NICE 4: Comparison with D3A/AdaRNN

Additional baselines from the TTA literature:
- D3A (Zhang et al., 2024) — binary drift detection
- AdaRNN (Du et al., CIKM 2021) — distribution matching during training

### NICE 5: Real-World Case Study

Apply to a genuinely streaming use case (energy trading, network traffic) with actual concept drift. More convincing than benchmark datasets.

---

## 4. Execution Roadmap

### Phase A: Complete Current Benchmark (1-2 days)

1. Wait for Transformer results to finish (PID 75474)
2. Validate all 48 experiment results
3. Add RGTTA+DynaTTA to runner (optional)

### Phase B: Multi-Seed Benchmark (3-5 days compute)

1. Run full 7-policy × 4 models × 4 ETT datasets × 3 horizons × 5 seeds
   = 420 policy evaluations (~50 hours M2)
2. Run synthetic datasets: 7 policies × 4 models × 8 datasets × 3 horizons × 3 seeds
   = 2,016 policy evaluations (~30 hours — shorter series)
3. Run Weather + Exchange: 7 policies × 4 models × 2 datasets × 3 horizons × 3 seeds
   = 504 policy evaluations (~15 hours)

**Total compute: ~95-100 hours (can parallelise some)**

### Phase C: Analysis & Statistical Tests (1-2 days)

1. Aggregate multi-seed results → mean ± std tables
2. Wilcoxon signed-rank tests (pairwise)
3. Friedman + Nemenyi critical difference diagram
4. Win count analysis with confidence intervals
5. Regime-tier analysis on synthetic datasets (match rate vs ground truth)

### Phase D: τ_high Ablation (1-2 days)

1. Sweep τ_high ∈ {0.70, 0.75, 0.80, 0.85, 0.90}
2. Fix model=GRU-Small, dataset=ETTh1, horizon=96, 3 seeds
3. Plot MSE vs τ_high with std bands
4. Identify sensitivity (is performance robust across ±0.05?)

### Phase E: Paper Rewrite (2-3 days)

1. Rewrite abstract with final numbers
2. Update results section with v3 multi-seed tables
3. Add statistical significance results (p-values, CD diagram)
4. Add synthetic dataset analysis section
5. Update figures (new bar charts, CD diagram, regime-tier plot)
6. Final editing pass

### Phase F: Submission Prep (1 day)

1. arXiv formatting check
2. Supplementary material PDF
3. Code release prep (clean repo, add LICENSE, anonymise if needed)
4. README for reproducibility

---

## 5. Target Venue Analysis

| Venue | Type | Deadline | Fit |
|-------|------|----------|-----|
| arXiv preprint | Preprint | Anytime | Immediate option — establish priority |
| ICML Workshop (TTA/CL) | Workshop | ~May 2026 | Good fit — TTA-focused audience |
| NeurIPS Workshop | Workshop | ~Aug 2026 | Good fit if results are strong |
| ECML-PKDD | Conference | ~Apr 2026 | Applications track — practical focus |
| CIKM | Conference | ~May 2026 | Applied ML — good venue for practical contribution |

**Recommended path:**
1. arXiv preprint immediately after Phase E (establish priority)
2. Workshop submission at ICML 2026 (if timeline works)
3. Full conference submission at ECML-PKDD or CIKM

---

## 6. Key Research Questions Still Open

1. **Is RGTTA+EWC consistently better than EWC alone?** Current 36-experiment results show a tie (7-7 wins). Multi-seed + more datasets needed to break the tie.

2. **Does checkpoint reuse actually help on recurring synthetic data?** This is the core thesis claim but hasn't been tested on `synth_recurring` yet.

3. **Is τ_high=0.80 robust?** No sensitivity analysis done. What if optimal τ differs by dataset?

4. **Does RGTTA+DynaTTA add value over RGTTA+EWC?** The hybrid hasn't been benchmarked.

5. **Why does DynaTTA dominate on ETTm data?** Its dynamic LR seems well-suited to smooth, slowly-varying distributions. Is this an artefact of ETT's nature or a genuine advantage?

6. **Can we characterise dataset properties that predict which policy wins?** This would be the most useful practical contribution — a meta-learning angle.

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RGTTA+EWC doesn't consistently beat baselines | Medium | High | Pivot to "conditional recommendation" framing — show WHEN it wins |
| Synthetic results don't match theory | Low | High | Diagnose similarity metric — may need threshold tuning per dataset |
| Multi-seed runs show high variance | Medium | Medium | Report honestly, increase seeds to 10 if needed |
| DynaTTA dominates everything | Low | High | Emphasise RGTTA's speed advantage + show it wins on non-smooth data |
| Reviewers want larger models | High | Medium | Acknowledge limitation, frame as "policy wraps any model" |

---

*This document is the research roadmap. Update after each major milestone. See `PAPER_IMPROVEMENT_PLAN.md` for paper-specific task tracking.*
