# Progress: Publication baselines (regime-switching / test-time adaptation)

**Single source of truth** for the work that adds 1–2 existing methods as baselines so the paper can say we compare **update policies** on the **same base model**. Read this at the start of any session; update it when status or next steps change.

Project Cursor rules (`.cursor/rules/`) enforce: (1) progress-discipline — read/update this file so we never lose context; (2) code-standards — high quality, same base model for baselines, tests.

---

## Goal

- Add **1–2 baselines** that are existing methods (regime-switching or test-time adaptation / model portfolios).
- **Same base model for all:** Our forecaster (e.g. GRU in `TimeSeriesTransformer`) is the **only** base architecture. Every method—ours, always-retrain, TTA, etc.—uses this same model. Only the **update/adaptation policy** differs. This gives a true comparison: we are comparing **when to retrain vs when to reuse/adapt**, not model A vs model B.
- Outcome: Benchmarks that compare (1) Always retrain, (2) Test-time adaptation (same model, different policy), (3) Optional: one regime-related baseline if feasible with same model, (4) Ours (regime-aware checkpoint reuse).

---

## Principle: Same base model

- **Yes:** Test-time adaptation (and any other baseline) must use the **same base model** (our GRU forecaster / `TimeSeriesTransformer`) so the comparison is fair.
- **Policy-only comparison:**
  - **Always retrain:** Same GRU; on new data, retrain on all accumulated data.
  - **Test-time adaptation (TTA):** Same GRU; on new data, run a small number of gradient steps on the new batch only (no checkpoint memory; adapt in place).
  - **Ours:** Same GRU; on new data, regime match → load checkpoint or retrain and save.
- No comparing our GRU to Chronos/PatchTST/etc. here; that would mix architecture and policy. Same model, different strategies only.

---

## Phases (plan)

### Phase 1: Research and choose baselines

- [x] **1.1** List 1–2 concrete methods that can be implemented **with our same base model** (GRU forecaster):
  - **TTA:** Test-time adaptation = continue training the same model on the new batch only (e.g. N steps, same loss). No regime memory. Straightforward.
  - **Optional second:** e.g. “always fine-tune on last K points” (sliding-window update), or a simple regime-switching variant that still uses our GRU (e.g. same model, different “when to reset” rule). Document choice and rationale.
- [x] **1.2** Write short “Baseline definitions” in this file (one paragraph each): what each baseline does, same model, same data protocol.

### Phase 2: Implementation

- [x] **2.1** Implement **test-time adaptation** policy using existing `CorrectedRegimeForecaster` / same GRU:
  - New policy: on `update_with_new_data`, never search regime memory; always take current model and run K gradient steps on the new batch only (no full retrain on all data). Same loss, same optimizer settings as our training.
  - Expose as a flag or a separate runner class so benchmark script can call “same model, TTA policy.”
- [x] **2.2** If second baseline chosen: implement it with same base model; same interface (incremental batches, same metrics).
- [x] **2.3** Unit or integration test: same base model instance, run one batch with “always retrain” vs “TTA” vs “ours” and check that only update logic differs (e.g. train time, number of params).

### Phase 3: Benchmarking

- [x] **3.1** Extend benchmark runner (or add script) to run:
  - Same data splits, same base model config: (1) Always retrain, (2) TTA, (3) Ours (regime-aware reuse).
- [x] **3.2** Report metrics: MAPE (or same as now), training time per batch, total time. Same format as current comprehensive benchmark.
- [x] **3.3** Run on same 33-dataset setup (or a subset for speed) and record results in `benchmarks/results/` and in this file (summary table).

### Phase 4: Documentation and paper narrative

- [x] **4.1** Update README: add “Comparison with other update policies” with same-base-model explanation and summary table (Always retrain vs TTA vs Ours).
- [x] **4.2** Add “Baseline definitions” to docs or README so reviewers see that all use the same base model.

---

## Current status

| Item | Status | Notes |
|------|--------|------|
| Progress file created | Done | This file. |
| Phase 1: Research and choose baselines | Done | TTA confirmed; second baseline deferred. |
| Phase 2: Implementation | Done | TTA in benchmarks/tta_forecaster.py; run_baseline_comparison.py added. |
| Phase 3: Benchmarking | Done | run_baseline_comparison.py runs three-way. |
| Phase 4: Documentation | Done | README section added. |

**Current phase:** All phases complete.

**Immediate to-do:**
- [ ] **Analytical / numerical study:** Find out (in a Jupyter notebook or by numbers) *why* we win in some cases against full train or TTA. Identify which datasets and scenario types favour regime-aware reuse; formulate and check hypotheses (e.g. recurring regimes → correct checkpoint; novel shock → wrong match; TTA overfits to small batch; full retrain dilutes regime signal). Document findings for the paper as a “why we win / when we lose” section.

**Optional todo (for later):**
- [ ] **Optional 1:** Run publication benchmark (ETT: ETTh1, ETTh2, ETTm1, ETTm2) to completion; document ETT numbers in README and `BENCHMARK_COMPARISON.md`.
- [ ] **Optional 2:** Add second baseline (e.g. “fine-tune on last K points only” with same base model); extend three-way script to four-way; document.


---

## Baseline definitions (Phase 1 complete)

- **Always retrain (current baseline):** Same GRU (`TimeSeriesTransformer`). On every new batch, retrain from scratch on **all** accumulated data. Already implemented in `benchmarks/baseline_forecaster.py`. Same data protocol: `fit(full_history)` each time in the benchmark.
- **Test-time adaptation (TTA):** Same GRU. On every new batch, **do not** retrain on full history; take the **current** model and run K gradient steps on the **new batch only** (same loss, same optimizer type). No checkpoint memory, no full retrain. Implemented in `benchmarks/tta_forecaster.py`; no changes to `src/regime_forecasting` or to our regime-aware forecaster.
- **Ours (regime-aware reuse):** Same GRU. On new data, match distribution to stored regimes; if match → load checkpoint (no training); if no match → retrain and save. Implemented in `CorrectedRegimeForecaster`.
- **Second baseline (optional):** Deferred. Option for later: "fine-tune on last K points only" (sliding window) with same base model.

---

## Documented three-way benchmark results (reproducible)

**Run:** 33 datasets, 4 initial training batches, 6 incremental batches per dataset. Same data as comprehensive benchmark. Command to reproduce:

```bash
PYTHONPATH=.:src:benchmarks python benchmarks/run_baseline_comparison.py
```

**Results (2026-02-18):**

| Method | Total training time | Avg. MAPE |
|--------|---------------------|-----------|
| Always retrain | 119.1 s | 20.81% |
| TTA | 62.6 s | 18.35% |
| Regime (ours) | 35.5 s | 21.50% |

- Full per-dataset results: `benchmarks/results/baseline_comparison/three_way_results.json`.
- Regime saves ~70% time vs baseline; TTA ~47%. All use same base model; only update policy differs.

---

## Where things live

- Base model: `src/regime_forecasting/models/transformer.py` (`TimeSeriesTransformer` – GRU).
- Forecaster / update logic: `src/regime_forecasting/core/forecaster.py` (`CorrectedRegimeForecaster`).
- Comprehensive benchmark: `benchmarks/run_comprehensive_benchmark.py`, `benchmarks/benchmark_runner.py`.
- Three-way comparison: `benchmarks/run_baseline_comparison.py`; results: `benchmarks/results/baseline_comparison/three_way_results.json`.
- **Why we win analysis:** `notebooks/why_regime_wins_analysis.ipynb` — loads three-way results, classifies wins/losses by scenario, two-regime toy MSE, exports tables for paper.
- All results: `benchmarks/results/` (`comprehensive/`, `baseline_comparison/`, `ablation/`, `publication/`). See also `benchmarks/results/BENCHMARK_COMPARISON.md`.

---

*Last updated: 2026-02-18 (Phases 2–4 complete; three-way results documented; README and progress updated for reproducibility).*
