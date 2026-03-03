# Copilot Instructions — RGTTA: Regime-Guided Test-Time Adaptation

These instructions apply to **all** Copilot interactions in this workspace.

---

## 1. Project Identity

**What:** A research project implementing **Regime-Guided Test-Time Adaptation (RGTTA)** for incremental time series forecasting. The core contribution is a **when-to-adapt and how-aggressively** strategy — not a new model architecture — that wraps any neural forecaster.

**Thesis:** When streaming time-series data arrives in batches, measuring distributional similarity to previously-seen regimes and using that similarity to *control adaptation intensity* yields a better accuracy–speed trade-off than fixed-strategy baselines (always-retrain, fixed-step TTA, fixed-EWC, dynamic-LR TTA). In particular:
- On **recurring regimes**, checkpoint reuse (specialist model) beats full-data retraining (generalist model).
- On **smooth/stationary data**, regime-guided adaptation is ≥ as accurate as TTA but significantly faster.
- On **novel distribution shocks**, aggressive adaptation + EWC regularisation preserves old knowledge while adapting quickly.

**Stage:** Pre-submission research. Definitive benchmark (Run #72) **completed**: 8 policies × 4 models × 14 datasets × 4 horizons × 3 seeds = 672 experiments. Results in `benchmarks/results/unified_v2_8pol/`. RG-policies win **69.6%** (156/224 seed-averaged). Paper draft in `paper/main.tex`. Next: figures, statistical tests, retrain baseline.

---

## 2. Architecture

### System Stack

| Layer | Location | Role |
|-------|----------|------|
| Public API | `src/regime_forecasting/__init__.py` | `RegimeAwareForecaster` wrapper |
| Core Engine | `src/regime_forecasting/core/forecaster.py` | Training, checkpoint mgmt, regime matching, prediction |
| Memory Module | `src/regime_forecasting/core/memory_module.py` | Checkpoint storage + distribution-based similarity search |
| Models | `src/regime_forecasting/models/` | 4 architectures sharing same interface |
| Preprocessing | `src/regime_forecasting/utils/data_utils.py` | MinMaxScaler [−1,1], lag features, STL decomposition |
| Evaluation | `src/regime_forecasting/utils/evaluation.py` | wMAPE, sMAPE, MSE, MAE, RMSE, direction accuracy |

### 6 Primary Update Policies (all in `benchmarks/`)

All policies use the **same base model** per experiment. Only the update strategy differs.

**Display names** use RG- prefix (RG-TTA, RG-EWC, RG-DynaTTA). **Code identifiers** use lowercase (rgtta, rgtta_ewc, rgtta_dynatta).

| # | Policy (display) | Code key | File | Strategy |
|---|-----------------|----------|------|----------|
| 1 | **TTA** | `tta` | `tta_forecaster.py` | Fixed K=20 steps, lr=3e-4 |
| 2 | **EWC** | `ewc` | `ewc_forecaster.py` | K=15 steps + Fisher penalty (λ=400) |
| 3 | **DynaTTA** | `dynatta` | `dynatta_forecaster.py` | Dynamic LR via sigmoid of 3 shift metrics |
| 4 | **RG-TTA** *(ours)* | `rgtta` | `rgtta_forecaster.py` | Regime-guided continuous TTA (v2) |
| 5 | **RG-EWC** *(ours)* | `rgtta_ewc` | `rgtta_forecaster.py` (use_ewc=True) | RG-TTA + flat EWC (λ=400) |
| 6 | **RG-DynaTTA** *(ours)* | `rgtta_dynatta` | `rgtta_dynatta_forecaster.py` | RG-TTA + DynaTTA dynamic LR |

> **Retrain** (full retrain on accumulated data) is planned as a 7th policy but not yet benchmarked. When added, it runs via the core engine with τ=999.

> **TAFAS/RG-TAFAS** excluded from primary comparison. TAFAS (Kim et al., AAAI 2025) collapses at long horizons (H=336/720, +100–400% vs TTA). Files retained: `tafas_forecaster.py`, `rgtta_tafas_forecaster.py`. Cited in paper.

#### v2 Algorithm Design (current)

RG-TTA v2 uses **continuous adaptation** — not the old 3-tier fixed-step system (v1).

- **Smooth LR**: `α = α_base × (1 + lr_sim_scale × (1 - sim))` — lower similarity → higher LR
- **Loss-driven early stopping**: patience=3, ε=0.005, min_steps=5, max_steps=25
- **Checkpoint loading**: requires `sim ≥ ckpt_sim_threshold` AND `ckpt_loss < current_loss × ckpt_loss_gate`
- **Diagnostic tiers** (for logging only): HIGH (sim≥0.85), MID (0.55–0.85), LOW (<0.55)
- **Frozen backbone**: Only `output_projection` is trainable (~10% of params)

### 4 Model Architectures (in study)

| Key | Class | File | Params |
|-----|-------|------|--------|
| gru_small | TimeSeriesTransformer | `transformer.py` (legacy name — is a GRU) | ~71K |
| itransformer | iTransformerForecaster | `itransformer_model.py` | ~123K (varies with input_dim) |
| patchtst | PatchTSTForecaster | `patchtst_model.py` | ~192K (channel-independent) |
| dlinear | DLinearForecaster | `dlinear_model.py` | ~37K–1.2M (depends on H) |

> **gru_large** (`LargeGRUForecaster`, ~330K params) is **excluded from the study**. With 330K params, even v2's max_steps=25 is insufficient for convergence per batch. RG-TTA is designed for compact models (≤150K params). File retained in `large_gru_model.py`.

### Benchmark Infrastructure

| Component | File |
|-----------|------|
| Unified runner (streaming) | `benchmarks/run_unified_benchmark.py` |
| Dataset loader | `benchmarks/data_loaders/standard_benchmarks.py` |
| Synthetic generator | `benchmarks/data_loaders/synthetic_regimes.py` |
| Paper number verification | `scripts/_verify_paper_numbers_v2.py` |
| Checkpoint analysis | `scripts/_ckpt_analysis.py` |

**Evaluation protocol:** Streaming-only. RG-TTA's checkpoint reuse, adaptive step budgets (5–25 via early stopping), and regime memory require stateful sequential processing. The sliding-window protocol (1 step/window, cumulative state) used by DynaTTA neutralises RG-TTA's key features. The sliding-window benchmark code (`run_sliding_window_benchmark.py`) is retained for reference but is **not** part of reported results.

---

## 3. Datasets (14 total)

**Real-World (6):** ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange
**Synthetic (8):** synth_stable, synth_trend_break, synth_slow_drift, synth_fast_switch, synth_recurring, synth_volatility, synth_shock_recovery, synth_multi_regime

---

## 4. Key Constants (v2)

### Benchmark Protocol
| Parameter | Value |
|-----------|-------|
| BATCH_SIZE | 750 |
| INITIAL_TRAIN_SIZE | 720 |
| MAX_BATCHES | 10 |
| SEQUENCE_LENGTH | 96 |
| Horizons | 96, 192, 336, 720 |
| Seeds | 3 (per experiment) |

### RG-TTA v2 Adaptation
| Parameter | Value | Description |
|-----------|-------|-------------|
| lr_base (α_base) | 3e-4 | Base learning rate |
| max_steps | 25 | Maximum adaptation steps per batch |
| min_steps | 5 | Minimum steps before early stopping |
| patience | 3 | Consecutive non-improving steps to stop |
| epsilon (ε) | 0.005 | Minimum loss improvement to reset patience |
| lr_sim_scale | 0.67 | LR boost factor for novel regimes |
| ckpt_sim_threshold | 0.75 | Minimum similarity to consider checkpoint |
| ckpt_loss_gate | 0.70 | Checkpoint must beat current loss × this |
| ewc_lambda | 400.0 | Flat EWC penalty (RG-EWC only) |
| Memory cap | 5 entries | FIFO eviction |
| Similarity weights | KS=0.3, Wass=0.3, Feat=0.2, Var=0.2 | Ensemble metric |

### Diagnostic Tiers (logging only, not algorithmic)
| Tier | Condition | Purpose |
|------|-----------|----------|
| HIGH | sim ≥ 0.85 | Near-identical regime — checkpoint reuse likely |
| MID | 0.55 ≤ sim < 0.85 | Moderate drift — standard adaptation |
| LOW | sim < 0.55 | Novel regime — aggressive adaptation |

---

## 5. Code Standards

- Clear names, short functions, one-line docstrings for public APIs.
- Type hints for function arguments and returns.
- Handle edge cases without silent wrong results.
- Do not introduce new dependencies without updating `pyproject.toml`.
- **Same base model always** for all 6 policies per experiment.
- Same random seed and data splits for all methods.

### Design Document Rule

**Every time** a policy is created, modified, or tuned, update `docs/RGTTA_DESIGN_DECISIONS.md` with what changed, why, and the result. Never skip this.

### Documentation-on-Change Rule (MANDATORY)

**Every time** any of the following happens, the corresponding documentation **must** be updated in the same session — no exceptions:

1. **Code change** (bug fix, optimisation, new feature, parameter change):
   - Update `docs/RGTTA_DESIGN_DECISIONS.md` with: what changed, why, before/after evidence.
   - Update `README.md` if the change affects any pseudocode, algorithm description, key design choices, constants, or timing claims.
   - If a constant changes (τ, λ, steps, LR), update **both** §4 in README and §4 Key Constants in copilot-instructions.

2. **Empirical finding** from a **solid benchmark** (≥ 3 seeds, ≥ 2 datasets — not a 1-seed smoke test):
   - Log the finding in `docs/RGTTA_DESIGN_DECISIONS.md` under a dated section.
   - If the finding changes our understanding of when a policy wins/loses, update the "When X wins" / "When X can struggle" sections in `README.md`.
   - Update `docs/PAPER_IMPROVEMENT_PLAN.md` if the finding affects paper claims.

3. **Run completion** (any benchmark, smoke test, or long-running process):
   - Update `docs/RUN_LOG.md` immediately — move from Active to Completed with outcome summary.

4. **Stale documentation detected** (README claims something the code no longer does):
   - Fix it immediately. Stale docs are bugs.

**What counts as a "solid study" vs "smoke test":**
- **Smoke test**: 1 seed, 1–2 datasets, purpose is to verify code works. Log in RUN_LOG only.
- **Solid study**: ≥ 3 seeds OR ≥ 3 datasets, purpose is to draw conclusions. Log in RUN_LOG **and** update DESIGN_DECISIONS + README if findings are new.

---

## 6. Progress Tracking

| File | Scope | Status |
|------|-------|--------|
| `docs/PAPER_IMPROVEMENT_PLAN.md` | Paper quality for arXiv | Active |
| `docs/RGTTA_DESIGN_DECISIONS.md` | Engineering notebook | Active |
| `docs/RUN_LOG.md` | Benchmark/test run tracker | Active |

**Rules:**
1. At session start, read the progress file. Check "Next action".
2. When completing a task, update immediately.
3. When starting a task, mark in-progress first.
4. Never re-plan from zero if a progress file exists.
5. If user says "continue" / "where were we?", read progress file and resume.

### Status Check Rule (MANDATORY)

If the user says **"status check"**, always return a compact snapshot with these items in order:
1. **Active run state** from `docs/RUN_LOG.md` (PID, status, log path, progress if known).
2. **Last command(s) executed** in terminal relevant to the active task.
3. **Current objective** and **next command planned**.
4. **Blocking issues** (if any) and what will be tried next.

For every long-running benchmark, ensure `docs/RUN_LOG.md` includes enough detail for status check recovery:
- exact command,
- PID,
- log file path,
- reason/goal,
- completion criteria (e.g., expected experiment count).

### Run Logging Rule

**IMMEDIATELY when launching** any benchmark, smoke test, or long-running Python process:
1. **Log it in `docs/RUN_LOG.md` BEFORE doing anything else** — record: timestamp, PID, command, reason, log file path. This is the FIRST action after the process starts, not after it finishes.
2. Add the entry to the "Active Runs" table right away so there is always a record of what was launched and on which PID.
3. When it finishes (or is killed), move it to "Completed Runs" with outcome summary.
4. At session start, check "Active Runs" — verify if any PIDs are still alive (`ps aux | grep <PID>`).
5. **Always launch in a new terminal** — never reuse an existing terminal for a new run.
6. **Do not monitor logs** unless the user explicitly asks. Let runs complete in background.
7. **Never skip logging** — even if a run is expected to be short. If it has a PID, it gets logged.
8. **After logging the run, STOP.** Do not tail logs, do not check progress, do not poll status. Wait for the user to ask. This is a strict rule — no continuous monitoring unless the user explicitly requests it.

---

## 8. VM Operations — Mandatory Procedures

### VM Registry (ALWAYS check RUN_LOG.md Active Runs for current PIDs before any VM action)

| VM Name | GCP Project | Zone | Role | Connect |
|---------|-------------|------|------|---------|
| `rgtta-benchmark` | `project-6e74dcd8-e9e5-4d74-80b` | `us-central1-a` | Retrain baseline | `gcloud config set project project-6e74dcd8-e9e5-4d74-80b && gcloud compute ssh rgtta-benchmark --zone=us-central1-a` |
| `rgtta-tta-bench` | `project-research-488102` | `us-central1-a` | TTA policies | `gcloud config set project project-research-488102 && gcloud compute ssh rgtta-tta-bench --zone=us-central1-a` |

> **CRITICAL:** `gcloud compute ssh` only works with `--command="..."` (short inline command). For multi-line commands, write a local script file and use `gcloud compute scp` + `-- bash /tmp/script.sh`. The `-- bash` form (double-dash, not `--command`) is required when the remote command contains `&` or `;` that must not be shell-expanded locally. **Never use `--project=` flag inline** — always `gcloud config set project` first.

### Checking VM Status

```bash
# List VMs and IPs
gcloud config set project project-6e74dcd8-e9e5-4d74-80b && gcloud compute instances list
gcloud config set project project-research-488102 && gcloud compute instances list

# Check running processes (safe — no pkill)
gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command="ps aux | grep python | grep -v grep | awk '{print \$2, \$3, \$11}'"

# Check log progress
gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command="grep -c '✅' ~/benchmark_retrain_v5.log; tail -3 ~/benchmark_retrain_v5.log"
```

### Uploading Files to a VM

```bash
# Single file (WORKS — scp handles spaces in local path fine)
gcloud compute scp --zone=us-central1-a \
  '/Users/indarkumar/Documents/Incremental_learning research/benchmarks/run_unified_benchmark.py' \
  rgtta-benchmark:~/rgtta/benchmarks/run_unified_benchmark.py

# Same for VM2 (add --project=project-research-488102)
gcloud compute scp --zone=us-central1-a --project=project-research-488102 \
  '/Users/indarkumar/Documents/Incremental_learning research/benchmarks/run_unified_benchmark.py' \
  rgtta-tta-bench:~/rgtta/benchmarks/run_unified_benchmark.py

# Always verify after upload:
gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command="grep -n 'ALL_POLICIES = \[' ~/rgtta/benchmarks/run_unified_benchmark.py | head -3"
```

**Files to always sync before a new run:** `run_unified_benchmark.py`, `rgtta_forecaster.py`, `rgtta_dynatta_forecaster.py`, `ewc_forecaster.py`, `tta_forecaster.py`, `dynatta_forecaster.py`.

### Killing a Running Benchmark (SAFE METHOD)

**NEVER use `pkill -f run_unified_benchmark`** — the string "run_unified_benchmark" appears in the SSH command itself, causing the SSH session to be killed and exit 255.

```bash
# Step 1: Get the PIDs (check RUN_LOG.md Active Runs first — PID is logged there)
gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command="pgrep -a python | grep run_unified | awk '{print \$1}'"

# Step 2: Kill by PID
gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command="kill <PID1> <PID2>; sleep 2; echo done"

# Step 3: Confirm dead
gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command="ps aux | grep python | grep -v grep | wc -l"
```

### Launching a New Benchmark Run on a VM

**Always use a local script file + scp + `-- bash`** to avoid shell quoting issues with `&`, `>`, `$!`:

```bash
# Step 1: Write the launch script locally
cat > /tmp/vm1_launch.sh << 'EOF'
cd ~/rgtta
export OMP_NUM_THREADS=1
export PYTHONPATH=.:src:benchmarks:benchmarks/data_loaders
nohup .venv/bin/python -u benchmarks/run_unified_benchmark.py \
  --policies retrain \
  --horizons 96 192 336 720 \
  --seeds 3 \
  --models gru_small itransformer patchtst dlinear \
  --results-dir benchmarks/results/unified_retrain_v5 \
  --workers 10 > ~/benchmark_retrain_v5.log 2>&1 & disown
echo "PID:$!"
EOF

# Step 2: Upload the script
gcloud compute scp --zone=us-central1-a /tmp/vm1_launch.sh rgtta-benchmark:/tmp/vm1_launch.sh

# Step 3: Run it (-- bash, NOT --command)
gcloud compute ssh rgtta-benchmark --zone=us-central1-a -- bash /tmp/vm1_launch.sh

# Step 4: Verify after ~15 seconds
sleep 15 && gcloud compute ssh rgtta-benchmark --zone=us-central1-a --command="ps aux | grep 'run_unified' | grep -v grep | wc -l; tail -3 ~/benchmark_retrain_v5.log"
```

**Key flags:**
- `-u` (unbuffered) on python — ensures log output appears immediately, not after buffer fills
- `disown` after `&` — detaches process from SSH session so it survives disconnect
- `nohup` — prevents SIGHUP on SSH close

### Standard Benchmark Commands

**VM1 — Retrain only:**
```
--policies retrain --horizons 96 192 336 720 --seeds 3 --models gru_small itransformer patchtst dlinear --results-dir benchmarks/results/unified_retrain_v5 --workers 10
```

**VM2 — All 6 TTA policies (no retrain, no TAFAS):**
```
--policies tta ewc dynatta rgtta rgtta_ewc rgtta_dynatta --horizons 96 192 336 720 --seeds 3 --models gru_small itransformer patchtst dlinear --results-dir benchmarks/results/unified_v2_8pol --workers 10
```

> Models: `gru_small itransformer patchtst dlinear` only. **`gru_large` is excluded** from all definitive runs (330K params, under-adapts at 12 steps).

### Pre-Action Checklist (MUST DO before any VM operation)

1. **Read `docs/RUN_LOG.md` Active Runs** — note current PIDs, log file paths, which project each VM is in.
2. **Set the correct gcloud project** (`gcloud config set project <project>`) before any `gcloud compute` command.
3. **Confirm VM is RUNNING** (`gcloud compute instances list`) before SSH.
4. **When killing**: use PID from RUN_LOG, not `pkill -f`.
5. **When launching**: always use script-file method, always verify with `wc -l` + `tail` after 15s.
6. **Log the new run in RUN_LOG.md IMMEDIATELY** after launch — before doing anything else.

---

## 7. Important Conventions

- **Similarity metric**: Ensemble of KS (0.3), Wasserstein-1 (0.3), normalised Euclidean (0.2), Variance-ratio (0.2). Not cosine.
- **Checkpoints co-save preprocessor state**: MinMaxScaler pickled inside checkpoint metadata.
- **Model naming**: `TimeSeriesTransformer` = GRU (legacy). iTransformer = `iTransformerForecaster`. PatchTST = `PatchTSTForecaster`.
- **Multivariate support**: Real-world datasets use multivariate input (all covariates as input channels). Synthetic datasets remain univariate. The target column’s raw name (e.g. `OT`) is excluded from feature_cols since `y` already represents it.
- **Frozen backbone**: All gradient-based TTA policies (TTA, EWC, DynaTTA, RG-TTA, RG-EWC, RG-DynaTTA) freeze the model backbone during adaptation. Only `output_projection` is trainable (~10% of params). This is implicit regularisation — prevents catastrophic forgetting of temporal representations.
- **Checkpoint loss gate (v2)**: Checkpoint loading requires `sim ≥ 0.75` AND `ckpt_loss < current_loss × 0.70` (30% improvement required). Prevents reverting to stale checkpoints on drifting data.
- **Smooth LR modulation (v2)**: `α = α_base × (1 + 0.67 × (1 − sim))`. No discrete tiers — LR scales continuously with novelty.
- **Loss-driven early stopping (v2)**: Stops adaptation when loss plateaus (patience=3, ε=0.005). Guarantees min_steps=5, caps at max_steps=25.
- **DynaTTA official code**: Found at `shivam-grover/DynaTTA`. Official implementation = TAFAS + DynamicGCM + dynamic LR. Our `dynatta_forecaster.py` correctly implements Algorithm 1's dynamic LR formula but applies it to full-model TTA (not frozen+GCM).
- **DynaTTA warmup**: `warmup_steps = warmup_factor * tta_steps * 3` (NOT `forecast_horizon`). Must be horizon-independent — batch protocol increments `n_adapt` by `tta_steps` per batch, not per-sample like sliding-window.
- **DynaTTA/RG-DynaTTA embedding extraction**: `_extract_embedding()` must handle all 4 architectures: GRU (input_projection → gru), PatchTST (patch_embed → encoder → pool), and fallback (raw input pooling for iTransformer/DLinear). PatchTST has `self.encoder` but expects patched input, not raw tensors.
- **Horizon independence**: All adaptation hyperparameters (step counts, LRs, thresholds, warmup) must be constant across horizons. Only model output size and initial_train_size may scale with `forecast_horizon`.
