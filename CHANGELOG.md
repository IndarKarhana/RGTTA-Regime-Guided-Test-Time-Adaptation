# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - Paper Improvements for arXiv

### Added
- **Paper improvement plan**: `docs/PAPER_IMPROVEMENT_PLAN.md` — 6-phase plan covering figures, standard ETT metrics, narrative reframe, EWC baseline, appendix, and style fixes
- **References expanded**: 20 → 32 citations (added LSTM, LayerNorm, Huber loss, gradient clipping, Autoformer, PatchTST, N-BEATS, Demšar, Box-Jenkins, Synaptic Intelligence, deep learning surveys)
- **Copilot instructions updated**: added paper-improvement progress file rule to `.github/copilot-instructions.md`

### Planned (see `docs/PAPER_IMPROVEMENT_PLAN.md`)
- Phase 1: 3 figures (workflow diagram, MAPE bar chart, threshold ablation plot)
- Phase 2: Standard MSE/MAE on ETT at horizons 96/192/336/720
- Phase 3: Narrative reframe (abstract, conclusion, remove redundancy)
- Phase 4: EWC baseline (4th comparison method)
- Phase 5: Appendix (full per-dataset results, hyperparameter table, dataset descriptions)
- Phase 6: Style fixes (separate limitations/future work, tighten writing)

## [0.3.0] - 2026-01-22

### Added
- **Dynamic Threshold Feature**: New `dynamic_threshold` parameter that automatically adapts similarity threshold based on data characteristics:
  - Coefficient of Variation (volatility)
  - Trend strength
  - Autocorrelation
  
- **Threshold Ablation Study**: Comprehensive analysis of similarity threshold impact on ETTm1 and ETTh1 datasets
  - Tested thresholds: [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
  - Key finding: Optimal threshold is dataset-dependent
  - Results saved in `benchmarks/results/ablation/`

- **New Methods**:
  - `get_threshold_history()`: Returns history of dynamically adjusted thresholds
  - `get_current_threshold()`: Returns current active similarity threshold

- **New Scripts**:
  - `benchmarks/run_ablation_study.py`: Full ablation study for threshold parameter
  - `benchmarks/test_dynamic_threshold.py`: Test script for dynamic threshold feature

### Changed
- Updated `RegimeAwareForecaster` API to expose `dynamic_threshold` and `verbose` parameters
- Updated README with ablation study results and dynamic threshold documentation
- Updated tuning guide with evidence-based recommendations

### Fixed
- Memory module now correctly updates threshold when dynamic threshold is enabled

## [0.2.0] - 2026-01-21

### Added
- Adaptive model selection based on data volatility (CV threshold = 15%)
- Publication benchmark on standard ETT datasets
- Comprehensive benchmark on 33 datasets (synthetic + real)

### Changed
- Default model architecture: GRU-based encoder (61,768 parameters)
- Improved numerical stability with weight clamping

## [0.1.0] - Initial Release

### Added
- Core regime-aware incremental forecasting framework
- Memory module for checkpoint storage and similarity matching
- Transformer-based time series model
- Basic demos and tests
