"""
RG-TTA: Regime-Guided Meta-Control for Test-Time Adaptation in Streaming Time Series

A library for building adaptive time series forecasting models that can detect
regime changes and incrementally train on distribution changes.

Design:
- Distribution matching IS regime detection (no separate flag detection)
- Match found → Load checkpoint model, forecast directly (no training)
- No match (distribution change) → Train full + partial checkpoints, save both
- Save ALL checkpoints (full and partial) with their distribution features
"""

from typing import Optional

from .core.forecaster import CorrectedRegimeForecaster
from .core.memory_module import MemoryModule
from .models.transformer import TimeSeriesTransformer
from .utils.distribution_detection import flag_intuitive_erratic_seasons

__version__ = "0.3.0"
__all__ = [
    "CorrectedRegimeForecaster",
    "MemoryModule",
    "TimeSeriesTransformer",
    "flag_intuitive_erratic_seasons",
    "RegimeAwareForecaster",
]


# Backwards-compatible wrapper matching older API expected by tests
class RegimeAwareForecaster:
    """
    Wrapper class providing backwards-compatible API.

        Core Logic:
        1. When new data arrives, extract distribution features
        2. Search memory for matching checkpoint
        3. MATCH FOUND: Load checkpoint → forecast (no training)
        4. NO MATCH (distribution change):
           - Train new FULL model from scratch on all data
           - Fine-tune previous FULL on last 3 seasons → save as PARTIAL
           - Save both with distribution features
           - Forecast using model chosen by model_selection (full/partial/adaptive)
    """

    def __init__(
        self,
        season_length: int = 12,
        forecast_horizon: int = 6,
        storage_path: str = "./checkpoints",
        model_config: Optional[dict] = None,
        similarity_threshold: float = 0.8,
        model_selection: str = "adaptive",  # "full", "partial", or "adaptive"
        dynamic_threshold: bool = False,  # NEW: Enable dynamic threshold
        verbose: bool = True,
        **kwargs,
    ):
        cfg = model_config or {}
        hidden_dim = cfg.get("hidden_dim", 64)
        num_layers = cfg.get("num_layers", 2)
        num_heads = cfg.get("num_heads", 4)

        # Create underlying corrected forecaster
        self._core = CorrectedRegimeForecaster(
            season_length=season_length,
            forecast_horizon=forecast_horizon,
            sequence_length=cfg.get("sequence_length", 24),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=cfg.get("dropout", 0.1),
            similarity_threshold=similarity_threshold,
            device=kwargs.get("device", "cpu"),
            storage_path=storage_path,
            model_selection=model_selection,
            dynamic_threshold=dynamic_threshold,
        )

        # Set logging level based on verbose flag
        import logging

        if not verbose:
            logging.getLogger("regime_forecasting").setLevel(logging.WARNING)

        # Expose convenient attributes expected by older APIs/demos
        self.memory_module = self._core.memory_module
        self.regime_detector = self._core.preprocessor

        self.season_length = season_length
        self.forecast_horizon = forecast_horizon
        self.similarity_threshold = similarity_threshold
        self.model_selection = model_selection
        self.dynamic_threshold = dynamic_threshold
        self.is_fitted = False
        self.model = None
        self.current_regime_id = None

    def fit_incremental(self, df, start_period: int = 24, epochs_per_regime: int = 20, **kwargs):
        """
        Fit on provided data.
        - First call: Initial training, creates first checkpoint
        - Subsequent calls: Checks for distribution match
          - Match: Load checkpoint (no training)
          - No match: Create new full + partial checkpoints
        """
        result = self._core.fit_incremental(df, epochs_per_segment=epochs_per_regime, **kwargs)
        self.is_fitted = True
        self.model = self._core.model
        self.current_regime_id = self._core.latest_full_checkpoint_id
        return result

    def predict(self, df, steps_ahead: int = 6):
        """Make predictions using current model."""
        return self._core.predict(context_df=df, steps_ahead=steps_ahead)

    def update_with_new_data(self, new_df, epochs: int = 20, **kwargs):
        """
        Update with new data (recommended API).
        Returns info about whether match was found or distribution change occurred.
        """
        result = self._core.update_with_new_data(new_df, epochs=epochs, **kwargs)
        self.model = self._core.model
        self.current_regime_id = self._core.latest_full_checkpoint_id
        return result

    def get_regime_info(self):
        """Get information about current regime and stored checkpoints."""
        core_info = self._core.get_regime_info()
        return {
            "current_regime_id": self.current_regime_id,
            "memory_module": self.memory_module,
            "regime_detector": self.regime_detector,
            "is_fitted": self.is_fitted,
            "total_checkpoints": core_info.get("total_checkpoints", 0),
            "checkpoint_ids": core_info.get("checkpoint_ids", []),
        }

    def get_threshold_history(self):
        """
        Get history of dynamically adjusted thresholds.
        Only available when dynamic_threshold=True.

        Returns:
            List of dicts with keys: threshold, cv, trend_strength, acf1
        """
        return self._core.threshold_history

    def get_current_threshold(self):
        """Get the current similarity threshold (may differ from base if dynamic)."""
        return self._core.similarity_threshold
