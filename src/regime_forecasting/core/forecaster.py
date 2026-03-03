"""
RegimeAwareForecaster with corrected design:
1. Distribution matching IS regime detection - no separate flag detection
2. Match found → Load checkpoint model, forecast directly (no training)
3. No match found (distribution change) → Train new full model from scratch, 
   fine-tune previous full model for partial checkpoint, save both with distributions
4. Save ALL checkpoints (full and partial) with their distribution features
"""
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy import stats

from ..core.memory_module import MemoryModule
from ..models.transformer import TimeSeriesTransformer, regime_aware_loss
from ..utils.data_utils import (
    DataPreprocessor,
    create_lagged_features,
    prepare_sequences,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CorrectedRegimeForecaster:
    """
    Regime-aware forecaster with correct design:
    - Distribution matching IS regime detection
    - Match found: use matched checkpoint for forecast (no training)
    - No match (distribution change): train full + partial checkpoints
    - Adaptive mode: automatically choose FULL or PARTIAL based on volatility
    - Dynamic threshold: automatically adjust similarity threshold based on data characteristics
    """
    
    # Volatility threshold for adaptive mode (coefficient of variation)
    VOLATILITY_THRESHOLD = 0.15  # 15% CV is considered high volatility
    
    # Dynamic threshold configuration
    DYNAMIC_THRESHOLD_CONFIG = {
        "min_threshold": 0.5,      # Minimum similarity threshold
        "max_threshold": 0.9,      # Maximum similarity threshold  
        "base_threshold": 0.75,    # Base threshold for stable data
        "volatility_factor": 0.2,  # How much to reduce threshold per unit CV
        "trend_factor": 0.1,       # Adjustment for trend strength
    }

    def __init__(
        self,
        season_length: int = 12,
        forecast_horizon: int = 6,
        sequence_length: int = 24,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        similarity_threshold: float = 0.8,
        device: str = "cpu",
        storage_path: str = "./checkpoints",
        model_selection: str = "adaptive",  # "full", "partial", or "adaptive"
        dynamic_threshold: bool = False,    # NEW: Enable dynamic threshold
        model_class: type = None,
        model_kwargs: dict = None,
        input_dim: int = 1,
        feature_cols: list = None,
    ):
        self.season_length = season_length
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length
        self.base_similarity_threshold = similarity_threshold
        self.similarity_threshold = similarity_threshold
        self.dynamic_threshold = dynamic_threshold
        self.device = torch.device(device)
        self.storage_path = storage_path
        
        # Model selection mode: "full", "partial", or "adaptive"
        if model_selection not in ["full", "partial", "adaptive"]:
            raise ValueError(f"model_selection must be 'full', 'partial', or 'adaptive', got '{model_selection}'")
        self.model_selection = model_selection

        # Model class — defaults to TimeSeriesTransformer (GRU)
        self._model_class = model_class or TimeSeriesTransformer
        self._model_kwargs = model_kwargs or {}
        self.input_dim = input_dim
        self.feature_cols = feature_cols

        # Model parameters
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # State variables
        self.model = None  # Current active model for forecasting
        self.preprocessor = DataPreprocessor()
        self.memory_module = MemoryModule(
            storage_path=storage_path, 
            similarity_threshold=similarity_threshold
        )
        self.exog_cols = []

        # Training data accumulation
        self.accumulated_data = None
        self.latest_full_checkpoint_id = None  # Track latest full model
        self.current_checkpoint_id = 0
        
        # Track threshold history for dynamic adjustment
        self.threshold_history = []

        logger.info(f"🚀 CorrectedRegimeForecaster initialized (device: {device})")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Dynamic threshold: {dynamic_threshold}")
        logger.info(f"   Season length: {season_length}")

    def extract_distribution_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract distribution features from time series data.
        Features: [mean, std, skewness, kurtosis, autocorr_lag1]
        """
        y = np.asarray(y, dtype=float)
        
        mean_val = float(np.mean(y))
        std_val = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
        skew_val = float(stats.skew(y)) if len(y) > 2 else 0.0
        kurt_val = float(stats.kurtosis(y)) if len(y) > 3 else 0.0
        
        if len(y) > 1:
            acf1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        else:
            acf1 = 0.0
            
        features = np.array([mean_val, std_val, skew_val, kurt_val, acf1], dtype=float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def _detect_volatility(self, y: np.ndarray) -> Tuple[float, bool]:
        """
        Detect if data is high volatility using coefficient of variation (CV).
        
        Args:
            y: Time series values
            
        Returns:
            Tuple of (coefficient_of_variation, is_high_volatility)
        """
        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]  # Remove NaNs
        
        if len(y) < 2:
            return 0.0, False
        
        mean_val = np.abs(np.mean(y))
        std_val = np.std(y, ddof=1)
        
        # Coefficient of variation (normalized volatility)
        if mean_val > 1e-8:
            cv = std_val / mean_val
        else:
            cv = std_val if std_val > 0 else 0.0
        
        is_high_volatility = cv > self.VOLATILITY_THRESHOLD
        
        return float(cv), is_high_volatility

    def compute_dynamic_threshold(self, y: np.ndarray) -> float:
        """
        Compute a dynamic similarity threshold based on data characteristics.
        
        The threshold adapts based on:
        1. Volatility (CV): Higher volatility → lower threshold (more reuse)
        2. Trend strength: Strong trends → higher threshold (more training)
        3. Autocorrelation: High autocorr → lower threshold (patterns persist)
        
        Args:
            y: Time series values to analyze
            
        Returns:
            Adapted similarity threshold
        """
        if not self.dynamic_threshold:
            return self.base_similarity_threshold
        
        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        
        if len(y) < 10:
            return self.base_similarity_threshold
        
        config = self.DYNAMIC_THRESHOLD_CONFIG
        
        # 1. Compute volatility (coefficient of variation)
        cv, _ = self._detect_volatility(y)
        
        # 2. Compute trend strength (normalized slope)
        try:
            x = np.arange(len(y))
            slope, _, r_value, _, _ = stats.linregress(x, y)
            trend_strength = abs(r_value)  # R² as trend strength [0, 1]
        except:
            trend_strength = 0.0
        
        # 3. Compute autocorrelation at lag 1
        try:
            if len(y) > 1:
                acf1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
                acf1 = max(0, acf1)  # Only consider positive autocorr
            else:
                acf1 = 0.0
        except:
            acf1 = 0.0
        
        # Start from base threshold
        threshold = config["base_threshold"]
        
        # Adjust for volatility: high CV → lower threshold
        # Intuition: volatile data benefits from more checkpoint reuse
        volatility_adjustment = -config["volatility_factor"] * cv
        
        # Adjust for trend: strong trend → slightly higher threshold  
        # Intuition: trending data may need fresh training
        trend_adjustment = config["trend_factor"] * trend_strength
        
        # Adjust for autocorrelation: high autocorr → lower threshold
        # Intuition: persistent patterns → reuse works better
        autocorr_adjustment = -0.05 * acf1
        
        # Apply adjustments
        threshold = threshold + volatility_adjustment + trend_adjustment + autocorr_adjustment
        
        # Clamp to valid range
        threshold = max(config["min_threshold"], min(config["max_threshold"], threshold))
        
        # Update memory module threshold
        self.similarity_threshold = threshold
        self.memory_module.similarity_threshold = threshold
        
        # Track history
        self.threshold_history.append({
            "threshold": threshold,
            "cv": cv,
            "trend_strength": trend_strength,
            "acf1": acf1
        })
        
        logger.info(f"🎯 Dynamic threshold: {threshold:.3f} (CV={cv:.3f}, trend={trend_strength:.3f}, acf={acf1:.3f})")
        
        return threshold

    def _choose_model_for_change(self, new_data: pd.DataFrame) -> str:
        """
        Choose which model to use on distribution change based on model_selection mode.
        
        Args:
            new_data: The new data batch
            
        Returns:
            "full" or "partial"
        """
        if self.model_selection == "full":
            return "full"
        elif self.model_selection == "partial":
            return "partial"
        else:  # adaptive
            # Use recent data to detect volatility
            y_values = new_data["y"].values
            cv, is_volatile = self._detect_volatility(y_values)
            
            if is_volatile:
                logger.info(f"📊 High volatility detected (CV={cv:.3f} > {self.VOLATILITY_THRESHOLD})")
                logger.info("   → Using PARTIAL model (recent data more relevant)")
                return "partial"
            else:
                logger.info(f"📊 Low volatility detected (CV={cv:.3f} ≤ {self.VOLATILITY_THRESHOLD})")
                logger.info("   → Using FULL model (more training data beneficial)")
                return "full"

    def fit_initial(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Initial training on historical data.
        Creates the first full checkpoint.
        
        Args:
            df: DataFrame with columns ['ds', 'y', 'unique_id']
            
        Returns:
            Training summary
        """
        logger.info("🎯 Starting initial training")

        # Ensure required columns
        if "unique_id" not in df.columns:
            df = df.copy()
            df["unique_id"] = "ts_001"

        df = df.sort_values("ds").reset_index(drop=True)

        # Create lagged features
        df = create_lagged_features(df, lags=[1, self.season_length])
        self.exog_cols = ["lag_1", f"lag_{self.season_length}"]
        logger.info(f"📈 Created lagged features: {self.exog_cols}")

        # Store accumulated data
        self.accumulated_data = df.copy()

        # Train full model from scratch
        logger.info("🔧 Training full model on all historical data...")
        train_result = self._train_model_from_scratch(
            df,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
        )

        if train_result["status"] == "completed":
            # Extract distribution features from last 3 seasons
            recent_data = df.tail(self.season_length * 3)
            distribution_features = self.extract_distribution_features(recent_data["y"].values)

            # Save as full checkpoint
            full_checkpoint_id = self._save_to_memory(
                checkpoint_type="full",
                data=df,
                distribution_features=distribution_features,
            )
            self.latest_full_checkpoint_id = full_checkpoint_id

            logger.info(f"✅ Initial training completed")
            logger.info(f"💾 Full checkpoint saved: {full_checkpoint_id}")

            return {
                "status": "completed",
                "checkpoint_id": full_checkpoint_id,
                "training_result": train_result,
            }
        else:
            return {"status": "failed", "reason": train_result.get("reason")}

    def update_with_new_data(
        self,
        new_df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Update forecaster with new data.
        
        Logic:
        1. Extract distribution features from new data (last 3 seasons)
        2. Search memory for matching checkpoint
        3. If MATCH FOUND: Load matched model, ready for forecast (no training)
        4. If NO MATCH (distribution change):
           - Train NEW full model from scratch on all accumulated data
           - Fine-tune PREVIOUS full model on new data → save as partial
           - Save both checkpoints with distribution features
           - Forecast using new full model
           
        Args:
            new_df: New data to incorporate
            
        Returns:
            Update summary with match/change info
        """
        logger.info("🔄 Processing new data...")

        if self.accumulated_data is None:
            raise ValueError("No initial training. Call fit_initial first.")

        # Prepare new data
        new_df = new_df.copy()
        if "unique_id" not in new_df.columns:
            new_df["unique_id"] = "ts_001"
        new_df = new_df.sort_values("ds").reset_index(drop=True)

        # Add lagged features to new data
        new_df = create_lagged_features(new_df, lags=[1, self.season_length])

        # Compute dynamic threshold if enabled (before similarity search)
        if self.dynamic_threshold:
            feature_window = min(len(new_df), self.season_length * 3)
            y_values = new_df.tail(feature_window)["y"].values
            self.compute_dynamic_threshold(y_values)

        # Extract distribution features from new data (last 3 seasons or all if shorter)
        feature_window = min(len(new_df), self.season_length * 3)
        new_distribution = self.extract_distribution_features(
            new_df.tail(feature_window)["y"].values
        )
        
        logger.info(f"📊 New data distribution features: {new_distribution}")

        # STEP 1: Search memory for matching checkpoint
        match_result = self.memory_module.find_similar_regime(new_distribution)

        if match_result is not None:
            # ========== MATCH FOUND: No training needed ==========
            logger.info(f"✅ MATCH FOUND: {match_result['regime_id']} (similarity: {match_result['similarity']:.3f})")
            logger.info("📚 Loading matched checkpoint model - no training required")

            # Load the matched model weights AND preprocessor
            self._load_checkpoint(match_result['weights'], match_result.get('metadata'))

            # Update accumulated data (still need to track all data)
            self.accumulated_data = pd.concat(
                [self.accumulated_data, new_df], ignore_index=True
            ).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)

            return {
                "action": "match_found",
                "matched_checkpoint": match_result['regime_id'],
                "similarity": match_result['similarity'],
                "training_performed": False,
                "message": "Loaded existing checkpoint, ready for forecast",
            }

        else:
            # ========== NO MATCH: Distribution change - need training ==========
            logger.info("🆕 NO MATCH: Distribution change detected!")
            logger.info("📈 Training new checkpoints...")

            # Combine accumulated data with new data
            combined_df = pd.concat(
                [self.accumulated_data, new_df], ignore_index=True
            ).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)
            
            # Ensure lagged features are present
            if "lag_1" not in combined_df.columns:
                combined_df = create_lagged_features(combined_df, lags=[1, self.season_length])

            # Get previous full model state (before training new one)
            previous_full_state = None
            if self.model is not None:
                previous_full_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # ----- TRAIN NEW FULL MODEL from scratch on ALL accumulated data -----
            logger.info("🔧 Training NEW FULL model on all accumulated data...")
            
            full_train_result = self._train_model_from_scratch(
                combined_df,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
            )

            if full_train_result["status"] != "completed":
                logger.warning("⚠️ Full model training failed")
                return {"action": "training_failed", "reason": full_train_result.get("reason")}

            # Save new full model state
            new_full_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

            # Extract distribution features for full model (all data)
            full_distribution = self.extract_distribution_features(combined_df["y"].values)

            # Save NEW FULL checkpoint
            new_full_checkpoint_id = self._save_to_memory(
                checkpoint_type="full",
                data=combined_df,
                distribution_features=full_distribution,
            )
            logger.info(f"💾 New FULL checkpoint saved: {new_full_checkpoint_id}")

            # ----- TRAIN PARTIAL MODEL: fine-tune previous full on last 3 seasons -----
            partial_checkpoint_id = None
            partial_state = None
            if previous_full_state is not None:
                logger.info("🔧 Training PARTIAL model (fine-tuning previous full)...")
                
                # Get recent data for partial checkpoint — must be large enough
                # to create training samples after lag feature creation.
                # Lag features consume `season_length` rows (NaN from lag_N),
                # then each sample needs `sequence_length + forecast_horizon` rows.
                # We need at least ~20 samples for meaningful fine-tuning.
                min_samples = 20
                min_required = self.season_length + self.sequence_length + self.forecast_horizon + min_samples
                partial_window_size = max(self.season_length * 3, min_required)
                partial_data = combined_df.tail(partial_window_size).copy()
                
                if len(partial_data) >= self.sequence_length + self.forecast_horizon:
                    # Load previous full model state
                    self.model.load_state_dict(previous_full_state)
                    
                    # Fine-tune on partial data
                    partial_train_result = self._train_on_data(
                        partial_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate * 0.5,  # Lower LR for fine-tuning
                        validation_split=validation_split,
                    )
                    
                    if partial_train_result["status"] == "completed":
                        # Extract distribution from partial window
                        partial_distribution = self.extract_distribution_features(
                            partial_data["y"].values
                        )
                        
                        # Save PARTIAL checkpoint
                        partial_checkpoint_id = self._save_to_memory(
                            checkpoint_type="partial",
                            data=partial_data,
                            distribution_features=partial_distribution,
                        )
                        logger.info(f"💾 PARTIAL checkpoint saved: {partial_checkpoint_id}")
                        
                        # Store partial model state for potential use
                        partial_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                else:
                    logger.warning("⚠️ Not enough data for partial checkpoint")

            # Choose which model to use for forecasting based on adaptive selection
            model_choice = self._choose_model_for_change(new_df)
            
            if model_choice == "partial" and partial_checkpoint_id is not None and partial_state is not None:
                # Use PARTIAL model for forecasting
                self.model.load_state_dict(partial_state)
                forecast_model_type = "partial"
            else:
                # Use NEW FULL model for forecasting (default behavior)
                self.model.load_state_dict(new_full_state)
                forecast_model_type = "full"
            
            self.latest_full_checkpoint_id = new_full_checkpoint_id

            # Update accumulated data
            self.accumulated_data = combined_df.copy()

            logger.info(f"✅ Distribution change handled - using {forecast_model_type} model for forecast")

            return {
                "action": "distribution_change",
                "training_performed": True,
                "full_checkpoint_id": new_full_checkpoint_id,
                "partial_checkpoint_id": partial_checkpoint_id,
                "full_training_result": full_train_result,
                "forecast_model_type": forecast_model_type,
                "message": f"New checkpoints created, using {forecast_model_type} model for forecast",
            }

    def predict(self, context_df: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        """
        Make predictions for future time steps.
        
        Args:
            context_df: Recent historical data (at least sequence_length rows)
            steps_ahead: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit_initial first.")

        self.model.eval()

        # Prepare context
        context_df = context_df.copy()
        if "unique_id" not in context_df.columns:
            context_df["unique_id"] = "ts_001"

        # Add lagged features
        context_df = create_lagged_features(context_df, lags=[1, self.season_length])

        # Fit preprocessor if needed
        if not self.preprocessor.is_fitted:
            context_df, _ = self.preprocessor.fit_transform(context_df, "y", self.exog_cols)
        else:
            context_df = self.preprocessor.transform(context_df, "y", self.exog_cols)

        # Ensure proper data types
        for col in ["y", "y_scaled"] + self.exog_cols:
            if col in context_df.columns:
                context_df[col] = pd.to_numeric(context_df[col], errors="coerce").fillna(0).astype(np.float64)

        # Build a single sequence from the most recent data (direct multi-horizon prediction)
        X_target_seq, X_exog_seq, _ = prepare_sequences(
            context_df.tail(self.sequence_length + self.forecast_horizon),
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, '_scaled_feature_cols', None),
        )

        if len(X_target_seq) == 0:
            # Fallback: build manual sequence
            vals = (
                context_df["y_scaled"].values
                if "y_scaled" in context_df.columns
                else context_df["y"].values
            )
            vals = np.array(vals, dtype=np.float64)
            if len(vals) == 0:
                return pd.DataFrame({"y_pred": [0.0] * steps_ahead})
            if len(vals) >= self.sequence_length:
                seq = vals[-self.sequence_length:]
            else:
                pad_val = float(vals[0]) if len(vals) > 0 else 0.0
                pad = np.full(self.sequence_length - len(vals), pad_val, dtype=np.float64)
                seq = np.concatenate([pad, vals])
            X_target_seq = np.array([seq.reshape(-1, 1)], dtype=np.float64)
            X_exog_seq = None
        else:
            X_target_seq = np.array(X_target_seq, dtype=np.float64)
            if X_exog_seq is not None:
                X_exog_seq = np.array(X_exog_seq, dtype=np.float64)

        X_target_seq = np.clip(X_target_seq, -5, 5)
        if X_exog_seq is not None:
            X_exog_seq = np.clip(X_exog_seq, -5, 5)

        Xt = torch.FloatTensor(X_target_seq[-1:]).to(self.device)
        Xe = (
            torch.FloatTensor(X_exog_seq[-1:]).to(self.device)
            if X_exog_seq is not None
            else None
        )
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            pred = self.model(Xt, Xe)  # [1, forecast_horizon]
            pred_scaled = pred[0].cpu().numpy()  # [forecast_horizon]

        # Inverse transform to original scale
        predictions = self.preprocessor.inverse_transform_target(pred_scaled)
        predictions = np.array(predictions, dtype=np.float64).flatten()

        # Handle NaN/Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            context_vals = context_df["y"].values[-self.sequence_length:]
            fallback = float(np.nanmean(context_vals)) if len(context_vals) > 0 else 0.0
            predictions = np.where(
                np.isnan(predictions) | np.isinf(predictions),
                fallback,
                predictions,
            )

        # Truncate or pad to requested steps
        if len(predictions) >= steps_ahead:
            predictions = predictions[:steps_ahead]
        else:
            last_val = predictions[-1] if len(predictions) > 0 else 0.0
            pad = np.full(steps_ahead - len(predictions), last_val)
            predictions = np.concatenate([predictions, pad])

        return pd.DataFrame({"y_pred": predictions})

    def _train_model_from_scratch(
        self,
        data: pd.DataFrame,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        validation_split: float,
    ) -> Dict:
        """Train a new model from scratch on the given data."""
        
        # Fit preprocessor
        data_scaled, _ = self.preprocessor.fit_transform(data, "y", self.exog_cols, feature_cols=self.feature_cols)

        # Build scaled feature column list for multivariate input
        self._scaled_feature_cols = None
        if self.feature_cols:
            self._scaled_feature_cols = [f"{c}_scaled" for c in self.feature_cols
                                         if f"{c}_scaled" in data_scaled.columns]
            if "y_scaled" not in self._scaled_feature_cols:
                self._scaled_feature_cols = ["y_scaled"] + self._scaled_feature_cols

        # Prepare sequences
        X_target, X_exog, y = prepare_sequences(
            data_scaled,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, '_scaled_feature_cols', None),
        )

        n_sequences = len(X_target)
        if n_sequences < 2:
            logger.warning(f"⚠️ Not enough sequences ({n_sequences})")
            return {"status": "skipped", "reason": "insufficient_sequences"}

        # Initialize NEW model
        n_exog = X_exog.shape[2] if X_exog is not None else 0
        self.model = self._model_class(
            input_dim=X_target.shape[2] if X_target.ndim == 3 else self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            forecast_horizon=self.forecast_horizon,
            season_length=self.season_length,
            exog_dim=n_exog,
            **self._model_kwargs,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🧠 New model initialized with {total_params} parameters")

        return self._run_training_loop(
            X_target, X_exog, y, epochs, batch_size, learning_rate, validation_split
        )

    def _train_on_data(
        self,
        data: pd.DataFrame,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        validation_split: float,
    ) -> Dict:
        """Continue training existing model on given data (fine-tuning)."""
        
        if self.model is None:
            return {"status": "failed", "reason": "no_model"}

        # Transform data (preprocessor should already be fitted)
        if not self.preprocessor.is_fitted:
            data_scaled, _ = self.preprocessor.fit_transform(data, "y", self.exog_cols)
        else:
            data_scaled = self.preprocessor.transform(data, "y", self.exog_cols)

        # Prepare sequences
        X_target, X_exog, y = prepare_sequences(
            data_scaled,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            exog_cols=self.exog_cols,
            feature_cols=getattr(self, '_scaled_feature_cols', None),
        )

        n_sequences = len(X_target)
        if n_sequences < 2:
            return {"status": "skipped", "reason": "insufficient_sequences"}

        return self._run_training_loop(
            X_target, X_exog, y, epochs, batch_size, learning_rate, validation_split
        )

    def _run_training_loop(
        self,
        X_target: np.ndarray,
        X_exog: Optional[np.ndarray],
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        validation_split: float,
    ) -> Dict:
        """Run the training loop with numerical stability safeguards."""
        
        n_sequences = len(X_target)
        n_train = max(1, int(n_sequences * (1 - validation_split)))
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, n_sequences)) if n_train < n_sequences else []

        # Clamp input data (now scaled to [-1, 1] by MinMaxScaler)
        X_target = np.clip(X_target, -5, 5)
        y = np.clip(y, -5, 5)
        if X_exog is not None:
            X_exog = np.clip(X_exog, -5, 5)

        # Convert to tensors and check for NaN in input data
        X_target_tensor = torch.FloatTensor(X_target).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        X_exog_tensor = (
            torch.FloatTensor(X_exog).to(self.device) if X_exog is not None else None
        )
        
        # Check for NaN/Inf in input tensors
        if torch.isnan(X_target_tensor).any() or torch.isinf(X_target_tensor).any():
            logger.warning("⚠️ NaN/Inf detected in input sequences, replacing with 0")
            X_target_tensor = torch.nan_to_num(X_target_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(y_tensor).any() or torch.isinf(y_tensor).any():
            logger.warning("⚠️ NaN/Inf detected in targets, replacing with 0")
            y_tensor = torch.nan_to_num(y_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Use lower learning rate for stability with large-scale data
        actual_lr = min(learning_rate, 0.0005)
        optimizer = optim.Adam(self.model.parameters(), lr=actual_lr, weight_decay=1e-4, eps=1e-8)
        
        # Learning rate scheduler for stability
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        
        best_val_loss = float("inf")
        best_model_state = None
        nan_count = 0  # Track consecutive NaN losses

        self.model.train()
        for epoch in range(epochs):
            np.random.shuffle(train_idx)
            epoch_losses = []

            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i: i + batch_size]
                batch_target = X_target_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]
                batch_exog = X_exog_tensor[batch_idx] if X_exog_tensor is not None else None

                optimizer.zero_grad()
                predictions = self.model(batch_target, batch_exog)
                loss = regime_aware_loss(batch_y, predictions)
                
                # Check for NaN loss - skip this batch if NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count > 10:
                        logger.warning("⚠️ Too many NaN losses, reinitializing model weights")
                        # Reinitialize model weights with smaller initialization
                        self.model._init_weights()
                        nan_count = 0
                    continue
                
                nan_count = 0  # Reset counter on valid loss
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                # Clamp weights after optimizer step to prevent explosion
                if hasattr(self.model, 'clamp_weights'):
                    self.model.clamp_weights(max_val=2.0)
                
                epoch_losses.append(loss.item())

            # Use median to avoid outlier influence
            avg_train_loss = np.median(epoch_losses) if epoch_losses else float('nan')
            
            # Check if model has NaN weights
            model_has_nan = any(torch.isnan(p).any() for p in self.model.parameters())
            
            # If NaN weights detected, reinitialize
            if model_has_nan:
                logger.warning("⚠️ NaN weights detected, reinitializing model")
                self.model._init_weights()
                model_has_nan = False

            # Validation
            if val_idx:
                self.model.eval()
                with torch.no_grad():
                    val_target = X_target_tensor[val_idx]
                    val_y = y_tensor[val_idx]
                    val_exog = X_exog_tensor[val_idx] if X_exog_tensor is not None else None
                    val_pred = self.model(val_target, val_exog)
                    val_loss = regime_aware_loss(val_y, val_pred).item()

                # Update learning rate based on validation loss
                # Only save model state if weights don't have NaN
                if not np.isnan(val_loss) and not np.isinf(val_loss) and not model_has_nan:
                    scheduler.step(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }

                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.6f}")
                self.model.train()
            else:
                # Without validation, use train loss for scheduler
                # Only save model state if weights don't have NaN
                if not np.isnan(avg_train_loss) and not np.isinf(avg_train_loss) and not model_has_nan:
                    scheduler.step(avg_train_loss)
                    if avg_train_loss < best_val_loss:
                        best_val_loss = avg_train_loss
                        best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}")

        # Restore best model if we have one
        if best_model_state is not None:
            # Final check that saved state doesn't have NaN
            has_nan_in_state = any(torch.isnan(v).any() for v in best_model_state.values())
            if has_nan_in_state:
                logger.warning("⚠️ Best model state contains NaN, reinitializing model")
                self.model.apply(self._init_weights)
            else:
                self.model.load_state_dict(best_model_state)
                logger.info(f"✅ Best model restored (val_loss: {best_val_loss:.4f})")
        else:
            logger.warning("⚠️ No valid model state saved, using final weights")

        return {
            "status": "completed",
            "final_train_loss": avg_train_loss if not np.isnan(avg_train_loss) else 0.0,
            "final_val_loss": best_val_loss if (val_idx and not np.isinf(best_val_loss)) else None,
            "n_sequences": n_sequences,
        }

    def _save_to_memory(
        self,
        checkpoint_type: str,
        data: pd.DataFrame,
        distribution_features: np.ndarray,
    ) -> str:
        """Save checkpoint to memory module with distribution features."""
        
        checkpoint_id = f"{checkpoint_type}_{self.current_checkpoint_id:03d}"
        self.current_checkpoint_id += 1

        weights = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        } if self.model else None

        metadata = {
            "checkpoint_type": checkpoint_type,
            "data_periods": len(data),
            "timestamp_range": (str(data["ds"].min()), str(data["ds"].max())),
            "preprocessor_state": pickle.dumps(self.preprocessor),
        }

        self.memory_module.save_checkpoint(
            weights=weights,
            regime_features=distribution_features,
            metadata=metadata,
            regime_id=checkpoint_id,
        )

        return checkpoint_id

    def _load_checkpoint(self, weights: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Load model weights AND preprocessor from checkpoint."""
        
        if self.model is None:
            # Initialize model structure if not exists
            n_exog = len(self.exog_cols) if self.exog_cols else 0
            self.model = self._model_class(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                forecast_horizon=self.forecast_horizon,
                season_length=self.season_length,
                exog_dim=n_exog,
                **self._model_kwargs,
            ).to(self.device)

        # Load weights
        state_dict = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)
        
        # Load preprocessor from checkpoint metadata if available
        if metadata is not None and 'preprocessor_state' in metadata:
            try:
                self.preprocessor = pickle.loads(metadata['preprocessor_state'])
                logger.info("✅ Model weights AND preprocessor loaded from checkpoint")
            except Exception as e:
                logger.warning(f"⚠️ Could not load preprocessor: {e}. Model weights loaded only.")
        else:
            logger.info("✅ Model weights loaded from checkpoint")

    def _load_model_weights(self, weights: Dict[str, Any]):
        """Load model weights from checkpoint (legacy method for backward compatibility)."""
        self._load_checkpoint(weights, None)

    def _init_weights(self, module):
        """Initialize module weights with Xavier uniform initialization."""
        import torch.nn as nn
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0, 0.1)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def get_memory_info(self) -> Dict[str, Any]:
        """Get information about stored checkpoints."""
        return self.memory_module.get_regime_info()

    def get_regime_info(self) -> Dict[str, Any]:
        """Get regime information (for compatibility with wrapper)."""
        return {
            "current_regime_id": self.latest_full_checkpoint_id,
            "memory_module": self.memory_module,
            "total_checkpoints": len(self.memory_module.checkpoints),
            "checkpoint_ids": self.memory_module.list_checkpoints(),
        }

    def save_state(self, filepath: str):
        """Save complete forecaster state."""
        state = {
            "accumulated_data": self.accumulated_data.to_dict() if self.accumulated_data is not None else None,
            "latest_full_checkpoint_id": self.latest_full_checkpoint_id,
            "current_checkpoint_id": self.current_checkpoint_id,
            "config": {
                "season_length": self.season_length,
                "forecast_horizon": self.forecast_horizon,
                "sequence_length": self.sequence_length,
                "similarity_threshold": self.similarity_threshold,
            },
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"💾 Forecaster state saved to {filepath}")

    # ========== Legacy API support ==========
    
    def fit_incremental(
        self,
        df: pd.DataFrame,
        epochs_per_segment: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Legacy API: Fits on provided data.
        If no prior training, does initial fit.
        Otherwise, treats as new data update.
        """
        if self.accumulated_data is None:
            return self.fit_initial(
                df,
                epochs=epochs_per_segment,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
            )
        else:
            return self.update_with_new_data(
                df,
                epochs=epochs_per_segment,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
            )
