"""
Data utilities for preprocessing and decomposition
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles normalization, decomposition, and feature engineering.

    Supports both univariate and multivariate modes:
      - Univariate: scales target column only.
      - Multivariate: also scales additional feature columns
        (stored as ``{col}_scaled`` in the DataFrame).
    """
    
    def __init__(self, season_length: int = 12):
        self.season_length = season_length
        # Use MinMaxScaler to ensure bounded values for numerical stability
        # This scales data to [0, 1] range which works well with neural networks
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.exog_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.is_fitted = False
        self._feature_cols: Optional[list] = None  # multivariate feature columns
        
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'y',
        exog_cols: Optional[list] = None,
        feature_cols: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit scalers and transform data.
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            exog_cols: List of exogenous variable columns (lag features)
            feature_cols: List of additional multivariate feature columns
                to scale.  Each column ``c`` gets a ``c_scaled`` column.
                These are the raw dataset features (e.g. HUFL, HULL, etc.).
            
        Returns:
            Transformed dataframe and transformation metadata
        """
        df_transformed = df.copy()
        metadata = {}
        
        # Transform target variable
        target_values = df[target_col].values.reshape(-1, 1)
        df_transformed[f'{target_col}_scaled'] = self.target_scaler.fit_transform(target_values).flatten()
        
        # MinMaxScaler uses data_min_, data_max_, and scale_
        metadata['target_scaler'] = {
            'min': float(self.target_scaler.data_min_[0]),
            'max': float(self.target_scaler.data_max_[0]),
            'scale': float(self.target_scaler.scale_[0])
        }
        
        # Transform exogenous variables (lag features)
        if exog_cols:
            exog_values = df[exog_cols].values
            df_transformed[[f'{col}_scaled' for col in exog_cols]] = self.exog_scaler.fit_transform(exog_values)
            
            metadata['exog_scaler'] = {
                'min': self.exog_scaler.data_min_.tolist(),
                'max': self.exog_scaler.data_max_.tolist(),
                'scale': self.exog_scaler.scale_.tolist()
            }
        
        # Transform multivariate feature columns (if provided)
        if feature_cols:
            # Only scale columns that exist and are not the target
            valid_feat_cols = [c for c in feature_cols if c in df.columns and c != target_col]
            if valid_feat_cols:
                feat_values = df[valid_feat_cols].values
                scaled = self.feature_scaler.fit_transform(feat_values)
                for j, col in enumerate(valid_feat_cols):
                    df_transformed[f'{col}_scaled'] = scaled[:, j]
                self._feature_cols = valid_feat_cols
                metadata['feature_scaler'] = {
                    'columns': valid_feat_cols,
                    'min': self.feature_scaler.data_min_.tolist(),
                    'max': self.feature_scaler.data_max_.tolist(),
                    'scale': self.feature_scaler.scale_.tolist(),
                }
        
        self.is_fitted = True
        logger.info("📊 Data normalization completed")
        
        return df_transformed, metadata
    
    def transform(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'y',
        exog_cols: Optional[list] = None
    ) -> pd.DataFrame:
        """Transform new data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        df_transformed = df.copy()
        
        # Transform target
        target_values = df[target_col].values.reshape(-1, 1)
        df_transformed[f'{target_col}_scaled'] = self.target_scaler.transform(target_values).flatten()
        
        # Transform exogenous variables (lag features)
        if exog_cols:
            exog_values = df[exog_cols].values
            df_transformed[[f'{col}_scaled' for col in exog_cols]] = self.exog_scaler.transform(exog_values)
        
        # Transform multivariate feature columns (if fitted)
        if self._feature_cols:
            valid_feat_cols = [c for c in self._feature_cols if c in df.columns]
            if valid_feat_cols:
                feat_values = df[valid_feat_cols].values
                scaled = self.feature_scaler.transform(feat_values)
                for j, col in enumerate(valid_feat_cols):
                    df_transformed[f'{col}_scaled'] = scaled[:, j]
        
        return df_transformed

    def update_scaler_range(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        exog_cols: Optional[list] = None,
    ) -> bool:
        """Incrementally expand scaler range when new data exceeds fitted bounds.

        Only *expands* — never shrinks.  This prevents scaling artefacts when
        streaming data drifts beyond the initial training range, while keeping
        the learned mapping stable for values already seen.

        Returns True if any scaler was updated, False otherwise.
        """
        if not self.is_fitted:
            return False

        updated = False

        # --- Target scaler ---
        new_vals = df[target_col].dropna().values
        if len(new_vals) > 0:
            new_min = float(np.min(new_vals))
            new_max = float(np.max(new_vals))
            old_min = float(self.target_scaler.data_min_[0])
            old_max = float(self.target_scaler.data_max_[0])
            if new_min < old_min or new_max > old_max:
                exp_min = min(old_min, new_min)
                exp_max = max(old_max, new_max)
                rng = exp_max - exp_min
                if rng > 1e-8:
                    self.target_scaler.data_min_[0] = exp_min
                    self.target_scaler.data_max_[0] = exp_max
                    self.target_scaler.data_range_[0] = rng
                    # feature_range = (-1, 1) → scale = 2/range, min_ = -1 - data_min*scale
                    self.target_scaler.scale_[0] = 2.0 / rng
                    self.target_scaler.min_[0] = -1.0 - exp_min * self.target_scaler.scale_[0]
                    updated = True
                    logger.info(f"📏 Scaler updated: y range [{old_min:.2f}, {old_max:.2f}] "
                                f"→ [{exp_min:.2f}, {exp_max:.2f}]")

        # --- Exog scaler ---
        if exog_cols:
            valid = [c for c in exog_cols if c in df.columns]
            if valid:
                vals = df[valid].values
                for j, col in enumerate(valid):
                    if j < len(self.exog_scaler.data_min_):
                        col_vals = vals[:, j]
                        col_vals = col_vals[np.isfinite(col_vals)]
                        if len(col_vals) == 0:
                            continue
                        new_min_j = float(np.min(col_vals))
                        new_max_j = float(np.max(col_vals))
                        old_min_j = float(self.exog_scaler.data_min_[j])
                        old_max_j = float(self.exog_scaler.data_max_[j])
                        if new_min_j < old_min_j or new_max_j > old_max_j:
                            exp_min_j = min(old_min_j, new_min_j)
                            exp_max_j = max(old_max_j, new_max_j)
                            rng_j = exp_max_j - exp_min_j
                            if rng_j > 1e-8:
                                self.exog_scaler.data_min_[j] = exp_min_j
                                self.exog_scaler.data_max_[j] = exp_max_j
                                self.exog_scaler.data_range_[j] = rng_j
                                self.exog_scaler.scale_[j] = 2.0 / rng_j
                                self.exog_scaler.min_[j] = -1.0 - exp_min_j * self.exog_scaler.scale_[j]
                                updated = True

        # --- Feature scaler (multivariate columns) ---
        if self._feature_cols:
            valid_feat = [c for c in self._feature_cols if c in df.columns]
            if valid_feat:
                vals = df[valid_feat].values
                for j, col in enumerate(valid_feat):
                    if j < len(self.feature_scaler.data_min_):
                        col_vals = vals[:, j]
                        col_vals = col_vals[np.isfinite(col_vals)]
                        if len(col_vals) == 0:
                            continue
                        new_min_j = float(np.min(col_vals))
                        new_max_j = float(np.max(col_vals))
                        old_min_j = float(self.feature_scaler.data_min_[j])
                        old_max_j = float(self.feature_scaler.data_max_[j])
                        if new_min_j < old_min_j or new_max_j > old_max_j:
                            exp_min_j = min(old_min_j, new_min_j)
                            exp_max_j = max(old_max_j, new_max_j)
                            rng_j = exp_max_j - exp_min_j
                            if rng_j > 1e-8:
                                self.feature_scaler.data_min_[j] = exp_min_j
                                self.feature_scaler.data_max_[j] = exp_max_j
                                self.feature_scaler.data_range_[j] = rng_j
                                self.feature_scaler.scale_[j] = 2.0 / rng_j
                                self.feature_scaler.min_[j] = -1.0 - exp_min_j * self.feature_scaler.scale_[j]
                                updated = True

        return updated

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target predictions"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")
            
        y_scaled_reshaped = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled_reshaped).flatten()
    
    def decompose_series(
        self, 
        y: np.ndarray, 
        model: str = 'additive'
    ) -> Dict[str, np.ndarray]:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            y: Time series values
            model: 'additive' or 'multiplicative'
            
        Returns:
            Dictionary with trend, seasonal, residual components
        """
        if len(y) < 2 * self.season_length:
            logger.warning(f"⚠️ Series too short for decomposition. Need at least {2 * self.season_length} points, got {len(y)}")
            return {
                'trend': np.full_like(y, np.mean(y)),
                'seasonal': np.zeros_like(y),
                'residual': y - np.mean(y)
            }
        
        try:
            # Create a temporary series with proper index
            ts = pd.Series(y, index=pd.date_range('2020-01-01', periods=len(y), freq='ME'))
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                ts, 
                model=model, 
                period=self.season_length,
                extrapolate_trend='freq'
            )
            
            return {
                'trend': decomposition.trend.bfill().ffill().values,
                'seasonal': decomposition.seasonal.fillna(0).values,
                'residual': decomposition.resid.fillna(0).values
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Decomposition failed: {e}. Using simple trend estimation.")
            
            # Fallback: simple linear trend
            x = np.arange(len(y))
            trend = np.polyval(np.polyfit(x, y, 1), x)
            residual = y - trend
            
            return {
                'trend': trend,
                'seasonal': np.zeros_like(y),
                'residual': residual
            }


def create_lagged_features(
    df: pd.DataFrame, 
    target_col: str = 'y', 
    lags: Optional[list] = None,
    season_length: int = 12
) -> pd.DataFrame:
    """
    Create lagged features for exogenous variables
    
    Args:
        df: Input dataframe with time series data
        target_col: Name of target column
        lags: List of lag values to create
        season_length: Seasonal period length
        
    Returns:
        DataFrame with additional lagged features
    """
    if lags is None:
        lags = [1, season_length]  # Default: 1-step and seasonal lags
    
    df_with_lags = df.copy()
    lag_cols = [f'lag_{lag}' for lag in lags]
    
    # Initialize lag columns
    for lag_col in lag_cols:
        df_with_lags[lag_col] = np.nan
    
    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds').reset_index(drop=True)
        group_indices = df_with_lags[df_with_lags['unique_id'] == uid].index
        
        for lag in lags:
            lag_col = f'lag_{lag}'
            lagged_values = group[target_col].shift(lag)
            df_with_lags.loc[group_indices, lag_col] = lagged_values.values
    
    # Forward fill initial NaN values
    for lag_col in lag_cols:
        df_with_lags[lag_col] = df_with_lags.groupby('unique_id')[lag_col].bfill()
    
    logger.info(f"📈 Created lagged features: {lag_cols}")
    
    return df_with_lags


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 24,
    forecast_horizon: int = 6,
    target_col: str = 'y_scaled',
    exog_cols: Optional[list] = None,
    feature_cols: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare sequences for training the transformer model.

    Supports both univariate and multivariate modes:
      - **Univariate** (feature_cols=None): X_target has shape (N, seq_len, 1).
      - **Multivariate** (feature_cols=['A','B',...]): X_target has shape
        (N, seq_len, len(feature_cols)).  The first column in feature_cols
        should be the scaled target (e.g. 'y_scaled'), but all columns are
        included as input channels so that models like iTransformer and
        PatchTST can leverage cross-variate information.
    
    Args:
        df: Processed dataframe
        sequence_length: Length of input sequences
        forecast_horizon: Number of steps to forecast
        target_col: Name of scaled target column (used for y labels)
        exog_cols: List of exogenous variable column names (lag features)
        feature_cols: List of scaled multivariate feature columns to use
            as the model's input channels.  If None, falls back to
            univariate (target_col only).  When provided, exog_cols are
            still used as additional exogenous context if present.
        
    Returns:
        Tuple of (X_target, X_exog, y) arrays
        - X_target: (N, seq_len, input_dim)  where input_dim = len(feature_cols) or 1
        - X_exog:   (N, seq_len, n_exog) or None
        - y:        (N, forecast_horizon)
    """
    X_target_list = []
    X_exog_list = [] 
    y_list = []
    
    # Determine which columns form the multivariate target input
    if feature_cols is not None and len(feature_cols) > 0:
        # Multivariate mode: use all specified feature columns
        use_feature_cols = [c for c in feature_cols if c in df.columns]
        if not use_feature_cols:
            use_feature_cols = None
    else:
        use_feature_cols = None
    
    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds').reset_index(drop=True)
        
        # Target values for y labels (always univariate — the column to forecast)
        target_values = group[target_col].values
        
        # Input features — univariate or multivariate
        if use_feature_cols is not None:
            input_values = group[use_feature_cols].values  # (T, n_features)
        else:
            input_values = None  # will use target_values reshaped to (-1, 1)
        
        if exog_cols:
            exog_values = group[exog_cols].values
        else:
            exog_values = None
        
        # Create sequences
        for i in range(len(group) - sequence_length - forecast_horizon + 1):
            # Input sequence
            if input_values is not None:
                X_target_seq = input_values[i:i + sequence_length]  # (seq_len, n_features)
            else:
                X_target_seq = target_values[i:i + sequence_length].reshape(-1, 1)
            X_target_list.append(X_target_seq)
            
            # Exogenous sequence
            if exog_values is not None:
                X_exog_seq = exog_values[i:i + sequence_length]
                X_exog_list.append(X_exog_seq)
            
            # Target forecast (always univariate)
            y_seq = target_values[i + sequence_length:i + sequence_length + forecast_horizon]
            y_list.append(y_seq)
    
    # Determine input_dim for the empty array fallback
    if use_feature_cols is not None:
        input_dim = len(use_feature_cols)
    else:
        input_dim = 1
    
    X_target = np.array(X_target_list) if X_target_list else np.empty((0, sequence_length, input_dim))
    X_exog = np.array(X_exog_list) if X_exog_list else None
    y = np.array(y_list) if y_list else np.empty((0, forecast_horizon))
    
    logger.info(f"📝 Created {len(X_target)} sequences (seq_len: {sequence_length}, "
                f"forecast: {forecast_horizon}, input_dim: {input_dim})")
    
    return X_target, X_exog, y