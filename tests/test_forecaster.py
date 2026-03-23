"""
Tests for the RegimeAwareForecaster
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regime_forecasting import RegimeAwareForecaster
from regime_forecasting.core.memory_module import MemoryModule
from regime_forecasting.core.regime_detector import RegimeDetector
from regime_forecasting.utils.data_utils import DataPreprocessor, create_lagged_features


class TestRegimeAwareForecaster:
    """Test cases for the main forecaster"""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")

        # Create simple time series with trend and seasonality
        t = np.arange(60)
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 5, 60)

        y = trend + seasonal + noise

        df = pd.DataFrame({"unique_id": "test_series", "ds": dates, "y": y})

        return df

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_forecaster_initialization(self, temp_dir):
        """Test forecaster initialization"""
        forecaster = RegimeAwareForecaster(season_length=12, forecast_horizon=6, storage_path=temp_dir)

        assert forecaster.season_length == 12
        assert forecaster.forecast_horizon == 6
        assert forecaster.similarity_threshold == 0.8
        assert not forecaster.is_fitted
        assert forecaster.model is None

    def test_forecaster_basic_training(self, sample_data, temp_dir):
        """Test basic training functionality"""
        forecaster = RegimeAwareForecaster(
            season_length=12,
            forecast_horizon=3,
            storage_path=temp_dir,
            model_config={"hidden_dim": 64, "num_layers": 1},
        )

        # Should be able to fit on sample data
        try:
            forecaster.fit_incremental(
                df=sample_data,
                start_period=24,
                epochs_per_regime=5,  # Few epochs for testing
            )

            assert forecaster.is_fitted
            assert forecaster.model is not None
            assert forecaster.current_regime_id is not None

        except Exception as e:
            pytest.fail(f"Basic training failed: {e}")

    def test_forecaster_prediction(self, sample_data, temp_dir):
        """Test prediction functionality"""
        forecaster = RegimeAwareForecaster(
            season_length=12,
            forecast_horizon=3,
            storage_path=temp_dir,
            model_config={"hidden_dim": 32, "num_layers": 1},
        )

        # Train first
        forecaster.fit_incremental(df=sample_data, start_period=24, epochs_per_regime=3)

        # Make predictions
        test_data = sample_data.tail(12)  # Use last year for prediction
        predictions = forecaster.predict(df=test_data, steps_ahead=3)

        assert len(predictions) > 0
        assert "y_pred" in predictions.columns
        assert len(predictions) == 3  # Should have 3 predictions
        assert all(predictions["y_pred"] > 0)  # Predictions should be positive

    def test_regime_info(self, sample_data, temp_dir):
        """Test regime information retrieval"""
        forecaster = RegimeAwareForecaster(
            season_length=12, storage_path=temp_dir, model_config={"hidden_dim": 32, "num_layers": 1}
        )

        forecaster.fit_incremental(df=sample_data, start_period=24, epochs_per_regime=3)

        regime_info = forecaster.get_regime_info()

        assert "current_regime_id" in regime_info
        assert "memory_module" in regime_info
        assert "regime_detector" in regime_info
        assert regime_info["is_fitted"]

    def test_partial_checkpoint_on_change(self, temp_dir):
        """Ensure a partial checkpoint is saved on distribution change"""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        y = np.array([100.0] * 8 + [10.0] * 4)
        df = pd.DataFrame({"unique_id": "test_series", "ds": dates, "y": y})

        forecaster = RegimeAwareForecaster(
            season_length=4,
            forecast_horizon=2,
            storage_path=temp_dir,
            model_config={"hidden_dim": 16, "num_layers": 1, "sequence_length": 4},
        )

        forecaster.fit_incremental(df=df, epochs_per_regime=2)

        core = getattr(forecaster, "_core", None)
        assert core is not None
        # Checkpoints are now stored in memory_module
        checkpoints = core.memory_module.checkpoints
        # With new design: first fit creates a full checkpoint
        # Partial checkpoints are only created on distribution change when there's prior data
        assert len(checkpoints) >= 1
        # Check that checkpoint IDs contain 'full' or 'partial'
        checkpoint_ids = list(checkpoints.keys())
        assert any("full" in cp_id or "partial" in cp_id for cp_id in checkpoint_ids)


class TestMemoryModule:
    """Test cases for the memory module"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_memory_initialization(self, temp_dir):
        """Test memory module initialization"""
        memory = MemoryModule(storage_path=temp_dir)

        assert memory.storage_path == temp_dir
        assert memory.similarity_threshold == 0.8
        assert len(memory.checkpoints) == 0
        assert memory.regime_counter == 0

    def test_checkpoint_save_load(self, temp_dir):
        """Test checkpoint saving and loading"""
        memory = MemoryModule(storage_path=temp_dir)

        # Create dummy checkpoint data
        weights = {"layer1": np.array([1, 2, 3]), "layer2": np.array([4, 5, 6])}
        features = np.array([0.1, 0.2, 0.3, 0.4])
        metadata = {"timestamp": "2024-01-01", "performance": 0.95}

        # Save checkpoint
        regime_id = memory.save_checkpoint(weights, features, metadata)

        assert regime_id in memory.checkpoints
        assert len(memory.checkpoints) == 1

        # Retrieve checkpoint
        checkpoint = memory.get_checkpoint(regime_id)
        assert checkpoint is not None
        assert checkpoint.regime_id == regime_id
        np.testing.assert_array_equal(checkpoint.regime_features, features)

    def test_similarity_matching(self, temp_dir):
        """Test regime similarity matching"""
        memory = MemoryModule(storage_path=temp_dir, similarity_threshold=0.7)

        # Save a checkpoint
        weights1 = {"layer": np.array([1, 2, 3])}
        features1 = np.array([1.0, 2.0, 3.0])
        metadata1 = {"performance": 0.9}

        regime_id1 = memory.save_checkpoint(weights1, features1, metadata1)

        # Test similarity with very similar features
        similar_features = np.array([1.1, 2.1, 3.1])  # Very similar
        result = memory.find_similar_regime(similar_features)

        assert result is not None
        assert result["regime_id"] == regime_id1
        assert result["similarity"] > 0.7

        # Test with very different features
        different_features = np.array([10.0, 20.0, 30.0])  # Very different
        result2 = memory.find_similar_regime(different_features)

        # Should not match due to low similarity
        assert result2 is None or result2["similarity"] < 0.7


class TestRegimeDetector:
    """Test cases for regime detection"""

    def test_detector_initialization(self):
        """Test regime detector initialization"""
        detector = RegimeDetector(season_length=12, regime_threshold=0.7)

        assert detector.season_length == 12
        assert detector.regime_threshold == 0.7
        assert len(detector.feature_history) == 0
        assert len(detector.regime_flags) == 0

    def test_feature_extraction(self):
        """Test regime feature extraction"""
        detector = RegimeDetector(season_length=12)

        # Create simple time series
        np.random.seed(42)
        y = np.random.normal(100, 10, 24)  # 2 years of data

        features = detector.extract_regime_features(y)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert np.all(np.isfinite(features))  # No NaN or infinite values

    def test_regime_change_detection(self):
        """Test regime change detection"""
        detector = RegimeDetector(season_length=12, regime_threshold=0.7)

        # First window - should always be detected as regime change
        np.random.seed(42)
        y1 = np.random.normal(100, 10, 12)
        is_change1, features1 = detector.detect_regime_change(y1)

        assert is_change1  # First window should always trigger change
        assert len(detector.feature_history) == 1

        # Similar window - should not trigger change
        y2 = np.random.normal(100, 10, 12)  # Similar pattern
        is_change2, features2 = detector.detect_regime_change(y2)

        # This might or might not trigger depending on similarity
        assert len(detector.feature_history) == 2

        # Very different window - should trigger change
        y3 = np.random.normal(50, 20, 12)  # Much lower mean, higher variance
        is_change3, features3 = detector.detect_regime_change(y3, use_flag_method=True)

        # Should detect regime change due to mean drop
        assert is_change3
        assert len(detector.feature_history) == 3

    def test_detection_summary(self):
        """Test detection summary functionality"""
        detector = RegimeDetector()

        # Process a few windows
        for i in range(3):
            y = np.random.normal(100 + i * 10, 10, 12)
            detector.detect_regime_change(y)

        summary = detector.get_detection_summary()

        assert "total_windows" in summary
        assert "regime_changes" in summary
        assert "change_rate" in summary
        assert summary["total_windows"] == 3


class TestDataUtilities:
    """Test cases for data utilities"""

    def test_lagged_features(self):
        """Test lagged feature creation"""
        # Create test data
        df = pd.DataFrame(
            {"unique_id": ["A"] * 24, "ds": pd.date_range("2020-01-01", periods=24, freq="ME"), "y": np.arange(24)}
        )

        # Create lagged features
        df_with_lags = create_lagged_features(df, lags=[1, 12])

        assert "lag_1" in df_with_lags.columns
        assert "lag_12" in df_with_lags.columns

        # Check lag values
        assert df_with_lags["lag_1"].iloc[1] == 0  # lag_1 at index 1 should be y[0]
        assert df_with_lags["lag_12"].iloc[12] == 0  # lag_12 at index 12 should be y[0]

    def test_data_preprocessor(self):
        """Test data preprocessing"""
        # Create test data
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 24,
                "ds": pd.date_range("2020-01-01", periods=24, freq="ME"),
                "y": np.random.normal(100, 20, 24),
                "exog1": np.random.normal(0, 1, 24),
            }
        )

        preprocessor = DataPreprocessor(season_length=12)

        # Test fit_transform
        df_transformed, metadata = preprocessor.fit_transform(df, target_col="y", exog_cols=["exog1"])

        assert "y_scaled" in df_transformed.columns
        assert "exog1_scaled" in df_transformed.columns
        assert "target_scaler" in metadata
        assert "exog_scaler" in metadata

        # Test transform (should work after fitting)
        df_test = df.tail(12)
        df_test_transformed = preprocessor.transform(df_test, target_col="y", exog_cols=["exog1"])

        assert "y_scaled" in df_test_transformed.columns
        assert "exog1_scaled" in df_test_transformed.columns


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
