"""
Standard Time Series Benchmark Datasets Loader

Downloads and loads standard benchmarks used in time series forecasting papers:
- ETTh1, ETTh2 (Electricity Transformer Temperature - hourly)
- ETTm1, ETTm2 (Electricity Transformer Temperature - 15-min)
- Weather (21 meteorological indicators)
- Electricity (321 clients' consumption)
- Traffic (862 sensors road occupancy)

Source: https://github.com/thuml/Time-Series-Library
"""

import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency — resolved at first use
_synthetic_module = None


def _get_synthetic_datasets():
    global _synthetic_module
    if _synthetic_module is None:
        from benchmarks.data_loaders.synthetic_regimes import SYNTHETIC_DATASETS, generate_all

        _synthetic_module = (SYNTHETIC_DATASETS, generate_all)
    return _synthetic_module


class StandardBenchmarkLoader:
    """Load standard time series forecasting benchmark datasets."""

    # Dataset URLs - ETT from original source, others need manual download
    DATASET_URLS = {
        "ETT": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/",
        # Weather, Exchange via HuggingFace thuml/Time-Series-Library
        "weather": "hf://thuml/Time-Series-Library/weather/weather.csv",
        "exchange": "hf://thuml/Time-Series-Library/exchange_rate/exchange_rate.csv",
        "electricity": "hf://thuml/Time-Series-Library/electricity/electricity.csv",
        "traffic": "hf://thuml/Time-Series-Library/traffic/traffic.csv",
        "illness": "hf://thuml/Time-Series-Library/illness/national_illness.csv",
    }

    # Priority datasets for publication (ordered by importance)
    PRIORITY_DATASETS = ["ETTh1", "ETTm1", "Weather", "Electricity"]

    # Dataset configurations
    DATASET_CONFIGS = {
        "ETTh1": {
            "file": "ETTh1.csv",
            "freq": "h",
            "target": "OT",
            "features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (hourly) - Station 1",
        },
        "ETTh2": {
            "file": "ETTh2.csv",
            "freq": "h",
            "target": "OT",
            "features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (hourly) - Station 2",
        },
        "ETTm1": {
            "file": "ETTm1.csv",
            "freq": "15min",
            "target": "OT",
            "features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (15-min) - Station 1",
        },
        "ETTm2": {
            "file": "ETTm2.csv",
            "freq": "15min",
            "target": "OT",
            "features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "description": "Electricity Transformer Temperature (15-min) - Station 2",
        },
        "Weather": {
            "file": "weather.csv",
            "freq": "10min",
            "target": "OT",
            "features": None,  # Use all features
            "description": "21 meteorological indicators from Weather Station",
        },
        "Electricity": {
            "file": "electricity.csv",
            "freq": "h",
            "target": "OT",
            "features": None,  # Use all features (321 clients)
            "description": "Electricity consumption of 321 clients",
        },
        "Traffic": {
            "file": "traffic.csv",
            "freq": "h",
            "target": "OT",
            "features": None,  # Use all features (862 sensors)
            "description": "Road occupancy rates from 862 sensors",
        },
        "Exchange": {
            "file": "exchange_rate.csv",
            "freq": "D",
            "target": "OT",
            "features": None,  # 8 country exchange rates
            "description": "Daily exchange rates of 8 countries (Lai et al., 2018)",
        },
        "ILI": {
            "file": "national_illness.csv",
            "freq": "W",
            "target": "OT",
            "features": None,  # 7 illness indicators
            "description": "Weekly influenza-like illness (ILI) patient data",
        },
        # --- Synthetic regime datasets (generated, not downloaded) ---
        "synth_stable": {
            "file": "synth_stable.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: single stable regime (no shifts)",
            "synthetic": True,
        },
        "synth_trend_break": {
            "file": "synth_trend_break.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: one abrupt trend reversal mid-series",
            "synthetic": True,
        },
        "synth_slow_drift": {
            "file": "synth_slow_drift.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: gradual mean drift (no sharp boundary)",
            "synthetic": True,
        },
        "synth_fast_switch": {
            "file": "synth_fast_switch.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: rapid regime switching every ~600 points",
            "synthetic": True,
        },
        "synth_recurring": {
            "file": "synth_recurring.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: recurring A-B-C-A-B-C (checkpoint reuse test)",
            "synthetic": True,
        },
        "synth_volatility": {
            "file": "synth_volatility.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: alternating low/high volatility",
            "synthetic": True,
        },
        "synth_shock_recovery": {
            "file": "synth_shock_recovery.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: stable → shock → recovery → new equilibrium",
            "synthetic": True,
        },
        "synth_multi_regime": {
            "file": "synth_multi_regime.csv",
            "freq": "h",
            "target": "y",
            "features": ["y"],
            "description": "Synthetic: 5+ regimes, varying lengths, some recurring",
            "synthetic": True,
        },
    }

    def __init__(self, data_dir: str = None):
        """
        Initialize the benchmark loader.

        Args:
            data_dir: Directory to store downloaded datasets
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_name: str) -> Path:
        """
        Download a dataset if not already present.

        Args:
            dataset_name: Name of dataset (ETTh1, ETTm1, Weather, etc.)

        Returns:
            Path to the downloaded file
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_CONFIGS.keys())}")

        config = self.DATASET_CONFIGS[dataset_name]
        filename = config["file"]
        filepath = self.data_dir / filename

        if filepath.exists():
            logger.info(f"Dataset {dataset_name} already exists at {filepath}")
            return filepath

        # Synthetic datasets: generate instead of download
        if config.get("synthetic", False):
            logger.info(f"Generating synthetic dataset {dataset_name}...")
            syn_datasets, gen_all = _get_synthetic_datasets()
            if dataset_name in syn_datasets:
                df = syn_datasets[dataset_name]()
                df.to_csv(filepath, index=False)
                logger.info(f"Generated {dataset_name} ({len(df)} rows) at {filepath}")
                return filepath
            else:
                raise ValueError(f"Unknown synthetic dataset: {dataset_name}")

        # Determine URL based on dataset type
        if dataset_name.startswith("ETT"):
            url = self.DATASET_URLS["ETT"] + filename
        elif dataset_name.lower() == "weather":
            url = self.DATASET_URLS["weather"]
        elif dataset_name.lower() == "exchange":
            url = self.DATASET_URLS["exchange"]
        elif dataset_name.lower() == "electricity":
            url = self.DATASET_URLS.get("electricity")
        elif dataset_name.lower() == "traffic":
            url = self.DATASET_URLS.get("traffic")
        elif dataset_name.lower() == "ili":
            url = self.DATASET_URLS.get("illness")
        else:
            raise ValueError(f"No URL configured for dataset: {dataset_name}")

        if url is None:
            raise RuntimeError(f"No download URL for {dataset_name}. Please download manually.")

        # HuggingFace Hub download
        if url.startswith("hf://"):
            logger.info(f"Downloading {dataset_name} from HuggingFace...")
            try:
                from huggingface_hub import hf_hub_download

                parts = url.replace("hf://", "").split("/", 1)
                repo_id = "/".join(parts[0:2]) if "/" in parts[0] else parts[0]
                hf_filename = parts[1] if len(parts) > 1 else filename
                # Parse: hf://thuml/Time-Series-Library/weather/weather.csv
                url_path = url.replace("hf://", "")
                repo_id = "/".join(url_path.split("/")[:2])
                hf_filename = "/".join(url_path.split("/")[2:])
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_filename,
                    repo_type="dataset",
                    local_dir=str(self.data_dir / "_hf_cache"),
                )
                import shutil

                shutil.copy2(downloaded, filepath)
                shutil.rmtree(self.data_dir / "_hf_cache", ignore_errors=True)
                logger.info(f"Downloaded {dataset_name} to {filepath}")
            except Exception as e:
                logger.error(f"HuggingFace download failed for {dataset_name}: {e}")
                raise RuntimeError(f"Could not download {dataset_name}: {e}")
        else:
            logger.info(f"Downloading {dataset_name} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Downloaded {dataset_name} to {filepath}")
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
                raise RuntimeError(f"Could not download {dataset_name}. Please download manually from {url}")

        return filepath

    def load_dataset(
        self, dataset_name: str, target_col: str = None, multivariate: bool = False, normalize: bool = False
    ) -> pd.DataFrame:
        """
        Load a benchmark dataset.

        Args:
            dataset_name: Name of dataset
            target_col: Target column to use (defaults to dataset's default)
            multivariate: If True, return all features; if False, return only target
            normalize: If True, normalize the data

        Returns:
            DataFrame with columns [date, unique_id, ds, y] (and features if multivariate)
        """
        filepath = self.download_dataset(dataset_name)
        config = self.DATASET_CONFIGS[dataset_name]

        # Load raw data
        df = pd.read_csv(filepath)

        # Synthetic datasets already have the right column format
        if config.get("synthetic", False):
            df["ds"] = pd.to_datetime(df["ds"])
            if "unique_id" not in df.columns:
                df["unique_id"] = dataset_name
            result = df[["unique_id", "ds", "y"]].copy()
            if normalize:
                mean, std = result["y"].mean(), result["y"].std()
                if std > 0:
                    result["y"] = (result["y"] - mean) / std
            logger.info(
                f"Loaded {dataset_name}: {len(result)} samples, "
                f"date range: {result['ds'].min()} to {result['ds'].max()}"
            )
            return result

        # Clean sentinel values (e.g. Weather dataset uses -9999 for missing)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            sentinel_mask = df[col] < -9000
            if sentinel_mask.any():
                count = sentinel_mask.sum()
                logger.info(f"  Replacing {count} sentinel values (-9999) in '{col}' with NaN → forward-fill")
                df.loc[sentinel_mask, col] = np.nan
        # Forward-fill then back-fill any NaNs introduced by sentinel replacement
        if df[numeric_cols].isna().any().any():
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Parse date column
        date_col = "date" if "date" in df.columns else df.columns[0]
        df["ds"] = pd.to_datetime(df[date_col])

        # Select target
        target = target_col or config["target"]
        if target not in df.columns:
            # If no specific target, use last column
            target = df.columns[-1] if df.columns[-1] != "ds" else df.columns[-2]

        # Prepare output dataframe
        if multivariate:
            # Keep all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            result = df[["ds"] + numeric_cols].copy()
            result["y"] = df[target]
        else:
            result = pd.DataFrame({"ds": df["ds"], "y": df[target]})

        # Add unique_id for compatibility with our framework
        result["unique_id"] = dataset_name

        # Normalize if requested
        if normalize:
            for col in result.select_dtypes(include=[np.number]).columns:
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std

        # Reorder columns
        cols = ["unique_id", "ds", "y"] + [c for c in result.columns if c not in ["unique_id", "ds", "y"]]
        result = result[cols]

        logger.info(
            f"Loaded {dataset_name}: {len(result)} samples, date range: {result['ds'].min()} to {result['ds'].max()}"
        )

        return result

    def prepare_incremental_batches(
        self,
        dataset_name: str,
        batch_size: int = 96,  # 96 hours = 4 days for hourly data
        initial_train_size: int = 720,  # 30 days for hourly data
        forecast_horizon: int = 24,
        target_col: str = None,
        multivariate: bool = False,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Prepare dataset for incremental learning evaluation.

        Args:
            dataset_name: Name of dataset
            batch_size: Size of each incremental batch
            initial_train_size: Size of initial training set
            forecast_horizon: Forecast horizon
            target_col: Target column
            multivariate: If True, keep all numeric feature columns

        Returns:
            Tuple of (initial_train_df, list_of_batch_dfs)
        """
        df = self.load_dataset(dataset_name, target_col=target_col, multivariate=multivariate)

        # Split into initial training and batches
        initial_df = df.iloc[:initial_train_size].copy()

        remaining = df.iloc[initial_train_size:].copy()
        batches = []

        for i in range(0, len(remaining), batch_size):
            batch = remaining.iloc[i : i + batch_size].copy()
            if len(batch) >= forecast_horizon:  # Only include if we can forecast
                batches.append(batch)

        logger.info(f"Prepared {dataset_name} for incremental learning:")
        logger.info(f"  Initial training: {len(initial_df)} samples")
        logger.info(f"  Number of batches: {len(batches)}")
        logger.info(f"  Batch size: {batch_size}")

        return initial_df, batches

    def get_standard_splits(
        self, dataset_name: str, train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get standard train/val/test splits as used in TSLib papers.

        Args:
            dataset_name: Name of dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = self.load_dataset(dataset_name)
        n = len(df)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(f"Split {dataset_name}:")
        logger.info(f"  Train: {len(train_df)} ({train_ratio * 100:.0f}%)")
        logger.info(f"  Val: {len(val_df)} ({val_ratio * 100:.0f}%)")
        logger.info(f"  Test: {len(test_df)} ({test_ratio * 100:.0f}%)")

        return train_df, val_df, test_df

    def list_available_datasets(self) -> List[Dict]:
        """List all available benchmark datasets with descriptions."""
        datasets = []
        for name, config in self.DATASET_CONFIGS.items():
            filepath = self.data_dir / config["file"]
            datasets.append(
                {
                    "name": name,
                    "description": config["description"],
                    "frequency": config["freq"],
                    "target": config["target"],
                    "downloaded": filepath.exists(),
                }
            )
        return datasets

    def download_all(self) -> None:
        """Download all benchmark datasets."""
        for name in self.DATASET_CONFIGS.keys():
            try:
                self.download_dataset(name)
            except Exception as e:
                logger.warning(f"Could not download {name}: {e}")


def create_regime_shift_scenario(
    df: pd.DataFrame, shift_points: List[int] = None, shift_magnitude: float = 0.3
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Create artificial regime shifts in a dataset for testing regime detection.

    Args:
        df: Input dataframe with 'y' column
        shift_points: Indices where regime shifts occur (auto-detected if None)
        shift_magnitude: Magnitude of artificial shifts

    Returns:
        Tuple of (modified_df, shift_points)
    """
    df = df.copy()

    if shift_points is None:
        # Create shifts at 25%, 50%, 75% of the data
        n = len(df)
        shift_points = [n // 4, n // 2, 3 * n // 4]

    y_std = df["y"].std()

    # Apply shifts
    current_shift = 0
    for i, point in enumerate(shift_points):
        # Alternate between positive and negative shifts
        current_shift = shift_magnitude * y_std * (1 if i % 2 == 0 else -1)
        df.loc[df.index[point:], "y"] += current_shift

    return df, shift_points


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    loader = StandardBenchmarkLoader()

    print("\n=== Available Datasets ===")
    for ds in loader.list_available_datasets():
        status = "✓" if ds["downloaded"] else "✗"
        print(f"  [{status}] {ds['name']}: {ds['description']}")

    print("\n=== Testing ETTh1 Loading ===")
    try:
        df = loader.load_dataset("ETTh1")
        print(f"  Loaded: {len(df)} samples")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")
        print(f"  Sample:\n{df.head()}")

        # Test incremental batches
        print("\n=== Testing Incremental Batch Preparation ===")
        initial, batches = loader.prepare_incremental_batches("ETTh1")
        print(f"  Initial training: {len(initial)} samples")
        print(f"  Number of batches: {len(batches)}")

    except Exception as e:
        print(f"  Error: {e}")
        print("  Note: Download datasets manually if automatic download fails")
