"""
Generate synthetic time series data with regime changes
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_timeseries(
    start_date: str = "2000-01-01",
    periods: int = 240,  # 20 years of monthly data
    regime_changes: List[int] = None,
    unique_ids: List[str] = None,
    noise_level: float = 0.1,
    seasonal_strength: float = 0.3
) -> pd.DataFrame:
    """
    Generate synthetic time series with multiple regimes
    
    Args:
        start_date: Starting date for the time series
        periods: Number of time periods
        regime_changes: List of periods where regimes change
        unique_ids: List of series identifiers
        noise_level: Amount of random noise to add
        seasonal_strength: Strength of seasonal patterns
        
    Returns:
        DataFrame with columns ['unique_id', 'ds', 'y']
    """
    if regime_changes is None:
        regime_changes = [48, 96, 144, 192]  # Changes every 4 years
    
    if unique_ids is None:
        unique_ids = ['series_001']
    
    # Define different regime patterns
    regimes = {
        'stable_growth': {
            'trend': 0.02,
            'seasonal_amplitude': 10,
            'volatility': 0.05,
            'base_level': 100
        },
        'high_volatility': {
            'trend': 0.01,
            'seasonal_amplitude': 25,
            'volatility': 0.15,
            'base_level': 120
        },
        'declining': {
            'trend': -0.01,
            'seasonal_amplitude': 8,
            'volatility': 0.08,
            'base_level': 90
        },
        'explosive_growth': {
            'trend': 0.05,
            'seasonal_amplitude': 15,
            'volatility': 0.12,
            'base_level': 110
        },
        'recession': {
            'trend': -0.03,
            'seasonal_amplitude': 5,
            'volatility': 0.20,
            'base_level': 70
        }
    }
    
    regime_sequence = ['stable_growth', 'high_volatility', 'declining', 'explosive_growth', 'recession']
    
    all_data = []
    
    for uid in unique_ids:
        logger.info(f"Generating data for {uid}")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods, freq='ME')
        
        # Initialize series
        y_values = []
        current_regime_idx = 0
        regime_periods = [0] + regime_changes + [periods]
        
        for period in range(periods):
            # Check if we need to switch regime
            if period in regime_changes:
                current_regime_idx += 1
                if current_regime_idx >= len(regime_sequence):
                    current_regime_idx = 0  # Cycle back
            
            regime_name = regime_sequence[current_regime_idx]
            regime_params = regimes[regime_name]
            
            # Calculate components
            t = period
            
            # Trend component
            trend = regime_params['base_level'] + regime_params['trend'] * t
            
            # Seasonal component (yearly + some higher frequency)
            seasonal = (
                regime_params['seasonal_amplitude'] * seasonal_strength * 
                (np.sin(2 * np.pi * t / 12) + 
                 0.3 * np.sin(2 * np.pi * t / 6) +
                 0.1 * np.sin(2 * np.pi * t / 3))
            )
            
            # Noise component
            noise = np.random.normal(0, regime_params['volatility'] * regime_params['base_level'])
            
            # Combine components
            y_value = trend + seasonal + noise
            
            # Ensure positive values
            y_value = max(y_value, 1.0)
            
            y_values.append(y_value)
        
        # Create DataFrame for this series
        series_data = pd.DataFrame({
            'unique_id': uid,
            'ds': dates,
            'y': y_values
        })
        
        all_data.append(series_data)
    
    # Combine all series
    final_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Generated {len(final_df)} observations across {len(unique_ids)} series")
    logger.info(f"Regime changes at periods: {regime_changes}")
    
    return final_df


def generate_test_data_similar_regime(
    base_data: pd.DataFrame,
    reference_regime_period: int = 48,  # Reference to second regime (high_volatility)
    test_periods: int = 24
) -> pd.DataFrame:
    """
    Generate test data similar to a historical regime for testing regime matching
    """
    # Get parameters similar to the reference regime (high_volatility)
    similar_regime_params = {
        'trend': 0.012,  # Slightly different but similar
        'seasonal_amplitude': 23,  # Close to 25
        'volatility': 0.14,  # Close to 0.15
        'base_level': 125  # Close to 120
    }
    
    last_date = base_data['ds'].max()
    start_date = last_date + timedelta(days=30)
    
    dates = pd.date_range(start=start_date, periods=test_periods, freq='ME')
    y_values = []
    
    for i in range(test_periods):
        t = reference_regime_period + i  # Continue from reference regime
        
        # Similar pattern to high_volatility regime
        trend = similar_regime_params['base_level'] + similar_regime_params['trend'] * t
        
        seasonal = (
            similar_regime_params['seasonal_amplitude'] * 0.3 * 
            (np.sin(2 * np.pi * t / 12) + 
             0.3 * np.sin(2 * np.pi * t / 6) +
             0.1 * np.sin(2 * np.pi * t / 3))
        )
        
        noise = np.random.normal(0, similar_regime_params['volatility'] * similar_regime_params['base_level'])
        
        y_value = max(trend + seasonal + noise, 1.0)
        y_values.append(y_value)
    
    test_df = pd.DataFrame({
        'unique_id': 'test_similar',
        'ds': dates,
        'y': y_values
    })
    
    logger.info(f"Generated {len(test_df)} test observations similar to historical regime")
    
    return test_df


def generate_test_data_new_regime(
    base_data: pd.DataFrame,
    test_periods: int = 24
) -> pd.DataFrame:
    """
    Generate test data with completely new regime pattern
    """
    # Completely new regime pattern
    new_regime_params = {
        'trend': 0.08,  # Much higher growth
        'seasonal_amplitude': 50,  # Much higher seasonality
        'volatility': 0.25,  # High volatility
        'base_level': 200,  # Much higher base level
        'cycle_length': 18  # Different seasonal cycle
    }
    
    last_date = base_data['ds'].max()
    start_date = last_date + timedelta(days=30)
    
    dates = pd.date_range(start=start_date, periods=test_periods, freq='ME')
    y_values = []
    
    for i in range(test_periods):
        t = i
        
        # New regime pattern with different characteristics
        trend = new_regime_params['base_level'] + new_regime_params['trend'] * t
        
        # Different seasonal pattern
        seasonal = (
            new_regime_params['seasonal_amplitude'] * 0.4 * 
            (np.sin(2 * np.pi * t / new_regime_params['cycle_length']) + 
             0.5 * np.cos(2 * np.pi * t / 9) +
             0.2 * np.sin(4 * np.pi * t / new_regime_params['cycle_length']))
        )
        
        noise = np.random.normal(0, new_regime_params['volatility'] * new_regime_params['base_level'])
        
        y_value = max(trend + seasonal + noise, 1.0)
        y_values.append(y_value)
    
    test_df = pd.DataFrame({
        'unique_id': 'test_new_regime',
        'ds': dates,
        'y': y_values
    })
    
    logger.info(f"Generated {len(test_df)} test observations with new regime pattern")
    
    return test_df


def plot_regime_data(df: pd.DataFrame, regime_changes: List[int] = None, save_path: str = None):
    """
    Plot the generated time series data with regime boundaries
    """
    if regime_changes is None:
        regime_changes = [48, 96, 144, 192]
    
    plt.figure(figsize=(15, 8))
    
    for uid in df['unique_id'].unique():
        series_data = df[df['unique_id'] == uid].sort_values('ds')
        plt.plot(series_data['ds'], series_data['y'], label=uid, linewidth=1.5)
    
    # Add regime change lines
    if regime_changes:
        start_date = df['ds'].min()
        for change_period in regime_changes:
            change_date = start_date + pd.DateOffset(months=change_period)
            if change_date <= df['ds'].max():
                plt.axvline(x=change_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.title('Synthetic Time Series with Regime Changes', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate main training data
    print("Generating main training data...")
    main_data = generate_synthetic_timeseries(
        start_date="2000-01-01",
        periods=240,  # 20 years
        regime_changes=[48, 96, 144, 192],  # Every 4 years
        unique_ids=['series_001']
    )
    
    # Generate test data - similar to historical regime
    print("Generating similar regime test data...")
    similar_test = generate_test_data_similar_regime(main_data, reference_regime_period=48)
    
    # Generate test data - completely new regime
    print("Generating new regime test data...")
    new_test = generate_test_data_new_regime(main_data)
    
    # Save datasets
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    main_data.to_csv(data_dir / "sample_data.csv", index=False)
    similar_test.to_csv(data_dir / "similar_regime_test.csv", index=False)
    new_test.to_csv(data_dir / "new_regime_test.csv", index=False)
    
    print(f"Main data shape: {main_data.shape}")
    print(f"Similar test shape: {similar_test.shape}")
    print(f"New regime test shape: {new_test.shape}")
    
    # Plot the data
    plot_regime_data(main_data, save_path=str(data_dir / "regime_plot.png"))
    
    print("Data generation complete!")
    print("Files saved:")
    print(f"- {data_dir / 'sample_data.csv'} (main training data)")
    print(f"- {data_dir / 'similar_regime_test.csv'} (similar regime test)")
    print(f"- {data_dir / 'new_regime_test.csv'} (new regime test)")
    print(f"- {data_dir / 'regime_plot.png'} (visualization)")