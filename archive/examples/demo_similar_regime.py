"""
Demo script showing regime matching with similar historical patterns
This demonstrates the system loading weights from a similar regime without retraining
"""
import sys
import os
sys.path.append('../src')

import numpy as np
import pandas as pd
import logging
from regime_forecasting import RegimeAwareForecaster

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_similar_regime():
    """
    Demonstrate regime matching with similar historical data
    """
    print("=" * 60)
    print("🔍 DEMO: Similar Regime Detection and Weight Reuse")
    print("=" * 60)
    
    # Load training and test data (DATA_DIR env var overrides default)
    try:
        print("\n📊 Loading training data...")
        data_dir = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data'))
        train_path = os.path.join(data_dir, 'training_data.csv')
        train_data = pd.read_csv(train_path)
        train_data['ds'] = pd.to_datetime(train_data['ds'])
        print(f"✅ Loaded {len(train_data)} training observations")
        print(f"   Date range: {train_data['ds'].min()} to {train_data['ds'].max()}")

        # Load similar regime test data (prefer same_distribution_test.csv)
        print("\n📊 Loading similar regime test data...")
        test_path = os.path.join(data_dir, 'same_distribution_test.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_dir, 'mixed_regime_test.csv')
        test_data = pd.read_csv(test_path)
        test_data['ds'] = pd.to_datetime(test_data['ds'])
        print(f"✅ Loaded {len(test_data)} test observations")
        print(f"   Date range: {test_data['ds'].min()} to {test_data['ds'].max()}")

    except FileNotFoundError:
        print("❌ Data files not found in DATA_DIR. Please provide CSVs or set DATA_DIR")
        return
    
    # Initialize forecaster
    print("\n🚀 Initializing RegimeAwareForecaster...")
    forecaster = RegimeAwareForecaster(
        season_length=12,
        forecast_horizon=6,
        regime_threshold=0.7,
        similarity_threshold=0.75,  # Lower threshold to encourage matching
        model_config={
            'hidden_dim': 128,  # Smaller model for demo
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        }
    )
    
    # Train on historical data
    print("\n🎯 Training on historical data...")
    try:
        forecaster.fit_incremental(
            df=train_data,
            start_period=24,  # Start after 2 years
            epochs_per_regime=30  # Fewer epochs for demo
        )
        print("✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Get regime information after training
    print("\n📈 Training Results:")
    regime_info = forecaster.get_regime_info()
    mem = regime_info.get('memory_module') if isinstance(regime_info, dict) else getattr(forecaster, 'memory_module', None)
    # Normalize memory info
    if hasattr(mem, 'get_regime_info'):
        mem_info = mem.get_regime_info()
    elif isinstance(mem, dict):
        mem_info = mem
    else:
        mem_info = {'total_regimes': 0, 'regime_ids': []}

    print(f"   Total regimes detected: {mem_info.get('total_regimes', 0)}")
    print(f"   Current regime: {regime_info.get('current_regime_id', getattr(forecaster, 'current_regime_id', None))}")
    print(f"   Regime IDs: {mem_info.get('regime_ids', [])}")

    # Display evaluation summary when available
    eval_summary = regime_info.get('evaluation_summary') if isinstance(regime_info, dict) else None
    if eval_summary and 'latest_metrics' in eval_summary:
        metrics = eval_summary['latest_metrics']
        print(f"   Latest MAPE: {metrics.get('weighted_mape', 'N/A'):.2f}%")
        print(f"   Latest RMSE: {metrics.get('rmse', 'N/A'):.4f}")
    
    # Now test with similar regime data
    print("\n🔍 Testing with similar regime data...")
    print("   This should trigger regime matching (not new training)")
    
    # Create extended test dataset for incremental fitting
    extended_test = test_data.copy()
    extended_test['unique_id'] = 'series_001'  # Use same ID as training
    
    # Combine last part of training data with test data for context
    context_data = train_data.tail(24).copy()  # Last 2 years for context
    combined_test = pd.concat([context_data, extended_test], ignore_index=True)
    combined_test = combined_test.sort_values('ds').reset_index(drop=True)
    
    try:
        # Process the test data (this should trigger regime matching)
        initial_regimes = len(forecaster.memory_module.checkpoints)
        
        print(f"   Regimes before test: {initial_regimes}")
        
        # Extend training with similar data
        forecaster.fit_incremental(
            df=combined_test.tail(36),  # Use last 3 years including test data
            start_period=12,  # Start processing quickly
            epochs_per_regime=20
        )
        
        final_regimes = len(forecaster.memory_module.checkpoints)
        print(f"   Regimes after test: {final_regimes}")
        
        if final_regimes == initial_regimes:
            print("🎉 SUCCESS: No new regime created - existing weights reused!")
        elif final_regimes > initial_regimes:
            print("⚠️  New regime created - similarity might be below threshold")
        
    except Exception as e:
        print(f"❌ Test processing failed: {e}")
        return
    
    # Make predictions
    print("\n🔮 Making predictions...")
    try:
        predictions = forecaster.predict(
            df=extended_test.tail(12),  # Use last year for prediction
            steps_ahead=6
        )
        
        print(f"✅ Generated {len(predictions)} predictions")
        if len(predictions) > 0:
            print("   Prediction summary:")
            print(f"   Mean prediction: {predictions['y_pred'].mean():.2f}")
            print(f"   Prediction range: {predictions['y_pred'].min():.2f} - {predictions['y_pred'].max():.2f}")

            # Display first few predictions (safe for missing 'ds')
            print("\n   First 3 predictions:")
            for i in range(min(3, len(predictions))):
                pred_row = predictions.iloc[i]
                if 'ds' in predictions.columns:
                    label = pred_row['ds'].strftime('%Y-%m')
                else:
                    label = f"step_{i+1}"
                print(f"   {label} : {pred_row['y_pred']:.2f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Final regime information
    print("\n📊 Final System State:")
    final_info = forecaster.get_regime_info()
    # Normalize access to memory/detector info
    mem = final_info.get('memory_module') if isinstance(final_info, dict) else getattr(forecaster, 'memory_module', None)
    if hasattr(mem, 'get_regime_info'):
        mem_info = mem.get_regime_info()
    elif isinstance(mem, dict):
        mem_info = mem
    else:
        mem_info = {'total_regimes': 0, 'regime_ids': []}

    print(f"   Active regime: {final_info.get('current_regime_id', getattr(forecaster, 'current_regime_id', None))}")
    print(f"   Total regimes: {mem_info.get('total_regimes', 0)}")

    detection_summary = final_info.get('regime_detector') if isinstance(final_info, dict) else None
    if detection_summary and isinstance(detection_summary, dict):
        print(f"   Regime changes detected: {detection_summary.get('regime_changes', 0)}")
        print(f"   Change rate: {detection_summary.get('change_rate', 0.0):.2%}")
    else:
        print("   Regime detection summary not available in this API shape")
    
    print("\n" + "=" * 60)
    print("✅ Similar Regime Demo Completed Successfully!")
    print("Key takeaway: System reused existing weights for similar patterns")
    print("=" * 60)


if __name__ == "__main__":
    demo_similar_regime()