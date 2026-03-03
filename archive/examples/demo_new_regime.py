"""
Demo script showing detection and handling of completely new regimes
This demonstrates the system creating new weights when no similar patterns exist
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


def demo_new_regime():
    """
    Demonstrate new regime detection and weight initialization
    """
    print("=" * 60)
    print("🆕 DEMO: New Regime Detection and Training")
    print("=" * 60)
    
    # Load training and new-regime test data (use DATA_DIR env var if set)
    try:
        print("\n📊 Loading training data...")
        data_dir = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data'))
        train_path = os.path.join(data_dir, 'training_data.csv')
        train_data = pd.read_csv(train_path)
        train_data['ds'] = pd.to_datetime(train_data['ds'])
        print(f"✅ Loaded {len(train_data)} training observations")

        # Load new regime test data (prefer different_distribution_test.csv)
        print("\n📊 Loading new regime test data...")
        test_path = os.path.join(data_dir, 'different_distribution_test.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_dir, 'test_diff_future_actuals.csv')
        test_data = pd.read_csv(test_path)
        test_data['ds'] = pd.to_datetime(test_data['ds'])
        print(f"✅ Loaded {len(test_data)} test observations")
        print(f"   This data has completely different patterns!")

    except FileNotFoundError:
        print("❌ Data files not found in DATA_DIR. Please provide CSVs or set DATA_DIR")
        return
    
    # Initialize forecaster with higher similarity threshold
    print("\n🚀 Initializing RegimeAwareForecaster...")
    forecaster = RegimeAwareForecaster(
        season_length=12,
        forecast_horizon=6,
        regime_threshold=0.7,
        similarity_threshold=0.85,  # Higher threshold - less likely to match
        model_config={
            'hidden_dim': 128,
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
            start_period=24,
            epochs_per_regime=30
        )
        print("✅ Initial training completed!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Get initial regime information
    print("\n📈 Initial Training Results:")
    initial_info = forecaster.get_regime_info()
    mem = initial_info.get('memory_module') if isinstance(initial_info, dict) else getattr(forecaster, 'memory_module', None)
    if hasattr(mem, 'get_regime_info'):
        mem_info = mem.get_regime_info()
    elif isinstance(mem, dict):
        mem_info = mem
    else:
        mem_info = {'total_regimes': 0, 'regime_ids': []}

    initial_regimes = mem_info.get('total_regimes', 0)
    print(f"   Initial regimes: {initial_regimes}")
    print(f"   Current regime: {initial_info.get('current_regime_id', getattr(forecaster, 'current_regime_id', None))}")
    print(f"   Regime IDs: {mem_info.get('regime_ids', [])}")

    # Display performance
    if isinstance(initial_info, dict) and 'evaluation_summary' in initial_info:
        eval_summary = initial_info['evaluation_summary']
        if 'latest_metrics' in eval_summary:
            metrics = eval_summary['latest_metrics']
            print(f"   Latest MAPE: {metrics.get('weighted_mape', 'N/A'):.2f}%")
    
    # Now test with completely new regime data
    print("\n🔍 Testing with NEW regime data...")
    print("   This should trigger NEW regime creation (not matching)")
    
    # Prepare test data
    extended_test = test_data.copy()
    extended_test['unique_id'] = 'series_001'  # Same ID for consistency
    
    # Add some context from training data
    context_data = train_data.tail(12).copy()
    combined_test = pd.concat([context_data, extended_test], ignore_index=True)
    combined_test = combined_test.sort_values('ds').reset_index(drop=True)
    
    print(f"   Combined test data shape: {combined_test.shape}")
    print(f"   New data characteristics:")
    print(f"     Mean: {extended_test['y'].mean():.2f}")
    print(f"     Std: {extended_test['y'].std():.2f}")
    print(f"     Min: {extended_test['y'].min():.2f}")
    print(f"     Max: {extended_test['y'].max():.2f}")
    
    try:
        # Process the new regime data
        print(f"\n   Processing new regime data...")
        print(f"   Regimes before: {len(forecaster.memory_module.checkpoints)}")
        
        # This should create a new regime
        forecaster.fit_incremental(
            df=combined_test.tail(30),  # Focus on new data
            start_period=6,
            epochs_per_regime=40  # More epochs for new regime
        )
        
        final_regimes = len(forecaster.memory_module.checkpoints)
        print(f"   Regimes after: {final_regimes}")
        
        if final_regimes > initial_regimes:
            print("🎉 SUCCESS: New regime created for novel patterns!")
            new_regime_count = final_regimes - initial_regimes
            print(f"   {new_regime_count} new regime(s) added to memory")
        else:
            print("⚠️  No new regime created - might have matched existing pattern")
        
    except Exception as e:
        print(f"❌ New regime processing failed: {e}")
        return
    
    # Analyze the new regime
    print("\n🔬 New Regime Analysis:")
    final_info = forecaster.get_regime_info()
    current_regime = final_info['current_regime_id']
    print(f"   Current active regime: {current_regime}")
    
    # Check if it's a new regime ID (normalize memory API)
    init_mem = initial_info.get('memory_module') if isinstance(initial_info, dict) else getattr(forecaster, 'memory_module', None)
    if hasattr(init_mem, 'get_regime_info'):
        init_mem_info = init_mem.get_regime_info()
    elif isinstance(init_mem, dict):
        init_mem_info = init_mem
    else:
        init_mem_info = {'regime_ids': []}

    if current_regime not in init_mem_info.get('regime_ids', []):
        print(f"   ✅ Confirmed: {current_regime} is a NEW regime!")
    else:
        print(f"   ⚠️  Still using existing regime: {current_regime}")
    
    # Test predictions with new regime
    print("\n🔮 Testing predictions with new regime...")
    try:
        predictions = forecaster.predict(
            df=extended_test.tail(12),
            steps_ahead=6
        )
        
        print(f"✅ Generated {len(predictions)} predictions")
        if len(predictions) > 0:
            print("   New regime prediction summary:")
            print(f"   Mean prediction: {predictions['y_pred'].mean():.2f}")
            print(f"   Prediction range: {predictions['y_pred'].min():.2f} - {predictions['y_pred'].max():.2f}")
            
            # Compare with test data scale
            test_mean = extended_test['y'].mean()
            pred_mean = predictions['y_pred'].mean()
            print(f"   Test data mean: {test_mean:.2f}")
            print(f"   Prediction mean: {pred_mean:.2f}")
            print(f"   Ratio: {pred_mean/test_mean:.2f}")
            
            # Show first few predictions (safe for missing 'ds')
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
    
    # Final system state
    print("\n📊 Final System State:")
    final_info = forecaster.get_regime_info()
    mem = final_info.get('memory_module') if isinstance(final_info, dict) else getattr(forecaster, 'memory_module', None)
    if hasattr(mem, 'get_regime_info'):
        mem_info = mem.get_regime_info()
    elif isinstance(mem, dict):
        mem_info = mem
    else:
        mem_info = {'total_regimes': 0, 'regime_ids': []}

    print(f"   Total regimes in memory: {mem_info.get('total_regimes', 0)}")
    print(f"   All regime IDs: {mem_info.get('regime_ids', [])}")
    print(f"   Active regime: {final_info.get('current_regime_id', getattr(forecaster, 'current_regime_id', None))}")

    # Show detection statistics
    detection_info = final_info.get('regime_detector') if isinstance(final_info, dict) else None
    print(f"\n   Detection Statistics:")
    if detection_info and isinstance(detection_info, dict):
        print(f"   Total windows processed: {detection_info.get('total_windows', 0)}")
        print(f"   Regime changes detected: {detection_info.get('regime_changes', 0)}")
        print(f"   Overall change rate: {detection_info.get('change_rate', 0.0):.2%}")
    else:
        print("   Detection statistics not available in this API shape")
    
    print("\n" + "=" * 60)
    print("✅ New Regime Demo Completed Successfully!")
    print("Key takeaway: System created new weights for unprecedented patterns")
    print("=" * 60)


if __name__ == "__main__":
    demo_new_regime()