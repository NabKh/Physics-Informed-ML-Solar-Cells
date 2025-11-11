"""
End-to-End Workflow

Here we will demonstrate the complete workflow from data extraction
to physics-informed ML prediction.

Author: Nabil Khossossi
Date: September 2025
"""

import sys
sys.path.append('./src')

import numpy as np
import pandas as pd
from data_extraction import MaterialsProjectExtractor
from descriptors import PhotovoltaicDescriptors
from models import PhysicsInformedModel
from visualization import PVVisualization

def main():
    """
    Run complete workflow demonstration.
    """
    print("=" * 80)
    print("Physics-Informed ML for Solar Cells - Quick Start")
    print("=" * 80)
    
    # Step 1: Extract Data
    print("\n[1/5] Extracting materials from Materials Project...")
    extractor = MaterialsProjectExtractor()
    df = extractor.get_photovoltaic_candidates(
        bandgap_range=(1.1, 1.7),
        n_materials=50,
        stability_threshold=0.1
    )
    print(f"  ✓ Retrieved {len(df)} materials")
    
    # Step 2: Compute Physics Descriptors
    print("\n[2/5] Computing physics-based descriptors...")
    descriptor_calc = PhotovoltaicDescriptors(temperature=300)
    df_features = descriptor_calc.compute_all(df)
    print(f"  ✓ Computed {len(df_features.columns) - len(df.columns)} new features")
    
    # Step 3: Prepare ML Dataset
    print("\n[3/5] Preparing machine learning dataset...")
    feature_cols = [
        'band_gap', 'energy_above_hull', 'formation_energy',
        'density', 'bandgap_deviation', 'stability_score', 'thermal_broadening'
    ]
    
    # Check which features exist
    feature_cols = [col for col in feature_cols if col in df_features.columns]
    
    X = df_features[feature_cols]
    y = df_features['sq_efficiency']  # Target: SQ efficiency
    
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Samples: {len(X)}")
    
    # Step 4: Train Physics-Informed Model
    print("\n[4/5] Training physics-informed ML model...")
    model = PhysicsInformedModel(
        model_type='random_forest',
        enforce_sq_limit=True
    )
    
    model.fit(X, y, feature_names=feature_cols, target_name='sq_efficiency')
    
    # Cross-validation
    cv_results = model.cross_validate(X, y, cv=5)
    
    # Feature importance
    importance = model.get_feature_importance()
    print("\n  Top 3 Most Important Features:")
    for idx, row in importance.head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.3f}")
    
    # Step 5: Visualizations
    print("\n[5/5] Creating visualizations...")
    viz = PVVisualization()
    
    # Plot SQ efficiency curve
    viz.plot_sq_efficiency_curve(df_features, 'figures/sq_curve_example.png')
    print("  ✓ SQ efficiency curve saved")
    
    # Plot feature importance
    viz.plot_feature_importance(importance, save_path='figures/feature_importance_example.png')
    print("  ✓ Feature importance plot saved")
    
    # Predictions
    y_pred = model.predict(X)
    viz.plot_prediction_parity(y.values, y_pred, 
                              title='Physics-Informed Model Predictions',
                              save_path='figures/parity_example.png')
    print("  ✓ Parity plot saved")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Analyzed {len(df)} materials")
    print(f"✓ Computed {len(df_features.columns)} total features")
    print(f"✓ Trained model with R² = {cv_results['r2_mean']:.3f}")
    print(f"✓ All figures saved to figures/")
    print("\nNext steps:")
    print("  - Explore notebooks/ for detailed tutorials")
    print("  - Run 'jupyter notebook' to try interactive analysis")
    print("  - Modify src/ modules for your specific application")
    
    return model, df_features


if __name__ == "__main__":
    model, data = main()
