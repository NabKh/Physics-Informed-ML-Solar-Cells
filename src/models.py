"""
Physics-Informed Machine Learning Models for Photovoltaics

This module implements ML models with physical constraints and interpretability
for solar cell materials discovery.

Author: Nabil Khossossi
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class PhysicsInformedModel:
    """
    Machine learning model with physics-based constraints for PV materials.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        enforce_bandgap_constraint: bool = True,
        enforce_sq_limit: bool = True,
        random_state: int = 42
    ):
        """
        Initialize physics-informed model.
        
        Parameters
        ----------
        model_type : str
            'random_forest', 'gradient_boosting', or 'ridge'
        enforce_bandgap_constraint : bool
            Enforce physical band gap constraints (Eg > 0)
        enforce_sq_limit : bool
            Enforce Shockley-Queisser efficiency limit
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.enforce_bandgap_constraint = enforce_bandgap_constraint
        self.enforce_sq_limit = enforce_sq_limit
        self.random_state = random_state
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        self.is_fitted = False
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        target_name: str = 'target'
    ):
        """
        Fit the model with physics constraints.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        y : Series or array
            Target values
        feature_names : list, optional
            Names of features
        target_name : str
            Name of target variable
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        
        # Apply physics constraints to predictions
        if self.enforce_constraints:
            y_pred = self._apply_constraints(y_pred, X)
        
        train_mae = mean_absolute_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        print(f"Training complete: MAE = {train_mae:.4f}, R² = {train_r2:.4f}")
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        apply_constraints: bool = True
    ) -> np.ndarray:
        """
        Make predictions with optional physics constraints.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        apply_constraints : bool
            Whether to apply physics constraints
            
        Returns
        -------
        ndarray
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_array)
        y_pred = self.model.predict(X_scaled)
        
        # Apply physics constraints
        if apply_constraints and self.enforce_constraints:
            y_pred = self._apply_constraints(y_pred, X_array)
        
        return y_pred
    
    def _apply_constraints(
        self,
        predictions: np.ndarray,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Apply physics-based constraints to predictions.
        
        Parameters
        ----------
        predictions : ndarray
            Raw model predictions
        features : ndarray
            Input features (to check band gap, etc.)
            
        Returns
        -------
        ndarray
            Constrained predictions
        """
        constrained = predictions.copy()
        
        # Constraint 1: Band gap must be positive
        if self.enforce_bandgap_constraint and self.target_name == 'band_gap':
            constrained = np.maximum(constrained, 0.1)  # Minimum 0.1 eV
            constrained = np.minimum(constrained, 6.0)  # Maximum 6.0 eV
        
        # Constraint 2: Efficiency cannot exceed Shockley-Queisser limit
        if self.enforce_sq_limit and 'efficiency' in self.target_name.lower():
            # Maximum theoretical efficiency ≈ 33.7%
            constrained = np.minimum(constrained, 0.337)
            constrained = np.maximum(constrained, 0.0)
        
        # Constraint 3: Energy must be physically reasonable
        if 'energy' in self.target_name.lower():
            # Formation energies typically between -5 and 5 eV/atom
            constrained = np.clip(constrained, -5, 5)
        
        return constrained
    
    @property
    def enforce_constraints(self) -> bool:
        """Check if any constraints are enabled."""
        return self.enforce_bandgap_constraint or self.enforce_sq_limit
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance with physical interpretation.
        
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not have feature importance")
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        else:
            feature_names = self.feature_names
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        y : Series or array
            Target values
        cv : int
            Number of folds
            
        Returns
        -------
        dict
            Cross-validation scores
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Perform cross-validation
        scores = cross_val_score(
            self.model, X_scaled, y_array,
            cv=cv, scoring='neg_mean_absolute_error'
        )
        
        r2_scores = cross_val_score(
            self.model, X_scaled, y_array,
            cv=cv, scoring='r2'
        )
        
        results = {
            'mae_mean': -scores.mean(),
            'mae_std': scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
        
        print(f"Cross-validation (k={cv}):")
        print(f"  MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"  R²:  {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        
        return results


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for solar cell materials.
    
    Optimize for:
    - High efficiency
    - High stability
    - Low cost
    - Manufacturability
    """
    
    def __init__(self):
        """Initialize multi-objective optimizer."""
        self.models = {}
        self.objectives = ['efficiency', 'stability', 'cost']
        
    def fit_objectives(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        objective_cols: Dict[str, str]
    ):
        """
        Fit separate models for each objective.
        
        Parameters
        ----------
        df : pd.DataFrame
            Materials data
        feature_cols : list
            Feature column names
        objective_cols : dict
            Mapping of objective names to column names
        """
        X = df[feature_cols]
        
        for obj_name, col_name in objective_cols.items():
            if col_name not in df.columns:
                print(f"Warning: {col_name} not in dataframe")
                continue
            
            y = df[col_name]
            
            # Create and fit model
            model = PhysicsInformedModel(
                model_type='random_forest',
                enforce_sq_limit=(obj_name == 'efficiency')
            )
            
            model.fit(X, y, feature_names=feature_cols, target_name=obj_name)
            self.models[obj_name] = model
            
        print(f"\nFitted {len(self.models)} objective models")
    
    def predict_pareto_front(
        self,
        X: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Predict Pareto front for multi-objective optimization.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for candidates
        weights : dict, optional
            Weights for each objective (must sum to 1)
            
        Returns
        -------
        pd.DataFrame
            Predictions with Pareto ranking
        """
        if weights is None:
            # Equal weighting
            n_obj = len(self.models)
            weights = {name: 1/n_obj for name in self.models.keys()}
        
        predictions = pd.DataFrame()
        
        # Get predictions for each objective
        for obj_name, model in self.models.items():
            predictions[obj_name] = model.predict(X)
        
        # Calculate weighted score
        predictions['weighted_score'] = sum(
            predictions[obj] * weights.get(obj, 0)
            for obj in self.models.keys()
        )
        
        # Rank by weighted score
        predictions['rank'] = predictions['weighted_score'].rank(ascending=False)
        
        return predictions


def example_usage():
    """
    Example usage of physics-informed models.
    """
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 100
    
    # Features
    band_gap = np.random.uniform(0.5, 3.0, n_samples)
    stability = np.random.uniform(0.0, 0.3, n_samples)
    density = np.random.uniform(2.0, 8.0, n_samples)
    
    # Target: SQ efficiency (with noise)
    def sq_eff(Eg):
        if Eg < 0.9:
            return 0.15 * Eg
        elif Eg < 1.8:
            return 0.33 * (1 - abs(Eg - 1.34) / 2)
        else:
            return 0.20 * np.exp(-(Eg - 1.8) / 0.5)
    
    efficiency = np.array([sq_eff(Eg) for Eg in band_gap])
    efficiency += np.random.normal(0, 0.02, n_samples)  # Add noise
    efficiency = np.clip(efficiency, 0, 0.35)
    
    # Create dataframe
    df = pd.DataFrame({
        'band_gap': band_gap,
        'stability_score': stability,
        'density': density,
        'efficiency': efficiency
    })
    
    # Prepare features and target
    feature_cols = ['band_gap', 'stability_score', 'density']
    X = df[feature_cols]
    y = df['efficiency']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=" * 80)
    print("Physics-Informed Model Example")
    print("=" * 80)
    
    # Train model WITHOUT constraints
    print("\n1. Standard Model (no constraints):")
    model_standard = PhysicsInformedModel(
        enforce_sq_limit=False
    )
    model_standard.fit(X_train, y_train, feature_names=feature_cols, target_name='efficiency')
    y_pred_standard = model_standard.predict(X_test, apply_constraints=False)
    
    mae_standard = mean_absolute_error(y_test, y_pred_standard)
    print(f"Test MAE: {mae_standard:.4f}")
    print(f"Max prediction: {y_pred_standard.max():.3f} (may violate SQ limit)")
    
    # Train model WITH constraints
    print("\n2. Physics-Informed Model (with SQ limit):")
    model_physics = PhysicsInformedModel(
        enforce_sq_limit=True
    )
    model_physics.fit(X_train, y_train, feature_names=feature_cols, target_name='efficiency')
    y_pred_physics = model_physics.predict(X_test)
    
    mae_physics = mean_absolute_error(y_test, y_pred_physics)
    print(f"Test MAE: {mae_physics:.4f}")
    print(f"Max prediction: {y_pred_physics.max():.3f} (≤ 0.337, SQ limit)")
    
    # Feature importance
    print("\n3. Feature Importance:")
    importance = model_physics.get_feature_importance()
    print(importance)
    
    # Cross-validation
    print("\n4. Cross-Validation:")
    cv_results = model_physics.cross_validate(X, y, cv=5)
    
    return model_physics, df


if __name__ == "__main__":
    model, data = example_usage()
