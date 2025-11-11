"""
Visualization Tools for Photovoltaic Materials Analysis

Publication-quality visualizations for materials discovery and ML results.

Author: Nabil Khossossi
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


class PVVisualization:
    """
    Visualization tools for photovoltaic materials analysis.
    """
    
    def __init__(self, style: str = 'nature'):
        """
        Initialize visualization toolkit.
        
        Parameters
        ----------
        style : str
            Figure style: 'nature', 'science', 'acs'
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Setup matplotlib style for publications."""
        if self.style == 'nature':
            # Nature journal style
            plt.rcParams['figure.figsize'] = (3.5, 2.625)  # Single column
            plt.rcParams['font.size'] = 7
        elif self.style == 'science':
            # Science journal style
            plt.rcParams['figure.figsize'] = (3.3, 2.5)
            plt.rcParams['font.size'] = 7
        elif self.style == 'acs':
            # ACS journal style
            plt.rcParams['figure.figsize'] = (3.25, 2.5)
            plt.rcParams['font.size'] = 8
    
    def plot_bandgap_distribution(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot band gap distribution with PV-relevant regions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Materials dataframe with 'band_gap' column
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Histogram
        ax.hist(df['band_gap'], bins=30, alpha=0.7, 
                color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Mark optimal regions
        ax.axvspan(1.1, 1.7, alpha=0.2, color='green', 
                   label='Single Junction Optimal')
        ax.axvspan(1.7, 2.0, alpha=0.2, color='orange', 
                   label='Top Cell (Tandem)')
        ax.axvline(1.34, color='red', linestyle='--', linewidth=2,
                   label='SQ Maximum (1.34 eV)')
        
        ax.set_xlabel('Band Gap (eV)', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Band Gap Distribution of PV Candidates')
        ax.legend(frameon=True, fontsize=8)
        ax.grid(alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sq_efficiency_curve(
        self,
        df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Shockley-Queisser efficiency limit curve.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            Materials to overlay on curve
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Theoretical SQ curve
        Eg_range = np.linspace(0.5, 3.5, 100)
        
        # Simplified SQ efficiency (from descriptors.py)
        def sq_eff(Eg):
            if Eg < 0.9:
                return 0.15 * Eg
            elif Eg < 1.8:
                return 0.33 * (1 - abs(Eg - 1.34) / 2)
            else:
                return 0.20 * np.exp(-(Eg - 1.8) / 0.5)
        
        efficiency = np.array([sq_eff(Eg) for Eg in Eg_range])
        
        # Plot SQ limit
        ax.plot(Eg_range, efficiency * 100, 'r-', linewidth=2.5,
                label='Shockley-Queisser Limit')
        
        # Fill under curve
        ax.fill_between(Eg_range, 0, efficiency * 100, alpha=0.1, color='red')
        
        # Mark optimal point
        ax.plot(1.34, 33.7, 'r*', markersize=15, 
                label='Maximum (1.34 eV, 33.7%)')
        
        # Overlay actual materials if provided
        if df is not None and 'band_gap' in df.columns and 'sq_efficiency' in df.columns:
            ax.scatter(df['band_gap'], df['sq_efficiency'] * 100,
                      alpha=0.6, s=50, c='steelblue', edgecolors='black',
                      linewidths=0.5, label='Materials')
        
        # Mark common materials
        common_materials = {
            'Si': (1.12, 29),
            'GaAs': (1.42, 33),
            'CdTe': (1.50, 32),
            'CIGS': (1.15, 29)
        }
        
        for name, (Eg, eff) in common_materials.items():
            ax.plot(Eg, eff, 'o', markersize=8, label=name)
            ax.annotate(name, xy=(Eg, eff), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Band Gap (eV)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Efficiency (%)', fontweight='bold', fontsize=11)
        ax.set_title('Shockley-Queisser Efficiency Limit', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=8)
        ax.grid(alpha=0.3, linestyle=':')
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0, 35)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance with physical interpretation.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            Feature importance from model
        top_n : int
            Number of top features to show
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Get top N features
        df_plot = importance_df.head(top_n).sort_values('importance')
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot)))
        bars = ax.barh(df_plot['feature'], df_plot['importance'], 
                       color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Importance', fontweight='bold', fontsize=11)
        ax.set_title('Feature Importance for PV Efficiency Prediction',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        
        # Add value labels
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            ax.text(row['importance'], i, f" {row['importance']:.3f}",
                   va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Prediction Parity',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot parity plot (predicted vs actual).
        
        Parameters
        ----------
        y_true : ndarray
            True values
        y_pred : ndarray
            Predicted values
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, 
                  c='steelblue', edgecolors='black', linewidths=0.5)
        
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Add metrics text box
        textstr = f'MAE = {mae:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               verticalalignment='top', bbox=props, fontsize=9)
        
        ax.set_xlabel('True Values', fontweight='bold', fontsize=11)
        ax.set_ylabel('Predicted Values', fontweight='bold', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multi_objective_pareto(
        self,
        df: pd.DataFrame,
        obj1: str,
        obj2: str,
        obj3: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Pareto front for multi-objective optimization.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe
        obj1, obj2 : str
            First two objectives (axes)
        obj3 : str, optional
            Third objective (color)
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        
        if obj3 is not None:
            # 3D projected to 2D with color
            scatter = ax.scatter(df[obj1], df[obj2], c=df[obj3],
                               s=100, cmap='viridis', alpha=0.7,
                               edgecolors='black', linewidths=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(obj3, fontweight='bold')
        else:
            # 2D plot
            ax.scatter(df[obj1], df[obj2], s=100, alpha=0.7,
                      c='steelblue', edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(obj1, fontweight='bold', fontsize=11)
        ax.set_ylabel(obj2, fontweight='bold', fontsize=11)
        ax.set_title('Multi-Objective Optimization', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_descriptor_correlation(
        self,
        df: pd.DataFrame,
        descriptors: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix of descriptors.
        
        Parameters
        ----------
        df : pd.DataFrame
            Materials dataframe
        descriptors : list
            List of descriptor names
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Calculate correlation matrix
        corr = df[descriptors].corr()
        
        # Plot heatmap
        im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', 
                      vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(descriptors)))
        ax.set_yticks(np.arange(len(descriptors)))
        ax.set_xticklabels(descriptors, rotation=45, ha='right')
        ax.set_yticklabels(descriptors)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontweight='bold')
        
        # Add correlation values
        for i in range(len(descriptors)):
            for j in range(len(descriptors)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha='center', va='center', color='black',
                             fontsize=8)
        
        ax.set_title('Descriptor Correlation Matrix', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def example_usage():
    """
    Example usage of visualization tools.
    """
    # Create sample data
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'band_gap': np.random.uniform(0.5, 3.0, n),
        'sq_efficiency': np.random.uniform(0.1, 0.35, n),
        'stability_score': np.random.uniform(0, 1, n),
        'formation_energy': np.random.uniform(-2, 2, n),
    })
    
    # Initialize visualizer
    viz = PVVisualization(style='nature')
    
    # Create plots
    print("Creating visualization examples...")
    
    # 1. Band gap distribution
    fig1 = viz.plot_bandgap_distribution(df, 'figures/bandgap_dist.png')
    print("✓ Band gap distribution")
    
    # 2. SQ efficiency curve
    fig2 = viz.plot_sq_efficiency_curve(df, 'figures/sq_curve.png')
    print("✓ SQ efficiency curve")
    
    # 3. Prediction parity
    y_true = np.random.uniform(0.1, 0.3, 50)
    y_pred = y_true + np.random.normal(0, 0.02, 50)
    fig3 = viz.plot_prediction_parity(y_true, y_pred, 
                                      save_path='figures/parity_plot.png')
    print("✓ Parity plot")
    
    print("\nAll figures saved to figures/ directory")


if __name__ == "__main__":
    example_usage()
