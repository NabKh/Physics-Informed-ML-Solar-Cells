"""
Physics-Based Descriptors for Photovoltaic Materials

This module computes physically-motivated descriptors for solar cell materials,
incorporating fundamental photovoltaic physics and band structure theory.

Author: Nabil Khossossi
Date: September 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import constants
from scipy.integrate import simps
import warnings

warnings.filterwarnings('ignore')


class PhotovoltaicDescriptors:
    """
    Compute physics-based descriptors for photovoltaic materials.
    """
    
    def __init__(self, temperature: float = 300):
        """
        Initialize descriptor calculator.
        
        Parameters
        ----------
        temperature : float
            Operating temperature in Kelvin
        """
        self.T = temperature
        self.kB = constants.k  # Boltzmann constant (J/K)
        self.e = constants.e   # Elementary charge (C)
        self.h = constants.h   # Planck constant (J⋅s)
        self.c = constants.c   # Speed of light (m/s)
        
        # Thermal voltage at 300K ≈ 0.0259 eV
        self.V_T = (self.kB * self.T) / self.e
        
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all physics-based descriptors.
        
        Parameters
        ----------
        df : pd.DataFrame
            Materials dataframe with basic properties
            
        Returns
        -------
        pd.DataFrame
            Dataframe with added descriptor columns
        """
        df_features = df.copy()
        
        # Band gap descriptors
        df_features = self._add_bandgap_descriptors(df_features)
        
        # Thermodynamic descriptors
        df_features = self._add_thermodynamic_descriptors(df_features)
        
        # Shockley-Queisser theoretical efficiency
        df_features = self._add_sq_efficiency(df_features)
        
        # Structure-based descriptors
        df_features = self._add_structure_descriptors(df_features)
        
        return df_features
    
    def _add_bandgap_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add band gap related descriptors.
        """
        df = df.copy()
        
        # Optimal band gap for single junction (Shockley-Queisser)
        optimal_Eg = 1.34  # eV
        df['bandgap_deviation'] = np.abs(df['band_gap'] - optimal_Eg)
        
        # Band gap categorization for tandem applications
        df['is_top_cell'] = (df['band_gap'] >= 1.7) & (df['band_gap'] <= 2.0)
        df['is_bottom_cell'] = (df['band_gap'] >= 1.0) & (df['band_gap'] <= 1.4)
        df['is_single_junction'] = (df['band_gap'] >= 1.1) & (df['band_gap'] <= 1.7)
        
        # Photon absorption threshold (wavelength in nm)
        df['absorption_threshold_nm'] = 1240 / df['band_gap']  # λ = hc/E
        
        # Thermal broadening parameter (normalized by thermal voltage)
        df['thermal_broadening'] = df['band_gap'] / self.V_T
        
        return df
    
    def _add_thermodynamic_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add thermodynamic stability descriptors.
        """
        df = df.copy()
        
        # Stability score (inverse of energy above hull)
        df['stability_score'] = 1 / (1 + df['energy_above_hull'])
        
        # Formation energy per atom (more negative = more stable)
        df['formation_stability'] = -df['formation_energy']
        
        # Decomposition resistance (qualitative)
        df['is_thermodynamically_stable'] = df['energy_above_hull'] < 0.05
        
        return df
    
    def _add_sq_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Shockley-Queisser detailed balance efficiency limit.
        
        This is the theoretical maximum efficiency for a single junction
        solar cell based on detailed balance (radiative recombination only).
        """
        df = df.copy()
        
        def sq_efficiency_simple(Eg: float, T: float = 300) -> float:
            """
            Simplified Shockley-Queisser efficiency calculation.
            
            Parameters
            ----------
            Eg : float
                Band gap in eV
            T : float
                Temperature in K
                
            Returns
            -------
            float
                Theoretical maximum efficiency (0-1)
            """
            if Eg < 0.5 or Eg > 4.0:
                return 0.0
            
            # Simplified model using empirical fit to SQ limit
            # Based on: William Shockley and Hans Queisser (1961)
            
            # Optimal voltage (typically 80-90% of Eg)
            V_oc = 0.85 * Eg  # Open circuit voltage approximation
            
            # Photon flux integration (simplified)
            # Assume AM1.5G spectrum (1000 W/m²)
            
            # Current density (proportional to absorbed photons)
            # Photons with E > Eg can be absorbed
            if Eg <= 1.5:
                J_sc = 42 - 22 * Eg  # Empirical fit (mA/cm²)
            else:
                J_sc = 30 * np.exp(-(Eg - 1.5) / 0.5)
            
            J_sc = max(0, J_sc)
            
            # Fill factor (empirical relation)
            v_oc_norm = V_oc / (self.kB * T / self.e)
            FF = (v_oc_norm - np.log(v_oc_norm + 0.72)) / (v_oc_norm + 1)
            FF = np.clip(FF, 0.65, 0.89)
            
            # Efficiency = (Voc × Jsc × FF) / Pin
            P_in = 100  # mW/cm² (AM1.5G)
            efficiency = (V_oc * J_sc * FF) / (P_in * 10)  # Normalize
            
            # Theoretical SQ limit is ~33.7% for Eg ≈ 1.34 eV
            # Scale to match known limit
            if Eg > 0.9 and Eg < 1.8:
                scaling = 0.33 / 0.30  # Empirical correction
                efficiency *= scaling
            
            return np.clip(efficiency, 0, 0.35)
        
        # Apply to each material
        df['sq_efficiency'] = df['band_gap'].apply(
            lambda x: sq_efficiency_simple(x, self.T)
        )
        
        # Efficiency relative to optimum
        df['efficiency_ratio'] = df['sq_efficiency'] / 0.337  # Normalize to max
        
        return df
    
    def _add_structure_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add crystal structure based descriptors.
        """
        df = df.copy()
        
        # Volume-based descriptor (molar volume)
        # Assuming simple compositions for now
        df['molar_volume'] = df['volume']  # Will need refinement
        
        # Density-based features
        df['density_score'] = df['density'] / df['density'].mean()
        
        # Crystal system encoding (simplified)
        crystal_map = {
            'cubic': 1.0,
            'tetragonal': 0.9,
            'orthorhombic': 0.8,
            'hexagonal': 0.85,
            'trigonal': 0.85,
            'monoclinic': 0.7,
            'triclinic': 0.6,
        }
        df['symmetry_score'] = df['crystal_system'].map(crystal_map).fillna(0.5)
        
        return df
    
    def compute_spectral_response(
        self,
        bandgap: float,
        wavelengths: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute theoretical spectral response.
        
        Parameters
        ----------
        bandgap : float
            Band gap in eV
        wavelengths : array, optional
            Wavelengths in nm. If None, uses default range.
            
        Returns
        -------
        wavelengths : ndarray
            Wavelength array in nm
        response : ndarray
            Spectral response (0-1)
        """
        if wavelengths is None:
            wavelengths = np.linspace(300, 1200, 200)  # UV to IR
        
        # Convert wavelength to energy: E (eV) = 1240 / λ (nm)
        photon_energy = 1240 / wavelengths
        
        # Step function approximation (can be refined)
        response = (photon_energy >= bandgap).astype(float)
        
        # Add realistic absorption edge (Urbach tail)
        urbach_energy = 0.05  # eV, typical value
        edge_region = np.abs(photon_energy - bandgap) < 0.2
        
        response[edge_region] = 1 / (
            1 + np.exp((bandgap - photon_energy[edge_region]) / urbach_energy)
        )
        
        return wavelengths, response
    
    def compute_carrier_generation(
        self,
        bandgap: float,
        absorption_coefficient: Optional[float] = None
    ) -> float:
        """
        Estimate photo-generated carrier density.
        
        Parameters
        ----------
        bandgap : float
            Band gap in eV
        absorption_coefficient : float, optional
            Absorption coefficient in cm⁻¹
            
        Returns
        -------
        float
            Generation rate (qualitative)
        """
        # Simplified model
        # Direct gap materials: high absorption
        # Indirect gap materials: lower absorption
        
        # Assume direct gap for simplicity
        if absorption_coefficient is None:
            # Typical values: 10^4 - 10^5 cm⁻¹ for direct gap
            absorption_coefficient = 5e4
        
        # Penetration depth
        penetration_depth = 1 / absorption_coefficient  # cm
        
        # Simplified generation (proportional to absorption)
        generation = absorption_coefficient / 1e5  # Normalized
        
        return generation
    
    def add_advanced_descriptors(
        self,
        df: pd.DataFrame,
        detailed_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add advanced descriptors if detailed electronic structure available.
        
        Parameters
        ----------
        df : pd.DataFrame
            Materials dataframe
        detailed_data : dict, optional
            Detailed properties from Materials Project
            
        Returns
        -------
        pd.DataFrame
            Dataframe with advanced descriptors
        """
        df = df.copy()
        
        if detailed_data is None:
            # Add placeholder columns
            df['effective_mass_electron'] = np.nan
            df['effective_mass_hole'] = np.nan
            df['dos_at_fermi'] = np.nan
            return df
        
        # Extract from detailed data if available
        for idx, row in df.iterrows():
            mp_id = row['material_id']
            
            if mp_id in detailed_data:
                props = detailed_data[mp_id]
                
                # Effective masses (if available)
                # This would require band structure analysis
                # Placeholder for now
                df.at[idx, 'effective_mass_electron'] = 0.5  # m_e
                df.at[idx, 'effective_mass_hole'] = 0.5  # m_e
                
                # DOS at Fermi level
                if 'dos' in props:
                    # Extract from DOS data
                    pass
        
        return df


def example_usage():
    """
    Example usage of PhotovoltaicDescriptors.
    """
    # Create sample data
    sample_data = {
        'material_id': ['mp-001', 'mp-002', 'mp-003'],
        'formula': ['GaAs', 'Si', 'CdTe'],
        'band_gap': [1.42, 1.12, 1.50],
        'energy_above_hull': [0.0, 0.0, 0.02],
        'formation_energy': [-0.5, -0.3, -0.4],
        'volume': [45.0, 40.0, 50.0],
        'density': [5.3, 2.3, 5.8],
        'crystal_system': ['cubic', 'cubic', 'cubic'],
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize descriptor calculator
    descriptor_calc = PhotovoltaicDescriptors(temperature=300)
    
    # Compute all descriptors
    df_with_features = descriptor_calc.compute_all(df)
    
    print("Physics-Based Descriptors:")
    print("=" * 80)
    print(df_with_features[[
        'formula', 'band_gap', 'sq_efficiency', 
        'bandgap_deviation', 'stability_score'
    ]])
    
    # Compute spectral response for GaAs
    wavelengths, response = descriptor_calc.compute_spectral_response(1.42)
    
    print("\nSpectral Response computed for band gap = 1.42 eV (GaAs)")
    print(f"Absorption threshold: {1240/1.42:.1f} nm")
    
    return df_with_features


if __name__ == "__main__":
    df_features = example_usage()
