"""
Materials Project Data Extraction Module

This module provides tools to extract photovoltaic-relevant materials
from the Materials Project database using their API.

Author: Nabil Khossossi
Date: September 2025
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
import warnings

warnings.filterwarnings('ignore')


class MaterialsProjectExtractor:
    """
    Extract and process materials data from Materials Project for
    photovoltaic applications.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the extractor with Materials Project API credentials.
        
        Parameters
        ----------
        api_key : str, optional
            Materials Project API key. If None, looks for MP_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get('MP_API_KEY')
        if not self.api_key:
            raise ValueError(
                "No API key provided. Get one from materialsproject.org"
            )
        
        self.mpr = MPRester(self.api_key)
        
    def get_photovoltaic_candidates(
        self,
        bandgap_range: Tuple[float, float] = (0.9, 2.0),
        n_materials: int = 100,
        stability_threshold: float = 0.1,
        elements_to_exclude: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Query Materials Project for potential photovoltaic materials.
        
        Parameters
        ----------
        bandgap_range : tuple
            (min, max) band gap in eV for PV applications
        n_materials : int
            Maximum number of materials to retrieve
        stability_threshold : float
            Energy above hull threshold (eV/atom)
        elements_to_exclude : list, optional
            Elements to exclude (e.g., radioactive, toxic)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with materials data
        """
        if elements_to_exclude is None:
            # Exclude radioactive, highly toxic, and rare elements
            elements_to_exclude = [
                'U', 'Th', 'Pu', 'Po', 'Ra', 'Ac',  # Radioactive
                'Hg', 'Pb', 'Cd',  # Toxic (though Pb used in perovskites)
            ]
        
        print(f"Querying Materials Project for PV candidates...")
        print(f"Band gap range: {bandgap_range} eV")
        print(f"Stability threshold: {stability_threshold} eV/atom")
        
        # Query materials
        docs = self.mpr.materials.summary.search(
            band_gap=bandgap_range,
            energy_above_hull=(0, stability_threshold),
            num_elements=(2, 5),  # Binary to quaternary compounds
            num_chunks=10,
            fields=[
                "material_id",
                "formula_pretty", 
                "band_gap",
                "energy_above_hull",
                "formation_energy_per_atom",
                "volume",
                "density",
                "symmetry",
                "is_stable",
                "theoretical",
            ]
        )
        
        # Convert to list and limit
        materials = list(docs)[:n_materials]
        
        print(f"Retrieved {len(materials)} materials")
        
        # Convert to DataFrame
        data_list = []
        for mat in materials:
            # Filter by excluded elements
            formula = mat.formula_pretty
            if any(elem in formula for elem in elements_to_exclude):
                continue
                
            data_list.append({
                'material_id': mat.material_id,
                'formula': mat.formula_pretty,
                'band_gap': mat.band_gap,
                'energy_above_hull': mat.energy_above_hull,
                'formation_energy': mat.formation_energy_per_atom,
                'volume': mat.volume,
                'density': mat.density,
                'crystal_system': mat.symmetry.crystal_system,
                'space_group': mat.symmetry.symbol,
                'is_stable': mat.is_stable,
                'is_theoretical': mat.theoretical,
            })
        
        df = pd.DataFrame(data_list)
        print(f"Final dataset: {len(df)} materials")
        
        return df
    
    def get_detailed_properties(
        self, 
        material_ids: List[str],
        properties: Optional[List[str]] = None
    ) -> Dict:
        """
        Get detailed properties for specific materials.
        
        Parameters
        ----------
        material_ids : list
            List of Materials Project IDs
        properties : list, optional
            Specific properties to retrieve
            
        Returns
        -------
        dict
            Detailed properties for each material
        """
        if properties is None:
            properties = [
                "structure",
                "band_gap",
                "efermi",
                "dos",
                "band_structure",
                "dielectric",
            ]
        
        detailed_data = {}
        
        for mp_id in material_ids:
            try:
                # Get material document
                doc = self.mpr.materials.summary.get_data_by_id(mp_id)
                
                detailed_data[mp_id] = {
                    'formula': doc.formula_pretty,
                    'band_gap': doc.band_gap,
                    'structure': doc.structure,
                }
                
                # Get electronic structure if available
                try:
                    dos_data = self.mpr.electronic_structure.dos.get_data_by_id(mp_id)
                    detailed_data[mp_id]['dos'] = dos_data
                except:
                    pass
                
                # Get dielectric properties if available
                try:
                    diel = self.mpr.materials.dielectric.get_data_by_id(mp_id)
                    detailed_data[mp_id]['dielectric'] = diel
                except:
                    pass
                    
            except Exception as e:
                print(f"Error getting data for {mp_id}: {e}")
                continue
        
        return detailed_data
    
    def filter_by_composition(
        self,
        df: pd.DataFrame,
        required_elements: Optional[List[str]] = None,
        forbidden_elements: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filter materials by elemental composition.
        
        Parameters
        ----------
        df : pd.DataFrame
            Materials dataframe
        required_elements : list, optional
            Elements that must be present
        forbidden_elements : list, optional
            Elements that must not be present
            
        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        df_filtered = df.copy()
        
        if required_elements:
            mask = df_filtered['formula'].apply(
                lambda x: all(elem in x for elem in required_elements)
            )
            df_filtered = df_filtered[mask]
        
        if forbidden_elements:
            mask = df_filtered['formula'].apply(
                lambda x: not any(elem in x for elem in forbidden_elements)
            )
            df_filtered = df_filtered[mask]
        
        return df_filtered
    
    def get_perovskite_materials(
        self,
        perovskite_type: str = 'halide',
        n_materials: int = 50
    ) -> pd.DataFrame:
        """
        Get perovskite materials specifically.
        
        Parameters
        ----------
        perovskite_type : str
            'halide' or 'oxide'
        n_materials : int
            Number of materials to retrieve
            
        Returns
        -------
        pd.DataFrame
            Perovskite materials data
        """
        if perovskite_type == 'halide':
            # Typical halide perovskite band gaps
            bandgap_range = (1.2, 2.5)
            # Look for Pb, Sn with I, Br, Cl
            required_elements = None  # We'll filter post-query
        elif perovskite_type == 'oxide':
            bandgap_range = (0.5, 3.0)
            required_elements = ['O']
        else:
            raise ValueError("perovskite_type must be 'halide' or 'oxide'")
        
        df = self.get_photovoltaic_candidates(
            bandgap_range=bandgap_range,
            n_materials=n_materials * 2  # Get extra for filtering
        )
        
        if perovskite_type == 'halide':
            # Filter for halide perovskites
            halides = ['I', 'Br', 'Cl']
            metals = ['Pb', 'Sn', 'Ge', 'Bi']
            
            mask = df['formula'].apply(
                lambda x: (any(h in x for h in halides) and 
                          any(m in x for m in metals))
            )
            df = df[mask].head(n_materials)
        
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        filename: str = 'materials_data.csv',
        output_dir: str = 'data/raw'
    ):
        """
        Save materials data to file.
        
        Parameters
        ----------
        df : pd.DataFrame
            Materials dataframe
        filename : str
            Output filename
        output_dir : str
            Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
        # Also save metadata
        metadata = {
            'n_materials': len(df),
            'bandgap_range': [df['band_gap'].min(), df['band_gap'].max()],
            'formulas': df['formula'].tolist()[:10],  # First 10
        }
        
        meta_file = filepath.replace('.csv', '_metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def example_usage():
    """
    Example usage of the MaterialsProjectExtractor.
    """
    # Initialize extractor
    extractor = MaterialsProjectExtractor()
    
    # Get general PV candidates
    print("=" * 60)
    print("Example 1: General PV Candidates")
    print("=" * 60)
    df_pv = extractor.get_photovoltaic_candidates(
        bandgap_range=(1.1, 1.7),  # Optimal for single junction
        n_materials=50,
        stability_threshold=0.1
    )
    print(f"\nTop 5 materials by band gap:")
    print(df_pv[['formula', 'band_gap', 'energy_above_hull']].head())
    
    # Get perovskite materials
    print("\n" + "=" * 60)
    print("Example 2: Halide Perovskites")
    print("=" * 60)
    df_perov = extractor.get_perovskite_materials(
        perovskite_type='halide',
        n_materials=30
    )
    print(f"\nFound {len(df_perov)} halide perovskites")
    print(df_perov[['formula', 'band_gap']].head(10))
    
    # Save data
    extractor.save_data(df_pv, 'pv_candidates.csv')
    extractor.save_data(df_perov, 'halide_perovskites.csv')
    
    return df_pv, df_perov


if __name__ == "__main__":
    df_pv, df_perov = example_usage()
