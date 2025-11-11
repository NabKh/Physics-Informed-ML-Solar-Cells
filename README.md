# Physics-Informed Machine Learning for Solar Cell Materials Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive tutorial and framework for applying physics-informed machine learning to photovoltaic materials discovery using Materials Project data.

## Overview

This repository demonstrates how to:
- Extract photovoltaic-relevant properties from Materials Project database
- Construct physics-based descriptors for solar cell materials
- Build interpretable ML models with physical constraints
- Accelerate materials screening while maintaining scientific rigor

**Key Philosophy**: Rather than treating ML as a black box, we incorporate fundamental photovoltaic physics (band structure, optical absorption, charge transport) into the learning framework.

## Quick Start

```python
from src.data_extraction import MaterialsProjectExtractor
from src.descriptors import PhotovoltaicDescriptors
from src.models import PhysicsInformedModel

# Extract materials data
extractor = MaterialsProjectExtractor(api_key="YOUR_API_KEY")
materials = extractor.get_photovoltaic_candidates(n_materials=100)

# Compute physics-based descriptors
descriptor_calc = PhotovoltaicDescriptors()
features = descriptor_calc.compute_all(materials)

# Train physics-informed model
model = PhysicsInformedModel(enforce_bandgap_constraint=True)
model.fit(features, target='efficiency')
```

## Tutorial Notebooks

1. **Materials Project Data Extraction** (`01_materials_project_data_extraction.ipynb`)
   - API setup and authentication
   - Querying relevant materials for PV applications
   - Data cleaning and preprocessing

2. **Physics Descriptors for Photovoltaics** (`02_physics_descriptors_for_photovoltaics.ipynb`)
   - Electronic band structure descriptors
   - Optical absorption features
   - Charge transport metrics
   - Stability indicators

3. **ML Models with Physics Constraints** (`03_ml_models_with_physics_constraints.ipynb`)
   - Incorporating Shockley-Queisser limit
   - Band gap optimization for different architectures
   - Multi-objective optimization (efficiency, stability, cost)

4. **Interpretable Predictions** (`04_interpretable_predictions.ipynb`)
   - Feature importance analysis
   - Physical interpretation of ML decisions
   - Uncertainty quantification

## 🔬 Physics-Informed Features

### Electronic Structure
- **Band gap (Eg)**: Optimal range 1.1-1.7 eV for single junction
- **Effective masses (m*)**: Impacts charge mobility
- **Band alignment**: Valence/conduction band positions
- **Density of states**: Near band edges

### Optical Properties
- **Absorption coefficient**: Direct vs indirect transitions
- **Spectral matching**: Solar spectrum overlap
- **Theoretical efficiency**: Based on detailed balance

### Stability & Synthesis
- **Formation energy**: Thermodynamic stability
- **Decomposition energy**: Against competing phases
- **Synthesizability score**: Likelihood of experimental realization

## 📊 Example Results

### Descriptor Correlation with Efficiency
![Descriptor Analysis](figures/descriptor_correlation.png)

### Physics-Constrained Predictions
![Predictions](figures/physics_constrained_predictions.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/NabKh/Physics-Informed-ML-Solar-Cells.git
cd Physics-Informed-ML-Solar-Cells

# Create conda environment
conda env create -f environment.yml
conda activate piml-pv

# Or use pip
pip install -r requirements.txt
```

### Materials Project API Key
1. Register at [Materials Project](https://materialsproject.org/)
2. Get your API key from dashboard
3. Set environment variable: `export MP_API_KEY="your_key_here"`

## 📖 Key Concepts

### Why Physics-Informed ML?

Traditional ML approaches may learn spurious correlations and violate physical laws. Our approach:

1. **Constrains predictions** to physically reasonable ranges (e.g., Eg > 0)
2. **Uses domain knowledge** in feature engineering
3. **Ensures interpretability** through physics-based features
4. **Incorporates physical laws** (e.g., detailed balance limit)

### Shockley-Queisser Limit Integration

```python
def theoretical_efficiency(bandgap, temperature=300):
    """
    Calculate theoretical efficiency limit based on detailed balance.
    Incorporates fundamental thermodynamic constraints.
    """
    # Implementation based on SQ limit
    pass
```

## Educational Goals

This repository serves as:
- **Tutorial** for students learning ML for materials science
- **Template** for researchers starting PV materials projects
- **Best practices** guide for physics-informed ML

## 📈 Use Cases

### 1. Rapid Screening
Screen 50-100 materials in minutes instead of months of DFT calculations

### 2. Design Space Exploration
Identify promising regions in composition-property space

### 3. Hypothesis Generation
Suggest unconventional materials for experimental validation

### 4. Teaching Tool
Demonstrate integration of physics and machine learning


## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

**Nabil Khossossi**
- Website: [sustai-nabil.com](https://sustai-nabil.com)
- Email: n.khossossi@differ.nl
- GitHub: [@NabKh](https://github.com/NabKh)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

**Note**: This is an educational and research tool. For production applications, additional validation and testing are required.
