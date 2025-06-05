# 3D Turbulent Free-Surface Flow Simulation with SPH Method

![Demonstration](demo.gif)

## Table of contents

- [Synopsis](#synopsis)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Technical Specifications](#technical-specifications)
- [Acknowledgements](#acknowledgements)

## Synopsis

This repository contains the implementation and results of my Master's thesis at the University of Liège (ULiège), focusing on simulating 3D turbulent free-surface flows with dynamic rigid bodies using the Smoothed Particle Hydrodynamics (SPH) method.

### Research objectives
- Simulate obstacle formation during flood events within a single software framework
- Leverage [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) for its advanced features:
  - Real-time GUI visualization
  - Highly optimized algorithms
  - Multi-physics simulation capabilities

## Features

- **Flexible simulation parameters**: Particle size-independent simulations
- **Automated workflow**: Batch scripts for building and running simulations
- **Data management**: Automated output organization and transfer
- **Extensible framework**: Python scripts for scene generation and analysis

##  Prerequisites

Before building this project, ensure you have the following installed:

- **CMake**: 3.31.7
- **Python**: 3.12.4 with virtual environment support
- **NVIDIA CUDA Toolkit**: 12.8 (for GPU acceleration)
- **Git** for version control

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rivlow/Master-thesis-free-surface-turbulent-flow-SPH.git
cd Master-thesis-free-surface-turbulent-flow-SPH
```

### 2. Build the project

Make sure to create a virtual python environment:
```bash
python -m venv venv
```

Then, the project includes an automated build script for Windows:
```bash
rebuild.bat
```

**Note**: The build script includes paths to CUDA Toolkit, Python interpreter, and pybind11. You may need to modify these paths in `rebuild.bat` according to your system configuration.

#### Build options
- Set `CLEAN_INSTALL=1` to use the original SPlisHSPlasH source code
- Set `CLEAN_INSTALL=0` to use modified source code

The build script automatically configures CMake with, for instance:
- AVX optimization
- Double precision floating-point
- GPU utilization for neighborhood search

## Project structure

```
Master-thesis-free-surface-turbulent-flow-SPH/
├── my_output/              # Simulation output data storage
│   └── local/             
│       └── my_simulations/
├── python_scripts/            # Custom Python scripts
│   ├── Tools_global/      # General scripts for data processing
│   ├── Transfer_data.py   # Output organization utility
│   └── validation_test/   # Validation and analysis scripts (Turbulent/Free_surface/Bridge)
├── Pictures/              # Figures for documentation
├── SPlisHSPlasH/         # Modified SPH source code
├── venv/                 # Python virtual environment
├── rebuild.bat           # Automated build script
└── run_simulation.bat    # Simulation execution script
```

### Key components

#### `my_output/`
Organized storage for simulation results, preventing conflicts when running multiple simulations with different parameters.

#### `Transfer_data.py`
Utility script that automatically copies simulation outputs from the default location to organized folders in `my_output/`.

##  Usage

### Running a simulation

1. **(If not already) Create python virtual environment**
	```bash
	python -m venv venv
	venv\Scripts\activate.bat
	pip install ensurepath
	```

2. **Generate scene parameters**
   ```bash
   python own_python/write_scenes/wrte_free_surface.py
   ```

3. **Execute simulation**
   ```bash
   run_simulation.bat
   ```

4. **Organize outputs**
   ```bash
   python own_python/Transfer_data.py
   ```

One reminds that the activate.bat script should always be used when creating terminal session.

### Modifying simulation parameters

All simulations are particle size-independent. To change spatial resolution:
1. Adjust particle radius in the scene generation script
2. Numerical parameters (e.g., `angular_viscosity` for vorticity) may need adjustment

## Technical specifications

### Hardware configuration
- **Laptop**: ASUS ROG Zephyrus G16 (GU605MI)
- **RAM**: 32 GB
- **CPU**: Intel Core Ultra 7 155H @ 3.80 GHz (22 logical processors)
- **GPU**: 
  - NVIDIA GeForce RTX 4070 Laptop GPU
  - Intel Arc Graphics
- **OS**: Windows 11

### Software utilised
- **Build system**: CMake 3.31.7
- **Programming language**: C++ with Python 3.12.4 bindings
- **GPU computing**: NVIDIA CUDA Toolkit 12.8
- **ParaView version**: ParaView 5.13.3
- **SPH framework**: Modified SPlisHSPlasH

## Acknowledgements

Special thanks to:
- **[Professor Pierre Archambeau](https://www.uliege.be/cms/c_9054334/fr/repertoire?uid=u016646)** - For his relentless support, guidance, and clear explanations during this Master's thesis
- **[Professor Jan Bender](https://animation.rwth-aachen.de/person/1/)** - For his immediate and helpful responses to technical questions during desperate moments
- The SPlisHSPlasH development team for providing an excellent SPH framework

---

**Author**: [Luca Santoro]  
**Institution**: University of Liège (ULiège)  
**Year**: 2024-2025  
**Contact**: [luca.santoro@student.uliege.be]