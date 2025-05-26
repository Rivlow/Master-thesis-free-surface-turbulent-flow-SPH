# 3D Turbulent Free-Surface Flow Simulation with SPH Method

## Table of Contents

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

### Research Objectives
- Simulate obstacle formation during flood events within a single software framework
- Leverage [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) for its advanced features:
  - Real-time GUI visualization
  - Highly optimized algorithms
  - Multi-physics simulation capabilities

## Features

- **Flexible Simulation Parameters**: Particle size-independent simulations
- **Automated Workflow**: Batch scripts for building and running simulations
- **Data Management**: Automated output organization and transfer
- **Extensible Framework**: Python scripts for scene generation and analysis

##  Prerequisites

Before building this project, ensure you have the following installed:

- **CMake**: 3.31.7
- **Python**: 3.12.4 with virtual environment support
- **NVIDIA CUDA Toolkit**: 12.8 (for GPU acceleration)
- **Git** for version control

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rivlow/Master-thesis-free-surface-turbulent-flow-SPH.git
cd Master-thesis-free-surface-turbulent-flow-SPH
```

### 2. Build the Project

The project includes an automated build script for Windows:

```bash
rebuild.bat
```

**Note**: The build script includes paths to CUDA Toolkit, Python interpreter, and pybind11. You may need to modify these paths in `rebuild.bat` according to your system configuration.

#### Build Options
- Set `CLEAN_INSTALL=1` to use the original SPlisHSPlasH source code
- Set `CLEAN_INSTALL=0` to use modified source code

The build script automatically configures CMake with, for instance:
- AVX optimization
- Double precision floating-point
- GPU utilization for neighborhood search

## Project Structure

```
Master-thesis-free-surface-turbulent-flow-SPH/
├── my_output/              # Simulation output data storage
│   └── local/             
│       └── my_simulations/
├── own_python/            # Custom Python scripts
│   ├── write_scenes/      # Scene generation scripts
│   ├── Transfer_data.py   # Output organization utility
│   └── validation_test/   # Validation and analysis scripts
├── Pictures/              # Figures for documentation
├── SPlisHSPlasH/         # Modified SPH source code
├── venv/                 # Python virtual environment
├── rebuild.bat           # Automated build script
└── run_simulation.bat    # Simulation execution script
```

### Key Components

#### `my_output/`
Organized storage for simulation results, preventing conflicts when running multiple simulations with different parameters.

#### `own_python/write_scenes/`
Python scripts that generate JSON parameter files for simulations. Each script defines:
- `json_path`: Location of simulation parameter file
- `output_path`: Destination for simulation outputs
- Automatic generation of summary files with simulation parameters

#### `Transfer_data.py`
Utility script that automatically copies simulation outputs from the default location to organized folders in `my_output/`.

##  Usage

### Running a Simulation

1. **CReate python virtual environment**
	```bash
	python -m venv venv
	venv\Scripts\activate.bat
	pip install ensurepath
	```

2. **Generate Scene Parameters**
   ```bash
   python own_python/write_scenes/wrte_free_surface.py
   ```

3. **Execute Simulation**
   ```bash
   run_simulation.bat
   ```

4. **Organize Outputs**
   ```bash
   python own_python/Transfer_data.py
   ```

One reminds that the activate.bat script should always be used when creating terminal session.

### Modifying Simulation Parameters

All simulations are particle size-independent. To change spatial resolution:
1. Adjust particle radius in the scene generation script
2. Numerical parameters (e.g., `angular_viscosity` for vorticity) may need adjustment

## Technical Specifications

### Hardware Configuration
- **Laptop**: ASUS ROG Zephyrus G16 (GU605MI)
- **RAM**: 32 GB
- **CPU**: Intel Core Ultra 7 155H @ 3.80 GHz (22 logical processors)
- **GPU**: 
  - NVIDIA GeForce RTX 4070 Laptop GPU
  - Intel Arc Graphics
- **OS**: Windows

### Software Stack
- **Build System**: CMake 3.31.7
- **Programming Language**: C++ with Python 3.12.4 bindings
- **GPU Computing**: NVIDIA CUDA Toolkit 12.8
- **SPH Framework**: Modified SPlisHSPlasH

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