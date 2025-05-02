@echo off
setlocal enabledelayedexpansion

:: 1 if you want to clone fresh, or 0 to use existing code
set CLEAN_INSTALL=0

:: If CLEAN_INSTALL is true, remove existing folder and clone the repo
if !CLEAN_INSTALL! EQU 1 (
    echo Removing existing folder and cloning repository...
    if exist SPlisHSPlasH rmdir /s /q SPlisHSPlasH
    git clone https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git
    cd SPlisHSPlasH
) else (
    echo Using existing code with your modifications...
    :: If we're not already in the SPlisHSPlasH folder, enter it
    if not exist build (
        if exist SPlisHSPlasH (
            cd SPlisHSPlasH
        )
    )
)

:: Compilation step common to both cases
echo Preparing compilation...
if exist build rmdir /s /q build
mkdir build
cd build

echo Configuring project with CMake...
cmake -G "Visual Studio 17 2022" -A x64 ^
-DCMAKE_MAKE_PROGRAM="C:/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/amd64/MSBuild.exe" ^
-DCMAKE_BUILD_TYPE=Release ^
-DUSE_GPU_NEIGHBORHOOD_SEARCH=ON ^
-DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" ^
-DUSE_DOUBLE_PRECISION=OFF ^
-DUSE_EMBEDDED_PYTHON=ON ^
-DUSE_OpenMP=ON ^
-DUSE_AVX=ON ^
-DUSE_IMGUI=ON ^
-DPYTHON_EXECUTABLE=C:/Users/lucas/AppData/Local/Programs/Python/Python312/python.exe ^
-Dpybind11_DIR=C:/Users/lucas/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11 ^
..

echo Building project...
cmake --build . --config Release

echo Moving to bin folder...
cd ../bin

echo Compilation done!