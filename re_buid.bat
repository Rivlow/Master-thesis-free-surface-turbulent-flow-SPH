if exist SPlisHSPlasH rmdir /s /q SPlisHSPlasH
git clone https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git
cd SPlisHSPlasH
if exist build rmdir /s /q build
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" ^
-DCMAKE_BUILD_TYPE=Release ^
-DUSE_DOUBLE_PRECISION=OFF ^
-DUSE_EMBEDDED_PYTHON=ON ^
-DUSE_OpenMP=ON ^
-DUSE_AVX=ON ^
-DUSE_IMGUI=ON ^
-DPYTHON_EXECUTABLE=C:/Users/lucas/AppData/Local/Programs/Python/Python312/python.exe ^
-Dpybind11_DIR=C:/Users/lucas/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11 ^
..
cmake --build . --config Release
cd ../bin