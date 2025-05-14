@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: Default values
set SCENE_FILE=..\data\Scenes\free_surface.json
set NO_GUI=0

:: Parse command line arguments
if not "%~1"=="" set SCENE_FILE=%~1
if not "%~2"=="" set NO_GUI=%~2

:: Change directory to SPlisHSPlasH/bin
cd "SPlisHSPlasH\bin"

:: Capture start time
set start_time=%time%

echo Running simulation...
echo Scene file: %SCENE_FILE%
if %NO_GUI%==1 (
    echo GUI disabled
    call :RUN_SIMULATION "--no-gui"
) else (
    echo GUI enabled
    call :RUN_SIMULATION ""
)
goto :CALCULATE_TIME

:RUN_SIMULATION
SPHSimulator %SCENE_FILE% %~1
exit /b

:CALCULATE_TIME
:: Capture end time, whether by interruption or normal completion
set end_time=%time%

if errorlevel 1 (
    echo.
    echo Simulation interrupted by user!
) else (
    echo Simulation completed!
)

:: Calculate time difference
set options="tokens=1-4 delims=:.,"
for /f %options% %%a in ("%start_time%") do set start_h=%%a&set /a start_m=%%b&set /a start_s=%%c&set /a start_ms=%%d
for /f %options% %%a in ("%end_time%") do set end_h=%%a&set /a end_m=%%b&set /a end_s=%%c&set /a end_ms=%%d

:: Calculate total time in hundredths of a second
set /a start_tot= (%start_h%*3600 + %start_m%*60 + %start_s%)*100 + %start_ms%
set /a end_tot= (%end_h%*3600 + %end_m%*60 + %end_s%)*100 + %end_ms%

:: If we cross midnight (end_tot < start_tot), add a day
if %end_tot% LSS %start_tot% set /a end_tot+=24*3600*100

:: Calculate difference
set /a diff_tot=%end_tot%-%start_tot%
set /a diff_h=%diff_tot% / 360000
set /a diff_m=(%diff_tot% - %diff_h%*360000) / 6000
set /a diff_s=(%diff_tot% - %diff_h%*360000 - %diff_m%*6000) / 100
set /a diff_ms=(%diff_tot% - %diff_h%*360000 - %diff_m%*6000 - %diff_s%*100)

:: Display execution time
echo.
echo Execution time: %diff_h%h %diff_m%m %diff_s%s %diff_ms%ms

endlocal