@echo off
setlocal

:: Check venv is activated
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error while running activate.bat
    exit /b 1
)

set SCENE_NAME=bridge
set NO_GUI=0

:: Parse arguments
if not "%~1"=="" set SCENE_NAME=%~1
if not "%~2"=="" set NO_GUI=%~2

:: set scene path
set SCENE_FILE=..\data\Scenes\%SCENE_NAME%.json

cd "SPlisHSPlasH\bin"

:: Find t_init
echo Running simulation...
for /f %%i in ('powershell -command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"') do set start_time=%%i

if %NO_GUI%==1 (
    SPHSimulator %SCENE_FILE% --no-gui
) else (
    SPHSimulator %SCENE_FILE%
)

:: Find elapsed time
for /f %%i in ('powershell -command "$start='%start_time%'; $end=Get-Date; $diff=New-TimeSpan $start $end; Write-Host $diff.ToString('hh\:mm\:ss')"') do set elapsed=%%i

echo total simulation time = %elapsed% >> "output\%SCENE_NAME%\simulation_summary.txt"

echo.
echo Simulation terminee - Temps ecoule: %elapsed%