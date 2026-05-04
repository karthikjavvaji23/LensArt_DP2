@echo off
REM ===========================================================================
REM  LensArt Hybrid — one-time venv setup
REM ===========================================================================
cd /d "%~dp0"
title LensArt Hybrid setup

echo.
echo ============================================================
echo   LensArt Hybrid — venv setup (ResNet50 + CLIP)
echo ============================================================
echo.

set "PYCMD="
where py >nul 2>nul && set "PYCMD=py -3"
if "%PYCMD%"=="" where python >nul 2>nul && set "PYCMD=python"
if "%PYCMD%"=="" (
    echo [ERROR] Python is not installed or not on PATH.
    pause & exit /b 1
)

echo [1/3] Creating .venv\
%PYCMD% -m venv .venv
if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )

echo [2/3] Upgrading pip
call .venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )

echo [3/3] Installing CPU-only PyTorch + CLIP + Flask (this takes ~5 minutes)
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )

echo.
echo ============================================================
echo   Setup complete.
echo   Place your trained ResNet50 weights at:
echo     models\resnet_model.pth
echo   Then double-click run.bat to start the app.
echo ============================================================
echo.
pause
