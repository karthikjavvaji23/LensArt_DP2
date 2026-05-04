@echo off
REM ===========================================================================
REM  LensArt Hybrid — launcher
REM ===========================================================================
cd /d "%~dp0"
title LensArt Hybrid

echo.
echo ============================================================
echo   LensArt Hybrid — Art Recognition + Discovery
echo   ResNet50 (style)  +  CLIP (artist + similarity)
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No virtual environment found. Run setup_venv.bat first.
    pause & exit /b 1
)
if not exist "models\resnet_model.pth" (
    echo [INFO] No ResNet weights at models\resnet_model.pth — style classification
    echo        will gracefully degrade. CLIP artist + similarity still work.
    echo.
)

start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:5000"

echo Starting LensArt at http://localhost:5000
echo (the browser opens automatically in 8 seconds)
echo Press Ctrl+C in this window to stop the server.
echo.

call .venv\Scripts\python.exe app.py

echo.
echo ============================================================
echo   LensArt server stopped.
echo ============================================================
pause
