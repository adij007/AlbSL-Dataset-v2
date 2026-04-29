@echo off
:: ============================================================
:: AlbSL Pipeline — Dependency Installer to D: Drive
:: Run this ONCE as Administrator before using the pipeline.
:: All packages, cache, and the virtualenv go to D:\albsl_env
:: ============================================================

setlocal EnableDelayedExpansion

set DRIVE=D:
set ENV_DIR=%DRIVE%\albsl_env
set CACHE_DIR=%DRIVE%\pip_cache

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║     AlbSL Pipeline — D: Drive Dependency Installer  ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: ── Check Python ────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Install Python 3.10+ first.
    pause & exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Found Python %PYVER%

:: ── Check D: drive exists ───────────────────────────────────
if not exist %DRIVE%\ (
    echo [ERROR] Drive %DRIVE% not found. Please check the DRIVE variable.
    pause & exit /b 1
)

:: ── Create directories ──────────────────────────────────────
echo.
echo [1/6] Creating directories on %DRIVE%...
mkdir "%ENV_DIR%" 2>nul
mkdir "%CACHE_DIR%" 2>nul
echo       %ENV_DIR%
echo       %CACHE_DIR%

:: ── Create virtualenv on D: ─────────────────────────────────
echo.
echo [2/6] Creating virtual environment at %ENV_DIR%...
if exist "%ENV_DIR%\Scripts\activate.bat" (
    echo       [SKIP] Virtual environment already exists.
) else (
    python -m venv "%ENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo       [OK] Created.
)

:: ── Activate venv ───────────────────────────────────────────
call "%ENV_DIR%\Scripts\activate.bat"
echo.
echo [3/6] Activated virtual environment.
echo       Python: %ENV_DIR%\Scripts\python.exe

:: ── Upgrade pip ─────────────────────────────────────────────
echo.
echo [4/6] Upgrading pip...
pip install --upgrade pip --cache-dir "%CACHE_DIR%" --quiet

:: ── Install Intel PyTorch/XPU stack ───────────────────────────
echo.
echo [5/6] Installing PyTorch + Intel Extension for PyTorch...
echo.

pip install torch torchvision torchaudio intel-extension-for-pytorch ^
    --cache-dir "%CACHE_DIR%"

if %errorlevel% neq 0 (
    echo [WARN] Intel XPU build failed. Trying CPU-only PyTorch...
    pip install torch torchvision torchaudio --cache-dir "%CACHE_DIR%"
)

:: ── Install all other heavy dependencies ────────────────────
echo.
echo [6/6] Installing all pipeline dependencies...
echo       This will take 5-15 minutes depending on connection speed.
echo.

pip install ^
    openvino ^
    openvino-dev[onnx,pytorch] ^
    onnxruntime-openvino ^
    mediapipe ^
    mmpose ^
    mmdet ^
    timm ^
    opencv-python-headless ^
    numpy ^
    ultralytics ^
    albumentations ^
    h5py ^
    tqdm ^
    rich ^
    scipy ^
    scikit-learn ^
    pandas ^
    Pillow ^
    kornia ^
    filterpy ^
    sympy ^
    decord ^
    vidgear ^
    imageio[ffmpeg] ^
    ffmpeg-python ^
    loguru ^
    --cache-dir "%CACHE_DIR%"

if %errorlevel% neq 0 (
    echo [ERROR] Some packages failed. Check output above.
    pause & exit /b 1
)

:: ── Create activation shortcut ──────────────────────────────
echo.
set SCRIPT_DIR=%~dp0
echo Creating activate_albsl.bat shortcut in Dependencies folder...
(
    echo @echo off
    echo call "%ENV_DIR%\Scripts\activate.bat"
    echo echo AlbSL environment activated. Python: %ENV_DIR%
    echo cmd /k
) > "%SCRIPT_DIR%activate_albsl.bat"

:: ── Summary ─────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║                   INSTALLATION DONE                  ║
echo  ╠══════════════════════════════════════════════════════╣
echo  ║  Virtual env  : %ENV_DIR%
echo  ║  Pip cache    : %CACHE_DIR%
echo  ║  Activate     : Dependencies\activate_albsl.bat       ║
echo  ║                                                       ║
echo  ║  Usage:                                               ║
echo  ║    Dependencies\activate_albsl.bat                    ║
echo  ║    python Script\extract_keypoints_v2.py ...          ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

pause
