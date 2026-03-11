@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  Art to Music - Windows Setup & Launch Script
::  Checks dependencies, sets up environment, trains model,
::  and launches the Streamlit application.
:: ============================================================

:: Color codes (Windows ANSI via PowerShell echo workaround)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%          Art to Music - Setup ^& Launch Script             %RESET%
echo %CYAN%============================================================%RESET%
echo.

:: ------------------------------------------------------------
:: STEP 1: Check Python 3.8+ is installed
:: ------------------------------------------------------------
echo %YELLOW%[INFO] Checking Python installation...%RESET%
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR] Python not found. Please install Python 3.8+ from https://python.org%RESET%
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)

if !PYMAJ! LSS 3 (
    echo %RED%[ERROR] Python 3.8+ required. Found version !PYVER!%RESET%
    exit /b 1
)
if !PYMAJ! EQU 3 (
    if !PYMIN! LSS 8 (
        echo %RED%[ERROR] Python 3.8+ required. Found version !PYVER!%RESET%
        exit /b 1
    )
)
echo %GREEN%[OK] Python !PYVER! detected%RESET%

:: ------------------------------------------------------------
:: STEP 2: Create virtual environment
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Setting up virtual environment...%RESET%
if not exist "venv\" (
    python -m venv venv
    if errorlevel 1 (
        echo %RED%[ERROR] Failed to create virtual environment%RESET%
        exit /b 1
    )
    echo %GREEN%[OK] Virtual environment created%RESET%
) else (
    echo %GREEN%[OK] Virtual environment already exists%RESET%
)

:: ------------------------------------------------------------
:: STEP 3: Activate virtual environment
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Activating virtual environment...%RESET%
call venv\Scripts\activate
if errorlevel 1 (
    echo %RED%[ERROR] Failed to activate virtual environment%RESET%
    exit /b 1
)
echo %GREEN%[OK] Virtual environment activated%RESET%

:: ------------------------------------------------------------
:: STEP 4: Upgrade pip
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Upgrading pip...%RESET%
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo %RED%[ERROR] Failed to upgrade pip%RESET%
    exit /b 1
)
echo %GREEN%[OK] pip upgraded successfully%RESET%

:: ------------------------------------------------------------
:: STEP 5: Install requirements
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Installing requirements (this may take several minutes)...%RESET%
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo %RED%[ERROR] Failed to install requirements. Check requirements.txt and your internet connection.%RESET%
    exit /b 1
)
echo %GREEN%[OK] All requirements installed%RESET%

:: ------------------------------------------------------------
:: STEP 6: Create necessary directories
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Creating project directories...%RESET%
if not exist "data\raw\" mkdir data\raw
if not exist "data\processed\" mkdir data\processed
if not exist "models\" mkdir models
if not exist "assets\" mkdir assets
if not exist "pipeline\" mkdir pipeline
if not exist "scripts\" mkdir scripts
if not exist "notebooks\" mkdir notebooks
echo %GREEN%[OK] Project directories ready%RESET%

:: ------------------------------------------------------------
:: STEP 7: Download YOLOv8 nano weights if not present
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Checking YOLOv8 weights...%RESET%
if not exist "models\yolov8n.pt" (
    echo %YELLOW%[INFO] Downloading YOLOv8 nano weights...%RESET%
    python -c "from ultralytics import YOLO; import shutil, os; m = YOLO('yolov8n.pt'); src = 'yolov8n.pt'; dst = os.path.join('models', 'yolov8n.pt'); shutil.move(src, dst) if os.path.exists(src) else None; print('Weights ready')" 2>&1
    if errorlevel 1 (
        echo %YELLOW%[INFO] YOLOv8 weights will be downloaded automatically on first use%RESET%
    ) else (
        echo %GREEN%[OK] YOLOv8 weights downloaded to models\yolov8n.pt%RESET%
    )
) else (
    echo %GREEN%[OK] YOLOv8 weights already present%RESET%
)

:: ------------------------------------------------------------
:: STEP 8: Copy dataset to data/raw if not present
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Checking dataset...%RESET%
if not exist "data\raw\spotify_dataset.csv" (
    if exist "spotify_dataset.csv" (
        copy "spotify_dataset.csv" "data\raw\spotify_dataset.csv" >nul
        echo %GREEN%[OK] Dataset copied to data\raw\spotify_dataset.csv%RESET%
    ) else (
        echo %YELLOW%[INFO] No dataset found — synthetic data will be generated%RESET%
    )
) else (
    echo %GREEN%[OK] Dataset already present in data\raw\%RESET%
)

:: ------------------------------------------------------------
:: STEP 9: Run dataset cleaning script if needed
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Checking for cleaned dataset...%RESET%
if not exist "data\processed\cleaned_dataset.csv" (
    echo %YELLOW%[INFO] Running dataset cleaning pipeline...%RESET%
    python scripts\clean_dataset.py
    if errorlevel 1 (
        echo %RED%[ERROR] Dataset cleaning failed. Check scripts\clean_dataset.py%RESET%
        exit /b 1
    )
    echo %GREEN%[OK] Dataset cleaned and saved to data\processed\cleaned_dataset.csv%RESET%
) else (
    echo %GREEN%[OK] Cleaned dataset already exists%RESET%
)

:: ------------------------------------------------------------
:: STEP 10: Train recommendation model if needed
:: ------------------------------------------------------------
echo.
echo %YELLOW%[INFO] Checking for trained recommendation model...%RESET%
if not exist "models\recommender_model.pkl" (
    echo %YELLOW%[INFO] Training recommendation model...%RESET%
    python scripts\train_recommender.py
    if errorlevel 1 (
        echo %RED%[ERROR] Model training failed. Check scripts\train_recommender.py%RESET%
        exit /b 1
    )
    echo %GREEN%[OK] Recommendation model trained and saved to models\recommender_model.pkl%RESET%
) else (
    echo %GREEN%[OK] Trained model already exists%RESET%
)

:: ------------------------------------------------------------
:: STEP 11: Launch Streamlit application
:: ------------------------------------------------------------
echo.
echo %CYAN%============================================================%RESET%
echo %GREEN%[OK] All setup steps completed successfully!%RESET%
echo %CYAN%============================================================%RESET%
echo.
echo %YELLOW%[INFO] Launching Art to Music application...%RESET%
echo %CYAN%[INFO] Open your browser at: http://localhost:8501%RESET%
echo %YELLOW%[INFO] Press Ctrl+C to stop the application%RESET%
echo.

streamlit run app.py --server.port 8501
if errorlevel 1 (
    echo %RED%[ERROR] Failed to launch Streamlit application%RESET%
    exit /b 1
)

endlocal
