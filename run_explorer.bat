@echo off
echo nGPT Explorer - Interactive Learning Suite
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python check_requirements.py
if %errorlevel% neq 0 (
    echo Failed to check requirements.
    echo Please install the required packages manually:
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

REM Test controller connectivity
echo.
echo Testing Xbox controller connectivity...
echo (If no controller is connected, you can use keyboard controls instead)
python test_controller.py

REM Run the explorer
echo.
echo Starting nGPT Explorer...
python ngpt_explorer.py

pause 