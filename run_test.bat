@echo off
cd /d "%~dp0"
echo Running face detection test...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if the GIF file exists
if not exist "gif\test_SAVEE.gif" (
    echo Error: GIF file not found at gif\test_SAVEE.gif
    pause
    exit /b 1
)

echo Starting face detection...
python functions/get_face_areas.py "gif/test_SAVEE.gif"

if errorlevel 1 (
    echo An error occurred while running the script
) else (
    echo Script completed successfully
)

pause 