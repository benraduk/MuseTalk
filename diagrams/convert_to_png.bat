@echo off
REM Mermaid to PNG Converter (Windows)
REM ==================================

echo 🎨 MERMAID TO PNG CONVERTER
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Run the conversion script
echo 🔄 Running conversion script...
python convert_to_png.py

echo.
echo 📋 Alternative methods if conversion failed:
echo.
echo 🌐 Online Converter (Easiest):
echo    1. Go to: https://mermaid.live/
echo    2. Copy content from .mmd files
echo    3. Paste and download PNG
echo.
echo 🔧 VS Code Extension:
echo    1. Install 'Mermaid Markdown Syntax Highlighting'
echo    2. Open .mmd file in VS Code
echo    3. Right-click → Export Mermaid Diagram
echo.

pause