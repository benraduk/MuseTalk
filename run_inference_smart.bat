@echo off
REM MuseTalk SMART Inference - Coordinate Caching + Optimized Batching
REM First run: Slow face detection, saves coordinates
REM Subsequent runs: Skip face detection, use cached coordinates = MUCH faster

echo 🧠 MuseTalk SMART Inference (Coordinate Caching Strategy)
echo 📁 Working directory: %CD%
echo 💡 Strategy: Cache face detection results for instant re-runs
echo ============================================================

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Activate conda environment if not already active
if not "%CONDA_DEFAULT_ENV%"=="musetalk" (
    echo 🔄 Activating musetalk environment...
    call conda activate musetalk
)

REM Check if coordinates already exist
set VIDEO_NAME=Canva_en
if exist "results\%VIDEO_NAME%.pkl" (
    echo 💾 Found cached coordinates! Skipping face detection...
    echo ⚡ This run will be MUCH faster!
    python scripts/inference.py --inference_config configs/inference/test.yaml --unet_config ./models/musetalkV15/musetalk.json --batch_size 16 --use_saved_coord
) else (
    echo 🔍 First run: Will perform face detection and cache results
    echo ⏳ This will take ~4 minutes, but future runs will be instant!
    python scripts/inference.py --inference_config configs/inference/test.yaml --unet_config ./models/musetalkV15/musetalk.json --batch_size 16
    echo 💾 Coordinates cached for future runs!
)

echo ============================================================
echo ✅ Smart inference completed!
