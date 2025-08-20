@echo off
REM MuseTalk OPTIMIZED Inference - Maximum Performance
REM Optimized inference batch sizes + coordinate caching

echo 🚀 MuseTalk OPTIMIZED Inference (High Performance Mode)
echo 📁 Working directory: %CD%
echo 🐍 Setting PYTHONPATH to include current directory
echo ⚡ Optimizations: 16 inference batch size + coordinate caching
echo ============================================================

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Activate conda environment if not already active
if not "%CONDA_DEFAULT_ENV%"=="musetalk" (
    echo 🔄 Activating musetalk environment...
    call conda activate musetalk
)

REM Run OPTIMIZED inference with optimized settings
echo ⚡ Running with optimized settings for maximum speed...
echo 📊 Batch size: 16 (AI inference optimization)
echo 💾 Coordinate caching: Enabled (skip face detection on re-runs)
echo 📁 Output directory: ./results/optimized/
python scripts/inference.py --inference_config configs/inference/test.yaml --unet_config ./models/musetalkV15/musetalk.json --batch_size 16 --use_saved_coord --result_dir "./results/optimized"

REM Restore original directory
echo ============================================================
echo ✅ Optimized inference completed (should be 3-4x faster!)
