@echo off
REM MuseTalk OPTIMIZED Inference - Maximum Performance
REM Optimized inference batch sizes + coordinate caching

echo 🚀 MuseTalk OPTIMIZED Inference (High Performance Mode)
echo 📁 Working directory: %CD%
echo 🐍 Setting PYTHONPATH to include current directory
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
echo 📁 Output directory: ./results/optimized/
python scripts/inference.py --enable_gpen_bfr --inference_config configs/inference/test.yaml --unet_config ./models/musetalkV15/musetalk.json --batch_size 16 --result_dir "./results"

REM Restore original directory
echo ============================================================
echo ✅ Inference completed
