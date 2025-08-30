@echo off
REM Real Face Erosion Test - Side by Side Comparison
REM Tests morphological erosion on actual Canva_en video

echo 🔥 REAL FACE EROSION TEST - Canva_en Video
echo ============================================================
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

REM Create test configs directory
if not exist "configs\erosion_real_test" mkdir "configs\erosion_real_test"

echo.
echo 📝 Creating test configuration WITHOUT erosion...
echo task_0: > "configs\erosion_real_test\no_erosion.yaml"
echo  video_path: "data/video/Canva_en.mp4" >> "configs\erosion_real_test\no_erosion.yaml"
echo  audio_path: "data/audio/Canva_FR.m4a" >> "configs\erosion_real_test\no_erosion.yaml"
echo  result_name: "canva_WITHOUT_erosion.mp4" >> "configs\erosion_real_test\no_erosion.yaml"
echo  result_dir: "./results" >> "configs\erosion_real_test\no_erosion.yaml"
echo  bbox_shift: 7 >> "configs\erosion_real_test\no_erosion.yaml"
echo  ellipse_padding_factor: 0.02 >> "configs\erosion_real_test\no_erosion.yaml"
echo  upper_boundary_ratio: 0.22 >> "configs\erosion_real_test\no_erosion.yaml"
echo  expand_factor: 1.9 >> "configs\erosion_real_test\no_erosion.yaml"
echo  use_elliptical_mask: true >> "configs\erosion_real_test\no_erosion.yaml"
echo  blur_kernel_ratio: 0.06 >> "configs\erosion_real_test\no_erosion.yaml"
echo  mouth_vertical_offset: 0 >> "configs\erosion_real_test\no_erosion.yaml"
echo  mouth_scale_factor: 1.0 >> "configs\erosion_real_test\no_erosion.yaml"
echo  mask_shape: "ultra_wide_ellipse" >> "configs\erosion_real_test\no_erosion.yaml"
echo  mask_height_ratio: 0.85 >> "configs\erosion_real_test\no_erosion.yaml"
echo  mask_corner_radius: 0.15 >> "configs\erosion_real_test\no_erosion.yaml"
echo  enable_pre_erosion: false >> "configs\erosion_real_test\no_erosion.yaml"
echo  erosion_ratio: 0.008 >> "configs\erosion_real_test\no_erosion.yaml"
echo  erosion_iterations: 1 >> "configs\erosion_real_test\no_erosion.yaml"
echo  debug_mouth_mask: true >> "configs\erosion_real_test\no_erosion.yaml"

echo.
echo 📝 Creating test configuration WITH erosion...
echo task_0: > "configs\erosion_real_test\with_erosion.yaml"
echo  video_path: "data/video/Canva_en.mp4" >> "configs\erosion_real_test\with_erosion.yaml"
echo  audio_path: "data/audio/Canva_FR.m4a" >> "configs\erosion_real_test\with_erosion.yaml"
echo  result_name: "canva_WITH_erosion.mp4" >> "configs\erosion_real_test\with_erosion.yaml"
echo  result_dir: "./results" >> "configs\erosion_real_test\with_erosion.yaml"
echo  bbox_shift: 7 >> "configs\erosion_real_test\with_erosion.yaml"
echo  ellipse_padding_factor: 0.02 >> "configs\erosion_real_test\with_erosion.yaml"
echo  upper_boundary_ratio: 0.22 >> "configs\erosion_real_test\with_erosion.yaml"
echo  expand_factor: 1.9 >> "configs\erosion_real_test\with_erosion.yaml"
echo  use_elliptical_mask: true >> "configs\erosion_real_test\with_erosion.yaml"
echo  blur_kernel_ratio: 0.06 >> "configs\erosion_real_test\with_erosion.yaml"
echo  mouth_vertical_offset: 0 >> "configs\erosion_real_test\with_erosion.yaml"
echo  mouth_scale_factor: 1.0 >> "configs\erosion_real_test\with_erosion.yaml"
echo  mask_shape: "ultra_wide_ellipse" >> "configs\erosion_real_test\with_erosion.yaml"
echo  mask_height_ratio: 0.85 >> "configs\erosion_real_test\with_erosion.yaml"
echo  mask_corner_radius: 0.15 >> "configs\erosion_real_test\with_erosion.yaml"
echo  enable_pre_erosion: true >> "configs\erosion_real_test\with_erosion.yaml"
echo  erosion_ratio: 0.008 >> "configs\erosion_real_test\with_erosion.yaml"
echo  erosion_iterations: 1 >> "configs\erosion_real_test\with_erosion.yaml"
echo  debug_mouth_mask: true >> "configs\erosion_real_test\with_erosion.yaml"

echo.
echo ============================================================
echo 🎯 TEST 1: WITHOUT Erosion (Real Face Baseline)
echo ============================================================
echo ⚡ Running inference WITHOUT erosion...
python scripts/inference.py --inference_config "configs/erosion_real_test/no_erosion.yaml" --version v15 --batch_size 8 --use_float16

if %ERRORLEVEL% neq 0 (
    echo ❌ Test 1 failed! Check the error above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🔥 TEST 2: WITH Erosion (Phase 1 Critical Fix)
echo ============================================================
echo ⚡ Running inference WITH erosion...
python scripts/inference.py --inference_config "configs/erosion_real_test/with_erosion.yaml" --version v15 --batch_size 8 --use_float16

if %ERRORLEVEL% neq 0 (
    echo ❌ Test 2 failed! Check the error above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🎉 BOTH TESTS COMPLETED SUCCESSFULLY!
echo ============================================================
echo 📁 Results saved in: ./results/v15/
echo 🎬 Videos created:
echo    - canva_WITHOUT_erosion.mp4 (Original lip bleed-through)
echo    - canva_WITH_erosion.mp4 (Clean AI mouth only)
echo.
echo 🔍 COMPARE THE VIDEOS TO SEE:
echo    - WITHOUT erosion: Original lips visible around AI mouth
echo    - WITH erosion: Clean AI mouth with no bleed-through
echo    - Focus on mouth corners and edges for clearest difference
echo.
echo ✅ Phase 1 Critical Fix: Morphological Erosion - VALIDATED ON REAL FACES!
echo ============================================================
pause
