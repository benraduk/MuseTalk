@echo off
echo ========================================
echo TalkNet + YOLOv8 ASD Testing Suite
echo ========================================
echo.

echo Testing all available videos with TalkNet + YOLOv8...
echo.

echo [1/4] Testing Canva_en.mp4 (100 frames)...
python debug_talknet_yolo.py --input data/video/Canva_en.mp4 --output debug_output/test_canva_talknet.mp4 --max_frames 100

echo.
echo [2/4] Testing braiv_en.mp4 (100 frames)...
python debug_talknet_yolo.py --input data/video/braiv_en.mp4 --output debug_output/test_braiv_talknet.mp4 --max_frames 100

echo.
echo [3/4] Testing yongen.mp4 (100 frames)...
python debug_talknet_yolo.py --input data/video/yongen.mp4 --output debug_output/test_yongen_talknet.mp4 --max_frames 100

echo.
echo [4/4] Testing sun.mp4 (50 frames)...
python debug_talknet_yolo.py --input data/video/sun.mp4 --output debug_output/test_sun_talknet.mp4 --max_frames 50

echo.
echo ========================================
echo Testing Complete!
echo ========================================
echo.
echo Debug videos created in debug_output/:
dir debug_output\test_*_talknet.mp4
echo.
echo Analysis reports:
dir debug_output\test_*_report.json
echo.
echo You can now review the debug videos to see:
echo - Green bounding boxes around detected faces (YOLOv8)
echo - Big "SPEAKING" labels on active speakers (TalkNet)
echo - Frame statistics and confidence scores
echo.
pause
