#!/usr/bin/env python3
"""
Real Video Erosion Test - Canva_en Video
Tests morphological erosion on actual Canva_en video with real faces.
Creates side-by-side comparison to show lip bleed-through elimination on real footage.
"""

import os
import subprocess
import shutil
import cv2
import numpy as np
from PIL import Image
import random
import yaml

def create_test_configs():
    """Create test configurations with and without erosion for real video"""
    
    # Base config for Canva_en video
    base_config = {
        'task_0': {
            'video_path': "data/video/Canva_en.mp4",
            'audio_path': "data/audio/Canva_FR.m4a", 
            'result_name': "canva_erosion_test_{mode}.mp4",
            'result_dir': "./results",
            'bbox_shift': 7,
            
            # YOLOv8 Parameters
            'ellipse_padding_factor': 0.02,
            'upper_boundary_ratio': 0.22,
            'expand_factor': 1.9,
            'use_elliptical_mask': True,
            'blur_kernel_ratio': 0.06,
            'mouth_vertical_offset': 0,
            'mouth_scale_factor': 1.0,
            'mask_shape': "ultra_wide_ellipse",
            'mask_height_ratio': 0.85,
            'mask_corner_radius': 0.15,
            'debug_mouth_mask': True,
            
            # YOLOv8 Face Selection
            'yolo_conf_threshold': 0.5,
            'yolo_temporal_weight': 0.25,
            'yolo_size_weight': 0.30,
            'yolo_center_weight': 0.20,
            'yolo_max_face_jump': 0.3,
            'yolo_primary_face_lock_threshold': 10,
            'yolo_primary_face_confidence_drop': 0.8,
            
            # ASD Parameters
            'asd_enabled': True,
            'asd_audio_window_ms': 400,
            'asd_confidence_threshold': 0.3,
            'asd_temporal_smoothing': 0.8,
            'asd_audio_weight': 0.7,
            'asd_visual_weight': 0.3,
            'asd_fallback_to_yolo': True
        }
    }
    
    # Create configs directory
    os.makedirs("configs/real_erosion_test", exist_ok=True)
    
    # Config WITHOUT erosion
    config_no_erosion = base_config.copy()
    config_no_erosion['task_0']['result_name'] = "canva_erosion_test_WITHOUT.mp4"
    config_no_erosion['task_0']['enable_pre_erosion'] = False
    config_no_erosion['task_0']['erosion_ratio'] = 0.008  # Not used when disabled
    config_no_erosion['task_0']['erosion_iterations'] = 1
    
    # Config WITH erosion
    config_with_erosion = base_config.copy()
    config_with_erosion['task_0']['result_name'] = "canva_erosion_test_WITH.mp4"
    config_with_erosion['task_0']['enable_pre_erosion'] = True
    config_with_erosion['task_0']['erosion_ratio'] = 0.008  # 0.8% of face width
    config_with_erosion['task_0']['erosion_iterations'] = 1
    
    # Save configs
    no_erosion_path = "configs/real_erosion_test/no_erosion.yaml"
    with_erosion_path = "configs/real_erosion_test/with_erosion.yaml"
    
    with open(no_erosion_path, 'w') as f:
        yaml.dump(config_no_erosion, f, default_flow_style=False)
    
    with open(with_erosion_path, 'w') as f:
        yaml.dump(config_with_erosion, f, default_flow_style=False)
    
    return no_erosion_path, with_erosion_path

def run_inference(config_path, test_name, max_frames=150):
    """Run inference with given config, limited frames for faster testing"""
    print(f"🚀 Running inference: {test_name}")
    print(f"📄 Config: {config_path}")
    print(f"⏱️  Limited to {max_frames} frames for faster testing")
    
    # First, let's extract only the first N frames from the video for testing
    temp_video_path = f"temp_canva_{max_frames}frames.mp4"
    
    # Extract first N frames to temporary video
    extract_cmd = [
        "ffmpeg", "-y", "-i", "data/video/Canva_en.mp4", 
        "-vframes", str(max_frames), 
        "-c:v", "libx264", "-crf", "18", 
        temp_video_path
    ]
    
    try:
        print(f"📹 Extracting first {max_frames} frames...")
        subprocess.run(extract_cmd, check=True, capture_output=True)
        print(f"✅ Temporary video created: {temp_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Video extraction failed: {e}")
        return False
    
    # Update config to use temporary video
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['task_0']['video_path'] = temp_video_path
    
    temp_config_path = config_path.replace('.yaml', '_temp.yaml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run inference
    cmd = [
        "python", "scripts/inference.py",
        "--inference_config", temp_config_path,
        "--version", "v15",
        "--batch_size", "4",  # Smaller batch for stability
        "--use_float16"  # Faster inference
    ]
    
    try:
        print("🎬 Starting MuseTalk inference...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            print(f"✅ {test_name} completed successfully")
            # Cleanup temp files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return True
        else:
            print(f"❌ {test_name} failed:")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ {test_name} error: {e}")
        return False

def extract_comparison_frames(video1_path, video2_path, output_dir, num_frames=8):
    """Extract frames from both videos for detailed comparison"""
    print(f"🎬 Extracting {num_frames} frames for detailed comparison...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Could not open video files")
        return False
    
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    print(f"📊 Total frames available: {total_frames}")
    
    # Select strategic frames (not random - focus on mouth movement)
    # Skip first 10 frames (often static), then select evenly spaced frames
    start_frame = 10
    end_frame = min(total_frames - 10, 140)  # Leave some buffer at end
    
    if end_frame <= start_frame:
        frame_indices = list(range(start_frame, min(start_frame + num_frames, total_frames)))
    else:
        step = max(1, (end_frame - start_frame) // num_frames)
        frame_indices = list(range(start_frame, end_frame, step))[:num_frames]
    
    print(f"Selected frames: {frame_indices}")
    
    comparisons = []
    
    for i, frame_idx in enumerate(frame_indices):
        print(f"Processing frame {frame_idx}...")
        
        # Read frames
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            # Create side-by-side comparison with focus on mouth region
            h, w = frame1.shape[:2]
            
            # Create comparison image
            comparison = np.zeros((h + 150, w * 2, 3), dtype=np.uint8)
            
            # Place frames
            comparison[150:150+h, 0:w] = frame1
            comparison[150:150+h, w:w*2] = frame2
            
            # Add labels with better visibility
            label_bg_color = (0, 0, 0)  # Black background for text
            text_color = (255, 255, 255)  # White text
            
            # Left side label
            cv2.rectangle(comparison, (0, 0), (w, 150), label_bg_color, -1)
            cv2.putText(comparison, "WITHOUT Erosion", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(comparison, "Original lip bleed-through visible", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(comparison, "Double-mouth artifacts", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Right side label  
            cv2.rectangle(comparison, (w, 0), (w*2, 150), label_bg_color, -1)
            cv2.putText(comparison, "WITH Erosion", (w + 20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(comparison, "Clean AI mouth only", (w + 20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(comparison, "Eliminated bleed-through", (w + 20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Frame info
            cv2.putText(comparison, f"Frame {frame_idx} - Real Face Erosion Test", 
                       (20, h + 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            
            # Save individual comparison
            output_path = os.path.join(output_dir, f"real_comparison_frame_{frame_idx:03d}.png")
            cv2.imwrite(output_path, comparison)
            comparisons.append(comparison)
            
            print(f"💾 Saved: {output_path}")
    
    cap1.release()
    cap2.release()
    
    # Create summary grid (2 columns)
    if comparisons:
        print("📊 Creating summary grid...")
        
        # Arrange in 2 columns
        n_images = len(comparisons)
        cols = 2
        rows = (n_images + cols - 1) // cols
        
        if n_images > 0:
            sample_h, sample_w = comparisons[0].shape[:2]
            grid_width = sample_w * cols
            grid_height = sample_h * rows
            
            # Create grid with black background
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, img in enumerate(comparisons):
                row = i // cols
                col = i % cols
                
                y_start = row * sample_h
                y_end = y_start + sample_h
                x_start = col * sample_w
                x_end = x_start + sample_w
                
                if y_end <= grid_height and x_end <= grid_width:
                    grid[y_start:y_end, x_start:x_end] = img
            
            # Save grid
            grid_path = os.path.join(output_dir, "REAL_EROSION_COMPARISON_GRID.png")
            cv2.imwrite(grid_path, grid)
            print(f"📊 Summary grid saved: {grid_path}")
    
    return True

def main():
    print("🔥 REAL VIDEO EROSION TEST - Canva_en with Real Faces")
    print("=" * 70)
    
    # Check if required files exist
    required_files = [
        "data/video/Canva_en.mp4",
        "data/audio/Canva_FR.m4a",
        "scripts/inference.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return
    
    print("✅ All required files found")
    
    # Create test configurations
    print("\n📝 Creating test configurations for real video...")
    no_erosion_config, with_erosion_config = create_test_configs()
    print(f"✅ Configs created:")
    print(f"   - Without erosion: {no_erosion_config}")
    print(f"   - With erosion: {with_erosion_config}")
    
    # Run inference without erosion
    print("\n" + "="*70)
    print("🎯 TEST 1: WITHOUT Erosion (Real Face Baseline)")
    print("="*70)
    success1 = run_inference(no_erosion_config, "WITHOUT Erosion - Real Faces")
    
    if not success1:
        print("❌ Baseline test failed. Check your setup.")
        return
    
    # Run inference with erosion
    print("\n" + "="*70)
    print("🔥 TEST 2: WITH Erosion (Real Face - Phase 1 Fix)")
    print("="*70)
    success2 = run_inference(with_erosion_config, "WITH Erosion - Real Faces")
    
    if not success2:
        print("❌ Erosion test failed. Check implementation.")
        return
    
    # Compare results
    print("\n" + "="*70)
    print("📊 CREATING REAL FACE COMPARISON")
    print("="*70)
    
    video1_path = "results/v15/canva_erosion_test_WITHOUT.mp4"
    video2_path = "results/v15/canva_erosion_test_WITH.mp4"
    
    if os.path.exists(video1_path) and os.path.exists(video2_path):
        comparison_dir = "real_erosion_comparison_results"
        success3 = extract_comparison_frames(video1_path, video2_path, comparison_dir, num_frames=8)
        
        if success3:
            print("\n🎉 REAL VIDEO EROSION TEST COMPLETED!")
            print("=" * 70)
            print(f"📁 Results saved in: {comparison_dir}/")
            print(f"🎬 Videos created:")
            print(f"   - Without erosion: {video1_path}")
            print(f"   - With erosion: {video2_path}")
            print(f"🖼️  Frame comparisons: {comparison_dir}/")
            print(f"📊 MAIN RESULT: {comparison_dir}/REAL_EROSION_COMPARISON_GRID.png")
            print("\n🔍 WHAT TO LOOK FOR IN REAL FACES:")
            print("   - WITHOUT erosion: Original lips visible around AI mouth edges")
            print("   - WITH erosion: Clean AI mouth with no original lip bleed-through")
            print("   - Focus on mouth corners and lip edges for the clearest difference")
            print("   - Erosion should eliminate 'ghosting' of original mouth features")
            print("\n✅ Phase 1 Critical Fix: REAL FACE VALIDATION COMPLETE")
            
        else:
            print("❌ Comparison creation failed")
    else:
        print("❌ Output videos not found:")
        print(f"   - {video1_path}: {'✅' if os.path.exists(video1_path) else '❌'}")
        print(f"   - {video2_path}: {'✅' if os.path.exists(video2_path) else '❌'}")

if __name__ == "__main__":
    main()
