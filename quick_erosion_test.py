#!/usr/bin/env python3
"""
Quick Morphological Erosion Test
Tests the erosion feature by running inference on Canva_en video with and without erosion.
Creates comparison outputs to demonstrate lip bleed-through elimination.
"""

import os
import subprocess
import shutil
import cv2
import numpy as np
from PIL import Image
import random

def create_test_configs():
    """Create test configurations with and without erosion"""
    
    # Base config
    base_config = """task_0:
 video_path: "data/video/Canva_en.mp4"
 audio_path: "data/audio/Canva_FR.m4a"
 result_name: "erosion_test_{mode}.mp4"
 result_dir: "./results"
 bbox_shift: 7
 
 # YOLOv8 Surgical Mouth Positioning Parameters - NATURAL SIZING
 ellipse_padding_factor: 0.02      # ULTRA coverage (smaller = larger coverage) 
 upper_boundary_ratio: 0.22        # Maximum face coverage (smaller = more coverage)  
 expand_factor: 1.9                # Generous face crop expansion for edge cases
 use_elliptical_mask: true         # Use elliptical mask (recommended)
 blur_kernel_ratio: 0.06           # Optimal mask smoothing for natural blending
 
 # Precise Mouth Positioning & Sizing - NATURAL SIZING
 mouth_vertical_offset: 0          # Vertical mouth position (0 = perfect YOLOv8 positioning)
 mouth_scale_factor: 1.0           # NO scaling - use natural YOLOv8 detected mouth size

 # Advanced Mask Shape Options - ULTRA WIDE COVERAGE
 mask_shape: "ultra_wide_ellipse"        # BEST: 80% wider than standard ellipse
 mask_height_ratio: 0.85           # Slightly reduced for more natural proportions
 mask_corner_radius: 0.15          # Softer corners for better blending

 # Morphological Erosion Parameters (NEW - Phase 1 Critical Fix)
{erosion_config}

 # Debug Options
 debug_mouth_mask: true            # Enable debug outputs for troubleshooting

 # YOLOv8 Face Selection Parameters (NEW)
 yolo_conf_threshold: 0.5          # Minimum confidence for face detection (0.2-0.8)
 yolo_temporal_weight: 0.25        # Weight for temporal consistency (prevent face switching)
 yolo_size_weight: 0.30            # Weight for face size (prefer larger faces)
 yolo_center_weight: 0.20          # Weight for center position (prefer centered faces)
 yolo_max_face_jump: 0.3           # Maximum allowed face center jump (as fraction of frame width)

  # Primary Face Locking (ANTI-GLITCH PROTECTION)
 yolo_primary_face_lock_threshold: 10  # Lock primary face after N consistent frames (5-20)
 yolo_primary_face_confidence_drop: 0.8  # Allow confidence drop to this ratio before unlocking (0.6-0.9)

 # Active Speaker Detection (ASD) - INTELLIGENT SPEAKER SELECTION
 asd_enabled: true                     # Enable/disable ASD (true/false)
 asd_audio_window_ms: 400             # Audio analysis window in milliseconds (300-600)
 asd_confidence_threshold: 0.3        # Minimum ASD confidence for speaker selection (0.2-0.6)
 asd_temporal_smoothing: 0.8          # Temporal smoothing factor for stability (0.5-0.9)
 asd_audio_weight: 0.7                # Weight for audio features in fusion (0.4-0.8)
 asd_visual_weight: 0.3               # Weight for visual features in fusion (0.2-0.6)
 asd_fallback_to_yolo: true           # Fallback to YOLOv8 selection when ASD uncertain (recommended: true)"""

    # Config without erosion
    no_erosion_config = base_config.format(
        mode="without_erosion",
        erosion_config="""enable_pre_erosion: false             # DISABLED for comparison
erosion_ratio: 0.008                  # Not used when disabled
erosion_iterations: 1                 # Not used when disabled"""
    )
    
    # Config with erosion
    with_erosion_config = base_config.format(
        mode="with_erosion", 
        erosion_config="""enable_pre_erosion: true              # ENABLED - eliminate lip bleed-through
erosion_ratio: 0.008                  # Erosion kernel size as ratio of face width (0.005-0.015)
erosion_iterations: 1                 # Number of erosion iterations (1-2)"""
    )
    
    # Save configs
    os.makedirs("configs/erosion_test", exist_ok=True)
    
    with open("configs/erosion_test/no_erosion.yaml", "w") as f:
        f.write(no_erosion_config)
    
    with open("configs/erosion_test/with_erosion.yaml", "w") as f:
        f.write(with_erosion_config)
    
    return "configs/erosion_test/no_erosion.yaml", "configs/erosion_test/with_erosion.yaml"

def run_inference(config_path, test_name):
    """Run inference with given config"""
    print(f"🚀 Running inference: {test_name}")
    print(f"📄 Config: {config_path}")
    
    cmd = [
        "python", "scripts/inference.py",
        "--inference_config", config_path,
        "--version", "v15",
        "--batch_size", "8",  # Smaller batch for stability
        "--use_float16"  # Faster inference
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print(f"✅ {test_name} completed successfully")
            return True
        else:
            print(f"❌ {test_name} failed:")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ {test_name} error: {e}")
        return False

def extract_comparison_frames(video1_path, video2_path, output_dir, num_frames=5):
    """Extract frames from both videos for comparison"""
    print(f"🎬 Extracting {num_frames} frames for comparison...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Could not open video files")
        return False
    
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Select random frames
    random.seed(42)
    frame_indices = sorted(random.sample(range(10, min(total_frames-10, 200)), num_frames))
    
    print(f"Selected frames: {frame_indices}")
    
    comparisons = []
    
    for i, frame_idx in enumerate(frame_indices):
        # Read frames
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            # Create side-by-side comparison
            h, w = frame1.shape[:2]
            comparison = np.zeros((h + 100, w * 2, 3), dtype=np.uint8)
            
            # Place frames
            comparison[100:100+h, 0:w] = frame1
            comparison[100:100+h, w:w*2] = frame2
            
            # Add labels
            cv2.putText(comparison, "WITHOUT Erosion", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(comparison, "(Original lip bleed-through)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.putText(comparison, "WITH Erosion", (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(comparison, "(Clean AI mouth only)", (w + 10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(comparison, f"Frame {frame_idx} - Morphological Erosion Comparison", 
                       (10, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save comparison
            output_path = os.path.join(output_dir, f"comparison_frame_{frame_idx:03d}.png")
            cv2.imwrite(output_path, comparison)
            comparisons.append(comparison)
            
            print(f"💾 Saved comparison: {output_path}")
    
    cap1.release()
    cap2.release()
    
    # Create summary grid
    if comparisons:
        print("📊 Creating summary grid...")
        
        # Stack vertically for easier viewing
        grid = np.vstack(comparisons)
        
        grid_path = os.path.join(output_dir, "erosion_comparison_summary.png")
        cv2.imwrite(grid_path, grid)
        print(f"📊 Summary saved: {grid_path}")
    
    return True

def main():
    print("🔥 Morphological Erosion Test - Phase 1 Critical Fix")
    print("=" * 60)
    
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
    print("\n📝 Creating test configurations...")
    no_erosion_config, with_erosion_config = create_test_configs()
    print(f"✅ Configs created:")
    print(f"   - Without erosion: {no_erosion_config}")
    print(f"   - With erosion: {with_erosion_config}")
    
    # Run inference without erosion
    print("\n" + "="*60)
    print("🎯 TEST 1: WITHOUT Erosion (Baseline)")
    print("="*60)
    success1 = run_inference(no_erosion_config, "WITHOUT Erosion")
    
    if not success1:
        print("❌ Baseline test failed. Check your setup.")
        return
    
    # Run inference with erosion
    print("\n" + "="*60)
    print("🔥 TEST 2: WITH Erosion (Phase 1 Fix)")
    print("="*60)
    success2 = run_inference(with_erosion_config, "WITH Erosion")
    
    if not success2:
        print("❌ Erosion test failed. Check implementation.")
        return
    
    # Compare results
    print("\n" + "="*60)
    print("📊 CREATING COMPARISON")
    print("="*60)
    
    video1_path = "results/v15/erosion_test_without_erosion.mp4"
    video2_path = "results/v15/erosion_test_with_erosion.mp4"
    
    if os.path.exists(video1_path) and os.path.exists(video2_path):
        comparison_dir = "erosion_comparison_results"
        success3 = extract_comparison_frames(video1_path, video2_path, comparison_dir)
        
        if success3:
            print("\n🎉 EROSION TEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📁 Results saved in: {comparison_dir}/")
            print(f"🎬 Videos created:")
            print(f"   - Without erosion: {video1_path}")
            print(f"   - With erosion: {video2_path}")
            print(f"🖼️  Frame comparisons: {comparison_dir}/")
            print(f"📊 Summary image: {comparison_dir}/erosion_comparison_summary.png")
            print("\n🔍 WHAT TO LOOK FOR:")
            print("   - WITHOUT erosion: Original lips visible around AI mouth (bleed-through)")
            print("   - WITH erosion: Clean AI mouth only, no original lip artifacts")
            print("   - Erosion should eliminate the 'double mouth' effect")
            
        else:
            print("❌ Comparison creation failed")
    else:
        print("❌ Output videos not found:")
        print(f"   - {video1_path}: {'✅' if os.path.exists(video1_path) else '❌'}")
        print(f"   - {video2_path}: {'✅' if os.path.exists(video2_path) else '❌'}")

if __name__ == "__main__":
    main()
