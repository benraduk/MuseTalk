#!/usr/bin/env python3
"""
Simple Erosion Test - Direct Function Testing
Tests the morphological erosion feature directly on the blending function.
Creates side-by-side comparison to show lip bleed-through elimination.
"""

import os
import cv2
import numpy as np
from PIL import Image
import sys

# Add the project root to Python path
sys.path.append('.')

try:
    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    print("✅ Successfully imported MuseTalk modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the MuseTalk directory and conda environment is activated")
    sys.exit(1)

def create_test_data():
    """Create simple test data for erosion testing"""
    print("🎨 Creating test data...")
    
    # Create a simple test image (640x480)
    height, width = 480, 640
    test_image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add a face-like region
    face_center_x, face_center_y = width // 2, height // 2
    face_width, face_height = 200, 250
    
    # Draw face outline
    cv2.rectangle(test_image, 
                 (face_center_x - face_width//2, face_center_y - face_height//2),
                 (face_center_x + face_width//2, face_center_y + face_height//2),
                 (180, 150, 120), -1)  # Skin color
    
    # Add original mouth (this should be eliminated by erosion)
    mouth_y = face_center_y + 30
    cv2.ellipse(test_image, (face_center_x, mouth_y), (40, 15), 0, 0, 360, (120, 80, 80), -1)
    
    # Create AI face (what we want to blend in)
    ai_face = np.ones((face_height, face_width, 3), dtype=np.uint8) * 180
    # Add AI mouth (different color to show the difference)
    ai_mouth_y = face_height//2 + 30
    cv2.ellipse(ai_face, (face_width//2, ai_mouth_y), (35, 12), 0, 0, 360, (80, 120, 160), -1)
    
    # Face bounding box
    bbox = [face_center_x - face_width//2, face_center_y - face_height//2,
            face_center_x + face_width//2, face_center_y + face_height//2]
    
    # Create dummy landmarks (5-point format)
    landmarks = [
        (face_center_x - 40, face_center_y - 40),  # left eye
        (face_center_x + 40, face_center_y - 40),  # right eye  
        (face_center_x, face_center_y),            # nose
        (face_center_x - 25, mouth_y),             # left mouth
        (face_center_x + 25, mouth_y)              # right mouth
    ]
    
    return test_image, ai_face, bbox, landmarks

def test_erosion_effect():
    """Test the erosion effect with side-by-side comparison"""
    print("🔥 Testing morphological erosion effect...")
    
    # Create test data
    original_image, ai_face, bbox, landmarks = create_test_data()
    
    # Initialize face parser
    try:
        fp = FaceParsing()
        print("✅ Face parser initialized")
    except Exception as e:
        print(f"❌ Face parser initialization failed: {e}")
        return False
    
    # Test parameters
    test_params = {
        'upper_boundary_ratio': 0.22,
        'expand': 1.9,
        'mode': 'jaw',
        'use_elliptical_mask': True,
        'ellipse_padding_factor': 0.02,
        'blur_kernel_ratio': 0.06,
        'landmarks': landmarks,
        'mouth_vertical_offset': 0.0,
        'mouth_scale_factor': 1.0,
        'debug_mouth_mask': True,
        'mask_shape': 'ultra_wide_ellipse',
        'mask_height_ratio': 0.85,
        'mask_corner_radius': 0.15
    }
    
    results = []
    labels = []
    
    # Test without erosion
    print("🔍 Testing WITHOUT erosion...")
    try:
        result_no_erosion = get_image(
            original_image, ai_face, bbox,
            fp=fp,
            enable_pre_erosion=False,
            erosion_ratio=0.008,
            erosion_iterations=1,
            debug_frame_idx=0,
            debug_output_dir="debug_erosion_test",
            **test_params
        )
        results.append(result_no_erosion)
        labels.append("WITHOUT Erosion\n(Original lip visible)")
        print("✅ No erosion test completed")
    except Exception as e:
        print(f"❌ No erosion test failed: {e}")
        return False
    
    # Test with erosion
    print("🔥 Testing WITH erosion...")
    try:
        result_with_erosion = get_image(
            original_image, ai_face, bbox,
            fp=fp,
            enable_pre_erosion=True,
            erosion_ratio=0.008,
            erosion_iterations=1,
            debug_frame_idx=1,
            debug_output_dir="debug_erosion_test",
            **test_params
        )
        results.append(result_with_erosion)
        labels.append("WITH Erosion\n(Clean AI mouth)")
        print("✅ Erosion test completed")
    except Exception as e:
        print(f"❌ Erosion test failed: {e}")
        return False
    
    # Create comparison image
    print("📊 Creating comparison...")
    h, w = original_image.shape[:2]
    comparison_width = w * len(results)
    comparison_height = h + 120  # Extra space for labels
    
    comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
    
    # Place images side by side
    for i, (result, label) in enumerate(zip(results, labels)):
        x_offset = i * w
        comparison[120:120+h, x_offset:x_offset+w] = result
        
        # Add labels
        lines = label.split('\n')
        cv2.putText(comparison, lines[0], (x_offset + 10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        if len(lines) > 1:
            cv2.putText(comparison, lines[1], (x_offset + 10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add title
    cv2.putText(comparison, "Morphological Erosion Test - Phase 1 Critical Fix", 
               (10, comparison_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save results
    os.makedirs("erosion_test_results", exist_ok=True)
    
    # Save individual results
    cv2.imwrite("erosion_test_results/original_image.png", original_image)
    cv2.imwrite("erosion_test_results/ai_face.png", ai_face)
    cv2.imwrite("erosion_test_results/result_no_erosion.png", result_no_erosion)
    cv2.imwrite("erosion_test_results/result_with_erosion.png", result_with_erosion)
    cv2.imwrite("erosion_test_results/comparison.png", comparison)
    
    print("💾 Results saved:")
    print("   - erosion_test_results/original_image.png")
    print("   - erosion_test_results/ai_face.png") 
    print("   - erosion_test_results/result_no_erosion.png")
    print("   - erosion_test_results/result_with_erosion.png")
    print("   - erosion_test_results/comparison.png")
    
    return True

def main():
    print("🔥 Simple Morphological Erosion Test")
    print("=" * 50)
    print("Testing the erosion feature directly on blending function")
    print("=" * 50)
    
    success = test_erosion_effect()
    
    if success:
        print("\n🎉 EROSION TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📁 Check erosion_test_results/ for output images")
        print("📊 View comparison.png to see the difference")
        print("\n🔍 WHAT TO LOOK FOR:")
        print("   - WITHOUT erosion: Original mouth should be visible around AI mouth")
        print("   - WITH erosion: Only AI mouth should be visible (clean overlay)")
        print("   - Erosion eliminates the 'double mouth' bleed-through effect")
        print("\n✅ Phase 1 Critical Fix: Morphological Erosion - IMPLEMENTED")
    else:
        print("\n❌ EROSION TEST FAILED")
        print("Check the error messages above for debugging")

if __name__ == "__main__":
    main()
