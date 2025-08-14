#!/usr/bin/env python3
"""
Test Phase 2: Enhanced FaceFusion face detection
"""
import sys
import os
import glob
import cv2
sys.path.append('.')

def test_facefusion_detector_standalone():
    """Test the FaceFusion detector in isolation"""
    print("🧪 Testing FaceFusion Detector (Standalone)")
    print("=" * 50)
    
    try:
        from musetalk.utils.facefusion_detection import FaceFusionDetector, test_facefusion_detector
        
        # Test detector initialization
        detector = test_facefusion_detector()
        
        # Test on a sample image if available
        test_images = glob.glob("data/video/*.png") + glob.glob("data/video/*.jpg") + glob.glob("assets/demo/*/*.png")
        
        if test_images:
            print(f"\n🔍 Testing detection on sample image: {test_images[0]}")
            frame = cv2.imread(test_images[0])
            
            if frame is not None:
                detections = detector.detect_multi_angle(frame)
                print(f"✅ Detection successful: {len(detections)} faces found")
                
                for i, detection in enumerate(detections):
                    print(f"  Face {i+1}: confidence={detection['confidence']:.2f}, source={detection['source']}")
            else:
                print("⚠️  Could not load test image")
        else:
            print("⚠️  No test images found - detector initialized but not tested on real images")
            
        return True
        
    except Exception as e:
        print(f"❌ FaceFusion detector test failed: {e}")
        return False

def test_phase2_preprocessing():
    """Test the Phase 2 preprocessing integration"""
    print("\n🧪 Testing Phase 2 Preprocessing Integration")
    print("=" * 50)
    
    # Find test images
    test_patterns = [
        "data/video/*.png",
        "data/video/*.jpg", 
        "assets/demo/*/*.png",
        "assets/demo/*/*.jpg"
    ]
    
    test_images = []
    for pattern in test_patterns:
        test_images.extend(glob.glob(pattern))
    
    if not test_images:
        print("❌ No test images found!")
        print("💡 Create test images with: ffmpeg -i your_video.mp4 -vframes 5 data/video/%08d.png")
        return False
    
    # Take first 3 images for quick test
    test_images = test_images[:3]
    print(f"🎯 Testing on {len(test_images)} images:")
    for img in test_images:
        print(f"  - {img}")
    
    try:
        from musetalk.utils.preprocessing import get_landmark_and_bbox_phase2
        
        print(f"\n🚀 Running Phase 2 enhanced detection...")
        coords_list, frames, metadata = get_landmark_and_bbox_phase2(test_images)
        
        # Analyze results
        detected_faces = sum(1 for coord in coords_list if coord != (0.0, 0.0, 0.0, 0.0))
        
        print(f"\n✅ Phase 2 Test Results:")
        print(f"   - Frames processed: {len(frames)}")
        print(f"   - Faces detected: {detected_faces}/{len(frames)}")
        print(f"   - Detection rate: {detected_faces/len(frames)*100:.1f}%")
        print(f"   - Metadata entries: {len(metadata)}")
        
        # Show detection details
        print(f"\n📋 Detection Details:")
        for i, meta in enumerate(metadata[:5]):  # First 5 frames
            coord = coords_list[i]
            if coord != (0.0, 0.0, 0.0, 0.0):
                print(f"   Frame {i}: {meta['orientation']}, angle={meta['angle']}°, confidence={meta['confidence']:.2f}, source={meta['source']}")
            else:
                print(f"   Frame {i}: No face detected")
        
        # Test success criteria
        if detected_faces > 0:
            print(f"\n🎉 Phase 2 test PASSED!")
            return True
        else:
            print(f"\n⚠️  Phase 2 test completed but no faces detected")
            return False
            
    except Exception as e:
        print(f"❌ Phase 2 preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_files():
    """Test that FaceFusion model files are present"""
    print("\n🧪 Testing FaceFusion Model Files")
    print("=" * 50)
    
    model_files = [
        "models/facefusion/scrfd_2.5g.onnx",
        "models/facefusion/scrfd_2.5g.hash",
        "models/facefusion/yoloface_8n.onnx", 
        "models/facefusion/yoloface_8n.hash",
        "models/facefusion/retinaface_10g.onnx",
        "models/facefusion/retinaface_10g.hash"
    ]
    
    found_models = []
    missing_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            found_models.append(model_file)
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"✅ {model_file} ({file_size:.1f} MB)")
        else:
            missing_models.append(model_file)
            print(f"❌ {model_file} - NOT FOUND")
    
    print(f"\nModel Status: {len(found_models)}/{len(model_files)} files found")
    
    if missing_models:
        print(f"\n⚠️  Missing models - run download script:")
        print(f"   Windows: download_weights.bat")
        print(f"   Linux/Mac: ./download_weights.sh")
        return False
    else:
        print(f"🎉 All FaceFusion models present!")
        return True

def main():
    """Run all Phase 2 tests"""
    print("🚀 PHASE 2 TESTING SUITE")
    print("=" * 60)
    
    # Test 1: Model files
    models_ok = test_model_files()
    
    # Test 2: FaceFusion detector
    detector_ok = test_facefusion_detector_standalone()
    
    # Test 3: Preprocessing integration  
    preprocessing_ok = test_phase2_preprocessing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 2 TEST SUMMARY")
    print("=" * 60)
    print(f"Model files: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"FaceFusion detector: {'✅ PASS' if detector_ok else '❌ FAIL'}")
    print(f"Preprocessing integration: {'✅ PASS' if preprocessing_ok else '❌ FAIL'}")
    
    if models_ok and detector_ok and preprocessing_ok:
        print(f"\n🎉 PHASE 2 READY - All tests passed!")
        print(f"📋 Next steps:")
        print(f"   1. Test on your problematic videos")
        print(f"   2. Compare detection rates vs original")
        print(f"   3. Begin Phase 3 rotation normalization")
    elif models_ok and detector_ok:
        print(f"\n⚠️  PHASE 2 PARTIALLY READY")
        print(f"   - Models and detector work")
        print(f"   - Need to fix preprocessing integration")
    else:
        print(f"\n❌ PHASE 2 NOT READY")
        print(f"   - Fix failing components before proceeding")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
