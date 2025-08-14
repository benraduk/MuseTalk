#!/usr/bin/env python3
"""
Test enhanced angle detection on specific frames
"""
import sys
import os
import glob
import cv2
import numpy as np
sys.path.append('.')

def test_specific_frames():
    """Test angle detection on the frames you mentioned"""
    print("🎯 Testing Enhanced Angle Detection on Specific Frames")
    print("=" * 60)
    
    # Test on the frames you mentioned
    test_frames = [
        "data/video/00001000.png",  # The angled frame you mentioned
        "data/video/00000728.png", 
        "data/video/00000001.png"
    ]
    
    # Also check for any available frames
    all_frames = glob.glob("data/video/*.png")
    if all_frames:
        # Add last few frames (likely to be angled based on your description)
        all_frames.sort()
        test_frames.extend(all_frames[-3:])  # Last 3 frames
    
    # Remove duplicates and filter existing files
    test_frames = list(set([f for f in test_frames if os.path.exists(f)]))
    
    if not test_frames:
        print("❌ No test frames found!")
        return False
    
    print(f"🔍 Testing {len(test_frames)} frames:")
    for frame_path in test_frames:
        print(f"  - {frame_path}")
    
    try:
        from musetalk.utils.facefusion_detection import FaceFusionDetector
        
        # Initialize detector
        detector = FaceFusionDetector()
        print(f"\n✅ Detector initialized")
        
        # Test each frame
        results = []
        for i, frame_path in enumerate(test_frames):
            print(f"\n📷 Frame {i+1}: {os.path.basename(frame_path)}")
            
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"  ❌ Could not load frame")
                continue
            
            # Detect faces with enhanced angle detection
            detections = detector.detect_multi_angle(frame)
            
            if detections:
                detection = detections[0]  # First detection
                angle = detection.get('angle', 0)
                confidence = detection.get('confidence', 0)
                source = detection.get('source', 'unknown')
                
                # Get detailed angle analysis if available
                angle_methods = detection.get('angle_methods', {})
                
                print(f"  ✅ Face detected:")
                print(f"     Final angle: {angle}°")
                print(f"     Confidence: {confidence:.2f}")
                print(f"     Source: {source}")
                
                if angle_methods:
                    print(f"     Angle breakdown:")
                    print(f"       - Bbox method: {angle_methods.get('bbox', 'N/A')}°")
                    print(f"       - Landmark method: {angle_methods.get('landmarks', 'N/A')}°") 
                    print(f"       - Visual method: {angle_methods.get('visual', 'N/A')}°")
                
                # Classify the angle
                if angle == 0:
                    orientation = "Frontal"
                elif 315 <= angle <= 360 or 0 <= angle <= 45:
                    orientation = "Nearly frontal"
                elif 45 < angle < 135:
                    orientation = "Right profile/angled"
                elif 135 <= angle <= 225:
                    orientation = "Upside down(?)"
                elif 225 < angle < 315:
                    orientation = "Left profile/angled"
                else:
                    orientation = "Unknown"
                
                print(f"     Classification: {orientation}")
                
                # Store result
                results.append({
                    'frame': os.path.basename(frame_path),
                    'angle': angle,
                    'confidence': confidence,
                    'orientation': orientation,
                    'methods': angle_methods
                })
                
            else:
                print(f"  ❌ No face detected")
                results.append({
                    'frame': os.path.basename(frame_path),
                    'angle': None,
                    'confidence': 0,
                    'orientation': 'No face',
                    'methods': {}
                })
        
        # Summary
        print(f"\n📊 ANGLE DETECTION SUMMARY")
        print("=" * 60)
        
        detected_faces = [r for r in results if r['angle'] is not None]
        angled_faces = [r for r in detected_faces if r['angle'] not in [0, 360]]
        
        print(f"Frames tested: {len(results)}")
        print(f"Faces detected: {len(detected_faces)}")
        print(f"Angled faces found: {len(angled_faces)}")
        
        if angled_faces:
            print(f"\n🎯 Angled faces detected:")
            for result in angled_faces:
                print(f"  {result['frame']}: {result['angle']}° ({result['orientation']})")
        
        # Check for the specific frame you mentioned
        frame_1000_result = next((r for r in results if '1000' in r['frame']), None)
        if frame_1000_result:
            print(f"\n🔍 Frame 1000 analysis:")
            if frame_1000_result['angle'] and frame_1000_result['angle'] != 0:
                print(f"  ✅ SUCCESS: Detected angle {frame_1000_result['angle']}° (was showing 0° before)")
            else:
                print(f"  ⚠️  Still showing {frame_1000_result['angle']}° - may need further tuning")
        
        return len(angled_faces) > 0
        
    except Exception as e:
        print(f"❌ Enhanced angle detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_angle_detection(frame_path: str):
    """Create a visual debug image showing angle detection"""
    print(f"\n🎨 Creating visual debug for: {frame_path}")
    
    if not os.path.exists(frame_path):
        print(f"❌ Frame not found: {frame_path}")
        return
    
    try:
        from musetalk.utils.facefusion_detection import FaceFusionDetector
        
        # Load frame
        frame = cv2.imread(frame_path)
        detector = FaceFusionDetector()
        
        # Detect with angle analysis
        detections = detector.detect_multi_angle(frame)
        
        if detections:
            detection = detections[0]
            bbox = detection['bbox']
            angle = detection.get('angle', 0)
            angle_methods = detection.get('angle_methods', {})
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw angle indicator
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw angle line
            angle_rad = np.radians(angle)
            line_length = min(x2 - x1, y2 - y1) // 3
            end_x = int(center_x + line_length * np.cos(angle_rad))
            end_y = int(center_y + line_length * np.sin(angle_rad))
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (255, 0, 0), 3)
            
            # Add text annotations
            cv2.putText(frame, f"Angle: {angle}°", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add method breakdown
            y_offset = y2 + 20
            for method, method_angle in angle_methods.items():
                if method != 'final':
                    cv2.putText(frame, f"{method}: {method_angle}°", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += 20
            
            # Save debug image
            debug_path = frame_path.replace('.png', '_angle_debug.png')
            cv2.imwrite(debug_path, frame)
            print(f"✅ Debug image saved: {debug_path}")
            
        else:
            print(f"❌ No face detected for visualization")
            
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Run enhanced angle detection tests"""
    print("🚀 ENHANCED ANGLE DETECTION TEST")
    print("=" * 60)
    
    # Test angle detection
    success = test_specific_frames()
    
    # Create visual debug for frame 1000 if it exists
    if os.path.exists("data/video/00001000.png"):
        visualize_angle_detection("data/video/00001000.png")
    
    # Summary
    print(f"\n" + "=" * 60)
    if success:
        print(f"🎉 Enhanced angle detection is working!")
        print(f"   - Detected angled faces that were previously showing 0°")
        print(f"   - Ready to proceed with Phase 3 rotation normalization")
    else:
        print(f"⚠️  Angle detection needs further tuning")
        print(f"   - May need to adjust heuristics or add more detection methods")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
