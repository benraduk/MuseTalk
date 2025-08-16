"""
FaceFusion-inspired face detection for multi-angle processing
Phase 2.1: Multi-angle face detection integration
"""
import cv2
import numpy as np
import os
import torch
from typing import List, Tuple, Optional, Dict, Any
from face_detection import FaceAlignment, LandmarksType

# Try to import ONNX runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available - using fallback detection only")

class FaceFusionDetector:
    """Multi-angle face detector using FaceFusion models"""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.models = {}
        self.model_info = {
            'scrfd': {
                'path': './models/facefusion/scrfd_2.5g.onnx',
                'input_size': (640, 640),
                'description': 'SCRFD - Best for multi-angle detection'
            },
            'yolo_face': {
                'path': './models/facefusion/yoloface_8n.onnx', 
                'input_size': (640, 640),
                'description': 'YOLO_Face - Good for extreme angles'
            },
            'retinaface': {
                'path': './models/facefusion/retinaface_10g.onnx',
                'input_size': (640, 640), 
                'description': 'RetinaFace - High quality for frontal faces'
            }
        }
        
        self.load_models()
        
        # Fallback to original FaceAlignment
        try:
            self.fa_fallback = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
            print("✅ FaceAlignment fallback initialized")
        except Exception as e:
            print(f"⚠️  FaceAlignment fallback failed: {e}")
            self.fa_fallback = None
    
    def load_models(self):
        """Load available FaceFusion ONNX models"""
        if not ONNX_AVAILABLE:
            print("⚠️  ONNX Runtime not available - skipping FaceFusion models")
            return
            
        loaded_count = 0
        
        for model_name, model_config in self.model_info.items():
            model_path = model_config['path']
            
            if os.path.exists(model_path):
                try:
                    # Set up ONNX providers based on device
                    if self.device == "cuda" and torch.cuda.is_available():
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    else:
                        providers = ['CPUExecutionProvider']
                    
                    # Load model
                    session = ort.InferenceSession(model_path, providers=providers)
                    self.models[model_name] = {
                        'session': session,
                        'config': model_config
                    }
                    
                    print(f"✅ {model_name} loaded: {model_config['description']}")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {model_name}: {e}")
            else:
                print(f"⚠️  Model not found: {model_path}")
        
        if loaded_count == 0:
            print("⚠️  No FaceFusion models loaded - will use fallback detection only")
        else:
            print(f"🎉 Successfully loaded {loaded_count} FaceFusion models")
    
    def detect_multi_angle(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces at multiple angles and return best detections with enhanced angle estimation
        Returns: List of {bbox, landmarks, angle, confidence, source}
        """
        all_detections = []
        
        # Try FaceFusion models first (if available)
        if self.models:
            for model_name in self.models.keys():
                detections = self._detect_with_model(frame, model_name, angle=0)
                all_detections.extend(detections)
                
                # If we got good detections, we can stop
                if detections and max(d['confidence'] for d in detections) > 0.7:
                    break
        
        # Re-enabled FaceAlignment fallback since ONNX parsing isn't implemented yet
        # Always try FaceAlignment for better angle detection (even if FaceFusion worked)
        # FaceAlignment gives us better bounding boxes for angle analysis
        fallback_detections = self._detect_with_facealignment(frame, angle=0)
        all_detections.extend(fallback_detections)
        
        # Select best detections and enhance with landmark-based angle detection
        best_detections = self._select_best_detections(all_detections)
        
        # Post-process to get better angle estimates
        enhanced_detections = []
        for detection in best_detections:
            enhanced_detection = self._enhance_angle_detection(detection, frame)
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _enhance_angle_detection(self, detection: Dict[str, Any], frame: np.ndarray) -> Dict[str, Any]:
        """
        Enhance angle detection using multiple methods
        """
        # Skip enhancement if detection already has detailed angle breakdown
        if 'angle_breakdown' in detection:
            # print(f"ENHANCE DEBUG: Skipping enhancement - already has angle breakdown")  # Debug only
            return detection
            
        bbox = detection['bbox']
        
        # Method 1: Bounding box analysis (already done)
        bbox_angle = detection.get('angle', 0)
        
        # Method 2: Skip landmark extraction (was causing OpenCV errors and not contributing)
        landmark_angle = 0
        
        # Method 3: Visual analysis of face region
        visual_angle = self._estimate_angle_from_visual_features(bbox, frame)
        
        # Combine methods with confidence weighting
        final_angle = self._combine_angle_estimates(bbox_angle, landmark_angle, visual_angle)
        
        # Debug output for enhancement (remove in production)
        # print(f"ENHANCE DEBUG: input_angle={bbox_angle}°, landmark={landmark_angle}°, visual={visual_angle}° → enhanced_final={final_angle}°")
        
        # Update detection with enhanced angle
        enhanced_detection = detection.copy()
        enhanced_detection['angle'] = final_angle
        enhanced_detection['angle_breakdown'] = {
            'bbox': bbox_angle,
            'landmark': landmark_angle, 
            'visual': visual_angle,
            'final': final_angle
        }
        
        return enhanced_detection
    
    def _estimate_angle_from_landmarks(self, bbox, frame: np.ndarray) -> int:
        """
        Try to extract landmarks and calculate angle using Phase 1 method
        """
        try:
            # Import the Phase 1 angle estimation function
            from musetalk.utils.preprocessing import estimate_face_angle
            
            # Try to get landmarks from the face region
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            face_crop = frame[y1:y2, x1:x2]
            
            # This is a simplified approach - ideally we'd use a proper landmark detector
            # For now, we'll use geometric analysis of the face region
            landmarks = self._extract_simple_landmarks(face_crop, bbox)
            
            if landmarks is not None and len(landmarks) >= 17:
                angle = estimate_face_angle(landmarks)
                return angle
            else:
                return 0  # Default to frontal if no landmarks
                
        except Exception as e:
            print(f"⚠️  Landmark-based angle estimation failed: {e}")
            return 0
    
    def _extract_simple_landmarks(self, face_crop: np.ndarray, bbox) -> Optional[np.ndarray]:
        """
        Extract simple landmarks using image analysis
        This is a simplified approach for demonstration
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Simple approach: create approximate landmarks based on face features
            h, w = gray.shape
            x1, y1, x2, y2 = bbox
            
            # Create simplified 68-point landmarks (approximated)
            landmarks = np.zeros((68, 2))
            
            # Jaw line (points 0-16) - this is what we need for angle calculation
            for i in range(17):
                # Approximate jaw line as an arc
                t = i / 16.0  # 0 to 1
                jaw_x = int(w * (0.1 + 0.8 * t))  # Left to right across face
                jaw_y = int(h * (0.7 + 0.2 * np.sin(np.pi * t)))  # Curved jaw line
                landmarks[i] = [jaw_x + x1, jaw_y + y1]
            
            # Add other landmark points (simplified)
            for i in range(17, 68):
                landmarks[i] = [w/2 + x1, h/2 + y1]  # Center points for other features
            
            return landmarks.astype(np.int32)
            
        except Exception as e:
            print(f"⚠️  Simple landmark extraction failed: {e}")
            return None
    
    def _estimate_angle_from_visual_features(self, bbox, frame: np.ndarray) -> int:
        """
        Estimate angle from visual features of the face
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            face_crop = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Analyze left vs right side brightness/detail
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            
            # Calculate variance (detail level) in each half
            left_variance = np.var(left_half)
            right_variance = np.var(right_half)
            
            # Calculate brightness difference
            left_brightness = np.mean(left_half)
            right_brightness = np.mean(right_half)
            
            # Heuristic: profile faces often have one side with less detail/different brightness
            variance_ratio = left_variance / right_variance if right_variance > 0 else 1.0
            brightness_diff = left_brightness - right_brightness
            
            # Estimate angle based on asymmetry (CONSERVATIVE thresholds)
            if variance_ratio < 0.3:  # VERY obvious right profile (left side much less detail)
                return 90   # Right profile
            elif variance_ratio > 3.0:  # VERY obvious left profile (right side much less detail)
                return 270  # Left profile  
            elif abs(brightness_diff) > 25:  # VERY significant brightness difference
                if brightness_diff > 25:
                    return 30   # Moderate right turn
                else:
                    return 330  # Moderate left turn
            else:
                return 0    # Default to frontal - be conservative
                
        except Exception as e:
            print(f"⚠️  Visual angle estimation failed: {e}")
            return 0
    
    def _combine_angle_estimates(self, bbox_angle: int, landmark_angle: int, visual_angle: int) -> int:
        """
        Combine multiple angle estimates with CONSERVATIVE weighting
        Prioritize bbox and landmark methods over visual method to avoid false positives
        """
        # CONSERVATIVE approach: prioritize more reliable methods
        
        # If bbox and landmark both say frontal (0°), but visual detects angle
        if bbox_angle == 0 and landmark_angle == 0:
            # If visual detects a reasonable angle (15-45°), consider it
            if visual_angle is not None and (15 <= visual_angle <= 45 or 315 <= visual_angle <= 345):
                return visual_angle  # Trust visual for moderate angles
            elif visual_angle is not None and (visual_angle <= 15 or visual_angle >= 345):
                return 0  # Very small angles, consider frontal
            elif visual_angle is not None and visual_angle != 0:
                # Visual shows extreme angle - be conservative, return moderate angle
                return 30 if visual_angle < 180 else 330
            else:
                return 0  # Default to frontal
        
        # If any method strongly indicates profile view, consider it
        angles = []
        weights = []
        
        if bbox_angle is not None:
            angles.append(bbox_angle)
            weights.append(0.4)  # Bbox method gets 40% weight
            
        if landmark_angle is not None:
            angles.append(landmark_angle)
            weights.append(0.4)  # Landmark method gets 40% weight
            
        if visual_angle is not None:
            angles.append(visual_angle)
            weights.append(0.2)  # Visual method gets only 20% weight
        
        if not angles:
            return 0
        
        # Weighted average
        weighted_sum = sum(angle * weight for angle, weight in zip(angles, weights))
        total_weight = sum(weights)
        avg_angle = weighted_sum / total_weight
        
        # Round to nearest 15 degrees for stability
        rounded_angle = round(avg_angle / 15) * 15
        
        return int(rounded_angle % 360)
    
    def _detect_with_model(self, frame: np.ndarray, model_name: str, angle: int = 0) -> List[Dict[str, Any]]:
        """Detect faces using specific FaceFusion model"""
        if model_name not in self.models:
            return []
        
        try:
            model_info = self.models[model_name]
            session = model_info['session']
            input_size = model_info['config']['input_size']
            
            # Prepare input (basic preprocessing for now)
            input_frame = self._preprocess_frame(frame, input_size)
            
            # Run inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_frame})
            
            # Post-process results (simplified for now)
            detections = self._postprocess_detection(outputs, frame.shape, model_name, frame)
            
            return detections
            
        except Exception as e:
            print(f"⚠️  Detection failed with {model_name}: {e}")
            return []
    
    def _preprocess_frame(self, frame: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess frame for ONNX model input"""
        # Resize frame
        resized = cv2.resize(frame, input_size)
        
        # Convert BGR to RGB and normalize
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_batch = np.expand_dims(input_tensor, axis=0)
        
        return input_batch
    
    def _postprocess_detection(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int], model_name: str, frame: np.ndarray = None) -> List[Dict[str, Any]]:
        """Post-process model outputs to extract face detections"""
        detections = []
        
        # This is a simplified post-processing - each model has different output formats
        # For now, we'll implement basic detection extraction
        
        try:
            # Basic confidence threshold
            confidence_threshold = 0.5
            
            # For demonstration, create a mock detection if outputs look reasonable
            if len(outputs) > 0 and outputs[0] is not None:
                # Mock detection in center of image (replace with actual parsing)
                h, w = original_shape[:2]
                center_x, center_y = w // 2, h // 2
                face_size = min(w, h) // 4
                
                bbox = [
                    max(0, center_x - face_size),
                    max(0, center_y - face_size), 
                    min(w, center_x + face_size),
                    min(h, center_y + face_size)
                ]
                
                # Calculate actual angle using our estimation methods
                bbox_angle = self._estimate_angle_from_bbox(bbox, original_shape)
                
                # Skip landmark extraction - visual method is working well
                # The simplified landmark extraction was causing OpenCV errors
                # and wasn't contributing to angle detection (always returned 0°)
                landmarks = None
                landmark_angle = 0
                
                visual_angle = self._estimate_angle_from_visual_features(bbox, frame) if frame is not None else 0
                
                # Combine angle estimates
                final_angle = self._combine_angle_estimates(bbox_angle, landmark_angle, visual_angle)
                
                # Debug output (remove in production)
                # print(f"DEBUG: bbox={bbox_angle}°, landmark={landmark_angle}°, visual={visual_angle}° → final={final_angle}°")
                
                detections.append({
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'confidence': 0.8,  # Mock confidence
                    'source': f'facefusion_{model_name}',
                    'angle': final_angle,
                    'angle_breakdown': {
                        'bbox': bbox_angle,
                        'landmark': landmark_angle,
                        'visual': visual_angle
                    }
                })
                
        except Exception as e:
            print(f"⚠️  Post-processing failed for {model_name}: {e}")
        
        return detections
    
    def _detect_with_facealignment(self, frame: np.ndarray, angle: int = 0) -> List[Dict[str, Any]]:
        """Fallback detection using original FaceAlignment with enhanced angle detection"""
        detections = []
        
        if self.fa_fallback is None:
            return detections
        
        try:
            # Use existing FaceAlignment detection
            # Convert single frame to batch array format
            frame_batch = np.array([frame])  # Convert list to numpy array
            batch_detections = self.fa_fallback.get_detections_for_batch(frame_batch)
            
            # Extract bboxes from the batch result (first item since we only passed one frame)
            if batch_detections and len(batch_detections) > 0:
                frame_detections = batch_detections[0]  # Get detections for our single frame
                if frame_detections is not None:
                    for bbox in frame_detections:
                        if bbox is not None and len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]  # Take first 4 coordinates
                            
                            # Calculate basic quality score based on face size
                            area = (x2 - x1) * (y2 - y1)
                            confidence = min(area / (200 * 200), 1.0)  # Normalize by expected face size
                            confidence = max(confidence, 0.3)  # Minimum confidence for fallback
                            
                            # Enhanced angle detection using bounding box analysis
                            face_angle = self._estimate_angle_from_bbox(bbox, frame.shape)
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'landmarks': None,
                                'confidence': confidence,
                                'source': 'facealignment_fallback',
                                'angle': face_angle
                            })
        except Exception as e:
            print(f"⚠️  FaceAlignment fallback failed: {e}")
        
        return detections
    
    def _estimate_angle_from_bbox(self, bbox, frame_shape):
        """
        Estimate face angle from bounding box characteristics
        This is a heuristic approach for when we don't have landmarks
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        frame_height, frame_width = frame_shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        
        # Calculate position relative to frame center
        bbox_center_x = (x1 + x2) / 2
        frame_center_x = frame_width / 2
        horizontal_offset = (bbox_center_x - frame_center_x) / frame_width
        
        # Heuristic angle estimation
        if aspect_ratio < 0.7:  # Very narrow - likely profile
            # Determine which side based on position
            if horizontal_offset > 0.1:  # Face on right side of frame
                return 270  # Left profile (face looking left)
            elif horizontal_offset < -0.1:  # Face on left side of frame  
                return 90   # Right profile (face looking right)
            else:
                return 45   # Angled
        elif aspect_ratio > 1.3:  # Very wide - unusual proportions
            return 45  # Assume angled
        else:
            # Normal aspect ratio - check for subtle angle indicators
            if abs(horizontal_offset) > 0.15:
                return 15 if horizontal_offset > 0 else 345  # Slight angle
            else:
                return 0  # Frontal
    
    def _get_landmarks_from_bbox(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """
        Extract landmarks from face region for better angle estimation
        This uses a simple approach - in a full implementation, you'd use a landmark detector
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            face_crop = frame[y1:y2, x1:x2]
            
            # Simple approach: use face detection with landmarks
            # This is a placeholder - real implementation would use dlib or similar
            if hasattr(self.fa_fallback, 'get_landmarks'):
                landmarks = self.fa_fallback.get_landmarks(face_crop)
                if landmarks is not None and len(landmarks) >= 68:
                    # Adjust landmarks to original frame coordinates
                    landmarks[:, 0] += x1
                    landmarks[:, 1] += y1
                    return landmarks
            
            return None
            
        except Exception as e:
            print(f"⚠️  Landmark extraction failed: {e}")
            return None
    
    def _select_best_detections(self, all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select best non-overlapping detections"""
        if not all_detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        
        # For now, just return the highest confidence detection
        # TODO: Implement proper Non-Maximum Suppression (NMS) in future iterations
        best_detection = sorted_detections[0]
        
        # Only return detection if confidence is reasonable
        if best_detection['confidence'] > 0.2:
            return [best_detection]
        else:
            return []
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models for debugging"""
        return {
            'onnx_available': ONNX_AVAILABLE,
            'models_loaded': list(self.models.keys()),
            'fallback_available': self.fa_fallback is not None,
            'total_models': len(self.models)
        }

# Quick test function
def test_facefusion_detector():
    """Test function to verify detector works"""
    print("🧪 Testing FaceFusion Detector...")
    
    detector = FaceFusionDetector()
    status = detector.get_model_status()
    
    print(f"📊 Detector Status:")
    print(f"   - ONNX Available: {status['onnx_available']}")
    print(f"   - Models Loaded: {status['models_loaded']}")
    print(f"   - Fallback Available: {status['fallback_available']}")
    print(f"   - Total Models: {status['total_models']}")
    
    return detector

if __name__ == "__main__":
    test_facefusion_detector()
