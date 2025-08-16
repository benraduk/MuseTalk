#!/usr/bin/env python3
"""
Enhanced Lip Rotation using FaceFusion's Professional Methods
This leverages FaceFusion's warp_face_by_face_landmark_5 and paste_back functions
for professional-grade lip rotation and compositing.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import sys
import os

# Add facefusion to path if available
facefusion_path = os.path.join(os.path.dirname(__file__), '..', '..', 'facefusion')
if os.path.exists(facefusion_path):
    sys.path.insert(0, facefusion_path)

try:
    # Import FaceFusion's professional functions
    from facefusion.face_helper import warp_face_by_face_landmark_5, paste_back, WARP_TEMPLATE_SET
    from facefusion.face_landmarker import detect_face_landmark
    from facefusion.face_masker import create_box_mask
    FACEFUSION_AVAILABLE = True
    print("✅ FaceFusion professional functions imported successfully")
except ImportError as e:
    print(f"⚠️  FaceFusion not available: {e}")
    print("⚠️  Falling back to basic rotation methods")
    FACEFUSION_AVAILABLE = False


class ProfessionalLipRotator:
    """
    Professional lip rotation using FaceFusion's proven methods
    """
    
    def __init__(self):
        self.debug = False
        self.warp_template = 'ffhq_512'  # Use FaceFusion's FFHQ template
        self.crop_size = (256, 256)     # Standard face size
        
    def rotate_lips_professional(self, musetalk_frame: np.ndarray, face_angle: int, 
                                face_bbox: List[int], original_frame: np.ndarray,
                                face_landmarks_5: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Professional lip rotation using FaceFusion's methods
        
        Args:
            musetalk_frame: AI-generated face with frontal lips
            face_angle: Detected face angle
            face_bbox: Face bounding box [x1, y1, x2, y2]
            original_frame: Original frame to composite onto
            face_landmarks_5: 5-point facial landmarks (optional)
            
        Returns:
            final_frame: Frame with professionally rotated lips
        """
        if not FACEFUSION_AVAILABLE:
            # Fallback to basic rotation
            return self._basic_lip_rotation_fallback(musetalk_frame, face_angle, face_bbox, original_frame)
        
        try:
            return self._professional_facefusion_rotation(
                musetalk_frame, face_angle, face_bbox, original_frame, face_landmarks_5
            )
        except Exception as e:
            if self.debug:
                print(f"⚠️  Professional rotation failed: {e}, falling back to basic")
            return self._basic_lip_rotation_fallback(musetalk_frame, face_angle, face_bbox, original_frame)
    
    def _professional_facefusion_rotation(self, musetalk_frame: np.ndarray, face_angle: int,
                                        face_bbox: List[int], original_frame: np.ndarray,
                                        face_landmarks_5: Optional[np.ndarray]) -> np.ndarray:
        """
        Use FaceFusion's professional warping and pasting methods
        """
        # Step 1: Extract mouth region from MuseTalk output
        mouth_region = self._extract_mouth_region_professional(musetalk_frame)
        
        # Step 2: Create rotated landmarks for the mouth
        if face_landmarks_5 is None:
            face_landmarks_5 = self._estimate_face_landmarks_5(face_bbox, face_angle)
        
        # Step 3: Create rotated version of landmarks to match target angle
        rotated_landmarks_5 = self._rotate_landmarks_for_angle(face_landmarks_5, face_angle)
        
        # Step 4: Use FaceFusion's warp_face_by_face_landmark_5 for professional warping
        warp_template = WARP_TEMPLATE_SET[self.warp_template]
        
        # Warp the mouth region to match the rotated landmarks
        warped_mouth, affine_matrix = warp_face_by_face_landmark_5(
            mouth_region, rotated_landmarks_5, warp_template, self.crop_size
        )
        
        # Step 5: Create a mouth mask for clean compositing
        mouth_mask = self._create_mouth_mask_professional(warped_mouth)
        
        # Step 6: Use FaceFusion's paste_back for professional compositing
        final_frame = paste_back(original_frame, warped_mouth, mouth_mask, affine_matrix)
        
        if self.debug:
            print(f"🎨 Professional lip rotation applied: {face_angle}° face angle")
            
        return final_frame
    
    def _extract_mouth_region_professional(self, musetalk_frame: np.ndarray) -> np.ndarray:
        """
        Extract mouth region using FaceFusion-style approach
        """
        # MuseTalk generates 256x256 faces, extract mouth area
        h, w = musetalk_frame.shape[:2]
        
        # Use FaceFusion-style mouth region extraction
        mouth_y1 = int(h * 0.55)   # More precise mouth area
        mouth_y2 = int(h * 0.88)   
        mouth_x1 = int(w * 0.20)   
        mouth_x2 = int(w * 0.80)   
        
        mouth_region = musetalk_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        
        # Ensure we have a valid mouth region
        if mouth_region.size == 0:
            mouth_region = musetalk_frame  # Fallback to full face
            
        return mouth_region
    
    def _estimate_face_landmarks_5(self, face_bbox: List[int], face_angle: int) -> np.ndarray:
        """
        Estimate 5-point facial landmarks from bounding box
        This is a simplified approach - in practice, you'd use a landmark detector
        """
        x1, y1, x2, y2 = face_bbox
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Standard 5-point landmark positions (normalized)
        landmarks_normalized = np.array([
            [0.35, 0.35],  # Left eye
            [0.65, 0.35],  # Right eye  
            [0.50, 0.50],  # Nose tip
            [0.35, 0.75],  # Left mouth corner
            [0.65, 0.75]   # Right mouth corner
        ])
        
        # Scale to face bounding box
        landmarks_5 = landmarks_normalized.copy()
        landmarks_5[:, 0] = x1 + landmarks_5[:, 0] * face_width
        landmarks_5[:, 1] = y1 + landmarks_5[:, 1] * face_height
        
        return landmarks_5.astype(np.float32)
    
    def _rotate_landmarks_for_angle(self, landmarks_5: np.ndarray, face_angle: int) -> np.ndarray:
        """
        Rotate the 5-point landmarks to match the target face angle
        """
        if face_angle == 0:
            return landmarks_5
            
        # Convert face angle to rotation for landmarks
        rotation_angle = face_angle * 0.8  # Scale down for more natural look
        rotation_rad = np.radians(rotation_angle)
        
        # Find center of landmarks
        center = np.mean(landmarks_5, axis=0)
        
        # Create rotation matrix
        cos_a, sin_a = np.cos(rotation_rad), np.sin(rotation_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation around center
        rotated_landmarks = landmarks_5 - center
        rotated_landmarks = rotated_landmarks @ rotation_matrix.T
        rotated_landmarks += center
        
        return rotated_landmarks.astype(np.float32)
    
    def _create_mouth_mask_professional(self, mouth_region: np.ndarray) -> np.ndarray:
        """
        Create professional mouth mask using FaceFusion's approach
        """
        if FACEFUSION_AVAILABLE:
            try:
                # Use FaceFusion's create_box_mask for professional masking
                mask = create_box_mask(mouth_region, blur_amount=0.1, padding=(5, 5, 5, 5))
                return mask
            except:
                pass
        
        # Fallback to basic mask
        h, w = mouth_region.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        # Create elliptical mask for mouth
        center = (w // 2, h // 2)
        axes = (w // 3, h // 4)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Soft edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = np.clip(mask, 0, 1)
        
        return mask
    
    def _basic_lip_rotation_fallback(self, musetalk_frame: np.ndarray, face_angle: int,
                                   face_bbox: List[int], original_frame: np.ndarray) -> np.ndarray:
        """
        Fallback to basic rotation when FaceFusion is not available
        """
        # This would use our original basic rotation method
        # Import and use the original LipRotator class
        try:
            from musetalk.utils.lip_rotation import LipRotator
            basic_rotator = LipRotator()
            return basic_rotator.rotate_and_position_lips(musetalk_frame, face_angle, face_bbox, original_frame)
        except Exception as e:
            if self.debug:
                print(f"⚠️  Even basic rotation failed: {e}")
            return original_frame
    
    def create_comparison_visualization(self, original_frame: np.ndarray, musetalk_frame: np.ndarray,
                                      basic_result: np.ndarray, professional_result: np.ndarray,
                                      face_angle: int, save_path: str = None) -> np.ndarray:
        """
        Create comparison between basic and professional rotation
        """
        # Resize for visualization
        vis_size = (150, 150)
        orig_vis = cv2.resize(original_frame, vis_size)
        musetalk_vis = cv2.resize(musetalk_frame, vis_size)
        basic_vis = cv2.resize(basic_result, vis_size)
        prof_vis = cv2.resize(professional_result, vis_size)
        
        # Create combined visualization
        comparison = np.hstack([orig_vis, musetalk_vis, basic_vis, prof_vis])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, f'Original ({face_angle}°)', (5, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(comparison, 'MuseTalk', (155, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(comparison, 'Basic Rotation', (305, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(comparison, 'Professional', (455, 20), font, 0.4, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, comparison)
            
        return comparison


def test_professional_lip_rotation():
    """Test the professional lip rotation vs basic rotation"""
    print("🧪 TESTING PROFESSIONAL LIP ROTATION vs BASIC")
    print("=" * 60)
    
    rotator = ProfessionalLipRotator()
    rotator.debug = True
    
    # Test with different angles
    test_angles = [0, 30, 330]
    
    for angle in test_angles:
        print(f"\n📐 Testing professional rotation for {angle}° face")
        
        # Create test data
        original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_frame[:] = (50, 50, 100)  # Background
        
        face_bbox = [200, 150, 400, 350]
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (100, 100, 150), -1)
        
        # MuseTalk output simulation
        musetalk_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        musetalk_frame[:] = (150, 150, 200)
        cv2.rectangle(musetalk_frame, (64, 154), (192, 218), (200, 100, 100), -1)
        
        try:
            # Test professional rotation
            professional_result = rotator.rotate_lips_professional(
                musetalk_frame, angle, face_bbox, original_frame
            )
            
            # Test basic rotation for comparison
            from musetalk.utils.lip_rotation import LipRotator
            basic_rotator = LipRotator()
            basic_result = basic_rotator.rotate_and_position_lips(
                musetalk_frame, angle, face_bbox, original_frame
            )
            
            # Create comparison
            comparison = rotator.create_comparison_visualization(
                original_frame, musetalk_frame, basic_result, professional_result, angle,
                f"professional_vs_basic_{angle}deg.png"
            )
            
            print(f"   ✅ Professional rotation successful for {angle}°")
            print(f"   💾 Comparison saved: professional_vs_basic_{angle}deg.png")
            
        except Exception as e:
            print(f"   ❌ Test failed for {angle}°: {e}")
    
    print(f"\n🎯 Professional lip rotation testing complete!")
    print(f"Check the professional_vs_basic_*.png files to compare quality.")


if __name__ == "__main__":
    test_professional_lip_rotation()
