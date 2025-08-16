#!/usr/bin/env python3
"""
Phase 3: Rotation Normalization for MuseTalk
Core implementation: rotate → MuseTalk → rotate back

This module provides the key functions for normalizing angled faces to frontal orientation
before MuseTalk processing, then restoring the original angle afterwards.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math

class RotationNormalizer:
    """
    Handles rotation normalization for angled faces in MuseTalk pipeline
    """
    
    def __init__(self):
        """Initialize the rotation normalizer"""
        self.debug = False  # Set to True for debug visualizations
        
    def normalize_face_rotation(self, face_crop: np.ndarray, angle: int, 
                              target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize an angled face to frontal orientation (0°)
        
        Args:
            face_crop: Input face image (cropped face region)
            angle: Detected face angle in degrees (0-360)
            target_size: Target size for MuseTalk processing (256x256)
            
        Returns:
            normalized_face: Face rotated to frontal orientation and resized
            rotation_metadata: Information needed to restore original angle
        """
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Invalid face crop provided")
            
        # Store original dimensions
        original_height, original_width = face_crop.shape[:2]
        
        # Calculate rotation needed to make face frontal
        # Convert angle to rotation needed (negative because we want to counter-rotate)
        rotation_angle = self._calculate_normalization_angle(angle)
        
        if abs(rotation_angle) < 5:  # Already frontal enough
            # Just resize without rotation
            normalized_face = cv2.resize(face_crop, target_size)
            rotation_metadata = {
                'rotation_applied': 0,
                'original_size': (original_width, original_height),
                'rotation_matrix': np.eye(2, 3, dtype=np.float32),
                'needs_restoration': False,
                'original_angle': angle
            }
        else:
            # Apply rotation to normalize to frontal
            normalized_face, rotation_matrix = self._rotate_image(face_crop, rotation_angle)
            
            # Resize to target size for MuseTalk
            normalized_face = cv2.resize(normalized_face, target_size)
            
            rotation_metadata = {
                'rotation_applied': rotation_angle,
                'original_size': (original_width, original_height),
                'rotation_matrix': rotation_matrix,
                'needs_restoration': True,
                'original_angle': angle
            }
        
        if self.debug:
            print(f"🔄 Normalized {angle}° face with {rotation_angle}° rotation")
            
        return normalized_face, rotation_metadata
    
    def restore_face_rotation(self, processed_face: np.ndarray, 
                            rotation_metadata: Dict[str, Any]) -> np.ndarray:
        """
        Restore the original face angle after MuseTalk processing
        
        Args:
            processed_face: Face processed by MuseTalk (256x256)
            rotation_metadata: Metadata from normalize_face_rotation
            
        Returns:
            restored_face: Face rotated back to original angle and size
        """
        if not rotation_metadata['needs_restoration']:
            # Just resize back to original size
            original_width, original_height = rotation_metadata['original_size']
            return cv2.resize(processed_face, (original_width, original_height))
        
        # Resize back to original size first
        original_width, original_height = rotation_metadata['original_size']
        resized_face = cv2.resize(processed_face, (original_width, original_height))
        
        # Apply reverse rotation to restore original angle
        reverse_rotation = -rotation_metadata['rotation_applied']
        restored_face, _ = self._rotate_image(resized_face, reverse_rotation)
        
        if self.debug:
            original_angle = rotation_metadata['original_angle']
            applied_rotation = rotation_metadata['rotation_applied']
            print(f"🔄 Restored face: {original_angle}° (reversed {applied_rotation}° normalization)")
            
        return restored_face
    
    def _calculate_normalization_angle(self, detected_angle: int) -> float:
        """
        Calculate the rotation needed to normalize face to frontal (0°)
        
        Args:
            detected_angle: Face angle detected by visual analysis (0-360°)
            
        Returns:
            rotation_angle: Angle to rotate for normalization (-180 to +180)
        """
        # Normalize detected angle to -180 to +180 range
        if detected_angle > 180:
            detected_angle = detected_angle - 360
            
        # For face normalization, we want to rotate opposite to detected angle
        # to bring the face to frontal (0°) orientation
        rotation_angle = -detected_angle
        
        # Clamp to reasonable rotation limits to avoid extreme distortions
        rotation_angle = max(-45, min(45, rotation_angle))
        
        return rotation_angle
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate image by specified angle while preserving all content
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            rotated_image: Rotated image
            rotation_matrix: 2x3 transformation matrix used
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to fit rotated content
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int(height * sin_angle + width * cos_angle)
        new_height = int(height * cos_angle + width * sin_angle)
        
        # Adjust rotation matrix to account for new image center
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        # Apply rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return rotated_image, rotation_matrix
    
    def create_debug_visualization(self, original_face: np.ndarray, 
                                 normalized_face: np.ndarray, 
                                 restored_face: np.ndarray,
                                 angle: int, save_path: str = None) -> np.ndarray:
        """
        Create debug visualization showing the rotation normalization process
        
        Args:
            original_face: Original angled face
            normalized_face: Face after normalization (frontal)
            restored_face: Face after restoration to original angle
            angle: Detected angle
            save_path: Optional path to save visualization
            
        Returns:
            debug_image: Combined visualization
        """
        # Resize all faces to same size for visualization
        vis_size = (200, 200)
        orig_vis = cv2.resize(original_face, vis_size)
        norm_vis = cv2.resize(normalized_face, vis_size)
        rest_vis = cv2.resize(restored_face, vis_size)
        
        # Create combined visualization
        debug_image = np.hstack([orig_vis, norm_vis, rest_vis])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_image, f'Original ({angle}°)', (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, 'Normalized (0°)', (210, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, f'Restored ({angle}°)', (410, 25), font, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, debug_image)
            
        return debug_image


def test_rotation_normalization():
    """Test the rotation normalization functionality"""
    print("🧪 TESTING ROTATION NORMALIZATION")
    print("=" * 50)
    
    normalizer = RotationNormalizer()
    normalizer.debug = True
    
    # Test with different angles
    test_angles = [0, 30, 330, 45, 315]
    
    for angle in test_angles:
        print(f"\n📐 Testing angle: {angle}°")
        
        # Create a test face image (simple pattern)
        test_face = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add asymmetric pattern to see rotation effect
        cv2.rectangle(test_face, (50, 80), (150, 120), (255, 255, 255), -1)  # Mouth area
        cv2.circle(test_face, (80, 60), 10, (255, 0, 0), -1)  # Left eye
        cv2.circle(test_face, (120, 60), 10, (0, 255, 0), -1)  # Right eye
        cv2.rectangle(test_face, (90, 140), (110, 160), (0, 0, 255), -1)  # Chin marker
        
        try:
            # Test normalization
            normalized_face, metadata = normalizer.normalize_face_rotation(test_face, angle)
            print(f"   ✅ Normalization successful")
            print(f"      Applied rotation: {metadata['rotation_applied']:.1f}°")
            print(f"      Needs restoration: {metadata['needs_restoration']}")
            
            # Test restoration
            restored_face = normalizer.restore_face_rotation(normalized_face, metadata)
            print(f"   ✅ Restoration successful")
            
            # Create debug visualization
            debug_vis = normalizer.create_debug_visualization(
                test_face, normalized_face, restored_face, angle,
                f"debug_rotation_{angle}deg.png"
            )
            print(f"   💾 Debug visualization saved: debug_rotation_{angle}deg.png")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    print(f"\n🎯 Rotation normalization testing complete!")
    print(f"Check the debug_rotation_*.png files to see the results.")


if __name__ == "__main__":
    test_rotation_normalization()
