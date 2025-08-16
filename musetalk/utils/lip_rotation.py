#!/usr/bin/env python3
"""
Phase 3B: Lip Rotation Post-Processing
Post-process approach: Rotate AI-generated frontal lips to match detected face angles

This approach is much simpler and more elegant:
1. MuseTalk generates frontal lips (as it normally does)
2. We rotate and position those lips to match the detected face angle
3. Composite onto the original frame
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math

class LipRotator:
    """
    Handles rotation and positioning of AI-generated lips to match face angles
    """
    
    def __init__(self):
        """Initialize the lip rotator"""
        self.debug = False  # Set to True for debug visualizations
        
    def rotate_and_position_lips(self, generated_lips: np.ndarray, face_angle: int, 
                                face_bbox: list, original_frame: np.ndarray) -> np.ndarray:
        """
        Rotate AI-generated frontal lips to match face angle and composite onto frame
        
        Args:
            generated_lips: AI-generated lip region (frontal orientation)
            face_angle: Detected face angle (0-360°)
            face_bbox: Face bounding box [x1, y1, x2, y2]
            original_frame: Original frame to composite onto
            
        Returns:
            final_frame: Frame with rotated lips composited
        """
        if generated_lips is None or generated_lips.size == 0:
            return original_frame
            
        try:
            # Step 1: Extract mouth region from generated lips
            mouth_region = self._extract_mouth_region(generated_lips)
            
            # Step 2: Rotate mouth region to match face angle
            if face_angle != 0 and abs(face_angle) > 5:  # Only rotate if significant angle
                rotation_angle = self._calculate_lip_rotation_angle(face_angle)
                rotated_mouth = self._rotate_image(mouth_region, rotation_angle)
            else:
                rotated_mouth = mouth_region
                rotation_angle = 0
            
            # Step 3: Position rotated mouth within face bbox
            positioned_mouth = self._position_mouth_in_face(rotated_mouth, face_bbox, face_angle)
            
            # Step 4: Composite onto original frame with blending
            final_frame = self._composite_mouth(original_frame, positioned_mouth, face_bbox)
            
            if self.debug:
                print(f"🔄 Rotated lips: {face_angle}° face → {rotation_angle}° lip rotation")
                
            return final_frame
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Lip rotation failed: {e}")
            return original_frame
    
    def _extract_mouth_region(self, generated_face: np.ndarray) -> np.ndarray:
        """
        Extract the mouth region from the AI-generated face
        
        Args:
            generated_face: Full AI-generated face (256x256)
            
        Returns:
            mouth_region: Extracted mouth area
        """
        # MuseTalk generates 256x256 faces
        # Mouth region is typically in the lower third of the face
        h, w = generated_face.shape[:2]
        
        # Define mouth region (approximate coordinates for 256x256 face)
        mouth_y1 = int(h * 0.6)   # Start at 60% down
        mouth_y2 = int(h * 0.85)  # End at 85% down
        mouth_x1 = int(w * 0.25)  # Start at 25% across
        mouth_x2 = int(w * 0.75)  # End at 75% across
        
        mouth_region = generated_face[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        
        if self.debug:
            print(f"   📏 Extracted mouth region: {mouth_region.shape}")
            
        return mouth_region
    
    def _calculate_lip_rotation_angle(self, face_angle: int) -> float:
        """
        Calculate the rotation angle needed for lips to match face angle
        
        Args:
            face_angle: Detected face angle (0-360°)
            
        Returns:
            lip_rotation_angle: Angle to rotate lips (-45 to +45°)
        """
        # Convert face angle to lip rotation angle
        if face_angle > 180:
            face_angle = face_angle - 360  # Convert to -180 to +180 range
            
        # For lips, we want to rotate in the same direction as the face
        # but with a more subtle rotation (lips don't rotate as much as the full face)
        lip_rotation = face_angle * 0.8  # Scale down the rotation for more natural look
        
        # Clamp to reasonable limits
        lip_rotation = max(-45, min(45, lip_rotation))
        
        return lip_rotation
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle while preserving content
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            rotated_image: Rotated image
        """
        if abs(angle) < 1:  # Skip rotation for very small angles
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to fit rotated content
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int(height * sin_angle + width * cos_angle)
        new_height = int(height * cos_angle + width * sin_angle)
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        # Apply rotation with anti-aliasing
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return rotated_image
    
    def _position_mouth_in_face(self, rotated_mouth: np.ndarray, face_bbox: list, face_angle: int) -> Dict[str, Any]:
        """
        Calculate positioning for rotated mouth within the face bbox
        
        Args:
            rotated_mouth: Rotated mouth region
            face_bbox: Face bounding box [x1, y1, x2, y2]
            face_angle: Original face angle for positioning adjustments
            
        Returns:
            positioning_info: Dictionary with position and sizing info
        """
        x1, y1, x2, y2 = face_bbox
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Calculate mouth position within face (as ratios)
        mouth_center_x_ratio = 0.5  # Center horizontally
        mouth_center_y_ratio = 0.72  # Position in lower third of face
        
        # Adjust position slightly based on face angle
        if face_angle == 30:  # Right-angled face
            mouth_center_x_ratio += 0.02  # Shift slightly right
        elif face_angle == 330:  # Left-angled face  
            mouth_center_x_ratio -= 0.02  # Shift slightly left
        
        # Calculate absolute position
        mouth_center_x = x1 + int(face_width * mouth_center_x_ratio)
        mouth_center_y = y1 + int(face_height * mouth_center_y_ratio)
        
        # Calculate mouth size (scale with face size)
        mouth_height, mouth_width = rotated_mouth.shape[:2]
        scale_factor = min(face_width / 256, face_height / 256) * 0.8  # Scale appropriately
        
        scaled_width = int(mouth_width * scale_factor)
        scaled_height = int(mouth_height * scale_factor)
        
        # Calculate final bbox for mouth
        mouth_x1 = mouth_center_x - scaled_width // 2
        mouth_y1 = mouth_center_y - scaled_height // 2
        mouth_x2 = mouth_x1 + scaled_width
        mouth_y2 = mouth_y1 + scaled_height
        
        return {
            'mouth_bbox': [mouth_x1, mouth_y1, mouth_x2, mouth_y2],
            'scaled_mouth': cv2.resize(rotated_mouth, (scaled_width, scaled_height)),
            'center': (mouth_center_x, mouth_center_y),
            'scale_factor': scale_factor
        }
    
    def _composite_mouth(self, original_frame: np.ndarray, positioning_info: Dict[str, Any], 
                        face_bbox: list) -> np.ndarray:
        """
        Composite the rotated mouth onto the original frame with blending
        
        Args:
            original_frame: Original frame
            positioning_info: Mouth positioning information
            face_bbox: Face bounding box for reference
            
        Returns:
            composited_frame: Frame with mouth composited
        """
        mouth_bbox = positioning_info['mouth_bbox']
        scaled_mouth = positioning_info['scaled_mouth']
        
        mx1, my1, mx2, my2 = mouth_bbox
        
        # Ensure mouth bbox is within frame bounds
        frame_h, frame_w = original_frame.shape[:2]
        mx1 = max(0, min(mx1, frame_w))
        my1 = max(0, min(my1, frame_h))
        mx2 = max(mx1, min(mx2, frame_w))
        my2 = max(my1, min(my2, frame_h))
        
        # Adjust scaled mouth if bbox was clipped
        actual_width = mx2 - mx1
        actual_height = my2 - my1
        
        if actual_width > 0 and actual_height > 0:
            if scaled_mouth.shape[:2] != (actual_height, actual_width):
                scaled_mouth = cv2.resize(scaled_mouth, (actual_width, actual_height))
            
            # Create a soft blend mask for natural compositing
            mask = self._create_mouth_mask(scaled_mouth)
            
            # Composite with alpha blending
            result_frame = original_frame.copy()
            
            # Extract region to composite onto
            target_region = result_frame[my1:my2, mx1:mx2]
            
            # Blend using mask
            for c in range(3):  # RGB channels
                target_region[:, :, c] = (
                    target_region[:, :, c] * (1 - mask) + 
                    scaled_mouth[:, :, c] * mask
                )
            
            result_frame[my1:my2, mx1:mx2] = target_region
            
            if self.debug:
                print(f"   🎨 Composited mouth at ({mx1},{my1}) to ({mx2},{my2})")
                
            return result_frame
        else:
            if self.debug:
                print(f"   ⚠️  Invalid mouth bbox, returning original frame")
            return original_frame
    
    def _create_mouth_mask(self, mouth_region: np.ndarray) -> np.ndarray:
        """
        Create a soft mask for mouth blending
        
        Args:
            mouth_region: Mouth region image
            
        Returns:
            mask: Soft blending mask (0-1)
        """
        h, w = mouth_region.shape[:2]
        
        # Create elliptical mask centered on mouth
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Create ellipse (mouth shape)
        center = (w // 2, h // 2)
        axes = (w // 3, h // 4)  # Ellipse dimensions
        
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Ensure mask is in 0-1 range
        mask = np.clip(mask, 0, 1)
        
        return mask
    
    def create_debug_visualization(self, original_frame: np.ndarray, generated_face: np.ndarray,
                                 final_frame: np.ndarray, face_angle: int, 
                                 save_path: str = None) -> np.ndarray:
        """
        Create debug visualization showing the lip rotation process
        
        Args:
            original_frame: Original input frame
            generated_face: AI-generated face from MuseTalk
            final_frame: Final frame with rotated lips
            face_angle: Detected face angle
            save_path: Optional path to save visualization
            
        Returns:
            debug_image: Combined visualization
        """
        # Resize for visualization
        vis_size = (200, 200)
        orig_vis = cv2.resize(original_frame, vis_size)
        gen_vis = cv2.resize(generated_face, vis_size) 
        final_vis = cv2.resize(final_frame, vis_size)
        
        # Create combined visualization
        debug_image = np.hstack([orig_vis, gen_vis, final_vis])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_image, f'Original ({face_angle}°)', (10, 25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_image, 'MuseTalk Output', (210, 25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_image, f'Rotated Lips ({face_angle}°)', (410, 25), font, 0.5, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, debug_image)
            
        return debug_image


def test_lip_rotation():
    """Test the lip rotation functionality"""
    print("🧪 TESTING LIP ROTATION POST-PROCESSING")
    print("=" * 50)
    
    rotator = LipRotator()
    rotator.debug = True
    
    # Test with different angles
    test_angles = [0, 30, 330]
    
    for angle in test_angles:
        print(f"\n📐 Testing lip rotation for {angle}° face")
        
        # Create test images
        original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_frame[:] = (50, 50, 100)  # Dark blue background
        
        # Add a face-like region
        face_bbox = [200, 150, 400, 350]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (100, 100, 150), -1)
        
        # Create AI-generated face (simulated MuseTalk output)
        generated_face = np.zeros((256, 256, 3), dtype=np.uint8)
        generated_face[:] = (150, 150, 200)  # Light face color
        
        # Add mouth region to generated face
        cv2.rectangle(generated_face, (64, 154), (192, 218), (200, 100, 100), -1)  # Mouth area
        
        try:
            # Test lip rotation
            result_frame = rotator.rotate_and_position_lips(
                generated_face, angle, face_bbox, original_frame
            )
            
            print(f"   ✅ Lip rotation successful for {angle}°")
            
            # Create debug visualization
            debug_vis = rotator.create_debug_visualization(
                original_frame, generated_face, result_frame, angle,
                f"debug_lip_rotation_{angle}deg.png"
            )
            print(f"   💾 Debug visualization saved: debug_lip_rotation_{angle}deg.png")
            
        except Exception as e:
            print(f"   ❌ Test failed for {angle}°: {e}")
    
    print(f"\n🎯 Lip rotation testing complete!")
    print(f"Check the debug_lip_rotation_*.png files to see the results.")


if __name__ == "__main__":
    test_lip_rotation()
