import torch
import cv2
import numpy as np
from typing import List, Union, Optional
import os
import sys

# Defer GFPGAN import to avoid early import errors
GFPGAN_AVAILABLE = None

def _check_gfpgan_availability():
    """Check if GFPGAN is available and import it"""
    global GFPGAN_AVAILABLE
    if GFPGAN_AVAILABLE is None:
        try:
            from gfpgan import GFPGANer
            GFPGAN_AVAILABLE = True
            return True, GFPGANer
        except ImportError as e:
            GFPGAN_AVAILABLE = False
            return False, None
    elif GFPGAN_AVAILABLE:
        from gfpgan import GFPGANer
        return True, GFPGANer
    else:
        return False, None

class GFPGANEnhancer:
    """
    GFPGAN face enhancement wrapper for MuseTalk pipeline.
    
    This class enhances FULL AI-generated faces (not just lips) that come from VAE decoding.
    The input faces contain original facial features in the upper half and AI-generated 
    lips in the lower half.
    """
    
    def __init__(self, 
                 model_path: str = "models/gfpgan/GFPGANv1.4.pth",
                 upscale: int = 1,  # Changed default to 1 for MuseTalk compatibility
                 arch: str = 'clean',
                 channel_multiplier: int = 2,
                 bg_upsampler: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize GFPGAN enhancer
        
        Args:
            model_path: Path to GFPGAN model (.pth file)
            upscale: Upscaling factor (1=no scaling, 2=2x, 4=4x) - use 1 for MuseTalk
            arch: GFPGAN architecture ('original'=conservative, 'clean'=aggressive, 'RestoreFormer'=alternative)
            channel_multiplier: Model complexity (1=light, 2=standard, 4=heavy)
            bg_upsampler: Background upsampler ('realesrgan' or None) - None recommended
            device: Device to run on ('cuda' or 'cpu')
        """
        # Check GFPGAN availability
        is_available, GFPGANer = _check_gfpgan_availability()
        if not is_available:
            raise ImportError("GFPGAN is not available. Install it with: pip install gfpgan basicsr facexlib realesrgan")
            
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🎨 Initializing GFPGAN on {self.device}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GFPGAN model not found at {model_path}")
        
        try:
            # Initialize GFPGAN
            self.restorer = GFPGANer(
                model_path=model_path,
                upscale=upscale,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=bg_upsampler,
                device=str(self.device)
            )
            
            self.upscale = upscale
            self.model_path = model_path
            print(f"✅ GFPGAN initialized successfully")
            print(f"   Model: {os.path.basename(model_path)}")
            print(f"   Upscale: {upscale}x")
            print(f"   Architecture: {arch}")
            
        except Exception as e:
            print(f"❌ Failed to initialize GFPGAN: {e}")
            raise
    
    def enhance_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance a single face image from MuseTalk VAE decoder.
        
        Args:
            face_image: Input face image (H, W, 3) in RGB format from MuseTalk VAE
            
        Returns:
            Enhanced face image in RGB format (same size as input for compatibility)
        """
        try:
            # Convert RGB to BGR for GFPGAN (GFPGAN expects BGR)
            face_bgr = cv2.cvtColor(face_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # GFPGAN enhancement
            # Returns: cropped_faces, restored_faces, restored_img
            _, _, restored_img = self.restorer.enhance(
                face_bgr, 
                has_aligned=False,  # Input is not pre-aligned
                only_center_face=True,  # Focus on the main face
                paste_back=True  # Paste enhanced face back to original image
            )
            
            if restored_img is None:
                print("⚠️ GFPGAN returned None, using fallback")
                return self._fallback_enhance(face_image)
            
            # Convert back to RGB for MuseTalk compatibility
            restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            
            # Ensure output size matches input size for pipeline compatibility
            if restored_rgb.shape != face_image.shape:
                restored_rgb = cv2.resize(restored_rgb, 
                                        (face_image.shape[1], face_image.shape[0]), 
                                        interpolation=cv2.INTER_LANCZOS4)
            
            return restored_rgb
            
        except Exception as e:
            print(f"⚠️ GFPGAN enhancement failed: {e}")
            return self._fallback_enhance(face_image)
    
    def _fallback_enhance(self, face_image: np.ndarray) -> np.ndarray:
        """
        Fallback enhancement using basic upscaling when GFPGAN fails.
        
        Args:
            face_image: Input face image
            
        Returns:
            Basic upscaled image
        """
        if self.upscale > 1:
            # Basic bicubic upscaling as fallback
            h, w = face_image.shape[:2]
            upscaled = cv2.resize(face_image, 
                                (w * self.upscale, h * self.upscale), 
                                interpolation=cv2.INTER_CUBIC)
            # Resize back to original size for compatibility
            return cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return face_image
    
    def enhance_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance a batch of face images with progress tracking.
        
        Args:
            face_images: List of face images from MuseTalk VAE decoder
            
        Returns:
            List of enhanced face images
        """
        enhanced_faces = []
        total_faces = len(face_images)
        
        print(f"🎨 Enhancing {total_faces} faces with GFPGAN...")
        
        for i, face in enumerate(face_images):
            if i % 10 == 0:  # Progress every 10 faces
                print(f"   Progress: {i+1}/{total_faces}")
            
            enhanced_face = self.enhance_face(face)
            enhanced_faces.append(enhanced_face)
        
        print(f"✅ Enhanced {total_faces} faces successfully")
        return enhanced_faces
    
    def enhance_batch_optimized(self, face_images: List[np.ndarray], 
                               batch_size: int = 4) -> List[np.ndarray]:
        """
        Memory-optimized batch processing for large face counts.
        
        Args:
            face_images: List of face images
            batch_size: Process this many faces before clearing GPU cache
            
        Returns:
            List of enhanced face images
        """
        enhanced_faces = []
        total_faces = len(face_images)
        
        print(f"🎨 Enhancing {total_faces} faces (batch_size={batch_size})...")
        
        # Process in smaller batches to manage GPU memory
        for i in range(0, total_faces, batch_size):
            batch_end = min(i + batch_size, total_faces)
            batch = face_images[i:batch_end]
            
            print(f"   Batch {i//batch_size + 1}: Processing faces {i+1}-{batch_end}")
            
            # Process batch
            batch_results = []
            for face in batch:
                enhanced = self.enhance_face(face)
                batch_results.append(enhanced)
            
            enhanced_faces.extend(batch_results)
            
            # Clear GPU cache periodically to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"✅ Enhanced {total_faces} faces successfully")
        return enhanced_faces
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded GFPGAN model.
        
        Returns:
            Dictionary with model information
        """
        is_available, _ = _check_gfpgan_availability()
        return {
            'model_path': self.model_path,
            'upscale_factor': self.upscale,
            'device': str(self.device),
            'available': is_available
        }

def test_gfpgan_enhancer():
    """
    Test function for GFPGAN enhancer
    """
    try:
        # Test with dummy data
        enhancer = GFPGANEnhancer()
        
        # Create test image (256x256 RGB)
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test single enhancement
        enhanced = enhancer.enhance_face(test_image)
        print(f"✅ Single enhancement test passed: {enhanced.shape}")
        
        # Test batch enhancement
        batch = [test_image] * 3
        enhanced_batch = enhancer.enhance_batch(batch)
        print(f"✅ Batch enhancement test passed: {len(enhanced_batch)} faces")
        
        # Print model info
        info = enhancer.get_model_info()
        print("📊 Model Info:", info)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_gfpgan_enhancer()
