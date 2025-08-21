"""
GPEN-BFR 256 Face Enhancer for MuseTalk
========================================

This module provides a wrapper for GPEN-BFR 256 face enhancement using ONNX runtime.
GPEN-BFR is specifically designed for 256x256 face images and provides more natural
results than GFPGAN while being more stable due to ONNX runtime.

Key Features:
- ONNX runtime (no PyTorch conflicts)
- Perfect 256x256 input/output size for MuseTalk
- Better identity preservation than GFPGAN
- More natural enhancement results
- GPU acceleration support

Usage:
    from face_enhancers.gpen_bfr_enhancer import GPENBFREnhancer
    
    enhancer = GPENBFREnhancer()
    enhanced_face = enhancer.enhance_face(face_image)
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

# Global flag to track ONNX availability
_onnx_available = None

def _check_onnx_availability():
    """Check if ONNX runtime is available and return appropriate providers"""
    global _onnx_available
    
    if _onnx_available is not None:
        return _onnx_available
    
    try:
        import onnxruntime as ort
        
        # Get available providers
        available_providers = ort.get_available_providers()
        
        # Prefer GPU providers
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            print("✅ CUDA provider available for GPEN-BFR")
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')
        
        _onnx_available = {
            'available': True,
            'onnxruntime': ort,
            'providers': providers
        }
        
        print(f"✅ ONNX Runtime available with providers: {providers}")
        return _onnx_available
        
    except ImportError as e:
        print(f"❌ ONNX Runtime not available: {e}")
        print("💡 Install with: pip install onnxruntime-gpu")
        _onnx_available = {'available': False, 'error': str(e)}
        return _onnx_available


class GPENBFREnhancer:
    """
    GPEN-BFR 256 face enhancer using ONNX runtime
    
    This class provides face enhancement specifically optimized for MuseTalk's
    256x256 VAE-decoded face images. It uses the GPEN-BFR model which provides
    better identity preservation and more natural results than GFPGAN.
    
    Args:
        model_path (str): Path to the GPEN-BFR ONNX model file
        device (str): Device preference ('auto', 'cuda', 'cpu')
        
    Example:
        enhancer = GPENBFREnhancer()
        enhanced_faces = enhancer.enhance_batch(vae_faces)
    """
    
    def __init__(self, 
                 model_path: str = "models/gpen_bfr/gpen_bfr_256.onnx",
                 device: str = "auto"):
        
        # Check ONNX availability
        onnx_info = _check_onnx_availability()
        if not onnx_info['available']:
            raise ImportError(f"ONNX Runtime not available: {onnx_info.get('error', 'Unknown error')}")
        
        self.ort = onnx_info['onnxruntime']
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GPEN-BFR model not found: {model_path}\n"
                f"💡 Download with: python download_weights.bat (Windows) or ./download_weights.sh (Linux/macOS)"
            )
        
        self.model_path = model_path
        self.device = device
        
        # Setup providers based on device preference
        self.providers = self._setup_providers(device, onnx_info['providers'])
        
        # Initialize ONNX session
        try:
            session_options = self.ort.SessionOptions()
            session_options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = self.ort.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=self.providers
            )
            
            # Get input/output names and shapes
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            
            print(f"✅ GPEN-BFR initialized successfully")
            print(f"   Model: {os.path.basename(model_path)}")
            print(f"   Providers: {self.session.get_providers()}")
            print(f"   Input shape: {self.input_shape}")
            print(f"   Output shape: {self.output_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPEN-BFR session: {e}")
    
    def _setup_providers(self, device: str, available_providers: List[str]) -> List[str]:
        """Setup ONNX execution providers based on device preference"""
        providers = []
        
        if device == "auto" or device == "cuda":
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
        
        if device == "auto" or device == "cpu":
            if 'CPUExecutionProvider' in available_providers:
                providers.append(('CPUExecutionProvider', {
                    'intra_op_num_threads': 4,
                    'inter_op_num_threads': 4,
                }))
        
        # Fallback to simple provider names if complex config fails
        if not providers:
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        return providers
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for GPEN-BFR model
        
        Args:
            face_image: Input face image (BGR format, any size)
            
        Returns:
            Preprocessed tensor ready for ONNX inference
        """
        # Ensure we have a valid image
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid input face image")
        
        # Resize to 256x256 if needed
        if face_image.shape[:2] != (256, 256):
            face_image = cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert BGR to RGB (GPEN-BFR expects RGB)
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] range (typical for GAN models)
        face_tensor = face_image.astype(np.float32) / 127.5 - 1.0
        
        # Convert HWC to CHW format
        face_tensor = np.transpose(face_tensor, (2, 0, 1))
        
        # Add batch dimension
        face_tensor = np.expand_dims(face_tensor, axis=0)
        
        return face_tensor
    
    def postprocess_face(self, output_tensor: np.ndarray) -> np.ndarray:
        """
        Postprocess GPEN-BFR output tensor to image
        
        Args:
            output_tensor: Model output tensor
            
        Returns:
            Enhanced face image in BGR format
        """
        # Remove batch dimension
        output_tensor = np.squeeze(output_tensor, axis=0)
        
        # Convert CHW to HWC
        output_tensor = np.transpose(output_tensor, (1, 2, 0))
        
        # Denormalize from [-1, 1] to [0, 255]
        output_tensor = (output_tensor + 1.0) * 127.5
        output_tensor = np.clip(output_tensor, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR for OpenCV compatibility
        output_tensor = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)
        
        return output_tensor
    
    def enhance_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance a single face image using GPEN-BFR
        
        Args:
            face_image: Input face image (BGR format, preferably 256x256)
            
        Returns:
            Enhanced face image (BGR format, 256x256)
        """
        try:
            # Preprocess
            input_tensor = self.preprocess_face(face_image)
            
            # Run inference
            output_tensor = self.session.run(
                [self.output_name], 
                {self.input_name: input_tensor}
            )[0]
            
            # Postprocess
            enhanced_face = self.postprocess_face(output_tensor)
            
            return enhanced_face
            
        except Exception as e:
            print(f"⚠️ GPEN-BFR enhancement failed for face: {e}")
            # Return original face on failure
            if face_image.shape[:2] != (256, 256):
                return cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            return face_image.copy()
    
    def enhance_batch(self, face_images: List[np.ndarray], show_progress: bool = True) -> List[np.ndarray]:
        """
        Enhance multiple face images
        
        Args:
            face_images: List of face images (BGR format)
            show_progress: Whether to show progress information
            
        Returns:
            List of enhanced face images (BGR format, 256x256)
        """
        enhanced_faces = []
        total_faces = len(face_images)
        
        for i, face in enumerate(face_images):
            if show_progress:
                print(f"🎨 Enhancing face {i+1}/{total_faces} with GPEN-BFR...")
            
            enhanced = self.enhance_face(face)
            enhanced_faces.append(enhanced)
        
        if show_progress:
            print(f"✅ Enhanced {total_faces} faces with GPEN-BFR")
        
        return enhanced_faces
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'providers': self.session.get_providers() if hasattr(self, 'session') else [],
            'input_shape': self.input_shape if hasattr(self, 'input_shape') else None,
            'output_shape': self.output_shape if hasattr(self, 'output_shape') else None,
            'device': self.device
        }


def test_gpen_bfr_availability():
    """Test function to check if GPEN-BFR can be initialized"""
    try:
        enhancer = GPENBFREnhancer()
        print("✅ GPEN-BFR is available and ready to use!")
        print(f"Model info: {enhancer.get_model_info()}")
        return True
    except Exception as e:
        print(f"❌ GPEN-BFR test failed: {e}")
        return False


if __name__ == "__main__":
    # Test GPEN-BFR availability
    print("🧪 Testing GPEN-BFR availability...")
    test_gpen_bfr_availability()
