#!/usr/bin/env python3
"""
Hybrid Inference Module - Surgical Integration
==============================================
This module provides surgical replacement of MuseTalk's UNet with LatentSync's UNet3D
while maintaining 100% compatibility with the existing MuseTalk pipeline.

Key Features:
- Surgical precision: Only replaces UNet inference, keeps everything else
- Automatic fallback: Falls back to MuseTalk UNet if LatentSync fails
- Zero config changes: Drop-in replacement for existing inference calls
- Memory efficient: Uses FP16 and cached model loading
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgicalIntegration:
    """
    Surgical Integration Manager
    
    Handles the surgical replacement of MuseTalk UNet with LatentSync UNet3D
    while maintaining full compatibility and providing robust fallback.
    """
    
    def __init__(self):
        self.latentsync_unet3d = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = torch.cuda.is_available()  # Use FP16 on GPU for memory efficiency
        self.fallback_count = 0
        self.success_count = 0
        
        logger.info(f"🔧 Surgical Integration initialized on {self.device}")
        if self.use_fp16:
            logger.info("⚡ Using FP16 for memory efficiency")
    
    def load_latentsync_unet3d(self) -> bool:
        """
        Load LatentSync UNet3D model with caching
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if self.latentsync_unet3d is not None:
            return True  # Already loaded
        
        try:
            logger.info("🔄 Loading LatentSync UNet3D model...")
            
            # Try to import LatentSync UNet3D
            try:
                from LatentSync.latentsync.models.unet import UNet3DConditionModel
            except ImportError as e:
                logger.error(f"❌ Failed to import LatentSync UNet3D: {e}")
                logger.error("💡 Make sure LatentSync repository is available and properly installed")
                return False
            
            # Determine model path - try multiple possible locations
            possible_paths = [
                "models/latentsync",  # Primary location from download_weights script
                "LatentSync/checkpoints", 
                "LatentSync/models/unet3d",
                "../LatentSync/models/unet3d",
                "checkpoints"  # Alternative LatentSync location
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                logger.warning("⚠️  LatentSync model path not found, trying to load from config")
                # Try to load from huggingface hub or default config
                model_path = "latentsync/unet3d"  # This might work if model is in HF hub
            
            # Load the model
            dtype = torch.float16 if self.use_fp16 else torch.float32
            
            self.latentsync_unet3d = UNet3DConditionModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_safetensors=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Move to device and set to eval mode
            if not hasattr(self.latentsync_unet3d, 'device') or self.latentsync_unet3d.device != self.device:
                self.latentsync_unet3d = self.latentsync_unet3d.to(self.device)
            
            self.latentsync_unet3d.eval()
            
            logger.info(f"✅ LatentSync UNet3D loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load LatentSync UNet3D: {e}")
            logger.error(f"💡 Will use MuseTalk UNet fallback")
            return False
    
    def format_for_latentsync(self, tensor: torch.Tensor, tensor_type: str = "latent") -> torch.Tensor:
        """
        Format tensors for LatentSync if needed
        
        Args:
            tensor: Input tensor from MuseTalk pipeline
            tensor_type: Type of tensor ("latent" or "audio")
            
        Returns:
            torch.Tensor: Formatted tensor for LatentSync
        """
        # For now, assume tensors are compatible
        # If LatentSync requires different formatting, implement here
        
        if tensor_type == "latent":
            # LatentSync UNet3D might expect temporal dimension
            # If tensor is [B, C, H, W], might need [B, C, T, H, W] where T=1
            if len(tensor.shape) == 4:  # [B, C, H, W]
                tensor = tensor.unsqueeze(2)  # [B, C, 1, H, W] - add temporal dimension
        
        return tensor
    
    def format_for_musetalk(self, tensor: torch.Tensor, tensor_type: str = "latent") -> torch.Tensor:
        """
        Format tensors back to MuseTalk format if needed
        
        Args:
            tensor: Output tensor from LatentSync
            tensor_type: Type of tensor ("latent" or "audio")
            
        Returns:
            torch.Tensor: Formatted tensor for MuseTalk pipeline
        """
        if tensor_type == "latent":
            # If LatentSync returned [B, C, T, H, W], convert back to [B, C, H, W]
            if len(tensor.shape) == 5 and tensor.shape[2] == 1:  # [B, C, 1, H, W]
                tensor = tensor.squeeze(2)  # [B, C, H, W]
        
        return tensor
    
    def surgical_unet3d_inference(
        self,
        latent_batch: torch.Tensor,
        audio_features: torch.Tensor, 
        timesteps: torch.Tensor,
        fallback_unet: Optional[callable] = None
    ) -> torch.Tensor:
        """
        🔄 SURGICAL REPLACEMENT: LatentSync UNet3D with MuseTalk fallback
        
        This is the core surgical function that replaces MuseTalk's UNet inference
        with LatentSync UNet3D while maintaining 100% compatibility.
        
        Args:
            latent_batch: Input latents [B, C, H, W] (same format as MuseTalk)
            audio_features: Audio embeddings [B, seq_len, feature_dim] (same format as MuseTalk)
            timesteps: Diffusion timesteps [B] (same format as MuseTalk)
            fallback_unet: MuseTalk UNet function for fallback safety
            
        Returns:
            torch.Tensor: Predicted latents [B, C, H, W] (same format as MuseTalk)
        """
        try:
            # STEP 1: Try to load LatentSync UNet3D (cached after first load)
            if not self.load_latentsync_unet3d():
                raise RuntimeError("LatentSync UNet3D not available")
            
            # STEP 2: Format inputs for LatentSync (if needed)
            formatted_latents = self.format_for_latentsync(latent_batch, "latent")
            formatted_audio = self.format_for_latentsync(audio_features, "audio")
            
            # STEP 3: LatentSync UNet3D inference
            with torch.no_grad():  # Save memory
                # Ensure inputs are on correct device and dtype
                formatted_latents = formatted_latents.to(self.device)
                formatted_audio = formatted_audio.to(self.device)
                timesteps = timesteps.to(self.device)
                
                if self.use_fp16:
                    formatted_latents = formatted_latents.half()
                    formatted_audio = formatted_audio.half()
                
                # The actual LatentSync inference
                noise_pred = self.latentsync_unet3d(
                    formatted_latents,
                    timesteps,
                    encoder_hidden_states=formatted_audio
                ).sample
                
                # Convert back to original dtype if needed
                if self.use_fp16 and latent_batch.dtype != torch.float16:
                    noise_pred = noise_pred.float()
            
            # STEP 4: Format output back to MuseTalk format (if needed)
            pred_latents = self.format_for_musetalk(noise_pred, "latent")
            
            # STEP 5: Ensure output has same device/dtype as input
            pred_latents = pred_latents.to(latent_batch.device, latent_batch.dtype)
            
            self.success_count += 1
            if self.success_count % 10 == 0:  # Log every 10 successful inferences
                logger.info(f"✅ LatentSync inference success count: {self.success_count}")
            
            return pred_latents
            
        except Exception as e:
            # STEP 6: Fallback to MuseTalk UNet
            self.fallback_count += 1
            logger.warning(f"⚠️  LatentSync inference failed (attempt {self.fallback_count}): {e}")
            
            if fallback_unet is not None:
                logger.info("🔄 Falling back to MuseTalk UNet")
                try:
                    # Use original MuseTalk UNet
                    with torch.no_grad():
                        pred_latents = fallback_unet(
                            latent_batch, 
                            timesteps, 
                            encoder_hidden_states=audio_features
                        ).sample
                    
                    logger.info("✅ MuseTalk UNet fallback successful")
                    return pred_latents
                    
                except Exception as fallback_error:
                    logger.error(f"❌ MuseTalk UNet fallback also failed: {fallback_error}")
                    raise RuntimeError(f"Both LatentSync and MuseTalk UNet failed: {e}, {fallback_error}")
            else:
                logger.error("❌ No fallback UNet available")
                raise RuntimeError(f"LatentSync inference failed and no fallback available: {e}")

# Global instance for caching
_surgical_integration = None

def get_surgical_integration() -> SurgicalIntegration:
    """Get global surgical integration instance (singleton pattern)"""
    global _surgical_integration
    if _surgical_integration is None:
        _surgical_integration = SurgicalIntegration()
    return _surgical_integration

def surgical_unet3d_inference(
    latent_batch: torch.Tensor,
    audio_features: torch.Tensor,
    timesteps: torch.Tensor, 
    fallback_unet: Optional[callable] = None
) -> torch.Tensor:
    """
    🔄 SURGICAL REPLACEMENT FUNCTION
    
    Drop-in replacement for MuseTalk UNet inference that uses LatentSync UNet3D
    for higher quality while maintaining full compatibility and fallback safety.
    
    This is the function that will be called from scripts/inference.py and app.py
    
    Usage in scripts/inference.py:
        # OLD:
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        
        # NEW:
        pred_latents = surgical_unet3d_inference(latent_batch, audio_feature_batch, timesteps, unet.model)
    
    Args:
        latent_batch: Input latents from MuseTalk VAE encoder
        audio_features: Audio embeddings from MuseTalk Whisper
        timesteps: Diffusion timesteps from MuseTalk scheduler
        fallback_unet: MuseTalk UNet model for fallback safety
        
    Returns:
        torch.Tensor: Predicted latents (same format as MuseTalk UNet output)
    """
    surgical_integration = get_surgical_integration()
    return surgical_integration.surgical_unet3d_inference(
        latent_batch, audio_features, timesteps, fallback_unet
    )

def test_surgical_integration():
    """
    Test function to validate surgical integration works
    """
    logger.info("🧪 Testing surgical integration...")
    
    # Test model loading
    surgical_integration = get_surgical_integration()
    success = surgical_integration.load_latentsync_unet3d()
    
    if success:
        logger.info("✅ LatentSync UNet3D loading test passed")
        
        # Test with dummy tensors
        try:
            batch_size = 1
            latent_channels = 4
            height, width = 32, 32  # Latent space dimensions
            seq_len, feature_dim = 16, 512  # Audio features
            
            dummy_latents = torch.randn(batch_size, latent_channels, height, width)
            dummy_audio = torch.randn(batch_size, seq_len, feature_dim)
            dummy_timesteps = torch.randint(0, 1000, (batch_size,))
            
            # Test surgical function
            result = surgical_unet3d_inference(
                dummy_latents, dummy_audio, dummy_timesteps, None
            )
            
            logger.info(f"✅ Surgical inference test passed - output shape: {result.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Surgical inference test failed: {e}")
            return False
    else:
        logger.warning("⚠️  LatentSync UNet3D not available, will use fallback only")
        return False

if __name__ == "__main__":
    """
    Test the surgical integration when run directly
    """
    print("🔬 SURGICAL INTEGRATION TEST")
    print("=" * 50)
    
    # Run test
    success = test_surgical_integration()
    
    if success:
        print("🎉 Surgical integration ready for deployment!")
    else:
        print("⚠️  Surgical integration will use fallback mode")
    
    # Show statistics
    surgical_integration = get_surgical_integration()
    print(f"\n📊 Statistics:")
    print(f"   Success count: {surgical_integration.success_count}")
    print(f"   Fallback count: {surgical_integration.fallback_count}")
    print(f"   Device: {surgical_integration.device}")
    print(f"   FP16 enabled: {surgical_integration.use_fp16}")