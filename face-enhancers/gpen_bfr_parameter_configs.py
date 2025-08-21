"""
GPEN-BFR Parameter Configurations for MuseTalk
==============================================

This module defines various parameter configurations for GPEN-BFR face enhancement.
Unlike GFPGAN, GPEN-BFR uses different approaches for parameter tuning including
preprocessing normalization, post-processing adjustments, and blending techniques.

Key Parameters:
- normalization_range: Input normalization range ([-1,1] vs [0,1])
- enhancement_strength: Blending ratio between original and enhanced
- contrast_boost: Post-processing contrast adjustment
- sharpening: Post-processing sharpening filter
- color_correction: Color space adjustments
- noise_reduction: Pre-processing noise reduction
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple

# Parameter configuration presets
GPEN_BFR_CONFIGS = {
    'NATURAL': {
        'name': 'Natural Enhancement',
        'description': 'Subtle enhancement preserving original characteristics',
        'normalization_range': (-1, 1),
        'enhancement_strength': 0.7,  # Blend 70% enhanced, 30% original
        'contrast_boost': 1.05,
        'sharpening': 0.3,
        'color_correction': 'none',
        'noise_reduction': 0.1,
        'gamma_correction': 1.0
    },
    
    'BALANCED': {
        'name': 'Balanced Enhancement',
        'description': 'Good balance between enhancement and naturalness',
        'normalization_range': (-1, 1),
        'enhancement_strength': 0.85,
        'contrast_boost': 1.1,
        'sharpening': 0.5,
        'color_correction': 'slight',
        'noise_reduction': 0.2,
        'gamma_correction': 1.05
    },
    
    'QUALITY_FOCUSED': {
        'name': 'Quality Focused',
        'description': 'Maximum quality enhancement for best visual results',
        'normalization_range': (-1, 1),
        'enhancement_strength': 1.0,  # Full enhancement
        'contrast_boost': 1.15,
        'sharpening': 0.7,
        'color_correction': 'moderate',
        'noise_reduction': 0.3,
        'gamma_correction': 1.1
    },
    
    'CONSERVATIVE': {
        'name': 'Conservative Enhancement',
        'description': 'Minimal enhancement for subtle improvements',
        'normalization_range': (-1, 1),
        'enhancement_strength': 0.5,
        'contrast_boost': 1.02,
        'sharpening': 0.1,
        'color_correction': 'none',
        'noise_reduction': 0.05,
        'gamma_correction': 1.0
    },
    
    'DRAMATIC': {
        'name': 'Dramatic Enhancement',
        'description': 'Strong enhancement for maximum visual impact',
        'normalization_range': (-1, 1),
        'enhancement_strength': 1.0,
        'contrast_boost': 1.25,
        'sharpening': 1.0,
        'color_correction': 'strong',
        'noise_reduction': 0.4,
        'gamma_correction': 1.15
    },
    
    'SKIN_FOCUS': {
        'name': 'Skin Focused',
        'description': 'Optimized for skin texture and smoothness',
        'normalization_range': (-1, 1),
        'enhancement_strength': 0.8,
        'contrast_boost': 1.08,
        'sharpening': 0.2,  # Less sharpening for smoother skin
        'color_correction': 'skin_tone',
        'noise_reduction': 0.5,  # More noise reduction for smooth skin
        'gamma_correction': 1.03
    },
    
    'DETAIL_ENHANCE': {
        'name': 'Detail Enhancement',
        'description': 'Focus on enhancing fine details and textures',
        'normalization_range': (-1, 1),
        'enhancement_strength': 0.9,
        'contrast_boost': 1.2,
        'sharpening': 0.8,  # High sharpening for details
        'color_correction': 'moderate',
        'noise_reduction': 0.1,  # Less noise reduction to preserve details
        'gamma_correction': 1.08
    },
    
    'LIPS_OPTIMIZED': {
        'name': 'Lips Optimized',
        'description': 'Specifically tuned for lip region enhancement',
        'normalization_range': (-1, 1),
        'enhancement_strength': 0.9,
        'contrast_boost': 1.12,
        'sharpening': 0.6,
        'color_correction': 'lip_enhance',
        'noise_reduction': 0.25,
        'gamma_correction': 1.06
    }
}


def apply_preprocessing(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Apply preprocessing based on configuration"""
    processed = image.copy()
    
    # Noise reduction
    if config['noise_reduction'] > 0:
        kernel_size = int(config['noise_reduction'] * 10) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        processed = cv2.bilateralFilter(processed, kernel_size, 
                                      config['noise_reduction'] * 80, 
                                      config['noise_reduction'] * 80)
    
    return processed


def apply_postprocessing(original: np.ndarray, enhanced: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Apply post-processing based on configuration"""
    
    # Enhancement strength blending
    strength = config['enhancement_strength']
    if strength < 1.0:
        blended = cv2.addWeighted(enhanced, strength, original, 1 - strength, 0)
    else:
        blended = enhanced.copy()
    
    # Contrast boost
    if config['contrast_boost'] != 1.0:
        blended = cv2.convertScaleAbs(blended, alpha=config['contrast_boost'], beta=0)
    
    # Gamma correction
    if config['gamma_correction'] != 1.0:
        gamma = config['gamma_correction']
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        blended = cv2.LUT(blended, table)
    
    # Sharpening
    if config['sharpening'] > 0:
        blended = apply_sharpening(blended, config['sharpening'])
    
    # Color correction
    if config['color_correction'] != 'none':
        blended = apply_color_correction(blended, config['color_correction'])
    
    return blended


def apply_sharpening(image: np.ndarray, strength: float) -> np.ndarray:
    """Apply unsharp masking for sharpening"""
    if strength <= 0:
        return image
    
    # Create gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    
    # Create unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return unsharp_mask


def apply_color_correction(image: np.ndarray, correction_type: str) -> np.ndarray:
    """Apply color correction based on type"""
    if correction_type == 'none':
        return image
    
    corrected = image.copy()
    
    if correction_type == 'slight':
        # Slight saturation boost
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    elif correction_type == 'moderate':
        # Moderate saturation and slight hue adjustment
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    elif correction_type == 'strong':
        # Strong color enhancement
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.05)  # Slight brightness boost
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    elif correction_type == 'skin_tone':
        # Optimize for skin tones
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.multiply(lab[:, :, 0], 1.02)  # Slight lightness boost
        lab[:, :, 1] = cv2.multiply(lab[:, :, 1], 0.98)  # Reduce green tint
        lab[:, :, 2] = cv2.multiply(lab[:, :, 2], 1.02)  # Slight warmth boost
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
    elif correction_type == 'lip_enhance':
        # Enhance lip colors (boost reds)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        # Create mask for red hues (lips)
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Enhance saturation in red areas
        hsv[:, :, 1] = np.where(red_mask > 0, 
                               cv2.multiply(hsv[:, :, 1], 1.3), 
                               cv2.multiply(hsv[:, :, 1], 1.1))
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return corrected


def get_config_by_name(config_name: str) -> Dict[str, Any]:
    """Get configuration by name"""
    return GPEN_BFR_CONFIGS.get(config_name.upper(), GPEN_BFR_CONFIGS['BALANCED'])


def list_available_configs() -> Dict[str, str]:
    """List all available configurations with descriptions"""
    return {name: config['description'] for name, config in GPEN_BFR_CONFIGS.items()}


def create_custom_config(base_config: str = 'BALANCED', **overrides) -> Dict[str, Any]:
    """Create a custom configuration based on a base config with overrides"""
    config = GPEN_BFR_CONFIGS[base_config.upper()].copy()
    config.update(overrides)
    return config


# Example usage and testing
if __name__ == "__main__":
    print("🎨 GPEN-BFR Parameter Configurations")
    print("=" * 50)
    
    configs = list_available_configs()
    for name, description in configs.items():
        print(f"📋 {name}: {description}")
    
    print(f"\n🔧 Total configurations available: {len(configs)}")
    
    # Show parameter details for a few key configs
    key_configs = ['NATURAL', 'BALANCED', 'QUALITY_FOCUSED']
    print(f"\n📊 Key Configuration Details:")
    print("-" * 40)
    
    for config_name in key_configs:
        config = GPEN_BFR_CONFIGS[config_name]
        print(f"\n{config_name}:")
        print(f"  Enhancement Strength: {config['enhancement_strength']}")
        print(f"  Contrast Boost: {config['contrast_boost']}")
        print(f"  Sharpening: {config['sharpening']}")
        print(f"  Color Correction: {config['color_correction']}")
        print(f"  Noise Reduction: {config['noise_reduction']}")
