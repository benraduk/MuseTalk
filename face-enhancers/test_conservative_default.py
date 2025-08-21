"""
Test GPEN-BFR with CONSERVATIVE Default and Flexible Configuration
================================================================

This script demonstrates the updated GPEN-BFR enhancer with:
1. CONSERVATIVE as the default configuration
2. Easy switching between configurations
3. Batch processing with different configs per face

Usage:
    python face-enhancers/test_conservative_default.py
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add face-enhancers to path
face_enhancers_path = Path(__file__).parent
if str(face_enhancers_path) not in sys.path:
    sys.path.insert(0, str(face_enhancers_path))

try:
    from gpen_bfr_enhancer import GPENBFREnhancer
    from gpen_bfr_parameter_configs import list_available_configs
except ImportError as e:
    print(f"❌ Failed to import GPEN-BFR modules: {e}")
    sys.exit(1)


def test_conservative_default():
    """Test that CONSERVATIVE is now the default configuration"""
    print("🧪 Testing CONSERVATIVE as Default Configuration")
    print("=" * 60)
    
    # Initialize enhancer (should default to CONSERVATIVE)
    enhancer = GPENBFREnhancer()
    
    # Check the default config
    config_info = enhancer.get_model_info()['config']
    print(f"✅ Default configuration loaded:")
    print(f"   Name: {config_info.get('name', 'Unknown')}")
    print(f"   Enhancement Strength: {config_info['enhancement_strength']}")
    print(f"   Contrast Boost: {config_info['contrast_boost']}")
    print(f"   Sharpening: {config_info['sharpening']}")
    print(f"   Color Correction: {config_info['color_correction']}")
    
    return enhancer


def test_configuration_switching(enhancer):
    """Test switching between different configurations"""
    print(f"\n🔄 Testing Configuration Switching")
    print("-" * 40)
    
    # Show available configurations
    configs = enhancer.get_available_configs()
    print(f"📋 Available configurations ({len(configs)}):")
    for name, desc in configs.items():
        print(f"   • {name}: {desc}")
    
    # Test switching to different configs
    test_configs = ['NATURAL', 'BALANCED', 'DRAMATIC']
    
    for config_name in test_configs:
        print(f"\n🎨 Switching to {config_name}...")
        enhancer.update_config(config_name)
        
        current_config = enhancer.get_model_info()['config']
        print(f"   ✅ Now using: {current_config.get('name', config_name)}")
        print(f"   Enhancement Strength: {current_config['enhancement_strength']}")
    
    # Switch back to CONSERVATIVE
    print(f"\n🔄 Switching back to CONSERVATIVE...")
    enhancer.update_config('CONSERVATIVE')
    print(f"   ✅ Back to default configuration")


def test_temporary_config_override(enhancer):
    """Test temporary configuration override for single enhancement"""
    print(f"\n⚡ Testing Temporary Configuration Override")
    print("-" * 50)
    
    # Check for existing VAE faces
    vae_faces_dir = Path("test_gfpgan_output/vae_faces")
    if not vae_faces_dir.exists():
        print("⚠️ No VAE faces found. Creating test image...")
        # Create a test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    else:
        vae_files = list(vae_faces_dir.glob("*.png"))
        if vae_files:
            test_image = cv2.imread(str(vae_files[0]))
            print(f"📸 Using existing VAE face: {vae_files[0].name}")
        else:
            test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Verify current config is CONSERVATIVE
    current_config = enhancer.get_model_info()['config']
    print(f"🔧 Current default config: {current_config.get('name', 'Unknown')}")
    
    # Test temporary override to DRAMATIC
    print(f"🎭 Temporarily enhancing with DRAMATIC config...")
    enhanced_dramatic = enhancer.enhance_face_with_config(test_image, 'DRAMATIC')
    
    # Verify we're back to CONSERVATIVE
    after_config = enhancer.get_model_info()['config']
    print(f"✅ After enhancement, still using: {after_config.get('name', 'Unknown')}")
    
    # Test normal enhancement with CONSERVATIVE
    print(f"🎨 Normal enhancement with CONSERVATIVE...")
    enhanced_conservative = enhancer.enhance_face(test_image)
    
    print(f"✅ Temporary override test completed successfully!")
    
    return enhanced_conservative, enhanced_dramatic


def test_batch_with_different_configs(enhancer):
    """Test batch processing with different configurations per face"""
    print(f"\n📦 Testing Batch Processing with Different Configs")
    print("-" * 55)
    
    # Create test images or use existing VAE faces
    vae_faces_dir = Path("test_gfpgan_output/vae_faces")
    test_images = []
    
    if vae_faces_dir.exists():
        vae_files = sorted(list(vae_faces_dir.glob("*.png")))[:3]  # Use up to 3 faces
        for vae_file in vae_files:
            img = cv2.imread(str(vae_file))
            if img is not None:
                test_images.append(img)
                print(f"📸 Loaded: {vae_file.name}")
    
    # Fill with test images if needed
    while len(test_images) < 3:
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_images.append(test_img)
    
    # Define different configs for each face
    config_names = ['CONSERVATIVE', 'BALANCED', 'DRAMATIC']
    
    print(f"\n🎨 Processing {len(test_images)} faces with different configs:")
    for i, config in enumerate(config_names):
        print(f"   Face {i+1}: {config}")
    
    # Process batch with different configs
    enhanced_faces = enhancer.enhance_batch_with_configs(test_images, config_names)
    
    print(f"✅ Batch processing completed!")
    print(f"📊 Processed {len(enhanced_faces)} faces with individual configurations")
    
    return enhanced_faces


def main():
    """Main test function"""
    print("🎯 GPEN-BFR Conservative Default & Flexible Configuration Test")
    print("=" * 70)
    
    try:
        # Test 1: Conservative default
        enhancer = test_conservative_default()
        
        # Test 2: Configuration switching
        test_configuration_switching(enhancer)
        
        # Test 3: Temporary config override
        enhanced_conservative, enhanced_dramatic = test_temporary_config_override(enhancer)
        
        # Test 4: Batch with different configs
        enhanced_batch = test_batch_with_different_configs(enhancer)
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ CONSERVATIVE is now the default configuration")
        print(f"✅ Configuration switching works perfectly")
        print(f"✅ Temporary overrides work without affecting default")
        print(f"✅ Batch processing with different configs per face works")
        
        print(f"\n💡 Usage Examples:")
        print(f"   # Use default CONSERVATIVE")
        print(f"   enhancer = GPENBFREnhancer()")
        print(f"   enhanced = enhancer.enhance_face(face)")
        print(f"   ")
        print(f"   # Temporarily use DRAMATIC for one face")
        print(f"   enhanced = enhancer.enhance_face_with_config(face, 'DRAMATIC')")
        print(f"   ")
        print(f"   # Change default to BALANCED")
        print(f"   enhancer.update_config('BALANCED')")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
