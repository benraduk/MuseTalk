"""
Quick GPEN-BFR Setup Test
========================

This script quickly tests if GPEN-BFR can be initialized and is ready to use.
Run this before the full test to verify your setup.

Usage:
    python face-enhancers/test_gpen_bfr_setup.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('.')
sys.path.append('..')

def test_onnx_installation():
    """Test if ONNX runtime is properly installed"""
    print("🔧 Testing ONNX Runtime installation...")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime installed successfully")
        print(f"   Available providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA provider available (GPU acceleration enabled)")
        else:
            print("⚠️ CUDA provider not available (CPU only)")
        
        return True
        
    except ImportError as e:
        print(f"❌ ONNX Runtime not installed: {e}")
        print("💡 Install with: pip install onnxruntime-gpu")
        return False


def test_model_download():
    """Test if GPEN-BFR model is downloaded"""
    print("\n📥 Testing GPEN-BFR model availability...")
    
    model_path = Path("models/gpen_bfr/gpen_bfr_256.onnx")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ GPEN-BFR model found: {model_path}")
        print(f"   Model size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ GPEN-BFR model not found: {model_path}")
        print("💡 Download with:")
        print("   Windows: download_weights.bat")
        print("   Linux/macOS: ./download_weights.sh")
        return False


def test_gpen_bfr_import():
    """Test if GPEN-BFR enhancer can be imported"""
    print("\n🔌 Testing GPEN-BFR enhancer import...")
    
    try:
        # Add face-enhancers to path
        import sys
        from pathlib import Path
        face_enhancers_path = Path(__file__).parent
        if str(face_enhancers_path) not in sys.path:
            sys.path.insert(0, str(face_enhancers_path))
        
        from gpen_bfr_enhancer import GPENBFREnhancer
        print("✅ GPEN-BFR enhancer imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import GPEN-BFR enhancer: {e}")
        print("💡 Make sure you're running from the MuseTalk root directory")
        return False


def test_gpen_bfr_initialization():
    """Test if GPEN-BFR enhancer can be initialized"""
    print("\n🚀 Testing GPEN-BFR enhancer initialization...")
    
    try:
        # Add face-enhancers to path
        import sys
        from pathlib import Path
        face_enhancers_path = Path(__file__).parent
        if str(face_enhancers_path) not in sys.path:
            sys.path.insert(0, str(face_enhancers_path))
        
        from gpen_bfr_enhancer import GPENBFREnhancer
        
        enhancer = GPENBFREnhancer()
        model_info = enhancer.get_model_info()
        
        print("✅ GPEN-BFR enhancer initialized successfully")
        print(f"   Providers: {model_info['providers']}")
        print(f"   Input shape: {model_info['input_shape']}")
        print(f"   Output shape: {model_info['output_shape']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize GPEN-BFR enhancer: {e}")
        return False


def test_simple_enhancement():
    """Test a simple enhancement operation"""
    print("\n🎨 Testing simple face enhancement...")
    
    try:
        import numpy as np
        import cv2
        
        # Add face-enhancers to path
        import sys
        from pathlib import Path
        face_enhancers_path = Path(__file__).parent
        if str(face_enhancers_path) not in sys.path:
            sys.path.insert(0, str(face_enhancers_path))
        
        from gpen_bfr_enhancer import GPENBFREnhancer
        
        # Create a simple test image (256x256, BGR)
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Initialize enhancer
        enhancer = GPENBFREnhancer()
        
        # Enhance the test image
        enhanced = enhancer.enhance_face(test_image)
        
        # Verify output
        if enhanced.shape == (256, 256, 3) and enhanced.dtype == np.uint8:
            print("✅ Simple enhancement test passed")
            print(f"   Input shape: {test_image.shape}")
            print(f"   Output shape: {enhanced.shape}")
            return True
        else:
            print(f"❌ Enhancement output has wrong format: {enhanced.shape}, {enhanced.dtype}")
            return False
            
    except Exception as e:
        print(f"❌ Simple enhancement test failed: {e}")
        return False


def main():
    """Run all setup tests"""
    print("🧪 GPEN-BFR Setup Test")
    print("=" * 40)
    
    tests = [
        ("ONNX Installation", test_onnx_installation),
        ("Model Download", test_model_download),
        ("GPEN-BFR Import", test_gpen_bfr_import),
        ("GPEN-BFR Initialization", test_gpen_bfr_initialization),
        ("Simple Enhancement", test_simple_enhancement),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! GPEN-BFR is ready to use.")
        print("💡 You can now run: python face-enhancers/test_gpen_bfr_on_vae_faces.py")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
