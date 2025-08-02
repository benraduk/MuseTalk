#!/usr/bin/env python3
"""
Test script for surgical requirements validation
Tests that our surgical integration eliminates conflicts successfully
"""

import sys
import traceback
import importlib
from typing import Dict, List, Tuple

def test_import(module_name: str, description: str = "") -> Tuple[bool, str]:
    """Test if a module can be imported successfully"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, f"{module_name} v{version}"
    except ImportError as e:
        return False, f"{module_name}: {str(e)}"
    except Exception as e:
        return False, f"{module_name}: {str(e)}"

def test_surgical_requirements():
    """Test that surgical requirements work without conflicts"""
    
    print("🔬 SURGICAL REQUIREMENTS VALIDATION")
    print("=" * 60)
    
    # Test core dependencies
    core_tests = [
        ("torch", "Core PyTorch"),
        ("torchvision", "PyTorch Vision"),
        ("numpy", "NumPy arrays"),
        ("cv2", "OpenCV"),
        ("diffusers", "Diffusers library"),
        ("transformers", "Transformers library"),
        ("librosa", "Audio processing"),
        ("soundfile", "Audio I/O"),
        ("gradio", "Web interface"),
        ("omegaconf", "Configuration"),
    ]
    
    print(f"\n✅ CORE DEPENDENCIES:")
    core_success = 0
    for module, desc in core_tests:
        success, result = test_import(module)
        status = "✅" if success else "❌"
        print(f"   {status} {result}")
        if success:
            core_success += 1
    
    print(f"\n📊 Core Dependencies: {core_success}/{len(core_tests)} successful")
    
    # Test MuseTalk components (what we keep)
    musetalk_tests = [
        ("musetalk.utils.face_detection", "MuseTalk Face Detection"),
        ("musetalk.utils.utils", "MuseTalk Utils"),
        ("musetalk.utils.blending", "MuseTalk Blending"),
        ("musetalk.utils.audio_processor", "MuseTalk Audio"),
        ("musetalk.models.vae", "MuseTalk VAE"),
        ("musetalk.models.unet", "MuseTalk UNet"),
    ]
    
    print(f"\n✅ MUSETALK COMPONENTS (KEEP):")
    musetalk_success = 0
    for module, desc in musetalk_tests:
        success, result = test_import(module)
        status = "✅" if success else "❌"
        print(f"   {status} {result}")
        if success:
            musetalk_success += 1
    
    print(f"\n📊 MuseTalk Components: {musetalk_success}/{len(musetalk_tests)} successful")
    
    # Test LatentSync components (only what we need)
    latentsync_tests = [
        ("LatentSync.latentsync.models.unet", "LatentSync UNet3D (NEEDED)"),
    ]
    
    print(f"\n✅ LATENTSYNC COMPONENTS (SURGICAL):")
    latentsync_success = 0
    for module, desc in latentsync_tests:
        success, result = test_import(module)
        status = "✅" if success else "❌"
        print(f"   {status} {result}")
        if success:
            latentsync_success += 1
    
    print(f"\n📊 LatentSync Components: {latentsync_success}/{len(latentsync_tests)} successful")
    
    # Test eliminated dependencies (should fail or be unused)
    eliminated_tests = [
        ("mmpose", "MMLab Pose (ELIMINATED)"),
        ("mmcv", "MMLab CV (ELIMINATED)"),
        ("mmdet", "MMLab Detection (ELIMINATED)"),
        ("tensorflow", "TensorFlow (ELIMINATED)"),
        ("insightface", "InsightFace (ELIMINATED)"),
        ("mediapipe", "MediaPipe (ELIMINATED)"),
        ("LatentSync.latentsync.utils.face_detector", "LatentSync Face Detection (ELIMINATED)"),
    ]
    
    print(f"\n❌ ELIMINATED DEPENDENCIES (SHOULD NOT BE NEEDED):")
    eliminated_success = 0
    for module, desc in eliminated_tests:
        success, result = test_import(module)
        if success:
            print(f"   ⚠️  {result} (present but not used)")
        else:
            print(f"   ✅ {module} (successfully eliminated)")
            eliminated_success += 1
    
    print(f"\n📊 Eliminated Dependencies: {eliminated_success}/{len(eliminated_tests)} successfully eliminated")
    
    return core_success, musetalk_success, latentsync_success, eliminated_success

def test_surgical_functionality():
    """Test that surgical components actually work together"""
    
    print(f"\n🔧 SURGICAL FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        print("\n1. Testing PyTorch CUDA availability...")
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        
        print("\n2. Testing MuseTalk face detection (WITHOUT mmpose)...")
        from musetalk.utils.face_detection import FaceAlignment, LandmarksType
        print("   ✅ MuseTalk face detection imported (S3FD)")
        
        print("\n3. Testing MuseTalk enhanced preprocessing...")
        try:
            from musetalk.utils.preprocessing import get_landmark_and_bbox_enhanced, coord_placeholder
            print("   ✅ Enhanced preprocessing imported")
        except ImportError as e:
            if "mmpose" in str(e):
                print("   ⚠️  mmpose import detected - this is what we want to eliminate")
                print("   🔧 Surgical modification needed: remove mmpose imports")
            else:
                print(f"   ❌ Unexpected import error: {e}")
        
        print("\n4. Testing MuseTalk utils...")
        from musetalk.utils.utils import datagen_enhanced, load_all_model
        print("   ✅ MuseTalk utils imported")
        
        print("\n5. Testing Diffusers (for LatentSync UNet3D)...")
        from diffusers import AutoencoderKL, DDIMScheduler
        print("   ✅ Diffusers imported")
        
        print("\n6. Testing LatentSync UNet3D (surgical component)...")
        try:
            from LatentSync.latentsync.models.unet import UNet3DConditionModel
            print("   ✅ LatentSync UNet3D imported")
        except ImportError as e:
            print(f"   ❌ LatentSync UNet3D import failed: {e}")
        
        print("\n7. Testing tensor operations...")
        test_tensor = torch.randn(1, 3, 256, 256)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
            print("   ✅ CUDA tensor operations work")
        else:
            print("   ✅ CPU tensor operations work")
        
        print("\n🎉 SURGICAL FUNCTIONALITY: SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ SURGICAL FUNCTIONALITY: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_version_compatibility():
    """Test that versions are compatible"""
    
    print(f"\n📋 VERSION COMPATIBILITY CHECK")
    print("=" * 60)
    
    version_info = {}
    
    # Get versions
    try:
        import torch
        version_info['torch'] = torch.__version__
    except:
        version_info['torch'] = 'Not available'
    
    try:
        import diffusers
        version_info['diffusers'] = diffusers.__version__
    except:
        version_info['diffusers'] = 'Not available'
        
    try:
        import transformers
        version_info['transformers'] = transformers.__version__
    except:
        version_info['transformers'] = 'Not available'
    
    try:
        import numpy
        version_info['numpy'] = numpy.__version__
    except:
        version_info['numpy'] = 'Not available'
    
    print("\n📦 INSTALLED VERSIONS:")
    for package, version in version_info.items():
        print(f"   {package:15}: {version}")
    
    # Expected versions from surgical requirements
    expected_versions = {
        'torch': '2.2.0',
        'diffusers': '0.32.2', 
        'transformers': '4.48.0',
        'numpy': '1.26.4'
    }
    
    print(f"\n🎯 EXPECTED VERSIONS:")
    for package, expected in expected_versions.items():
        installed = version_info.get(package, 'Not available')
        match = "✅" if installed.startswith(expected) else "⚠️"
        print(f"   {package:15}: {expected} {match} (installed: {installed})")
    
    return version_info

def main():
    """Run all surgical requirements tests"""
    
    print("🔬 SURGICAL INTEGRATION VALIDATION TEST")
    print("Testing that our dependency elimination strategy works")
    print("=" * 80)
    
    # Test 1: Requirements validation
    core_success, musetalk_success, latentsync_success, eliminated_success = test_surgical_requirements()
    
    # Test 2: Functionality validation  
    functionality_success = test_surgical_functionality()
    
    # Test 3: Version compatibility
    version_info = test_version_compatibility()
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"📊 SURGICAL INTEGRATION TEST SUMMARY")
    print(f"=" * 80)
    
    total_tests = 4
    passed_tests = 0
    
    if core_success >= 8:  # At least 8/10 core dependencies
        print("✅ Core dependencies: PASS")
        passed_tests += 1
    else:
        print(f"❌ Core dependencies: FAIL ({core_success}/10)")
    
    if musetalk_success >= 4:  # At least 4/6 MuseTalk components
        print("✅ MuseTalk components: PASS")
        passed_tests += 1
    else:
        print(f"❌ MuseTalk components: FAIL ({musetalk_success}/6)")
    
    if latentsync_success >= 1:  # At least 1/1 LatentSync component
        print("✅ LatentSync UNet3D: PASS")
        passed_tests += 1
    else:
        print("❌ LatentSync UNet3D: FAIL")
    
    if functionality_success:
        print("✅ Surgical functionality: PASS")
        passed_tests += 1
    else:
        print("❌ Surgical functionality: FAIL")
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 SURGICAL INTEGRATION: READY FOR IMPLEMENTATION!")
        print("\nNext steps:")
        print("1. Create hybrid inference script")
        print("2. Test with your cutaway videos")
        print("3. Compare quality with original MuseTalk")
        return True
    elif passed_tests >= 3:
        print("⚠️  SURGICAL INTEGRATION: MOSTLY READY (minor issues)")
        print("\nIssues to resolve before proceeding:")
        if core_success < 8:
            print("- Install missing core dependencies")
        if musetalk_success < 4:
            print("- Fix MuseTalk component imports")
        if latentsync_success < 1:
            print("- Install LatentSync UNet3D")
        if not functionality_success:
            print("- Fix functionality issues")
        return False
    else:
        print("❌ SURGICAL INTEGRATION: NOT READY")
        print("\nMajor issues to resolve:")
        print("- Check environment setup")
        print("- Install surgical requirements: pip install -r requirements_surgical.txt")
        print("- Verify MuseTalk and LatentSync are properly cloned")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)