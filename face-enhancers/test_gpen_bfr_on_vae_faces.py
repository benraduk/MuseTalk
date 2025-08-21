"""
GPEN-BFR Test Script for MuseTalk VAE Faces
===========================================

This script tests GPEN-BFR face enhancement on existing VAE-decoded faces
from MuseTalk. It creates before/after comparisons and generates a comprehensive
analysis of the enhancement quality.

Usage:
    python face-enhancers/test_gpen_bfr_on_vae_faces.py

Requirements:
    - Existing VAE faces in test_gfpgan_output/vae_faces/ (from previous tests)
    - GPEN-BFR model downloaded to models/gpen_bfr/gpen_bfr_256.onnx
    - ONNX runtime installed (onnxruntime-gpu recommended)
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os
from typing import List, Tuple

# Add the project root to Python path
sys.path.append('.')
sys.path.append('..')

try:
    # Add face-enhancers to path
    from pathlib import Path
    face_enhancers_path = Path(__file__).parent
    if str(face_enhancers_path) not in sys.path:
        sys.path.insert(0, str(face_enhancers_path))
    
    from gpen_bfr_enhancer import GPENBFREnhancer
except ImportError as e:
    print(f"❌ Failed to import GPEN-BFR enhancer: {e}")
    print("💡 Make sure you're running from the MuseTalk root directory")
    sys.exit(1)


def extract_vae_faces_if_needed():
    """Extract VAE faces if they don't exist yet"""
    vae_faces_dir = Path("test_gfpgan_output/vae_faces")
    
    if vae_faces_dir.exists() and list(vae_faces_dir.glob("*.png")):
        print(f"✅ Found existing VAE faces in {vae_faces_dir}")
        return True
    
    print("❌ No VAE faces found. We need to extract them first.")
    print("💡 This script will create some sample faces for testing...")
    
    # Create sample faces directory
    vae_faces_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some sample 256x256 faces for testing
    sample_faces = []
    
    # Sample face 1: Gradient pattern
    face1 = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        face1[i, :, 0] = i  # Red gradient
        face1[:, i, 1] = i  # Green gradient
    face1[:, :, 2] = 128  # Blue constant
    sample_faces.append(("sample_face_01.png", face1))
    
    # Sample face 2: Checkerboard pattern
    face2 = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i//32 + j//32) % 2 == 0:
                face2[i:i+32, j:j+32] = [200, 150, 100]
            else:
                face2[i:i+32, j:j+32] = [100, 150, 200]
    sample_faces.append(("sample_face_02.png", face2))
    
    # Sample face 3: Circular pattern
    face3 = np.zeros((256, 256, 3), dtype=np.uint8)
    center = (128, 128)
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            face3[i, j] = [int(dist) % 256, int(dist * 1.5) % 256, int(dist * 2) % 256]
    sample_faces.append(("sample_face_03.png", face3))
    
    # Save sample faces
    for filename, face in sample_faces:
        cv2.imwrite(str(vae_faces_dir / filename), face)
        print(f"📸 Created sample face: {filename}")
    
    print(f"✅ Created {len(sample_faces)} sample faces for testing")
    return True


def test_gpen_bfr_enhancement():
    """Test GPEN-BFR on existing VAE faces"""
    print("🎨 GPEN-BFR Enhancement Test")
    print("=" * 50)
    
    # Ensure we have VAE faces to test on
    if not extract_vae_faces_if_needed():
        return False
    
    # Find VAE faces
    vae_faces_dir = Path("test_gfpgan_output/vae_faces")
    vae_files = sorted(list(vae_faces_dir.glob("*.png")))
    
    if not vae_files:
        print("❌ No VAE face files found.")
        return False
    
    print(f"📸 Found {len(vae_files)} VAE faces to enhance")
    
    # Create output directories
    output_dir = Path("test_gpen_bfr_output")
    output_dir.mkdir(exist_ok=True)
    
    enhanced_dir = output_dir / "enhanced_faces"
    enhanced_dir.mkdir(exist_ok=True)
    
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize GPEN-BFR enhancer
        print("\n🔧 Initializing GPEN-BFR enhancer...")
        enhancer = GPENBFREnhancer()
        
        # Process each face
        enhancement_results = []
        
        for i, vae_file in enumerate(vae_files):
            print(f"\n🎨 Processing {vae_file.name}...")
            
            # Load original face
            original = cv2.imread(str(vae_file))
            if original is None:
                print(f"⚠️ Could not load {vae_file}")
                continue
            
            print(f"   Original size: {original.shape}")
            
            # Enhance with GPEN-BFR
            enhanced = enhancer.enhance_face(original)
            print(f"   Enhanced size: {enhanced.shape}")
            
            # Save enhanced face
            enhanced_path = enhanced_dir / f"gpen_bfr_{vae_file.name}"
            cv2.imwrite(str(enhanced_path), enhanced)
            
            # Create side-by-side comparison
            comparison = create_comparison_image(original, enhanced, vae_file.name)
            comparison_path = comparison_dir / f"comparison_gpen_bfr_{vae_file.name}"
            cv2.imwrite(str(comparison_path), comparison)
            
            # Calculate quality metrics
            metrics = calculate_quality_metrics(original, enhanced)
            enhancement_results.append({
                'filename': vae_file.name,
                'original_path': str(vae_file),
                'enhanced_path': str(enhanced_path),
                'comparison_path': str(comparison_path),
                'metrics': metrics
            })
            
            print(f"✅ Enhanced: {enhanced_path.name}")
            print(f"📊 Comparison: {comparison_path.name}")
            print(f"📈 Quality improvement: {metrics['improvement_score']:.2f}")
        
        # Generate comprehensive analysis
        create_analysis_report(enhancement_results, analysis_dir)
        
        # Create comparison grid
        create_comparison_grid(enhancement_results, output_dir)
        
        print(f"\n🎉 GPEN-BFR enhancement completed!")
        print(f"📁 Results saved in: {output_dir}")
        print(f"📊 Analysis report: {analysis_dir / 'enhancement_analysis.txt'}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPEN-BFR test failed: {e}")
        print("💡 Make sure ONNX runtime is installed: pip install onnxruntime-gpu")
        print("💡 Make sure GPEN-BFR model is downloaded: run download_weights.bat")
        return False


def create_comparison_image(original: np.ndarray, enhanced: np.ndarray, filename: str) -> np.ndarray:
    """Create a side-by-side comparison image with labels"""
    # Ensure both images are the same size
    if original.shape != enhanced.shape:
        original = cv2.resize(original, (enhanced.shape[1], enhanced.shape[0]))
    
    # Create side-by-side comparison
    comparison = np.hstack([original, enhanced])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Original label (green)
    cv2.putText(comparison, "Original VAE", (10, 30), font, font_scale, (0, 255, 0), thickness)
    
    # Enhanced label (red)
    cv2.putText(comparison, "GPEN-BFR Enhanced", (266, 30), font, font_scale, (0, 0, 255), thickness)
    
    # Filename at bottom
    cv2.putText(comparison, filename, (10, comparison.shape[0] - 10), font, 0.5, (255, 255, 255), 1)
    
    return comparison


def calculate_quality_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """Calculate quality metrics for enhancement comparison"""
    # Convert to grayscale for some metrics
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness (Laplacian variance)
    orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
    enh_sharpness = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
    
    # Calculate contrast (standard deviation)
    orig_contrast = orig_gray.std()
    enh_contrast = enh_gray.std()
    
    # Calculate brightness (mean)
    orig_brightness = orig_gray.mean()
    enh_brightness = enh_gray.mean()
    
    # Calculate MSE and PSNR
    mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Simple improvement score (higher is better)
    sharpness_improvement = (enh_sharpness - orig_sharpness) / orig_sharpness if orig_sharpness > 0 else 0
    contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast if orig_contrast > 0 else 0
    improvement_score = (sharpness_improvement + contrast_improvement) / 2
    
    return {
        'original_sharpness': orig_sharpness,
        'enhanced_sharpness': enh_sharpness,
        'sharpness_improvement': sharpness_improvement,
        'original_contrast': orig_contrast,
        'enhanced_contrast': enh_contrast,
        'contrast_improvement': contrast_improvement,
        'original_brightness': orig_brightness,
        'enhanced_brightness': enh_brightness,
        'mse': mse,
        'psnr': psnr,
        'improvement_score': improvement_score
    }


def create_analysis_report(results: List[dict], analysis_dir: Path):
    """Create a comprehensive analysis report"""
    report_path = analysis_dir / "enhancement_analysis.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GPEN-BFR Enhancement Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary statistics
        if results:
            avg_sharpness_improvement = np.mean([r['metrics']['sharpness_improvement'] for r in results])
            avg_contrast_improvement = np.mean([r['metrics']['contrast_improvement'] for r in results])
            avg_improvement_score = np.mean([r['metrics']['improvement_score'] for r in results])
            avg_psnr = np.mean([r['metrics']['psnr'] for r in results if r['metrics']['psnr'] != float('inf')])
            
            f.write(f"Summary Statistics:\n")
            f.write(f"  Total faces processed: {len(results)}\n")
            f.write(f"  Average sharpness improvement: {avg_sharpness_improvement:.3f}\n")
            f.write(f"  Average contrast improvement: {avg_contrast_improvement:.3f}\n")
            f.write(f"  Average improvement score: {avg_improvement_score:.3f}\n")
            f.write(f"  Average PSNR: {avg_psnr:.2f} dB\n\n")
        
        # Individual results
        f.write("Individual Results:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            f.write(f"\nFile: {result['filename']}\n")
            metrics = result['metrics']
            f.write(f"  Sharpness: {metrics['original_sharpness']:.2f} → {metrics['enhanced_sharpness']:.2f} ({metrics['sharpness_improvement']:+.3f})\n")
            f.write(f"  Contrast: {metrics['original_contrast']:.2f} → {metrics['enhanced_contrast']:.2f} ({metrics['contrast_improvement']:+.3f})\n")
            f.write(f"  Brightness: {metrics['original_brightness']:.2f} → {metrics['enhanced_brightness']:.2f}\n")
            f.write(f"  PSNR: {metrics['psnr']:.2f} dB\n")
            f.write(f"  Improvement Score: {metrics['improvement_score']:.3f}\n")
        
        # Recommendations
        f.write(f"\n\nRecommendations:\n")
        f.write("-" * 20 + "\n")
        
        if results:
            if avg_improvement_score > 0.1:
                f.write("✅ GPEN-BFR shows significant improvement over original VAE faces\n")
                f.write("✅ Recommended for integration into MuseTalk pipeline\n")
            elif avg_improvement_score > 0.05:
                f.write("⚠️ GPEN-BFR shows moderate improvement\n")
                f.write("⚠️ Consider testing with different parameters or models\n")
            else:
                f.write("❌ GPEN-BFR shows minimal improvement\n")
                f.write("❌ May not be worth the computational cost\n")
        
        f.write(f"\nGenerated on: {Path().cwd()}\n")
    
    print(f"📊 Analysis report saved: {report_path}")


def create_comparison_grid(results: List[dict], output_dir: Path):
    """Create a grid comparison of all enhancements"""
    if not results:
        return
    
    # Load first 3 results for grid (or all if less than 3)
    grid_results = results[:3]
    grid_images = []
    
    for result in grid_results:
        # Load original and enhanced
        original = cv2.imread(result['original_path'])
        enhanced = cv2.imread(result['enhanced_path'])
        
        if original is not None and enhanced is not None:
            # Create side-by-side for this face
            pair = np.hstack([original, enhanced])
            
            # Add filename label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(pair, result['filename'], (10, pair.shape[0] - 10), 
                       font, 0.5, (255, 255, 255), 1)
            
            grid_images.append(pair)
    
    if grid_images:
        # Stack vertically
        grid = np.vstack(grid_images)
        
        # Add title
        title_height = 60
        title_img = np.zeros((title_height, grid.shape[1], 3), dtype=np.uint8)
        
        cv2.putText(title_img, "GPEN-BFR Enhancement Results", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(title_img, "Original (Left) vs Enhanced (Right)", 
                   (20, title_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Combine title and grid
        final_grid = np.vstack([title_img, grid])
        
        grid_path = output_dir / "gpen_bfr_comparison_grid.png"
        cv2.imwrite(str(grid_path), final_grid)
        print(f"📊 Comparison grid saved: {grid_path}")


def main():
    """Main function"""
    print("🧪 GPEN-BFR Test Script for MuseTalk")
    print("=" * 50)
    
    # Test GPEN-BFR availability first
    try:
        # Add face-enhancers to path
        from pathlib import Path
        face_enhancers_path = Path(__file__).parent
        if str(face_enhancers_path) not in sys.path:
            sys.path.insert(0, str(face_enhancers_path))
        
        from gpen_bfr_enhancer import test_gpen_bfr_availability
        if not test_gpen_bfr_availability():
            print("❌ GPEN-BFR is not available. Please check installation.")
            return False
    except Exception as e:
        print(f"❌ Failed to test GPEN-BFR availability: {e}")
        return False
    
    # Run the enhancement test
    success = test_gpen_bfr_enhancement()
    
    if success:
        print("\n🎉 GPEN-BFR test completed successfully!")
        print("📁 Check the test_gpen_bfr_output/ directory for results")
        print("💡 If results look good, you can integrate GPEN-BFR into MuseTalk")
    else:
        print("\n❌ GPEN-BFR test failed")
        print("💡 Check the error messages above for troubleshooting")
    
    return success


if __name__ == "__main__":
    main()
