"""
GPEN-BFR Parameter Testing Script
================================

This script tests different GPEN-BFR parameter configurations on VAE faces
to help you find the optimal settings for your specific use case.

Usage:
    python face-enhancers/test_gpen_bfr_parameters.py

The script will:
1. Load existing VAE faces
2. Test multiple parameter configurations
3. Create comparison grids showing the differences
4. Generate a detailed analysis report
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os
from typing import List, Dict, Any

# Add face-enhancers to path
face_enhancers_path = Path(__file__).parent
if str(face_enhancers_path) not in sys.path:
    sys.path.insert(0, str(face_enhancers_path))

try:
    from gpen_bfr_enhancer import GPENBFREnhancer
    from gpen_bfr_parameter_configs import GPEN_BFR_CONFIGS, list_available_configs
except ImportError as e:
    print(f"❌ Failed to import GPEN-BFR modules: {e}")
    sys.exit(1)


def test_parameter_configurations():
    """Test different parameter configurations on VAE faces"""
    print("🎨 GPEN-BFR Parameter Testing")
    print("=" * 60)
    
    # Check for existing VAE faces
    vae_faces_dir = Path("test_gfpgan_output/vae_faces")
    if not vae_faces_dir.exists():
        print("❌ No VAE faces found. Run the main GPEN-BFR test first.")
        return False
    
    vae_files = sorted(list(vae_faces_dir.glob("*.png")))
    if not vae_files:
        print("❌ No VAE face files found.")
        return False
    
    print(f"📸 Found {len(vae_files)} VAE faces to test")
    
    # Create output directories
    output_dir = Path("test_gpen_bfr_parameters_output")
    output_dir.mkdir(exist_ok=True)
    
    configs_dir = output_dir / "configurations"
    configs_dir.mkdir(exist_ok=True)
    
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)
    
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Test configurations to compare
    test_configs = ['NATURAL', 'BALANCED', 'QUALITY_FOCUSED', 'CONSERVATIVE', 'DRAMATIC', 'SKIN_FOCUS', 'LIPS_OPTIMIZED']
    
    print(f"\n🧪 Testing {len(test_configs)} configurations:")
    for config in test_configs:
        print(f"   • {config}: {GPEN_BFR_CONFIGS[config]['description']}")
    
    # Process each VAE face with each configuration
    results = {}
    
    for i, vae_file in enumerate(vae_files[:2]):  # Test on first 2 faces to save time
        print(f"\n📸 Processing {vae_file.name} ({i+1}/{min(2, len(vae_files))})...")
        
        # Load original face
        original = cv2.imread(str(vae_file))
        if original is None:
            continue
        
        face_results = {'original': original, 'enhanced': {}, 'metrics': {}}
        
        # Test each configuration
        for config_name in test_configs:
            print(f"   🎨 Testing {config_name}...")
            
            try:
                # Initialize enhancer with specific config
                enhancer = GPENBFREnhancer(config_name=config_name)
                
                # Enhance face
                enhanced = enhancer.enhance_face(original)
                
                # Save enhanced face
                config_filename = f"{config_name.lower()}_{vae_file.name}"
                config_path = configs_dir / config_filename
                cv2.imwrite(str(config_path), enhanced)
                
                # Store results
                face_results['enhanced'][config_name] = enhanced
                
                # Calculate quality metrics
                metrics = calculate_enhancement_metrics(original, enhanced)
                face_results['metrics'][config_name] = metrics
                
                print(f"      ✅ Quality score: {metrics['improvement_score']:.2f}")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                continue
        
        results[vae_file.stem] = face_results
    
    # Create comparison grids
    create_configuration_comparisons(results, comparisons_dir)
    
    # Generate analysis report
    create_parameter_analysis_report(results, analysis_dir)
    
    print(f"\n🎉 Parameter testing completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Check comparison grids in: {comparisons_dir}")
    print(f"📈 Analysis report: {analysis_dir / 'parameter_analysis.txt'}")
    
    return True


def calculate_enhancement_metrics(original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
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
    
    # Calculate SSIM (simplified version)
    ssim = calculate_ssim(orig_gray, enh_gray)
    
    # Simple improvement score
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
        'ssim': ssim,
        'improvement_score': improvement_score
    }


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate simplified SSIM"""
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    return ssim


def create_configuration_comparisons(results: Dict, comparisons_dir: Path):
    """Create comparison grids showing different configurations"""
    
    for face_name, face_data in results.items():
        print(f"📊 Creating comparison grid for {face_name}...")
        
        original = face_data['original']
        enhanced_faces = face_data['enhanced']
        
        if not enhanced_faces:
            continue
        
        # Create a grid showing original + all configurations
        configs = list(enhanced_faces.keys())
        grid_cols = min(4, len(configs) + 1)  # +1 for original
        grid_rows = (len(configs) + 1 + grid_cols - 1) // grid_cols
        
        # Calculate grid size
        face_size = 256
        grid_width = grid_cols * face_size
        grid_height = grid_rows * face_size + 40 * grid_rows  # Extra space for labels
        
        # Create grid image
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Add original image
        y_offset = 0
        x_offset = 0
        
        # Place original
        grid[y_offset:y_offset+face_size, x_offset:x_offset+face_size] = original
        cv2.putText(grid, "ORIGINAL", (x_offset + 5, y_offset + face_size + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Place enhanced versions
        for i, (config_name, enhanced_face) in enumerate(enhanced_faces.items()):
            col = (i + 1) % grid_cols
            row = (i + 1) // grid_cols
            
            x_offset = col * face_size
            y_offset = row * (face_size + 40)
            
            if y_offset + face_size <= grid_height and x_offset + face_size <= grid_width:
                grid[y_offset:y_offset+face_size, x_offset:x_offset+face_size] = enhanced_face
                
                # Add label
                cv2.putText(grid, config_name, (x_offset + 5, y_offset + face_size + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Add quality score
                score = face_data['metrics'][config_name]['improvement_score']
                cv2.putText(grid, f"Score: {score:.2f}", (x_offset + 5, y_offset + face_size + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Save grid
        grid_path = comparisons_dir / f"config_comparison_{face_name}.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"   ✅ Saved: {grid_path.name}")


def create_parameter_analysis_report(results: Dict, analysis_dir: Path):
    """Create a detailed analysis report of parameter testing"""
    report_path = analysis_dir / "parameter_analysis.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GPEN-BFR Parameter Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        all_scores = []
        config_scores = {}
        
        for face_name, face_data in results.items():
            for config_name, metrics in face_data['metrics'].items():
                score = metrics['improvement_score']
                all_scores.append(score)
                
                if config_name not in config_scores:
                    config_scores[config_name] = []
                config_scores[config_name].append(score)
        
        # Configuration rankings
        f.write("📊 Configuration Rankings (by average improvement score):\n")
        f.write("-" * 55 + "\n")
        
        config_averages = {config: np.mean(scores) for config, scores in config_scores.items()}
        sorted_configs = sorted(config_averages.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (config_name, avg_score) in enumerate(sorted_configs, 1):
            config_desc = GPEN_BFR_CONFIGS[config_name]['description']
            f.write(f"{rank}. {config_name}: {avg_score:.3f} - {config_desc}\n")
        
        f.write(f"\n📈 Performance Analysis:\n")
        f.write("-" * 25 + "\n")
        
        best_config = sorted_configs[0][0]
        worst_config = sorted_configs[-1][0]
        
        f.write(f"🥇 Best performing: {best_config} (avg: {sorted_configs[0][1]:.3f})\n")
        f.write(f"🥉 Lowest performing: {worst_config} (avg: {sorted_configs[-1][1]:.3f})\n")
        f.write(f"📊 Score range: {min(all_scores):.3f} to {max(all_scores):.3f}\n")
        f.write(f"📊 Average improvement: {np.mean(all_scores):.3f}\n")
        
        # Detailed metrics for each configuration
        f.write(f"\n📋 Detailed Configuration Analysis:\n")
        f.write("-" * 40 + "\n")
        
        for config_name in sorted([c for c in config_scores.keys()]):
            f.write(f"\n{config_name}:\n")
            config_info = GPEN_BFR_CONFIGS[config_name]
            f.write(f"  Description: {config_info['description']}\n")
            f.write(f"  Enhancement Strength: {config_info['enhancement_strength']}\n")
            f.write(f"  Contrast Boost: {config_info['contrast_boost']}\n")
            f.write(f"  Sharpening: {config_info['sharpening']}\n")
            f.write(f"  Color Correction: {config_info['color_correction']}\n")
            f.write(f"  Average Score: {config_averages[config_name]:.3f}\n")
        
        # Recommendations
        f.write(f"\n💡 Recommendations:\n")
        f.write("-" * 20 + "\n")
        
        if config_averages[best_config] > 2.0:
            f.write(f"✅ {best_config} provides excellent enhancement quality\n")
        elif config_averages[best_config] > 1.0:
            f.write(f"✅ {best_config} provides good enhancement quality\n")
        else:
            f.write(f"⚠️ All configurations show modest improvement\n")
        
        f.write(f"🎯 For MuseTalk integration, consider using: {best_config}\n")
        
        if 'LIPS_OPTIMIZED' in config_averages:
            lips_score = config_averages['LIPS_OPTIMIZED']
            f.write(f"💋 For lip-sync applications, LIPS_OPTIMIZED scored: {lips_score:.3f}\n")
    
    print(f"📊 Analysis report saved: {report_path}")


def main():
    """Main function"""
    print("🧪 GPEN-BFR Parameter Testing Script")
    print("=" * 60)
    
    # Show available configurations
    configs = list_available_configs()
    print(f"\n📋 Available Configurations ({len(configs)}):")
    for name, desc in configs.items():
        print(f"   • {name}: {desc}")
    
    # Run parameter testing
    success = test_parameter_configurations()
    
    if success:
        print("\n🎉 Parameter testing completed successfully!")
        print("💡 Check the results to find the best configuration for your use case")
    else:
        print("\n❌ Parameter testing failed")
    
    return success


if __name__ == "__main__":
    main()
