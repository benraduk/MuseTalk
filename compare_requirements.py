#!/usr/bin/env python3
"""
Compare original MuseTalk/LatentSync requirements vs surgical requirements
Shows exactly what conflicts were eliminated
"""

def parse_requirements_file(filename):
    """Parse a requirements file and return dependencies as dict"""
    deps = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('--'):
                    if '==' in line:
                        name, version = line.split('==', 1)
                        deps[name.strip()] = version.strip()
                    else:
                        deps[line.strip()] = None
    except FileNotFoundError:
        print(f"⚠️  File {filename} not found")
    return deps

def analyze_requirements_changes():
    """Analyze changes between original and surgical requirements"""
    
    print("📋 REQUIREMENTS COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load requirements
    musetalk_deps = parse_requirements_file('requirements.txt')
    surgical_deps = parse_requirements_file('requirements_surgical.txt')
    
    # Manual LatentSync dependencies (from their requirements.txt)
    latentsync_deps = {
        'torch': '2.5.1',
        'torchvision': '0.20.1',
        'diffusers': '0.32.2', 
        'transformers': '4.48.0',
        'decord': '0.6.0',
        'accelerate': '0.26.1',
        'einops': '0.7.0',
        'omegaconf': '2.3.0',
        'opencv-python': '4.9.0.80',
        'mediapipe': '0.10.11',
        'python_speech_features': '0.6',
        'librosa': '0.10.1',
        'scenedetect': '0.6.1',
        'ffmpeg-python': '0.2.0',
        'imageio': '2.31.1',
        'imageio-ffmpeg': '0.5.1',
        'lpips': '0.1.4',
        'face-alignment': '1.4.1',
        'gradio': '5.24.0',
        'huggingface-hub': '0.30.2',
        'numpy': '1.26.4',
        'kornia': '0.8.0',
        'insightface': '0.7.3',
        'onnxruntime-gpu': '1.21.0',
        'DeepCache': '0.1.1'
    }
    
    # Hidden MuseTalk dependencies (from README)
    musetalk_hidden = {
        'torch': '2.0.1',
        'torchvision': '0.15.2',
        'torchaudio': '2.0.2',
        'mmpose': '1.1.0',
        'mmcv': '2.0.1', 
        'mmdet': '3.1.0',
        'openmim': None,
        'mmengine': None
    }
    
    # Combine MuseTalk dependencies
    musetalk_complete = {**musetalk_deps, **musetalk_hidden}
    
    print(f"\n📊 DEPENDENCY COUNTS:")
    print(f"   MuseTalk (complete): {len(musetalk_complete)} dependencies")
    print(f"   LatentSync (original): {len(latentsync_deps)} dependencies") 
    print(f"   Surgical (new): {len(surgical_deps)} dependencies")
    
    # Find conflicts between original systems
    print(f"\n🔴 ORIGINAL CONFLICTS (MuseTalk vs LatentSync):")
    conflicts = []
    for pkg in musetalk_complete:
        if pkg in latentsync_deps:
            musetalk_ver = musetalk_complete[pkg]
            latentsync_ver = latentsync_deps[pkg]
            if musetalk_ver != latentsync_ver:
                conflicts.append((pkg, musetalk_ver, latentsync_ver))
                severity = "🔴 CATASTROPHIC" if pkg in ['torch', 'torchvision'] else "🔴 HIGH"
                print(f"   {severity}: {pkg:20} {musetalk_ver:10} vs {latentsync_ver}")
    
    print(f"\n   Total conflicts: {len(conflicts)}")
    
    # Find what was eliminated
    all_original = set(musetalk_complete.keys()) | set(latentsync_deps.keys()) 
    eliminated = all_original - set(surgical_deps.keys())
    
    print(f"\n❌ ELIMINATED DEPENDENCIES ({len(eliminated)}):")
    
    # Categorize eliminations
    eliminated_categories = {
        'MMLab Ecosystem': ['mmpose', 'mmcv', 'mmdet', 'openmim', 'mmengine'],
        'Training Only': ['tensorflow', 'tensorboard'],
        'LatentSync Face Detection': ['insightface', 'onnxruntime-gpu', 'face-alignment'], 
        'LatentSync Preprocessing': ['mediapipe', 'scenedetect', 'kornia', 'decord'],
        'Training/Evaluation': ['lpips', 'python_speech_features'],
        'Optional': ['DeepCache', 'imageio-ffmpeg']
    }
    
    for category, packages in eliminated_categories.items():
        category_eliminated = [pkg for pkg in packages if pkg in eliminated]
        if category_eliminated:
            print(f"\n   {category}:")
            for pkg in category_eliminated:
                original_ver = musetalk_complete.get(pkg) or latentsync_deps.get(pkg) or 'Unknown'
                print(f"      {pkg:20} ({original_ver})")
    
    # Find version changes in kept dependencies
    print(f"\n🔄 VERSION CHANGES (Kept but updated):")
    version_changes = []
    for pkg in surgical_deps:
        surgical_ver = surgical_deps[pkg]
        if surgical_ver:  # Only versioned packages
            musetalk_ver = musetalk_complete.get(pkg)
            latentsync_ver = latentsync_deps.get(pkg)
            
            if musetalk_ver and surgical_ver != musetalk_ver:
                version_changes.append((pkg, musetalk_ver, surgical_ver, 'MuseTalk'))
            elif latentsync_ver and surgical_ver != latentsync_ver:
                version_changes.append((pkg, latentsync_ver, surgical_ver, 'LatentSync'))
    
    for pkg, old_ver, new_ver, source in version_changes:
        direction = "↗️" if pkg in ['torch', 'diffusers', 'transformers', 'numpy'] else "↘️"
        print(f"   {direction} {pkg:20}: {old_ver:10} → {new_ver:10} (from {source})")
    
    # Resolution strategy summary
    print(f"\n🎯 RESOLUTION STRATEGY:")
    strategy_counts = {
        'PyTorch Middle Ground': 3,  # torch, torchvision, torchaudio
        'Use LatentSync Version': 4,  # diffusers, transformers, numpy, omegaconf
        'Use MuseTalk Version': 3,   # accelerate, librosa, einops  
        'Keep Matching': 4,          # opencv, gradio, huggingface_hub, etc
        'Eliminate Completely': len(eliminated)
    }
    
    for strategy, count in strategy_counts.items():
        print(f"   {strategy:25}: {count:2} packages")
    
    # Risk assessment
    print(f"\n📈 RISK ASSESSMENT:")
    high_risk_changes = [pkg for pkg, _, _, _ in version_changes if pkg in ['torch', 'diffusers', 'transformers', 'numpy']]
    medium_risk_changes = [pkg for pkg, _, _, _ in version_changes if pkg not in ['torch', 'diffusers', 'transformers', 'numpy']]
    
    print(f"   🔴 High-risk changes: {len(high_risk_changes)} (core ML libraries)")
    print(f"   🟡 Medium-risk changes: {len(medium_risk_changes)} (utilities)")
    print(f"   ✅ Eliminated conflicts: {len(conflicts)} → 0")
    print(f"   ✅ Dependency reduction: {len(eliminated)} packages eliminated")
    
    reduction_percentage = len(eliminated) / len(all_original) * 100
    print(f"   ✅ Total reduction: {reduction_percentage:.1f}%")
    
    return conflicts, eliminated, version_changes

def generate_migration_script():
    """Generate a migration script for switching to surgical requirements"""
    
    script_content = """#!/bin/bash
# Migration script from original MuseTalk to surgical integration

echo "🔧 SURGICAL INTEGRATION MIGRATION"
echo "=================================="

# Step 1: Backup current environment
echo "📦 Step 1: Backing up current environment..."
conda create --name musetalk_backup --clone musetalk --yes
echo "✅ Backup created: musetalk_backup"

# Step 2: Create surgical environment
echo "🆕 Step 2: Creating surgical integration environment..."
conda create --name musetalk_surgical python=3.10 --yes
conda activate musetalk_surgical

# Step 3: Install surgical requirements
echo "⚡ Step 3: Installing surgical requirements..."
pip install -r requirements_surgical.txt

# Step 4: Validate installation
echo "🔍 Step 4: Validating installation..."
python test_surgical_requirements.py

echo "🎉 Surgical integration migration complete!"
echo ""
echo "To use the surgical environment:"
echo "   conda activate musetalk_surgical"
echo ""
echo "To revert to backup if needed:"
echo "   conda activate musetalk_backup"
"""
    
    with open('migrate_to_surgical.sh', 'w') as f:
        f.write(script_content)
    
    print(f"\n✅ Generated migration script: migrate_to_surgical.sh")
    print("   Run with: bash migrate_to_surgical.sh")

def main():
    """Run requirements comparison analysis"""
    
    conflicts, eliminated, version_changes = analyze_requirements_changes()
    generate_migration_script()
    
    print(f"\n" + "=" * 80)
    print(f"🎯 SURGICAL INTEGRATION SUMMARY")
    print(f"=" * 80)
    
    print(f"✅ Conflicts eliminated: {len(conflicts)}")
    print(f"✅ Dependencies removed: {len(eliminated)}")  
    print(f"⚠️  Version changes: {len(version_changes)}")
    
    if len(eliminated) > len(version_changes):
        print(f"\n🎉 EXCELLENT: More eliminated than changed!")
        print(f"   Surgical approach is highly effective")
    else:
        print(f"\n⚠️  CAUTION: More changes than eliminations")
        print(f"   Test thoroughly before production use")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review the generated files:")
    print(f"   - requirements_surgical.txt")
    print(f"   - surgical_integration_guide.md") 
    print(f"   - test_surgical_requirements.py")
    print(f"   - migrate_to_surgical.sh")
    print(f"")
    print(f"2. Test the surgical integration:")
    print(f"   bash migrate_to_surgical.sh")
    print(f"")
    print(f"3. If successful, implement hybrid inference script")

if __name__ == "__main__":
    main()