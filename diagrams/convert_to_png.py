#!/usr/bin/env python3
"""
Mermaid to PNG Converter Script
===============================
Converts all .mmd files in the diagrams folder to PNG format
"""

import os
import subprocess
import sys
from pathlib import Path

def check_mermaid_cli():
    """Check if mermaid CLI is installed"""
    try:
        result = subprocess.run(['mmdc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Mermaid CLI found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Mermaid CLI not found")
    print("📋 Installation instructions:")
    print("   1. Install Node.js: https://nodejs.org/")
    print("   2. Install mermaid CLI: npm install -g @mermaid-js/mermaid-cli")
    print("   3. Or use online converter: https://mermaid.live/")
    return False

def convert_mermaid_to_png():
    """Convert all .mmd files to PNG"""
    diagrams_dir = Path(__file__).parent
    mmd_files = list(diagrams_dir.glob("*.mmd"))
    
    if not mmd_files:
        print("❌ No .mmd files found in diagrams folder")
        return False
    
    print(f"🔍 Found {len(mmd_files)} Mermaid diagrams to convert:")
    for mmd_file in mmd_files:
        print(f"   - {mmd_file.name}")
    
    if not check_mermaid_cli():
        return False
    
    print(f"\n🔄 Converting diagrams to PNG...")
    
    success_count = 0
    for mmd_file in mmd_files:
        png_file = mmd_file.with_suffix('.png')
        
        try:
            # Convert using mermaid CLI
            result = subprocess.run([
                'mmdc', 
                '-i', str(mmd_file),
                '-o', str(png_file),
                '--theme', 'default',
                '--backgroundColor', 'white'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ {mmd_file.name} → {png_file.name}")
                success_count += 1
            else:
                print(f"   ❌ {mmd_file.name} - Error: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ {mmd_file.name} - Exception: {e}")
    
    print(f"\n📊 Conversion complete: {success_count}/{len(mmd_files)} successful")
    
    if success_count == len(mmd_files):
        print("🎉 All diagrams converted successfully!")
        return True
    else:
        print("⚠️  Some conversions failed. Check errors above.")
        return False

def show_alternative_methods():
    """Show alternative conversion methods"""
    print(f"\n📋 ALTERNATIVE CONVERSION METHODS:")
    print(f"")
    print(f"🌐 Option 1: Online Converter (Easiest)")
    print(f"   1. Go to: https://mermaid.live/")
    print(f"   2. Copy content from .mmd files")
    print(f"   3. Paste and download PNG")
    print(f"")
    print(f"🔧 Option 2: VS Code Extension")
    print(f"   1. Install 'Mermaid Markdown Syntax Highlighting'")
    print(f"   2. Open .mmd file in VS Code")  
    print(f"   3. Right-click → Export Mermaid Diagram")
    print(f"")
    print(f"📦 Option 3: Docker (No Node.js needed)")
    print(f"   docker run --rm -v $(pwd):/data minlag/mermaid-cli -i diagram.mmd -o diagram.png")

def main():
    """Main conversion function"""
    print("🎨 MERMAID TO PNG CONVERTER")
    print("=" * 50)
    
    # Check if we're in the diagrams directory
    current_dir = Path.cwd()
    if current_dir.name != 'diagrams':
        diagrams_path = current_dir / 'diagrams'
        if diagrams_path.exists():
            os.chdir(diagrams_path)
            print(f"📁 Changed to diagrams directory: {diagrams_path}")
        else:
            print("❌ Please run this script from the diagrams folder or repository root")
            return
    
    # Try to convert
    success = convert_mermaid_to_png()
    
    # Show alternatives if conversion failed
    if not success:
        show_alternative_methods()

if __name__ == "__main__":
    main()