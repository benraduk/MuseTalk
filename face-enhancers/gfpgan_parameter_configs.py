"""
GFPGAN parameter configurations for different enhancement styles
Use these configs to get different results from GFPGAN
"""

# Different GFPGAN configurations for various enhancement styles
GFPGAN_CONFIGS = {
    'natural': {
        'model_path': 'models/gfpgan/GFPGANv1.3.pth',  # Less aggressive than v1.4
        'upscale': 1,
        'arch': 'original',  # More conservative than 'clean'
        'channel_multiplier': 1,  # Lighter processing
        'bg_upsampler': None,
        'description': 'Minimal enhancement, preserves original features'
    },
    
    'conservative': {
        'model_path': 'models/gfpgan/GFPGANv1.2.pth',  # Oldest, most stable
        'upscale': 1,
        'arch': 'original',
        'channel_multiplier': 1,
        'bg_upsampler': None,
        'description': 'Very light touch, minimal artifacts'
    },
    
    'balanced': {
        'model_path': 'models/gfpgan/GFPGANv1.3.pth',
        'upscale': 1,
        'arch': 'clean',  # More processing than 'original'
        'channel_multiplier': 2,
        'bg_upsampler': None,
        'description': 'Good balance of enhancement and naturalness'
    },
    
    'quality_focused': {
        'model_path': 'models/gfpgan/GFPGANv1.4.pth',  # Latest model
        'upscale': 1,
        'arch': 'clean',
        'channel_multiplier': 2,
        'bg_upsampler': None,
        'description': 'Maximum quality, may over-enhance some faces'
    },
    
    'restoreformer': {
        'model_path': 'models/gfpgan/GFPGANv1.4.pth',
        'upscale': 1,
        'arch': 'RestoreFormer',  # Alternative architecture
        'channel_multiplier': 2,
        'bg_upsampler': None,
        'description': 'Different enhancement style, good for variety'
    }
}

def get_gfpgan_config(style='balanced'):
    """
    Get GFPGAN configuration for a specific style
    
    Args:
        style: One of 'natural', 'conservative', 'balanced', 'quality_focused', 'restoreformer'
    
    Returns:
        Dictionary with GFPGAN parameters
    """
    if style not in GFPGAN_CONFIGS:
        print(f"⚠️ Unknown style '{style}'. Available: {list(GFPGAN_CONFIGS.keys())}")
        style = 'balanced'
    
    return GFPGAN_CONFIGS[style].copy()

def print_all_configs():
    """
    Print all available GFPGAN configurations
    """
    print("🎨 Available GFPGAN Configurations:")
    print("=" * 50)
    
    for style, config in GFPGAN_CONFIGS.items():
        print(f"\n📋 {style.upper()}:")
        print(f"   Model: {config['model_path'].split('/')[-1]}")
        print(f"   Architecture: {config['arch']}")
        print(f"   Channel Multiplier: {config['channel_multiplier']}")
        print(f"   Description: {config['description']}")

# Integration example for MuseTalk
def create_enhanced_gfpgan_wrapper():
    """
    Example of how to use these configs in the MuseTalk pipeline
    """
    example_code = '''
# In scripts/inference.py, after VAE decoding:

from musetalk.utils.gfpgan_enhancer import GFPGANEnhancer
from gfpgan_parameter_configs import get_gfpgan_config

# Choose your preferred style
config = get_gfpgan_config('balanced')  # or 'natural', 'conservative', etc.

# Initialize GFPGAN with chosen config
gfpgan_enhancer = GFPGANEnhancer(**config)

# Apply enhancement
recon = vae.decode_latents(pred_latents)
if gfpgan_enhancer is not None:
    enhanced_recon = []
    for face_img in recon:
        enhanced_face = gfpgan_enhancer.enhance_face(face_img)
        enhanced_recon.append(enhanced_face)
    recon = enhanced_recon
'''
    
    print("🔧 Integration Example:")
    print("=" * 25)
    print(example_code)

if __name__ == "__main__":
    print_all_configs()
    create_enhanced_gfpgan_wrapper()
