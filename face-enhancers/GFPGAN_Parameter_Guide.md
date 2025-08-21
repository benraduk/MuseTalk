# 🎨 GFPGAN Parameter Guide

## 📋 Key Parameters You Can Adjust

### 1. **Model Version** (`model_path`)
- **GFPGANv1.4.pth**: Latest, most aggressive enhancement
- **GFPGANv1.3.pth**: Balanced, often more stable
- **GFPGANv1.2.pth**: Conservative, minimal artifacts

### 2. **Architecture** (`arch`)
- **'clean'**: More aggressive face restoration (default)
- **'original'**: Less aggressive, preserves more original features
- **'RestoreFormer'**: Alternative architecture, different style

### 3. **Channel Multiplier** (`channel_multiplier`)
- **2**: Default, full model capacity
- **1**: Reduced model complexity, lighter processing
- **4**: Maximum capacity (if supported), heaviest processing

### 4. **Upscale Factor** (`upscale`)
- **1**: No upscaling, preserve original size (recommended for MuseTalk)
- **2**: 2x upscaling (512x512 output)
- **4**: 4x upscaling (1024x1024 output)

### 5. **Background Upsampler** (`bg_upsampler`)
- **None**: No background processing (recommended)
- **'realesrgan'**: Enhance background too (may cause artifacts)

## 🎯 Recommended Combinations

### For **Natural Look** (Minimal Changes):
```python
GFPGANEnhancer(
    model_path="models/gfpgan/GFPGANv1.3.pth",
    upscale=1,
    arch='original',
    channel_multiplier=1,
    bg_upsampler=None
)
```

### For **Balanced Enhancement**:
```python
GFPGANEnhancer(
    model_path="models/gfpgan/GFPGANv1.3.pth",
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)
```

### For **Maximum Quality** (may over-enhance):
```python
GFPGANEnhancer(
    model_path="models/gfpgan/GFPGANv1.4.pth",
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)
```

## 🔧 Custom Testing

Run `python test_gfpgan_parameters.py` to test all combinations and find your preferred settings.

## 💡 Tips for Better Results

1. **Start Conservative**: Begin with GFPGANv1.3 + 'original' arch
2. **Avoid Upscaling**: Keep upscale=1 for MuseTalk compatibility
3. **Test Multiple Models**: Different faces may work better with different versions
4. **Compare Side-by-Side**: Use the comparison images to evaluate quality
5. **Consider Face Type**: Some parameters work better for certain face types

## ⚠️ Common Issues

- **Over-enhancement**: Try older model versions or 'original' architecture
- **Artifacts**: Reduce channel_multiplier or use conservative settings
- **Loss of Identity**: Use GFPGANv1.2 or minimal processing settings
- **Unnatural Skin**: Try 'original' architecture instead of 'clean'
