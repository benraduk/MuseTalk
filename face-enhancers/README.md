# Face Enhancers for MuseTalk

This folder contains all face enhancement models and utilities for improving the quality of AI-generated faces in the MuseTalk pipeline.

## 📁 File Organization

### **GPEN-BFR (Recommended)**
- `gpen_bfr_enhancer.py` - ONNX-based GPEN-BFR 256 wrapper
- `test_gpen_bfr_on_vae_faces.py` - Test script for GPEN-BFR enhancement

### **GFPGAN (Legacy)**
- `gfpgan_enhancer.py` - PyTorch-based GFPGAN wrapper (has import issues)
- `gfpgan_parameter_configs.py` - GFPGAN parameter configurations
- `GFPGAN_Parameter_Guide.md` - Guide for GFPGAN parameters
- `gfpgan-plan.md` - Original GFPGAN integration plan

## 🎯 **Recommended Approach: GPEN-BFR**

GPEN-BFR is the recommended face enhancement solution because:

### **Technical Advantages**
- ✅ **ONNX Runtime**: No PyTorch conflicts, more stable
- ✅ **Perfect Size**: 256x256 input/output (matches MuseTalk exactly)
- ✅ **Better Quality**: More natural results than GFPGAN
- ✅ **GPU Acceleration**: CUDA support through ONNX
- ✅ **Cross-Platform**: Works consistently across systems

### **Quality Benefits**
- 🎨 **Natural Enhancement**: Less over-processing than GFPGAN
- 👁️ **Identity Preservation**: Maintains original facial features
- 💋 **Lip Quality**: Specifically good for lip region enhancement
- 🖼️ **Artifact Reduction**: Fewer visual artifacts

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install onnx onnxruntime-gpu
```

### **2. Download GPEN-BFR Model**
```bash
# Windows
download_weights.bat

# Linux/macOS
./download_weights.sh
```

### **3. Test GPEN-BFR**
```bash
python face-enhancers/test_gpen_bfr_on_vae_faces.py
```

### **4. Use in Code**
```python
from face_enhancers.gpen_bfr_enhancer import GPENBFREnhancer

# Initialize enhancer
enhancer = GPENBFREnhancer()

# Enhance single face
enhanced_face = enhancer.enhance_face(face_image)

# Enhance multiple faces
enhanced_faces = enhancer.enhance_batch(face_images)
```

## 📊 **Integration into MuseTalk**

To integrate GPEN-BFR into the main MuseTalk pipeline:

### **1. Import in scripts/inference.py**
```python
from face_enhancers.gpen_bfr_enhancer import GPENBFREnhancer
```

### **2. Add Command Line Arguments**
```python
parser.add_argument("--enable_gpen_bfr", action="store_true", 
                   help="Enable GPEN-BFR face enhancement")
parser.add_argument("--gpen_bfr_model", type=str, 
                   default="models/gpen_bfr/gpen_bfr_256.onnx",
                   help="Path to GPEN-BFR model")
```

### **3. Initialize Enhancer**
```python
gpen_bfr_enhancer = None
if args.enable_gpen_bfr:
    try:
        gpen_bfr_enhancer = GPENBFREnhancer(model_path=args.gpen_bfr_model)
    except Exception as e:
        print(f"⚠️ GPEN-BFR initialization failed: {e}")
```

### **4. Apply Enhancement After VAE Decoding**
```python
# After: recon = vae.decode_latents(pred_latents)
if gpen_bfr_enhancer is not None:
    enhanced_recon = gpen_bfr_enhancer.enhance_batch(recon)
    recon = enhanced_recon
```

## 🔧 **Model Information**

### **GPEN-BFR 256**
- **Input Size**: 256x256 RGB
- **Output Size**: 256x256 RGB  
- **Model Size**: ~45MB
- **Architecture**: GAN with Prior Embedding
- **Template**: arcface_128 face alignment
- **Source**: FaceFusion project

### **Download URLs**
- **Model**: `https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.onnx`
- **Location**: `models/gpen_bfr/gpen_bfr_256.onnx`

## 🧪 **Testing**

### **Test Scripts**
- `test_gpen_bfr_on_vae_faces.py` - Comprehensive GPEN-BFR testing
- Creates before/after comparisons
- Generates quality metrics and analysis
- Produces comparison grids

### **Test Outputs**
- `test_gpen_bfr_output/enhanced_faces/` - Enhanced face images
- `test_gpen_bfr_output/comparisons/` - Side-by-side comparisons
- `test_gpen_bfr_output/analysis/` - Quality analysis reports
- `test_gpen_bfr_output/gpen_bfr_comparison_grid.png` - Overview grid

## 📈 **Performance**

### **Speed**
- **Single Face**: ~50-100ms (GPU) / ~200-500ms (CPU)
- **Batch Processing**: Efficient for multiple faces
- **Memory**: ~1-2GB GPU memory

### **Quality Metrics**
- **Sharpness**: Typically 10-30% improvement
- **Contrast**: Typically 5-15% improvement  
- **PSNR**: Usually 25-35 dB
- **Visual Quality**: Significant improvement in lip detail

## ⚠️ **Troubleshooting**

### **Common Issues**

#### **"ONNX Runtime not available"**
```bash
pip install onnxruntime-gpu
# OR for CPU only:
pip install onnxruntime
```

#### **"GPEN-BFR model not found"**
```bash
# Download the model
python download_weights.bat  # Windows
./download_weights.sh        # Linux/macOS
```

#### **"CUDA provider not available"**
- Install CUDA toolkit
- Install onnxruntime-gpu
- Falls back to CPU automatically

#### **Memory Issues**
- Reduce batch size
- Use CPU provider
- Close other GPU applications

## 🔄 **Migration from GFPGAN**

If you were using GFPGAN, migrating to GPEN-BFR is straightforward:

### **Old GFPGAN Code**
```python
from face_enhancers.gfpgan_enhancer import GFPGANEnhancer
enhancer = GFPGANEnhancer()
```

### **New GPEN-BFR Code**
```python
from face_enhancers.gpen_bfr_enhancer import GPENBFREnhancer
enhancer = GPENBFREnhancer()
```

The API is nearly identical, making migration seamless!

## 📝 **Expected ONNX Warnings**

When loading GPEN-BFR, you may see ONNX Runtime warnings about "graph inputs" and "const folding". These are **completely harmless** optimization notices from the model itself and don't affect functionality or quality. They indicate that the model could be re-exported for slightly better performance, but this is not necessary for normal use.

## 📚 **Additional Resources**

- [FaceFusion Project](https://github.com/facefusion/facefusion) - Source of GPEN-BFR models
- [ONNX Runtime Documentation](https://onnxruntime.ai/) - ONNX runtime details
- [GPEN Paper](https://arxiv.org/abs/2105.06070) - Original GPEN research

---

**Last Updated**: August 2025  
**Recommended Model**: GPEN-BFR 256  
**Status**: Production Ready ✅
