# BraivTalk - Enhanced MuseTalk with Cutaway Handling

<strong>Advanced Audio-Driven Lip-Sync with Robust Frame Processing</strong>

An enhanced version of MuseTalk that provides **robust cutaway handling** and **optimized GPU performance** for reliable lip-sync generation in real-world video scenarios.

## 🎯 **Project Goals**

BraivTalk addresses critical limitations in the original MuseTalk by focusing on:

- **🎬 Cutaway Handling**: Seamlessly processes videos with face transitions, cutaways, and non-face frames
- **🔄 Frame Continuity**: Prevents video freezing when faces disappear from frame
- **⚡ GPU Optimization**: Configurable batch processing for efficient hardware utilization
- **🛡️ Reliability**: Robust preprocessing and error handling for production use
- **🎵 Audio Sync**: Perfect audio-visual synchronization throughout entire videos

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.10+
- CUDA-capable GPU (recommended)
- FFmpeg installed
- Git LFS (for model downloads)

### **1. Clone and Setup**

```bash
git clone https://github.com/benraduk/braivtalk.git
cd braivtalk

# Setup environment (recommended)
conda create -n braivtalk python=3.10
conda activate braivtalk

# Install core dependencies
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.32.2 transformers==4.48.0 numpy==1.26.4
pip install librosa soundfile opencv-python gradio huggingface_hub
pip install omegaconf tqdm yacs av accelerate

# Install CUDA runtime and cuDNN for FaceFusion-style GPU acceleration
conda install nvidia/label/cuda-12.9.1::cuda-runtime nvidia/label/cudnn-9.10.0::cudnn

# Install TensorRT for optimal ONNX GPU performance
pip install tensorrt==10.12.0.36 --extra-index-url https://pypi.nvidia.com

# Install remaining dependencies (includes onnxruntime-gpu)
pip install -r requirements.txt
```

### **2. Download Models**

```bash
# Windows
./download_weights.bat

# Linux/Mac
chmod +x download_weights.sh
./download_weights.sh
```

### **3. Quick Test**

```bash
python -m scripts.inference \
  --inference_config configs/inference/test.yaml \
  --result_dir results/test \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json \
  --version v15 \
  --ffmpeg_path ffmpeg-master-latest-win64-gpl-shared/bin
```

### **4. Gradio Web Interface**

```bash
python app.py
```

Access the web interface at `http://localhost:7860`

## 🔧 **Key Enhancements**

### **Enhanced Frame Processing**
- **Cutaway Detection**: Automatically identifies frames without faces
- **Frame Passthrough**: Original frames are preserved for non-face segments
- **Continuous Processing**: Video never freezes or gets stuck on missing faces
- **Smart Batching**: Processes face frames efficiently while bypassing others

### **Optimized Performance**
- **Configurable Batch Sizes**: Adjustable for different GPU capabilities
- **Memory Management**: Conservative settings prevent system crashes
- **GPU Acceleration**: Maximizes CUDA utilization where possible
- **FP16 Support**: Optional mixed precision for memory efficiency

### **Robust Pipeline**
- **Error Handling**: Graceful fallbacks for edge cases
- **Path Normalization**: Cross-platform file handling
- **Dependency Management**: Optional components for flexibility
- **Audio Synchronization**: Reliable FFmpeg integration

## 📁 **Project Structure**

```
braivtalk/
├── scripts/
│   ├── inference.py              # Main inference with cutaway handling
│   ├── preprocess.py             # Enhanced preprocessing
│   └── realtime_inference.py     # Real-time processing
├── musetalk/
│   ├── models/                   # Core MuseTalk models (VAE, UNet)
│   ├── utils/
│   │   ├── preprocessing.py      # Enhanced face detection & landmarks
│   │   ├── utils.py              # Data generation with passthrough support
│   │   └── blending.py           # Image composition
│   └── whisper/                  # Audio feature extraction
├── configs/
│   └── inference/                # Configuration files
├── data/
│   ├── video/                    # Input videos
│   └── audio/                    # Input audio files
├── models/                       # Downloaded model weights
└── results/                      # Generated outputs
```

## 🛠️ **For Developers**

### **Key File Modifications**

| File | Enhancement | Purpose |
|------|-------------|---------|
| `musetalk/utils/preprocessing.py` | Face detection optimization | Handles cutaway frames gracefully |
| `musetalk/utils/utils.py` | Enhanced data generation | Supports both processing and passthrough |
| `scripts/inference.py` | Cutaway handling logic | Main inference with frame continuity |
| `app.py` | Web interface updates | Gradio interface with enhanced pipeline |

### **Development Setup**

```bash
# Clone repository
git clone https://github.com/benraduk/braivtalk.git
cd braivtalk

# Create development environment
conda create -n braivtalk-dev python=3.10
conda activate braivtalk-dev

# Install dependencies (see Quick Start section)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
conda install nvidia/label/cuda-12.9.1::cuda-runtime nvidia/label/cudnn-9.10.0::cudnn
pip install tensorrt==10.12.0.36 --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt

# Download models
./download_weights.bat  # or .sh for Linux/Mac
```

### **Testing**

```bash
# Quick dependency check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__, 'Providers:', onnxruntime.get_available_providers())"

# Test inference pipeline
python -m scripts.inference --inference_config configs/inference/test.yaml --result_dir results/dev_test --unet_model_path models/musetalkV15/unet.pth --unet_config models/musetalkV15/musetalk.json --version v15
```

## ⚙️ **Performance Tuning**

### **Batch Size Configuration**

Adjust batch sizes based on your GPU memory:

```python
# Conservative (4GB+ GPU)
batch_size = 4
batch_size_fa = 1

# Moderate (8GB+ GPU) 
batch_size = 12
batch_size_fa = 1

# Aggressive (16GB+ GPU)
batch_size = 24
batch_size_fa = 2
```

### **Memory Optimization**

- Use FP16 precision: `--fp16` flag
- Reduce batch sizes if experiencing OOM errors
- Close other GPU applications during processing
- Monitor GPU memory usage with `nvidia-smi`

## 🎬 **Development Status**

### **✅ Completed Features**
- [x] Cutaway frame detection and handling
- [x] Enhanced preprocessing with face detection
- [x] Passthrough frame processing
- [x] GPU batch optimization
- [x] FFmpeg integration fixes
- [x] Cross-platform compatibility
- [x] Gradio web interface
- [x] Comprehensive error handling

### **🚧 In Progress**
- [ ] Real-time processing optimization
- [ ] Advanced face detection models
- [ ] Multi-speaker support

### **📋 Planned Features**
- [ ] Video quality enhancement
- [ ] Advanced audio processing
- [ ] Batch video processing
- [ ] API endpoints
- [ ] Docker containerization
- [ ] Model fine-tuning tools

## 🐛 **Troubleshooting**

### **Common Issues**

**Video freezes during cutaways**
- ✅ Fixed: Enhanced frame processing handles this automatically

**Out of memory errors**
- Reduce `batch_size` in inference scripts
- Use `batch_size_fa=1` for face detection
- Close other GPU applications

**FFmpeg not found**
- Install FFmpeg and add to PATH
- Use `--ffmpeg_path` argument to specify location

**ONNX Runtime GPU issues**
- Verify CUDA installation: `nvidia-smi`
- Check ONNX providers: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
- Fallback to CPU: `pip uninstall onnxruntime-gpu && pip install onnxruntime`

**Audio sync issues**
- ✅ Fixed: Improved FFmpeg command handling
- Ensure input audio and video have matching durations

## 📚 **Documentation**

- **[Dependency Tree](diagrams/02_dependency_tree.mmd)** - Project dependencies
- **[Code Structure](diagrams/03_code_structure.mmd)** - Architecture overview
- **[Diagrams](diagrams/)** - Technical documentation

## 📄 **License**

This project builds upon MuseTalk and maintains compatibility with its licensing terms.

## 🙏 **Citations**

```bibtex
@article{zhang2024musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and Zeng, Yubin and Zhan, Chao and Huang, Junxin and He, Yingjie and Zhou, Wenjiang},
  journal={arXiv preprint arXiv:2410.10122},
  year={2024}
}
```

## 🔗 **Links**

- **[Original MuseTalk](https://github.com/TMElyralab/MuseTalk)** - Base repository
- **[Gradio](https://gradio.app/)** - Web interface framework
- **[Architecture Diagrams](diagrams/)** - Detailed technical documentation

---

*BraivTalk - Enhanced MuseTalk v2.0.0*  
*Last updated: January 2025*