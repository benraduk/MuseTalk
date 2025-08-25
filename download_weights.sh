#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper models/gpen_bfr models/face_detection/weights models/talknet

# Install required packages
pip install -U "huggingface_hub[cli]"
pip install gdown

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.0 weights
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
hf download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
hf download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download DWPose weights
hf download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"





# Download Face Parse Bisent weights
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

# Download GPEN-BFR models
echo "📥 Downloading GPEN-BFR models..."
curl -L "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.onnx" \
  -o $CheckpointsDir/gpen_bfr/gpen_bfr_256.onnx

# Download YOLOv8 Face Detection ONNX model
echo "📥 Downloading YOLOv8 Face Detection model..."
curl -L "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx" \
  -o $CheckpointsDir/face_detection/weights/yoloface_8n.onnx

# Download TalkNet Active Speaker Detection models
echo "📥 Downloading TalkNet ASD models..."

# Download pre-trained TalkNet model for active speaker detection
curl -L "https://github.com/TaoRuijie/TalkNet-ASD/releases/download/v0.1/pretrain_TalkSet.model" \
  -o $CheckpointsDir/talknet/pretrain_TalkSet.model

# Download S3FD face detection model for TalkNet
curl -L "https://github.com/TaoRuijie/TalkNet-ASD/releases/download/v0.1/sfd_face.pth" \
  -o $CheckpointsDir/talknet/sfd_face.pth

# Alternative download links (if above fails)
# You can also download from: https://drive.google.com/drive/folders/1WkJNKVetOOGXNaWoUNWaJLOKvbhvVRhd
# Manual download instructions will be provided if automatic download fails

echo "✅ All weights have been downloaded successfully!" 
