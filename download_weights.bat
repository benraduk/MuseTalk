@echo off
setlocal

:: Set the checkpoints directory
set CheckpointsDir=models

:: Create necessary directories
mkdir %CheckpointsDir%\musetalk
mkdir %CheckpointsDir%\musetalkV15
mkdir %CheckpointsDir%\syncnet

mkdir %CheckpointsDir%\dwpose
mkdir %CheckpointsDir%\face-parse-bisent
mkdir %CheckpointsDir%\sd-vae-ft-mse
mkdir %CheckpointsDir%\whisper
mkdir %CheckpointsDir%\gpen_bfr
mkdir %CheckpointsDir%\face_detection
mkdir %CheckpointsDir%\face_detection\weights

:: Install required packages
pip install -U "huggingface_hub[cli]"
pip install gdown

:: Set HuggingFace endpoint
set HF_ENDPOINT=https://hf-mirror.com

:: Download MuseTalk weights
hf download TMElyralab/MuseTalk --local-dir %CheckpointsDir%

:: Download SD VAE weights
hf download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae --include "config.json" "diffusion_pytorch_model.bin"

:: Download Whisper weights
hf download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

:: Download DWPose weights
hf download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose --include "dw-ll_ucoco_384.pth"

:: Download SyncNet weights
hf download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet --include "latentsync_syncnet.pt"



:: Download Face Parse Bisent weights (using gdown)
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O %CheckpointsDir%\face-parse-bisent\79999_iter.pth

:: Download ResNet weights
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o %CheckpointsDir%\face-parse-bisent\resnet18-5c106cde.pth

:: Download GPEN-BFR models
echo Downloading GPEN-BFR models...
curl -L "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.onnx" -o %CheckpointsDir%\gpen_bfr\gpen_bfr_256.onnx

:: Download YOLOv8 Face Detection ONNX model
echo Downloading YOLOv8 Face Detection model...
curl -L "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx" -o %CheckpointsDir%\face_detection\weights\yoloface_8n.onnx

echo All weights have been downloaded successfully!
endlocal 