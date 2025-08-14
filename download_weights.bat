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
mkdir %CheckpointsDir%\facefusion

:: Install required packages
pip install -U "huggingface_hub[cli]"
pip install gdown

:: Set HuggingFace endpoint
set HF_ENDPOINT=https://hf-mirror.com

:: Download MuseTalk weights
huggingface-cli download TMElyralab/MuseTalk --local-dir %CheckpointsDir%

:: Download SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae --include "config.json" "diffusion_pytorch_model.bin"

:: Download Whisper weights
huggingface-cli download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

:: Download DWPose weights
huggingface-cli download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose --include "dw-ll_ucoco_384.pth"

:: Download SyncNet weights
huggingface-cli download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet --include "latentsync_syncnet.pt"



:: Download Face Parse Bisent weights (using gdown)
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O %CheckpointsDir%\face-parse-bisent\79999_iter.pth

:: Download ResNet weights
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o %CheckpointsDir%\face-parse-bisent\resnet18-5c106cde.pth

:: Download FaceFusion face detection models for Phase 2 multi-angle processing
echo Downloading FaceFusion face detection models...

:: Download SCRFD model (primary choice for multi-angle detection)
curl -L https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.onnx -o %CheckpointsDir%\facefusion\scrfd_2.5g.onnx
curl -L https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.hash -o %CheckpointsDir%\facefusion\scrfd_2.5g.hash

:: Download YOLO_Face model (backup option for extreme angles)
curl -L https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx -o %CheckpointsDir%\facefusion\yoloface_8n.onnx
curl -L https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.hash -o %CheckpointsDir%\facefusion\yoloface_8n.hash

:: Download RetinaFace model (high quality for frontal faces)
curl -L https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.onnx -o %CheckpointsDir%\facefusion\retinaface_10g.onnx
curl -L https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.hash -o %CheckpointsDir%\facefusion\retinaface_10g.hash

echo FaceFusion models downloaded for Phase 2 multi-angle face detection!
echo All weights have been downloaded successfully!
endlocal 