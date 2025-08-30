#!/usr/bin/env python3
"""
Morphological Erosion Test Script
Compares mouth mask blending with and without erosion on 10 random frames from Canva_en video.
Creates side-by-side comparison images to demonstrate the elimination of lip bleed-through.
"""

import os
import cv2
import numpy as np
import random
import pickle
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf

# Import MuseTalk modules
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

def extract_random_frames(video_path, num_frames=10, seed=42):
    """Extract random frames from video for testing"""
    print(f"🎬 Extracting {num_frames} random frames from {video_path}")
    
    # Set random seed for reproducible results
    random.seed(seed)
    np.random.seed(seed)
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    # Select random frame indices
    frame_indices = sorted(random.sample(range(0, min(total_frames, 1000)), num_frames))
    print(f"Selected frames: {frame_indices}")
    
    # Extract frames
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    
    cap.release()
    return frames, frame_indices

def setup_models(device):
    """Initialize MuseTalk models"""
    print("🤖 Loading MuseTalk models...")
    
    # Load models
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalk/config.json",
        device=device
    )
    
    # Initialize audio processor and Whisper
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device).eval()
    whisper.requires_grad_(False)
    
    # Initialize face parser
    fp = FaceParsing()
    
    return vae, unet, pe, whisper, audio_processor, fp

def generate_ai_mouth(frame, bbox, vae, unet, pe, whisper, audio_processor, device):
    """Generate AI mouth for a frame using dummy audio"""
    try:
        x1, y1, x2, y2 = bbox
        
        # Crop and resize face
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        # Get latents
        latents = vae.get_latents_for_unet(crop_frame)
        latents = latents.to(device)
        
        # Use dummy audio features (silent audio)
        dummy_audio = np.zeros((1, 80, 16))  # Silent spectrogram
        dummy_audio = torch.from_numpy(dummy_audio).to(device).float()
        audio_features = pe(dummy_audio)
        
        # Generate AI mouth
        timesteps = torch.tensor([0], device=device)
        with torch.no_grad():
            pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_features).sample
            ai_mouth = vae.decode_latents(pred_latents)
        
        # Resize back to original face size
        ai_mouth = cv2.resize(ai_mouth.astype(np.uint8), (x2-x1, y2-y1))
        
        return ai_mouth
        
    except Exception as e:
        print(f"Warning: AI mouth generation failed: {e}")
        # Return original crop as fallback
        return frame[y1:y2, x1:x2]

def create_comparison_image(original_frame, ai_mouth, bbox, landmarks, fp, frame_idx, 
                          erosion_ratios=[0.0, 0.008], output_dir="test_results"):
    """Create side-by-side comparison of different erosion settings"""
    
    results = []
    labels = []
    
    for erosion_ratio in erosion_ratios:
        enable_erosion = erosion_ratio > 0.0
        
        # Apply blending with current erosion setting
        try:
            blended_frame = get_image(
                original_frame, ai_mouth, bbox,
                upper_boundary_ratio=0.22,
                expand=1.9,
                mode="jaw",
                fp=fp,
                use_elliptical_mask=True,
                ellipse_padding_factor=0.02,
                blur_kernel_ratio=0.06,
                landmarks=landmarks,
                mouth_vertical_offset=0.0,
                mouth_scale_factor=1.0,
                debug_mouth_mask=False,
                mask_shape="ultra_wide_ellipse",
                mask_height_ratio=0.85,
                mask_corner_radius=0.15,
                enable_pre_erosion=enable_erosion,
                erosion_ratio=erosion_ratio,
                erosion_iterations=1
            )
            
            results.append(blended_frame)
            if enable_erosion:
                labels.append(f"WITH Erosion\n(ratio={erosion_ratio})")
            else:
                labels.append("WITHOUT Erosion\n(original)")
                
        except Exception as e:
            print(f"Error in blending with erosion_ratio={erosion_ratio}: {e}")
            results.append(original_frame)
            labels.append(f"ERROR\n(ratio={erosion_ratio})")
    
    # Create side-by-side comparison
    h, w = original_frame.shape[:2]
    comparison_width = w * len(results)
    comparison_height = h + 100  # Extra space for labels
    
    comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
    
    # Place images side by side
    for i, (result, label) in enumerate(zip(results, labels)):
        x_offset = i * w
        comparison[100:100+h, x_offset:x_offset+w] = result
        
        # Add label
        cv2.putText(comparison, label.split('\n')[0], (x_offset + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        if '\n' in label:
            cv2.putText(comparison, label.split('\n')[1], (x_offset + 10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add frame info
    cv2.putText(comparison, f"Frame {frame_idx} - Morphological Erosion Comparison", 
               (10, comparison_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Test morphological erosion on random video frames")
    parser.add_argument("--video_path", default="data/video/Canva_en.mp4", help="Path to test video")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of random frames to test")
    parser.add_argument("--output_dir", default="erosion_test_results", help="Output directory for results")
    parser.add_argument("--device", default="cuda:0", help="Device to use for inference")
    parser.add_argument("--erosion_ratios", nargs='+', type=float, default=[0.0, 0.008], 
                       help="Erosion ratios to test (0.0 = no erosion)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Extract random frames
    frames, frame_indices = extract_random_frames(args.video_path, args.num_frames)
    
    if not frames:
        print("❌ No frames extracted. Check video path.")
        return
    
    # Setup models
    try:
        vae, unet, pe, whisper, audio_processor, fp = setup_models(device)
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        print("Make sure model weights are available in ./models/")
        return
    
    # Process landmarks for all frames
    print("🎯 Extracting landmarks...")
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames temporarily
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    # Get landmarks and bboxes
    coord_list, frame_list, landmarks_list = get_landmark_and_bbox(frame_paths, bbox_shift=0)
    
    # Process each frame
    print("🔥 Testing erosion on frames...")
    
    comparison_images = []
    
    for i, (frame, bbox, landmarks, frame_idx) in enumerate(tqdm(
        zip(frames, coord_list, landmarks_list, frame_indices), 
        total=len(frames), desc="Processing frames")):
        
        if bbox is None or len(bbox) != 4:
            print(f"⚠️ Frame {frame_idx}: No face detected, skipping")
            continue
            
        try:
            # Generate AI mouth
            ai_mouth = generate_ai_mouth(frame, bbox, vae, unet, pe, whisper, audio_processor, device)
            
            # Create comparison
            comparison = create_comparison_image(
                frame, ai_mouth, bbox, landmarks, fp, frame_idx, 
                erosion_ratios=args.erosion_ratios, output_dir=args.output_dir
            )
            
            comparison_images.append(comparison)
            
            # Save individual comparison
            output_path = os.path.join(args.output_dir, f"comparison_frame_{frame_idx:03d}.png")
            cv2.imwrite(output_path, comparison)
            print(f"💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_idx}: {e}")
            continue
    
    # Create summary grid
    if comparison_images:
        print("📊 Creating summary grid...")
        
        # Calculate grid dimensions
        n_images = len(comparison_images)
        grid_cols = min(2, n_images)
        grid_rows = (n_images + grid_cols - 1) // grid_cols
        
        # Get dimensions
        sample_h, sample_w = comparison_images[0].shape[:2]
        grid_width = sample_w * grid_cols
        grid_height = sample_h * grid_rows
        
        # Create grid
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        for i, img in enumerate(comparison_images):
            row = i // grid_cols
            col = i % grid_cols
            
            y_start = row * sample_h
            y_end = y_start + sample_h
            x_start = col * sample_w
            x_end = x_start + sample_w
            
            grid[y_start:y_end, x_start:x_end] = img
        
        # Save grid
        grid_path = os.path.join(args.output_dir, "erosion_comparison_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"📊 Summary grid saved: {grid_path}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"✅ Erosion test completed!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"🔍 Check individual comparisons and the summary grid")
    print(f"🎯 Look for elimination of original lip bleed-through in erosion images")

if __name__ == "__main__":
    main()
