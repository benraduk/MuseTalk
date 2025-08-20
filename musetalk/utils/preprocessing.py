import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm

# Initialize face detection model (core MuseTalk functionality)
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

# Removed get_bbox_range - DWPose dependency eliminated
    

def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Simplified face detection using only FaceAlignment.
    Returns face bounding boxes for frames with faces, coord_placeholder for frames without.
    """
    frames = read_imgs(img_list)
    batch_size_fa = 1  # 🔧 REVERTED: Batch processing is slower for SFD face detector
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    
    if upperbondrange != 0:
        print(f'🔍 Face detection with bbox_shift: {upperbondrange}')
    else:
        print('🔍 Face detection with default bbox')
    
    for fb in tqdm(batches, desc="Detecting faces"):
        # Get face bounding boxes using FaceAlignment
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        for j, f in enumerate(bbox):
            if f is None:  # No face detected
                coords_list += [coord_placeholder]
                continue
            
            # Apply bbox_shift if specified (simple vertical adjustment)
            x1, y1, x2, y2 = f
            if upperbondrange != 0:
                y1 = max(0, y1 + upperbondrange)  # Shift top boundary
            
            # Ensure bbox stays within image bounds
            img_height, img_width = fb[j].shape[:2]
            x1 = max(0, x1)
            x2 = min(img_width, x2)
            y1 = max(0, y1)
            y2 = min(img_height, y2)
            
            coords_list += [(x1, y1, x2, y2)]
    
    print("="*80)
    print(f"✅ Face detection complete: {len(frames)} frames processed")
    face_count = sum(1 for coord in coords_list if coord != coord_placeholder)
    print(f"📊 Faces detected: {face_count}/{len(frames)} frames")
    print("="*80)
    
    return coords_list, frames


# Removed get_landmark_and_bbox_enhanced - DWPose dependency eliminated
# The main get_landmark_and_bbox function now handles all cases
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
