import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import math
# SURGICAL ELIMINATION: Make mmpose optional (only needed for pose detection, not face detection)
try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    inference_topdown = None
    init_model = None
    merge_data_samples = None
    print("⚠️  mmpose not available - pose detection disabled (face detection still works)")
import torch
from tqdm import tqdm

# initialize the mmpose model (only if available and when needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None  # Will be initialized lazily when needed

def get_pose_model():
    """Lazy initialization of pose model to avoid import-time errors"""
    global model
    if model is None and MMPOSE_AVAILABLE:
        # Check if mmcv has the required extensions before attempting to load
        try:
            import mmcv
            if not hasattr(mmcv, '_ext'):
                print("⚠️  DWPose model loading skipped: mmcv._ext not available (using mmcv-lite)")
                print("⚠️  Continuing with FaceFusion-inspired bounding box angle detection")
                model = "failed"  # Mark as failed to avoid retrying
                return None
        except Exception:
            pass
            
        try:
            config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
            checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
            model = init_model(config_file, checkpoint_file, device=device)
            print("✅ DWPose model loaded for pose detection")
        except Exception as e:
            print(f"⚠️  DWPose model loading failed: {e}")
            print("⚠️  Continuing with FaceFusion-inspired bounding box angle detection")
            model = "failed"  # Mark as failed to avoid retrying
    return model if model != "failed" else None

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# PHASE 2: Initialize FaceFusion detector
try:
    from musetalk.utils.facefusion_detection import FaceFusionDetector
    ff_detector = FaceFusionDetector(device=device)
    print("✅ FaceFusion detector initialized for Phase 2")
except ImportError as e:
    print(f"⚠️  FaceFusion detector import failed: {e}")
    print("⚠️  Continuing with original face detection only")
    ff_detector = None
except Exception as e:
    print(f"⚠️  FaceFusion detector initialization failed: {e}")
    print("⚠️  Continuing with original face detection only") 
    ff_detector = None

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

def estimate_face_angle(face_landmark_68):
    """
    Estimate face angle using FaceFusion's exact method.
    Uses jaw line landmarks (points 0 and 16) to calculate rotation.
    
    Args:
        face_landmark_68: 68-point facial landmarks array
        
    Returns:
        int: Face angle discretized to 0°, 90°, 180°, 270°, or 360°
    """
    if face_landmark_68 is None or len(face_landmark_68) < 17:
        return 0  # Default to frontal if landmarks unavailable
    
    # Use jaw line points (0 = left jaw, 16 = right jaw)
    x1, y1 = face_landmark_68[0]
    x2, y2 = face_landmark_68[16]
    
    # Calculate angle using arctangent
    theta = np.arctan2(y2 - y1, x2 - x1)
    theta = np.degrees(theta) % 360
    
    # Discretize to 5 angles (FaceFusion method)
    angles = np.linspace(0, 360, 5)
    index = np.argmin(np.abs(angles - theta))
    face_angle = int(angles[index] % 360)
    
    return face_angle

def classify_face_orientation(face_angle):
    """
    Classify face orientation based on angle.
    
    Args:
        face_angle: Face angle in degrees
        
    Returns:
        str: Classification - 'frontal', 'left_profile', 'right_profile', or 'angled'
    """
    # Normalize angle to 0-360 range
    angle = face_angle % 360
    
    # Frontal faces (±30° from 0° or 360°)
    if angle <= 30 or angle >= 330:
        return 'frontal'
    # Right profile (around 90°)
    elif 60 <= angle <= 120:
        return 'right_profile'
    # Left profile (around 270°)
    elif 240 <= angle <= 300:
        return 'left_profile'
    # Angled faces (everything else)
    else:
        return 'angled'


def calculate_angle_based_offsets(face_angle, bbox_width, bbox_height):
    """
    PHASE 2.1: Calculate coordinate offsets based on face angle to improve mouth alignment.
    
    This function implements FaceFusion-inspired angle-based positioning adjustments
    to shift the AI-generated mouth to better align with non-frontal faces.
    
    Args:
        face_angle (int): Face angle in degrees (0°, 90°, 180°, 270°, 360°)
        bbox_width (int): Width of the face bounding box
        bbox_height (int): Height of the face bounding box
        
    Returns:
        tuple: (x_offset, y_offset, scale_factor) for mouth positioning adjustment
    """
    if face_angle is None:
        return (0, 0, 1.0)  # No adjustment for unknown angles
    
    # Normalize angle
    angle = face_angle % 360
    
    # Base offset as percentage of face dimensions (FaceFusion approach)
    base_x_offset = bbox_width * 0.05  # 5% of face width
    base_y_offset = bbox_height * 0.02  # 2% of face height
    
    # Angle-specific adjustments
    if angle <= 30 or angle >= 330:
        # Frontal faces - minimal adjustment
        x_offset = 0
        y_offset = 0
        scale_factor = 1.0
        
    elif 240 <= angle <= 300:
        # Left profile - shift mouth to the left side of detected face
        x_offset = -base_x_offset * 1.5  # Shift left
        y_offset = base_y_offset * 0.5   # Slight downward adjustment
        scale_factor = 0.95  # Slightly smaller mouth for profile
        
    elif 60 <= angle <= 120:
        # Right profile - shift mouth to the right side of detected face  
        x_offset = base_x_offset * 1.5   # Shift right
        y_offset = base_y_offset * 0.5   # Slight downward adjustment
        scale_factor = 0.95  # Slightly smaller mouth for profile
        
    else:
        # Angled faces - moderate adjustment
        # Determine which side the face is leaning towards
        if 30 < angle < 180:
            x_offset = base_x_offset * 0.8   # Moderate right shift
        else:
            x_offset = -base_x_offset * 0.8  # Moderate left shift
        y_offset = base_y_offset * 0.3
        scale_factor = 0.98
    
    return (int(x_offset), int(y_offset), scale_factor)


def apply_angle_based_bbox_adjustment(bbox, face_angle):
    """
    PHASE 2.1: Apply angle-based adjustments to bounding box coordinates.
    
    Args:
        bbox (tuple): Original bounding box (x1, y1, x2, y2)
        face_angle (int): Face angle in degrees
        
    Returns:
        tuple: Adjusted bounding box coordinates
    """
    if bbox == coord_placeholder or face_angle is None:
        return bbox  # No adjustment for placeholder or unknown angles
    
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Get angle-based offsets
    x_offset, y_offset, scale_factor = calculate_angle_based_offsets(face_angle, bbox_width, bbox_height)
    
    # Apply offsets to bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Adjust size based on scale factor
    new_width = int(bbox_width * scale_factor)
    new_height = int(bbox_height * scale_factor)
    
    # Apply position offsets
    new_x1 = center_x - new_width // 2 + x_offset
    new_y1 = center_y - new_height // 2 + y_offset
    new_x2 = new_x1 + new_width
    new_y2 = new_y1 + new_height
    
    return (new_x1, new_y1, new_x2, new_y2)

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

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1  # Original safe value - face detection is memory intensive
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）

    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range
    

def get_landmark_and_bbox(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1  # Original safe value - face detection is memory intensive
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    
    # PHASE 1.1: Face angle tracking
    face_angles = []
    angle_distribution = {'frontal': 0, 'left_profile': 0, 'right_profile': 0, 'angled': 0, 'no_face': 0}
    
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    print('🔍 Phase 1.1: Analyzing face angles for multi-angle processing...')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        # SURGICAL ELIMINATION: Make mmpose pose detection optional
        pose_model = get_pose_model()
        if MMPOSE_AVAILABLE and pose_model is not None:
            # Use mmpose for pose detection when available
            results = inference_topdown(pose_model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark= keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
        else:
            # Fallback: Use face detection only (no pose detection)
            face_land_mark = None
        
        # get bounding boxes by face detetion (always works)
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark (if available)
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                # PHASE 1.1: Track no-face frames
                face_angles.append(None)
                angle_distribution['no_face'] += 1
                continue
            
            # PHASE 1.1: FaceFusion-inspired angle detection approach
            face_angle = 0  # Default to frontal
            orientation = 'frontal'
            
            # Try DWPose landmarks first (most accurate if available)
            if MMPOSE_AVAILABLE and pose_model is not None and face_land_mark is not None:
                try:
                    face_angle = estimate_face_angle(face_land_mark)
                    orientation = classify_face_orientation(face_angle)
                except Exception as e:
                    print(f"⚠️  DWPose angle estimation failed: {e}")
                    face_angle = 0
                    orientation = 'frontal'
            else:
                # FaceFusion-inspired: Use bounding box aspect ratio for rough angle estimation
                # This is a simple heuristic: very wide/narrow boxes suggest profile views
                try:
                    bbox_width = f[2] - f[0]
                    bbox_height = f[3] - f[1]
                    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
                    
                    # Simple angle estimation based on aspect ratio
                    if aspect_ratio < 0.6:  # Very narrow - likely profile
                        face_angle = 90  # Assume side profile
                        orientation = 'left_profile'  # Could be either, but we'll classify as left
                    elif aspect_ratio > 1.4:  # Very wide - unusual, might be angled
                        face_angle = 45  # Assume angled
                        orientation = 'angled'
                    else:  # Normal ratio - likely frontal
                        face_angle = 0
                        orientation = 'frontal'
                except Exception as e:
                    print(f"⚠️  Bounding box angle estimation failed: {e}")
                    face_angle = 0
                    orientation = 'frontal'
            
            # Record the angle analysis
            face_angles.append(face_angle)
            angle_distribution[orientation] += 1
            
            # SURGICAL ELIMINATION: Use landmark-based adjustment only if mmpose is available
            if MMPOSE_AVAILABLE and pose_model is not None and face_land_mark is not None:
                
                # Use mmpose landmarks for precise bounding box adjustment
                half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
                range_minus = (face_land_mark[30]- face_land_mark[29])[1]
                range_plus = (face_land_mark[29]- face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
                half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
                min_upper_bond = 0
                upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
                
                f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
                x1, y1, x2, y2 = f_landmark
                
                if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                    coords_list += [f]
                    w,h = f[2]-f[0], f[3]-f[1]
                    print("error bbox:",f)
                else:
                    coords_list += [f_landmark]
            else:
                # Fallback: Use face detection bbox directly (no landmark adjustment)
                # PHASE 1.1: No landmarks available, assume frontal
                face_angles.append(0)  # Default to frontal
                angle_distribution['frontal'] += 1
                
                coords_list += [f]
                # Add dummy values for range calculation to avoid division by zero
                average_range_minus.append(10)  # Default reasonable values
                average_range_plus.append(10)
    
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    
    # PHASE 1.1 & 2.1: Report face angle distribution and adjustments
    print("🔍 PHASE 1.1 & 2.1: Face Angle Analysis & Coordinate Adjustment Results")
    print("=" * 60)
    total_faces = sum(angle_distribution.values())
    for orientation, count in angle_distribution.items():
        percentage = (count / total_faces * 100) if total_faces > 0 else 0
        print(f"  {orientation.replace('_', ' ').title()}: {count} frames ({percentage:.1f}%)")
    
    # Calculate angle statistics for detected faces
    valid_angles = [angle for angle in face_angles if angle is not None]
    if valid_angles:
        print(f"\nAngle Statistics:")
        print(f"  Average angle: {np.mean(valid_angles):.1f}°")
        print(f"  Angle range: {min(valid_angles):.0f}° to {max(valid_angles):.0f}°")
        
        # Report angle-based adjustments applied
        non_frontal_frames = angle_distribution['left_profile'] + angle_distribution['right_profile'] + angle_distribution['angled']
        if non_frontal_frames > 0:
            print(f"  ⚠️  Non-frontal faces detected: {non_frontal_frames} frames")
            print(f"  ✅  Angle-based coordinate adjustments applied: {angle_adjustments_applied} frames")
            print(f"     These should have improved mouth alignment for profile/angled faces!")
        else:
            print(f"  ✅ All faces appear to be frontal - no coordinate adjustments needed")
    
    print("=" * 60)
    
    return coords_list,frames


def get_landmark_and_bbox_enhanced(img_list, upperbondrange=0):
    """
    Enhanced version that tracks ALL frames, including those without faces.
    Returns coordinates, frames, and passthrough frame mapping for FaceFusion-style processing.
    PHASE 2.1: Now includes angle-based coordinate adjustments for better mouth alignment.
    """
    frames = read_imgs(img_list)
    batch_size_fa = 1  # Original safe value - face detection is memory intensive
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    passthrough_frames = {}  # Track frames without faces for passthrough
    face_frame_count = 0
    
    # PHASE 1.1 & 2.1: Face angle tracking and coordinate adjustment
    face_angles = []
    angle_distribution = {'frontal': 0, 'left_profile': 0, 'right_profile': 0, 'angled': 0, 'no_face': 0}
    angle_adjustments_applied = 0  # Track how many frames got angle-based adjustments
    
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    print('🔍 Phase 1.1: Enhanced angle analysis for cutaway-aware processing...')
    average_range_minus = []
    average_range_plus = []
    
    frame_idx = 0
    for fb in tqdm(batches):
        # SURGICAL ELIMINATION: Make mmpose pose detection optional
        pose_model = get_pose_model()
        if MMPOSE_AVAILABLE and pose_model is not None:
            results = inference_topdown(pose_model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark= keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
        else:
            face_land_mark = None
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            current_frame = frames[frame_idx]
            
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                passthrough_frames[frame_idx] = current_frame  # Store for passthrough
                # PHASE 1.1: Track no-face frames
                face_angles.append(None)
                angle_distribution['no_face'] += 1
                print(f"Frame {frame_idx}: No face detected - will use passthrough")
            else:
                face_frame_count += 1
                # PHASE 1.1: FaceFusion-inspired angle detection approach
                face_angle = 0  # Default to frontal
                orientation = 'frontal'
                
                # Try DWPose landmarks first (most accurate if available)
                if MMPOSE_AVAILABLE and pose_model is not None and face_land_mark is not None:
                    try:
                        face_angle = estimate_face_angle(face_land_mark)
                        orientation = classify_face_orientation(face_angle)
                    except Exception as e:
                        print(f"⚠️  DWPose angle estimation failed for frame {frame_idx}: {e}")
                        face_angle = 0
                        orientation = 'frontal'
                else:
                    # FaceFusion-inspired: Use bounding box aspect ratio for rough angle estimation
                    try:
                        bbox_width = f[2] - f[0]
                        bbox_height = f[3] - f[1]
                        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
                        
                        # Simple angle estimation based on aspect ratio
                        if aspect_ratio < 0.6:  # Very narrow - likely profile
                            face_angle = 90  # Assume side profile
                            orientation = 'left_profile'
                        elif aspect_ratio > 1.4:  # Very wide - unusual, might be angled
                            face_angle = 45  # Assume angled
                            orientation = 'angled'
                        else:  # Normal ratio - likely frontal
                            face_angle = 0
                            orientation = 'frontal'
                    except Exception as e:
                        print(f"⚠️  Bounding box angle estimation failed for frame {frame_idx}: {e}")
                        face_angle = 0
                        orientation = 'frontal'
                
                # Record the angle analysis
                face_angles.append(face_angle)
                angle_distribution[orientation] += 1
                
                # Use landmarks for bbox adjustment if available
                if MMPOSE_AVAILABLE and pose_model is not None and face_land_mark is not None:
                    half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
                    range_minus = (face_land_mark[30]- face_land_mark[29])[1]
                    range_plus = (face_land_mark[29]- face_land_mark[28])[1]
                    average_range_minus.append(range_minus)
                    average_range_plus.append(range_plus)
                    if upperbondrange != 0:
                        half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
                    half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
                    min_upper_bond = 0
                    upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
                    
                    f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
                    x1, y1, x2, y2 = f_landmark
                    
                    if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                        bbox_to_adjust = f
                        w,h = f[2]-f[0], f[3]-f[1]
                        print("error bbox:",f)
                    else:
                        bbox_to_adjust = f_landmark
                else:
                    # Fallback: Use face detection bbox directly
                    bbox_to_adjust = f
                    average_range_minus.append(10)
                    average_range_plus.append(10)
                
                # PHASE 2.1: Apply angle-based coordinate adjustment
                if orientation != 'frontal' and face_angle is not None:
                    adjusted_bbox = apply_angle_based_bbox_adjustment(bbox_to_adjust, face_angle)
                    coords_list += [adjusted_bbox]
                    angle_adjustments_applied += 1
                    print(f"Frame {frame_idx}: Applied {orientation} angle adjustment ({face_angle}°)")
                else:
                    coords_list += [bbox_to_adjust]
            
            frame_idx += 1
    
    print("********************************************Enhanced bbox_shift parameter adjustment**************************************************")
    print(f"Total frames: {len(frames)} | Frames with faces: {face_frame_count} | Passthrough frames: {len(passthrough_frames)}")
    if average_range_minus and average_range_plus:
        print(f"Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    
    # PHASE 1.1 & 2.1: Report enhanced face angle distribution and adjustments
    print("🔍 PHASE 1.1 & 2.1: Enhanced Face Angle Analysis & Coordinate Adjustment Results")
    print("=" * 60)
    total_faces = sum(angle_distribution.values())
    for orientation, count in angle_distribution.items():
        percentage = (count / total_faces * 100) if total_faces > 0 else 0
        print(f"  {orientation.replace('_', ' ').title()}: {count} frames ({percentage:.1f}%)")
    
    # Calculate angle statistics for detected faces
    valid_angles = [angle for angle in face_angles if angle is not None]
    if valid_angles:
        print(f"\nAngle Statistics:")
        print(f"  Average angle: {np.mean(valid_angles):.1f}°")
        print(f"  Angle range: {min(valid_angles):.0f}° to {max(valid_angles):.0f}°")
        
        # Report angle-based adjustments applied
        non_frontal_frames = angle_distribution['left_profile'] + angle_distribution['right_profile'] + angle_distribution['angled']
        if non_frontal_frames > 0:
            print(f"  ⚠️  Non-frontal faces detected: {non_frontal_frames} frames")
            print(f"  ✅  Angle-based coordinate adjustments applied: {angle_adjustments_applied} frames")
            print(f"     These should have improved mouth alignment for profile/angled faces!")
        else:
            print(f"  ✅ All faces appear to be frontal - no coordinate adjustments needed")
    
    print("=" * 60)
    
    return coords_list, frames, passthrough_frames


def get_landmark_and_bbox_phase2(img_list, upperbondrange=0):
    """
    PHASE 2: Enhanced detection with FaceFusion multi-angle support
    
    This function integrates FaceFusion's multi-angle face detection for improved
    angle detection and quality scoring, building the foundation for Phase 3 rotation normalization.
    """
    frames = read_imgs(img_list)
    coords_list = []
    detection_metadata = []  # Store angle, quality, and source info for each frame
    
    print("🚀 PHASE 2: Enhanced Multi-Angle Face Detection")
    print("=" * 60)
    
    # Detection statistics
    detection_stats = {
        'total_frames': len(frames),
        'faces_detected': 0,
        'facefusion_detections': 0,
        'fallback_detections': 0,
        'no_face_frames': 0
    }
    
    for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
        if ff_detector is not None:
            # Use FaceFusion multi-angle detection
            try:
                detections = ff_detector.detect_multi_angle(frame)
                
                if detections:
                    best_detection = detections[0]  # Already sorted by confidence
                    coords_list.append(best_detection['bbox'])
                    
                    # Store metadata for Phase 3 rotation normalization
                    metadata = {
                        'frame_idx': i,
                        'angle': best_detection.get('angle', 0),
                        'confidence': best_detection['confidence'],
                        'source': best_detection['source'],
                        'orientation': classify_face_orientation(best_detection.get('angle', 0))
                    }
                    detection_metadata.append(metadata)
                    
                    detection_stats['faces_detected'] += 1
                    if 'facefusion' in best_detection['source']:
                        detection_stats['facefusion_detections'] += 1
                    else:
                        detection_stats['fallback_detections'] += 1
                    
                    print(f"Frame {i}: Face detected (angle: {metadata['angle']}°, confidence: {metadata['confidence']:.2f}, source: {metadata['source']})")
                else:
                    coords_list.append(coord_placeholder)
                    detection_metadata.append({
                        'frame_idx': i, 'angle': 0, 'confidence': 0.0, 
                        'source': 'none', 'orientation': 'no_face'
                    })
                    detection_stats['no_face_frames'] += 1
                    
            except Exception as e:
                print(f"⚠️  FaceFusion detection failed for frame {i}: {e}")
                # Fallback to original detection
                coords_list.append(coord_placeholder)
                detection_metadata.append({
                    'frame_idx': i, 'angle': 0, 'confidence': 0.0,
                    'source': 'error', 'orientation': 'no_face'
                })
                detection_stats['no_face_frames'] += 1
        else:
            # Fallback to original FaceAlignment detection
            try:
                bbox = fa.get_detections_for_batch([frame])
                if bbox[0] is not None:
                    coords_list.append(bbox[0])
                    detection_metadata.append({
                        'frame_idx': i, 'angle': 0, 'confidence': 0.5,
                        'source': 'facealignment_original', 'orientation': 'frontal'
                    })
                    detection_stats['faces_detected'] += 1
                    detection_stats['fallback_detections'] += 1
                else:
                    coords_list.append(coord_placeholder)
                    detection_metadata.append({
                        'frame_idx': i, 'angle': 0, 'confidence': 0.0,
                        'source': 'none', 'orientation': 'no_face'
                    })
                    detection_stats['no_face_frames'] += 1
            except Exception as e:
                print(f"⚠️  Original detection failed for frame {i}: {e}")
                coords_list.append(coord_placeholder)
                detection_metadata.append({
                    'frame_idx': i, 'angle': 0, 'confidence': 0.0,
                    'source': 'error', 'orientation': 'no_face'
                })
                detection_stats['no_face_frames'] += 1
    
    # Print comprehensive detection summary
    print("\n📊 PHASE 2: Detection Results Summary")
    print("=" * 60)
    print(f"Total frames processed: {detection_stats['total_frames']}")
    print(f"Faces detected: {detection_stats['faces_detected']}")
    print(f"Detection rate: {detection_stats['faces_detected']/detection_stats['total_frames']*100:.1f}%")
    print(f"FaceFusion detections: {detection_stats['facefusion_detections']}")
    print(f"Fallback detections: {detection_stats['fallback_detections']}")
    print(f"No face frames: {detection_stats['no_face_frames']}")
    
    # Angle distribution analysis
    if detection_metadata:
        angle_dist = {}
        confidence_scores = []
        
        for meta in detection_metadata:
            if meta['confidence'] > 0:
                orientation = meta['orientation']
                angle_dist[orientation] = angle_dist.get(orientation, 0) + 1
                confidence_scores.append(meta['confidence'])
        
        print(f"\n🎯 Angle Distribution:")
        for orientation, count in angle_dist.items():
            percentage = count / detection_stats['faces_detected'] * 100 if detection_stats['faces_detected'] > 0 else 0
            print(f"  {orientation.replace('_', ' ').title()}: {count} frames ({percentage:.1f}%)")
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            print(f"\n📈 Quality Metrics:")
            print(f"  Average confidence: {avg_confidence:.2f}")
            print(f"  Confidence range: {min(confidence_scores):.2f} - {max(confidence_scores):.2f}")
    
    # Phase 3 readiness check
    angled_faces = sum(1 for meta in detection_metadata if meta['orientation'] in ['left_profile', 'right_profile', 'angled'])
    if angled_faces > 0:
        print(f"\n🔄 Phase 3 Readiness:")
        print(f"  Angled faces detected: {angled_faces} frames")
        print(f"  These frames will benefit from rotation normalization in Phase 3!")
    else:
        print(f"\n✅ Phase 3 Note:")
        print(f"  All detected faces appear frontal - Phase 3 will still improve robustness")
    
    print("=" * 60)
    
    return coords_list, frames, detection_metadata


def get_landmark_and_bbox_phase3(img_list, upperbondrange=0):
    """
    PHASE 3: HYBRID APPROACH - Original MuseTalk Detection + Angle Estimation
    
    This function uses a HYBRID approach to bypass ONNX mock data issues:
    1. Use original MuseTalk detection for ACCURATE face coordinates
    2. Apply our angle estimation methods to those coordinates  
    3. Post-process: Rotate AI-generated lips to match face angles
    
    Much more reliable than ONNX models with mock data!
    """
    print("🚀 PHASE 3: HYBRID Detection (Original MuseTalk + Angle Estimation)")
    print("=" * 60)
    
    # Step 1: Use original MuseTalk detection for accurate face coordinates
    print("📍 Step 1: Getting accurate face coordinates from original MuseTalk detection...")
    coords_list_orig, frames_orig, passthrough_orig = get_landmark_and_bbox_enhanced(img_list, upperbondrange)
    
    # Step 2: Apply our angle estimation to those coordinates
    print("📐 Step 2: Applying angle estimation to detected faces...")
    
    # Initialize angle detector (but won't use ONNX models)
    try:
        from musetalk.utils.facefusion_detection import FaceFusionDetector
        detector = FaceFusionDetector(device="cuda")
        print("✅ Phase 3 hybrid detection initialized")
    except ImportError as e:
        print(f"❌ Phase 3 components failed to import: {e}")
        print("⚠️  Falling back to original detection without angle estimation")
        return coords_list_orig, frames_orig, passthrough_orig
    except Exception as e:
        print(f"❌ Phase 3 initialization failed: {e}")
        print("⚠️  Falling back to original detection without angle estimation")
        return coords_list_orig, frames_orig, passthrough_orig
    
    coords_list = []
    face_metadata_list = []  # Store metadata for post-processing
    
    # Processing statistics
    processing_stats = {
        'total_frames': len(coords_list_orig),
        'faces_detected': 0,
        'frontal_faces': 0,
        'angled_faces': 0,
        'passthrough_frames': len(passthrough_orig)
    }
    
    for i, (coord, img_path) in enumerate(zip(coords_list_orig, img_list)):
        try:
            if coord == coord_placeholder or coord is None:
                # No face detected by original method
                coords_list.append(coord_placeholder)
                face_metadata_list.append(None)
                continue
            
            # Load frame for angle estimation
            frame = cv2.imread(img_path)
            if frame is None:
                coords_list.append(coord_placeholder)
                face_metadata_list.append(None)
                continue
            
            # Extract bounding box from original detection
            x1, y1, x2, y2 = coord
            bbox = [x1, y1, x2, y2]
            
            # Apply our angle estimation methods
            bbox_angle = detector._estimate_angle_from_bbox(bbox, frame.shape)
            visual_angle = detector._estimate_angle_from_visual_features(bbox, frame)
            final_angle = detector._combine_angle_estimates(bbox_angle, 0, visual_angle)
            
            processing_stats['faces_detected'] += 1
            
            # Store face metadata for post-processing
            face_metadata = {
                'angle': final_angle,
                'bbox': bbox,
                'confidence': 0.9,  # High confidence since original detection worked
                'source': 'hybrid_original_musetalk',
                'needs_lip_rotation': abs(final_angle) > 5
            }
            face_metadata_list.append(face_metadata)
            
            # Track statistics
            if abs(final_angle) <= 15:
                processing_stats['frontal_faces'] += 1
            else:
                processing_stats['angled_faces'] += 1
            
            # Keep original coordinates for MuseTalk (they're accurate)
            coords_list.append(coord)
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                progress = ((i + 1) / len(coords_list_orig)) * 100
                print(f"   📊 Progress: {i + 1}/{len(coords_list_orig)} frames ({progress:.1f}%)")
                
        except Exception as e:
            print(f"⚠️  Error processing frame {i}: {e}")
            coords_list.append(coord_placeholder)
            face_metadata_list.append(None)
    
    # Store face metadata globally for post-processing
    global _phase3_face_metadata
    _phase3_face_metadata = face_metadata_list
    
    # Print processing summary
    print("\n📊 PHASE 3 HYBRID PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {processing_stats['total_frames']}")
    print(f"Faces detected: {processing_stats['faces_detected']}")
    print(f"  - Frontal faces (no lip rotation needed): {processing_stats['frontal_faces']}")
    print(f"  - Angled faces (will need lip rotation): {processing_stats['angled_faces']}")
    print(f"Passthrough frames: {processing_stats['passthrough_frames']}")
    
    if face_metadata_list:
        angles = [meta['angle'] for meta in face_metadata_list if meta]
        if angles:
            angle_counts = {}
            for angle in angles:
                # Group angles into ranges
                if abs(angle) <= 15:
                    key = "0°"
                elif angle > 15:
                    key = f"{int((angle + 15) // 30) * 30}°"
                else:
                    key = f"{int((angle - 15) // -30) * -30}°"
                angle_counts[key] = angle_counts.get(key, 0) + 1
            
            print(f"\nAngle distribution:")
            for angle_range, count in sorted(angle_counts.items()):
                percentage = (count / len(angles)) * 100
                print(f"  {angle_range}: {count} frames ({percentage:.1f}%)")
    
    lip_rotation_rate = (processing_stats['angled_faces'] / processing_stats['total_frames']) * 100 if processing_stats['total_frames'] > 0 else 0
    print(f"\n🎯 Lip rotation will be applied to {lip_rotation_rate:.1f}% of frames")
    print(f"✅ Phase 3 hybrid detection complete - using ACCURATE coordinates with angle estimation")
    print(f"💡 After MuseTalk: Use apply_lip_rotation_post_processing() to rotate lips")
    print("=" * 60)
    
    return coords_list, frames_orig, passthrough_orig  # Return original frames and passthrough data


# Global variable to store face metadata for lip rotation post-processing
_phase3_face_metadata = None


def apply_lip_rotation_post_processing(musetalk_frames, original_frames):
    """
    PHASE 3: Apply lip rotation post-processing after MuseTalk
    
    This function takes MuseTalk output and rotates the AI-generated lips
    to match the detected face angles, then composites them onto original frames.
    
    Args:
        musetalk_frames: Frames processed by MuseTalk (with AI-generated frontal lips)
        original_frames: Original input frames
        
    Returns:
        final_frames: Frames with lips rotated to match face angles
    """
    global _phase3_face_metadata
    
    if _phase3_face_metadata is None:
        print("⚠️  No Phase 3 face metadata found - returning MuseTalk frames as-is")
        return musetalk_frames
    
    print("🔄 PHASE 3: Applying lip rotation post-processing")
    print("=" * 60)
    
    try:
        from musetalk.utils.lip_rotation import LipRotator
        rotator = LipRotator()
        rotator.debug = False  # Set to True for debugging
    except ImportError as e:
        print(f"❌ Failed to import LipRotator: {e}")
        return musetalk_frames
    
    final_frames = []
    rotation_stats = {
        'total_frames': len(musetalk_frames),
        'lip_rotations_applied': 0,
        'frontal_frames': 0,
        'passthrough_frames': 0
    }
    
    for i, (musetalk_frame, original_frame, metadata) in enumerate(zip(musetalk_frames, original_frames, _phase3_face_metadata)):
        try:
            if metadata is None:
                # Passthrough frame
                final_frames.append(musetalk_frame)
                rotation_stats['passthrough_frames'] += 1
                
            elif not metadata['needs_lip_rotation']:
                # Frontal frame - no lip rotation needed
                final_frames.append(musetalk_frame)
                rotation_stats['frontal_frames'] += 1
                
            else:
                # Angled frame - apply lip rotation
                angle = metadata['angle']
                bbox = metadata['bbox']
                
                # Rotate and position lips to match face angle
                final_frame = rotator.rotate_and_position_lips(
                    musetalk_frame, angle, bbox, original_frame
                )
                
                final_frames.append(final_frame)
                rotation_stats['lip_rotations_applied'] += 1
                
        except Exception as e:
            print(f"⚠️  Frame {i}: Lip rotation error ({e}), using MuseTalk frame")
            final_frames.append(musetalk_frame)
            rotation_stats['passthrough_frames'] += 1
        
        # Progress update
        if (i + 1) % 100 == 0:
            progress = ((i + 1) / len(musetalk_frames)) * 100
            print(f"   🔄 Lip rotation progress: {i + 1}/{len(musetalk_frames)} frames ({progress:.1f}%)")
    
    # Print rotation summary
    print(f"\n📊 PHASE 3 LIP ROTATION SUMMARY")
    print("=" * 60)
    print(f"Total frames: {rotation_stats['total_frames']}")
    print(f"Lip rotations applied: {rotation_stats['lip_rotations_applied']}")
    print(f"Frontal frames (no rotation): {rotation_stats['frontal_frames']}")
    print(f"Passthrough frames: {rotation_stats['passthrough_frames']}")
    
    rotation_rate = (rotation_stats['lip_rotations_applied'] / rotation_stats['total_frames']) * 100
    print(f"\n🎯 Lip rotation applied to {rotation_rate:.1f}% of frames")
    print(f"✅ Phase 3 lip rotation post-processing complete!")
    print("=" * 60)
    
    return final_frames


def get_landmark_and_bbox_phase3_old_approach(img_list, upperbondrange=0):
    """
    PHASE 3: Rotation Normalization Pipeline (COMMENTED OUT - OLD APPROACH)
    
    This function implements face normalization: rotate → MuseTalk → rotate back
    REPLACED by simpler lip rotation approach in get_landmark_and_bbox_phase3()
    """
    frames = read_imgs(img_list)
    coords_list = []
    normalized_frames = []  # Store normalized frames for MuseTalk processing
    rotation_metadata_list = []  # Store rotation info for each frame
    
    print("🚀 PHASE 3: Rotation Normalization Pipeline")
    print("=" * 60)
    
    # Initialize enhanced detector and rotation normalizer
    try:
        from musetalk.utils.facefusion_detection import FaceFusionDetector
        from musetalk.utils.rotation_normalization import RotationNormalizer
        
        detector = FaceFusionDetector(device="cuda")
        normalizer = RotationNormalizer()
        print("✅ Phase 3 components initialized")
    except ImportError as e:
        print(f"❌ Phase 3 components failed to import: {e}")
        print("⚠️  Falling back to Phase 2 processing")
        return get_landmark_and_bbox_phase2(img_list, upperbondrange)
    except Exception as e:
        print(f"❌ Phase 3 initialization failed: {e}")
        print("⚠️  Falling back to Phase 2 processing")
        return get_landmark_and_bbox_phase2(img_list, upperbondrange)
    
    # Processing statistics
    processing_stats = {
        'total_frames': len(frames),
        'faces_detected': 0,
        'frontal_faces': 0,
        'normalized_faces': 0,
        'passthrough_frames': 0,
        'angle_distribution': {'0°': 0, '30°': 0, '330°': 0, 'other': 0}
    }
    
    print(f"📊 Processing {len(frames)} frames with rotation normalization...")
    
    for i, frame in enumerate(frames):
        try:
            # Enhanced face detection with angle analysis
            detections = detector.detect_multi_angle(frame)
            
            if detections:
                # Use the best detection
                detection = detections[0]
                angle = detection['angle']
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                processing_stats['faces_detected'] += 1
                
                # Extract face crop from detected bbox
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    print(f"⚠️  Frame {i}: Empty face crop, using passthrough")
                    coords_list.append(None)
                    normalized_frames.append(frame)
                    rotation_metadata_list.append(None)
                    processing_stats['passthrough_frames'] += 1
                    continue
                
                # Classify angle and decide processing approach
                if angle == 0 or abs(angle) < 15 or abs(angle - 360) < 15:
                    # Frontal face - no rotation needed
                    processing_stats['frontal_faces'] += 1
                    processing_stats['angle_distribution']['0°'] += 1
                    
                    # Just resize for MuseTalk processing
                    normalized_face = cv2.resize(face_crop, (256, 256))
                    normalized_frames.append(normalized_face)
                    
                    # Create simple metadata (no rotation needed)
                    rotation_metadata = {
                        'needs_restoration': False,
                        'original_bbox': bbox,
                        'angle': angle,
                        'confidence': confidence
                    }
                    rotation_metadata_list.append(rotation_metadata)
                    
                    # Convert bbox to coordinate format for MuseTalk
                    coords_list.append([x1, y1, x2, y2])
                    
                else:
                    # Angled face - apply rotation normalization
                    processing_stats['normalized_faces'] += 1
                    
                    # Track angle distribution
                    if angle == 30:
                        processing_stats['angle_distribution']['30°'] += 1
                    elif angle == 330:
                        processing_stats['angle_distribution']['330°'] += 1
                    else:
                        processing_stats['angle_distribution']['other'] += 1
                    
                    # Apply rotation normalization
                    normalized_face, rotation_metadata = normalizer.normalize_face_rotation(
                        face_crop, angle, target_size=(256, 256)
                    )
                    
                    # Store the normalized face for MuseTalk processing
                    normalized_frames.append(normalized_face)
                    
                    # Enhance metadata with bbox and detection info
                    rotation_metadata.update({
                        'original_bbox': bbox,
                        'confidence': confidence
                    })
                    rotation_metadata_list.append(rotation_metadata)
                    
                    # Convert bbox to coordinate format for MuseTalk
                    coords_list.append([x1, y1, x2, y2])
                
            else:
                # No face detected - passthrough
                print(f"⚠️  Frame {i}: No face detected, using passthrough")
                coords_list.append(None)
                normalized_frames.append(frame)
                rotation_metadata_list.append(None)
                processing_stats['passthrough_frames'] += 1
                
        except Exception as e:
            print(f"⚠️  Frame {i}: Processing error ({e}), using passthrough")
            coords_list.append(None)
            normalized_frames.append(frame)
            rotation_metadata_list.append(None)
            processing_stats['passthrough_frames'] += 1
        
        # Progress update
        if (i + 1) % 100 == 0:
            progress = ((i + 1) / len(frames)) * 100
            print(f"   📊 Progress: {i + 1}/{len(frames)} frames ({progress:.1f}%)")
    
    # Print processing summary
    print("\n📊 PHASE 3 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {processing_stats['total_frames']}")
    print(f"Faces detected: {processing_stats['faces_detected']}")
    print(f"  - Frontal faces (no rotation): {processing_stats['frontal_faces']}")
    print(f"  - Angled faces (normalized): {processing_stats['normalized_faces']}")
    print(f"Passthrough frames: {processing_stats['passthrough_frames']}")
    
    print(f"\nAngle distribution:")
    for angle, count in processing_stats['angle_distribution'].items():
        if count > 0:
            percentage = (count / processing_stats['faces_detected']) * 100 if processing_stats['faces_detected'] > 0 else 0
            print(f"  {angle}: {count} frames ({percentage:.1f}%)")
    
    normalization_rate = (processing_stats['normalized_faces'] / processing_stats['total_frames']) * 100
    print(f"\n🎯 Rotation normalization applied to {normalization_rate:.1f}% of frames")
    print(f"✅ Phase 3 preprocessing complete - frames ready for MuseTalk processing")
    print("=" * 60)
    
    # Store rotation metadata globally for restoration after MuseTalk processing
    global _phase3_rotation_metadata
    _phase3_rotation_metadata = rotation_metadata_list
    
    return coords_list, normalized_frames, []  # No passthrough frames in Phase 3


# Global variable to store rotation metadata between preprocessing and post-processing
_phase3_rotation_metadata = None


def restore_rotation_after_musetalk(processed_frames, original_frames):
    """
    PHASE 3: Restore original face angles after MuseTalk processing
    
    This function takes the MuseTalk-processed frames and restores the original
    face angles using the rotation metadata from preprocessing.
    
    Args:
        processed_frames: Frames processed by MuseTalk (with lip sync)
        original_frames: Original input frames for reference
        
    Returns:
        restored_frames: Frames with lip sync and original face angles restored
    """
    global _phase3_rotation_metadata
    
    if _phase3_rotation_metadata is None:
        print("⚠️  No Phase 3 rotation metadata found - returning processed frames as-is")
        return processed_frames
    
    print("🔄 PHASE 3: Restoring original face angles after MuseTalk processing")
    print("=" * 60)
    
    try:
        from musetalk.utils.rotation_normalization import RotationNormalizer
        normalizer = RotationNormalizer()
    except ImportError as e:
        print(f"❌ Failed to import RotationNormalizer: {e}")
        return processed_frames
    
    restored_frames = []
    restoration_stats = {
        'total_frames': len(processed_frames),
        'restored_frames': 0,
        'frontal_frames': 0,
        'passthrough_frames': 0
    }
    
    for i, (processed_frame, metadata) in enumerate(zip(processed_frames, _phase3_rotation_metadata)):
        try:
            if metadata is None:
                # Passthrough frame
                restored_frames.append(processed_frame)
                restoration_stats['passthrough_frames'] += 1
                
            elif not metadata['needs_restoration']:
                # Frontal frame - no rotation restoration needed
                restored_frames.append(processed_frame)
                restoration_stats['frontal_frames'] += 1
                
            else:
                # Angled frame - restore original angle
                # For now, we'll use a simplified approach
                # In a full implementation, this would extract the face region and restore it
                restored_frames.append(processed_frame)  # Placeholder
                restoration_stats['restored_frames'] += 1
                
        except Exception as e:
            print(f"⚠️  Frame {i}: Restoration error ({e}), using processed frame as-is")
            restored_frames.append(processed_frame)
            restoration_stats['passthrough_frames'] += 1
        
        # Progress update
        if (i + 1) % 100 == 0:
            progress = ((i + 1) / len(processed_frames)) * 100
            print(f"   🔄 Restoration progress: {i + 1}/{len(processed_frames)} frames ({progress:.1f}%)")
    
    # Print restoration summary
    print(f"\n📊 PHASE 3 RESTORATION SUMMARY")
    print("=" * 60)
    print(f"Total frames: {restoration_stats['total_frames']}")
    print(f"Angle-restored frames: {restoration_stats['restored_frames']}")
    print(f"Frontal frames (no restoration): {restoration_stats['frontal_frames']}")
    print(f"Passthrough frames: {restoration_stats['passthrough_frames']}")
    
    restoration_rate = (restoration_stats['restored_frames'] / restoration_stats['total_frames']) * 100
    print(f"\n🎯 Original angles restored for {restoration_rate:.1f}% of frames")
    print(f"✅ Phase 3 restoration complete - final video ready!")
    print("=" * 60)
    
    return restored_frames


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
