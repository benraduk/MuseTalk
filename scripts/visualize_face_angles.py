#!/usr/bin/env python3
"""
Phase 1.4: Face Angle Visualization Tool
========================================

This script creates debug visualizations of face angle detection to help understand
how our FaceFusion-inspired angle estimation works on different face orientations.

Features:
- Overlays detected face angles on frames
- Shows bounding boxes and confidence scores
- Generates angle distribution charts
- Exports debug frames with annotations
- Provides detailed angle analysis reports

Usage:
    python scripts/visualize_face_angles.py --input_video data/video/Canva_en.mp4 --output_dir results/angle_debug
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from tqdm import tqdm
from pathlib import Path
import torch

# Add project root to path
sys.path.append('.')

from musetalk.utils.preprocessing import get_landmark_and_bbox, estimate_face_angle, classify_face_orientation
from musetalk.utils.blending import get_image
from face_detection import FaceAlignment, LandmarksType


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize face angles for debugging')
    parser.add_argument('--input_video', type=str, required=True, help='Input video file')
    parser.add_argument('--output_dir', type=str, default='results/angle_debug', help='Output directory for debug images')
    parser.add_argument('--max_frames', type=int, default=100, help='Maximum frames to process (for quick testing)')
    parser.add_argument('--sample_rate', type=int, default=5, help='Sample every N frames')
    parser.add_argument('--bbox_size', type=int, default=256, help='Bounding box size for face detection')
    return parser.parse_args()


def draw_angle_overlay(img, bbox, angle, orientation, confidence=1.0):
    """Draw angle information overlay on image"""
    img_vis = img.copy()
    
    if bbox is None or len(bbox) != 4:
        return img_vis
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    color_map = {
        'frontal': (0, 255, 0),      # Green
        'left_profile': (255, 0, 0),  # Blue  
        'right_profile': (0, 0, 255), # Red
        'angled': (0, 255, 255),     # Yellow
        'no_face': (128, 128, 128)   # Gray
    }
    
    color = color_map.get(orientation, (255, 255, 255))
    
    # Draw bounding box
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
    
    # Draw angle text
    text = f"{orientation.replace('_', ' ').title()}: {angle}°"
    font_scale = 0.6
    thickness = 2
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(img_vis, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
    
    # Draw text
    cv2.putText(img_vis, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Draw confidence if available
    if confidence < 1.0:
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(img_vis, conf_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return img_vis


def create_angle_distribution_chart(angle_stats, output_path):
    """Create angle distribution visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of orientation distribution
    orientations = list(angle_stats['distribution'].keys())
    counts = list(angle_stats['distribution'].values())
    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FFD700', '#808080']  # Green, Blue, Red, Yellow, Gray
    
    ax1.pie(counts, labels=orientations, autopct='%1.1f%%', colors=colors[:len(orientations)])
    ax1.set_title('Face Orientation Distribution', fontsize=14, fontweight='bold')
    
    # Histogram of actual angles
    if angle_stats['angles']:
        angles = [a for a in angle_stats['angles'] if a is not None]
        if angles:
            ax2.hist(angles, bins=36, range=(0, 360), alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Face Angle (degrees)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Face Angle Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add vertical lines for key angles
            for angle, color in [(0, 'green'), (90, 'blue'), (180, 'red'), (270, 'orange')]:
                ax2.axvline(x=angle, color=color, linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'No valid angles detected', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Face Angle Distribution (No Data)', fontsize=14)
    else:
        ax2.text(0.5, 0.5, 'No angle data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Face Angle Distribution (No Data)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Angle distribution chart saved: {output_path}")


def analyze_video_angles(video_path, output_dir, max_frames=100, sample_rate=5, bbox_size=256):
    """Analyze face angles in video and create debug visualizations"""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'debug_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Load models (minimal for face detection only)
    print("🔄 Loading face detection models...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
        print("✅ Face detection model loaded")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video info: {total_frames} frames, {fps:.2f} FPS")
    print(f"🎯 Processing every {sample_rate} frames, max {max_frames} frames")
    
    # Initialize statistics
    angle_stats = {
        'distribution': {'frontal': 0, 'left_profile': 0, 'right_profile': 0, 'angled': 0, 'no_face': 0},
        'angles': [],
        'total_frames': 0,
        'processed_frames': 0
    }
    
    frame_idx = 0
    processed_count = 0
    
    # Process frames
    pbar = tqdm(total=min(max_frames, total_frames // sample_rate), desc="Processing frames")
    
    while cap.isOpened() and processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on sample rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        try:
            # Resize frame for processing
            frame_resized = cv2.resize(frame, (bbox_size, bbox_size))
            
            # Get face detection (using same method as preprocessing)
            fb = [frame_resized]  # Batch of 1 frame
            bbox = fa.get_detections_for_batch(np.asarray(fb))
            
            if bbox[0] is not None:
                # Face detected
                x1, y1, x2, y2 = bbox[0]
                
                # Simple angle estimation using bounding box aspect ratio
                # This matches our fallback method in preprocessing.py
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 1.0
                
                # Convert aspect ratio to angle estimate
                if aspect_ratio > 1.2:
                    face_angle = 90  # More profile-like
                    orientation = 'right_profile'
                elif aspect_ratio < 0.8:
                    face_angle = 270  # More profile-like (other side)
                    orientation = 'left_profile'
                elif 0.9 <= aspect_ratio <= 1.1:
                    face_angle = 0   # More frontal
                    orientation = 'frontal'
                else:
                    face_angle = 45  # Angled
                    orientation = 'angled'
                
                angle_stats['angles'].append(face_angle)
                angle_stats['distribution'][orientation] += 1
                
                # Create debug visualization
                debug_frame = draw_angle_overlay(frame, bbox[0], face_angle, orientation)
                
            else:
                # No face detected
                face_angle = None
                orientation = 'no_face'
                angle_stats['distribution']['no_face'] += 1
                debug_frame = frame.copy()
                
                # Add "No Face" text
                cv2.putText(debug_frame, "NO FACE DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save debug frame
            debug_filename = f"frame_{frame_idx:06d}_{orientation}_{face_angle if face_angle else 'none'}.jpg"
            debug_path = os.path.join(frames_dir, debug_filename)
            cv2.imwrite(debug_path, debug_frame)
            
            angle_stats['processed_frames'] += 1
            processed_count += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"⚠️  Error processing frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    cap.release()
    pbar.close()
    
    angle_stats['total_frames'] = frame_idx
    
    # Create summary report
    report = {
        'video_info': {
            'path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'processed_frames': angle_stats['processed_frames'],
            'sample_rate': sample_rate
        },
        'angle_analysis': angle_stats,
        'statistics': {
            'face_detection_rate': (angle_stats['processed_frames'] - angle_stats['distribution']['no_face']) / angle_stats['processed_frames'] * 100 if angle_stats['processed_frames'] > 0 else 0,
            'avg_angle': np.mean([a for a in angle_stats['angles'] if a is not None]) if angle_stats['angles'] else 0,
            'angle_range': {
                'min': min([a for a in angle_stats['angles'] if a is not None]) if angle_stats['angles'] else 0,
                'max': max([a for a in angle_stats['angles'] if a is not None]) if angle_stats['angles'] else 0
            }
        }
    }
    
    # Save report
    report_path = os.path.join(output_dir, 'angle_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Analysis report saved: {report_path}")
    
    # Create visualization chart
    chart_path = os.path.join(output_dir, 'angle_distribution.png')
    create_angle_distribution_chart(angle_stats, chart_path)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 FACE ANGLE ANALYSIS SUMMARY")
    print("="*60)
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"🔢 Processed: {angle_stats['processed_frames']} frames")
    print(f"👤 Face detection rate: {report['statistics']['face_detection_rate']:.1f}%")
    print(f"📊 Angle distribution:")
    for orientation, count in angle_stats['distribution'].items():
        percentage = count / angle_stats['processed_frames'] * 100 if angle_stats['processed_frames'] > 0 else 0
        print(f"   • {orientation.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    if angle_stats['angles']:
        print(f"📐 Average angle: {report['statistics']['avg_angle']:.1f}°")
        print(f"📏 Angle range: {report['statistics']['angle_range']['min']}° - {report['statistics']['angle_range']['max']}°")
    
    print(f"📁 Debug frames saved in: {frames_dir}")
    print("="*60)
    
    return report


def main():
    args = parse_args()
    
    print("🎬 Face Angle Visualization Tool - Phase 1.4")
    print("=" * 50)
    
    if not os.path.exists(args.input_video):
        print(f"❌ Input video not found: {args.input_video}")
        return
    
    # Run analysis
    report = analyze_video_angles(
        video_path=args.input_video,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        sample_rate=args.sample_rate,
        bbox_size=args.bbox_size
    )
    
    if report:
        print(f"\n✅ Analysis complete! Check results in: {args.output_dir}")
    else:
        print("\n❌ Analysis failed!")


if __name__ == "__main__":
    main()
