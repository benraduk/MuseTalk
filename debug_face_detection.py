#!/usr/bin/env python3
"""
Face Detection Debug & Visualization Tool
==========================================

This tool creates comprehensive face detection visualization videos showing:
- ALL detected faces with bounding boxes and landmarks
- Primary face selection and reasoning
- Frame statistics and detection quality metrics
- Face switching events and temporal consistency

Usage:
    python debug_face_detection.py --input_video "data/video/test.mp4" --output_video "debug_output/analysis.mp4"
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Add musetalk utils to path
sys.path.append('musetalk/utils')
from face_detection.api import YOLOv8_face


class FaceDetectionDebugger:
    def __init__(self, config_path="configs/inference/test.yaml", model_path="models/face_detection/weights/yolov8n-face.onnx"):
        """Initialize the face detection debugger"""
        
        # Initialize YOLOv8 detector
        if os.path.exists(model_path):
            self.yolo_detector = YOLOv8_face(path=model_path)
        else:
            print(f"WARNING: Model not found at {model_path}")
            print("Trying fallback path...")
            fallback_path = "face_detection/weights/yolov8n-face.onnx"
            if os.path.exists(fallback_path):
                self.yolo_detector = YOLOv8_face(path=fallback_path)
            else:
                raise FileNotFoundError(f"YOLOv8 model not found at {model_path} or {fallback_path}")
        
        # Generate distinct colors for different faces
        self.colors = self._generate_face_colors(20)  # Support up to 20 faces
        
        # Statistics tracking
        self.frame_stats = []
        self.face_switches = []
        self.total_faces_detected = 0
        
        print("Face Detection Debugger initialized successfully!")
        print(f"Model: {model_path}")
        print(f"Confidence threshold: {self.yolo_detector.conf_threshold}")
        print(f"Primary face lock threshold: {self.yolo_detector.primary_face_lock_threshold}")
    
    def _generate_face_colors(self, num_colors):
        """Generate distinct colors for face visualization"""
        colors = []
        for i in range(num_colors):
            # Use HSV color space for better color distribution
            hue = int(180 * i / num_colors)
            color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color_bgr))
        return colors
    
    def create_debug_video(self, input_video, output_video, options=None):
        """Create comprehensive face detection debug video"""
        
        # Default options
        if options is None:
            options = {
                'show_landmarks': True,
                'show_confidence': True,
                'highlight_primary': True,
                'color_faces': True,
                'add_statistics': True,
                'track_face_switches': True
            }
        
        print(f"Processing video: {input_video}")
        print(f"Output will be saved to: {output_video}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise ValueError(f"Could not create output video: {output_video}")
        
        # Process frames
        frame_idx = 0
        previous_primary_center = None
        
        print("Processing frames...")
        with tqdm(total=total_frames, desc="Analyzing faces") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLOv8 detection on frame
                det_bboxes, det_conf, det_classid, landmarks = self.yolo_detector.detect(frame)
                
                # Create debug frame
                debug_frame = frame.copy()
                
                # Track statistics
                num_faces = len(det_bboxes) if len(det_bboxes) > 0 else 0
                self.total_faces_detected += num_faces
                
                # Visualize ALL faces
                if num_faces > 0:
                    debug_frame = self._visualize_all_faces(
                        debug_frame, det_bboxes, det_conf, landmarks, frame_idx, options
                    )
                    
                    # Determine primary face selection (simulate the selection logic)
                    primary_face_idx = self._simulate_primary_face_selection(
                        det_bboxes, det_conf, landmarks, frame.shape
                    )
                    
                    # Highlight primary face
                    if options.get('highlight_primary', True) and primary_face_idx is not None:
                        debug_frame = self._highlight_primary_face(
                            debug_frame, det_bboxes, primary_face_idx, frame_idx
                        )
                        
                        # Track face switches
                        if options.get('track_face_switches', True):
                            current_center = self._get_face_center(det_bboxes[primary_face_idx])
                            if previous_primary_center is not None:
                                distance = np.linalg.norm(np.array(current_center) - np.array(previous_primary_center))
                                if distance > width * 0.1:  # Significant movement
                                    self.face_switches.append({
                                        'frame': frame_idx,
                                        'distance': distance,
                                        'prev_center': previous_primary_center,
                                        'curr_center': current_center
                                    })
                            previous_primary_center = current_center
                
                # Add frame statistics
                if options.get('add_statistics', True):
                    debug_frame = self._add_frame_statistics(debug_frame, num_faces, frame_idx)
                
                # Record frame statistics
                self.frame_stats.append({
                    'frame': frame_idx,
                    'faces_detected': num_faces,
                    'primary_locked': self.yolo_detector.primary_face_locked,
                    'confidences': det_conf.tolist() if len(det_conf) > 0 else []
                })
                
                # Write frame
                writer.write(debug_frame)
                frame_idx += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        writer.release()
        
        # Generate report
        self._generate_report(output_video, total_frames)
        
        print(f"Debug video created successfully: {output_video}")
        print(f"Total frames processed: {frame_idx}")
        print(f"Average faces per frame: {self.total_faces_detected / max(frame_idx, 1):.2f}")
        print(f"Face switches detected: {len(self.face_switches)}")
    
    def _visualize_all_faces(self, frame, bboxes, confidences, landmarks, frame_idx, options):
        """Draw bounding boxes, landmarks, and info for all detected faces"""
        
        for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
            color = self.colors[i % len(self.colors)] if options.get('color_faces', True) else (0, 255, 255)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw landmarks if available and requested
            if options.get('show_landmarks', True) and len(landmarks) > i:
                face_landmarks = landmarks[i]
                if len(face_landmarks) >= 5:  # 5-point landmarks
                    for point in face_landmarks:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)
            
            # Add face info
            if options.get('show_confidence', True):
                info_text = f"Face {i}: {conf:.3f}"
                # Add background for better readability
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1-25), (x1 + text_size[0] + 5, y1-5), color, -1)
                cv2.putText(frame, info_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _simulate_primary_face_selection(self, bboxes, confidences, landmarks, frame_shape):
        """Simulate the primary face selection logic to determine which face would be selected"""
        if len(bboxes) == 0:
            return None
        
        if len(bboxes) == 1:
            return 0
        
        # Use the same logic as YOLOv8_face._select_best_face but just return the index
        frame_height, frame_width = frame_shape[:2]
        frame_center_x, frame_center_y = frame_width / 2, frame_height / 2
        scores = []
        
        for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
            x1, y1, x2, y2 = bbox
            
            # Calculate face properties
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            face_center_x = (x1 + x2) / 2
            face_center_y = (y1 + y2) / 2
            
            # 1. Confidence score
            confidence_score = float(conf)
            
            # 2. Size score
            frame_area = frame_width * frame_height
            size_score = min(face_area / (frame_area * 0.1), 1.0)
            
            # 3. Center position score
            distance_to_center = np.sqrt((face_center_x - frame_center_x)**2 + 
                                       (face_center_y - frame_center_y)**2)
            max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
            center_score = 1.0 - (distance_to_center / max_distance)
            
            # Simple weighted combination
            total_score = confidence_score * 0.4 + size_score * 0.3 + center_score * 0.3
            scores.append(total_score)
        
        return np.argmax(scores)
    
    def _highlight_primary_face(self, frame, bboxes, primary_idx, frame_idx):
        """Highlight the selected primary face with special indicators"""
        if primary_idx < len(bboxes):
            x1, y1, x2, y2 = bboxes[primary_idx].astype(int)
            
            # Draw thick primary face border
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 4)  # Green for primary
            
            # Add "PRIMARY" label with background
            label = "PRIMARY FACE"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1-35), (x1 + text_size[0] + 10, y1-5), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1+5, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Add lock status if applicable
            if self.yolo_detector.primary_face_locked:
                lock_label = "LOCKED"
                cv2.putText(frame, lock_label, (x2-80, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _add_frame_statistics(self, frame, num_faces, frame_idx):
        """Add frame-level statistics overlay"""
        stats_text = [
            f"Frame: {frame_idx}",
            f"Faces: {num_faces}",
            f"Primary Locked: {'Yes' if self.yolo_detector.primary_face_locked else 'No'}",
            f"Lock Frames: {self.yolo_detector.frames_since_primary_established}",
            f"Switches: {len(self.face_switches)}"
        ]
        
        # Add semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return frame
    
    def _get_face_center(self, bbox):
        """Get the center point of a face bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _generate_report(self, output_video, total_frames):
        """Generate a detailed analysis report"""
        report_path = output_video.replace('.mp4', '_report.json')
        
        # Calculate statistics
        avg_faces = self.total_faces_detected / max(total_frames, 1)
        switch_rate = len(self.face_switches) / max(total_frames, 1) * 100
        
        # Confidence statistics
        all_confidences = []
        for frame_stat in self.frame_stats:
            all_confidences.extend(frame_stat['confidences'])
        
        confidence_stats = {}
        if all_confidences:
            confidence_stats = {
                'mean': float(np.mean(all_confidences)),
                'std': float(np.std(all_confidences)),
                'min': float(np.min(all_confidences)),
                'max': float(np.max(all_confidences))
            }
        
        report = {
            'video_analysis': {
                'total_frames': total_frames,
                'total_faces_detected': self.total_faces_detected,
                'average_faces_per_frame': avg_faces,
                'face_switches': len(self.face_switches),
                'switch_rate_percent': switch_rate
            },
            'confidence_statistics': confidence_stats,
            'face_switches': self.face_switches[:10],  # First 10 switches
            'detection_quality': {
                'frames_with_faces': sum(1 for stat in self.frame_stats if stat['faces_detected'] > 0),
                'frames_with_multiple_faces': sum(1 for stat in self.frame_stats if stat['faces_detected'] > 1),
                'primary_lock_achieved': any(stat['primary_locked'] for stat in self.frame_stats)
            }
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Face Detection Debug & Visualization Tool')
    parser.add_argument('--input_video', required=True, help='Input video file path')
    parser.add_argument('--output_video', required=True, help='Output debug video file path')
    parser.add_argument('--config', default='configs/inference/test.yaml', help='Configuration file path')
    parser.add_argument('--model_path', default='models/face_detection/weights/yolov8n-face.onnx', help='YOLOv8 model path')
    
    # Visualization options
    parser.add_argument('--show_landmarks', type=bool, default=True, help='Show facial landmarks')
    parser.add_argument('--show_confidence', type=bool, default=True, help='Show confidence scores')
    parser.add_argument('--highlight_primary', type=bool, default=True, help='Highlight primary face')
    parser.add_argument('--color_faces', type=bool, default=True, help='Use different colors for faces')
    parser.add_argument('--add_statistics', type=bool, default=True, help='Add frame statistics')
    parser.add_argument('--track_face_switches', type=bool, default=True, help='Track face switching events')
    
    args = parser.parse_args()
    
    # Prepare options
    options = {
        'show_landmarks': args.show_landmarks,
        'show_confidence': args.show_confidence,
        'highlight_primary': args.highlight_primary,
        'color_faces': args.color_faces,
        'add_statistics': args.add_statistics,
        'track_face_switches': args.track_face_switches
    }
    
    try:
        # Initialize debugger
        debugger = FaceDetectionDebugger(config_path=args.config, model_path=args.model_path)
        
        # Create debug video
        debugger.create_debug_video(args.input_video, args.output_video, options)
        
        print("Face detection debugging completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
