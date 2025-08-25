#!/usr/bin/env python3
"""
TalkNet + YOLOv8 Debug Visualization Script
==========================================

Creates a debug video showing:
1. YOLOv8 face detection (bounding boxes + landmarks)
2. TalkNet active speaker detection (big labels on speaking faces)
3. Combined visualization on Canva_en video

Usage:
    python debug_talknet_yolo.py --input data/video/Canva_en.mp4 --output debug_output/talknet_yolo_debug.mp4
"""

import cv2
import numpy as np
import argparse
import os
import sys
import json
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from musetalk.utils.face_detection.api import YOLOv8_face
from talknet_asd import TalkNetYOLOv8ASD


class TalkNetYOLODebugger:
    """Debug visualizer for TalkNet + YOLOv8 integration"""
    
    def __init__(self, yolo_conf_threshold=0.5):
        # Initialize YOLOv8 face detector
        self.yolo_detector = YOLOv8_face(
            path='models/face_detection/weights/yoloface_8n.onnx',
            conf_thres=yolo_conf_threshold
        )
        
        # Initialize TalkNet ASD
        self.talknet_asd = TalkNetYOLOv8ASD(
            audio_window_ms=400,
            confidence_threshold=0.3,
            audio_weight=0.6,
            visual_weight=0.4
        )
        
        # Colors for visualization
        self.face_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Statistics
        self.frame_stats = []
        
    def create_debug_video(self, input_video, output_video, max_frames=None):
        """Create debug video with YOLOv8 + TalkNet visualization"""
        print(f"Creating TalkNet + YOLOv8 debug video...")
        print(f"Input: {input_video}")
        print(f"Output: {output_video}")
        
        # Initialize TalkNet audio
        audio_initialized = self.talknet_asd.initialize_audio(input_video)
        if not audio_initialized:
            print("WARNING: Audio initialization failed - proceeding with visual-only analysis")
        
        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        start_time = time.time()
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                debug_frame = self._process_frame(frame, frame_idx, fps)
                
                # Write frame
                writer.write(debug_frame)
                
                frame_idx += 1
                pbar.update(1)
                
                # Progress update
                if frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = frame_idx / elapsed
                    pbar.set_postfix({
                        'FPS': f'{fps_processed:.1f}',
                        'Faces': self.frame_stats[-1]['faces'] if self.frame_stats else 0
                    })
        
        # Cleanup
        cap.release()
        writer.release()
        
        # Generate report
        self._generate_report(output_video.replace('.mp4', '_report.json'))
        
        elapsed = time.time() - start_time
        print(f"Debug video created successfully!")
        print(f"Processing time: {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
        print(f"Output: {output_video}")
    
    def _process_frame(self, frame, frame_idx, fps):
        """Process single frame with YOLOv8 + TalkNet"""
        debug_frame = frame.copy()
        
        # 1. YOLOv8 Face Detection
        yolo_results = self.yolo_detector.get_detections_for_batch([frame], fps)
        
        if yolo_results[0] is None:
            # No faces detected
            self._add_frame_info(debug_frame, frame_idx, [], [], [])
            self.frame_stats.append({
                'frame': frame_idx,
                'faces': [],
                'speaker_scores': [],
                'active_speaker': None
            })
            return debug_frame
        
        # Extract face data
        face_bbox = yolo_results[0]
        faces = [face_bbox]  # YOLOv8 returns single best face
        
        # For now, use the single best face from YOLOv8
        # TODO: Modify YOLOv8 to return all detections for multi-face scenarios
        all_faces = [face_bbox] if face_bbox is not None else []
        all_landmarks = [None]  # YOLOv8 landmarks not easily accessible in current implementation
        
        # 2. TalkNet Active Speaker Detection
        speaker_scores = []
        active_speaker = None
        
        if len(all_faces) > 0:
            try:
                speaker_scores = self.talknet_asd.detect_active_speaker(
                    all_faces, all_landmarks, frame_idx, fps
                )
                active_speaker = self.talknet_asd.get_best_speaker(speaker_scores)
            except Exception as e:
                print(f"TalkNet error at frame {frame_idx}: {e}")
                speaker_scores = [0.0] * len(all_faces)
        
        # 3. Visualization
        debug_frame = self._visualize_detections(
            debug_frame, all_faces, all_landmarks, speaker_scores, active_speaker
        )
        
        # 4. Add frame information
        self._add_frame_info(debug_frame, frame_idx, all_faces, speaker_scores, active_speaker)
        
        # 5. Store statistics
        self.frame_stats.append({
            'frame': frame_idx,
            'faces': len(all_faces),
            'speaker_scores': speaker_scores,
            'active_speaker': active_speaker
        })
        
        return debug_frame
    

    
    def _visualize_detections(self, frame, faces, landmarks_list, speaker_scores, active_speaker):
        """Visualize YOLOv8 faces + TalkNet speaker detection"""
        
        for i, face_bbox in enumerate(faces):
            x1, y1, x2, y2 = map(int, face_bbox)
            
            # Choose color based on speaker status
            if i == active_speaker:
                color = (0, 255, 0)  # Green for active speaker
                thickness = 4
            else:
                color = self.face_colors[i % len(self.face_colors)]
                thickness = 2
            
            # Draw face bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw landmarks if available
            if i < len(landmarks_list) and landmarks_list[i] is not None:
                landmarks = landmarks_list[i]
                for point in landmarks:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)
            
            # Add face ID and confidence
            face_label = f"Face {i}"
            if i < len(speaker_scores):
                face_label += f" ({speaker_scores[i]:.2f})"
            
            cv2.putText(frame, face_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Big "SPEAKING" label for active speaker
            if i == active_speaker:
                label_text = "SPEAKING"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                
                # Position label above face
                label_x = x1 + (x2 - x1 - text_size[0]) // 2
                label_y = y1 - 30
                
                # Background rectangle for better visibility
                cv2.rectangle(frame, 
                            (label_x - 10, label_y - text_size[1] - 10),
                            (label_x + text_size[0] + 10, label_y + 10),
                            (0, 255, 0), -1)
                
                # Text
                cv2.putText(frame, label_text, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        return frame
    
    def _add_frame_info(self, frame, frame_idx, faces, speaker_scores, active_speaker):
        """Add frame information overlay"""
        info_lines = [
            f"Frame: {frame_idx}",
            f"Faces: {len(faces)}",
            f"Active Speaker: {active_speaker if active_speaker is not None else 'None'}",
            f"TalkNet + YOLOv8 Debug"
        ]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text
        y_offset = 35
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def _generate_report(self, report_path):
        """Generate analysis report"""
        if not self.frame_stats:
            return
        
        # Calculate statistics
        total_frames = len(self.frame_stats)
        frames_with_faces = sum(1 for stat in self.frame_stats if 
                               (isinstance(stat['faces'], int) and stat['faces'] > 0) or
                               (isinstance(stat['faces'], list) and len(stat['faces']) > 0))
        frames_with_speaker = sum(1 for stat in self.frame_stats if stat['active_speaker'] is not None)
        
        # Handle both int and list types for faces
        face_counts = []
        for stat in self.frame_stats:
            if isinstance(stat['faces'], int):
                face_counts.append(stat['faces'])
            elif isinstance(stat['faces'], list):
                face_counts.append(len(stat['faces']))
            else:
                face_counts.append(0)
        
        avg_faces = np.mean(face_counts) if face_counts else 0
        
        report = {
            'summary': {
                'total_frames': total_frames,
                'frames_with_faces': frames_with_faces,
                'frames_with_active_speaker': frames_with_speaker,
                'face_detection_rate': frames_with_faces / total_frames * 100,
                'speaker_detection_rate': frames_with_speaker / total_frames * 100,
                'average_faces_per_frame': avg_faces
            },
            'frame_stats': self.frame_stats[:100]  # First 100 frames for brevity
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved: {report_path}")
        print(f"Face detection rate: {report['summary']['face_detection_rate']:.1f}%")
        print(f"Speaker detection rate: {report['summary']['speaker_detection_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='TalkNet + YOLOv8 Debug Visualization')
    parser.add_argument('--input', default='data/video/Canva_en.mp4',
                       help='Input video path')
    parser.add_argument('--output', default='debug_output/talknet_yolo_debug.mp4',
                       help='Output debug video path')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--yolo_conf', type=float, default=0.5,
                       help='YOLOv8 confidence threshold')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        return
    
    # Create debugger
    debugger = TalkNetYOLODebugger(yolo_conf_threshold=args.yolo_conf)
    
    # Create debug video
    try:
        debugger.create_debug_video(args.input, args.output, args.max_frames)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
