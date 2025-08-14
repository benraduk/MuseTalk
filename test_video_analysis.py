#!/usr/bin/env python3
"""
Full Video Face Angle Analysis Tool
Tests the enhanced FaceFusion detection on complete videos and generates detailed reports
"""
import cv2
import sys
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('.')

def analyze_video_face_angles(video_path, output_dir="video_analysis", max_frames=None, sample_rate=1):
    """
    Analyze face angles throughout a complete video
    
    Args:
        video_path: Path to input video
        output_dir: Directory for output reports and visualizations
        max_frames: Maximum frames to process (None = all frames)
        sample_rate: Process every Nth frame (1 = every frame, 5 = every 5th frame)
    """
    print("🎬 VIDEO FACE ANGLE ANALYSIS")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Sample rate: Every {sample_rate} frame(s)")
    if max_frames:
        print(f"Max frames: {max_frames}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    try:
        from musetalk.utils.facefusion_detection import FaceFusionDetector
        detector = FaceFusionDetector(device="cuda")
        print("✅ FaceFusion detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {duration:.2f} seconds")
    print()
    
    # Analysis data storage
    frame_data = []
    angle_distribution = defaultdict(int)
    detection_stats = {
        'total_processed': 0,
        'faces_detected': 0,
        'no_face_frames': 0,
        'model_usage': defaultdict(int),
        'angle_methods_success': defaultdict(int)
    }
    
    # Process frames
    frame_idx = 0
    processed_count = 0
    
    print("🔄 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames based on sample rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
            
        # Stop if max_frames reached
        if max_frames and processed_count >= max_frames:
            break
        
        timestamp = frame_idx / fps if fps > 0 else 0
        
        try:
            # Detect faces and angles
            detections = detector.detect_multi_angle(frame)
            
            frame_info = {
                'frame_number': frame_idx,
                'timestamp': timestamp,
                'faces_detected': len(detections),
                'detections': []
            }
            
            if detections:
                detection_stats['faces_detected'] += 1
                
                for i, det in enumerate(detections):
                    angle = det['angle']
                    confidence = det['confidence']
                    source = det['source']
                    
                    # Classify angle
                    if angle <= 15 or angle >= 345:
                        angle_category = "frontal"
                    elif 15 < angle <= 75 or 285 <= angle < 345:
                        angle_category = "slight_angle"
                    elif 75 < angle <= 105 or 255 <= angle < 285:
                        angle_category = "profile"
                    else:
                        angle_category = "extreme_angle"
                    
                    detection_info = {
                        'face_id': i,
                        'angle': angle,
                        'angle_category': angle_category,
                        'confidence': confidence,
                        'source_model': source,
                        'bbox': det['bbox']
                    }
                    
                    # Add angle breakdown if available
                    if 'angle_breakdown' in det:
                        breakdown = det['angle_breakdown']
                        detection_info['angle_breakdown'] = {
                            'bbox_method': breakdown['bbox'],
                            'landmark_method': breakdown['landmark'],
                            'visual_method': breakdown['visual']
                        }
                        
                        # Track which methods succeeded
                        if breakdown['bbox'] != 0:
                            detection_stats['angle_methods_success']['bbox'] += 1
                        if breakdown['landmark'] != 0:
                            detection_stats['angle_methods_success']['landmark'] += 1
                        if breakdown['visual'] != 0:
                            detection_stats['angle_methods_success']['visual'] += 1
                    
                    frame_info['detections'].append(detection_info)
                    
                    # Update statistics
                    angle_distribution[angle_category] += 1
                    detection_stats['model_usage'][source] += 1
                    
            else:
                detection_stats['no_face_frames'] += 1
            
            frame_data.append(frame_info)
            detection_stats['total_processed'] += 1
            processed_count += 1
            
            # Progress update
            if processed_count % 50 == 0:
                progress = (processed_count / (max_frames or total_frames // sample_rate)) * 100
                print(f"   Processed {processed_count} frames ({progress:.1f}%)")
                
        except Exception as e:
            print(f"⚠️  Error processing frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"✅ Processing complete: {processed_count} frames analyzed")
    print()
    
    # Generate reports
    generate_analysis_reports(frame_data, detection_stats, angle_distribution, 
                            video_path, output_dir, fps, sample_rate)

def generate_analysis_reports(frame_data, detection_stats, angle_distribution, 
                            video_path, output_dir, fps, sample_rate):
    """Generate comprehensive analysis reports"""
    
    print("📋 GENERATING REPORTS...")
    
    # 1. JSON Report (detailed data)
    report_data = {
        'video_info': {
            'path': video_path,
            'analysis_date': datetime.now().isoformat(),
            'fps': fps,
            'sample_rate': sample_rate
        },
        'summary_statistics': detection_stats,
        'angle_distribution': dict(angle_distribution),
        'frame_data': frame_data
    }
    
    json_path = os.path.join(output_dir, 'face_angle_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"✅ Detailed JSON report: {json_path}")
    
    # 2. CSV Report (for spreadsheet analysis)
    csv_data = []
    for frame in frame_data:
        if frame['faces_detected'] > 0:
            for det in frame['detections']:
                row = {
                    'frame_number': frame['frame_number'],
                    'timestamp': f"{frame['timestamp']:.2f}s",
                    'face_id': det['face_id'],
                    'angle': det['angle'],
                    'angle_category': det['angle_category'],
                    'confidence': det['confidence'],
                    'source_model': det['source_model']
                }
                
                # Add breakdown if available
                if 'angle_breakdown' in det:
                    breakdown = det['angle_breakdown']
                    row.update({
                        'bbox_angle': breakdown['bbox_method'],
                        'landmark_angle': breakdown['landmark_method'],
                        'visual_angle': breakdown['visual_method']
                    })
                
                csv_data.append(row)
        else:
            # No face detected
            csv_data.append({
                'frame_number': frame['frame_number'],
                'timestamp': f"{frame['timestamp']:.2f}s",
                'face_id': -1,
                'angle': 'NO_FACE',
                'angle_category': 'no_detection',
                'confidence': 0,
                'source_model': 'none'
            })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'face_angles.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV report: {csv_path}")
    
    # 3. Text Summary Report
    txt_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("FACE ANGLE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Video: {video_path}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Rate: Every {sample_rate} frame(s)\n\n")
        
        f.write("DETECTION STATISTICS:\n")
        f.write(f"  Total frames processed: {detection_stats['total_processed']}\n")
        f.write(f"  Frames with faces: {detection_stats['faces_detected']}\n")
        f.write(f"  Frames without faces: {detection_stats['no_face_frames']}\n")
        f.write(f"  Detection rate: {(detection_stats['faces_detected']/detection_stats['total_processed']*100):.1f}%\n\n")
        
        f.write("ANGLE DISTRIBUTION:\n")
        for category, count in angle_distribution.items():
            percentage = (count / sum(angle_distribution.values()) * 100) if angle_distribution else 0
            f.write(f"  {category}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("MODEL USAGE:\n")
        for model, count in detection_stats['model_usage'].items():
            percentage = (count / sum(detection_stats['model_usage'].values()) * 100) if detection_stats['model_usage'] else 0
            f.write(f"  {model}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("ANGLE DETECTION METHODS SUCCESS:\n")
        for method, count in detection_stats['angle_methods_success'].items():
            f.write(f"  {method}: {count} successful detections\n")
    
    print(f"✅ Summary report: {txt_path}")
    
    # 4. Generate visualizations
    generate_visualizations(frame_data, angle_distribution, detection_stats, output_dir)
    
    print()
    print("📊 QUICK SUMMARY:")
    print(f"   Processed: {detection_stats['total_processed']} frames")
    print(f"   Face detection rate: {(detection_stats['faces_detected']/detection_stats['total_processed']*100):.1f}%")
    print(f"   Most common angle: {max(angle_distribution.items(), key=lambda x: x[1])[0] if angle_distribution else 'N/A'}")
    print(f"   Primary model: {max(detection_stats['model_usage'].items(), key=lambda x: x[1])[0] if detection_stats['model_usage'] else 'N/A'}")

def generate_visualizations(frame_data, angle_distribution, detection_stats, output_dir):
    """Generate visualization charts"""
    try:
        # 1. Angle Distribution Pie Chart
        if angle_distribution:
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            labels = list(angle_distribution.keys())
            sizes = list(angle_distribution.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            
            plt.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
            plt.title('Face Angle Distribution')
            
            # 2. Model Usage Bar Chart
            plt.subplot(1, 2, 2)
            if detection_stats['model_usage']:
                models = list(detection_stats['model_usage'].keys())
                counts = list(detection_stats['model_usage'].values())
                
                plt.bar(models, counts, color=['#ff7f7f', '#7f7fff', '#7fff7f'])
                plt.title('Detection Model Usage')
                plt.xlabel('Model')
                plt.ylabel('Detections')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'analysis_charts.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Visualization charts: {viz_path}")
            
        # 3. Timeline of angles (if enough data)
        angles_over_time = []
        timestamps = []
        
        for frame in frame_data:
            if frame['faces_detected'] > 0:
                # Use first face if multiple detected
                angle = frame['detections'][0]['angle']
                angles_over_time.append(angle)
                timestamps.append(frame['timestamp'])
        
        if len(angles_over_time) > 10:  # Only if we have enough data points
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, angles_over_time, 'b-', alpha=0.7, linewidth=1)
            plt.scatter(timestamps, angles_over_time, c=angles_over_time, cmap='viridis', s=20, alpha=0.6)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Face Angle (degrees)')
            plt.title('Face Angle Over Time')
            plt.colorbar(label='Angle (degrees)')
            plt.grid(True, alpha=0.3)
            
            timeline_path = os.path.join(output_dir, 'angle_timeline.png')
            plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Angle timeline: {timeline_path}")
            
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")

def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze face angles in video')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output', '-o', default='video_analysis', help='Output directory')
    parser.add_argument('--max-frames', '-m', type=int, help='Maximum frames to process')
    parser.add_argument('--sample-rate', '-s', type=int, default=5, help='Process every N frames (default: 5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    analyze_video_face_angles(args.video, args.output, args.max_frames, args.sample_rate)

if __name__ == "__main__":
    # Example usage if run without arguments
    if len(sys.argv) == 1:
        print("📖 USAGE EXAMPLES:")
        print()
        print("# Analyze entire video (every 5th frame)")
        print("python test_video_analysis.py data/video/Canva_en.mp4")
        print()
        print("# Analyze first 100 frames only")
        print("python test_video_analysis.py data/video/Canva_en.mp4 --max-frames 100")
        print()
        print("# Analyze every frame (slower but more detailed)")
        print("python test_video_analysis.py data/video/Canva_en.mp4 --sample-rate 1")
        print()
        print("# Custom output directory")
        print("python test_video_analysis.py data/video/Canva_en.mp4 --output my_analysis")
        print()
        
        # Try to run a quick test if video exists
        test_video = "data/video/Canva_en.mp4"
        if os.path.exists(test_video):
            print(f"🧪 Running quick test on {test_video} (first 50 frames)...")
            analyze_video_face_angles(test_video, "quick_test_analysis", max_frames=50, sample_rate=5)
        else:
            print(f"⚠️  Test video not found: {test_video}")
            print("Please specify a video file path as argument.")
    else:
        main()
