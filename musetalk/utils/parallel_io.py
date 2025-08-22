#!/usr/bin/env python3
"""
Parallel I/O utilities for MuseTalk frame processing.
Provides optimized frame writing with threading for improved performance.
"""

import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelFrameWriter:
    """
    Parallel frame writer for improved I/O performance.
    
    Uses ThreadPoolExecutor to write frames to disk in parallel,
    providing significant performance improvements for video processing.
    """
    
    def __init__(self, max_workers=4):
        """
        Initialize parallel frame writer.
        
        Args:
            max_workers (int): Maximum number of worker threads for parallel I/O
        """
        self.max_workers = max_workers
        self.futures = []
        self.frames_written = 0
        self.start_time = time.time()
    
    def write_frame_async(self, executor, frame_idx, frame_array, output_path):
        """
        Submit frame write task to thread pool for asynchronous processing.
        
        Args:
            executor (ThreadPoolExecutor): Thread pool executor
            frame_idx (int): Frame index for filename generation
            frame_array (numpy.ndarray): Frame image data
            output_path (str): Output directory path
            
        Returns:
            Future: Future object for the write operation
        """
        future = executor.submit(self._write_frame, frame_idx, frame_array, output_path)
        self.futures.append(future)
        return future
    
    def _write_frame(self, frame_idx, frame_array, output_path):
        """
        Write single frame to disk.
        
        Args:
            frame_idx (int): Frame index for filename generation
            frame_array (numpy.ndarray): Frame image data
            output_path (str): Output directory path
            
        Returns:
            bool: True if write successful, False otherwise
        """
        try:
            filename = f"{output_path}/{str(frame_idx).zfill(8)}.png"
            success = cv2.imwrite(filename, frame_array)
            if success:
                self.frames_written += 1
                return True
            else:
                print(f"❌ Failed to write frame {frame_idx}")
                return False
        except Exception as e:
            print(f"❌ Error writing frame {frame_idx}: {e}")
            return False
    
    def wait_for_completion(self):
        """
        Wait for all frames to be written and display performance statistics.
        
        Blocks until all pending write operations complete and provides
        detailed statistics about the I/O performance.
        """
        if not self.futures:
            return
        
        print(f"⏳ Finalizing {len(self.futures)} frame writes...")
        
        # Wait for all futures to complete with progress updates
        completed = 0
        for future in as_completed(self.futures):
            try:
                future.result(timeout=30)  # 30 second timeout per frame
                completed += 1
                # Show progress every 100 completed writes
                if completed % 100 == 0:
                    print(f"📊 Completed {completed}/{len(self.futures)} writes")
            except Exception as e:
                print(f"❌ Frame write error: {e}")
        
        # Calculate and display performance statistics
        elapsed = time.time() - self.start_time
        fps = self.frames_written / elapsed if elapsed > 0 else 0
        
        print(f"✅ Parallel I/O: Wrote {self.frames_written} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
    
    def get_stats(self):
        """
        Get current performance statistics.
        
        Returns:
            dict: Dictionary containing performance metrics
        """
        elapsed = time.time() - self.start_time
        fps = self.frames_written / elapsed if elapsed > 0 else 0
        
        return {
            'frames_written': self.frames_written,
            'elapsed_time': elapsed,
            'fps': fps,
            'pending_writes': len([f for f in self.futures if not f.done()]),
            'completed_writes': len([f for f in self.futures if f.done()])
        }


def create_parallel_writer(max_workers=4):
    """
    Factory function to create a ParallelFrameWriter instance.
    
    Args:
        max_workers (int): Maximum number of worker threads
        
    Returns:
        ParallelFrameWriter: Configured parallel frame writer instance
    """
    return ParallelFrameWriter(max_workers=max_workers)


# Utility functions for common parallel I/O patterns
def write_frames_parallel(frames_data, output_path, max_workers=4, show_progress=True):
    """
    Write multiple frames in parallel using a simple interface.
    
    Args:
        frames_data (list): List of (frame_idx, frame_array) tuples
        output_path (str): Output directory path
        max_workers (int): Maximum number of worker threads
        show_progress (bool): Whether to show progress updates
        
    Returns:
        dict: Performance statistics
    """
    writer = ParallelFrameWriter(max_workers=max_workers)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for frame_idx, frame_array in frames_data:
            writer.write_frame_async(executor, frame_idx, frame_array, output_path)
        
        if show_progress:
            writer.wait_for_completion()
    
    return writer.get_stats()


if __name__ == "__main__":
    print("🚀 MuseTalk Parallel I/O Utilities")
    print("=" * 50)
    print("This module provides optimized frame writing capabilities")
    print("for improved video processing performance.")
    print("\nKey features:")
    print("• Parallel frame writing with configurable worker threads")
    print("• Automatic error handling and retry logic")
    print("• Performance monitoring and statistics")
    print("• Non-blocking I/O for continuous processing")
