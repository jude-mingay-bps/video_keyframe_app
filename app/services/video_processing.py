import cv2
import yt_dlp
import base64
import threading
import time
import numpy as np
import gc
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.video.io.VideoFileClip import VideoFileClip
from .yolo_handler import predict_on_frame
from .movenet_handler import predict_pose_on_frame
import multiprocessing
from collections import OrderedDict


# Global cache for annotations with size limits and timestamp tracking
class LRUCache:
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new
                self.cache[key] = value
                self.timestamps[key] = time.time()
                
                # Remove oldest if over limit
                if len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
    
    def __len__(self):
        with self.lock:
            return len(self.cache)
    
    def values(self):
        with self.lock:
            return self.cache.values()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            gc.collect()
    
    def cleanup_old(self, max_age=3600):
        """Remove entries older than max_age seconds"""
        current_time = time.time()
        with self.lock:
            to_remove = []
            for key, timestamp in self.timestamps.items():
                if current_time - timestamp > max_age:
                    to_remove.append(key)
            
            for key in to_remove:
                if key in self.cache:
                    del self.cache[key]
                del self.timestamps[key]
            
            if to_remove:
                gc.collect()

frame_annotations_cache = LRUCache(max_size=200)
pose_annotations_cache = LRUCache(max_size=200)
annotation_lock = threading.Lock()

def clear_annotations_cache():
    """Clear the annotations cache."""
    frame_annotations_cache.clear()
    pose_annotations_cache.clear()

def cleanup_old_annotations():
    """Clean up old annotations from cache."""
    frame_annotations_cache.cleanup_old(max_age=3600)
    pose_annotations_cache.cleanup_old(max_age=3600)

def get_frame_annotations(frame_num):
    """Get annotations for a specific frame from cache."""
    return frame_annotations_cache.get(frame_num)

def set_frame_annotations(frame_num, annotations):
    """Set annotations for a specific frame in cache."""
    frame_annotations_cache.set(frame_num, annotations)

def get_pose_annotations(frame_num):
    """Get pose annotations for a specific frame from cache."""
    return pose_annotations_cache.get(frame_num)

def set_pose_annotations(frame_num, annotations):
    """Set pose annotations for a specific frame in cache."""
    pose_annotations_cache.set(frame_num, annotations)

def process_frame_annotations(frame_data, confidence_threshold=0.25):
    """Process YOLO annotations for a single frame."""
    frame_num = frame_data['frame_num']
    frame_base64 = frame_data['data']
    
    # Check if annotations already exist
    if get_frame_annotations(frame_num) is not None:
        return
    
    # Run YOLO prediction
    annotations, status = predict_on_frame(frame_base64, confidence_threshold)
    
    # Store in cache
    set_frame_annotations(frame_num, {
        'annotations': annotations,
        'status': status,
        'processed': True
    })

def start_background_annotation_processing(frames, confidence_threshold=0.25, max_workers=None):
    """Start background processing of YOLO annotations for all frames."""
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 6)
    
    def process_batch():
        # Clean up old annotations first
        cleanup_old_annotations()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frame processing tasks
            future_to_frame = {
                executor.submit(process_frame_annotations, frame, confidence_threshold): frame
                for frame in frames
            }
            
            # Process completed tasks
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing frame {frame['frame_num']}: {e}")
                    # Store error in cache
                    set_frame_annotations(frame['frame_num'], {
                        'annotations': None,
                        'status': f"Error: {str(e)}",
                        'processed': True
                    })
    
    # Start processing in background thread
    background_thread = threading.Thread(target=process_batch, daemon=True)
    background_thread.start()
    
    return background_thread


def download_youtube_video(url, output_path):
    """Download YouTube video using yt-dlp."""
    if output_path.endswith('.mp4'):
        output_path = output_path[:-4]

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path + '.%(ext)s',
        'quiet': False,
        'no_warnings': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return True, info.get('title', 'YouTube Video')
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return False, str(e)




def extract_single_frame(video_path, frame_pos, fps, max_width=960, quality=50):
    """Extract a single frame from video at specified position."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Resize frame to speed up encoding
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Fast JPEG encoding
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    time_val = frame_pos / fps
    
    return {
        'data': frame_base64,
        'frame_num': frame_pos,
        'time': time_val
    }

def get_optimal_quality(frame_count):
    """Get optimal JPEG quality based on frame count."""
    if frame_count < 50:
        return 75  # High quality for few frames
    elif frame_count < 200:
        return 60  # Medium quality
    else:
        return 45  # Lower quality for many frames

def extract_frames(video_path, start_time, duration=30, target_fps=10, start_background_processing=True, confidence_threshold=0.25):
    """Extract frames from video using parallel processing."""
    try:
        # Clear previous annotations cache
        clear_annotations_cache()
        
        # Use OpenCV for video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Calculate frame positions
        start_frame = int(start_time * fps)
        end_frame = min(int((start_time + duration) * fps), total_frames)
        frame_interval = max(1, int(fps / target_fps))
        
        frame_positions = list(range(start_frame, end_frame, frame_interval))
        
        # Determine optimal quality based on frame count
        quality = get_optimal_quality(len(frame_positions))
        
        # Sequential frame extraction with shared video capture (much faster)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames = []
        try:
            for frame_pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Resize frame to speed up encoding
                height, width = frame.shape[:2]
                if width > 960:
                    scale = 960 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Fast JPEG encoding
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                time_val = frame_pos / fps
                
                frames.append({
                    'data': frame_base64,
                    'frame_num': frame_pos,
                    'time': time_val
                })
        finally:
            cap.release()
        
        print(f"Extracted {len(frames)} frames using sequential processing")
        
        # Start background annotation processing AFTER frames are sent
        if start_background_processing and frames and len(frames) > 0:
            # Delay the background processing slightly
            def delayed_processing():
                time.sleep(0.5)
                print(f"Starting background YOLO processing for {len(frames)} frames...")
                start_background_annotation_processing(frames, confidence_threshold)
            
            threading.Thread(target=delayed_processing, daemon=True).start()
        
        return frames
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None


def extract_timeline_thumbnails(video_path, num_thumbnails=20):
    """Extract a set of thumbnails for the entire video timeline."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        return []

    thumbnails = []
    frame_interval = max(1, frame_count // num_thumbnails)

    for i in range(num_thumbnails):
        frame_num = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        height = 90
        if frame.shape[0] == 0:
            continue

        aspect_ratio = frame.shape[1] / frame.shape[0]
        width = int(height * aspect_ratio)
        resized_frame = cv2.resize(frame, (width, height))

        _, buffer = cv2.imencode('.jpg', resized_frame)
        thumb_base64 = base64.b64encode(buffer).decode('utf-8')
        thumbnails.append(thumb_base64)

    cap.release()
    return thumbnails