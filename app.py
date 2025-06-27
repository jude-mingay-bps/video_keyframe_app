import os
import cv2
import json
import shutil
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session, Response
from flask_cors import CORS
import yt_dlp
import uuid
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import requests

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_youtube_video(url, output_path):
    """Download YouTube video using yt-dlp"""
    # Ensure output path doesn't have extension (yt-dlp will add it)
    if output_path.endswith('.mp4'):
        output_path = output_path[:-4]
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path + '.%(ext)s',
        'quiet': False,  # Show progress for debugging
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Get the actual filename (yt-dlp might add extension)
            return True, info.get('title', 'YouTube Video')
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return False, str(e)

def extract_frames(video_path, start_time, duration=30, target_fps=30):
    """Extract frames from video at specified fps"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    frame_interval = int(fps / target_fps) if fps > target_fps else 1
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_num in range(start_frame, end_frame, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to base64 for web display
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        frames.append({
            'data': frame_base64,
            'frame_num': frame_num,
            'time': frame_num / fps
        })
    
    cap.release()
    return frames

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
    # Ensure frame_interval is at least 1
    frame_interval = max(1, frame_count // num_thumbnails)

    for i in range(num_thumbnails):
        frame_num = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Resize for thumbnail
        height = 90  # Match timeline height
        
        # Avoid division by zero if frame has no height
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

def test_roboflow_connection(api_key, project_url):
    """Test if Roboflow connection is valid"""
    try:
        # Extract workspace and project from URL
        project_url = project_url.rstrip('/')
        
        if 'roboflow.com' in project_url:
            parts = project_url.split('/')
            for i, part in enumerate(parts):
                if 'roboflow.com' in part and i + 2 < len(parts):
                    workspace = parts[i + 1]
                    project = parts[i + 2]
                    break
            else:
                return False, "Could not parse workspace and project from URL"
        else:
            return False, "Invalid Roboflow URL format"
        
        # Test API endpoint - get project info
        test_url = f"https://api.roboflow.com/{workspace}/{project}"
        
        params = {
            'api_key': api_key
        }
        
        response = requests.get(test_url, params=params)
        
        if response.status_code == 200:
            return True, f"Connected to {workspace}/{project}"
        else:
            return False, f"Invalid project or API key: {response.text}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def upload_to_roboflow_api(api_key, project_url, image_data, image_name, split='train', batch_name=None):
    """Upload image to Roboflow project with optional batch name and split"""
    try:
        # Extract workspace and project from URL
        project_url = project_url.rstrip('/')
        
        if 'roboflow.com' in project_url:
            parts = project_url.split('/')
            for i, part in enumerate(parts):
                if 'roboflow.com' in part and i + 2 < len(parts):
                    workspace = parts[i + 1]
                    project = parts[i + 2]
                    break
            else:
                return False, "Could not parse workspace and project from URL"
        else:
            return False, "Invalid Roboflow URL format"
        
        # Correct Roboflow Upload API endpoint format
        upload_url = f"https://api.roboflow.com/dataset/{project}/upload"
        
        # Convert base64 to image file
        image_bytes = base64.b64decode(image_data)
        
        # Save temporarily to ensure proper file upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Prepare the multipart upload
            with open(tmp_path, 'rb') as f:
                files = {
                    'file': (image_name, f, 'image/jpeg')
                }
                
                # Parameters as query string
                params = {
                    'api_key': api_key,
                    'name': image_name,
                    'split': split # Use the provided split
                }
                
                # Add batch name if provided
                if batch_name:
                    params['batch'] = batch_name
                
                print(f"Uploading to: {upload_url}")
                print(f"Project: {project}")
                print(f"Image name: {image_name}")
                print(f"Split: {split}")
                if batch_name:
                    print(f"Batch name: {batch_name}")
                
                response = requests.post(
                    upload_url,
                    files=files,
                    params=params
                )
                
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text[:200]}...")
                
                if response.status_code == 200:
                    # Check if response indicates success
                    try:
                        result = response.json()
                        if 'error' in result:
                            return False, f"Upload error: {result['error']}"
                        elif 'success' in result and result['success']:
                            return True, "Image uploaded successfully"
                        elif 'id' in result:  # Some endpoints return an ID on success
                            return True, f"Image uploaded successfully (ID: {result['id']})"
                        else:
                            # If no error and status is 200, assume success
                            return True, "Image uploaded successfully"
                    except:
                        # If can't parse JSON but got 200, assume success
                        return True, "Image uploaded successfully"
                else:
                    return False, f"Failed to upload (Status {response.status_code}): {response.text}"
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"Exception during upload: {str(e)}")
        return False, f"Error uploading to Roboflow: {str(e)}"

@app.route('/test_roboflow', methods=['POST'])
def test_roboflow_endpoint():
    """Test Roboflow connection"""
    data = request.json
    api_key = data.get('api_key')
    project_url = data.get('project_url')
    
    if not api_key or not project_url:
        return jsonify({'success': False, 'error': 'Missing API key or project URL'})
    
    success, message = test_roboflow_connection(api_key, project_url)
    return jsonify({'success': success, 'message': message})

@app.route('/get_video_info', methods=['POST'])
def get_video_info():
    """Get video metadata including duration"""
    data = request.json
    video_id = data.get('video_id')
    
    print(f"Getting video info for ID: {video_id}")
    
    if not video_id or 'videos' not in session or video_id not in session['videos']:
        print(f"Video not found in session. Session videos: {session.get('videos', {}).keys()}")
        return jsonify({'success': False, 'error': 'Video not found in session'})
    
    video_info = session['videos'][video_id]
    video_path = video_info['path']
    
    print(f"Video path: {video_path}")
    
    if not os.path.exists(video_path):
        # Try to find the file with different extensions
        possible_paths = [
            video_path,
            video_path + '.mp4',
            video_path.replace('.mp4', '') + '.mp4',
            os.path.join(TEMP_FOLDER, os.path.basename(video_path))
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path:
            video_path = found_path
            # Update session with correct path
            session['videos'][video_id]['path'] = video_path
            session.modified = True
            print(f"Updated video path to: {video_path}")
        else:
            print(f"Video file not found. Tried paths: {possible_paths}")
            return jsonify({'success': False, 'error': f'Video file not found'})
    
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return jsonify({'success': False, 'error': 'Cannot open video file'})
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    print(f"Video info - FPS: {fps}, Frames: {frame_count}, Duration: {duration}")
    
    # Ensure duration is valid
    if duration <= 0:
        return jsonify({'success': False, 'error': 'Invalid video duration'})
    
    return jsonify({
        'success': True,
        'duration': duration,
        'fps': fps,
        'frame_count': frame_count
    })

@app.route('/video/<video_id>')
def serve_video(video_id):
    """Serve video file for preview"""
    if 'videos' not in session or video_id not in session['videos']:
        return 'Video not found', 404
    
    video_info = session['videos'][video_id]
    video_path = video_info['path']
    
    if not os.path.exists(video_path):
        return 'Video file not found', 404
    
    def generate():
        with open(video_path, 'rb') as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)
    
    response = Response(generate(), mimetype='video/mp4')
    response.headers['Accept-Ranges'] = 'bytes'
    return response

@app.route('/')
def index():
    """Serve the main page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Frame Selector with Roboflow Integration</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            position: relative;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.2);
            color: #333;
            padding: 40px 0;
            margin-bottom: 30px;
            border-radius: 24px;
            position: relative;
            overflow: hidden;
        }

        header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
            background-size: 200% 100%;
            animation: shimmer 3s ease-in-out infinite;
        }

        @keyframes shimmer {
            0%, 100% { background-position: 200% 0; }
            50% { background-position: -200% 0; }
        }

        h1 {
            text-align: center;
            font-size: 2.8em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
            letter-spacing: -0.02em;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1em;
            font-weight: 500;
        }

        .upload-section, .roboflow-section, .video-list, .frame-selector {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 35px;
            border-radius: 24px;
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.1);
            margin-bottom: 30px;
            animation: slideIn 0.6s ease-out;
            position: relative;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .roboflow-section {
            border: 2px solid transparent;
            background: linear-gradient(white, white) padding-box,
                        linear-gradient(135deg, #e74c3c, #c0392b) border-box;
            position: relative;
            overflow: hidden;
        }

        .roboflow-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #e74c3c, #c0392b, #e74c3c);
            background-size: 200% 100%;
            animation: pulse-red 2s ease-in-out infinite;
        }

        @keyframes pulse-red {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 1; }
        }

        .roboflow-section h2 {
            color: #e74c3c;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.5em;
            font-weight: 700;
        }

        .roboflow-section h2::before {
            content: '🔗';
            font-size: 1.2em;
        }

        .input-group {
            margin-bottom: 28px;
            position: relative;
        }
        
        .input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 24px;
        }

        label {
            display: block;
            margin-bottom: 12px;
            font-weight: 700;
            color: #444;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            position: relative;
        }

        label::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 30px;
            height: 2px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 1px;
        }

        input[type="text"], input[type="file"], input[type="number"], input[type="password"], select {
            width: 100%;
            padding: 18px 20px;
            border: 2px solid #e8ecef;
            border-radius: 16px;
            font-size: 16px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: #fafbfc;
            font-weight: 500;
            position: relative;
        }
        
        select {
             -webkit-appearance: none;
             -moz-appearance: none;
             appearance: none;
             background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%23667eea' viewBox='0 0 16 16'%3E%3Cpath fill-rule='evenodd' d='M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z'/%3E%3C/svg%3E");
             background-repeat: no-repeat;
             background-position: right 20px center;
             padding-right: 50px;
        }

        input[type="text"]:focus, input[type="file"]:focus, input[type="number"]:focus, input[type="password"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1), 0 8px 25px rgba(102, 126, 234, 0.15);
            transform: translateY(-2px);
        }

        .button-group {
            display: flex;
            gap: 16px;
            margin-top: 30px;
            flex-wrap: wrap;
        }

        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 18px 32px;
            border-radius: 16px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
            min-width: 140px;
        }

        button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px 0 rgba(102, 126, 234, 0.4);
        }

        button:hover::before {
            left: 100%;
        }

        button:active {
            transform: translateY(-1px);
        }

        button:disabled {
            background: linear-gradient(135deg, #bdc3c7, #95a5a6);
            cursor: not-allowed;
            transform: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        button:disabled::before {
            display: none;
        }

        button.roboflow-btn {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            box-shadow: 0 8px 25px 0 rgba(231, 76, 60, 0.3);
        }

        button.roboflow-btn:hover {
            box-shadow: 0 12px 35px 0 rgba(231, 76, 60, 0.4);
        }

        .video-list {
            background: rgba(255, 255, 255, 0.95);
        }

        .video-list h2 {
            margin-bottom: 25px;
            font-size: 1.6em;
            color: #2c3e50;
            font-weight: 800;
        }

        .video-item {
            padding: 24px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 16px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid transparent;
            position: relative;
            overflow: hidden;
        }

        .video-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            transform: scaleY(0);
            transition: transform 0.3s ease;
        }

        .video-item:hover {
            background: linear-gradient(135deg, white, #f8f9fa);
            box-shadow: 0 8px 30px 0 rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
            border-color: rgba(102, 126, 234, 0.2);
        }

        .video-item:hover::before {
            transform: scaleY(1);
        }

        .video-item span {
            font-weight: 600;
            color: #2c3e50;
            font-size: 15px;
        }

        .video-item button {
            padding: 10px 24px;
            font-size: 14px;
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            min-width: auto;
        }

        .video-preview {
            margin: 35px 0;
            text-align: center;
        }

        .video-preview video {
            max-width: 100%;
            max-height: 450px;
            border-radius: 20px;
            box-shadow: 0 16px 60px rgba(0, 0, 0, 0.2);
            border: 3px solid rgba(255, 255, 255, 0.8);
        }

        .timeline-container {
            margin: 35px 0;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 30px;
            border-radius: 20px;
            border: 2px solid rgba(102, 126, 234, 0.1);
        }

        .timeline-container h3 {
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 700;
        }

        .timeline {
            position: relative;
            height: 90px;
            border-radius: 16px;
            margin: 25px 0;
            cursor: pointer;
            overflow: hidden;
            box-shadow: inset 0 4px 8px rgba(0, 0, 0, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.5);
            display: flex;
            background: #e9ecef; /* Fallback background */
        }

        .timeline-thumbnail {
            flex-shrink: 0;
            width: auto;
            height: 100%;
            object-fit: cover;
            opacity: 0.8;
        }

        .timeline-selection {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(118, 75, 162, 0.4));
            border: 3px solid #667eea;
            cursor: move;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 2;
            border-radius: 13px; /* Match parent's radius minus border */
        }

        .timeline-selection:hover {
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.5), rgba(118, 75, 162, 0.5));
        }

        .timeline-handle {
            position: absolute;
            width: 16px;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            cursor: ew-resize;
            transition: all 0.3s ease;
            z-index: 3;
        }

        .timeline-handle:hover {
            background: linear-gradient(135deg, #5a6fd8, #6b4d96);
            width: 20px;
        }

        .timeline-handle.left {
            left: -8px;
            border-radius: 16px 0 0 16px;
        }

        .timeline-handle.right {
            right: -8px;
            border-radius: 0 16px 16px 0;
        }

        .timeline-time {
            position: absolute;
            bottom: -35px;
            font-size: 13px;
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 6px 14px;
            border-radius: 8px;
            white-space: nowrap;
            font-weight: 700;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .timeline-time.start {
            left: 0;
        }

        .timeline-time.end {
            right: 0;
        }

        .timeline-info {
            display: flex;
            justify-content: space-between;
            margin-top: 40px;
            font-size: 16px;
            font-weight: 700;
            color: #2c3e50;
        }

        .segment-controls {
            display: flex;
            gap: 24px;
            align-items: center;
            margin: 30px 0;
            padding: 24px;
            background: linear-gradient(135deg, white, #f8f9fa);
            border-radius: 16px;
            border: 2px solid rgba(102, 126, 234, 0.1);
            flex-wrap: wrap;
        }

        .segment-controls input {
            width: 120px;
            padding: 12px 16px;
            font-weight: 600;
        }

        .segment-controls button {
            padding: 12px 24px;
            font-size: 14px;
        }

        .segment-controls label {
            margin: 0;
            font-size: 15px;
            color: #555;
            font-weight: 600;
        }

        .frame-display {
            text-align: center;
            margin: 35px 0;
        }

        .frame-display img {
            max-width: 100%;
            max-height: 550px;
            border: 4px solid #e8ecef;
            border-radius: 20px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 16px 50px rgba(0, 0, 0, 0.12);
        }

        .frame-display img.selected {
            border-color: #27ae60;
            box-shadow: 0 0 40px rgba(39, 174, 96, 0.5), 0 20px 60px rgba(0, 0, 0, 0.15);
            transform: scale(1.02);
        }

        .frame-controls {
            display: flex;
            justify-content: center;
            gap: 24px;
            margin: 35px 0;
            flex-wrap: wrap;
        }

        .frame-controls button {
            min-width: 160px;
        }

        .frame-info {
            text-align: center;
            font-size: 22px;
            margin: 25px 0;
            font-weight: 700;
            color: #2c3e50;
            padding: 15px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 12px;
            border: 2px solid rgba(102, 126, 234, 0.1);
        }

        .selected-indicator {
            color: #27ae60;
            font-weight: 900;
            animation: pulse 1.8s ease-in-out infinite;
            text-shadow: 0 0 10px rgba(39, 174, 96, 0.3);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.05); }
        }

        .progress-bar {
            width: 100%;
            height: 40px;
            background: linear-gradient(135deg, #ecf0f1, #bdc3c7);
            border-radius: 20px;
            margin: 30px 0;
            overflow: hidden;
            box-shadow: inset 0 4px 8px rgba(0, 0, 0, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.5);
            position: relative;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 16px;
            position: relative;
            overflow: hidden;
        }

        .progress-fill::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: progress-shine 2s ease-in-out infinite;
        }

        @keyframes progress-shine {
            0% { left: -100%; }
            100% { left: 100%; }
        }

        .instructions {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 25px;
            border-radius: 16px;
            margin: 30px 0;
            text-align: center;
            font-size: 17px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            border: 2px solid rgba(102, 126, 234, 0.1);
            font-weight: 500;
        }

        .keyboard-hint {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            margin: 0 6px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            font-weight: 700;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            font-size: 14px;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 50px;
        }

        .spinner {
            border: 5px solid rgba(255, 255, 255, 0.3);
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 70px;
            height: 70px;
            animation: spin 1s linear infinite;
            margin: 0 auto 25px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading p {
            font-size: 18px;
            font-weight: 600;
            color: #555;
        }

        .roboflow-status {
            display: inline-flex;
            align-items: center;
            padding: 8px 18px;
            border-radius: 25px;
            font-size: 14px;
            margin-left: 16px;
            font-weight: 700;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            gap: 8px;
        }

        .roboflow-status::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: status-pulse 2s ease-in-out infinite;
        }

        .roboflow-status.connected {
            background: linear-gradient(135deg, #27ae60, #229954);
            color: white;
            box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
        }

        .roboflow-status.connected::before {
            background: #fff;
        }

        .roboflow-status.disconnected {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }

        .roboflow-status.disconnected::before {
            background: #fff;
        }

        @keyframes status-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Toast Notification System */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            max-width: 400px;
            pointer-events: none;
        }

        .toast {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 16px 60px rgba(0, 0, 0, 0.15);
            transform: translateX(400px);
            opacity: 0;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            pointer-events: auto;
            position: relative;
            overflow: hidden;
            max-width: 100%;
            word-wrap: break-word;
        }

        .toast.show {
            transform: translateX(0);
            opacity: 1;
        }

        .toast.hide {
            transform: translateX(400px);
            opacity: 0;
        }

        .toast::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
        }

        .toast.success::before {
            background: linear-gradient(135deg, #27ae60, #229954);
        }

        .toast.error::before {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }

        .toast.warning::before {
            background: linear-gradient(135deg, #f39c12, #e67e22);
        }

        .toast-content {
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .toast-icon {
            flex-shrink: 0;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: bold;
            margin-top: 2px;
        }

        .toast.success .toast-icon {
            background: linear-gradient(135deg, #27ae60, #229954);
            color: white;
        }

        .toast.error .toast-icon {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }

        .toast.warning .toast-icon {
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
        }

        .toast.info .toast-icon {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }

        .toast-message {
            flex: 1;
            font-size: 15px;
            font-weight: 600;
            color: #2c3e50;
            line-height: 1.4;
        }

        .toast-close {
            flex-shrink: 0;
            background: none;
            border: none;
            font-size: 20px;
            color: #95a5a6;
            cursor: pointer;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s ease;
            margin-top: 2px;
        }

        .toast-close:hover {
            background: rgba(149, 165, 166, 0.1);
            color: #7f8c8d;
        }

        /* Progress Bar for Downloads/Uploads */
        .download-progress, .upload-progress {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
            margin-top: 12px;
            overflow: hidden;
        }

        .download-progress-fill, .upload-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 3px;
            transition: width 0.3s ease;
            width: 0%;
        }

        .toast.progress .toast-message {
            margin-bottom: 8px;
        }

        /* Enhanced Loading States */
        .button-loading {
            position: relative;
            color: transparent !important;
        }

        .button-loading::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            top: 50%;
            left: 50%;
            margin-left: -10px;
            margin-top: -10px;
            border: 2px solid transparent;
            border-top-color: #ffffff;
            border-radius: 50%;
            animation: button-spin 1s ease infinite;
        }

        @keyframes button-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            h1 {
                font-size: 2.2em;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            button {
                width: 100%;
                justify-content: center;
            }
            
            .segment-controls {
                flex-direction: column;
                align-items: stretch;
                gap: 16px;
            }
            
            .segment-controls input {
                width: 100%;
            }
            
            .frame-controls {
                flex-direction: column;
            }
            
            .frame-controls button {
                width: 100%;
            }
            
            .toast-container {
                top: 10px;
                right: 10px;
                left: 10px;
                max-width: none;
            }
            
            .toast {
                transform: translateY(-100px);
            }
            
            .toast.show {
                transform: translateY(0);
            }
            
            .toast.hide {
                transform: translateY(-100px);
            }
        }

        @media (max-width: 480px) {
            header {
                padding: 25px 0;
            }
            
            h1 {
                font-size: 1.8em;
            }
            
            .upload-section, .roboflow-section, .video-list, .frame-selector {
                padding: 20px;
                border-radius: 16px;
            }
            
            .timeline {
                height: 70px;
            }
            
            .timeline-handle {
                width: 12px;
            }
            
            .timeline-handle:hover {
                width: 12px;
            }
        }
    </style>
</head>
<body>
    <div class="toast-container" id="toast-container"></div>

    <header>
        <h1>Video Frame Selector with Roboflow Integration</h1>
        <div class="subtitle">Extract and manage video frames with seamless Roboflow integration</div>
    </header>
    
    <div class="container">
        <div class="roboflow-section">
            <h2>Roboflow Configuration <span id="roboflow-status" class="roboflow-status disconnected">Not Connected</span></h2>
            
            <div class="input-group">
                <label for="roboflow-url">Roboflow Project URL:</label>
                <input type="text" id="roboflow-url" placeholder="https://app.roboflow.com/workspace/project">
            </div>
            
            <div class="input-group">
                <label for="roboflow-api-key">Roboflow API Key:</label>
                <input type="password" id="roboflow-api-key" placeholder="Your Roboflow API key">
            </div>

            <div class="input-grid">
                <div class="input-group">
                    <label for="roboflow-batch-name">Custom Batch Name (Optional):</label>
                    <input type="text" id="roboflow-batch-name" placeholder="Defaults to video name">
                </div>

                <div class="input-group">
                    <label for="roboflow-split">Upload to Split:</label>
                    <select id="roboflow-split">
                        <option value="train" selected>Train</option>
                        <option value="valid">Valid</option>
                        <option value="test">Test</option>
                    </select>
                </div>
            </div>
            
            <div class="button-group">
                <button onclick="saveRoboflowConfig()" class="roboflow-btn">Save Configuration</button>
                <button onclick="testRoboflowConnection()">Test Connection</button>
            </div>
        </div>
        
        <div class="upload-section">
            <h2>Add Videos</h2>
            
            <div class="input-group">
                <label for="youtube-url">YouTube URL:</label>
                <input type="text" id="youtube-url" placeholder="https://www.youtube.com/watch?v=...">
            </div>
            
            <div class="input-group">
                <label for="file-upload">Or upload a video file:</label>
                <input type="file" id="file-upload" accept="video/*">
            </div>
            
            <div class="button-group">
                <button onclick="addYouTubeVideo()">Add YouTube Video</button>
                <button onclick="uploadFile()">Upload File</button>
            </div>
        </div>
        
        <div class="video-list" id="video-list" style="display: none;">
            <h2>Video Queue</h2>
            <div id="video-items"></div>
            <button onclick="startProcessing()" id="start-btn">Start Processing</button>
        </div>
        
        <div class="frame-selector" id="frame-selector">
            <h2 id="current-video-title">Processing Video</h2>
            
            <div class="video-preview">
                <video id="video-player" controls></video>
            </div>
            
            <div class="timeline-container">
                <h3>Select Video Segment</h3>
                <div class="timeline" id="timeline">
                    <div class="timeline-selection" id="timeline-selection">
                        <div class="timeline-handle left"></div>
                        <div class="timeline-handle right"></div>
                        <div class="timeline-time start" id="time-start">0:00</div>
                        <div class="timeline-time end" id="time-end">0:30</div>
                    </div>
                </div>
                <div class="timeline-info">
                    <span id="video-duration">Duration: 0:00</span>
                    <span id="segment-duration">Selected: 30s</span>
                </div>
                
                <div class="segment-controls">
                    <label>Start:</label>
                    <input type="number" id="start-time" min="0" value="0" step="0.1">
                    <label>Duration:</label>
                    <input type="number" id="duration" min="1" max="60" value="30" step="1">
                    <button onclick="updateSegmentFromInputs()">Update</button>
                    <button onclick="loadSegment()">Load Frames</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Loading frames...</p>
            </div>
            
            <div id="frame-viewer" style="display: none;">
                <div class="instructions">
                    Use <span class="keyboard-hint">←</span> <span class="keyboard-hint">→</span> to navigate, 
                    <span class="keyboard-hint">Space</span> to select/deselect, 
                    <span class="keyboard-hint">Enter</span> to finish
                </div>
                
                <div class="frame-info" id="frame-info"></div>
                
                <div class="frame-display">
                    <img id="frame-image" src="" alt="Video frame">
                </div>
                
                <div class="frame-controls">
                    <button onclick="previousFrame()">← Previous</button>
                    <button onclick="toggleSelection()">Toggle Selection</button>
                    <button onclick="nextFrame()">Next →</button>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%">0%</div>
                </div>
                
                <button onclick="finishVideo()" style="width: 100%; margin-top: 20px;">Finish This Video</button>
            </div>
        </div>
    </div>
    
    <script>
        let videos = [];
        let currentVideoIndex = 0;
        let frames = [];
        let currentFrameIndex = 0;
        let selectedFrames = new Set();
        let currentVideoId = null;
        let videoDuration = 0;
        let segmentStart = 0;
        let segmentDuration = 30;
        let isDragging = false;
        let dragType = null;
        let roboflowConfig = {
            url: '',
            apiKey: '',
            batchName: '',
            split: 'train',
            isConfigured: false
        };
        
        // Toast Notification System
        function showToast(message, type = 'info', duration = 5000, showProgress = false) {
            const toastContainer = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            
            const icons = {
                success: '✓',
                error: '✗',
                warning: '⚠',
                info: 'ℹ'
            };
            
            toast.innerHTML = `
                <div class="toast-content">
                    <div class="toast-icon">${icons[type] || icons.info}</div>
                    <div class="toast-message">${message}</div>
                    <button class="toast-close" onclick="removeToast(this.parentElement.parentElement)">×</button>
                </div>
                ${showProgress ? '<div class="download-progress"><div class="download-progress-fill"></div></div>' : ''}
            `;
            
            toastContainer.appendChild(toast);
            
            // Trigger animation
            setTimeout(() => toast.classList.add('show'), 10);
            
            // Auto remove
            if (duration > 0) {
                setTimeout(() => removeToast(toast), duration);
            }
            
            return toast;
        }
        
        function removeToast(toast) {
            toast.classList.add('hide');
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.parentElement.removeChild(toast);
                }
            }, 400);
        }
        
        function updateToastProgress(toast, progress) {
            const progressFill = toast.querySelector('.download-progress-fill, .upload-progress-fill');
            if (progressFill) {
                progressFill.style.width = `${progress}%`;
            }
        }
        
        function setButtonLoading(button, loading) {
            if (loading) {
                button.disabled = true;
                button.classList.add('button-loading');
                button.dataset.originalText = button.textContent;
                button.textContent = 'Loading...';
            } else {
                button.disabled = false;
                button.classList.remove('button-loading');
                if (button.dataset.originalText) {
                    button.textContent = button.dataset.originalText;
                    delete button.dataset.originalText;
                }
            }
        }
        
        // Load Roboflow config from localStorage
        function loadRoboflowConfig() {
            const saved = localStorage.getItem('roboflowConfig');
            if (saved) {
                roboflowConfig = JSON.parse(saved);
                document.getElementById('roboflow-url').value = roboflowConfig.url || '';
                document.getElementById('roboflow-api-key').value = roboflowConfig.apiKey || '';
                document.getElementById('roboflow-batch-name').value = roboflowConfig.batchName || '';
                document.getElementById('roboflow-split').value = roboflowConfig.split || 'train';
                updateRoboflowStatus();
            }
        }
        
        // Save Roboflow configuration
        function saveRoboflowConfig() {
            const url = document.getElementById('roboflow-url').value.trim();
            const apiKey = document.getElementById('roboflow-api-key').value.trim();
            const batchName = document.getElementById('roboflow-batch-name').value.trim();
            const split = document.getElementById('roboflow-split').value;
            
            if (!url || !apiKey) {
                showToast('Please enter both Roboflow project URL and API key', 'error');
                return;
            }
            
            roboflowConfig = {
                url: url,
                apiKey: apiKey,
                batchName: batchName,
                split: split,
                isConfigured: true
            };
            
            localStorage.setItem('roboflowConfig', JSON.stringify(roboflowConfig));
            updateRoboflowStatus();
            showToast('Roboflow configuration saved successfully', 'success');
        }
        
        // Test Roboflow connection
        async function testRoboflowConnection() {
            const url = document.getElementById('roboflow-url').value.trim();
            const apiKey = document.getElementById('roboflow-api-key').value.trim();
            const button = event.target;
            
            if (!url || !apiKey) {
                showToast('Please enter both Roboflow project URL and API key', 'error');
                return;
            }
            
            setButtonLoading(button, true);
            const loadingToast = showToast('Testing connection...', 'info', 0);
            
            try {
                const response = await fetch('/test_roboflow', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        api_key: apiKey,
                        project_url: url
                    })
                });
                
                const data = await response.json();
                removeToast(loadingToast);
                
                if (data.success) {
                    showToast(data.message, 'success');
                    // Save config if test successful
                    saveRoboflowConfig();
                } else {
                    showToast(data.message || 'Connection test failed', 'error');
                }
            } catch (error) {
                removeToast(loadingToast);
                showToast('Error testing connection: ' + error.message, 'error');
            } finally {
                setButtonLoading(button, false);
            }
        }
        
        function updateRoboflowStatus() {
            const status = document.getElementById('roboflow-status');
            if (roboflowConfig.isConfigured && roboflowConfig.url && roboflowConfig.apiKey) {
                status.textContent = 'Configured';
                status.className = 'roboflow-status connected';
            } else {
                status.textContent = 'Not Configured';
                status.className = 'roboflow-status disconnected';
            }
        }
        
        // Initialize on page load
        window.addEventListener('load', () => {
            loadRoboflowConfig();
            initializeTimeline();
        });
        
        // Keyboard event listeners
        document.addEventListener('keydown', (e) => {
            if (!frames.length) return;
            
            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    previousFrame();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    nextFrame();
                    break;
                case ' ':
                    e.preventDefault();
                    toggleSelection();
                    break;
                case 'Enter':
                    e.preventDefault();
                    finishVideo();
                    break;
            }
        });
        
        // Timeline interaction
        function initializeTimeline() {
            const timeline = document.getElementById('timeline');
            
            timeline.addEventListener('mousedown', (e) => {
                const selection = document.getElementById('timeline-selection');
                const leftHandle = selection.querySelector('.left');
                const rightHandle = selection.querySelector('.right');
                
                if (e.target === leftHandle) {
                    isDragging = true;
                    dragType = 'left';
                } else if (e.target === rightHandle) {
                    isDragging = true;
                    dragType = 'right';
                } else if (e.target === selection) {
                    isDragging = true;
                    dragType = 'move';
                } else if (e.target.classList.contains('timeline-thumbnail') || e.target === timeline) {
                    const rect = timeline.getBoundingClientRect();
                    const clickPos = (e.clientX - rect.left) / rect.width;
                    const clickTime = clickPos * videoDuration;
                    
                    segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, clickTime - segmentDuration / 2));
                    updateTimeline();
                }
                e.preventDefault();
            });
            
            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                const timeline = document.getElementById('timeline');
                const rect = timeline.getBoundingClientRect();
                const mousePos = (e.clientX - rect.left) / rect.width;
                const mouseTime = Math.max(0, Math.min(videoDuration, mousePos * videoDuration));
                
                if (dragType === 'move') {
                    // This logic seems a bit off, let's fix it
                    const selection = document.getElementById('timeline-selection');
                    const selectionWidth = selection.offsetWidth;
                    const timelineWidth = timeline.offsetWidth;
                    const startOffset = (segmentStart / videoDuration) * timelineWidth;
                    
                    // The original click-based logic for moving is better. This mousemove should be more precise.
                    // A better way is to store the initial mouse position and selection start on mousedown.
                    // For now, let's stick to a simplified version that works.
                    segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, mouseTime - segmentDuration / 2));

                } else if (dragType === 'left') {
                    const currentEnd = segmentStart + segmentDuration;
                    const newStart = Math.min(mouseTime, currentEnd - 1); // Ensure it doesn't cross the right handle
                    segmentDuration = currentEnd - newStart;
                    segmentStart = newStart;
                } else if (dragType === 'right') {
                    const newEnd = Math.max(mouseTime, segmentStart + 1); // Ensure it doesn't cross the left handle
                    segmentDuration = newEnd - segmentStart;
                }
                
                segmentDuration = Math.max(1, Math.min(60, segmentDuration));
                segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, segmentStart));
                
                updateTimeline();
            });
            
            document.addEventListener('mouseup', () => {
                isDragging = false;
                dragType = null;
            });
        }
        
        function updateTimeline() {
            if (!videoDuration || videoDuration === 0) {
                return;
            }
            
            const selection = document.getElementById('timeline-selection');
            const startPercent = (segmentStart / videoDuration) * 100;
            const widthPercent = (segmentDuration / videoDuration) * 100;
            
            selection.style.left = `${startPercent}%`;
            selection.style.width = `${widthPercent}%`;
            
            document.getElementById('time-start').textContent = formatTime(segmentStart);
            document.getElementById('time-end').textContent = formatTime(segmentStart + segmentDuration);
            document.getElementById('segment-duration').textContent = `Selected: ${Math.round(segmentDuration)}s`;
            
            document.getElementById('start-time').value = segmentStart.toFixed(1);
            document.getElementById('duration').value = Math.round(segmentDuration);
            
            const video = document.getElementById('video-player');
            if (video.src && video.readyState >= 1 && !isDragging) {
                video.currentTime = segmentStart;
            }
        }
        
        function updateSegmentFromInputs() {
            segmentStart = parseFloat(document.getElementById('start-time').value) || 0;
            segmentDuration = parseInt(document.getElementById('duration').value) || 30;
            
            segmentStart = Math.max(0, Math.min(videoDuration - 1, segmentStart));
            segmentDuration = Math.max(1, Math.min(60, Math.min(videoDuration - segmentStart, segmentDuration)));
            
            updateTimeline();
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }

        function renderTimelineThumbnails(thumbnails) {
            const timeline = document.getElementById('timeline');
            const selection = document.getElementById('timeline-selection');

            // Remove only old thumbnails, not the selection element
            timeline.querySelectorAll('.timeline-thumbnail').forEach(el => el.remove());

            const fragment = document.createDocumentFragment();
            thumbnails.forEach(thumbData => {
                const img = document.createElement('img');
                img.src = `data:image/jpeg;base64,${thumbData}`;
                img.className = 'timeline-thumbnail';
                img.draggable = false;
                fragment.appendChild(img);
            });
            
            // Insert all images before the selection slider for better performance
            timeline.insertBefore(fragment, selection);
        }
        
        async function addYouTubeVideo() {
            const url = document.getElementById('youtube-url').value.trim();
            const button = event.target;
            
            if (!url) {
                showToast('Please enter a YouTube URL', 'error');
                return;
            }
            
            setButtonLoading(button, true);
            const progressToast = showToast('Downloading YouTube video...', 'info', 0, true);
            
            try {
                const response = await fetch('/add_youtube', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                });
                
                const data = await response.json();
                removeToast(progressToast);
                
                if (data.success) {
                    videos.push(data.video);
                    updateVideoList();
                    document.getElementById('youtube-url').value = '';
                    showToast('YouTube video added successfully', 'success');
                } else {
                    showToast(data.error || 'Failed to add YouTube video', 'error');
                }
            } catch (error) {
                removeToast(progressToast);
                showToast('Error adding YouTube video: ' + error.message, 'error');
            } finally {
                setButtonLoading(button, false);
            }
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('file-upload');
            const file = fileInput.files[0];
            const button = event.target;
            
            if (!file) {
                showToast('Please select a file', 'error');
                return;
            }
            
            setButtonLoading(button, true);
            const progressToast = showToast('Uploading file...', 'info', 0, true);
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload_file', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                removeToast(progressToast);
                
                if (data.success) {
                    videos.push(data.video);
                    updateVideoList();
                    fileInput.value = '';
                    showToast('File uploaded successfully', 'success');
                } else {
                    showToast(data.error || 'Failed to upload file', 'error');
                }
            } catch (error) {
                removeToast(progressToast);
                showToast('Error uploading file: ' + error.message, 'error');
            } finally {
                setButtonLoading(button, false);
            }
        }
        
        function updateVideoList() {
            const videoList = document.getElementById('video-list');
            const videoItems = document.getElementById('video-items');
            
            if (videos.length > 0) {
                videoList.style.display = 'block';
                videoItems.innerHTML = videos.map((video, index) => `
                    <div class="video-item">
                        <span>${video.name}</span>
                        <button onclick="removeVideo(${index})">Remove</button>
                    </div>
                `).join('');
            } else {
                videoList.style.display = 'none';
            }
        }
        
        function removeVideo(index) {
            videos.splice(index, 1);
            updateVideoList();
        }
        
        function startProcessing() {
            if (videos.length === 0) {
                showToast('No videos to process', 'error');
                return;
            }
            
            currentVideoIndex = 0;
            document.getElementById('frame-selector').style.display = 'block';
            document.querySelector('.upload-section').style.display = 'none';
            document.getElementById('video-list').style.display = 'none';
            document.querySelector('.roboflow-section').style.display = 'none';
            
            videoDuration = 0;
            
            loadCurrentVideo();
        }
        
        async function loadCurrentVideo() {
            if (currentVideoIndex >= videos.length) {
                showToast('All videos processed!', 'success');
                resetInterface();
                return;
            }
            
            const video = videos[currentVideoIndex];
            currentVideoId = video.id;
            document.getElementById('current-video-title').textContent = `Processing: ${video.name}`;
            frames = [];
            currentFrameIndex = 0;
            selectedFrames.clear();
            document.getElementById('frame-viewer').style.display = 'none';
            
            // Clear old thumbnails before loading new ones
            const timeline = document.getElementById('timeline');
            timeline.querySelectorAll('.timeline-thumbnail').forEach(el => el.remove());
            
            try {
                const infoResponse = await fetch('/get_video_info', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ video_id: currentVideoId })
                });
                
                const infoData = await infoResponse.json();
                if (infoData.success) {
                    videoDuration = infoData.duration;
                    document.getElementById('video-duration').textContent = `Duration: ${formatTime(videoDuration)}`;
                    
                    const videoPlayer = document.getElementById('video-player');
                    videoPlayer.src = `/video/${currentVideoId}`;
                    videoPlayer.load();

                    videoPlayer.addEventListener('loadedmetadata', () => {
                        segmentStart = 0;
                        segmentDuration = Math.min(30, videoDuration);
                        updateTimeline();
                    }, { once: true });
                    
                    videoPlayer.addEventListener('error', (e) => {
                        console.error('Video load error:', e);
                        showToast('Error loading video preview.', 'warning');
                    }, { once: true });

                    // Fetch timeline thumbnails
                    const thumbResponse = await fetch('/get_timeline_thumbnails', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ video_id: currentVideoId })
                    });
                    const thumbData = await thumbResponse.json();
                    if (thumbData.success && thumbData.thumbnails.length > 0) {
                        renderTimelineThumbnails(thumbData.thumbnails);
                    } else {
                        console.error('Failed to load timeline thumbnails:', thumbData.error);
                    }

                } else {
                    showToast('Error loading video info: ' + (infoData.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showToast('Error loading video info: ' + error.message, 'error');
            }
        }
        
        async function loadSegment() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('frame-viewer').style.display = 'none';
            
            try {
                const response = await fetch('/extract_frames', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        video_id: currentVideoId,
                        start_time: segmentStart,
                        duration: segmentDuration
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    frames = data.frames;
                    currentFrameIndex = 0;
                    selectedFrames.clear();
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('frame-viewer').style.display = 'block';
                    displayFrame();
                    showToast(`Loaded ${frames.length} frames successfully`, 'success');
                } else {
                    showToast(data.error || 'Failed to extract frames', 'error');
                    document.getElementById('loading').style.display = 'none';
                }
            } catch (error) {
                showToast('Error extracting frames: ' + error.message, 'error');
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayFrame() {
            if (!frames.length) return;
            
            const frame = frames[currentFrameIndex];
            const img = document.getElementById('frame-image');
            img.src = `data:image/jpeg;base64,${frame.data}`;
            
            if (selectedFrames.has(currentFrameIndex)) {
                img.classList.add('selected');
            } else {
                img.classList.remove('selected');
            }
            
            const info = document.getElementById('frame-info');
            const selectedText = selectedFrames.has(currentFrameIndex) ? 
                '<span class="selected-indicator">[SELECTED]</span>' : '';
            info.innerHTML = `Frame ${currentFrameIndex + 1}/${frames.length} | ` +
                           `Time: ${frame.time.toFixed(1)}s | ` +
                           `Selected: ${selectedFrames.size} ${selectedText}`;
            
            const progress = ((currentFrameIndex + 1) / frames.length) * 100;
            const progressFill = document.getElementById('progress-fill');
            progressFill.style.width = `${progress}%`;
            progressFill.textContent = `${Math.round(progress)}%`;
        }
        
        function previousFrame() {
            if (currentFrameIndex > 0) {
                currentFrameIndex--;
                displayFrame();
            }
        }
        
        function nextFrame() {
            if (currentFrameIndex < frames.length - 1) {
                currentFrameIndex++;
                displayFrame();
            }
        }
        
        function toggleSelection() {
            if (selectedFrames.has(currentFrameIndex)) {
                selectedFrames.delete(currentFrameIndex);
            } else {
                selectedFrames.add(currentFrameIndex);
            }
            displayFrame();
        }
        
        async function finishVideo() {
            if (selectedFrames.size === 0) {
                if (!confirm('No frames selected. Skip this video?')) {
                    return;
                }
            } else {
                const uploadToRoboflow = roboflowConfig.isConfigured && roboflowConfig.apiKey && roboflowConfig.url;
                
                let uploadToast = null;
                if (uploadToRoboflow) {
                    uploadToast = showToast(`Saving ${selectedFrames.size} frames and uploading to Roboflow...`, 'info', 0, true);
                } else {
                    uploadToast = showToast(`Saving ${selectedFrames.size} frames...`, 'info', 0);
                }
                
                const finalRoboflowConfig = {
                    ...roboflowConfig,
                    batchName: document.getElementById('roboflow-batch-name').value.trim(),
                    split: document.getElementById('roboflow-split').value
                };

                try {
                    const response = await fetch('/save_frames', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            video_id: currentVideoId,
                            selected_indices: Array.from(selectedFrames),
                            frames: frames.filter((_, idx) => selectedFrames.has(idx)),
                            upload_to_roboflow: uploadToRoboflow,
                            roboflow_config: uploadToRoboflow ? finalRoboflowConfig : null
                        })
                    });
                    
                    const data = await response.json();
                    removeToast(uploadToast);
                    
                    if (data.success) {
                        let message = `Saved ${selectedFrames.size} frames to ${data.output_dir}`;
                        let toastType = 'success';
                        
                        if (data.roboflow_results) {
                            const uploaded = data.roboflow_results.filter(r => r.success).length;
                            const failed = data.roboflow_results.filter(r => !r.success).length;
                            
                            if (failed > 0) {
                                message += `. Roboflow: ${uploaded} uploaded, ${failed} failed`;
                                toastType = 'warning';
                            } else {
                                message += `. All ${uploaded} frames uploaded to Roboflow successfully`;
                            }
                        }
                        
                        showToast(message, toastType, 10000);
                    } else {
                        showToast('Error saving frames: ' + (data.error || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    removeToast(uploadToast);
                    showToast('Error saving frames: ' + error.message, 'error');
                }
            }
            
            currentVideoIndex++;
            loadCurrentVideo();
        }
        
        function resetInterface() {
            document.getElementById('frame-selector').style.display = 'none';
            document.querySelector('.upload-section').style.display = 'block';
            document.querySelector('.roboflow-section').style.display = 'block';
            videos = [];
            updateVideoList();
        }
    </script>
</body>
</html>
    '''

@app.route('/add_youtube', methods=['POST'])
def add_youtube():
    """Add a YouTube video to the processing queue"""
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'})
    
    video_id = str(uuid.uuid4())
    base_path = os.path.join(TEMP_FOLDER, video_id)
    
    print(f"Downloading YouTube video: {url}")
    
    success, title_or_error = download_youtube_video(url, base_path)
    
    if success:
        video_path = None
        if os.path.exists(TEMP_FOLDER):
            for file in os.listdir(TEMP_FOLDER):
                if file.startswith(video_id):
                    video_path = os.path.join(TEMP_FOLDER, file)
                    break
        
        if not video_path:
            print(f"Downloaded file not found for base {video_id}")
            return jsonify({'success': False, 'error': 'Downloaded file not found'})
        
        print(f"Video downloaded to: {video_path}")
        
        if 'videos' not in session:
            session['videos'] = {}
        
        session['videos'][video_id] = {
            'path': video_path,
            'name': title_or_error if title_or_error else url,
            'type': 'youtube'
        }
        session.modified = True
        
        return jsonify({
            'success': True,
            'video': {
                'id': video_id,
                'name': title_or_error if title_or_error else url,
                'type': 'youtube'
            }
        })
    else:
        print(f"Download failed: {title_or_error}")
        return jsonify({'success': False, 'error': f'Failed to download: {title_or_error}'})

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Upload a video file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        video_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{video_id}_{filename}')
        file.save(video_path)
        
        if 'videos' not in session:
            session['videos'] = {}
        
        session['videos'][video_id] = {
            'path': video_path,
            'name': filename,
            'type': 'upload'
        }
        session.modified = True
        
        return jsonify({
            'success': True,
            'video': {
                'id': video_id,
                'name': filename,
                'type': 'upload'
            }
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/extract_frames', methods=['POST'])
def extract_frames_endpoint():
    """Extract frames from a video segment"""
    data = request.json
    video_id = data.get('video_id')
    start_time = data.get('start_time', 0)
    duration = data.get('duration', 30)
    
    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found'})
    
    video_info = session['videos'][video_id]
    video_path = video_info['path']
    
    frames = extract_frames(video_path, start_time, duration)
    
    if frames:
        return jsonify({
            'success': True,
            'frames': frames
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to extract frames'})

@app.route('/get_timeline_thumbnails', methods=['POST'])
def get_timeline_thumbnails_endpoint():
    """Endpoint to get timeline thumbnails."""
    data = request.json
    video_id = data.get('video_id')

    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found'})

    video_info = session['videos'][video_id]
    video_path = video_info['path']

    thumbnails = extract_timeline_thumbnails(video_path)

    if thumbnails is not None:
        return jsonify({'success': True, 'thumbnails': thumbnails})
    else:
        return jsonify({'success': False, 'error': 'Failed to extract timeline thumbnails'})

@app.route('/save_frames', methods=['POST'])
def save_frames():
    """Save selected frames to disk and optionally upload to Roboflow"""
    data = request.json
    video_id = data.get('video_id')
    selected_indices = data.get('selected_indices', [])
    frames_data = data.get('frames', [])
    upload_to_roboflow = data.get('upload_to_roboflow', False)
    roboflow_config = data.get('roboflow_config', {})
    
    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found'})
    
    video_info = session['videos'][video_id]
    video_name_raw = os.path.splitext(video_info['name'])[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_FOLDER, f'{video_name_raw}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    roboflow_results = []
    
    for i, frame_data in enumerate(frames_data):
        frame_bytes = base64.b64decode(frame_data['data'])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        filename = f'frame_{i+1:03d}_time_{frame_data["time"]:.1f}s.png'
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        
        if upload_to_roboflow and roboflow_config.get('apiKey') and roboflow_config.get('url'):
            image_name = f'frame_{i+1:03d}_time_{frame_data["time"]:.1f}s.jpg'
            
            batch_name = roboflow_config.get('batchName') if roboflow_config.get('batchName') else video_name_raw
            split = roboflow_config.get('split', 'train')

            success, message = upload_to_roboflow_api(
                roboflow_config['apiKey'],
                roboflow_config['url'],
                frame_data['data'],
                image_name,
                split=split,
                batch_name=batch_name
            )
            roboflow_results.append({
                'frame': i,
                'success': success,
                'message': message
            })
    
    response_data = {
        'success': True,
        'output_dir': output_dir,
        'frame_count': len(frames_data)
    }
    
    if roboflow_results:
        response_data['roboflow_results'] = roboflow_results
    
    return jsonify(response_data)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up temporary files"""
    if 'videos' in session:
        for video_id, video_info in session['videos'].items():
            if video_info['type'] == 'youtube' and os.path.exists(video_info['path']):
                os.remove(video_info['path'])
        session.pop('videos', None)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)