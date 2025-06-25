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

def extract_frames(video_path, start_time, duration=30, target_fps=5):
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
    <title>Video Frame Selector</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
        }
        
        h1 {
            text-align: center;
            font-size: 2em;
        }
        
        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        input[type="text"], input[type="file"], input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus, input[type="file"]:focus, input[type="number"]:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        button:hover {
            background-color: #2980b9;
        }
        
        button:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }
        
        .video-list {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .video-item {
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 5px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .video-item:hover {
            background-color: #f8f9fa;
        }
        
        .frame-selector {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: none;
        }
        
        .video-preview {
            margin: 20px 0;
            text-align: center;
        }
        
        .video-preview video {
            max-width: 100%;
            max-height: 400px;
            border-radius: 5px;
        }
        
        .timeline-container {
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        
        .timeline {
            position: relative;
            height: 80px;
            background: #e9ecef;
            border-radius: 5px;
            margin: 20px 0;
            cursor: pointer;
            overflow: hidden;
        }
        
        .timeline-selection {
            position: absolute;
            height: 100%;
            background: rgba(52, 152, 219, 0.3);
            border: 2px solid #3498db;
            border-radius: 5px;
            cursor: move;
        }
        
        .timeline-handle {
            position: absolute;
            width: 10px;
            height: 100%;
            background: #3498db;
            cursor: ew-resize;
        }
        
        .timeline-handle.left {
            left: -5px;
            border-radius: 5px 0 0 5px;
        }
        
        .timeline-handle.right {
            right: -5px;
            border-radius: 0 5px 5px 0;
        }
        
        .timeline-time {
            position: absolute;
            bottom: -25px;
            font-size: 12px;
            background: #34495e;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            white-space: nowrap;
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
            margin-top: 30px;
            font-size: 14px;
        }
        
        .segment-controls {
            display: flex;
            gap: 15px;
            align-items: center;
            margin: 20px 0;
        }
        
        .segment-controls input {
            width: 100px;
        }
        
        .frame-display {
            text-align: center;
            margin: 20px 0;
        }
        
        .frame-display img {
            max-width: 100%;
            max-height: 500px;
            border: 3px solid #ddd;
            border-radius: 5px;
        }
        
        .frame-display img.selected {
            border-color: #27ae60;
            box-shadow: 0 0 20px rgba(39, 174, 96, 0.3);
        }
        
        .frame-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        
        .frame-info {
            text-align: center;
            font-size: 18px;
            margin: 15px 0;
        }
        
        .selected-indicator {
            color: #27ae60;
            font-weight: bold;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background-color: #ecf0f1;
            border-radius: 15px;
            margin: 20px 0;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background-color: #3498db;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .instructions {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: center;
        }
        
        .keyboard-hint {
            display: inline-block;
            background-color: #34495e;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 0 5px;
            font-family: monospace;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            color: #e74c3c;
            padding: 10px;
            background-color: #fee;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .success {
            color: #27ae60;
            padding: 10px;
            background-color: #efe;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>Video Frame Selector</h1>
    </header>
    
    <div class="container">
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
        
        <div id="messages"></div>
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
            const selection = document.getElementById('timeline-selection');
            const leftHandle = selection.querySelector('.left');
            const rightHandle = selection.querySelector('.right');
            
            // Click on timeline to move selection
            timeline.addEventListener('click', (e) => {
                if (e.target === timeline) {
                    const rect = timeline.getBoundingClientRect();
                    const clickPos = (e.clientX - rect.left) / rect.width;
                    const clickTime = clickPos * videoDuration;
                    
                    // Center the selection on click, maintaining duration
                    segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, clickTime - segmentDuration / 2));
                    updateTimeline();
                }
            });
            
            // Drag entire selection
            selection.addEventListener('mousedown', (e) => {
                if (e.target === selection) {
                    isDragging = true;
                    dragType = 'move';
                    e.preventDefault();
                }
            });
            
            // Drag left handle
            leftHandle.addEventListener('mousedown', (e) => {
                isDragging = true;
                dragType = 'left';
                e.preventDefault();
                e.stopPropagation();
            });
            
            // Drag right handle
            rightHandle.addEventListener('mousedown', (e) => {
                isDragging = true;
                dragType = 'right';
                e.preventDefault();
                e.stopPropagation();
            });
            
            // Mouse move for dragging
            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                
                const timeline = document.getElementById('timeline');
                const rect = timeline.getBoundingClientRect();
                const mousePos = (e.clientX - rect.left) / rect.width;
                const mouseTime = mousePos * videoDuration;
                
                if (dragType === 'move') {
                    segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, mouseTime - segmentDuration / 2));
                } else if (dragType === 'left') {
                    const newStart = Math.max(0, Math.min(segmentStart + segmentDuration - 1, mouseTime));
                    segmentDuration = segmentStart + segmentDuration - newStart;
                    segmentStart = newStart;
                } else if (dragType === 'right') {
                    const newEnd = Math.max(segmentStart + 1, Math.min(videoDuration, mouseTime));
                    segmentDuration = newEnd - segmentStart;
                }
                
                // Limit duration
                segmentDuration = Math.max(1, Math.min(60, segmentDuration));
                
                updateTimeline();
            });
            
            // Mouse up to stop dragging
            document.addEventListener('mouseup', () => {
                isDragging = false;
                dragType = null;
            });
        }
        
        // Initialize timeline on page load
        window.addEventListener('load', () => {
            initializeTimeline();
        });
        
        function updateTimeline() {
            if (!videoDuration || videoDuration === 0) {
                console.warn('Video duration not set');
                return;
            }
            
            const selection = document.getElementById('timeline-selection');
            const startPercent = (segmentStart / videoDuration) * 100;
            const widthPercent = (segmentDuration / videoDuration) * 100;
            
            selection.style.left = `${startPercent}%`;
            selection.style.width = `${widthPercent}%`;
            
            // Update time displays
            document.getElementById('time-start').textContent = formatTime(segmentStart);
            document.getElementById('time-end').textContent = formatTime(segmentStart + segmentDuration);
            document.getElementById('segment-duration').textContent = `Selected: ${Math.round(segmentDuration)}s`;
            
            // Update input fields
            document.getElementById('start-time').value = segmentStart.toFixed(1);
            document.getElementById('duration').value = Math.round(segmentDuration);
            
            // Update video player time
            const video = document.getElementById('video-player');
            if (video.src && video.readyState >= 2) {
                video.currentTime = segmentStart;
            }
        }
        
        function updateSegmentFromInputs() {
            segmentStart = parseFloat(document.getElementById('start-time').value) || 0;
            segmentDuration = parseInt(document.getElementById('duration').value) || 30;
            
            // Validate
            segmentStart = Math.max(0, Math.min(videoDuration - 1, segmentStart));
            segmentDuration = Math.max(1, Math.min(60, Math.min(videoDuration - segmentStart, segmentDuration)));
            
            updateTimeline();
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
        
        function showMessage(message, type = 'info') {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = type === 'error' ? 'error' : 'success';
            messageDiv.textContent = message;
            messagesDiv.appendChild(messageDiv);
            
            setTimeout(() => {
                messageDiv.remove();
            }, 5000);
        }
        
        async function addYouTubeVideo() {
            const url = document.getElementById('youtube-url').value.trim();
            if (!url) {
                showMessage('Please enter a YouTube URL', 'error');
                return;
            }
            
            showMessage('Downloading YouTube video... This may take a moment.');
            
            try {
                const response = await fetch('/add_youtube', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                });
                
                const data = await response.json();
                if (data.success) {
                    videos.push(data.video);
                    updateVideoList();
                    document.getElementById('youtube-url').value = '';
                    showMessage('YouTube video added successfully');
                } else {
                    showMessage(data.error || 'Failed to add YouTube video', 'error');
                }
            } catch (error) {
                showMessage('Error adding YouTube video: ' + error.message, 'error');
            }
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('file-upload');
            const file = fileInput.files[0];
            
            if (!file) {
                showMessage('Please select a file', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload_file', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.success) {
                    videos.push(data.video);
                    updateVideoList();
                    fileInput.value = '';
                    showMessage('File uploaded successfully');
                } else {
                    showMessage(data.error || 'Failed to upload file', 'error');
                }
            } catch (error) {
                showMessage('Error uploading file', 'error');
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
                showMessage('No videos to process', 'error');
                return;
            }
            
            currentVideoIndex = 0;
            document.getElementById('frame-selector').style.display = 'block';
            document.querySelector('.upload-section').style.display = 'none';
            document.getElementById('video-list').style.display = 'none';
            
            // Reset video duration to ensure proper initialization
            videoDuration = 0;
            
            loadCurrentVideo();
        }
        
        async function loadCurrentVideo() {
            if (currentVideoIndex >= videos.length) {
                showMessage('All videos processed!');
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
            
            // Load video metadata
            try {
                const response = await fetch('/get_video_info', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ video_id: currentVideoId })
                });
                
                const data = await response.json();
                if (data.success) {
                    videoDuration = data.duration;
                    document.getElementById('video-duration').textContent = `Duration: ${formatTime(videoDuration)}`;
                    
                    // Set video player source
                    const videoPlayer = document.getElementById('video-player');
                    videoPlayer.src = `/video/${currentVideoId}`;
                    
                    // Wait for video to load metadata
                    videoPlayer.addEventListener('loadedmetadata', () => {
                        // Reset segment selection
                        segmentStart = 0;
                        segmentDuration = Math.min(30, videoDuration);
                        updateTimeline();
                    }, { once: true });
                    
                    // Handle video load error
                    videoPlayer.addEventListener('error', (e) => {
                        console.error('Video load error:', e);
                        showMessage('Error loading video preview. You can still process frames.', 'error');
                    }, { once: true });
                } else {
                    showMessage('Error loading video info: ' + (data.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showMessage('Error loading video info: ' + error.message, 'error');
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
                } else {
                    showMessage(data.error || 'Failed to extract frames', 'error');
                    document.getElementById('loading').style.display = 'none';
                }
            } catch (error) {
                showMessage('Error extracting frames', 'error');
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
            
            // Update progress bar
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
                // Save selected frames
                try {
                    const response = await fetch('/save_frames', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            video_id: currentVideoId,
                            selected_indices: Array.from(selectedFrames),
                            frames: frames.filter((_, idx) => selectedFrames.has(idx))
                        })
                    });
                    
                    const data = await response.json();
                    if (data.success) {
                        showMessage(`Saved ${selectedFrames.size} frames to ${data.output_dir}`);
                    } else {
                        showMessage('Error saving frames', 'error');
                    }
                } catch (error) {
                    showMessage('Error saving frames', 'error');
                }
            }
            
            // Move to next video
            currentVideoIndex++;
            loadCurrentVideo();
        }
        
        function resetInterface() {
            document.getElementById('frame-selector').style.display = 'none';
            document.querySelector('.upload-section').style.display = 'block';
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
    
    # Generate unique ID for this video
    video_id = str(uuid.uuid4())
    base_path = os.path.join(TEMP_FOLDER, video_id)
    
    print(f"Downloading YouTube video: {url}")
    
    # Download the video
    success, title_or_error = download_youtube_video(url, base_path)
    
    if success:
        # Find the actual downloaded file
        video_path = None
        possible_files = [
            base_path,
            base_path + '.mp4',
            base_path + '.webm',
            base_path + '.mkv'
        ]
        
        # Also check for files that start with our base name
        if os.path.exists(TEMP_FOLDER):
            for file in os.listdir(TEMP_FOLDER):
                if file.startswith(video_id):
                    video_path = os.path.join(TEMP_FOLDER, file)
                    break
        
        # If not found, check the possible files
        if not video_path:
            for path in possible_files:
                if os.path.exists(path):
                    video_path = path
                    break
        
        if not video_path:
            print(f"Downloaded file not found. Checked: {possible_files}")
            print(f"Files in temp folder: {os.listdir(TEMP_FOLDER)}")
            return jsonify({'success': False, 'error': 'Downloaded file not found'})
        
        print(f"Video downloaded to: {video_path}")
        
        # Store video info in session
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
        # Generate unique ID
        video_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{video_id}_{filename}')
        file.save(video_path)
        
        # Store video info in session
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
    
    # Extract frames
    frames = extract_frames(video_path, start_time, duration)
    
    if frames:
        return jsonify({
            'success': True,
            'frames': frames
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to extract frames'})

@app.route('/save_frames', methods=['POST'])
def save_frames():
    """Save selected frames to disk"""
    data = request.json
    video_id = data.get('video_id')
    selected_indices = data.get('selected_indices', [])
    frames_data = data.get('frames', [])
    
    if not video_id or 'videos' not in session or video_id not in session['videos']:
        return jsonify({'success': False, 'error': 'Video not found'})
    
    video_info = session['videos'][video_id]
    video_name = os.path.splitext(video_info['name'])[0]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_FOLDER, f'{video_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame
    for i, frame_data in enumerate(frames_data):
        frame_bytes = base64.b64decode(frame_data['data'])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        filename = f'frame_{i+1:03d}_time_{frame_data["time"]:.1f}s.png'
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
    
    return jsonify({
        'success': True,
        'output_dir': output_dir,
        'frame_count': len(frames_data)
    })

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
