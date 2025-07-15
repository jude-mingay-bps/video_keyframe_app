let dragStartX = 0;
let initialSegmentStart = 0;
let videos = [];
let currentVideoIndex = 0;
let frames = [];
let currentFrameIndex = 0;
let selectedFrames = new Set();
let framePredictions = new Map(); // Store predictions for each frame
let currentVideoId = null;
let videoDuration = 0;
let segmentStart = 0;
let segmentDuration = 30;
let isDragging = false;
let dragType = null;
let predictionMode = false; // Whether YOLO predictions are enabled
let modelInfo = null;
let roboflowConfig = {
    url: '',
    apiKey: '',
    batchName: '',
    split: 'train',
    isConfigured: false
};

// --- NEW --- Interactive bounding box state
let correctionMode = false;
let currentBoundingBoxes = [];
let correctedAnnotations = new Map(); // Store corrections per frame

// Initialize confidence threshold slider
document.getElementById('confidence-threshold').addEventListener('input', (e) => {
    document.getElementById('confidence-value').textContent = e.target.value;
});

// Add this after the confidence threshold slider initialization
document.getElementById('target-fps').addEventListener('input', (e) => {
    document.getElementById('fps-value').textContent = e.target.value;
});

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

    setTimeout(() => toast.classList.add('show'), 10);

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
        button.textContent = '';
    } else {
        button.disabled = false;
        button.classList.remove('button-loading');
        if (button.dataset.originalText) {
            button.textContent = button.dataset.originalText;
            delete button.dataset.originalText;
        }
    }
}

// YOLO Model functions
async function loadModelInfo() {
    try {
        const response = await fetch('/get_model_info');
        const data = await response.json();

        if (data.success) {
            modelInfo = data.model_info;
            updateYoloStatus(true);
            displayModelInfo(modelInfo);
        } else {
            updateYoloStatus(false);
            displayModelError(data.error);
        }
    } catch (error) {
        updateYoloStatus(false);
        displayModelError('Failed to connect to model service');
    }
}

function updateYoloStatus(connected) {
    const status = document.getElementById('yolo-status');
    if (connected) {
        status.textContent = 'Model Loaded';
        status.className = 'yolo-status connected';
    } else {
        status.textContent = 'No Model';
        status.className = 'yolo-status disconnected';
    }
}

function displayModelInfo(info) {
    const container = document.getElementById('model-info-container');
    const classNames = Object.values(info.classes).slice(0, 10); // Show first 10 classes
    const moreClasses = info.num_classes - 10;

    container.innerHTML = `
        <div class="model-info">
            <h4>Model: ${info.name}</h4>
            <p><strong>Classes:</strong> ${info.num_classes}</p>
            <div class="class-list">
                ${classNames.map(name => `<span class="class-tag">${name}</span>`).join('')}
                ${moreClasses > 0 ? `<span class="class-tag">+${moreClasses} more</span>` : ''}
            </div>
        </div>
    `;
}

function displayModelError(error) {
    const container = document.getElementById('model-info-container');
    container.innerHTML = `
        <div class="model-info" style="border-color: rgba(231, 76, 60, 0.3); background: rgba(231, 76, 60, 0.1);">
            <h4 style="color: #e74c3c;">⚠️ Model Not Available</h4>
            <p>${error}</p>
            <p><small>Place a YOLO model file (.pt, .onnx, .engine) in the 'models' folder to enable predictions.</small></p>
        </div>
    `;
}

function togglePredictionMode() {
    const toggle = document.getElementById('prediction-toggle');
    predictionMode = !predictionMode;

    if (predictionMode) {
        toggle.classList.add('active');
        showToast('Automatic predictions enabled.', 'info', 3000);
        // --- FIX ---
        // Immediately run a prediction on the current frame when the toggle is enabled.
        runPrediction();
    } else {
        toggle.classList.remove('active');
        showToast('Automatic predictions disabled.', 'info', 3000);
    }
}


function updatePredictButtonVisibility() {
    const predictBtn = document.getElementById('predict-btn');
    if (predictionMode && modelInfo) {
        predictBtn.style.display = 'block';
    } else {
        predictBtn.style.display = 'none';
    }
}

async function runPrediction() {
    // --- FIX ---
    // First, check if a prediction already exists for this frame. If so, do nothing.
    if (framePredictions.has(currentFrameIndex)) {
        // Just make sure the display is up-to-date with the existing prediction.
        updateFrameDisplay();
        return;
    }

    // Original guard clause to ensure predictions are wanted and possible.
    if (!predictionMode || !modelInfo || !frames.length) {
        return;
    }

    // This part remains the same, executing the prediction via fetch.
    const button = document.getElementById('predict-btn'); // Note: This button is hidden but we can keep the logic
    if(button) setButtonLoading(button, true); // Safely handle if button exists

    try {
        const currentFrame = frames[currentFrameIndex];
        const confidence = parseFloat(document.getElementById('confidence-threshold').value);

        const response = await fetch('/predict_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame_data: currentFrame.data,
                confidence: confidence
            })
        });

        const data = await response.json();

        if (data.success) {
            framePredictions.set(currentFrameIndex, {
                annotations: data.annotations,
                frame_data: data.frame_data
            });

            updateFrameDisplay();

            if (data.annotations.length > 0) {
                const correctionBtn = document.getElementById('correction-btn');
                if(correctionBtn) correctionBtn.style.display = 'block';
            }
        } else {
            showToast('Prediction failed: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('Error running prediction: ' + error.message, 'error');
    } finally {
        if(button) setButtonLoading(button, false);
    }
}

function hideAnnotations() {
    document.getElementById('annotation-info').classList.remove('active');
    document.getElementById('correction-controls').classList.remove('active');
    if (correctionMode) {
        toggleCorrectionMode(); // Exit correction mode if annotations are hidden
    }
}

// --- NEW/MODIFIED --- Interactive bounding box and annotation display functions

function toggleCorrectionMode() {
    correctionMode = !correctionMode;
    const controls = document.getElementById('correction-controls');

    if (correctionMode) {
        controls.classList.add('active');
        updateBoundingBoxDisplay();
    } else {
        controls.classList.remove('active');
        clearBoundingBoxDisplay();
    }
}

function createInteractiveBoundingBoxes(annotations, imageElement) {
    clearBoundingBoxDisplay();
    if (!annotations || !annotations.length) return;

    const overlay = document.getElementById('bbox-overlay');

    // Wait for image to be fully loaded
    if (!imageElement.complete || !imageElement.naturalHeight) {
        imageElement.onload = () => createInteractiveBoundingBoxes(annotations, imageElement);
        return;
    }

    // Get image position relative to its container
    const container = imageElement.parentElement;
    const containerRect = container.getBoundingClientRect();
    const imgRect = imageElement.getBoundingClientRect();

    // Calculate offset of image within container
    const offsetX = imgRect.left - containerRect.left;
    const offsetY = imgRect.top - containerRect.top;

    // Get the actual displayed size of the image
    const displayWidth = imageElement.offsetWidth;
    const displayHeight = imageElement.offsetHeight;
    const naturalWidth = imageElement.naturalWidth;
    const naturalHeight = imageElement.naturalHeight;

    // Calculate scale factors
    const scaleX = displayWidth / naturalWidth;
    const scaleY = displayHeight / naturalHeight;

    // Position and size overlay to match image exactly
    overlay.style.width = displayWidth + 'px';
    overlay.style.height = displayHeight + 'px';
    overlay.style.position = 'absolute';
    overlay.style.top = offsetY + 'px';
    overlay.style.left = offsetX + 'px';

    annotations.forEach((annotation, index) => {
        const [x1, y1, x2, y2] = annotation.bbox_xyxy;

        // Scale coordinates to match displayed image size
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;

        const bbox = document.createElement('div');
        bbox.className = 'bbox-item';
        bbox.style.left = `${left}px`;
        bbox.style.top = `${top}px`;
        bbox.style.width = `${width}px`;
        bbox.style.height = `${height}px`;
        bbox.style.position = 'absolute';

        // Check if this annotation has been corrected
        const frameKey = `${currentVideoId}_${currentFrameIndex}`;
        const corrections = correctedAnnotations.get(frameKey) || new Set();
        const isCorrected = corrections.has(index);

        // Create label
        const label = document.createElement('div');
        label.className = 'bbox-label';

        if (isCorrected) {
            bbox.classList.add('misclassified');
            label.textContent = 'Other';
        } else {
            // Set color based on class
            const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22'];
            bbox.style.borderColor = colors[annotation.class_id % colors.length];
            label.textContent = `${annotation.class_name}: ${(annotation.confidence * 100).toFixed(1)}%`;
        }

        bbox.appendChild(label);

        // Add click handler if in correction mode
        if (correctionMode) {
            bbox.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleAnnotationCorrection(index);
            });
        }

        overlay.appendChild(bbox);
        currentBoundingBoxes.push({ element: bbox, annotation: annotation, index: index, corrected: isCorrected });
    });
}


function toggleAnnotationCorrection(annotationIndex) {
    const frameKey = `${currentVideoId}_${currentFrameIndex}`;
    if (!correctedAnnotations.has(frameKey)) {
        correctedAnnotations.set(frameKey, new Set());
    }

    const corrections = correctedAnnotations.get(frameKey);
    const bbox = currentBoundingBoxes.find(b => b.index === annotationIndex);
    if (!bbox) return;

    if (corrections.has(annotationIndex)) {
        corrections.delete(annotationIndex);
    } else {
        corrections.add(annotationIndex);
    }

    updateAnnotationDisplay();
    updateBoundingBoxDisplay(); // Redraw with new state
}

function clearAllCorrections() {
    const frameKey = `${currentVideoId}_${currentFrameIndex}`;
    correctedAnnotations.delete(frameKey);
    updateAnnotationDisplay();
    updateBoundingBoxDisplay();
}

function clearBoundingBoxDisplay() {
    const overlay = document.getElementById('bbox-overlay');
    overlay.innerHTML = '';
    currentBoundingBoxes = [];
}

function updateBoundingBoxDisplay() {
    if (!correctionMode) {
        clearBoundingBoxDisplay();
        return;
    };

    const img = document.getElementById('frame-image');
    if (img.src && framePredictions.has(currentFrameIndex)) {
        const prediction = framePredictions.get(currentFrameIndex);
        if (img.complete) {
            createInteractiveBoundingBoxes(prediction.annotations, img);
        } else {
            img.onload = () => createInteractiveBoundingBoxes(prediction.annotations, img);
        }
    } else {
        clearBoundingBoxDisplay();
    }
}

function updateAnnotationDisplay() {
    if (!framePredictions.has(currentFrameIndex)) {
        hideAnnotations();
        return;
    }

    const prediction = framePredictions.get(currentFrameIndex);
    const annotations = prediction.annotations;
    const frameKey = `${currentVideoId}_${currentFrameIndex}`;
    const corrections = correctedAnnotations.get(frameKey) || new Set();

    const annotationList = document.getElementById('annotation-list');
    const annotationInfo = document.getElementById('annotation-info');

    if (annotations && annotations.length > 0) {
        annotationList.innerHTML = annotations.map((ann, index) => {
            const isCorrected = corrections.has(index);
            const itemClass = isCorrected ? 'annotation-item other-class' : 'annotation-item';
            const displayName = isCorrected ? 'Other' : ann.class_name;
            const confidence = isCorrected ? '100.0' : (ann.confidence * 100).toFixed(1);

            return `
                <div class="${itemClass}">
                    <span>${displayName}</span>
                    <span class="annotation-confidence">${confidence}%</span>
                </div>`;
        }).join('');
        annotationInfo.classList.add('active');
    } else {
        annotationList.innerHTML = '<div class="annotation-item">No detections found</div>';
        annotationInfo.classList.add('active');
    }
}

function updateFrameDisplay() {
    if (!frames.length) return;

    const img = document.getElementById('frame-image');

    // --- FIX ---
    // Cancel any previous onload events to prevent old annotations from being redrawn on the new frame.
    img.onload = null;

    const prediction = framePredictions.get(currentFrameIndex);
    const frameSource = prediction ? prediction.frame_data : frames[currentFrameIndex].data;
    img.src = `data:image/jpeg;base64,${frameSource}`;

    clearBoundingBoxDisplay();

    if (prediction) {
        img.classList.add('predicted');
        updateAnnotationDisplay();
        // Set the onload event for the CURRENT frame.
        img.onload = () => createInteractiveBoundingBoxes(prediction.annotations, img);
        if(img.complete) img.onload();
    } else {
        img.classList.remove('predicted');
        hideAnnotations();
    }

    if (selectedFrames.has(currentFrameIndex)) {
        img.classList.add('selected');
    } else {
        img.classList.remove('selected');
    }
}

// --- END NEW/MODIFIED ---

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
    loadModelInfo();
    initializeTimeline();
    updatePredictButtonVisibility();
});

// Keyboard event listeners
document.addEventListener('keydown', (e) => {
    if (document.querySelector('.frame-selector').style.display !== 'block' || !frames.length) return;

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
        case 'p':
        case 'P':
            e.preventDefault();
            runPrediction();
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
        if (e.button !== 0) return; // Only main left click
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
            segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, mouseTime - segmentDuration / 2));
        } else if (dragType === 'left') {
            const currentEnd = segmentStart + segmentDuration;
            const newStart = Math.min(mouseTime, currentEnd - 1);
            segmentDuration = currentEnd - newStart;
            segmentStart = newStart;
        } else if (dragType === 'right') {
            const newEnd = Math.max(mouseTime, segmentStart + 1);
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

    timeline.querySelectorAll('.timeline-thumbnail').forEach(el => el.remove());

    const fragment = document.createDocumentFragment();
    thumbnails.forEach(thumbData => {
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${thumbData}`;
        img.className = 'timeline-thumbnail';
        img.draggable = false;
        fragment.appendChild(img);
    });

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
    document.querySelector('.frame-selector').style.display = 'block';
    document.querySelector('.upload-section').style.display = 'none';
    document.getElementById('video-list').style.display = 'none';
    document.querySelector('.roboflow-section').style.display = 'none';
    document.querySelector('.yolo-section').style.display = 'none';
    document.querySelector('header').style.display = 'none';

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
    framePredictions.clear();
    // Do NOT clear correctedAnnotations here, so corrections persist if user goes back
    document.getElementById('frame-viewer').style.display = 'none';
    hideAnnotations();

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
    hideAnnotations();

    const targetFps = parseInt(document.getElementById('target-fps').value);

    // Show estimated frame count
    const estimatedFrames = Math.floor(segmentDuration * targetFps);
    const loadingText = document.querySelector('#loading p');
    loadingText.textContent = `Loading ~${estimatedFrames} frames at ${targetFps} FPS...`;

    try {
        const response = await fetch('/extract_frames', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: currentVideoId,
                start_time: segmentStart,
                duration: segmentDuration,
                target_fps: targetFps  // Add this parameter
            })
        });

        const data = await response.json();
        if (data.success) {
            frames = data.frames;
            currentFrameIndex = 0;
            selectedFrames.clear();
            framePredictions.clear();
            document.getElementById('loading').style.display = 'none';
            document.getElementById('frame-viewer').style.display = 'block';
            updateFrameDisplay();
            showToast(`Loaded ${frames.length} frames at ${targetFps} FPS`, 'success');

            // Automatically run prediction on the first frame if the mode is active
            if (predictionMode) {
                runPrediction();
            }
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

    updateFrameDisplay();

    const info = document.getElementById('frame-info');
    const selectedText = selectedFrames.has(currentFrameIndex) ?
        '<span class="selected-indicator">[SELECTED]</span>' : '';
    const predictionText = framePredictions.has(currentFrameIndex) ?
        '<span class="predicted-indicator">[PREDICTED]</span>' : '';
    info.innerHTML = `Frame ${currentFrameIndex + 1}/${frames.length} | ` +
                   `Time: ${frame.time.toFixed(1)}s | ` +
                   `Selected: ${selectedFrames.size} ${selectedText} ${predictionText}`;

    const progress = ((currentFrameIndex + 1) / frames.length) * 100;
    const progressFill = document.getElementById('progress-fill');
    progressFill.style.width = `${progress}%`;
    progressFill.textContent = `${Math.round(progress)}%`;
}

function previousFrame() {
    if (currentFrameIndex > 0) {
        currentFrameIndex--;
        updateFrameDisplay();
        // Automatically run prediction on the new frame if the mode is active
        if (predictionMode) {
            runPrediction();
        }
    }
}

function nextFrame() {
    if (currentFrameIndex < frames.length - 1) {
        currentFrameIndex++;
        updateFrameDisplay();
        // Automatically run prediction on the new frame if the mode is active
        if (predictionMode) {
            runPrediction();
        }
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

        // --- MODIFIED to include corrections ---
        const selectedFrameData = Array.from(selectedFrames).map(frameIndex => {
            const frameData = {
                ...frames[frameIndex],
                frameIndex: frameIndex
            };

            if (framePredictions.has(frameIndex)) {
                const prediction = framePredictions.get(frameIndex);
                const frameKey = `${currentVideoId}_${frameIndex}`;
                const corrections = correctedAnnotations.get(frameKey) || new Set();

                const correctedAnnotationsList = prediction.annotations.map((ann, index) => {
                    if (corrections.has(index)) {
                        return {
                            ...ann,
                            class_name: 'Other',
                            class_id: 999, // Special ID for "Other"
                            confidence: 1.0,
                            was_corrected: true
                        };
                    }
                    return ann;
                });

                frameData.predictions = {
                    annotations: correctedAnnotationsList,
                    annotated_frame: prediction.annotated_frame || null
                };
            } else {
                frameData.predictions = null;
            }

            return frameData;
        });

        try {
            const response = await fetch('/save_frames', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_id: currentVideoId,
                    frames: selectedFrameData, // Send modified data
                    upload_to_roboflow: uploadToRoboflow,
                    roboflow_config: uploadToRoboflow ? finalRoboflowConfig : null
                })
            });

            const data = await response.json();
            removeToast(uploadToast);

            if (data.success) {
                let message = `Saved ${data.frame_count} frames to ${data.output_dir}`;
                let toastType = 'success';

                if (data.roboflow_results) {
                    const uploaded = data.roboflow_results.filter(r => r.success).length;
                    const failed = data.roboflow_results.filter(r => !r.success).length;

                    if (failed > 0) {
                        message += `. Roboflow: ${uploaded} uploaded, ${failed} failed`;
                        toastType = 'warning';
                    } else {
                        message += `. All ${uploaded} frames uploaded to Roboflow.`;
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

function showMainMenu() {
    document.querySelector('.frame-selector').style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    document.querySelector('.roboflow-section').style.display = 'block';
    document.querySelector('.yolo-section').style.display = 'block';
    document.querySelector('header').style.display = 'block';
    if (videos.length > 0) {
        document.getElementById('video-list').style.display = 'block';
    }
}

function resetInterface() {
    showMainMenu();
    videos = [];
    updateVideoList();
    correctedAnnotations.clear();
}
