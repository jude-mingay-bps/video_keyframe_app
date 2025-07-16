let backgroundProcessingInterval = null;

async function loadSegment() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('frame-viewer').style.display = 'none';
    document.getElementById('processing-status').style.display = 'none';
    
    // Reset the first frame flag
    if (typeof isFirstFrame !== 'undefined') {
        isFirstFrame = true;
    }
    
    // Reset the current image source
    if (typeof currentImageSrc !== 'undefined') {
        currentImageSrc = null;
    }
    
    // Reset the frame image visibility
    const frameImage = document.getElementById('frame-image');
    if (frameImage) {
        frameImage.style.display = 'none';
        frameImage.style.opacity = '1';
    }
    
    hideAnnotations();

    const targetFps = parseInt(document.getElementById('target-fps').value);
    const estimatedFrames = Math.floor(segmentDuration * targetFps);
    const loadingText = document.querySelector('#loading p');
    loadingText.textContent = `Loading ~${estimatedFrames} frames at ${targetFps} FPS...`;

    try {
        const confidence = parseFloat(document.getElementById('confidence-threshold').value);
        const response = await fetch('/extract_frames', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: currentVideoId,
                start_time: segmentStart,
                duration: segmentDuration,
                target_fps: targetFps,
                confidence_threshold: confidence
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
            displayFrame();
            showToast(`Loaded ${frames.length} frames at ${targetFps} FPS`, 'success');

            // Start monitoring background processing if prediction mode is enabled
            if (predictionMode && modelInfo) {
                startBackgroundProcessingMonitor();
            }

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

function startBackgroundProcessingMonitor() {
    // Clear any existing interval
    if (backgroundProcessingInterval) {
        clearInterval(backgroundProcessingInterval);
    }

    // Show the processing status
    document.getElementById('processing-status').style.display = 'block';

    // Poll for status updates
    backgroundProcessingInterval = setInterval(async () => {
        try {
            const response = await fetch('/get_background_processing_status');
            const data = await response.json();

            if (data.success) {
                const progressBar = document.getElementById('processing-progress');
                const progressInfo = document.getElementById('processing-info');

                progressBar.style.width = `${data.progress_percentage}%`;
                progressInfo.textContent = `${data.processed_frames} / ${data.total_frames} frames`;

                if (data.is_complete || data.progress_percentage >= 100) {
                    clearInterval(backgroundProcessingInterval);
                    backgroundProcessingInterval = null;
                    
                    // Keep showing for 2 seconds after completion
                    setTimeout(() => {
                        document.getElementById('processing-status').style.display = 'none';
                    }, 2000);
                }
            }
        } catch (error) {
            console.error('Error checking background processing status:', error);
        }
    }, 1000); // Check every second
}

function stopBackgroundProcessingMonitor() {
    if (backgroundProcessingInterval) {
        clearInterval(backgroundProcessingInterval);
        backgroundProcessingInterval = null;
    }
}

function displayFrame() {
    if (!frames.length) return;

    const info = document.getElementById('frame-info');
    const selectedText = selectedFrames.has(currentFrameIndex) ?
        '<span class="selected-indicator">[SELECTED]</span>' : '';
    const predictionText = framePredictions.has(currentFrameIndex) ?
        '<span class="predicted-indicator">[PREDICTED]</span>' : '';
    info.innerHTML = `Frame ${currentFrameIndex + 1}/${frames.length} | ` +
                   `Time: ${frames[currentFrameIndex].time.toFixed(1)}s | ` +
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
        displayFrame();
        if (predictionMode) {
            runPrediction();
        }
        // Update pose annotations if in pose mode
        if (typeof updatePoseAnnotationsOnFrameChange === 'function') {
            updatePoseAnnotationsOnFrameChange(currentFrameIndex);
        }
    }
}

function nextFrame() {
    if (currentFrameIndex < frames.length - 1) {
        currentFrameIndex++;
        updateFrameDisplay();
        displayFrame();
        if (predictionMode) {
            runPrediction();
        }
        // Update pose annotations if in pose mode
        if (typeof updatePoseAnnotationsOnFrameChange === 'function') {
            updatePoseAnnotationsOnFrameChange(currentFrameIndex);
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
