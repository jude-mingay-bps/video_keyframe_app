let backgroundProcessingInterval = null;

// Frame preloading system
class FramePreloader {
    constructor() {
        this.preloadedFrames = new Map();
        this.preloadRadius = 3;
        this.maxCacheSize = 50;
        this.isPreloading = false;
    }
    
    preloadAroundFrame(frameIndex) {
        if (this.isPreloading || !frames.length) return;
        
        this.isPreloading = true;
        
        const start = Math.max(0, frameIndex - this.preloadRadius);
        const end = Math.min(frames.length, frameIndex + this.preloadRadius + 1);
        
        // Preload frames around current frame
        for (let i = start; i < end; i++) {
            if (!this.preloadedFrames.has(i)) {
                this.preloadFrame(i);
            }
        }
        
        // Clean up old preloaded frames
        this.cleanupOldFrames(frameIndex);
        
        this.isPreloading = false;
    }
    
    preloadFrame(index) {
        if (this.preloadedFrames.has(index) || !frames[index]) return;
        
        const img = new Image();
        img.onload = () => {
            this.preloadedFrames.set(index, img);
            // Limit cache size
            if (this.preloadedFrames.size > this.maxCacheSize) {
                this.cleanupOldFrames(index);
            }
        };
        img.onerror = () => {
            console.warn(`Failed to preload frame ${index}`);
        };
        img.src = `data:image/jpeg;base64,${frames[index].data}`;
    }
    
    cleanupOldFrames(currentIndex) {
        if (this.preloadedFrames.size <= this.maxCacheSize) return;
        
        const toRemove = [];
        for (const [index, img] of this.preloadedFrames) {
            const distance = Math.abs(index - currentIndex);
            if (distance > this.preloadRadius * 2) {
                toRemove.push(index);
            }
        }
        
        // Remove furthest frames first
        toRemove.sort((a, b) => Math.abs(b - currentIndex) - Math.abs(a - currentIndex));
        
        while (this.preloadedFrames.size > this.maxCacheSize && toRemove.length > 0) {
            const indexToRemove = toRemove.pop();
            this.preloadedFrames.delete(indexToRemove);
        }
    }
    
    getPreloadedFrame(index) {
        return this.preloadedFrames.get(index);
    }
    
    clear() {
        this.preloadedFrames.clear();
    }
}

const framePreloader = new FramePreloader();

async function loadSegment() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('frame-viewer').style.display = 'none';
    document.getElementById('processing-status').style.display = 'none';
    
    // Clear preloader cache
    framePreloader.clear();
    
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

            // Start preloading frames around current frame
            setTimeout(() => {
                framePreloader.preloadAroundFrame(currentFrameIndex);
            }, 100);

            // Always start monitoring background processing when frames are loaded
            startBackgroundProcessingMonitor();

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
        
        // Preload frames around new position
        framePreloader.preloadAroundFrame(currentFrameIndex);
        
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
        
        // Preload frames around new position
        framePreloader.preloadAroundFrame(currentFrameIndex);
        
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
