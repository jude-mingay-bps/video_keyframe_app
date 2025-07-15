async function loadSegment() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('frame-viewer').style.display = 'none';
    hideAnnotations();

    const targetFps = parseInt(document.getElementById('target-fps').value);
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
                target_fps: targetFps
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

    updateFrameDisplay();

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
        if (predictionMode) {
            runPrediction();
        }
    }
}

function nextFrame() {
    if (currentFrameIndex < frames.length - 1) {
        currentFrameIndex++;
        updateFrameDisplay();
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
