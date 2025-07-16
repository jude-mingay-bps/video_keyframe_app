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
        runPrediction();
    } else {
        toggle.classList.remove('active');
        showToast('Automatic predictions disabled.', 'info', 3000);
    }
}

function updatePredictButtonVisibility() {
    const predictBtn = document.getElementById('predict-btn');
    if (predictionMode && modelInfo) {
        if(predictBtn) predictBtn.style.display = 'block';
    } else {
        if(predictBtn) predictBtn.style.display = 'none';
    }
}

async function runPrediction() {
    if (framePredictions.has(currentFrameIndex)) {
        updateFrameDisplay();
        return;
    }

    if (!predictionMode || !modelInfo || !frames.length) {
        return;
    }

    const button = document.getElementById('predict-btn');
    if(button) setButtonLoading(button, true);

    try {
        const currentFrame = frames[currentFrameIndex];
        const confidence = parseFloat(document.getElementById('confidence-threshold').value);

        // First check server cache for pre-processed annotations
        const cacheResponse = await fetch('/get_cached_annotations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame_num: currentFrame.frame_num
            })
        });

        const cacheData = await cacheResponse.json();
        
        if (cacheData.success && cacheData.cached && cacheData.annotations !== null) {
            // Use cached annotations
            framePredictions.set(currentFrameIndex, {
                annotations: cacheData.annotations,
                frame_data: currentFrame.data,
                fromCache: true
            });

            updateFrameDisplay();

            if (cacheData.annotations.length > 0) {
                const correctionBtn = document.getElementById('correction-btn');
                if(correctionBtn) correctionBtn.style.display = 'block';
            }
        } else {
            // No cache available, run prediction
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
                    frame_data: data.frame_data,
                    fromCache: false
                });

                updateFrameDisplay();

                if (data.annotations.length > 0) {
                    const correctionBtn = document.getElementById('correction-btn');
                    if(correctionBtn) correctionBtn.style.display = 'block';
                }
            } else {
                showToast('Prediction failed: ' + data.error, 'error');
            }
        }
    } catch (error) {
        showToast('Error running prediction: ' + error.message, 'error');
    } finally {
        if(button) setButtonLoading(button, false);
    }
}
