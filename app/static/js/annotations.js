function hideAnnotations() {
    const annotationInfo = document.getElementById('annotation-info');
    const correctionControls = document.getElementById('correction-controls');
    
    // Use opacity transition instead of immediate hide
    annotationInfo.classList.remove('active');
    correctionControls.classList.remove('active');
    
    if (correctionMode) {
        toggleCorrectionMode();
    }
}

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

    if (!imageElement.complete || !imageElement.naturalHeight) {
        setTimeout(() => createInteractiveBoundingBoxes(annotations, imageElement), 50);
        return;
    }

    const naturalWidth = imageElement.naturalWidth;
    const naturalHeight = imageElement.naturalHeight;
    
    if (!naturalWidth || !naturalHeight) {
        setTimeout(() => createInteractiveBoundingBoxes(annotations, imageElement), 50);
        return;
    }
    
    // Create a wrapper that matches the image exactly
    const wrapper = document.createElement('div');
    wrapper.style.position = 'absolute';
    wrapper.style.top = '0';
    wrapper.style.left = '0';
    wrapper.style.width = '100%';
    wrapper.style.height = '100%';
    wrapper.style.pointerEvents = 'none';
    
    // Clear and setup overlay
    overlay.innerHTML = '';
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.right = '0';
    overlay.style.bottom = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.appendChild(wrapper);

    annotations.forEach((annotation, index) => {
        const [x1, y1, x2, y2] = annotation.bbox_xyxy;
        
        // Convert to percentages based on natural image dimensions
        const leftPercent = (x1 / naturalWidth) * 100;
        const topPercent = (y1 / naturalHeight) * 100;
        const widthPercent = ((x2 - x1) / naturalWidth) * 100;
        const heightPercent = ((y2 - y1) / naturalHeight) * 100;

        const bbox = document.createElement('div');
        bbox.className = 'bbox-item';
        bbox.style.left = `${leftPercent}%`;
        bbox.style.top = `${topPercent}%`;
        bbox.style.width = `${widthPercent}%`;
        bbox.style.height = `${heightPercent}%`;
        bbox.style.position = 'absolute';

        const frameKey = `${currentVideoId}_${currentFrameIndex}`;
        const corrections = correctedAnnotations.get(frameKey) || new Set();
        const isCorrected = corrections.has(index);

        const label = document.createElement('div');
        label.className = 'bbox-label';

        if (isCorrected) {
            bbox.classList.add('misclassified');
            label.textContent = 'Other';
        } else {
            const colors = [
                'rgba(52, 152, 219, 0.85)',   // Blue
                'rgba(231, 76, 60, 0.85)',     // Red
                'rgba(46, 204, 113, 0.85)',    // Green
                'rgba(243, 156, 18, 0.85)',    // Orange
                'rgba(155, 89, 182, 0.85)',    // Purple
                'rgba(26, 188, 156, 0.85)',    // Turquoise
                'rgba(52, 73, 94, 0.85)',      // Dark gray
                'rgba(230, 126, 34, 0.85)'     // Dark orange
            ];
            bbox.style.borderColor = colors[annotation.class_id % colors.length];
            label.textContent = `${annotation.class_name}: ${(annotation.confidence * 100).toFixed(1)}%`;
        }

        bbox.appendChild(label);

        if (correctionMode) {
            bbox.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleAnnotationCorrection(index);
            });
        }

        wrapper.appendChild(bbox);
        currentBoundingBoxes.push({ element: bbox, annotation: annotation, index: index, corrected: isCorrected });
    });
    
    // Ensure wrapper has pointer events for child elements
    wrapper.style.pointerEvents = 'none';
    const bboxItems = wrapper.querySelectorAll('.bbox-item');
    bboxItems.forEach(item => {
        item.style.pointerEvents = 'auto';
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
    updateBoundingBoxDisplay();
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
    const img = document.getElementById('frame-image');
    if (img.src && framePredictions.has(currentFrameIndex)) {
        const prediction = framePredictions.get(currentFrameIndex);
        if (prediction.annotations && prediction.annotations.length > 0) {
            if (img.complete) {
                createInteractiveBoundingBoxes(prediction.annotations, img);
            } else {
                img.onload = () => createInteractiveBoundingBoxes(prediction.annotations, img);
            }
        } else {
            clearBoundingBoxDisplay();
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

let pendingImageLoad = null;
let isFirstFrame = true;
let currentImageSrc = null;

function updateFrameDisplay() {
    if (!frames.length) return;

    const img = document.getElementById('frame-image');
    const prediction = framePredictions.get(currentFrameIndex);
    const frameSource = prediction ? prediction.frame_data : frames[currentFrameIndex].data;
    
    // Direct update - no fancy transitions
    img.src = `data:image/jpeg;base64,${frameSource}`;
    img.style.display = 'block';
    img.style.opacity = '1';
    
    // Update annotations
    if (prediction) {
        img.classList.add('predicted');
        updateAnnotationDisplay();
        // Create bounding boxes after a short delay
        setTimeout(() => {
            if (prediction.annotations && prediction.annotations.length > 0) {
                createInteractiveBoundingBoxes(prediction.annotations, img);
            }
        }, 50);
    } else {
        img.classList.remove('predicted');
        hideAnnotations();
        clearBoundingBoxDisplay();
    }

    if (selectedFrames.has(currentFrameIndex)) {
        img.classList.add('selected');
    } else {
        img.classList.remove('selected');
    }
    
    // Update pose annotations if MoveNet is active
    if (typeof updatePoseAnnotationsOnFrameChange === 'function') {
        updatePoseAnnotationsOnFrameChange(currentFrameIndex);
    }
    
    // Update YOLO pose annotations if YOLO pose is active
    if (typeof updateYoloPoseAnnotationsOnFrameChange === 'function') {
        updateYoloPoseAnnotationsOnFrameChange(currentFrameIndex);
    }
}
