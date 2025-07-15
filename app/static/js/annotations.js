function hideAnnotations() {
    document.getElementById('annotation-info').classList.remove('active');
    document.getElementById('correction-controls').classList.remove('active');
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
        imageElement.onload = () => createInteractiveBoundingBoxes(annotations, imageElement);
        return;
    }

    const container = imageElement.parentElement;
    const containerRect = container.getBoundingClientRect();
    const imgRect = imageElement.getBoundingClientRect();
    const offsetX = imgRect.left - containerRect.left;
    const offsetY = imgRect.top - containerRect.top;
    const displayWidth = imageElement.offsetWidth;
    const displayHeight = imageElement.offsetHeight;
    const naturalWidth = imageElement.naturalWidth;
    const naturalHeight = imageElement.naturalHeight;
    const scaleX = displayWidth / naturalWidth;
    const scaleY = displayHeight / naturalHeight;

    overlay.style.width = displayWidth + 'px';
    overlay.style.height = displayHeight + 'px';
    overlay.style.position = 'absolute';
    overlay.style.top = offsetY + 'px';
    overlay.style.left = offsetX + 'px';

    annotations.forEach((annotation, index) => {
        const [x1, y1, x2, y2] = annotation.bbox_xyxy;
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

        const frameKey = `${currentVideoId}_${currentFrameIndex}`;
        const corrections = correctedAnnotations.get(frameKey) || new Set();
        const isCorrected = corrections.has(index);

        const label = document.createElement('div');
        label.className = 'bbox-label';

        if (isCorrected) {
            bbox.classList.add('misclassified');
            label.textContent = 'Other';
        } else {
            const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22'];
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
    img.onload = null;

    const prediction = framePredictions.get(currentFrameIndex);
    const frameSource = prediction ? prediction.frame_data : frames[currentFrameIndex].data;
    img.src = `data:image/jpeg;base64,${frameSource}`;

    clearBoundingBoxDisplay();

    if (prediction) {
        img.classList.add('predicted');
        updateAnnotationDisplay();
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
