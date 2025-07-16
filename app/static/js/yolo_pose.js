// YOLO Pose Detection functionality
let yoloPoseModelInfo = null;
let yoloPoseMode = false;
let frameYoloPoseAnnotations = new Map();

// Load YOLO pose model information
async function loadYoloPoseModelInfo() {
    try {
        const response = await fetch('/get_yolo_pose_model_info');
        const data = await response.json();
        
        if (data.success) {
            yoloPoseModelInfo = data.model_info;
            console.log('YOLO pose model loaded:', yoloPoseModelInfo);
            updateYoloPoseButtonVisibility();
        } else {
            console.warn('YOLO pose model not available:', data.error);
            yoloPoseModelInfo = null;
            updateYoloPoseButtonVisibility();
        }
    } catch (error) {
        console.error('Error loading YOLO pose model info:', error);
        yoloPoseModelInfo = null;
        updateYoloPoseButtonVisibility();
    }
}

// Update YOLO pose prediction button visibility
function updateYoloPoseButtonVisibility() {
    const poseBtn = document.getElementById('predict-yolo-pose-btn');
    const modelStatus = document.getElementById('yolo-pose-model-status');
    
    console.log('Updating YOLO pose button visibility. Model info:', yoloPoseModelInfo);
    
    if (yoloPoseModelInfo && !yoloPoseModelInfo.name.includes('Not Available')) {
        if (poseBtn) {
            poseBtn.style.display = 'block';
            console.log('YOLO pose button shown');
        }
        if (modelStatus) {
            modelStatus.textContent = `YOLO Pose Model: ${yoloPoseModelInfo.name}`;
            modelStatus.className = 'yolo-pose-status connected';
        }
    } else {
        if (poseBtn) {
            poseBtn.style.display = 'none';
            console.log('YOLO pose button hidden');
        }
        if (modelStatus) {
            modelStatus.textContent = 'YOLO Pose Model: Not Available';
            modelStatus.className = 'yolo-pose-status disconnected';
        }
    }
}

// Toggle YOLO pose prediction mode
function toggleYoloPoseMode() {
    yoloPoseMode = !yoloPoseMode;
    const btn = document.getElementById('predict-yolo-pose-btn');
    const icon = btn.querySelector('i');
    
    if (yoloPoseMode) {
        btn.classList.add('active');
        icon.className = 'fas fa-eye-slash';
        btn.querySelector('span').textContent = 'Hide YOLO Poses';
        
        // Show pose annotations on current frame
        if (currentFrameIndex >= 0 && currentFrameIndex < frames.length) {
            showYoloPoseAnnotationsOnFrame(currentFrameIndex);
        }
    } else {
        btn.classList.remove('active');
        icon.className = 'fas fa-user';
        btn.querySelector('span').textContent = 'Show YOLO Poses';
        
        // Hide pose annotations
        hideYoloPoseAnnotations();
    }
}

// Run YOLO pose prediction on current frame
async function predictYoloPoseOnFrame(frameIndex) {
    if (!yoloPoseModelInfo || yoloPoseModelInfo.name.includes('Not Available')) {
        showToast('YOLO pose model not available', 'error');
        return;
    }
    
    if (frameIndex < 0 || frameIndex >= frames.length) {
        return;
    }
    
    const frame = frames[frameIndex];
    const confidence = parseFloat(document.getElementById('confidence-threshold')?.value || 0.3);
    
    // Show loading state
    const btn = document.getElementById('predict-yolo-pose-btn');
    if (btn) setButtonLoading(btn, true);
    
    try {
        const response = await fetch('/predict_yolo_pose_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                frame_data: frame.data,
                confidence: confidence
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            frameYoloPoseAnnotations.set(frameIndex, data.pose_annotations);
            
            if (yoloPoseMode) {
                displayYoloPoseAnnotations(frameIndex, data.pose_annotations);
            }
            
            // Only show toast for errors, not success
        } else {
            showToast(`YOLO pose estimation failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error during YOLO pose prediction:', error);
        showToast('YOLO pose prediction failed', 'error');
    } finally {
        // Hide loading state
        const btn = document.getElementById('predict-yolo-pose-btn');
        if (btn) setButtonLoading(btn, false);
    }
}

// Display YOLO pose annotations on frame
function displayYoloPoseAnnotations(frameIndex, poseAnnotations) {
    console.log('displayYoloPoseAnnotations called with:', poseAnnotations);
    
    if (!poseAnnotations || poseAnnotations.length === 0) {
        console.log('No YOLO pose annotations to display');
        return;
    }
    
    const canvas = document.getElementById('yolo-pose-canvas');
    const frameImg = document.getElementById('frame-image');
    
    if (!canvas || !frameImg) {
        console.error('YOLO pose canvas or frame image not found');
        console.log('Canvas:', canvas);
        console.log('Frame image:', frameImg);
        return;
    }
    
    // Wait for frame image to load if it's not ready
    if (!frameImg.complete || frameImg.naturalWidth === 0) {
        console.log('Frame image not loaded yet, waiting...');
        frameImg.onload = () => displayYoloPoseAnnotations(frameIndex, poseAnnotations);
        return;
    }
    
    // Set canvas size to match frame
    canvas.width = frameImg.offsetWidth;
    canvas.height = frameImg.offsetHeight;
    canvas.style.display = 'block';
    
    console.log('YOLO pose canvas dimensions:', canvas.width, 'x', canvas.height);
    console.log('Frame image dimensions:', frameImg.naturalWidth, 'x', frameImg.naturalHeight);
    console.log('Frame image display size:', frameImg.offsetWidth, 'x', frameImg.offsetHeight);
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Scale factors for drawing (scale to match actual frame display size)
    const scaleX = canvas.width / frameImg.naturalWidth;
    const scaleY = canvas.height / frameImg.naturalHeight;
    
    console.log('YOLO pose scale factors:', scaleX, scaleY);
    
    // Define colors for different persons (different from MoveNet - orange/red theme)
    const colors = [
        '#FF6B35', '#F7931E', '#FFD23F', '#FF4444', '#E74C3C',
        '#D35400', '#E67E22', '#F39C12', '#FF8C00', '#CC5500'
    ];
    
    // Define skeleton connections (COCO format)
    const skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4], // head
        [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], // arms
        [5, 11], [6, 12], [11, 12], // torso
        [11, 13], [12, 14], [13, 15], [14, 16] // legs
    ];
    
    // Draw poses for each person
    poseAnnotations.forEach((person, personIndex) => {
        const color = colors[personIndex % colors.length];
        
        // Create keypoint lookup
        const keypointLookup = {};
        person.keypoints.forEach(kpt => {
            keypointLookup[kpt.index] = kpt;
        });
        
        // Draw skeleton connections
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        skeleton.forEach(([startIdx, endIdx]) => {
            const startKpt = keypointLookup[startIdx];
            const endKpt = keypointLookup[endIdx];
            
            if (startKpt && endKpt) {
                ctx.beginPath();
                ctx.moveTo(startKpt.x * scaleX, startKpt.y * scaleY);
                ctx.lineTo(endKpt.x * scaleX, endKpt.y * scaleY);
                ctx.stroke();
            }
        });
        
        // Draw keypoints
        ctx.fillStyle = color;
        person.keypoints.forEach(kpt => {
            const x = kpt.x * scaleX;
            const y = kpt.y * scaleY;
            
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw confidence score
            ctx.fillStyle = 'white';
            ctx.font = '11px Arial';
            ctx.fillText(kpt.confidence.toFixed(2), x + 7, y - 7);
            ctx.fillStyle = color;
        });
        
        // Draw person info
        if (person.keypoints.length > 0) {
            const firstKpt = person.keypoints[0];
            ctx.fillStyle = 'rgba(255, 107, 53, 0.8)';
            ctx.fillRect(firstKpt.x * scaleX, firstKpt.y * scaleY - 30, 180, 25);
            
            ctx.fillStyle = 'white';
            ctx.font = '13px Arial';
            ctx.fillText(`YOLO Person ${personIndex + 1} (${person.num_keypoints} kpts)`, 
                        firstKpt.x * scaleX + 5, firstKpt.y * scaleY - 12);
        }
    });
}

// Show YOLO pose annotations on specific frame
async function showYoloPoseAnnotationsOnFrame(frameIndex) {
    // Check if we already have pose annotations for this frame
    if (frameYoloPoseAnnotations.has(frameIndex)) {
        displayYoloPoseAnnotations(frameIndex, frameYoloPoseAnnotations.get(frameIndex));
        return;
    }
    
    // Run YOLO pose prediction
    await predictYoloPoseOnFrame(frameIndex);
}

// Hide YOLO pose annotations
function hideYoloPoseAnnotations() {
    const canvas = document.getElementById('yolo-pose-canvas');
    if (canvas) {
        canvas.style.display = 'none';
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// Initialize YOLO pose canvas
function initializeYoloPoseCanvas() {
    const frameContainer = document.querySelector('.frame-display');
    if (!frameContainer) {
        console.error('Frame display container not found');
        return;
    }
    
    // Create YOLO pose canvas if it doesn't exist
    let yoloPoseCanvas = document.getElementById('yolo-pose-canvas');
    if (!yoloPoseCanvas) {
        yoloPoseCanvas = document.createElement('canvas');
        yoloPoseCanvas.id = 'yolo-pose-canvas';
        yoloPoseCanvas.style.position = 'absolute';
        yoloPoseCanvas.style.top = '0';
        yoloPoseCanvas.style.left = '0';
        yoloPoseCanvas.style.pointerEvents = 'none';
        yoloPoseCanvas.style.display = 'none';
        yoloPoseCanvas.style.zIndex = '12'; // Higher than MoveNet canvas
        frameContainer.appendChild(yoloPoseCanvas);
        console.log('YOLO pose canvas created and added to frame container');
        console.log('Frame container style:', frameContainer.style.position || 'static');
        console.log('Frame container position:', frameContainer.getBoundingClientRect());
    }
}

// Update YOLO pose annotations when frame changes
function updateYoloPoseAnnotationsOnFrameChange(frameIndex) {
    if (yoloPoseMode) {
        // Add a small delay to ensure frame image is loaded
        setTimeout(() => {
            showYoloPoseAnnotationsOnFrame(frameIndex);
        }, 150);
    } else {
        hideYoloPoseAnnotations();
    }
}

// Initialize YOLO pose functionality
document.addEventListener('DOMContentLoaded', function() {
    loadYoloPoseModelInfo();
    initializeYoloPoseCanvas();
    
    // Add event listener for YOLO pose prediction button
    const yoloPoseBtn = document.getElementById('predict-yolo-pose-btn');
    if (yoloPoseBtn) {
        yoloPoseBtn.addEventListener('click', toggleYoloPoseMode);
    }
});

// Export functions for use in other modules
window.yoloPose = {
    loadYoloPoseModelInfo,
    toggleYoloPoseMode,
    predictYoloPoseOnFrame,
    updateYoloPoseAnnotationsOnFrameChange,
    initializeYoloPoseCanvas
};