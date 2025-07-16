// MoveNet Lightning functionality - Updated to fix loader issues
let movenetModelInfo = null;
let poseMode = false;
let framePoseAnnotations = new Map();

// Load MoveNet model information
async function loadMoveNetModelInfo() {
    try {
        const response = await fetch('/get_movenet_model_info');
        const data = await response.json();
        
        if (data.success) {
            movenetModelInfo = data.model_info;
            console.log('MoveNet model loaded:', movenetModelInfo);
            updatePoseButtonVisibility();
        } else {
            console.warn('MoveNet model not available:', data.error);
            movenetModelInfo = null;
            updatePoseButtonVisibility();
        }
    } catch (error) {
        console.error('Error loading MoveNet model info:', error);
        movenetModelInfo = null;
        updatePoseButtonVisibility();
    }
}

// Update pose prediction button visibility
function updatePoseButtonVisibility() {
    const poseBtn = document.getElementById('predict-pose-btn');
    const modelStatus = document.getElementById('movenet-model-status');
    
    console.log('Updating pose button visibility. Model info:', movenetModelInfo);
    
    if (movenetModelInfo && !movenetModelInfo.name.includes('Not Available')) {
        if (poseBtn) {
            poseBtn.style.display = 'block';
            console.log('Pose button shown');
        }
        if (modelStatus) {
            modelStatus.textContent = `MoveNet Model: ${movenetModelInfo.name}`;
            modelStatus.className = 'movenet-status connected';
        }
    } else {
        if (poseBtn) {
            poseBtn.style.display = 'none';
            console.log('Pose button hidden');
        }
        if (modelStatus) {
            modelStatus.textContent = 'MoveNet Model: Not Available';
            modelStatus.className = 'movenet-status disconnected';
        }
    }
}

// Toggle pose prediction mode
function togglePoseMode() {
    poseMode = !poseMode;
    const btn = document.getElementById('predict-pose-btn');
    const icon = btn.querySelector('i');
    
    if (poseMode) {
        btn.classList.add('active');
        icon.className = 'fas fa-eye-slash';
        btn.querySelector('span').textContent = 'Hide Poses';
        
        // Show pose annotations on current frame
        if (currentFrameIndex >= 0 && currentFrameIndex < frames.length) {
            showPoseAnnotationsOnFrame(currentFrameIndex);
        }
    } else {
        btn.classList.remove('active');
        icon.className = 'fas fa-running';
        btn.querySelector('span').textContent = 'Show Poses';
        
        // Hide pose annotations
        hidePoseAnnotations();
    }
}

// Run pose prediction on current frame
async function predictPoseOnFrame(frameIndex) {
    if (!movenetModelInfo || movenetModelInfo.name.includes('Not Available')) {
        showToast('MoveNet model not available', 'error');
        return;
    }
    
    if (frameIndex < 0 || frameIndex >= frames.length) {
        return;
    }
    
    const frame = frames[frameIndex];
    const confidence = parseFloat(document.getElementById('confidence-threshold')?.value || 0.3);
    
    // Show loading state
    const btn = document.getElementById('predict-pose-btn');
    if (btn) setButtonLoading(btn, true);
    
    try {
        const response = await fetch('/predict_pose_frame', {
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
            framePoseAnnotations.set(frameIndex, data.pose_annotations);
            
            if (poseMode) {
                displayPoseAnnotations(frameIndex, data.pose_annotations);
            }
            
            // Only show toast for errors, not success
        } else {
            showToast(`Pose estimation failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error during pose prediction:', error);
        showToast('Pose prediction failed', 'error');
    } finally {
        // Hide loading state
        const btn = document.getElementById('predict-pose-btn');
        if (btn) setButtonLoading(btn, false);
    }
}

// Display pose annotations on frame
function displayPoseAnnotations(frameIndex, poseAnnotations) {
    console.log('displayPoseAnnotations called with:', poseAnnotations);
    
    if (!poseAnnotations || poseAnnotations.length === 0) {
        console.log('No pose annotations to display');
        return;
    }
    
    const canvas = document.getElementById('pose-canvas');
    const frameImg = document.getElementById('frame-image');
    
    if (!canvas || !frameImg) {
        console.error('Canvas or frame image not found');
        console.log('Canvas:', canvas);
        console.log('Frame image:', frameImg);
        return;
    }
    
    // Wait for frame image to load if it's not ready
    if (!frameImg.complete || frameImg.naturalWidth === 0) {
        console.log('Frame image not loaded yet, waiting...');
        frameImg.onload = () => displayPoseAnnotations(frameIndex, poseAnnotations);
        return;
    }
    
    // Set canvas size to match frame
    canvas.width = frameImg.offsetWidth;
    canvas.height = frameImg.offsetHeight;
    canvas.style.display = 'block';
    
    console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
    console.log('Frame image dimensions:', frameImg.naturalWidth, 'x', frameImg.naturalHeight);
    console.log('Frame image display size:', frameImg.offsetWidth, 'x', frameImg.offsetHeight);
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Scale factors for drawing (scale to match actual frame display size)
    const scaleX = canvas.width / frameImg.naturalWidth;
    const scaleY = canvas.height / frameImg.naturalHeight;
    
    console.log('Scale factors:', scaleX, scaleY);
    
    // Define colors for different persons
    const colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
        '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A'
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
        ctx.lineWidth = 2;
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
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw confidence score
            ctx.fillStyle = 'white';
            ctx.font = '10px Arial';
            ctx.fillText(kpt.confidence.toFixed(2), x + 6, y - 6);
            ctx.fillStyle = color;
        });
        
        // Draw person info
        if (person.keypoints.length > 0) {
            const firstKpt = person.keypoints[0];
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(firstKpt.x * scaleX, firstKpt.y * scaleY - 25, 150, 20);
            
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.fillText(`Person ${personIndex + 1} (${person.num_keypoints} kpts)`, 
                        firstKpt.x * scaleX + 5, firstKpt.y * scaleY - 10);
        }
    });
}

// Show pose annotations on specific frame
async function showPoseAnnotationsOnFrame(frameIndex) {
    // Check if we already have pose annotations for this frame
    if (framePoseAnnotations.has(frameIndex)) {
        displayPoseAnnotations(frameIndex, framePoseAnnotations.get(frameIndex));
        return;
    }
    
    // Try to get cached pose annotations
    const frame = frames[frameIndex];
    try {
        const response = await fetch('/get_cached_pose_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                frame_num: frame.frame_num
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.cached) {
            framePoseAnnotations.set(frameIndex, data.pose_annotations);
            displayPoseAnnotations(frameIndex, data.pose_annotations);
        } else {
            // Run pose prediction
            await predictPoseOnFrame(frameIndex);
        }
    } catch (error) {
        console.error('Error getting cached pose annotations:', error);
        await predictPoseOnFrame(frameIndex);
    }
}

// Hide pose annotations
function hidePoseAnnotations() {
    const canvas = document.getElementById('pose-canvas');
    if (canvas) {
        canvas.style.display = 'none';
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// Initialize pose canvas
function initializePoseCanvas() {
    const frameContainer = document.querySelector('.frame-display');
    if (!frameContainer) {
        console.error('Frame display container not found');
        return;
    }
    
    // Create pose canvas if it doesn't exist
    let poseCanvas = document.getElementById('pose-canvas');
    if (!poseCanvas) {
        poseCanvas = document.createElement('canvas');
        poseCanvas.id = 'pose-canvas';
        poseCanvas.style.position = 'absolute';
        poseCanvas.style.top = '0';
        poseCanvas.style.left = '0';
        poseCanvas.style.pointerEvents = 'none';
        poseCanvas.style.display = 'none';
        poseCanvas.style.zIndex = '10';
        frameContainer.appendChild(poseCanvas);
        console.log('Pose canvas created and added to frame container');
        console.log('Frame container style:', frameContainer.style.position || 'static');
        console.log('Frame container position:', frameContainer.getBoundingClientRect());
    }
}

// Update pose annotations when frame changes
function updatePoseAnnotationsOnFrameChange(frameIndex) {
    if (poseMode) {
        // Add a small delay to ensure frame image is loaded
        setTimeout(() => {
            showPoseAnnotationsOnFrame(frameIndex);
        }, 100);
    } else {
        hidePoseAnnotations();
    }
}

// Initialize MoveNet functionality
document.addEventListener('DOMContentLoaded', function() {
    loadMoveNetModelInfo();
    initializePoseCanvas();
    
    // Add event listener for pose prediction button
    const poseBtn = document.getElementById('predict-pose-btn');
    if (poseBtn) {
        poseBtn.addEventListener('click', togglePoseMode);
    }
});

// Export functions for use in other modules
window.movenet = {
    loadMoveNetModelInfo,
    togglePoseMode,
    predictPoseOnFrame,
    updatePoseAnnotationsOnFrameChange,
    initializePoseCanvas
};