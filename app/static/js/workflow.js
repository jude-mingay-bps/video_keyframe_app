// Workflow management
let currentStep = 1;

function initializeWorkflow() {
    // Show only the first step initially
    showStep(1);
    updateStepProgress();
}

function showStep(stepNumber) {
    // Hide all steps
    const steps = document.querySelectorAll('.workflow-step');
    steps.forEach(step => {
        step.style.display = 'none';
        step.classList.remove('active');
    });
    
    // Show the selected step
    const targetStep = document.getElementById(`step-${stepNumber}`);
    if (targetStep) {
        targetStep.style.display = 'block';
        targetStep.classList.add('active');
        currentStep = stepNumber;
        updateStepProgress();
    }
}

function proceedToStep(stepNumber) {
    // Validate current step before proceeding
    if (validateCurrentStep()) {
        showStep(stepNumber);
        
        // Initialize Step 3 when entering it
        if (stepNumber === 3) {
            currentVideoIndex = 0;
            loadCurrentVideo();
        }
    }
}

function goBackToStep(stepNumber) {
    showStep(stepNumber);
}

function validateCurrentStep() {
    switch (currentStep) {
        case 1:
            // Check if at least one video is added
            const videoItems = document.querySelectorAll('#video-items .video-item');
            if (videoItems.length === 0) {
                showToast('Please add at least one video before proceeding', 'error');
                return false;
            }
            return true;
        case 2:
            // Model configuration is optional, always valid
            return true;
        case 3:
            // Frame selection validation can be added here
            return true;
        case 4:
            // Roboflow configuration validation can be added here
            return true;
        default:
            return true;
    }
}

function updateStepProgress() {
    const stepElements = document.querySelectorAll('.step');
    
    stepElements.forEach((element, index) => {
        const stepNumber = index + 1;
        element.classList.remove('active', 'completed');
        
        if (stepNumber < currentStep) {
            element.classList.add('completed');
        } else if (stepNumber === currentStep) {
            element.classList.add('active');
        }
    });
}

// Override existing functions to work with workflow
function startProcessing() {
    const videoItems = document.querySelectorAll('#video-items .video-item');
    if (videoItems.length === 0) {
        showToast('Please add at least one video first', 'error');
        return;
    }
    
    // Proceed to model configuration
    proceedToStep(2);
}

// Modified video addition functions
async function addYouTubeVideo() {
    const urlInput = document.getElementById('youtube-url');
    const url = urlInput.value.trim();
    const button = event.target;
    
    if (!url) {
        showToast('Please enter a YouTube URL', 'error');
        return;
    }
    
    // Validate YouTube URL
    if (!isValidYouTubeUrl(url)) {
        showToast('Please enter a valid YouTube URL', 'error');
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
            addVideoToQueue(data.video.name, 'youtube');
            urlInput.value = '';
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
        showToast('Please select a video file', 'error');
        return;
    }
    
    // Validate file type
    if (!isValidVideoFile(file)) {
        showToast('Please select a valid video file', 'error');
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
            addVideoToQueue(data.video.name, 'upload');
            fileInput.value = '';
            updateFileDisplay(); // Reset file display
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

function addVideoToQueue(videoName, type) {
    const videoItems = document.getElementById('video-items');
    const videoItem = document.createElement('div');
    videoItem.className = 'video-item';
    videoItem.dataset.videoIndex = videos.length - 1; // Store the index for removal
    
    videoItem.innerHTML = `
        <span>${videoName}</span>
        <button onclick="removeVideoFromQueue(this)" class="btn-secondary">Remove</button>
    `;
    
    videoItems.appendChild(videoItem);
    
    // Show the start processing button
    const startBtn = document.getElementById('start-btn');
    startBtn.style.display = 'block';
}

function removeVideoFromQueue(button) {
    const videoItem = button.parentElement;
    const videoIndex = parseInt(videoItem.dataset.videoIndex);
    
    // Remove from global videos array
    videos.splice(videoIndex, 1);
    
    // Update all remaining video items' indices
    const videoItems = document.querySelectorAll('#video-items .video-item');
    videoItems.forEach((item, index) => {
        if (index > videoIndex) {
            item.dataset.videoIndex = index - 1;
        }
    });
    
    // Remove the UI element
    videoItem.remove();
    
    // Hide start button if no videos left
    const remainingItems = document.querySelectorAll('#video-items .video-item');
    const startBtn = document.getElementById('start-btn');
    if (remainingItems.length === 0) {
        startBtn.style.display = 'none';
    }
}

function isValidYouTubeUrl(url) {
    const regex = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)/;
    return regex.test(url);
}

function isValidVideoFile(file) {
    const validTypes = [
        'video/mp4', 'video/webm', 'video/ogg', 'video/avi', 'video/mov', 'video/wmv',
        'video/x-msvideo', 'video/quicktime', 'video/x-ms-wmv', 'video/x-flv',
        'video/3gpp', 'video/x-matroska', 'video/mp2t'
    ];
    
    // Check MIME type first
    if (validTypes.includes(file.type)) {
        return true;
    }
    
    // Fallback to extension checking if MIME type is unknown
    const validExtensions = ['.mp4', '.webm', '.ogg', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.3gp', '.m4v'];
    const fileName = file.name.toLowerCase();
    return validExtensions.some(ext => fileName.endsWith(ext));
}

// YouTube title extraction is handled by the backend now

// This function is now integrated into uploadFile() above

function showToast(message, type = 'info', duration = 3000, persistent = false) {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add to container
    const container = document.getElementById('toast-container');
    container.appendChild(toast);
    
    // Auto remove after duration (unless persistent)
    if (!persistent && duration > 0) {
        setTimeout(() => {
            toast.remove();
        }, duration);
    }
    
    return toast; // Return for manual removal
}

function removeToast(toast) {
    if (toast && toast.parentElement) {
        toast.remove();
    }
}

function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.classList.add('button-loading');
        button.disabled = true;
    } else {
        button.classList.remove('button-loading');
        button.disabled = false;
    }
}

// Model tabs functionality
function initializeModelTabs() {
    const modelTabs = document.querySelectorAll('.model-tab');
    const modelConfigs = document.querySelectorAll('.model-config');
    
    modelTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const model = this.dataset.model;
            
            // Remove active class from all tabs and configs
            modelTabs.forEach(t => t.classList.remove('active'));
            modelConfigs.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding config
            this.classList.add('active');
            document.getElementById(`${model}-config`).classList.add('active');
        });
    });
}

// Update file display when file is selected
function updateFileDisplay() {
    const fileInput = document.getElementById('file-upload');
    const fileDisplay = document.getElementById('file-display');
    
    if (fileInput.files && fileInput.files[0]) {
        const file = fileInput.files[0];
        fileDisplay.querySelector('span').textContent = file.name;
        fileDisplay.classList.add('has-file');
    } else {
        fileDisplay.querySelector('span').textContent = 'Choose video file...';
        fileDisplay.classList.remove('has-file');
    }
}

// Initialize workflow when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeWorkflow();
    initializeModelTabs();
});