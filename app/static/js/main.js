// Global state variables
let videos = [];
let currentVideoIndex = 0;
let frames = [];
let currentFrameIndex = 0;
let selectedFrames = new Set();
let framePredictions = new Map();
let currentVideoId = null;
let videoDuration = 0;
let segmentStart = 0;
let segmentDuration = 30;
let predictionMode = false;
let modelInfo = null;
let roboflowConfig = {
    url: '',
    apiKey: '',
    batchName: '',
    split: 'train',
    isConfigured: false
};
let correctionMode = false;
let currentBoundingBoxes = [];
let correctedAnnotations = new Map();

// Initialize the application
window.addEventListener('load', () => {
    loadRoboflowConfig();
    loadModelInfo();
    initializeTimeline();
    updatePredictButtonVisibility();
    initializeEventListeners();
});

function startProcessing() {
    if (videos.length === 0) {
        showToast('No videos to process', 'error');
        return;
    }

    currentVideoIndex = 0;
    showFrameSelector();
    videoDuration = 0;
    loadCurrentVideo();
}

function resetInterface() {
    showMainMenu();
    videos = [];
    updateVideoList();
    correctedAnnotations.clear();
}
