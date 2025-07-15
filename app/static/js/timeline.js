let isDragging = false;
let dragType = null;

function initializeTimeline() {
    const timeline = document.getElementById('timeline');

    timeline.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        const selection = document.getElementById('timeline-selection');
        const leftHandle = selection.querySelector('.left');
        const rightHandle = selection.querySelector('.right');

        if (e.target === leftHandle) {
            isDragging = true;
            dragType = 'left';
        } else if (e.target === rightHandle) {
            isDragging = true;
            dragType = 'right';
        } else if (e.target === selection) {
            isDragging = true;
            dragType = 'move';
        } else if (e.target.classList.contains('timeline-thumbnail') || e.target === timeline) {
            const rect = timeline.getBoundingClientRect();
            const clickPos = (e.clientX - rect.left) / rect.width;
            const clickTime = clickPos * videoDuration;

            segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, clickTime - segmentDuration / 2));
            updateTimeline();
        }
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const timeline = document.getElementById('timeline');
        const rect = timeline.getBoundingClientRect();
        const mousePos = (e.clientX - rect.left) / rect.width;
        const mouseTime = Math.max(0, Math.min(videoDuration, mousePos * videoDuration));

        if (dragType === 'move') {
            segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, mouseTime - segmentDuration / 2));
        } else if (dragType === 'left') {
            const currentEnd = segmentStart + segmentDuration;
            const newStart = Math.min(mouseTime, currentEnd - 1);
            segmentDuration = currentEnd - newStart;
            segmentStart = newStart;
        } else if (dragType === 'right') {
            const newEnd = Math.max(mouseTime, segmentStart + 1);
            segmentDuration = newEnd - segmentStart;
        }

        segmentDuration = Math.max(1, Math.min(60, segmentDuration));
        segmentStart = Math.max(0, Math.min(videoDuration - segmentDuration, segmentStart));

        updateTimeline();
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        dragType = null;
    });
}

function updateTimeline() {
    if (!videoDuration || videoDuration === 0) {
        return;
    }

    const selection = document.getElementById('timeline-selection');
    const startPercent = (segmentStart / videoDuration) * 100;
    const widthPercent = (segmentDuration / videoDuration) * 100;

    selection.style.left = `${startPercent}%`;
    selection.style.width = `${widthPercent}%`;

    document.getElementById('time-start').textContent = formatTime(segmentStart);
    document.getElementById('time-end').textContent = formatTime(segmentStart + segmentDuration);
    document.getElementById('segment-duration').textContent = `Selected: ${Math.round(segmentDuration)}s`;

    document.getElementById('start-time').value = segmentStart.toFixed(1);
    document.getElementById('duration').value = Math.round(segmentDuration);

    const video = document.getElementById('video-player');
    if (video.src && video.readyState >= 1 && !isDragging) {
        video.currentTime = segmentStart;
    }
}

function updateSegmentFromInputs() {
    segmentStart = parseFloat(document.getElementById('start-time').value) || 0;
    segmentDuration = parseInt(document.getElementById('duration').value) || 30;

    segmentStart = Math.max(0, Math.min(videoDuration - 1, segmentStart));
    segmentDuration = Math.max(1, Math.min(60, Math.min(videoDuration - segmentStart, segmentDuration)));

    updateTimeline();
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function renderTimelineThumbnails(thumbnails) {
    const timeline = document.getElementById('timeline');
    const selection = document.getElementById('timeline-selection');

    timeline.querySelectorAll('.timeline-thumbnail').forEach(el => el.remove());

    const fragment = document.createDocumentFragment();
    thumbnails.forEach(thumbData => {
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${thumbData}`;
        img.className = 'timeline-thumbnail';
        img.draggable = false;
        fragment.appendChild(img);
    });

    timeline.insertBefore(fragment, selection);
}
