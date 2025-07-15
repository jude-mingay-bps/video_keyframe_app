// Toast Notification System
function showToast(message, type = 'info', duration = 5000, showProgress = false) {
    const toastContainer = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    };

    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-message">${message}</div>
            <button class="toast-close" onclick="removeToast(this.parentElement.parentElement)">×</button>
        </div>
        ${showProgress ? '<div class="download-progress"><div class="download-progress-fill"></div></div>' : ''}
    `;

    toastContainer.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);

    if (duration > 0) {
        setTimeout(() => removeToast(toast), duration);
    }

    return toast;
}

function removeToast(toast) {
    toast.classList.add('hide');
    setTimeout(() => {
        if (toast.parentElement) {
            toast.parentElement.removeChild(toast);
        }
    }, 400);
}

function updateToastProgress(toast, progress) {
    const progressFill = toast.querySelector('.download-progress-fill, .upload-progress-fill');
    if (progressFill) {
        progressFill.style.width = `${progress}%`;
    }
}

function setButtonLoading(button, loading) {
    if (loading) {
        button.disabled = true;
        button.classList.add('button-loading');
        button.dataset.originalText = button.textContent;
        button.textContent = '';
    } else {
        button.disabled = false;
        button.classList.remove('button-loading');
        if (button.dataset.originalText) {
            button.textContent = button.dataset.originalText;
            delete button.dataset.originalText;
        }
    }
}

function updateVideoList() {
    const videoList = document.getElementById('video-list');
    const videoItems = document.getElementById('video-items');

    if (videos.length > 0) {
        videoList.style.display = 'block';
        videoItems.innerHTML = videos.map((video, index) => `
            <div class="video-item">
                <span>${video.name}</span>
                <button onclick="removeVideo(${index})">Remove</button>
            </div>
        `).join('');
    } else {
        videoList.style.display = 'none';
    }
}

function showFrameSelector() {
    document.querySelector('.frame-selector').style.display = 'block';
    document.querySelector('.upload-section').style.display = 'none';
    document.getElementById('video-list').style.display = 'none';
    document.querySelector('.roboflow-section').style.display = 'none';
    document.querySelector('.yolo-section').style.display = 'none';
    document.querySelector('header').style.display = 'none';
}

function showMainMenu() {
    document.querySelector('.frame-selector').style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    document.querySelector('.roboflow-section').style.display = 'block';
    document.querySelector('.yolo-section').style.display = 'block';
    document.querySelector('header').style.display = 'block';
    if (videos.length > 0) {
        document.getElementById('video-list').style.display = 'block';
    }
}
