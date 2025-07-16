let keyNavigationThrottle = false;
let keyNavigationDelay = 50; // milliseconds

function initializeEventListeners() {
    document.getElementById('confidence-threshold').addEventListener('input', (e) => {
        document.getElementById('confidence-value').textContent = e.target.value;
    });

    document.getElementById('target-fps').addEventListener('input', (e) => {
        document.getElementById('fps-value').textContent = e.target.value;
    });

    document.addEventListener('keydown', (e) => {
        if (document.querySelector('.frame-selector').style.display !== 'block' || !frames.length) return;

        switch(e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                if (!keyNavigationThrottle) {
                    keyNavigationThrottle = true;
                    previousFrame();
                    setTimeout(() => {
                        keyNavigationThrottle = false;
                    }, keyNavigationDelay);
                }
                break;
            case 'ArrowRight':
                e.preventDefault();
                if (!keyNavigationThrottle) {
                    keyNavigationThrottle = true;
                    nextFrame();
                    setTimeout(() => {
                        keyNavigationThrottle = false;
                    }, keyNavigationDelay);
                }
                break;
            case ' ':
                e.preventDefault();
                toggleSelection();
                break;
            case 'p':
            case 'P':
                e.preventDefault();
                if(predictionMode) runPrediction();
                break;
            case 'Enter':
                e.preventDefault();
                finishVideo();
                break;
        }
    });
}
