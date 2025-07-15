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
                previousFrame();
                break;
            case 'ArrowRight':
                e.preventDefault();
                nextFrame();
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
