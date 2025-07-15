async function addYouTubeVideo() {
    const url = document.getElementById('youtube-url').value.trim();
    const button = event.target;

    if (!url) {
        showToast('Please enter a YouTube URL', 'error');
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
            updateVideoList();
            document.getElementById('youtube-url').value = '';
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
        showToast('Please select a file', 'error');
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
            updateVideoList();
            fileInput.value = '';
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

function removeVideo(index) {
    videos.splice(index, 1);
    updateVideoList();
}

async function loadCurrentVideo() {
    if (currentVideoIndex >= videos.length) {
        showToast('All videos processed!', 'success');
        resetInterface();
        return;
    }

    const video = videos[currentVideoIndex];
    currentVideoId = video.id;
    document.getElementById('current-video-title').textContent = `Processing: ${video.name}`;
    frames = [];
    currentFrameIndex = 0;
    selectedFrames.clear();
    framePredictions.clear();
    document.getElementById('frame-viewer').style.display = 'none';
    hideAnnotations();

    const timeline = document.getElementById('timeline');
    timeline.querySelectorAll('.timeline-thumbnail').forEach(el => el.remove());

    try {
        const infoResponse = await fetch('/get_video_info', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: currentVideoId })
        });

        const infoData = await infoResponse.json();
        if (infoData.success) {
            videoDuration = infoData.duration;
            document.getElementById('video-duration').textContent = `Duration: ${formatTime(videoDuration)}`;

            const videoPlayer = document.getElementById('video-player');
            videoPlayer.src = `/video/${currentVideoId}`;
            videoPlayer.load();

            videoPlayer.addEventListener('loadedmetadata', () => {
                segmentStart = 0;
                segmentDuration = Math.min(30, videoDuration);
                updateTimeline();
            }, { once: true });

            videoPlayer.addEventListener('error', (e) => {
                console.error('Video load error:', e);
                showToast('Error loading video preview.', 'warning');
            }, { once: true });

            const thumbResponse = await fetch('/get_timeline_thumbnails', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_id: currentVideoId })
            });
            const thumbData = await thumbResponse.json();
            if (thumbData.success && thumbData.thumbnails.length > 0) {
                renderTimelineThumbnails(thumbData.thumbnails);
            } else {
                console.error('Failed to load timeline thumbnails:', thumbData.error);
            }

        } else {
            showToast('Error loading video info: ' + (infoData.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        showToast('Error loading video info: ' + error.message, 'error');
    }
}

async function finishVideo() {
    if (selectedFrames.size === 0) {
        if (!confirm('No frames selected. Skip this video?')) {
            return;
        }
    } else {
        const uploadToRoboflow = roboflowConfig.isConfigured && roboflowConfig.apiKey && roboflowConfig.url;

        let uploadToast = null;
        if (uploadToRoboflow) {
            uploadToast = showToast(`Saving ${selectedFrames.size} frames and uploading to Roboflow...`, 'info', 0, true);
        } else {
            uploadToast = showToast(`Saving ${selectedFrames.size} frames...`, 'info', 0);
        }

        const finalRoboflowConfig = {
            ...roboflowConfig,
            batchName: document.getElementById('roboflow-batch-name').value.trim(),
            split: document.getElementById('roboflow-split').value
        };

        const selectedFrameData = Array.from(selectedFrames).map(frameIndex => {
            const frameData = {
                ...frames[frameIndex],
                frameIndex: frameIndex
            };

            if (framePredictions.has(frameIndex)) {
                const prediction = framePredictions.get(frameIndex);
                const frameKey = `${currentVideoId}_${frameIndex}`;
                const corrections = correctedAnnotations.get(frameKey) || new Set();

                const correctedAnnotationsList = prediction.annotations.map((ann, index) => {
                    if (corrections.has(index)) {
                        return {
                            ...ann,
                            class_name: 'Other',
                            class_id: 999,
                            confidence: 1.0,
                            was_corrected: true
                        };
                    }
                    return ann;
                });

                frameData.predictions = {
                    annotations: correctedAnnotationsList,
                    annotated_frame: prediction.annotated_frame || null
                };
            } else {
                frameData.predictions = null;
            }

            return frameData;
        });

        try {
            const response = await fetch('/save_frames', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_id: currentVideoId,
                    frames: selectedFrameData,
                    upload_to_roboflow: uploadToRoboflow,
                    roboflow_config: uploadToRoboflow ? finalRoboflowConfig : null
                })
            });

            const data = await response.json();
            removeToast(uploadToast);

            if (data.success) {
                let message = `Saved ${data.frame_count} frames to ${data.output_dir}`;
                let toastType = 'success';

                if (data.roboflow_results) {
                    const uploaded = data.roboflow_results.filter(r => r.success).length;
                    const failed = data.roboflow_results.filter(r => !r.success).length;

                    if (failed > 0) {
                        message += `. Roboflow: ${uploaded} uploaded, ${failed} failed`;
                        toastType = 'warning';
                    } else {
                        message += `. All ${uploaded} frames uploaded to Roboflow.`;
                    }
                }

                showToast(message, toastType, 10000);
            } else {
                showToast('Error saving frames: ' + (data.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            removeToast(uploadToast);
            showToast('Error saving frames: ' + error.message, 'error');
        }
    }

    currentVideoIndex++;
    loadCurrentVideo();
}
