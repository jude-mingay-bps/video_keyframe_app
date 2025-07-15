function loadRoboflowConfig() {
    const saved = localStorage.getItem('roboflowConfig');
    if (saved) {
        roboflowConfig = JSON.parse(saved);
        document.getElementById('roboflow-url').value = roboflowConfig.url || '';
        document.getElementById('roboflow-api-key').value = roboflowConfig.apiKey || '';
        document.getElementById('roboflow-batch-name').value = roboflowConfig.batchName || '';
        document.getElementById('roboflow-split').value = roboflowConfig.split || 'train';
        updateRoboflowStatus();
    }
}

function saveRoboflowConfig() {
    const url = document.getElementById('roboflow-url').value.trim();
    const apiKey = document.getElementById('roboflow-api-key').value.trim();
    const batchName = document.getElementById('roboflow-batch-name').value.trim();
    const split = document.getElementById('roboflow-split').value;

    if (!url || !apiKey) {
        showToast('Please enter both Roboflow project URL and API key', 'error');
        return;
    }

    roboflowConfig = {
        url: url,
        apiKey: apiKey,
        batchName: batchName,
        split: split,
        isConfigured: true
    };

    localStorage.setItem('roboflowConfig', JSON.stringify(roboflowConfig));
    updateRoboflowStatus();
    showToast('Roboflow configuration saved successfully', 'success');
}

async function testRoboflowConnection() {
    const url = document.getElementById('roboflow-url').value.trim();
    const apiKey = document.getElementById('roboflow-api-key').value.trim();
    const button = event.target;

    if (!url || !apiKey) {
        showToast('Please enter both Roboflow project URL and API key', 'error');
        return;
    }

    setButtonLoading(button, true);
    const loadingToast = showToast('Testing connection...', 'info', 0);

    try {
        const response = await fetch('/test_roboflow', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                api_key: apiKey,
                project_url: url
            })
        });

        const data = await response.json();
        removeToast(loadingToast);

        if (data.success) {
            showToast(data.message, 'success');
            saveRoboflowConfig();
        } else {
            showToast(data.message || 'Connection test failed', 'error');
        }
    } catch (error) {
        removeToast(loadingToast);
        showToast('Error testing connection: ' + error.message, 'error');
    } finally {
        setButtonLoading(button, false);
    }
}

function updateRoboflowStatus() {
    const status = document.getElementById('roboflow-status');
    if (roboflowConfig.isConfigured && roboflowConfig.url && roboflowConfig.apiKey) {
        status.textContent = 'Configured';
        status.className = 'roboflow-status connected';
    } else {
        status.textContent = 'Not Configured';
        status.className = 'roboflow-status disconnected';
    }
}
