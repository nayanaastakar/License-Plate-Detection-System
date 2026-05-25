const form = document.getElementById('detect-form');
const statusEl = document.getElementById('status');
const results = document.getElementById('results');
const cards = document.getElementById('cards');
const title = document.getElementById('result-title');
const outputFolder = document.getElementById('output-folder');
const textLink = document.getElementById('text-link');
const useDefault = document.getElementById('use-default');
const useDefaultImage = document.getElementById('use-default-image');
const modeTabs = document.querySelectorAll('.mode-tab');
const uploadPanel = document.getElementById('upload-panel');
const cameraPanel = document.getElementById('camera-panel');
const cameraIndex = document.getElementById('camera-index');
const minArea = document.getElementById('min-area');
const startCamera = document.getElementById('start-camera');
const stopCamera = document.getElementById('stop-camera');
const cameraStream = document.getElementById('camera-stream');
const runDetection = document.getElementById('run-detection');
let cameraPoll = null;

function setStatus(message, state = 'ready') {
    statusEl.textContent = message;
    statusEl.className = `status ${state === 'ready' ? '' : state}`.trim();
}

function renderResults(data) {
    results.hidden = false;
    title.textContent = data.count ? 'Final number plate image' : 'No number plate detected';
    outputFolder.textContent = data.output_folder;
    textLink.href = data.text_url || '#';
    textLink.hidden = !data.text_url;
    cards.innerHTML = '';

    if (!data.detections.length) {
        cards.innerHTML = '<p>No plate region was detected. Try a clearer or closer vehicle video.</p>';
        return;
    }

    const item = data.detections[0];
    const card = document.createElement('article');
    card.className = 'card card-final';
    card.innerHTML = `
            <a href="${item.annotated_frame}" target="_blank">
                <img src="${item.annotated_frame}" alt="Annotated frame ${item.frame}">
            </a>
            <div class="card-body">
                <strong>Final plate from frame ${item.frame}</strong>
                <p class="plate-text">${item.plate_text || 'Text not recognized'}</p>
                <p>Box: ${item.x}, ${item.y}, ${item.width} x ${item.height}</p>
                <a href="${item.crop_file}" target="_blank">
                    <img class="crop" src="${item.crop_file}" alt="Final number plate">
                </a>
            </div>
        `;
    cards.appendChild(card);
}

function switchMode(mode) {
    modeTabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.mode === mode));
    uploadPanel.hidden = mode !== 'upload';
    cameraPanel.hidden = mode !== 'camera';
    results.hidden = mode === 'camera' && !cameraStream.src;
    if (mode === 'camera') {
        setStatus('Camera ready');
    } else {
        stopCameraStream();
        setStatus('Ready');
    }
}

function stopCameraStream() {
    if (cameraPoll) {
        clearInterval(cameraPoll);
        cameraPoll = null;
    }
    cameraStream.removeAttribute('src');
    startCamera.disabled = false;
    stopCamera.disabled = true;
}

async function resetCameraDetection() {
    await fetch('/api/camera/reset', { method: 'POST' });
}

async function refreshCameraDetection() {
    const response = await fetch('/api/camera/latest');
    const data = await response.json();
    if (!data.count) {
        return;
    }

    const detection = data.detection;
    renderResults({
        count: 1,
        detections: [{
            annotated_frame: detection.annotated_frame,
            crop_file: detection.crop_file,
            frame: 'live',
            plate_text: detection.plate_text,
            text_file: detection.text_file,
            x: detection.x,
            y: detection.y,
            width: detection.width,
            height: detection.height,
        }],
        output_folder: detection.output_folder,
        text_url: detection.text_file,
    });
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    runDetection.disabled = true;
    setStatus('Processing...', 'busy');

    try {
        const formData = new FormData(form);
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (data.status !== 'success') {
            throw new Error(data.message);
        }
        renderResults(data);
        setStatus('Complete');
    } catch (error) {
        setStatus(error.message, 'error');
    } finally {
        runDetection.disabled = false;
    }
});

modeTabs.forEach((tab) => {
    tab.addEventListener('click', () => switchMode(tab.dataset.mode));
});

startCamera.addEventListener('click', async () => {
    await resetCameraDetection();
    results.hidden = true;
    const params = new URLSearchParams({
        camera: cameraIndex.value || '0',
        min_area: minArea.value || '120',
        t: Date.now().toString(),
    });
    cameraStream.src = `/camera_feed?${params.toString()}`;
    startCamera.disabled = true;
    stopCamera.disabled = false;
    setStatus('Camera running', 'busy');
    cameraPoll = setInterval(refreshCameraDetection, 2000);
});

cameraStream.addEventListener('error', () => {
    stopCameraStream();
    setStatus('Camera stream failed. Try camera index 0 or close other camera apps.', 'error');
});

stopCamera.addEventListener('click', async () => {
    stopCameraStream();
    await resetCameraDetection();
    setStatus('Camera stopped');
});

useDefault.addEventListener('change', () => {
    if (useDefault.checked) {
        useDefaultImage.checked = false;
    }
});

useDefaultImage.addEventListener('change', () => {
    if (useDefaultImage.checked) {
        useDefault.checked = false;
    }
});
