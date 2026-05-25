const form = document.getElementById('detect-form');
const statusEl = document.getElementById('status');
const results = document.getElementById('results');
const cards = document.getElementById('cards');
const title = document.getElementById('result-title');
const outputFolder = document.getElementById('output-folder');
const textLink = document.getElementById('text-link');

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

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = form.querySelector('button');
    button.disabled = true;
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
        button.disabled = false;
    }
});
