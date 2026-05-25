const form = document.getElementById('detect-form');
const statusEl = document.getElementById('status');
const results = document.getElementById('results');
const cards = document.getElementById('cards');
const title = document.getElementById('result-title');
const outputFolder = document.getElementById('output-folder');
const csvLink = document.getElementById('csv-link');
const jsonLink = document.getElementById('json-link');

function setStatus(message, state = 'ready') {
    statusEl.textContent = message;
    statusEl.className = `status ${state === 'ready' ? '' : state}`.trim();
}

function renderResults(data) {
    results.hidden = false;
    title.textContent = `${data.count} detection${data.count === 1 ? '' : 's'} found`;
    outputFolder.textContent = data.output_folder;
    csvLink.href = data.csv_url;
    jsonLink.href = data.json_url;
    cards.innerHTML = '';

    if (!data.detections.length) {
        cards.innerHTML = '<p>No plate regions were detected. Try a clearer or closer vehicle video.</p>';
        return;
    }

    data.detections.forEach((item) => {
        const card = document.createElement('article');
        card.className = 'card';
        card.innerHTML = `
            <a href="${item.annotated_frame}" target="_blank">
                <img src="${item.annotated_frame}" alt="Annotated frame ${item.frame}">
            </a>
            <div class="card-body">
                <strong>Frame ${item.frame}</strong>
                <p>Box: ${item.x}, ${item.y}, ${item.width} x ${item.height}</p>
                <a href="${item.crop_file}" target="_blank">
                    <img class="crop" src="${item.crop_file}" alt="Plate crop ${item.index}">
                </a>
            </div>
        `;
        cards.appendChild(card);
    });
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
