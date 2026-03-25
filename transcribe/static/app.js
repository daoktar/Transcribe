/* ============================================================
   Media Transcriber - app.js
   Vanilla JS single-page app communicating with FastAPI backend
   via REST + SSE.
   ============================================================ */

// --------------- App State ---------------

const state = {
    jobId: null,
    filenames: [],
    files: [],
    nativePaths: [],
    activeTab: 'upload',
    activeSubtab: 'transcript',
    currentFileIndex: 0,
    eventSource: null,
    lastStatuses: [],
};

// --------------- Utilities ---------------

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function isNative() {
    return !!(window.pywebview && window.pywebview.api);
}

// --------------- Initialization ---------------

document.addEventListener('DOMContentLoaded', async () => {
    await loadConfig();
    setupTabNavigation();
    setupDropZone();
    setupTranscribeButton();
    setupCancelButton();
    setupCopyButton();
    setupDownloadButton();
    setupRetryButton();
    setupTranscriptSubtabs();
    setupMobileLabels();
});

// --------------- Config ---------------

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const config = await res.json();

        // Populate format tags
        const tagsEl = document.getElementById('format-tags');
        const extensions = config.supported_extensions || config.extensions || [];
        if (extensions.length && tagsEl) {
            const shown = extensions.slice(0, 5);
            const more = extensions.length - shown.length;
            let html = shown.map(ext => `<span class="format-tag">${escapeHtml(ext.replace('.', '').toUpperCase())}</span>`).join('');
            if (more > 0) {
                const remaining = extensions.slice(5).map(ext => ext.replace('.', '').toUpperCase());
                html += `<span class="format-tag" title="${escapeHtml(remaining.join(', '))}">+${more} More</span>`;
            }
            tagsEl.innerHTML = html;
        }

        // Populate model select
        const models = config.model_choices || config.models || [];
        const modelSelect = document.getElementById('model-select');
        if (models.length && modelSelect) {
            modelSelect.innerHTML = models
                .map(m => `<option value="${escapeHtml(m)}"${m === 'large-v3' ? ' selected' : ''}>${escapeHtml(m)}</option>`)
                .join('');
        }
    } catch (err) {
        console.error('Failed to load config:', err);
    }
}

// --------------- Tab Navigation ---------------

function setupTabNavigation() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
}

function switchTab(tabName) {
    state.activeTab = tabName;

    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `tab-${tabName}`);
    });
}

// --------------- Drop Zone ---------------

function setupDropZone() {
    const zone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    // Click to browse
    zone.addEventListener('click', async (e) => {
        if (e.target === fileInput) return;

        // Native file picker via pywebview
        if (isNative() && window.pywebview.api.pick_files) {
            try {
                const paths = await window.pywebview.api.pick_files();
                if (paths && paths.length > 0) {
                    state.nativePaths = paths;
                    state.files = [];
                    state.filenames = paths.map(p => p.split('/').pop().split('\\').pop());
                    updateDropZoneFiles(state.filenames);
                }
            } catch (err) {
                console.error('Native file picker error:', err);
            }
            return;
        }

        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files);
        if (files.length === 0) return;
        state.files = files;
        state.nativePaths = [];
        state.filenames = files.map(f => f.name);
        updateDropZoneFiles(state.filenames);
    });

    // Drag events
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;
        state.files = files;
        state.nativePaths = [];
        state.filenames = files.map(f => f.name);
        updateDropZoneFiles(state.filenames);
    });
}

function updateDropZoneFiles(filenames) {
    const zone = document.getElementById('drop-zone');
    const transcribeBtn = document.getElementById('transcribe-btn');

    if (filenames.length === 0) {
        zone.classList.remove('has-files');
        zone.innerHTML = `
            <div class="drop-zone-content">
                <svg class="drop-icon" width="64" height="64" viewBox="0 0 64 64" fill="none">
                    <rect x="8" y="8" width="48" height="48" rx="8" stroke="#3e90ff" stroke-width="2" stroke-dasharray="4 4"/>
                    <path d="M32 22v20M22 32l10-10 10 10" stroke="#3e90ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <p class="drop-text">Drop files here</p>
                <p class="drop-subtext">or click to browse</p>
            </div>
            <input type="file" id="file-input" multiple hidden>
        `;
        transcribeBtn.disabled = true;
        return;
    }

    zone.classList.add('has-files');
    const content = zone.querySelector('.drop-zone-content');
    if (content) {
        content.innerHTML = `
            <svg class="drop-icon" width="64" height="64" viewBox="0 0 64 64" fill="none">
                <rect x="8" y="8" width="48" height="48" rx="8" stroke="#3e90ff" stroke-width="2"/>
                <path d="M20 34l8 8 16-16" stroke="#3e90ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <p class="drop-text">${filenames.length} file(s) selected</p>
            <p class="drop-subtext">Click to change or drop new files</p>
        `;
    }
    transcribeBtn.disabled = false;
}

// --------------- Transcribe ---------------

function setupTranscribeButton() {
    const btn = document.getElementById('transcribe-btn');
    btn.addEventListener('click', async () => {
        if (state.files.length === 0 && state.nativePaths.length === 0) return;

        btn.disabled = true;

        try {
            // Step 1: Upload files
            const formData = new FormData();
            if (state.nativePaths.length > 0) {
                // Native mode: send paths as form fields
                state.nativePaths.forEach(p => formData.append('paths', p));
            } else {
                state.files.forEach(f => formData.append('files', f));
            }

            const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
            if (!uploadRes.ok) throw new Error('Upload failed');
            const uploadData = await uploadRes.json();

            state.jobId = uploadData.job_id;
            state.filenames = uploadData.filenames;

            // Step 2: Start transcription
            const settings = {
                model: document.getElementById('model-select').value,
                language: document.getElementById('language-input').value.trim(),
                diarize: document.getElementById('diarize-checkbox').checked,
                hf_token: document.getElementById('hf-token').value.trim(),
                save_alongside: false,
                original_paths: state.nativePaths.length > 0 ? state.nativePaths : [],
            };

            const transcribeRes = await fetch(`/api/transcribe/${state.jobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings),
            });
            if (!transcribeRes.ok) throw new Error('Transcribe request failed');

            // Step 3: Switch to process tab and connect SSE
            switchTab('process');
            resetProgressUI();
            connectSSE(state.jobId);

            // Show cancel button
            const cancelBtn = document.getElementById('cancel-btn');
            cancelBtn.hidden = false;
            cancelBtn.disabled = false;
            cancelBtn.textContent = 'Cancel Remaining';

        } catch (err) {
            console.error('Transcription start error:', err);
            alert('Error: ' + err.message);
            btn.disabled = false;
        }
    });
}

function resetProgressUI() {
    updateProgressBar(0, 'Waiting to start...');
    document.getElementById('job-queue-list').innerHTML =
        '<p class="queue-empty">Preparing files...</p>';
}

// --------------- SSE Progress ---------------

function connectSSE(jobId) {
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }

    const es = new EventSource(`/api/progress/${jobId}`);
    state.eventSource = es;

    es.onmessage = (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch {
            return;
        }

        if (data.type === 'progress') {
            updateProgressBar(data.fraction, data.message);
            updateJobQueue(state.filenames, data.statuses, data.errors);
            if (data.statuses) state.lastStatuses = data.statuses;
        }

        if (data.type === 'complete') {
            es.close();
            state.eventSource = null;
            if (!data.statuses && state.lastStatuses.length) {
                data.statuses = state.lastStatuses;
            }
            onTranscriptionComplete(data);
        }

        if (data.type === 'error') {
            es.close();
            state.eventSource = null;
            alert('Error: ' + data.message);
            document.getElementById('transcribe-btn').disabled = false;
        }
    };

    es.onerror = () => {
        es.close();
        state.eventSource = null;
        console.error('SSE connection lost');
    };
}

// --------------- Progress Bar ---------------

function updateProgressBar(fraction, message) {
    const pct = Math.max(0, Math.min(100, Math.round(fraction * 100)));
    const fill = document.getElementById('progress-bar-fill');
    const msg = document.getElementById('progress-message');
    const pctEl = document.getElementById('progress-pct');

    fill.style.width = pct + '%';
    msg.textContent = message;
    pctEl.textContent = pct + '%';
    pctEl.style.visibility = (pct === 0 || pct === 100) ? 'hidden' : 'visible';

    if (pct >= 100) {
        fill.classList.add('done');
    } else {
        fill.classList.remove('done');
    }
}

// --------------- Job Queue ---------------

function updateJobQueue(filenames, statuses, errors) {
    const list = document.getElementById('job-queue-list');
    list.innerHTML = filenames.map((name, i) => {
        const status = statuses[i] || 'pending';
        const errorNote = errors[i]
            ? `<div class="job-note">${escapeHtml(errors[i])}</div>`
            : '';
        const miniBar = status === 'processing'
            ? '<div class="job-progress-mini"><div class="job-progress-mini-fill"></div></div>'
            : '';
        return `
            <div class="job-card ${status}">
                <div class="job-card-row">
                    <span class="status-badge ${status}">${status.toUpperCase()}</span>
                </div>
                <div class="job-filename">${escapeHtml(name)}</div>
                ${errorNote}
                ${miniBar}
            </div>
        `;
    }).join('');
}

// --------------- Transcription Complete ---------------

async function onTranscriptionComplete(data) {
    const firstIndex = data.first_result_index || 0;
    state.currentFileIndex = firstIndex;
    state.lastStatuses = data.statuses || [];

    // Switch to review tab
    switchTab('review');

    // Show toolbar
    document.getElementById('review-toolbar').hidden = false;

    // Render file switcher pills
    renderFileSwitcher(state.filenames, firstIndex, state.lastStatuses);

    // Load first result
    await loadFileResult(firstIndex);

    // Re-enable transcribe button for next run
    document.getElementById('transcribe-btn').disabled = false;
}

// --------------- Review Tab ---------------

let _loadRequestId = 0;

async function loadFileResult(fileIndex) {
    const requestId = ++_loadRequestId;
    state.currentFileIndex = fileIndex;

    try {
        const res = await fetch(`/api/result/${state.jobId}/${fileIndex}`);
        if (!res.ok) throw new Error('Failed to load result');
        if (requestId !== _loadRequestId) return; // stale request
        const result = await res.json();

        // Transcript text
        document.getElementById('transcript-text').textContent = result.text || '';

        // Speaker text — show empty state if no speakers
        const speakerTextEl = document.getElementById('speaker-text');
        if (result.has_speakers) {
            speakerTextEl.textContent = result.speaker_text || '';
        } else {
            speakerTextEl.innerHTML = '<div class="speaker-empty-state">No speaker data available. Enable speaker detection and retry.</div>';
        }

        // Show/hide speaker sub-tab
        const tabsEl = document.getElementById('transcript-tabs');
        if (result.has_speakers) {
            tabsEl.hidden = false;
        } else {
            tabsEl.hidden = true;
            switchSubtab('transcript');
        }

        // Show/hide retry button — only when diarize was requested AND there was an error
        const retryBtn = document.getElementById('retry-btn');
        retryBtn.hidden = !(result.diarize_requested && result.diarize_error);

        // Render summary
        renderSummary(result);

        // Update file switcher active state
        renderFileSwitcher(state.filenames, fileIndex, state.lastStatuses);

        // Show transcript content
        document.getElementById('transcript-content').hidden = false;

    } catch (err) {
        if (requestId === _loadRequestId) {
            console.error('Failed to load file result:', err);
        }
    }
}

function renderSummary(result) {
    const bar = document.getElementById('summary-bar');
    bar.hidden = false;

    const lang = result.language || 'Unknown';
    const segs = result.segments_count ?? 0;
    const spk = result.speakers != null ? result.speakers : 'N/A';

    bar.innerHTML = `
        <div class="summary-stat"><strong>${escapeHtml(lang)}</strong> language</div>
        <div class="summary-stat"><strong>${escapeHtml(String(segs))}</strong> segments</div>
        <div class="summary-stat"><strong>${escapeHtml(String(spk))}</strong> speakers</div>
    `;
}

function renderFileSwitcher(filenames, activeIndex, statuses) {
    const container = document.getElementById('file-switcher');
    if (filenames.length <= 1) {
        container.hidden = true;
        return;
    }
    container.hidden = false;
    container.innerHTML = filenames.map((name, i) => {
        const active = i === activeIndex ? 'active' : '';
        const status = (statuses && statuses[i]) || 'done';
        const statusClass = status === 'error' ? 'error' : (status === 'done' ? 'done' : 'warning');
        return `
            <button class="file-pill ${active}" data-index="${i}">
                <span class="pill-status ${statusClass}"></span>
                <span class="pill-name">${escapeHtml(name)}</span>
            </button>
        `;
    }).join('');

    container.querySelectorAll('.file-pill').forEach(pill => {
        pill.addEventListener('click', () => {
            const idx = parseInt(pill.dataset.index, 10);
            loadFileResult(idx);
        });
    });
}

// --------------- Transcript Sub-tabs ---------------

function setupTranscriptSubtabs() {
    const tabs = document.querySelectorAll('.transcript-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            switchSubtab(tab.dataset.subtab);
        });
    });
}

function switchSubtab(subtab) {
    state.activeSubtab = subtab;

    document.querySelectorAll('.transcript-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.subtab === subtab);
    });

    const transcriptText = document.getElementById('transcript-text');
    const speakerText = document.getElementById('speaker-text');

    if (subtab === 'transcript') {
        transcriptText.hidden = false;
        speakerText.hidden = true;
    } else {
        transcriptText.hidden = true;
        speakerText.hidden = false;
    }
}

// --------------- Mobile Button Labels ---------------

function setupMobileLabels() {
    function updateLabels() {
        const narrow = window.innerWidth <= 480;
        const downloadBtn = document.getElementById('download-btn');
        const copyBtn = document.getElementById('copy-btn');
        if (downloadBtn) downloadBtn.textContent = narrow ? 'Download' : 'Download ALL (.zip)';
        if (copyBtn && copyBtn.textContent !== 'Copied!' && copyBtn.textContent !== 'Copy failed') {
            copyBtn.textContent = narrow ? 'Copy' : 'Copy Text';
        }
    }
    updateLabels();
    window.addEventListener('resize', updateLabels);
}

// --------------- Copy Button ---------------

function copyToClipboard(text) {
    const btn = document.getElementById('copy-btn');

    function showFeedback(ok) {
        btn.textContent = ok ? 'Copied!' : 'Copy failed';
        const narrow = window.innerWidth <= 480;
        setTimeout(() => { btn.textContent = narrow ? 'Copy' : 'Copy Text'; }, 1500);
    }

    function fallbackCopy() {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.cssText = 'position:fixed;opacity:0;left:-9999px';
        document.body.appendChild(ta);
        ta.select();
        try {
            showFeedback(document.execCommand('copy'));
        } catch {
            showFeedback(false);
        }
        ta.remove();
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
            .then(() => showFeedback(true))
            .catch(() => fallbackCopy());
    } else {
        fallbackCopy();
    }
}

function setupCopyButton() {
    document.getElementById('copy-btn').addEventListener('click', () => {
        const activeText = state.activeSubtab === 'speakers'
            ? document.getElementById('speaker-text').textContent
            : document.getElementById('transcript-text').textContent;
        copyToClipboard(activeText);
    });
}

// --------------- Download Button ---------------

function setupDownloadButton() {
    document.getElementById('download-btn').addEventListener('click', async () => {
        if (!state.jobId) return;

        // Native mode: use pywebview save dialog
        if (isNative() && window.pywebview.api.save_transcript) {
            try {
                await window.pywebview.api.save_transcript(`/api/download/${state.jobId}`);
            } catch (err) {
                console.error('Native download error:', err);
            }
            return;
        }

        // Browser mode: direct download
        window.location.href = `/api/download/${state.jobId}`;
    });
}

// --------------- Cancel Button ---------------

function setupCancelButton() {
    const btn = document.getElementById('cancel-btn');
    btn.addEventListener('click', async () => {
        if (!state.jobId) return;
        if (!confirm('Cancel all remaining files in the queue?')) return;

        btn.disabled = true;
        btn.textContent = 'Cancelling...';

        try {
            await fetch(`/api/cancel/${state.jobId}`, { method: 'POST' });
        } catch (err) {
            console.error('Cancel error:', err);
        }
    });
}

// --------------- Retry Button ---------------

function setupRetryButton() {
    document.getElementById('retry-btn').addEventListener('click', async () => {
        if (!state.jobId) return;

        const hfToken = document.getElementById('hf-token').value.trim();
        const retryBtn = document.getElementById('retry-btn');
        retryBtn.disabled = true;

        try {
            const res = await fetch(`/api/retry-diarize/${state.jobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hf_token: hfToken }),
            });
            if (!res.ok) throw new Error('Retry request failed');

            // Switch to process tab and reconnect SSE
            switchTab('process');
            resetProgressUI();
            connectSSE(state.jobId);

            // Override complete handler to go back to review
            const origOnMessage = state.eventSource.onmessage;
            state.eventSource.onmessage = (event) => {
                let data;
                try {
                    data = JSON.parse(event.data);
                } catch {
                    return;
                }

                if (data.type === 'progress') {
                    updateProgressBar(data.fraction, data.message);
                    updateJobQueue(state.filenames, data.statuses, data.errors);
                }

                if (data.type === 'complete') {
                    state.eventSource.close();
                    state.eventSource = null;
                    switchTab('review');
                    loadFileResult(state.currentFileIndex);
                    retryBtn.disabled = false;
                }

                if (data.type === 'error') {
                    state.eventSource.close();
                    state.eventSource = null;
                    alert('Retry error: ' + data.message);
                    retryBtn.disabled = false;
                }
            };

        } catch (err) {
            console.error('Retry error:', err);
            alert('Retry failed: ' + err.message);
            retryBtn.disabled = false;
        }
    });
}
