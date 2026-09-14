// WebSocket connection
const socket = io();

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const mediaPreview = document.getElementById('mediaPreview');
const previewImg = document.getElementById('previewImg');
const previewVideo = document.getElementById('previewVideo');
const previewAudio = document.getElementById('previewAudio');
const audioFileName = document.getElementById('audioFileName');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const urlInput = document.getElementById('urlInput');

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const resultsCard = document.getElementById('resultsCard');
const agentStatus = document.getElementById('agentStatus');
const agentStatusText = document.getElementById('agentStatusText');

// State
let uploadedFile = null;
let uploadedFilePath = null;
let analysisComplete = false;

// ===== WebSocket Events =====
socket.on('connect', () => {
    console.log('Connected to server');
    statusDot.classList.add('connected');
    statusText.textContent = 'Connected';
    addSystemMessage('Connected to Deepfake Detection Agent System');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    statusDot.classList.remove('connected');
    statusText.textContent = 'Disconnected';
});

socket.on('status', (data) => {
    console.log('Status:', data);
});

socket.on('agent_status', (data) => {
    console.log('Agent status:', data);
    showAgentStatus(data.message);
});

socket.on('detection_result', (data) => {
    console.log('Detection result:', data);
    displayResults(data);
});

socket.on('analysis_result', (data) => {
    console.log('Analysis result:', data);
    displayExplanation(data.explanation);
    hideAgentStatus();
    analysisComplete = true;
    chatInput.disabled = false;
    sendBtn.disabled = false;
});

socket.on('question_answer', (data) => {
    console.log('Question answer:', data);
    addAgentMessage(data.answer);
    hideAgentStatus();
});

socket.on('error', (data) => {
    console.error('Error:', data);
    addSystemMessage(`Error: ${data.message}`, 'error');
    hideAgentStatus();
});

socket.on('session_saved', (data) => {
    console.log('Session saved:', data);
    addSystemMessage(`Session saved: ${data.session_id}`);
});

// ===== File Upload Handling =====
if (browseBtn) {
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
}

uploadArea.addEventListener('click', (e) => {
    if (e.target.id !== 'removeBtn' && !e.target.classList.contains('btn-browse')) {
        fileInput.click();
    }
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

if (urlInput) {
    urlInput.addEventListener('input', () => {
        if (urlInput.value.trim().length > 5 && !uploadedFile) {
            analyzeBtn.disabled = false;
        } else if (!uploadedFile) {
            analyzeBtn.disabled = true;
        }
    });
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearUpload();
});

function handleFile(file) {
    // Validate file size (64MB max)
    if (file.size > 64 * 1024 * 1024) {
        alert('File size must be less than 64MB');
        return;
    }

    uploadedFile = file;
    document.querySelector('.upload-content').style.display = 'none';
    mediaPreview.style.display = 'block';

    previewImg.style.display = 'none';
    previewVideo.style.display = 'none';
    previewAudio.style.display = 'none';

    const reader = new FileReader();

    if (file.type.startsWith('image/')) {
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
        reader.onload = (e) => {
            previewVideo.src = e.target.result;
            previewVideo.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else if (file.type.startsWith('audio/')) {
        audioFileName.textContent = file.name;
        previewAudio.style.display = 'flex';
    } else {
        // Fallback for unrecognized extension
        audioFileName.textContent = file.name;
        previewAudio.style.display = 'flex';
    }

    analyzeBtn.disabled = false;
}

function clearUpload() {
    uploadedFile = null;
    uploadedFilePath = null;
    fileInput.value = '';
    if (urlInput) urlInput.value = '';
    
    document.querySelector('.upload-content').style.display = 'block';
    mediaPreview.style.display = 'none';
    previewImg.src = '';
    previewVideo.src = '';
    
    analyzeBtn.disabled = true;
    resultsCard.style.display = 'none';
    analysisComplete = false;
    chatInput.disabled = true;
    sendBtn.disabled = true;
}

// ===== Analyze Media =====
analyzeBtn.addEventListener('click', async () => {
    if (!uploadedFile && (!urlInput || !urlInput.value.trim())) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Uploading...';

    const formData = new FormData();
    if (uploadedFile) {
        formData.append('file', uploadedFile);
    }

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        let data;
        const responseText = await response.text();
        try {
            data = JSON.parse(responseText);
        } catch (jsonErr) {
            throw new Error(`Server error (${response.status}): ${responseText.substring(0, 120)}...`);
        }

        if (response.ok && data.success) {
            uploadedFilePath = data.filepath;
            analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyzing...';
            resultsCard.style.display = 'none';

            const mediaName = uploadedFile ? uploadedFile.name : urlInput.value;
            addUserMessage(`Analyzing media content: ${mediaName}`);

            socket.emit('analyze_image', {
                filepath: uploadedFilePath,
                query: ''
            });
        } else {
            throw new Error(data.error || `Upload failed with status ${response.status}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        addSystemMessage(`Upload failed: ${error.message}`, 'error');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Media';
    }
});

// ===== Chat =====
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !chatInput.disabled) {
        sendMessage();
    }
});

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addUserMessage(message);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    socket.emit('ask_question', { question: message });
}

// ===== Message Display =====
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addAgentMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    chatInput.disabled = false;
    sendBtn.disabled = false;
}

function addSystemMessage(text, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ===== Results Display =====
function displayResults(data) {
    resultsCard.style.display = 'block';

    const predictionLabel = document.getElementById('predictionLabel');
    predictionLabel.textContent = data.label;
    predictionLabel.className = 'prediction-label ' + data.label.toLowerCase();

    const confidence = (data.confidence * 100).toFixed(1);
    document.getElementById('confidenceValue').textContent = confidence + '%';

    const meterFill = document.getElementById('meterFill');
    setTimeout(() => {
        meterFill.style.width = confidence + '%';
    }, 100);

    const detailedScores = document.getElementById('detailedScores');
    detailedScores.innerHTML = '';

    for (const [label, score] of Object.entries(data.all_scores)) {
        const scoreItem = document.createElement('div');
        scoreItem.className = 'score-item';
        scoreItem.innerHTML = `
            <span class="score-label">${label}</span>
            <span class="score-value">${(score * 100).toFixed(2)}%</span>
        `;
        detailedScores.appendChild(scoreItem);
    }

    // Render frame-by-frame analysis for videos if present
    const videoFramesSection = document.getElementById('videoFramesSection');
    const framesGrid = document.getElementById('framesGrid');
    
    const frames = data.frame_results || (data.forensic_certificate && data.forensic_certificate.multi_dimensional_evidence && data.forensic_certificate.multi_dimensional_evidence.frame_by_frame_analysis);

    if (frames && frames.length > 0) {
        framesGrid.innerHTML = '';
        frames.forEach((f, idx) => {
            const card = document.createElement('div');
            card.style.cssText = 'background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; min-width: 130px; text-align: center;';
            const statusColor = f.fake_probability > 0.5 ? '#b91c1c' : '#15803d';
            card.innerHTML = `
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">Frame ${idx + 1} (${f.timestamp_sec}s)</div>
                <div style="font-size: 0.85rem; font-weight: 800; color: ${statusColor}; margin: 4px 0;">${f.label}</div>
                <div style="font-size: 0.75rem; color: #334155;">Fake: ${(f.fake_probability * 100).toFixed(1)}%</div>
            `;
            framesGrid.appendChild(card);
        });
        videoFramesSection.style.display = 'block';
    } else {
        videoFramesSection.style.display = 'none';
    }

    // Render Auditable Forensic Certificate
    const forensicCertCard = document.getElementById('forensicCertCard');
    if (data.forensic_certificate) {
        const cert = data.forensic_certificate;
        const ev = cert.multi_dimensional_evidence || {};

        document.getElementById('certIdVal').textContent = cert.certificate_id || 'N/A';
        document.getElementById('certRiskVal').textContent = (cert.composite_risk_score * 100).toFixed(1) + '%';
        
        const spectral = ev.spectral_frequency_analysis ? (ev.spectral_frequency_analysis.high_frequency_anomaly_score * 100).toFixed(1) + '%' : 'N/A';
        const spatial = ev.spatial_inconsistency_analysis ? (ev.spatial_inconsistency_analysis.spatial_inconsistency_score * 100).toFixed(1) + '%' : 'N/A';

        document.getElementById('certSpectralVal').textContent = spectral;
        document.getElementById('certSpatialVal').textContent = spatial;

        const certBadge = document.getElementById('certBadge');
        certBadge.textContent = cert.auditable_verdict || 'AUDITABLE';
        certBadge.style.background = cert.composite_risk_score > 0.5 ? '#ef4444' : '#10b981';
        certBadge.style.color = '#fff';

        forensicCertCard.style.display = 'block';
    } else {
        forensicCertCard.style.display = 'none';
    }

    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Another Media';
}

function displayExplanation(text) {
    const explanationSection = document.getElementById('explanationSection');
    const explanationText = document.getElementById('explanationText');

    explanationSection.style.display = 'block';
    explanationText.textContent = text;

    addAgentMessage(text);
}

// ===== Agent Status =====
function showAgentStatus(message) {
    agentStatusText.textContent = message;
    agentStatus.style.display = 'block';
}

function hideAgentStatus() {
    agentStatus.style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

console.log('Deepfake Detection Web App initialized with video & media support');
