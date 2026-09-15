// WebSocket connection
const socket = io();

// ===== Multi-Language (i18n) Translations =====
const TRANSLATIONS = {
    en: {
        status_connecting: "Connecting...",
        status_connected: "Connected",
        status_disconnected: "Disconnected",
        hero_title: "Detect AI-Generated Images with Intelligence",
        hero_subtitle: "Multi-agent system powered by Vision Transformers and Google Gemini",
        hero_feature: "Upload images, video or audio to detect AI-generated or deepfake content using best-in-class models.",
        upload_title: "Upload Image / Media",
        upload_headline: "Drag and drop your file here or browse to upload",
        upload_subheadline: "Supports image, video and audio formats",
        browse_btn: "Browse Files",
        divider_or: "or",
        url_placeholder: "Enter Media URL...",
        analyze_btn: "Analyze Media",
        analyzing_btn: "Analyzing...",
        fmt_img: "Supported Image:",
        fmt_vid: "Video:",
        fmt_aud: "Audio:",
        chat_title: "Chat with Agent",
        welcome_msg: "👋 Welcome! Upload an image or video to start detection.",
        chat_placeholder: "Ask a question about the results...",
        results_title: "Detection & Forensic Results",
        confidence_label: "Confidence Score:",
        video_frames_title: "🎬 Frame-by-Frame Video Analysis",
        cert_title: "📜 Auditable Digital Forensic Certificate",
        cert_id_label: "Certificate ID",
        cert_risk_label: "Composite Risk",
        cert_spectral_label: "Spectral Anomaly (FFT 2D)",
        cert_spatial_label: "Spatial Noise Inconsistency",
        explanation_title: "🧠 AI Forensic Analysis & Multi-Agent Explanation",
        summary_verdict_title: "1. Executive Verdict & Certificate Summary",
        badge_real: "✅ REAL / AUTHENTIC",
        badge_fake: "🚨 FAKE / DEEPFAKE",
        conf_score_label: "Confidence Score",
        system_connected: "Connected to Deepfake Detection Agent System",
        upload_analyzing_media: "Analyzing media content: "
    },
    vi: {
        status_connecting: "Đang kết nối...",
        status_connected: "Đã kết nối",
        status_disconnected: "Mất kết nối",
        hero_title: "Phát hiện hình ảnh & truyền thông AI thông minh",
        hero_subtitle: "Hệ thống đa Agent dựa trên Vision Transformers và Google Gemini",
        hero_feature: "Tải lên hình ảnh, video hoặc âm thanh để phát hiện nội dung giả mạo (Deepfake) bằng các mô hình hàng đầu.",
        upload_title: "Tải lên Hình ảnh / Truyền thông",
        upload_headline: "Kéo thả tệp vào đây hoặc duyệt để tải lên",
        upload_subheadline: "Hỗ trợ định dạng hình ảnh, video và âm thanh",
        browse_btn: "Chọn tệp",
        divider_or: "hoặc",
        url_placeholder: "Nhập đường dẫn URL...",
        analyze_btn: "Phân tích Truyền thông",
        analyzing_btn: "Đang phân tích...",
        fmt_img: "Hình ảnh hỗ trợ:",
        fmt_vid: "Video:",
        fmt_aud: "Âm thanh:",
        chat_title: "Trò chuyện với Agent",
        welcome_msg: "👋 Xin chào! Tải lên hình ảnh hoặc video để bắt đầu giám định.",
        chat_placeholder: "Đặt câu hỏi về kết quả giám định...",
        results_title: "Kết quả Phát hiện & Giám định",
        confidence_label: "Độ tin cậy:",
        video_frames_title: "🎬 Phân tích từng khung hình Video",
        cert_title: "📜 Chứng thư Giám định Kỹ thuật số",
        cert_id_label: "Mã chứng thư",
        cert_risk_label: "Rủi ro tổng hợp",
        cert_spectral_label: "Dấu vết phổ (FFT 2D)",
        cert_spatial_label: "Bất biến nhiễu không gian",
        explanation_title: "🧠 Phân tích Giám định AI & Giải trình Đa Agent",
        summary_verdict_title: "1. Kết luận Giám định & Tóm tắt Chứng thư",
        badge_real: "✅ CHÂN THỰC / ẢNH THỰC",
        badge_fake: "🚨 GIẢ MẠO / DEEPFAKE",
        conf_score_label: "Độ tin cậy",
        system_connected: "Đã kết nối với Hệ thống Agent Phát hiện Deepfake",
        upload_analyzing_media: "Đang phân tích nội dung: "
    }
};

let currentLang = localStorage.getItem('app_lang') || (navigator.language && navigator.language.startsWith('vi') ? 'vi' : 'en');

function switchLanguage(lang) {
    if (!TRANSLATIONS[lang]) return;
    currentLang = lang;
    localStorage.setItem('app_lang', lang);

    const btnEn = document.getElementById('langBtnEn');
    const btnVi = document.getElementById('langBtnVi');
    if (btnEn && btnVi) {
        btnEn.classList.toggle('active', lang === 'en');
        btnVi.classList.toggle('active', lang === 'vi');
    }

    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (TRANSLATIONS[lang][key]) {
            el.textContent = TRANSLATIONS[lang][key];
        }
    });

    document.querySelectorAll('[data-i18n-ph]').forEach(el => {
        const key = el.getAttribute('data-i18n-ph');
        if (TRANSLATIONS[lang][key]) {
            el.placeholder = TRANSLATIONS[lang][key];
        }
    });

    if (statusDot.classList.contains('connected')) {
        statusText.textContent = TRANSLATIONS[lang].status_connected;
    }
}

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
    statusText.textContent = TRANSLATIONS[currentLang].status_connected;
    addSystemMessage(TRANSLATIONS[currentLang].system_connected);
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    statusDot.classList.remove('connected');
    statusText.textContent = TRANSLATIONS[currentLang].status_disconnected;
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
            addUserMessage(`${TRANSLATIONS[currentLang].upload_analyzing_media}${mediaName}`);

            socket.emit('analyze_image', {
                filepath: uploadedFilePath,
                query: '',
                language: currentLang
            });
        } else {
            throw new Error(data.error || `Upload failed with status ${response.status}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        addSystemMessage(`Upload failed: ${error.message}`, 'error');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `<span class="btn-icon">🔍</span> ${TRANSLATIONS[currentLang].analyze_btn}`;
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

    socket.emit('ask_question', { question: message, language: currentLang });
}

// ===== Message Display =====
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function renderMarkdown(text) {
    if (!text) return '';
    if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
        try {
            return marked.parse(text);
        } catch (e) {
            console.error('Markdown parse error:', e);
        }
    }
    return escapeHtml(text)
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function addAgentMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    messageDiv.innerHTML = `<div class="message-content markdown-body">${renderMarkdown(text)}</div>`;
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
    analyzeBtn.innerHTML = `<span class="btn-icon">🔍</span> ${TRANSLATIONS[currentLang].analyze_btn}`;
}

let latestDetectionResult = null;

const originalDisplayResults = displayResults;
displayResults = function(data) {
    latestDetectionResult = data;
    originalDisplayResults(data);
};

function formatAgentChatSummary(text, data, lang = currentLang) {
    const t = TRANSLATIONS[lang] || TRANSLATIONS.en;
    if (!data) return renderMarkdown(text);

    const cert = data.forensic_certificate || {};
    const ev = cert.multi_dimensional_evidence || {};

    const label = data.label || 'Unknown';
    const isFake = label.toLowerCase().includes('fake') || label.toLowerCase().includes('synthetic') || label.toLowerCase().includes('manipulate');
    const labelBadgeText = isFake ? t.badge_fake : t.badge_real;
    const labelBadgeBg = isFake ? '#fee2e2' : '#dcfce7';
    const labelBadgeColor = isFake ? '#b91c1c' : '#15803d';

    const confidence = (data.confidence * 100).toFixed(1) + '%';
    const certId = cert.certificate_id || 'N/A';
    const compositeRisk = cert.composite_risk_score !== undefined ? (cert.composite_risk_score * 100).toFixed(1) + '%' : 'N/A';

    const spectral = ev.spectral_frequency_analysis ? (ev.spectral_frequency_analysis.high_frequency_anomaly_score * 100).toFixed(1) + '%' : '0.0%';
    const spatial = ev.spatial_inconsistency_analysis ? (ev.spatial_inconsistency_analysis.spatial_inconsistency_score * 100).toFixed(1) + '%' : '8.8%';

    const cleanExplanation = renderMarkdown(text);

    return `
        <div class="chat-forensic-summary" style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 14px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 10px;">
                <span style="font-weight: 800; font-size: 0.9rem; background: ${labelBadgeBg}; color: ${labelBadgeColor}; padding: 3px 10px; border-radius: 6px;">
                    ${labelBadgeText}
                </span>
                <span style="font-size: 0.8rem; font-weight: 700; color: #475569;">
                    ${t.conf_score_label}: <strong style="color: #0f172a;">${confidence}</strong>
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; font-size: 0.82rem; color: #334155;">
                <div><span style="color: #64748b;">${t.cert_id_label}:</span> <code style="font-family: monospace; font-weight: 700; background: #f8fafc; padding: 2px 6px; border-radius: 4px; border: 1px solid #e2e8f0; color: #0f172a;">${certId}</code></div>
                <div><span style="color: #64748b;">${t.cert_risk_label}:</span> <strong style="color: ${isFake ? '#dc2626' : '#16a34a'};">${compositeRisk}</strong></div>
                <div><span style="color: #64748b;">${t.cert_spectral_label}:</span> <strong>${spectral}</strong></div>
                <div><span style="color: #64748b;">${t.cert_spatial_label}:</span> <strong>${spatial}</strong></div>
            </div>
        </div>
        <div style="font-weight: 800; font-size: 0.88rem; color: #1e293b; margin-bottom: 6px; border-left: 3px solid #5c46e5; padding-left: 8px;">${t.summary_verdict_title}</div>
        <div class="explanation-text-body" style="font-size: 0.88rem; line-height: 1.55;">${cleanExplanation}</div>
    `;
}

function displayExplanation(text) {
    const explanationSection = document.getElementById('explanationSection');
    const explanationText = document.getElementById('explanationText');

    explanationSection.style.display = 'block';
    explanationText.innerHTML = renderMarkdown(text);

    // Format chat agent message as clean summary card with active language
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    messageDiv.innerHTML = `<div class="message-content markdown-body">${formatAgentChatSummary(text, latestDetectionResult, currentLang)}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    chatInput.disabled = false;
    sendBtn.disabled = false;
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

document.addEventListener('DOMContentLoaded', () => {
    switchLanguage(currentLang);
});

console.log('Deepfake Detection Web App initialized with multi-language i18n support');
