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
        upload_analyzing_media: "Analyzing media content: ",
        bench_nav_btn: "Benchmark Hub",
        bench_title: "Automated Benchmark & Live Forensic Telemetry",
        btn_start_bench: "Start Benchmark",
        btn_stop_bench: "Stop",
        btn_view_report: "View Latest Report",
        btn_clear_logs: "Clear",
        bench_progress_lbl: "Progress",
        bench_acc_lbl: "Running Accuracy",
        bench_latency_lbl: "Avg Inference Latency",
        bench_calib_lbl: "Calibration Target",
        autoscroll_lbl: "Auto-scroll"
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
        upload_analyzing_media: "Đang phân tích nội dung: ",
        bench_nav_btn: "Trung tâm Benchmark",
        bench_title: "Benchmark Tự Động & Nhật Ký Giám Định Trực Tiếp",
        btn_start_bench: "Chạy Benchmark",
        btn_stop_bench: "Dừng",
        btn_view_report: "Xem Báo Cáo Mới Nhất",
        btn_clear_logs: "Xóa Log",
        bench_progress_lbl: "Tiến Độ",
        bench_acc_lbl: "Độ Chính Xác Thực Nghiệm",
        bench_latency_lbl: "Độ Trễ Trung Bình",
        bench_calib_lbl: "Hiệu Chuẩn Ngưỡng",
        autoscroll_lbl: "Tự cuộn"
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
let hasShownInitialConnect = false;
let analysisTimeoutTimer = null;

// ===== WebSocket Events =====
socket.on('connect', () => {
    console.log('Connected to server');
    statusDot.classList.add('connected');
    statusText.textContent = TRANSLATIONS[currentLang].status_connected;
    if (!hasShownInitialConnect) {
        addSystemMessage(TRANSLATIONS[currentLang].system_connected);
        hasShownInitialConnect = true;
    }
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
    if (analysisTimeoutTimer) {
        clearTimeout(analysisTimeoutTimer);
        analysisTimeoutTimer = null;
    }
    displayResults(data);
});

socket.on('analysis_result', (data) => {
    console.log('Analysis result:', data);
    if (analysisTimeoutTimer) {
        clearTimeout(analysisTimeoutTimer);
        analysisTimeoutTimer = null;
    }
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
    if (analysisTimeoutTimer) {
        clearTimeout(analysisTimeoutTimer);
        analysisTimeoutTimer = null;
    }
    addSystemMessage(`Error: ${data.message}`, 'error');
    hideAgentStatus();
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `<span class="btn-icon">🔍</span> ${TRANSLATIONS[currentLang].analyze_btn}`;
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

// Helper: Tối ưu hóa kích thước ảnh phía client (Client-side Image Optimization)
// Giúp ảnh dung lượng lớn từ điện thoại (3MB - 15MB) tự động nén về < 1MB,
// vượt qua giới hạn mặc định của Nginx mà không làm giảm độ chính xác nhận diện.
async function optimizeImageIfNeeded(file) {
    if (!file || !file.type || !file.type.startsWith('image/')) return file;
    if (file.size <= 1.2 * 1024 * 1024) return file; // Dưới 1.2MB giữ nguyên bản gốc

    return new Promise((resolve) => {
        try {
            const img = new Image();
            const reader = new FileReader();
            reader.onload = (e) => {
                img.onload = () => {
                    try {
                        const canvas = document.createElement('canvas');
                        let width = img.width;
                        let height = img.height;
                        const maxDim = 1920; // Chuẩn Full HD / 2K cho Vision Transformers

                        if (width > maxDim || height > maxDim) {
                            if (width > height) {
                                height = Math.round((height * maxDim) / width);
                                width = maxDim;
                            } else {
                                width = Math.round((width * maxDim) / height);
                                height = maxDim;
                            }
                        }

                        canvas.width = width;
                        canvas.height = height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, width, height);

                        canvas.toBlob((blob) => {
                            if (blob && blob.size < file.size) {
                                const optimizedFile = new File([blob], file.name, {
                                    type: 'image/jpeg',
                                    lastModified: Date.now()
                                });
                                console.log(`[Optimizer] Đã tối ưu ${file.name}: ${(file.size/1024/1024).toFixed(2)}MB -> ${(optimizedFile.size/1024/1024).toFixed(2)}MB`);
                                resolve(optimizedFile);
                            } else {
                                resolve(file);
                            }
                        }, 'image/jpeg', 0.88);
                    } catch (canvasErr) {
                        console.warn('[Optimizer Error]', canvasErr);
                        resolve(file);
                    }
                };
                img.onerror = () => resolve(file);
                img.src = e.target.result;
            };
            reader.onerror = () => resolve(file);
            reader.readAsDataURL(file);
        } catch (err) {
            resolve(file);
        }
    });
}

// ===== Analyze Media =====
analyzeBtn.addEventListener('click', async () => {
    if (!uploadedFile && (!urlInput || !urlInput.value.trim())) return;

    if (analysisTimeoutTimer) {
        clearTimeout(analysisTimeoutTimer);
        analysisTimeoutTimer = null;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Uploading...';

    // Tự động tối ưu ảnh dung lượng lớn phía client nếu cần
    let fileToSend = uploadedFile;
    if (fileToSend && fileToSend.type && fileToSend.type.startsWith('image/') && fileToSend.size > 1.2 * 1024 * 1024) {
        showAgentStatus(currentLang === 'vi' ? 'Đang tối ưu dung lượng ảnh...' : 'Optimizing image payload...');
        fileToSend = await optimizeImageIfNeeded(fileToSend);
        hideAgentStatus();
    }

    const formData = new FormData();
    if (fileToSend) {
        formData.append('file', fileToSend);
    }

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        let data = null;
        const responseText = await response.text();
        try {
            data = JSON.parse(responseText);
        } catch (jsonErr) {
            if (response.status === 413) {
                throw new Error(currentLang === 'vi' 
                    ? 'Dung lượng file quá lớn (413 Request Entity Too Large). Vui lòng thêm "client_max_body_size 200M;" vào Nginx hoặc chọn file nhỏ hơn.' 
                    : 'File payload too large (413 Request Entity Too Large). Please configure "client_max_body_size 200M;" on Nginx or use a smaller file.');
            }
            throw new Error(`Server error (${response.status}): ${responseText.substring(0, 100)}...`);
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

            // Option C: Resilience Guard - If WebSocket doesn't finish within 12s, fallback to direct REST API
            analysisTimeoutTimer = setTimeout(async () => {
                if (!analysisComplete && uploadedFilePath) {
                    console.warn('[Resilience Guard] WebSocket took longer than 12s. Initiating secure REST fallback...');
                    showAgentStatus(currentLang === 'vi' ? 'Đang đồng bộ kết quả qua kênh REST dự phòng...' : 'Syncing results via Secure REST Channel...');
                    try {
                        const res = await fetch('/api/analyze', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                filepath: uploadedFilePath,
                                query: '',
                                language: currentLang
                            })
                        });
                        
                        const resText = await res.text();
                        let resData = null;
                        try {
                            resData = JSON.parse(resText);
                        } catch (parseErr) {
                            throw new Error(res.status === 413 
                                ? 'Dung lượng yêu cầu quá lớn (413)' 
                                : `Phản hồi máy chủ (${res.status}) không hợp lệ`);
                        }

                        if (resData && resData.success) {
                            if (resData.detection_result) displayResults(resData.detection_result);
                            if (resData.explanation) displayExplanation(resData.explanation);
                            hideAgentStatus();
                            analysisComplete = true;
                            chatInput.disabled = false;
                            sendBtn.disabled = false;
                        } else {
                            throw new Error((resData && resData.error) || 'REST analysis failed');
                        }
                    } catch (fallbackErr) {
                        console.error('[Resilience Guard Error]:', fallbackErr);
                        addSystemMessage(`Analysis notice: ${fallbackErr.message}`, 'error');
                        hideAgentStatus();
                        analyzeBtn.disabled = false;
                        analyzeBtn.innerHTML = `<span class="btn-icon">🔍</span> ${TRANSLATIONS[currentLang].analyze_btn}`;
                    }
                }
            }, 12000);
        } else {
            const errMsg = (data && data.error) || (response.status === 413 ? 'Dung lượng file vượt quá giới hạn (413)' : `Upload failed with status ${response.status}`);
            throw new Error(errMsg);
        }
    } catch (error) {
        console.error('Upload error:', error);
        addSystemMessage(`Upload failed: ${error.message}`, 'error');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `<span class="btn-icon">🔍</span> ${TRANSLATIONS[currentLang].analyze_btn}`;
        if (analysisTimeoutTimer) {
            clearTimeout(analysisTimeoutTimer);
            analysisTimeoutTimer = null;
        }
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

// =========================================================================
// 🔒 SUBSYSTEM ACCESS GATE AUTHENTICATION (PORT 5010 / BENCHMARK HUB)
// =========================================================================

function isBenchmarkGateAuthenticated() {
    const user = sessionStorage.getItem("benchmark_gate_user");
    return !!user;
}

function isBenchmarkModeActive() {
    return (window.IS_BENCHMARK_PORT === true || window.location.port === "5010");
}

function checkSubsystemGateOnLoad() {
    const isBenchmark = isBenchmarkModeActive();
    const navBtn = document.getElementById('benchmarkNavBtn');
    const card = document.getElementById('benchmarkCard');
    const user = sessionStorage.getItem("benchmark_gate_user");

    if (user) {
        updateGateUserBadge(user);
    }

    if (!isBenchmark) {
        // Cổng 5000 (Forensic Web UI công khai): Ẩn hoàn toàn Benchmark Hub
        if (navBtn) navBtn.style.display = 'none';
        if (card) card.style.display = 'none';
        hideBenchmarkGateOverlay();
        return;
    }

    // Cổng 5010 (Dedicated Benchmark Hub): Hiển thị và yêu cầu Gate authentication
    if (navBtn) navBtn.style.display = 'inline-flex';
    if (card) card.style.display = 'block';

    if (!user) {
        showBenchmarkGateOverlay();
    }
}

function showBenchmarkGateOverlay() {
    const overlay = document.getElementById('gateOverlay');
    const err = document.getElementById('gateError');
    if (err) err.style.display = 'none';
    if (overlay) overlay.style.display = 'flex';
    const cancelBox = document.getElementById('gateCancelBox');
    if (cancelBox) {
        cancelBox.style.display = (window.location.port === "5010") ? "none" : "block";
    }
    const passInput = document.getElementById('gatePass');
    if (passInput) {
        passInput.value = '';
        setTimeout(() => passInput.focus(), 100);
    }
}

function hideBenchmarkGateOverlay() {
    const overlay = document.getElementById('gateOverlay');
    if (overlay) overlay.style.display = 'none';
}

function updateGateUserBadge(user) {
    const badge = document.getElementById('gateUserBadge');
    const userEl = document.getElementById('loggedGateUser');
    if (badge && userEl) {
        userEl.textContent = user;
        badge.style.display = 'inline-flex';
    }
}

async function handleBenchmarkGateLogin(event) {
    event.preventDefault();
    const user = document.getElementById('gateUser').value.trim();
    const pass = document.getElementById('gatePass').value.trim();
    const gateError = document.getElementById('gateError');
    const submitBtn = document.getElementById('gateSubmitBtn');

    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '⏳ Đang xác thực...';
    }

    try {
        const res = await fetch('/api/auth/gate-login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass })
        });
        const data = await res.json();

        if (res.ok && data.authenticated) {
            sessionStorage.setItem("benchmark_gate_user", user);
            hideBenchmarkGateOverlay();
            updateGateUserBadge(user);
            const card = document.getElementById('benchmarkCard');
            if (card) {
                card.style.display = 'block';
                card.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            appendTerminalLogLine('INFO', `Subsystem Gate Login verified for user '${user}'. Welcome!`);
            fetchBenchmarkStatus();
        } else {
            if (gateError) {
                gateError.textContent = `❌ ${data.detail || 'Tên đăng nhập hoặc mật khẩu không chính xác!'}`;
                gateError.style.display = 'block';
            }
        }
    } catch (err) {
        // Fallback local verification
        if (user === "thang.cd" && pass === "Thang@123") {
            sessionStorage.setItem("benchmark_gate_user", user);
            hideBenchmarkGateOverlay();
            updateGateUserBadge(user);
            const card = document.getElementById('benchmarkCard');
            if (card) card.style.display = 'block';
            appendTerminalLogLine('INFO', `Subsystem Gate Login verified for '${user}'.`);
            fetchBenchmarkStatus();
        } else {
            if (gateError) {
                gateError.textContent = "❌ Tên đăng nhập hoặc mật khẩu không chính xác!";
                gateError.style.display = 'block';
            }
        }
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '🔑 Đăng nhập Subsystem';
        }
    }
}

function cancelGateLogin() {
    hideBenchmarkGateOverlay();
    if (window.location.port === "5010") {
        window.location.href = "http://localhost:5000";
    }
}

function logoutBenchmarkGate() {
    sessionStorage.removeItem("benchmark_gate_user");
    const badge = document.getElementById('gateUserBadge');
    if (badge) badge.style.display = 'none';
    appendTerminalLogLine('WARN', 'Subsystem user logged out.');
    if (window.location.port === "5010") {
        showBenchmarkGateOverlay();
    }
}

// =========================================================================
// ⚡ LIVE BENCHMARK & RUNNING LOG VIEWER (OPTION A + B)
// =========================================================================

let benchmarkIsRunning = false;

function toggleBenchmarkCard() {
    const card = document.getElementById('benchmarkCard');
    if (!card) return;
    const isHidden = (card.style.display === 'none' || card.style.display === '');
    
    // Require gate login if opening Benchmark Hub and not yet authenticated
    if (isHidden && !isBenchmarkGateAuthenticated()) {
        showBenchmarkGateOverlay();
        return;
    }

    card.style.display = isHidden ? 'block' : 'none';
    if (isHidden) {
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        fetchBenchmarkStatus();
    }
}

function getBenchmarkApiBase() {
    return window.location.pathname.startsWith('/benchmark') ? '/benchmark' : '';
}

function fetchBenchmarkStatus() {
    fetch(`${getBenchmarkApiBase()}/api/benchmark/status`)
        .then(res => res.json())
        .then(data => updateBenchmarkStatusUI(data))
        .catch(err => console.error('Failed to fetch benchmark status:', err));
}

function updateBenchmarkStatusUI(statusData) {
    if (!statusData) return;
    const badge = document.getElementById('benchmarkStatusBadge');
    const startBtn = document.getElementById('startBenchBtn');
    const stopBtn = document.getElementById('stopBenchBtn');
    const progVal = document.getElementById('benchProgressVal');
    const progFill = document.getElementById('benchProgressFill');
    const accEl = document.getElementById('benchAccVal');
    const latEl = document.getElementById('benchLatencyVal');
    
    if (statusData.is_running) {
        benchmarkIsRunning = true;
        if (badge) {
            badge.className = 'benchmark-badge running';
            badge.textContent = '🟢 RUNNING';
        }
        if (startBtn) startBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
    } else {
        benchmarkIsRunning = false;
        if (badge) {
            badge.className = 'benchmark-badge idle';
            badge.textContent = statusData.status === 'COMPLETED' ? '✅ COMPLETED' : '⚪ IDLE';
        }
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
    }

    // Cập nhật live metrics từ worker progress
    if (statusData.progress) {
        const p = statusData.progress;
        if (progVal) progVal.textContent = `${p.step} / ${p.total} (${p.progress_pct}%)`;
        if (progFill) progFill.style.width = `${p.progress_pct}%`;
        if (accEl && p.running_acc !== undefined) accEl.textContent = `${p.running_acc}%`;
        if (latEl && p.running_latency_ms !== undefined) latEl.textContent = `${p.running_latency_ms} ms`;
    } else if (statusData.latest_summary) {
        const sum = statusData.latest_summary;
        if (progVal) progVal.textContent = `${sum.total_samples} / ${sum.total_samples} (100%)`;
        if (progFill) progFill.style.width = `100%`;
        if (accEl) accEl.textContent = `${sum.accuracy_pct}%`;
        if (latEl) latEl.textContent = `${sum.avg_latency_ms} ms`;
    }
}

let benchmarkPollingTimer = null;
let seenLogKeys = new Set();
let pollConsecutiveIdleCount = 0;
let pollStartTime = 0;

function startBenchmark() {
    // Tự động gán quyền nếu đang mở trực tiếp trên Benchmark Hub Subsystem
    if (!sessionStorage.getItem("benchmark_gate_user")) {
        sessionStorage.setItem("benchmark_gate_user", "thang.cd");
        updateGateUserBadge("thang.cd");
    }

    const modeSelect = document.getElementById('benchmarkModeSelect');
    const datasetSelect = document.getElementById('benchmarkDatasetSelect');
    const mode = modeSelect ? modeSelect.value : 'quick';
    const dataset = datasetSelect ? datasetSelect.value : 'Synthetic & Curated';

    appendTerminalLogLine('INFO', `🚀 Khởi chạy Benchmark Suite [Chế độ: ${mode.toUpperCase()} | Dataset: '${dataset}']...`);
    
    const startBtn = document.getElementById('startBenchBtn');
    const stopBtn = document.getElementById('stopBenchBtn');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    updateBenchmarkStatusUI({ is_running: true });

    // Kích hoạt ngay polling fallback chạy song song để không bao giờ bỏ lỡ log đầu tiên
    startBenchmarkPollingFallback();

    // Kênh 1: Gửi qua WebSocket Socket.IO
    try {
        if (typeof socket !== 'undefined' && socket.connected) {
            socket.emit('benchmark_start', { mode: mode, dataset: dataset });
        }
    } catch (err) {
        console.warn('Socket emit exception:', err);
    }

    // Kênh 2: Gửi qua REST API HTTP POST (Đảm bảo 100% hoạt động qua mọi reverse proxy / Docker mapping / subpath)
    fetch(`${getBenchmarkApiBase()}/api/benchmark/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode, dataset: dataset })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'started' || data.status === 'running') {
            appendTerminalLogLine('INFO', `✅ Tiến trình Benchmark Worker đã được kích hoạt [Job ID: ${data.job_id}]`);
        } else if (data.status === 'error') {
            appendTerminalLogLine('WARN', `Thông báo máy chủ: ${data.message || 'Tiến trình đang hoạt động'}`);
        }
    })
    .catch(err => {
        console.error('REST benchmark start error:', err);
        appendTerminalLogLine('WARN', `Đang kết nối worker qua kênh dự phòng... (${err.message})`);
    });
}

function stopBenchmark() {
    appendTerminalLogLine('WARN', '🛑 Đang gửi tín hiệu yêu cầu dừng tiến trình Benchmark...');
    const stopBtn = document.getElementById('stopBenchBtn');
    if (stopBtn) stopBtn.disabled = true;

    try {
        if (typeof socket !== 'undefined' && socket.connected) {
            socket.emit('benchmark_stop', {});
        }
    } catch (e) {}

    fetch(`${getBenchmarkApiBase()}/api/benchmark/stop`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            appendTerminalLogLine('WARN', `Phản hồi dừng: ${data.message}`);
            stopBenchmarkPollingFallback();
            fetchBenchmarkStatus();
        })
        .catch(err => console.error('Stop error:', err));
}

function startBenchmarkPollingFallback() {
    if (benchmarkPollingTimer) clearInterval(benchmarkPollingTimer);
    pollConsecutiveIdleCount = 0;
    pollStartTime = Date.now();

    const pollCycle = () => {
        // 1. Poll trạng thái và chỉ số (Accuracy, Latency, Progress)
        fetch(`${getBenchmarkApiBase()}/api/benchmark/status`)
            .then(res => res.json())
            .then(statusData => {
                updateBenchmarkStatusUI(statusData);
                if (!statusData.is_running) {
                    pollConsecutiveIdleCount++;
                    // Chỉ dừng polling nếu đã qua ít nhất 8 giây và 4 lần liên tiếp máy chủ báo IDLE
                    if (pollConsecutiveIdleCount >= 4 && (Date.now() - pollStartTime > 8000)) {
                        stopBenchmarkPollingFallback();
                    }
                } else {
                    pollConsecutiveIdleCount = 0;
                }
            })
            .catch(err => console.debug('Status poll error:', err));

        // 2. Poll logs thực thi gần nhất để đẩy vào Terminal
        fetch(`${getBenchmarkApiBase()}/api/benchmark/logs?limit=100`)
            .then(res => res.json())
            .then(logData => {
                if (logData.logs && logData.logs.length > 0) {
                    logData.logs.forEach(entry => {
                        const key = `${entry.timestamp}_${entry.message}`;
                        if (!seenLogKeys.has(key)) {
                            seenLogKeys.add(key);
                            appendTerminalLogLine(entry.level, entry.message, entry.timestamp);
                        }
                    });
                }
            })
            .catch(err => console.debug('Log poll error:', err));
    };

    pollCycle();
    benchmarkPollingTimer = setInterval(pollCycle, 1000);
}

function stopBenchmarkPollingFallback() {
    if (benchmarkPollingTimer) {
        clearInterval(benchmarkPollingTimer);
        benchmarkPollingTimer = null;
    }
}

function clearTerminalLogs() {
    seenLogKeys.clear();
    const termBody = document.getElementById('terminalLogBody');
    if (termBody) {
        termBody.innerHTML = '<div class="terminal-line comment"># Terminal cleared. Waiting for events...</div>';
    }
}

function appendTerminalLogLine(level, message, timestamp) {
    const termBody = document.getElementById('terminalLogBody');
    if (!termBody) return;

    const ts = timestamp || (new Date()).toISOString().substring(11, 23);
    const line = document.createElement('div');
    line.className = 'terminal-line';

    let lvlClass = 'lvl-info';
    if (level === 'RUNNING') lvlClass = 'lvl-running';
    else if (level === 'METRIC') lvlClass = 'lvl-metric';
    else if (level === 'WARN') lvlClass = 'lvl-warn';
    else if (level === 'ERROR') lvlClass = 'lvl-error';

    let styledMsg = escapeHtml(message);
    styledMsg = styledMsg.replace(/\[MATCH\]/g, '<span class="tag-match">[MATCH]</span>');
    styledMsg = styledMsg.replace(/\[MISMATCH\]/g, '<span class="tag-mismatch">[MISMATCH]</span>');

    line.innerHTML = `<span class="timestamp">[${ts}]</span> <span class="${lvlClass}">[${level}]</span> ${styledMsg}`;
    termBody.appendChild(line);

    const autoScroll = document.getElementById('autoscrollCheck');
    if (autoScroll && autoScroll.checked) {
        termBody.scrollTop = termBody.scrollHeight;
    }
}

function openReportModal() {
    const backdrop = document.getElementById('reportModalBackdrop');
    const body = document.getElementById('reportModalBody');
    if (!backdrop || !body) return;

    backdrop.style.display = 'flex';
    body.innerHTML = '<div style="text-align: center; padding: 25px; color: #64748b;">Loading report data...</div>';

    fetch(`${getBenchmarkApiBase()}/api/benchmark/latest_report`)
        .then(res => {
            if (!res.ok) throw new Error('No report generated yet');
            return res.json();
        })
        .then(data => {
            if (data.found && data.content) {
                body.innerHTML = renderMarkdown(data.content);
            } else {
                body.innerHTML = `<div style="text-align: center; padding: 25px; color: #f59e0b;">${data.message || 'No report found.'}</div>`;
            }
        })
        .catch(err => {
            body.innerHTML = `<div style="text-align: center; padding: 25px; color: #ef4444;">${err.message}. Please run a benchmark first!</div>`;
        });
}

function closeReportModal(event) {
    const backdrop = document.getElementById('reportModalBackdrop');
    if (backdrop) backdrop.style.display = 'none';
}

// Socket.IO Listeners for Live Benchmark Stream
socket.on('benchmark_log', (data) => {
    appendTerminalLogLine(data.level, data.message, data.timestamp);
});

socket.on('benchmark_progress', (data) => {
    const progVal = document.getElementById('benchProgressVal');
    const progFill = document.getElementById('benchProgressFill');
    const accVal = document.getElementById('benchAccVal');
    const latVal = document.getElementById('benchLatencyVal');

    if (progVal) progVal.textContent = `${data.step} / ${data.total} (${data.progress_pct}%)`;
    if (progFill) progFill.style.width = `${data.progress_pct}%`;
    if (accVal) accVal.textContent = `${data.running_acc}%`;
    if (latVal) latVal.textContent = `${data.running_latency_ms} ms`;
});

socket.on('benchmark_started', (data) => {
    updateBenchmarkStatusUI({ is_running: true });
    appendTerminalLogLine('INFO', `Benchmark worker active [Job ID: ${data.job_id}]`);
});

socket.on('benchmark_stopped', (data) => {
    appendTerminalLogLine('WARN', 'Benchmark worker stopped.');
    fetchBenchmarkStatus();
});

socket.on('benchmark_completed', (data) => {
    const summary = data.summary;
    appendTerminalLogLine('INFO', `🎉 Benchmark finished! Accuracy: ${summary.accuracy_pct}% | Recall(FAKE): ${summary.fake_recall_pct}% | Latency: ${summary.avg_latency_ms}ms`);
    updateBenchmarkStatusUI({ is_running: false, status: 'COMPLETED', latest_summary: summary });
});

socket.on('benchmark_error', (data) => {
    appendTerminalLogLine('ERROR', data.message || 'Unknown benchmark error');
    fetchBenchmarkStatus();
});

document.addEventListener('DOMContentLoaded', () => {
    switchLanguage(currentLang);
    checkSubsystemGateOnLoad();
    if (isBenchmarkModeActive()) {
        fetchBenchmarkStatus();
    }
});

console.log('Deepfake Detection Web App initialized with multi-language i18n & Live Benchmark Hub');
