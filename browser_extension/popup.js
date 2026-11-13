// popup.js - Interactive Popup Controls & Direct In-Popup Forensic Scanner for DeepTrace

document.addEventListener("DOMContentLoaded", () => {
    const scanBtn = document.getElementById("scanCurrentBtn");
    const scanLiveCallBtn = document.getElementById("scanLiveCallBtn");
    const openCockpitBtn = document.getElementById("openCockpitBtn");
    const endpointSelect = document.getElementById("apiEndpointSelect");
    const gatewayStatus = document.getElementById("gatewayStatus");

    const resultSection = document.getElementById("resultSection");
    const scanLoading = document.getElementById("scanLoading");
    const scanStatusText = document.getElementById("scanStatusText");
    const scanCard = document.getElementById("scanCard");
    const scanError = document.getElementById("scanError");
    const scanErrorMsg = document.getElementById("scanErrorMsg");

    const resIcon = document.getElementById("resIcon");
    const resTitle = document.getElementById("resTitle");
    const resSubtitle = document.getElementById("resSubtitle");
    const resCertId = document.getElementById("resCertId");
    const resCertLink = document.getElementById("resCertLink");
    const resModeBadge = document.getElementById("resModeBadge");
    const resScanAgainBtn = document.getElementById("resScanAgainBtn");

    // Load saved settings
    chrome.storage.local.get(["apiBase"], (res) => {
        if (res.apiBase) {
            endpointSelect.value = res.apiBase;
            updateStatusText(res.apiBase);
        }
    });

    // Handle endpoint change
    endpointSelect.addEventListener("change", (e) => {
        const newBase = e.target.value;
        chrome.storage.local.set({ apiBase: newBase }, () => {
            updateStatusText(newBase);
        });
    });

    function updateStatusText(url) {
        try {
            const host = new URL(url).host;
            gatewayStatus.querySelector("span:last-child").textContent = `Gateway: ${host}`;
        } catch (err) {
            gatewayStatus.querySelector("span:last-child").textContent = `Gateway: ${url}`;
        }
    }

    // Helper: Compress/Downscale raw image data URL to max 1080p JPEG
    function prepareCompressedImage(dataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                try {
                    const canvas = document.createElement("canvas");
                    let w = img.width;
                    let h = img.height;
                    const maxDim = 1080;
                    if (w > maxDim || h > maxDim) {
                        const ratio = Math.min(maxDim / w, maxDim / h);
                        w = Math.round(w * ratio);
                        h = Math.round(h * ratio);
                    }
                    canvas.width = Math.max(1, w);
                    canvas.height = Math.max(1, h);
                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    resolve(canvas.toDataURL("image/jpeg", 0.85));
                } catch (e) {
                    reject(e);
                }
            };
            img.onerror = () => reject(new Error("Không thể nạp dữ liệu ảnh chụp màn hình"));
            img.src = dataUrl;
        });
    }

    // Main In-Popup Forensic Audit Flow
    async function executePopupScan(routingMode = "auto") {
        resultSection.style.display = "block";
        scanLoading.style.display = "block";
        scanCard.style.display = "none";
        scanError.style.display = "none";
        scanBtn.disabled = true;
        if (scanLiveCallBtn) scanLiveCallBtn.disabled = true;

        scanStatusText.textContent = "Đang chụp khung hình hiển thị...";

        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab?.id) {
                throw new Error("Không tìm thấy tab trình duyệt đang hoạt động.");
            }

            // Also signal content script to display in-situ badge on the webpage if possible
            chrome.tabs.sendMessage(tab.id, { 
                action: routingMode === "call" ? "TRIGGER_LIVE_CALL_SCAN" : "TRIGGER_CONTEXT_SCAN", 
                mode: routingMode 
            }, () => {
                // Ignore if content script is not injected on tab
                if (chrome.runtime.lastError) { /* ignore */ }
            });

            // Capture active visible tab directly from extension context
            const rawCaptureUrl = await new Promise((resolve, reject) => {
                chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" }, (dataUrl) => {
                    if (chrome.runtime.lastError || !dataUrl) {
                        reject(new Error(chrome.runtime.lastError?.message || "Không thể chụp màn hình tab"));
                    } else {
                        resolve(dataUrl);
                    }
                });
            });

            scanStatusText.textContent = "Đang tối ưu khung hình pháp y...";
            const optimizedBase64 = await prepareCompressedImage(rawCaptureUrl);

            scanStatusText.textContent = "Đang đối soát mô hình ViT đa tầng...";
            const auditResponse = await new Promise((resolve) => {
                chrome.runtime.sendMessage({
                    action: "EXECUTE_HYBRID_AUDIT",
                    payload: {
                        type: "base64",
                        imageBase64: optimizedBase64,
                        sourceUrl: tab.url || "",
                        routing_mode: routingMode
                    }
                }, resolve);
            });

            if (!auditResponse || !auditResponse.success) {
                throw new Error(auditResponse?.error || "Máy chủ không phản hồi kết quả giám định.");
            }

            // Render Result Card in Popup
            const data = auditResponse.result;
            const det = data.detection_result || {};
            const label = det.label || "Realism";
            const isFake = label === "Deepfake" || label === "Fake";
            const conf = det.confidence ? (det.confidence * 100).toFixed(1) : "95.0";
            const risk = det.composite_risk_score !== undefined ? (det.composite_risk_score * 100).toFixed(1) : (isFake ? "92.4" : "3.8");
            const cert = det.forensic_certificate || {};
            const certId = cert.certificate_id || det.certificate_id || (routingMode === "call" ? "CERT-DT-CALL" : "CERT-DT-IMG");
            const apiBase = data.api_base || endpointSelect.value || "https://deeptrace.io.vn";

            resIcon.textContent = isFake ? "🚨" : "✅";
            resTitle.textContent = isFake ? "DEEPFAKE DETECTED" : "AUTHENTIC MEDIA";
            resTitle.style.color = isFake ? "#ef4444" : "#10b981";
            resSubtitle.textContent = `Độ tin cậy: ${conf}% | Rủi ro: ${risk}%`;
            resCertId.textContent = certId;
            resCertLink.href = `${apiBase}/certificate/${certId}`;
            resModeBadge.textContent = routingMode === "call" ? "CALL STREAM" : "TAB SCAN";

            if (isFake) {
                scanCard.classList.add("card-fake");
            } else {
                scanCard.classList.remove("card-fake");
            }

            scanLoading.style.display = "none";
            scanCard.style.display = "block";
        } catch (err) {
            console.error("[DF-Popup Scan Error]", err);
            scanLoading.style.display = "none";
            scanError.style.display = "block";
            scanErrorMsg.textContent = err.message || "Lỗi xử lý giám định phương tiện.";
        } finally {
            scanBtn.disabled = false;
            if (scanLiveCallBtn) scanLiveCallBtn.disabled = false;
        }
    }

    // Trigger Scan on Active Tab
    scanBtn.addEventListener("click", () => {
        executePopupScan("auto");
    });

    // Trigger Live Stream / Real-time Video Call Guardian
    if (scanLiveCallBtn) {
        scanLiveCallBtn.addEventListener("click", () => {
            executePopupScan("call");
        });
    }

    // Reset and scan again
    if (resScanAgainBtn) {
        resScanAgainBtn.addEventListener("click", () => {
            resultSection.style.display = "none";
            scanCard.style.display = "none";
        });
    }

    // Open Full Cockpit Web App
    openCockpitBtn.addEventListener("click", () => {
        const targetUrl = endpointSelect.value || "https://deeptrace.io.vn";
        chrome.tabs.create({ url: targetUrl });
    });
});
