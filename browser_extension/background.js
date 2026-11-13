// background.js - Service Worker for DeepTrace Forensic Shield (Manifest V3)

const PRIMARY_API_BASE = "https://deeptrace.io.vn";
const FALLBACK_API_BASE = "https://detection.dtpsoft.vn";

// Initialize Context Menus and Storage Defaults
chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.get(["apiBase"], (res) => {
        if (!res.apiBase) {
            chrome.storage.local.set({ apiBase: PRIMARY_API_BASE });
        }
    });

    chrome.contextMenus.create({
        id: "df-scan-context",
        title: "🛡️ Giám định Deepfake Phương Tiện Này...",
        contexts: ["video", "image", "link", "page"]
    });
});

// Context Menu Action
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "df-scan-context" && tab?.id) {
        chrome.tabs.sendMessage(tab.id, {
            action: "TRIGGER_CONTEXT_SCAN",
            mediaUrl: info.srcUrl || info.linkUrl || info.pageUrl
        });
    }
});

// Get currently configured backend endpoint
async function getApiBase() {
    return new Promise((resolve) => {
        chrome.storage.local.get(["apiBase"], (res) => {
            resolve(res.apiBase || PRIMARY_API_BASE);
        });
    });
}

// Robust fetch with configurable timeout via AbortController
async function fetchWithTimeout(url, options = {}, timeoutMs = 50000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(timeoutId);
        return response;
    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === "AbortError") {
            throw new Error(`Quá thời gian chờ phản hồi từ máy chủ (${Math.round(timeoutMs / 1000)}s timeout).`);
        }
        throw err;
    }
}

// Handle messages from content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "CAPTURE_ACTIVE_TAB") {
        try {
            chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
                if (chrome.runtime.lastError || !dataUrl) {
                    sendResponse({ 
                        success: false, 
                        error: chrome.runtime.lastError?.message || "Không thể chụp màn hình tab hiện tại" 
                    });
                } else {
                    sendResponse({ success: true, dataUrl });
                }
            });
        } catch (e) {
            sendResponse({ success: false, error: e.message });
        }
        return true; // Keep message channel open for async response
    }

    if (request.action === "EXECUTE_HYBRID_AUDIT") {
        // Ping service worker to prevent premature termination during long analysis
        const pingInterval = setInterval(() => {
            chrome.runtime.getPlatformInfo(() => {});
        }, 10000);

        executeForensicPipeline(request.payload)
            .then((result) => {
                clearInterval(pingInterval);
                sendResponse({ success: true, result });
            })
            .catch((err) => {
                clearInterval(pingInterval);
                console.error("[DF-Shield Audit Pipeline Error]", err);
                sendResponse({ success: false, error: err.message || "Lỗi quy trình giám định máy chủ" });
            });
        return true; // Keep message channel open for async response
    }
});

/**
 * Execute Client-Server Hybrid Forensic Pipeline with automatic failover
 */
async function executeForensicPipeline(payload) {
    const primaryBase = await getApiBase();
    const candidates = [primaryBase];
    if (primaryBase !== FALLBACK_API_BASE) {
        candidates.push(FALLBACK_API_BASE);
    }

    let lastError = null;
    for (const apiBase of candidates) {
        try {
            console.log(`[DF-Shield] Attempting Forensic Gateway: ${apiBase}`);
            return await runPipelineAgainstServer(apiBase, payload);
        } catch (err) {
            console.warn(`[DF-Shield] Gateway ${apiBase} failed:`, err.message);
            lastError = err;
            // If Gateway returned a client-side payload error (not network/timeout/5xx), don't retry fallback
            if (err.message && err.message.includes("400")) {
                throw err;
            }
        }
    }

    throw lastError || new Error("Không thể kết nối tới các Cổng Pháp Y Số.");
}

async function runPipelineAgainstServer(apiBase, payload) {
    // Step 1: Upload / Ingest Media
    const uploadEndpoint = `${apiBase}/api/upload`;
    let uploadBody = {};

    if (payload.type === "url" && payload.url) {
        uploadBody = { url: payload.url };
    } else if (payload.type === "base64" && payload.imageBase64) {
        uploadBody = { 
            image_base64: payload.imageBase64,
            source_url: payload.sourceUrl || ""
        };
    } else {
        throw new Error("Không có dữ liệu media hoặc URL hợp lệ.");
    }

    let uploadRes = await fetchWithTimeout(uploadEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(uploadBody)
    }, 25000);

    // Fallback: If Gateway returned 413 (Payload Too Large), retry with direct URL if available
    if (uploadRes.status === 413 && payload.sourceUrl && payload.sourceUrl.startsWith("http")) {
        console.warn("[DF-Shield] Received 413 on base64, falling back to direct URL ingestion:", payload.sourceUrl);
        uploadRes = await fetchWithTimeout(uploadEndpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: payload.sourceUrl })
        }, 25000);
    }

    if (!uploadRes.ok) {
        const errText = await uploadRes.text();
        throw new Error(`Lỗi tải lên Gateway (${uploadRes.status}): ${errText.substring(0, 100)}`);
    }

    const uploadData = await uploadRes.json();
    if (!uploadData.success || !uploadData.filepath) {
        throw new Error(uploadData.error || "Không thể nạp luồng phương tiện vào máy chủ forensic.");
    }

    // Step 2: Analyze Media through Tri-Stream Engine
    const analyzeEndpoint = `${apiBase}/api/analyze`;
    const analyzeRes = await fetchWithTimeout(analyzeEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            filepath: uploadData.filepath,
            language: "vi",
            source_url: payload.url || payload.sourceUrl || "",
            routing_mode: payload.routing_mode || "auto"
        })
    }, 55000);

    if (!analyzeRes.ok) {
        const errText = await analyzeRes.text();
        throw new Error(`Lỗi giám định phân tích (${analyzeRes.status}): ${errText.substring(0, 100)}`);
    }

    const analyzeData = await analyzeRes.json();
    if (!analyzeData.success) {
        throw new Error(analyzeData.error || "Quá trình phân tích đa chiều thất bại.");
    }

    // Return unified forensic decision bundle
    return {
        detection_result: analyzeData.detection_result,
        explanation: analyzeData.explanation,
        filename: uploadData.filename,
        source_url: payload.url || payload.sourceUrl || "",
        api_base: apiBase
    };
}
