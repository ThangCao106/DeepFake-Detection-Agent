// background.js - Service Worker for Deepfake Forensic Shield (Manifest V3)

const DEFAULT_API_BASE = "https://detection.dtpsoft.vn";

// Initialize Context Menus and Storage Defaults
chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.get(["apiBase"], (res) => {
        if (!res.apiBase) {
            chrome.storage.local.set({ apiBase: DEFAULT_API_BASE });
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
            resolve(res.apiBase || DEFAULT_API_BASE);
        });
    });
}

// Handle messages from content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "CAPTURE_ACTIVE_TAB") {
        chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
            if (chrome.runtime.lastError || !dataUrl) {
                sendResponse({ success: false, error: chrome.runtime.lastError?.message || "Failed to capture tab" });
            } else {
                sendResponse({ success: true, dataUrl });
            }
        });
        return true; // Keep message channel open for async response
    }

    if (request.action === "EXECUTE_HYBRID_AUDIT") {
        executeForensicPipeline(request.payload)
            .then((result) => sendResponse({ success: true, result }))
            .catch((err) => sendResponse({ success: false, error: err.message }));
        return true;
    }
});

/**
 * Execute Client-Server Hybrid Forensic Pipeline
 */
async function executeForensicPipeline(payload) {
    const apiBase = await getApiBase();
    console.log(`[DF-Shield] Dispatching to Forensic Gateway: ${apiBase}`);

    // Step 1: Upload / Ingest Media
    let uploadEndpoint = `${apiBase}/api/upload`;
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

    let uploadRes = await fetch(uploadEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(uploadBody)
    });

    // Fallback: If Gateway returned 413 (Payload Too Large from reverse proxy), fallback to sending direct media URL if available
    if (uploadRes.status === 413 && payload.sourceUrl && payload.sourceUrl.startsWith("http")) {
        console.warn("[DF-Shield] Received 413 on base64, falling back to direct URL ingestion:", payload.sourceUrl);
        uploadRes = await fetch(uploadEndpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: payload.sourceUrl })
        });
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
    const analyzeRes = await fetch(analyzeEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            filepath: uploadData.filepath,
            language: "vi",
            source_url: payload.url || payload.sourceUrl || "",
            routing_mode: "auto"
        })
    });

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

