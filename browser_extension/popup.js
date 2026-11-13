// popup.js - Interactive Popup Controls for Deepfake Forensic Shield

document.addEventListener("DOMContentLoaded", () => {
    const scanBtn = document.getElementById("scanCurrentBtn");
    const openCockpitBtn = document.getElementById("openCockpitBtn");
    const endpointSelect = document.getElementById("apiEndpointSelect");
    const gatewayStatus = document.getElementById("gatewayStatus");

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

    // Trigger Scan on Active Tab
    scanBtn.addEventListener("click", async () => {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab?.id) {
            chrome.tabs.sendMessage(tab.id, { action: "TRIGGER_CONTEXT_SCAN", mode: "standard" });
            window.close(); // Close popup so user sees in-page scan banner
        }
    });

    // Trigger Live Stream / Real-time Video Call Guardian
    const scanLiveCallBtn = document.getElementById("scanLiveCallBtn");
    if (scanLiveCallBtn) {
        scanLiveCallBtn.addEventListener("click", async () => {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tab?.id) {
                chrome.tabs.sendMessage(tab.id, { action: "TRIGGER_LIVE_CALL_SCAN", mode: "live_call" });
                window.close();
            }
        });
    }

    // Open Full Cockpit Web App
    openCockpitBtn.addEventListener("click", () => {
        const targetUrl = endpointSelect.value || "https://detection.dtpsoft.vn";
        chrome.tabs.create({ url: targetUrl });
    });
});

