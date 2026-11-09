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
            chrome.tabs.sendMessage(tab.id, { action: "TRIGGER_CONTEXT_SCAN" });
            window.close(); // Close popup so user sees in-page scan banner
        }
    });

    // Open Full Cockpit Web App
    openCockpitBtn.addEventListener("click", () => {
        const targetUrl = endpointSelect.value || "https://detection.dtpsoft.vn";
        chrome.tabs.create({ url: targetUrl });
    });
});
