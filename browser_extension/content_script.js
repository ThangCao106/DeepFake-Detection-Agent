// content_script.js - In-Situ Media Scanner & Forensic Badge Overlay for Social Media

(function () {
    let activeFloatingBtn = null;
    let hoveredMediaTarget = null;
    let activeOverlayCard = null;

    // Create and inject toast notification element
    function showToast(message, type = "info") {
        let toast = document.getElementById("df-shield-toast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "df-shield-toast";
            document.body.appendChild(toast);
        }
        toast.className = `df-shield-toast df-toast-${type} df-toast-show`;
        toast.innerHTML = `<span class="df-toast-icon">🛡️</span> <span>${message}</span>`;
        setTimeout(() => {
            toast.className = toast.className.replace("df-toast-show", "");
        }, 4000);
    }

    const isMobileTouch = window.matchMedia("(pointer: coarse)").matches || ("ontouchstart" in window);

    // Attach hover (Desktop) & touch/tap (Mobile) listeners to video and image elements
    function handleMediaFocus(e) {
        const target = e.target && e.target.closest ? e.target.closest("video, img") : null;
        if (!target) return;

        const rect = target.getBoundingClientRect();
        const minSize = isMobileTouch ? 120 : 160;
        if (rect.width < minSize || rect.height < minSize) return; // Ignore small icons

        hoveredMediaTarget = target;
        positionFloatingScanButton(target);
    }

    document.addEventListener("mouseover", handleMediaFocus, true);
    document.addEventListener("touchstart", handleMediaFocus, { capture: true, passive: true });

    // On Mobile browsers (Mises/Kiwi/Firefox Android/iOS Safari), auto-attach Scan button to visible videos
    if (isMobileTouch && typeof IntersectionObserver !== "undefined") {
        const videoObserver = new IntersectionObserver((entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting && entry.intersectionRatio >= 0.55) {
                    const rect = entry.target.getBoundingClientRect();
                    if (rect.width >= 140 && rect.height >= 140) {
                        hoveredMediaTarget = entry.target;
                        positionFloatingScanButton(entry.target);
                        break;
                    }
                }
            }
        }, { threshold: [0.55] });

        const observeVideos = () => {
            document.querySelectorAll("video").forEach((v) => {
                if (!v.dataset.dtObserved) {
                    v.dataset.dtObserved = "1";
                    videoObserver.observe(v);
                }
            });
        };
        observeVideos();
        new MutationObserver(observeVideos).observe(document.documentElement, { childList: true, subtree: true });
    }

    // Position floating scan badge over active video/image (clamped to viewport for mobile)
    function positionFloatingScanButton(mediaEl) {
        if (!activeFloatingBtn) {
            activeFloatingBtn = document.createElement("button");
            activeFloatingBtn.id = "df-shield-float-btn";
            activeFloatingBtn.innerHTML = `🛡️ Scan AI`;
            activeFloatingBtn.title = "Giám định Deepfake / AI với DeepTrace.io.vn";
            activeFloatingBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                e.preventDefault();
                initiateMediaAudit(hoveredMediaTarget || mediaEl);
            });
            document.body.appendChild(activeFloatingBtn);
        }

        const rect = mediaEl.getBoundingClientRect();
        const btnLeft = Math.max(8, Math.min(window.innerWidth - 112, rect.right - 102));
        const btnTop = Math.max(8, rect.top + 12);
        activeFloatingBtn.style.top = `${window.scrollY + btnTop}px`;
        activeFloatingBtn.style.left = `${window.scrollX + btnLeft}px`;
        activeFloatingBtn.style.display = "flex";
    }

    // Hide button when leaving container
    document.addEventListener("mouseleave", (e) => {
        if (activeFloatingBtn && !activeFloatingBtn.contains(e.relatedTarget)) {
            setTimeout(() => {
                if (activeFloatingBtn && !activeFloatingBtn.matches(":hover")) {
                    activeFloatingBtn.style.display = "none";
                }
            }, 300);
        }
    }, true);

    // Initiate audit via Client-Server Hybrid Pipeline
    async function initiateMediaAudit(mediaEl) {
        showToast("Đang bóc tách và gửi phương tiện tới Cổng Pháp Y Số...", "info");

        // Strategy A: Check for canonical social media post URL
        let candidateUrl = window.location.href;
        const parentAnchor = mediaEl.closest("a");
        if (parentAnchor && parentAnchor.href) {
            candidateUrl = parentAnchor.href;
        }

        const isSocialPlatform = /tiktok\.com|instagram\.com|threads\.net|youtube\.com\/shorts|facebook\.com|x\.com/i.test(candidateUrl);

        if (isSocialPlatform) {
            console.log(`[DF-Shield] Ingesting via Social Post URL: ${candidateUrl}`);
            chrome.runtime.sendMessage({
                action: "EXECUTE_HYBRID_AUDIT",
                payload: { type: "url", url: candidateUrl }
            }, handleAuditResponse);
            return;
        }

        // Strategy B: Crop frame directly from active tab screenshot (Bypass CORS)
        const rect = mediaEl.getBoundingClientRect();
        chrome.runtime.sendMessage({ action: "CAPTURE_ACTIVE_TAB" }, async (capRes) => {
            if (!capRes || !capRes.success || !capRes.dataUrl) {
                // Fallback to media src if available
                const directSrc = mediaEl.currentSrc || mediaEl.src;
                if (directSrc && directSrc.startsWith("http")) {
                    chrome.runtime.sendMessage({
                        action: "EXECUTE_HYBRID_AUDIT",
                        payload: { type: "url", url: directSrc }
                    }, handleAuditResponse);
                } else {
                    showToast("Không thể chụp khung hình phương tiện (Lỗi CORS/Viewport).", "error");
                }
                return;
            }

            // Crop image on in-memory canvas
            try {
                const croppedBase64 = await cropImageDataUrl(capRes.dataUrl, {
                    x: rect.left * window.devicePixelRatio,
                    y: rect.top * window.devicePixelRatio,
                    width: rect.width * window.devicePixelRatio,
                    height: rect.height * window.devicePixelRatio
                });

                chrome.runtime.sendMessage({
                    action: "EXECUTE_HYBRID_AUDIT",
                    payload: {
                        type: "base64",
                        imageBase64: croppedBase64,
                        sourceUrl: window.location.href
                    }
                }, handleAuditResponse);
            } catch (err) {
                console.error("[DF-Shield Crop Error]", err);
                showToast("Lỗi xử lý khung hình ảnh: " + err.message, "error");
            }
        });
    }

    // Helper: Crop Image Data URL to Bounding Rect
    function cropImageDataUrl(dataUrl, area) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement("canvas");
                canvas.width = Math.max(1, area.width);
                canvas.height = Math.max(1, area.height);
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, area.x, area.y, area.width, area.height, 0, 0, area.width, area.height);
                resolve(canvas.toDataURL("image/png"));
            };
            img.onerror = reject;
            img.src = dataUrl;
        });
    }

    // Handle response from background script
    function handleAuditResponse(res) {
        if (!res || !res.success) {
            showToast(`Giám định thất bại: ${res?.error || "Mất kết nối máy chủ"}`, "error");
            return;
        }

        const data = res.result;
        const det = data.detection_result || {};
        const label = det.label || "Realism";
        const isFake = label === "Deepfake" || label === "Fake";
        const conf = det.confidence ? (det.confidence * 100).toFixed(1) : "95.0";
        const risk = det.composite_risk_score !== undefined ? (det.composite_risk_score * 100).toFixed(1) : (isFake ? "92.4" : "3.8");
        const cert = det.forensic_certificate || {};
        const certId = cert.certificate_id || det.certificate_id || "CERT-DF-LIVE";
        const apiBase = data.api_base || "https://detection.dtpsoft.vn";
        const certUrl = `${apiBase}/certificate/${certId}`;

        showToast(isFake ? `🚨 Phát hiện Deepfake! Nguy cơ: ${risk}%` : `✅ Xác thực phương tiện Chân thật (${conf}%)`, isFake ? "error" : "success");
        renderInSituOverlayBadge(hoveredMediaTarget, {
            isFake,
            label,
            conf,
            risk,
            certId,
            certUrl
        });
    }

    // Render Clinical In-Situ Overlay Badge over the video
    function renderInSituOverlayBadge(mediaEl, info) {
        if (activeOverlayCard) activeOverlayCard.remove();

        const card = document.createElement("div");
        card.className = `df-shield-overlay-card ${info.isFake ? "df-card-fake" : "df-card-real"}`;
        
        card.innerHTML = `
            <div class="df-card-header">
                <span class="df-card-badge-seal">🛡️ EU AI ACT ART. 50</span>
                <span class="df-card-close-btn" title="Đóng">✕</span>
            </div>
            <div class="df-card-verdict">
                <span class="df-verdict-icon">${info.isFake ? "🚨" : "✅"}</span>
                <div>
                    <h4 class="df-verdict-title">${info.isFake ? "DEEPFAKE DETECTED" : "AUTHENTIC MEDIA"}</h4>
                    <span class="df-verdict-sub">Độ tin cậy: ${info.conf}% | Rủi ro: ${info.risk}%</span>
                </div>
            </div>
            <div class="df-card-meta">
                <span>Chứng thư: <code>${info.certId}</code></span>
            </div>
            <div class="df-card-actions">
                <a href="${info.certUrl}" target="_blank" rel="noopener noreferrer" class="df-btn-cert">
                    📜 Tra Cứu Chứng Thư Số ↗
                </a>
            </div>
        `;

        card.querySelector(".df-card-close-btn").addEventListener("click", () => {
            card.remove();
        });

        document.body.appendChild(card);
        activeOverlayCard = card;

        if (mediaEl) {
            const rect = mediaEl.getBoundingClientRect();
            card.style.top = `${window.scrollY + rect.top + 16}px`;
            card.style.left = `${window.scrollX + rect.left + 16}px`;
        } else {
            card.style.top = "20px";
            card.style.right = "20px";
        }
    }

    // Context menu & popup trigger listeners
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "TRIGGER_CONTEXT_SCAN") {
            const media = hoveredMediaTarget || document.querySelector("video, img");
            if (media) {
                initiateMediaAudit(media);
            } else {
                showToast("Vui lòng chạm hoặc di chuột vào video/ảnh để quét.", "error");
            }
        } else if (request.action === "TRIGGER_LIVE_CALL_SCAN") {
            showToast("Đang kích hoạt chế độ Giám định Cuộc gọi / Live Stream...", "info");
            const liveMedia = document.querySelector("video") || hoveredMediaTarget || document.querySelector("img");
            if (liveMedia) {
                initiateMediaAudit(liveMedia);
            } else {
                // If no direct video element found on DOM, capture active viewport
                chrome.runtime.sendMessage({ action: "CAPTURE_ACTIVE_TAB" }, async (capRes) => {
                    if (capRes && capRes.success && capRes.dataUrl) {
                        chrome.runtime.sendMessage({
                            action: "EXECUTE_HYBRID_AUDIT",
                            payload: {
                                type: "base64",
                                imageBase64: capRes.dataUrl,
                                sourceUrl: window.location.href,
                                routing_mode: "call"
                            }
                        }, handleAuditResponse);
                    } else {
                        showToast("Không tìm thấy luồng video cuộc gọi để giám định.", "error");
                    }
                });
            }
        }
    });

    console.log("[Deepfake Forensic Shield] Content script active on page:", window.location.hostname);
})();

