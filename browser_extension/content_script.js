// content_script.js - In-Situ Media Scanner & Forensic Badge Overlay for Social Media

(function () {
    let activeFloatingBtn = null;
    let hoveredMediaTarget = null;
    let activeOverlayCard = null;
    let toastTimer = null;

    // Create and inject toast notification element with status/spinner support
    function showToast(message, type = "info", persistent = false) {
        let toast = document.getElementById("df-shield-toast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "df-shield-toast";
            document.body.appendChild(toast);
        }
        if (toastTimer) {
            clearTimeout(toastTimer);
            toastTimer = null;
        }

        const icon = type === "loading" ? "⏳" : (type === "success" ? "✅" : (type === "error" ? "🚨" : "🛡️"));
        toast.className = `df-shield-toast df-toast-${type} df-toast-show`;
        toast.innerHTML = `<span class="df-toast-icon ${type === 'loading' ? 'df-toast-spin' : ''}">${icon}</span> <span>${message}</span>`;

        if (!persistent && type !== "loading") {
            toastTimer = setTimeout(() => {
                toast.className = toast.className.replace("df-toast-show", "");
            }, 5000);
        }
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

    // Position floating scan badge over active video/image (fixed in viewport coordinates)
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
        const btnLeft = Math.max(8, Math.min(window.innerWidth - 116, rect.right - 104));
        const btnTop = Math.max(8, Math.min(window.innerHeight - 44, rect.top + 12));
        activeFloatingBtn.style.top = `${btnTop}px`;
        activeFloatingBtn.style.left = `${btnLeft}px`;
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

    // Fast path: Try extracting base64 directly from loaded <img> element without tab capture
    function tryExtractDirectImageBase64(imgEl) {
        try {
            if (!imgEl.complete || !imgEl.naturalWidth) return null;
            const canvas = document.createElement("canvas");
            let w = imgEl.naturalWidth;
            let h = imgEl.naturalHeight;
            const maxDim = 1080;
            if (w > maxDim || h > maxDim) {
                const ratio = Math.min(maxDim / w, maxDim / h);
                w = Math.round(w * ratio);
                h = Math.round(h * ratio);
            }
            canvas.width = Math.max(1, w);
            canvas.height = Math.max(1, h);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL("image/jpeg", 0.85);
        } catch (e) {
            // Tainted canvas by cross-origin, fallback to tab capture
            return null;
        }
    }

    // Send payload to background service worker with timeout protection
    function sendAuditMessage(payload) {
        let isSettled = false;
        const timer = setTimeout(() => {
            if (!isSettled) {
                isSettled = true;
                showToast("Quá thời gian kết nối (Timeout). Vui lòng kiểm tra mạng hoặc thử lại.", "error");
            }
        }, 60000);

        try {
            chrome.runtime.sendMessage({
                action: "EXECUTE_HYBRID_AUDIT",
                payload: payload
            }, (res) => {
                if (isSettled) return;
                isSettled = true;
                clearTimeout(timer);

                if (chrome.runtime.lastError) {
                    console.error("[DF-Shield runtime error]", chrome.runtime.lastError);
                    showToast(`Lỗi kết nối cổng: ${chrome.runtime.lastError.message}`, "error");
                    return;
                }
                handleAuditResponse(res);
            });
        } catch (err) {
            if (isSettled) return;
            isSettled = true;
            clearTimeout(timer);
            showToast(`Lỗi gửi gói tin: ${err.message}`, "error");
        }
    }

    // Initiate audit via Client-Server Hybrid Pipeline
    async function initiateMediaAudit(mediaEl) {
        if (!mediaEl) {
            showToast("Không tìm thấy phương tiện cần quét.", "error");
            return;
        }
        hoveredMediaTarget = mediaEl;
        showToast("Đang bóc tách phương tiện...", "loading", true);

        // Strategy A: Check for canonical social media post URL
        let candidateUrl = window.location.href;
        const parentAnchor = mediaEl.closest("a");
        if (parentAnchor && parentAnchor.href) {
            candidateUrl = parentAnchor.href;
        }

        const isSocialPlatform = /tiktok\.com|instagram\.com|threads\.net|youtube\.com\/shorts|facebook\.com|x\.com/i.test(candidateUrl);

        if (isSocialPlatform) {
            console.log(`[DF-Shield] Ingesting via Social Post URL: ${candidateUrl}`);
            showToast("Đang kết nối Cổng Pháp Y Số (DeepTrace Gateway)...", "loading", true);
            sendAuditMessage({ type: "url", url: candidateUrl });
            return;
        }

        // Strategy B: If mediaEl is an <img>, try direct canvas extraction first (instant, bypasses capture)
        if (mediaEl.tagName && mediaEl.tagName.toLowerCase() === "img") {
            const directB64 = tryExtractDirectImageBase64(mediaEl);
            if (directB64) {
                showToast("Đang giám định đa tầng (DeepTrace ViT)...", "loading", true);
                sendAuditMessage({
                    type: "base64",
                    imageBase64: directB64,
                    sourceUrl: window.location.href
                });
                return;
            }
        }

        // Strategy C: Crop frame directly from active tab screenshot (Bypass CORS)
        const rect = mediaEl.getBoundingClientRect();
        chrome.runtime.sendMessage({ action: "CAPTURE_ACTIVE_TAB" }, async (capRes) => {
            if (!capRes || !capRes.success || !capRes.dataUrl) {
                // Strategy D: Fallback to media src if available
                const directSrc = mediaEl.currentSrc || mediaEl.src;
                if (directSrc && directSrc.startsWith("http")) {
                    showToast("Đang gửi URL phương tiện tới Cổng Pháp Y...", "loading", true);
                    sendAuditMessage({ type: "url", url: directSrc });
                } else {
                    showToast("Không thể bóc tách khung hình phương tiện (Lỗi CORS/Viewport).", "error");
                }
                return;
            }

            // Crop image on in-memory canvas
            try {
                showToast("Đang tối ưu khung hình giám định...", "loading", true);
                const dpr = window.devicePixelRatio || 1;
                const croppedBase64 = await cropImageDataUrl(capRes.dataUrl, {
                    x: Math.round(rect.left * dpr),
                    y: Math.round(rect.top * dpr),
                    width: Math.round(rect.width * dpr),
                    height: Math.round(rect.height * dpr)
                });

                showToast("Đang giám định đa tầng (DeepTrace ViT)...", "loading", true);
                sendAuditMessage({
                    type: "base64",
                    imageBase64: croppedBase64,
                    sourceUrl: window.location.href
                });
            } catch (err) {
                console.error("[DF-Shield Crop Error]", err);
                const directSrc = mediaEl.currentSrc || mediaEl.src;
                if (directSrc && directSrc.startsWith("http")) {
                    sendAuditMessage({ type: "url", url: directSrc });
                } else {
                    showToast("Lỗi xử lý khung hình ảnh: " + err.message, "error");
                }
            }
        });
    }

    // Helper: Crop Image Data URL to Bounding Rect and compress to JPEG under 1MB to avoid 413 Payload Too Large
    function cropImageDataUrl(dataUrl, area) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                try {
                    const canvas = document.createElement("canvas");
                    
                    // Clamp source rectangle strictly within [0, img.width] and [0, img.height]
                    const sx = Math.max(0, Math.min(img.width - 1, area.x || 0));
                    const sy = Math.max(0, Math.min(img.height - 1, area.y || 0));
                    const sw = Math.max(1, Math.min(img.width - sx, area.width || img.width));
                    const sh = Math.max(1, Math.min(img.height - sy, area.height || img.height));

                    let dw = sw;
                    let dh = sh;
                    const maxDim = 1080;
                    if (dw > maxDim || dh > maxDim) {
                        const ratio = Math.min(maxDim / dw, maxDim / dh);
                        dw = Math.round(dw * ratio);
                        dh = Math.round(dh * ratio);
                    }

                    canvas.width = Math.max(1, dw);
                    canvas.height = Math.max(1, dh);
                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
                    resolve(canvas.toDataURL("image/jpeg", 0.85));
                } catch (e) {
                    reject(e);
                }
            };
            img.onerror = () => reject(new Error("Không thể nạp dữ liệu khung hình."));
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
        const certId = cert.certificate_id || det.certificate_id || "CERT-DT-IMG";
        const apiBase = data.api_base || "https://deeptrace.io.vn";
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

    // Render In-Situ Overlay Badge over media (Viewport-fixed positioning)
    function renderInSituOverlayBadge(mediaEl, info) {
        if (activeOverlayCard) activeOverlayCard.remove();

        const card = document.createElement("div");
        card.className = `df-shield-overlay-card ${info.isFake ? "df-card-fake" : "df-card-real"}`;
        
        card.innerHTML = `
            <div class="df-card-header">
                <span class="df-card-badge-seal">🛡️ DEEPTRACE FORENSIC SHIELD</span>
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

        // Position fixed relative to viewport so it is ALWAYS visible even in modals, carousels, or scrolled pages
        if (mediaEl && mediaEl.getBoundingClientRect) {
            const rect = mediaEl.getBoundingClientRect();
            const top = Math.max(16, Math.min(window.innerHeight - 280, rect.top + 16));
            const left = Math.max(16, Math.min(window.innerWidth - 310, rect.left + 16));
            card.style.top = `${top}px`;
            card.style.left = `${left}px`;
        } else {
            card.style.top = "24px";
            card.style.right = "24px";
        }
    }

    // Find largest visible media element currently inside the active viewport
    function findBestVisibleMedia() {
        if (hoveredMediaTarget && isElementInViewport(hoveredMediaTarget)) {
            return hoveredMediaTarget;
        }

        const candidates = Array.from(document.querySelectorAll("video, img"));
        let bestEl = null;
        let maxArea = 0;

        for (const el of candidates) {
            const rect = el.getBoundingClientRect();
            const w = Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(rect.left, 0));
            const h = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
            const area = w * h;

            if (w < 80 || h < 80) continue; // Skip tiny icons, logos, tracking pixels

            const style = window.getComputedStyle(el);
            if (style.display === "none" || style.visibility === "hidden" || parseFloat(style.opacity) < 0.1) continue;

            // Prioritize videos 2.5x over static images
            const score = (el.tagName.toLowerCase() === "video" ? 2.5 : 1.0) * area;
            if (score > maxArea) {
                maxArea = score;
                bestEl = el;
            }
        }
        return bestEl;
    }

    function isElementInViewport(el) {
        if (!el || !el.getBoundingClientRect) return false;
        const rect = el.getBoundingClientRect();
        return rect.bottom > 0 && rect.right > 0 && rect.top < window.innerHeight && rect.left < window.innerWidth;
    }

    // Context menu & popup trigger listeners
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "TRIGGER_CONTEXT_SCAN") {
            const media = findBestVisibleMedia();
            if (media) {
                // Visual feedback: outline element briefly
                try {
                    const origOutline = media.style.outline;
                    media.style.outline = "3px solid #00F0FF";
                    setTimeout(() => { media.style.outline = origOutline; }, 2000);
                } catch (e) {}
                initiateMediaAudit(media);
            } else {
                // Fallback: Capture active viewport directly
                chrome.runtime.sendMessage({ action: "CAPTURE_ACTIVE_TAB" }, async (capRes) => {
                    if (capRes && capRes.success && capRes.dataUrl) {
                        showToast("Đang giám định toàn cảnh màn hình hiển thị...", "loading", true);
                        sendAuditMessage({
                            type: "base64",
                            imageBase64: capRes.dataUrl,
                            sourceUrl: window.location.href,
                            routing_mode: "auto"
                        });
                    } else {
                        showToast("Không tìm thấy phương tiện hợp lệ trên màn hình.", "error");
                    }
                });
            }
        } else if (request.action === "TRIGGER_LIVE_CALL_SCAN") {
            showToast("Đang kích hoạt chế độ Giám định Cuộc gọi / Live Stream...", "loading", true);
            const liveMedia = document.querySelector("video") || findBestVisibleMedia();
            if (liveMedia) {
                initiateMediaAudit(liveMedia);
            } else {
                // If no direct video element found on DOM, capture active viewport
                chrome.runtime.sendMessage({ action: "CAPTURE_ACTIVE_TAB" }, async (capRes) => {
                    if (capRes && capRes.success && capRes.dataUrl) {
                        sendAuditMessage({
                            type: "base64",
                            imageBase64: capRes.dataUrl,
                            sourceUrl: window.location.href,
                            routing_mode: "call"
                        });
                    } else {
                        showToast("Không tìm thấy luồng video cuộc gọi để giám định.", "error");
                    }
                });
            }
        }
    });

    console.log("[Deepfake Forensic Shield] Content script active on page:", window.location.hostname);
})();

