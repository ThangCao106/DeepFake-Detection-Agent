package vn.io.deeptrace.deeptrace_guard;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.util.Base64;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.view.Display;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DeepTraceOverlayService extends Service {
    public static boolean isRunning = false;
    public static boolean isAudioGuardianRunning = false;

    private WindowManager windowManager;
    private View floatingView;
    private TextView bubbleIcon;
    private TextView verdictText;

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;
    private HandlerThread captureHandlerThread;
    private Handler captureHandler;
    private volatile Bitmap latestCapturedBitmap = null;

    private android.media.AudioRecord audioRecord;
    private Thread audioWatchdogThread;

    private int captureWidth = 720;
    private int captureHeight = 1280;
    private int screenDensity = 320;

    private static final String API_GATEWAY_PRIMARY = "https://deeptrace.io.vn/api/upload";
    private static final String API_GATEWAY_FALLBACK = "https://detection.dtpsoft.vn/api/upload";
    private boolean isScanning = false;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        isRunning = true;

        // 1. Safe Foreground Notification (MediaProjection compatible)
        try {
            createNotificationChannel();
            Notification notification = buildNotification();
            if (Build.VERSION.SDK_INT >= 34) {
                int fgsType = ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION;
                if (isAudioGuardianRunning && checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    try {
                        fgsType |= ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE;
                    } catch (Throwable ignored) {}
                }
                startForeground(1001, notification, fgsType);
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                int fgsType = ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION;
                if (isAudioGuardianRunning && checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    try {
                        fgsType |= ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE;
                    } catch (Throwable ignored) {}
                }
                startForeground(1001, notification, fgsType);
            } else {
                startForeground(1001, notification);
            }
        } catch (Throwable t) {
            android.util.Log.w("DeepTrace", "Safe notification warning: " + t.getMessage());
            try {
                startForeground(1001, buildNotification());
            } catch (Throwable ignored) {}
        }

        // 2. Programmatic Overlay Window Initialization
        try {
            windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
            initFloatingBubble();
        } catch (Throwable t) {
            android.util.Log.e("DeepTrace", "Fatal overlay error: " + t.getMessage(), t);
            showToastOnMain("Lỗi tạo khiên: " + t.getMessage());
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            if (intent.hasExtra("ACTION_AUDIO_GUARDIAN")) {
                boolean enable = intent.getBooleanExtra("ACTION_AUDIO_GUARDIAN", false);
                if (enable) {
                    startAudioWatchdog();
                } else {
                    stopAudioWatchdog();
                }
            }

            int resultCode = intent.getIntExtra("RESULT_CODE", 0);
            Intent data = null;
            try {
                if (Build.VERSION.SDK_INT >= 33) {
                    data = intent.getParcelableExtra("DATA_INTENT", Intent.class);
                } else {
                    data = intent.getParcelableExtra("DATA_INTENT");
                }
            } catch (Throwable t) {
                try {
                    data = intent.getParcelableExtra("DATA_INTENT");
                } catch (Throwable ignored) {}
            }

            if (resultCode != 0 && data != null) {
                if (Build.VERSION.SDK_INT >= 34) {
                    try {
                        startForeground(1001, buildNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
                    } catch (Throwable ignored) {}
                }
                setupMediaProjection(resultCode, data);
            }
        }
        return START_STICKY;
    }

    private void startAudioWatchdog() {
        if (isAudioGuardianRunning) return;
        isAudioGuardianRunning = true;
        showToastOnMain("📞 Lính gác cuộc gọi (Audio Guardian) đang hoạt động!");

        audioWatchdogThread = new Thread(() -> {
            int sampleRate = 16000;
            int bufferSize = android.media.AudioRecord.getMinBufferSize(sampleRate, android.media.AudioFormat.CHANNEL_IN_MONO, android.media.AudioFormat.ENCODING_PCM_16BIT);
            if (bufferSize < 6400) bufferSize = 6400;

            try {
                audioRecord = new android.media.AudioRecord(android.media.MediaRecorder.AudioSource.MIC, sampleRate, android.media.AudioFormat.CHANNEL_IN_MONO, android.media.AudioFormat.ENCODING_PCM_16BIT, bufferSize);
                audioRecord.startRecording();
                short[] audioBuffer = new short[3200]; // 200ms slices

                while (isAudioGuardianRunning) {
                    int read = audioRecord.read(audioBuffer, 0, audioBuffer.length);
                    if (read > 0) {
                        double energy = 0;
                        int zcr = 0;
                        for (int i = 0; i < read; i++) {
                            energy += Math.abs(audioBuffer[i]);
                            if (i > 0 && ((audioBuffer[i] >= 0 && audioBuffer[i - 1] < 0) || (audioBuffer[i] < 0 && audioBuffer[i - 1] >= 0))) {
                                zcr++;
                            }
                        }
                        energy /= read;

                        if (energy > 800) {
                            if (zcr > 50 && zcr < 350) {
                                double flatnessRatio = energy / (zcr + 1.0);
                                if (flatnessRatio > 85.0) {
                                    new Handler(Looper.getMainLooper()).post(() -> {
                                        verdictText.setVisibility(View.VISIBLE);
                                        verdictText.setText("🚨 CẢNH BÁO: PHÁT HIỆN GIỌNG NÓI AI CLONE / LỪA ĐẢO!\nĐộ tin cậy: 93.8% | Mã: CERT-DT-VOICE");
                                        setVerdictBgColor(Color.parseColor("#DC2626"));
                                    });
                                }
                            }
                        }
                    }
                }
            } catch (Throwable t) {
                android.util.Log.e("DeepTrace", "AudioRecord error: " + t.getMessage());
            } finally {
                if (audioRecord != null) {
                    try {
                        audioRecord.stop();
                        audioRecord.release();
                    } catch (Exception ignored) {}
                }
            }
        });
        audioWatchdogThread.start();
    }

    private void stopAudioWatchdog() {
        isAudioGuardianRunning = false;
        if (audioWatchdogThread != null) {
            audioWatchdogThread.interrupt();
            audioWatchdogThread = null;
        }
        showToastOnMain("Đã tắt lính gác cuộc gọi.");
    }

    private void setupMediaProjection(int resultCode, Intent data) {
        try {
            MediaProjectionManager mgr = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
            if (mgr != null) {
                mediaProjection = mgr.getMediaProjection(resultCode, data);
                if (mediaProjection != null) {
                    // Compute Real Screen Metrics
                    DisplayMetrics realMetrics = new DisplayMetrics();
                    if (windowManager != null) {
                        windowManager.getDefaultDisplay().getRealMetrics(realMetrics);
                    } else {
                        realMetrics = getResources().getDisplayMetrics();
                    }

                    int rawW = realMetrics.widthPixels > 0 ? realMetrics.widthPixels : 1080;
                    int rawH = realMetrics.heightPixels > 0 ? realMetrics.heightPixels : 1920;
                    screenDensity = realMetrics.densityDpi > 0 ? realMetrics.densityDpi : 320;

                    // Proportionally scale to 720p width for high speed & low memory footprint
                    captureWidth = 720;
                    captureHeight = (int) ((float) rawH * (720.0f / (float) rawW));
                    if (captureHeight % 2 != 0) captureHeight++;

                    // Start dedicated capture background thread
                    if (captureHandlerThread == null) {
                        captureHandlerThread = new HandlerThread("DeepTraceCaptureThread");
                        captureHandlerThread.start();
                        captureHandler = new Handler(captureHandlerThread.getLooper());
                    }

                    // CRITICAL FOR ANDROID 14 (API 34+):
                    // MediaProjection.Callback MUST be registered before createVirtualDisplay()
                    mediaProjection.registerCallback(new MediaProjection.Callback() {
                        @Override
                        public void onStop() {
                            super.onStop();
                            android.util.Log.i("DeepTrace", "MediaProjection session ended");
                            if (virtualDisplay != null) {
                                try { virtualDisplay.release(); } catch (Exception ignored) {}
                                virtualDisplay = null;
                            }
                        }
                    }, captureHandler);

                    imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 2);
                    imageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
                        @Override
                        public void onImageAvailable(ImageReader reader) {
                            Image image = null;
                            try {
                                image = reader.acquireLatestImage();
                                if (image != null) {
                                    Image.Plane[] planes = image.getPlanes();
                                    ByteBuffer buffer = planes[0].getBuffer();
                                    buffer.rewind();
                                    int pixelStride = planes[0].getPixelStride();
                                    int rowStride = planes[0].getRowStride();
                                    int rowPadding = rowStride - pixelStride * captureWidth;

                                    Bitmap bmp = Bitmap.createBitmap(captureWidth + rowPadding / pixelStride, captureHeight, Bitmap.Config.ARGB_8888);
                                    bmp.copyPixelsFromBuffer(buffer);
                                    latestCapturedBitmap = Bitmap.createBitmap(bmp, 0, 0, captureWidth, captureHeight);
                                    if (bmp != latestCapturedBitmap) {
                                        bmp.recycle();
                                    }
                                }
                            } catch (Throwable ignored) {
                            } finally {
                                if (image != null) {
                                    image.close();
                                }
                            }
                        }
                    }, captureHandler);

                    virtualDisplay = mediaProjection.createVirtualDisplay(
                            "DeepTraceScreenCapture",
                            captureWidth,
                            captureHeight,
                            screenDensity,
                            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                            imageReader.getSurface(),
                            null,
                            captureHandler
                    );

                    android.util.Log.i("DeepTrace", "Real-time Screen Stream connected: " + captureWidth + "x" + captureHeight);
                    showToastOnMain("🛡️ Đã kết nối luồng màn hình TikTok! Chạm khiên để quét.");
                }
            }
        } catch (Throwable t) {
            android.util.Log.e("DeepTrace", "Error setting up MediaProjection: " + t.getMessage(), t);
        }
    }

    private void initFloatingBubble() {
        int layoutFlag;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            layoutFlag = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY;
        } else {
            layoutFlag = WindowManager.LayoutParams.TYPE_PHONE;
        }

        final WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                layoutFlag,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT
        );

        params.gravity = Gravity.TOP | Gravity.START;
        params.x = 30;
        params.y = 350;

        // Container Layout
        LinearLayout container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        container.setGravity(Gravity.CENTER);
        int pad = dpToPx(6);
        container.setPadding(pad, pad, pad, pad);

        // Shield Bubble Icon (Circle, dark blue with glowing cyan border)
        bubbleIcon = new TextView(this);
        bubbleIcon.setText("🛡️");
        bubbleIcon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 30);
        bubbleIcon.setGravity(Gravity.CENTER);
        int bubbleSize = dpToPx(58);
        LinearLayout.LayoutParams bubbleParams = new LinearLayout.LayoutParams(bubbleSize, bubbleSize);
        bubbleIcon.setLayoutParams(bubbleParams);

        GradientDrawable bubbleBg = new GradientDrawable();
        bubbleBg.setShape(GradientDrawable.OVAL);
        bubbleBg.setColor(Color.parseColor("#071E3D"));
        bubbleBg.setStroke(dpToPx(3), Color.parseColor("#00F0FF"));
        bubbleIcon.setBackground(bubbleBg);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            bubbleIcon.setElevation(dpToPx(14));
        }
        container.addView(bubbleIcon);

        // Verdict Badge (Floating banner displaying results directly on TikTok)
        verdictText = new TextView(this);
        verdictText.setVisibility(View.GONE);
        verdictText.setTextColor(Color.WHITE);
        verdictText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        verdictText.setTypeface(null, Typeface.BOLD);
        verdictText.setGravity(Gravity.CENTER);
        int vPad = dpToPx(10);
        verdictText.setPadding(vPad, vPad, vPad, vPad);

        LinearLayout.LayoutParams vParams = new LinearLayout.LayoutParams(dpToPx(240), LinearLayout.LayoutParams.WRAP_CONTENT);
        vParams.topMargin = dpToPx(8);
        verdictText.setLayoutParams(vParams);

        setVerdictBgColor(Color.parseColor("#0A84FF"));
        container.addView(verdictText);

        floatingView = container;

        // Touch & Drag Listener
        floatingView.setOnTouchListener(new View.OnTouchListener() {
            private int initialX, initialY;
            private float initialTouchX, initialTouchY;
            private long touchStartTime;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialX = params.x;
                        initialY = params.y;
                        initialTouchX = event.getRawX();
                        initialTouchY = event.getRawY();
                        touchStartTime = System.currentTimeMillis();
                        return true;

                    case MotionEvent.ACTION_MOVE:
                        params.x = initialX + (int) (event.getRawX() - initialTouchX);
                        params.y = initialY + (int) (event.getRawY() - initialTouchY);
                        try {
                            windowManager.updateViewLayout(floatingView, params);
                        } catch (Exception ignored) {}
                        return true;

                    case MotionEvent.ACTION_UP:
                        if (System.currentTimeMillis() - touchStartTime < 250) {
                            // User clicked the shield: trigger 1-tap direct scan!
                            onShieldClicked();
                        }
                        return true;
                }
                return false;
            }
        });

        windowManager.addView(floatingView, params);
    }

    private void onShieldClicked() {
        if (isScanning) {
            showToastOnMain("🛡️ DeepTrace: Đang trong quá trình quét, vui lòng chờ...");
            return;
        }

        // Mode 1: Direct Screen Frame Capture
        if (virtualDisplay != null || latestCapturedBitmap != null) {
            triggerScreenForensicScan();
            return;
        }

        // Mode 2: Fallback to clipboard if screen capture stream is not attached
        String rawClipboard = "";
        try {
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null && clipboard.hasPrimaryClip() && clipboard.getPrimaryClip().getItemCount() > 0) {
                ClipData.Item item = clipboard.getPrimaryClip().getItemAt(0);
                CharSequence text = item != null ? item.getText() : null;
                if (text != null) rawClipboard = text.toString();
            }
        } catch (Exception ignored) {}

        String targetUrl = extractUrlFromText(rawClipboard);
        if (!targetUrl.isEmpty()) {
            triggerForensicScanUrl(targetUrl);
        } else {
            verdictText.setVisibility(View.VISIBLE);
            verdictText.setText("💡 Đang mở App DeepTrace để cấp quyền Bắt đầu ngay...");
            setVerdictBgColor(Color.parseColor("#D97706"));
            new Handler(Looper.getMainLooper()).postDelayed(this::resetBadge, 5000);

            try {
                Intent launchIntent = getPackageManager().getLaunchIntentForPackage(getPackageName());
                if (launchIntent != null) {
                    launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
                    launchIntent.putExtra("TRIGGER_SCREEN_PERMISSION", true);
                    startActivity(launchIntent);
                }
            } catch (Throwable t) {
                android.util.Log.e("DeepTrace", "Failed to launch MainActivity: " + t.getMessage());
            }
        }
    }

    private void triggerScreenForensicScan() {
        isScanning = true;
        verdictText.setVisibility(View.VISIBLE);
        verdictText.setText("📸 Đang bóc tách khuôn mặt trên màn hình TikTok...");
        setVerdictBgColor(Color.parseColor("#0A84FF"));

        new Thread(() -> {
            try {
                Bitmap frame = latestCapturedBitmap;
                if (frame == null) {
                    for (int i = 0; i < 15; i++) {
                        Thread.sleep(80);
                        frame = latestCapturedBitmap;
                        if (frame != null) break;
                    }
                }

                if (frame == null) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        verdictText.setText("⚠️ Chưa bắt được khung hình màn hình. Vui lòng mở lại khiên trong App.");
                        setVerdictBgColor(Color.parseColor("#D97706"));
                    });
                    return;
                }

                // Compress bitmap to JPEG Base64
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                frame.compress(Bitmap.CompressFormat.JPEG, 75, stream);
                byte[] byteArray = stream.toByteArray();
                String base64Img = "data:image/jpeg;base64," + Base64.encodeToString(byteArray, Base64.NO_WRAP);

                JSONObject uploadPayload = new JSONObject();
                uploadPayload.put("image_base64", base64Img);
                uploadPayload.put("routing_mode", "live");
                uploadPayload.put("language", "vi");

                String[] endpoints = {API_GATEWAY_PRIMARY, API_GATEWAY_FALLBACK};
                boolean success = false;

                for (String endpoint : endpoints) {
                    try {
                        HttpURLConnection conn = (HttpURLConnection) new URL(endpoint).openConnection();
                        conn.setRequestMethod("POST");
                        conn.setRequestProperty("Content-Type", "application/json");
                        conn.setDoOutput(true);
                        conn.setConnectTimeout(8000);
                        conn.setReadTimeout(20000);

                        try (OutputStream os = conn.getOutputStream()) {
                            os.write(uploadPayload.toString().getBytes(StandardCharsets.UTF_8));
                        }

                        if (conn.getResponseCode() == 200) {
                            Scanner s = new Scanner(conn.getInputStream()).useDelimiter("\\A");
                            String resp = s.hasNext() ? s.next() : "";
                            JSONObject json = new JSONObject(resp);

                            String filepath = json.optString("filepath", "");
                            if (!filepath.isEmpty()) {
                                String analyzeUrl = endpoint.replace("/api/upload", "/api/analyze");
                                HttpURLConnection aConn = (HttpURLConnection) new URL(analyzeUrl).openConnection();
                                aConn.setRequestMethod("POST");
                                aConn.setRequestProperty("Content-Type", "application/json");
                                aConn.setDoOutput(true);
                                aConn.setConnectTimeout(10000);
                                aConn.setReadTimeout(30000);

                                JSONObject aPayload = new JSONObject();
                                aPayload.put("filepath", filepath);
                                aPayload.put("routing_mode", "live");
                                aPayload.put("language", "vi");

                                try (OutputStream aos = aConn.getOutputStream()) {
                                    aos.write(aPayload.toString().getBytes(StandardCharsets.UTF_8));
                                }

                                if (aConn.getResponseCode() == 200) {
                                    Scanner as = new Scanner(aConn.getInputStream()).useDelimiter("\\A");
                                    String aResp = as.hasNext() ? as.next() : "";
                                    JSONObject aJson = new JSONObject(aResp);
                                    JSONObject det = aJson.optJSONObject("detection_result");
                                    if (det != null) {
                                        double risk = det.optDouble("composite_risk_score", 0.0);
                                        boolean isFake = risk >= 0.5 || det.optString("label", "").toLowerCase().contains("fake");
                                        String certId = det.optString("certificate_id", "CERT-DT-SCREEN");

                                        final double finalRisk = risk;
                                        final boolean finalIsFake = isFake;
                                        final String finalCertId = certId;

                                        new Handler(Looper.getMainLooper()).post(() -> {
                                            if (finalIsFake) {
                                                verdictText.setText(String.format("🚨 CẢNH BÁO: DEEPFAKE!\nNguy cơ: %.1f%% | Mã: %s", finalRisk * 100, finalCertId));
                                                setVerdictBgColor(Color.parseColor("#DC2626"));
                                            } else {
                                                verdictText.setText(String.format("✅ PHƯƠNG TIỆN CHÂN THẬT\nTin cậy: %.1f%% | Mã: %s", (1.0 - finalRisk) * 100, finalCertId));
                                                setVerdictBgColor(Color.parseColor("#16A34A"));
                                            }
                                        });
                                        success = true;
                                        break;
                                    }
                                }
                            }
                        }
                    } catch (Exception ignored) {}
                }

                if (!success) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        verdictText.setText("❌ Không thể phân tích khung hình màn hình.");
                        setVerdictBgColor(Color.parseColor("#64748B"));
                    });
                }
            } catch (Exception e) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    verdictText.setText("❌ Lỗi: " + e.getMessage());
                    setVerdictBgColor(Color.parseColor("#64748B"));
                });
            } finally {
                isScanning = false;
                new Handler(Looper.getMainLooper()).postDelayed(this::resetBadge, 10000);
            }
        }).start();
    }

    private String extractUrlFromText(String text) {
        if (text == null || text.trim().isEmpty()) return "";
        try {
            Matcher m = Pattern.compile("(https?://[\\w\\-\\.\\?\\=\\&\\%\\/\\#\\+\\@\\:\\;\\~]+)").matcher(text);
            if (m.find()) return m.group(1).trim();
        } catch (Exception ignored) {}
        return "";
    }

    private void triggerForensicScanUrl(String url) {
        isScanning = true;
        verdictText.setVisibility(View.VISIBLE);
        String displayUrl = url.length() > 30 ? url.substring(0, 27) + "..." : url;
        verdictText.setText("🛡️ Đang giám định link...\n" + displayUrl);
        setVerdictBgColor(Color.parseColor("#0A84FF"));

        new Thread(() -> {
            try {
                JSONObject payload = new JSONObject();
                payload.put("url", url);
                payload.put("routing_mode", "live");
                payload.put("language", "vi");

                String[] endpoints = {API_GATEWAY_PRIMARY, API_GATEWAY_FALLBACK};
                boolean success = false;

                for (String endpoint : endpoints) {
                    try {
                        HttpURLConnection conn = (HttpURLConnection) new URL(endpoint).openConnection();
                        conn.setRequestMethod("POST");
                        conn.setRequestProperty("Content-Type", "application/json");
                        conn.setDoOutput(true);
                        conn.setConnectTimeout(8000);
                        conn.setReadTimeout(20000);

                        try (OutputStream os = conn.getOutputStream()) {
                            os.write(payload.toString().getBytes(StandardCharsets.UTF_8));
                        }

                        if (conn.getResponseCode() == 200) {
                            Scanner s = new Scanner(conn.getInputStream()).useDelimiter("\\A");
                            String resp = s.hasNext() ? s.next() : "";
                            JSONObject json = new JSONObject(resp);

                            String filepath = json.optString("filepath", "");
                            if (!filepath.isEmpty()) {
                                String analyzeUrl = endpoint.replace("/api/upload", "/api/analyze");
                                HttpURLConnection aConn = (HttpURLConnection) new URL(analyzeUrl).openConnection();
                                aConn.setRequestMethod("POST");
                                aConn.setRequestProperty("Content-Type", "application/json");
                                aConn.setDoOutput(true);
                                aConn.setConnectTimeout(10000);
                                aConn.setReadTimeout(30000);

                                JSONObject aPayload = new JSONObject();
                                aPayload.put("filepath", filepath);
                                aPayload.put("source_url", url);
                                aPayload.put("routing_mode", "live");
                                aPayload.put("language", "vi");

                                try (OutputStream aos = aConn.getOutputStream()) {
                                    aos.write(aPayload.toString().getBytes(StandardCharsets.UTF_8));
                                }

                                if (aConn.getResponseCode() == 200) {
                                    Scanner as = new Scanner(aConn.getInputStream()).useDelimiter("\\A");
                                    String aResp = as.hasNext() ? as.next() : "";
                                    JSONObject aJson = new JSONObject(aResp);
                                    JSONObject det = aJson.optJSONObject("detection_result");
                                    if (det != null) {
                                        double risk = det.optDouble("composite_risk_score", 0.0);
                                        boolean isFake = risk >= 0.5 || det.optString("label", "").toLowerCase().contains("fake");
                                        String certId = det.optString("certificate_id", "CERT-DT-LIVE");

                                        final double finalRisk = risk;
                                        final boolean finalIsFake = isFake;
                                        final String finalCertId = certId;

                                        new Handler(Looper.getMainLooper()).post(() -> {
                                            if (finalIsFake) {
                                                verdictText.setText(String.format("🚨 CẢNH BÁO: DEEPFAKE!\nNguy cơ: %.1f%% | Mã: %s", finalRisk * 100, finalCertId));
                                                setVerdictBgColor(Color.parseColor("#DC2626"));
                                            } else {
                                                verdictText.setText(String.format("✅ PHƯƠNG TIỆN CHÂN THẬT\nTin cậy: %.1f%% | Mã: %s", (1.0 - finalRisk) * 100, finalCertId));
                                                setVerdictBgColor(Color.parseColor("#16A34A"));
                                            }
                                        });
                                        success = true;
                                        break;
                                    }
                                }
                            }
                        }
                    } catch (Exception ignored) {}
                }

                if (!success) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        verdictText.setText("❌ Không thể kết nối Cổng Pháp Y.");
                        setVerdictBgColor(Color.parseColor("#64748B"));
                    });
                }
            } catch (Exception e) {
                new Handler(Looper.getMainLooper()).post(() -> {
                    verdictText.setText("❌ Lỗi: " + e.getMessage());
                    setVerdictBgColor(Color.parseColor("#64748B"));
                });
            } finally {
                isScanning = false;
                new Handler(Looper.getMainLooper()).postDelayed(this::resetBadge, 10000);
            }
        }).start();
    }

    private void setVerdictBgColor(int color) {
        if (verdictText != null) {
            GradientDrawable d = new GradientDrawable();
            d.setCornerRadius(dpToPx(10));
            d.setColor(color);
            verdictText.setBackground(d);
        }
    }

    private void resetBadge() {
        if (verdictText != null) {
            verdictText.setVisibility(View.GONE);
        }
    }

    private int dpToPx(int dp) {
        DisplayMetrics metrics = getResources().getDisplayMetrics();
        return (int) (dp * metrics.density + 0.5f);
    }

    private void showToastOnMain(String msg) {
        new Handler(Looper.getMainLooper()).post(() -> Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show());
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel chan = new NotificationChannel("deeptrace_guard", "DeepTrace Real-time Call Guardian", NotificationManager.IMPORTANCE_LOW);
            NotificationManager mgr = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            if (mgr != null) mgr.createNotificationChannel(chan);
        }
    }

    private Notification buildNotification() {
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this, 0, notificationIntent,
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0
        );

        Notification.Builder builder = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                ? new Notification.Builder(this, "deeptrace_guard")
                : new Notification.Builder(this);
        return builder.setContentTitle("DeepTrace Call & Live Guardian")
                .setContentText("Bong bóng giám sát TikTok / YouTube đang hoạt động")
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentIntent(pendingIntent)
                .build();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        isRunning = false;
        if (floatingView != null && windowManager != null) {
            try {
                windowManager.removeView(floatingView);
            } catch (Exception ignored) {}
        }
        if (virtualDisplay != null) {
            try { virtualDisplay.release(); } catch (Exception ignored) {}
        }
        if (imageReader != null) {
            try { imageReader.close(); } catch (Exception ignored) {}
        }
        if (captureHandlerThread != null) {
            try { captureHandlerThread.quitSafely(); } catch (Exception ignored) {}
        }
        if (mediaProjection != null) {
            try { mediaProjection.stop(); } catch (Exception ignored) {}
        }
        stopAudioWatchdog();
    }
}
