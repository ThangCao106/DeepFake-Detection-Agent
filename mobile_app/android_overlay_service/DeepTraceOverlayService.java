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
import android.os.IBinder;
import android.os.Looper;
import android.util.Base64;
import android.util.DisplayMetrics;
import android.util.TypedValue;
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

    private WindowManager windowManager;
    private View floatingView;
    private TextView bubbleIcon;
    private TextView verdictText;

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;
    private int screenWidth, screenHeight, screenDensity;

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

        DisplayMetrics metrics = getResources().getDisplayMetrics();
        screenWidth = metrics.widthPixels;
        screenHeight = metrics.heightPixels;
        screenDensity = metrics.densityDpi;

        // 1. Safe Foreground Notification (MediaProjection compatible)
        try {
            createNotificationChannel();
            Notification notification = buildNotification();
            if (Build.VERSION.SDK_INT >= 34) {
                startForeground(1001, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(1001, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
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
            int resultCode = intent.getIntExtra("RESULT_CODE", 0);
            Intent data = intent.getParcelableExtra("DATA_INTENT");
            if (resultCode != 0 && data != null) {
                setupMediaProjection(resultCode, data);
            }
        }
        return START_STICKY;
    }

    private void setupMediaProjection(int resultCode, Intent data) {
        try {
            MediaProjectionManager mgr = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
            if (mgr != null) {
                mediaProjection = mgr.getMediaProjection(resultCode, data);
                if (mediaProjection != null) {
                    imageReader = ImageReader.newInstance(screenWidth, screenHeight, PixelFormat.RGBA_8888, 2);
                    virtualDisplay = mediaProjection.createVirtualDisplay(
                            "DeepTraceScreenCapture",
                            screenWidth,
                            screenHeight,
                            screenDensity,
                            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                            imageReader.getSurface(),
                            null,
                            null
                    );
                    android.util.Log.i("DeepTrace", "Screen Capture Engine ready for 1-Tap inspection!");
                    showToastOnMain("🛡️ Sẵn sàng quét trực tiếp màn hình TikTok!");
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

        // Mode 1 (Primary): Direct Screen Frame Capture (No link needed!)
        if (imageReader != null) {
            triggerScreenForensicScan();
            return;
        }

        // Mode 2 (Fallback): URL extraction from clipboard
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
            verdictText.setText("💡 Mẹo: Mở App DeepTrace và bật lại công tắc Khiên nổi để kích hoạt chế độ Chụp Màn Hình Trực Tiếp!");
            setVerdictBgColor(Color.parseColor("#D97706"));
            new Handler(Looper.getMainLooper()).postDelayed(this::resetBadge, 6000);
        }
    }

    private void triggerScreenForensicScan() {
        isScanning = true;
        verdictText.setVisibility(View.VISIBLE);
        verdictText.setText("📸 Đang chụp & quét khuôn mặt trên màn hình...");
        setVerdictBgColor(Color.parseColor("#0A84FF"));

        new Thread(() -> {
            try {
                Bitmap frame = captureCurrentFrame();
                if (frame == null) {
                    Thread.sleep(200);
                    frame = captureCurrentFrame();
                }

                if (frame == null) {
                    new Handler(Looper.getMainLooper()).post(() -> {
                        verdictText.setText("⚠️ Không thể chụp khung hình màn hình.");
                        setVerdictBgColor(Color.parseColor("#D97706"));
                    });
                    return;
                }

                // Compress bitmap to JPEG Base64
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                frame.compress(Bitmap.CompressFormat.JPEG, 80, stream);
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

    private Bitmap captureCurrentFrame() {
        if (imageReader == null) return null;
        Image image = null;
        try {
            image = imageReader.acquireLatestImage();
            if (image == null) return null;

            Image.Plane[] planes = image.getPlanes();
            ByteBuffer buffer = planes[0].getBuffer();
            int pixelStride = planes[0].getPixelStride();
            int rowStride = planes[0].getRowStride();
            int rowPadding = rowStride - pixelStride * screenWidth;

            Bitmap bitmap = Bitmap.createBitmap(screenWidth + rowPadding / pixelStride, screenHeight, Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(buffer);
            return Bitmap.createBitmap(bitmap, 0, 0, screenWidth, screenHeight);
        } catch (Exception e) {
            android.util.Log.e("DeepTrace", "Error capturing frame: " + e.getMessage());
            return null;
        } finally {
            if (image != null) {
                image.close();
            }
        }
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
            virtualDisplay.release();
        }
        if (imageReader != null) {
            imageReader.close();
        }
        if (mediaProjection != null) {
            mediaProjection.stop();
        }
    }
}
