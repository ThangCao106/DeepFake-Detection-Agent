package vn.io.deeptrace.guard;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
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
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * DeepTrace Floating Overlay Service (Call & Live Stream Guardian for Android)
 * Monitors active Zalo / FaceTime / Messenger / TikTok Live sessions via non-invasive on-demand capture.
 */
public class DeepTraceOverlayService extends Service {
    private WindowManager windowManager;
    private View floatingView;
    private ImageView bubbleIcon;
    private TextView verdictText;

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;
    private int screenWidth, screenHeight, screenDensity;

    private static final String API_ENDPOINT = "https://deeptrace.io.vn/api/analyze";
    private boolean isScanning = false;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        startForeground(1001, buildNotification());

        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        DisplayMetrics metrics = getResources().getDisplayMetrics();
        screenWidth = metrics.widthPixels;
        screenHeight = metrics.heightPixels;
        screenDensity = metrics.densityDpi;

        initFloatingBubble();
    }

    private void initFloatingBubble() {
        int layoutFlag = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                : WindowManager.LayoutParams.TYPE_PHONE;

        final WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                layoutFlag,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT
        );

        params.gravity = Gravity.TOP | Gravity.START;
        params.x = 20;
        params.y = 250;

        floatingView = LayoutInflater.from(this).inflate(R.layout.overlay_bubble_layout, null);
        bubbleIcon = floatingView.findViewById(R.id.bubble_icon);
        verdictText = floatingView.findViewById(R.id.verdict_badge);

        // Touch listener for dragging and single-click scanning
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
                        windowManager.updateViewLayout(floatingView, params);
                        return true;

                    case MotionEvent.ACTION_UP:
                        if (System.currentTimeMillis() - touchStartTime < 250) {
                            // User clicked the bubble: Trigger real-time forensic inspection
                            triggerCallForensicScan();
                        }
                        return true;
                }
                return false;
            }
        });

        windowManager.addView(floatingView, params);
    }

    /**
     * Captures a single frame from the live video call (Zalo / FaceTime / Stream) and calls DeepTrace Engine
     */
    public void triggerCallForensicScan() {
        if (isScanning) return;
        isScanning = true;

        verdictText.setVisibility(View.VISIBLE);
        verdictText.setText("🛡️ Đang giám định cuộc gọi...");
        verdictText.setBackgroundColor(0xFF0A84FF);

        new Thread(() -> {
            try {
                Bitmap frame = captureCurrentFrame();
                if (frame == null) {
                    showToastOnMain("Không thể chụp luồng video cuộc gọi.");
                    resetBadge();
                    return;
                }

                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                frame.compress(Bitmap.CompressFormat.JPEG, 85, stream);
                String base64Image = "data:image/jpeg;base64," + Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP);

                // Call DeepTrace API
                JSONObject payload = new JSONObject();
                payload.put("type", "base64");
                payload.put("imageBase64", base64Image);
                payload.put("routing_mode", "call");
                payload.put("language", "vi");

                HttpURLConnection conn = (HttpURLConnection) new URL(API_ENDPOINT).openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setDoOutput(true);
                conn.setConnectTimeout(8000);
                conn.setReadTimeout(12000);

                try (OutputStream os = conn.getOutputStream()) {
                    os.write(payload.toString().getBytes(StandardCharsets.UTF_8));
                }

                if (conn.getResponseCode() == 200) {
                    java.util.Scanner s = new java.util.Scanner(conn.getInputStream()).useDelimiter("\\A");
                    String respStr = s.hasNext() ? s.next() : "";
                    JSONObject res = new JSONObject(respStr);
                    JSONObject det = res.optJSONObject("detection_result");
                    if (det != null) {
                        double risk = det.optDouble("composite_risk_score", 0.0);
                        boolean isFake = risk >= 0.7318;
                        String certId = det.optString("certificate_id", "CERT-DT-CALL");

                        new Handler(Looper.getMainLooper()).post(() -> {
                            if (isFake) {
                                verdictText.setText(String.format("🚨 CẢNH BÁO GIẢ MẠO! (Rủi ro: %.1f%%)\nMã: %s", risk * 100, certId));
                                verdictText.setBackgroundColor(0xFFDC2626);
                            } else {
                                verdictText.setText(String.format("✅ CUỘC GỌI AN TOÀN (Tin cậy: %.1f%%)\nMã: %s", (1.0 - risk) * 100, certId));
                                verdictText.setBackgroundColor(0xFF16A34A);
                            }
                        });
                    }
                } else {
                    showToastOnMain("Lỗi máy chủ giám định (" + conn.getResponseCode() + ")");
                }
            } catch (Exception e) {
                showToastOnMain("Lỗi: " + e.getMessage());
            } finally {
                isScanning = false;
                new Handler(Looper.getMainLooper()).postDelayed(this::resetBadge, 8000);
            }
        }).start();
    }

    private Bitmap captureCurrentFrame() {
        if (imageReader == null) return null;
        Image image = imageReader.acquireLatestImage();
        if (image == null) return null;

        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * screenWidth;

        Bitmap bitmap = Bitmap.createBitmap(screenWidth + rowPadding / pixelStride, screenHeight, Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        image.close();
        return Bitmap.createBitmap(bitmap, 0, 0, screenWidth, screenHeight);
    }

    private void resetBadge() {
        if (verdictText != null) {
            verdictText.setVisibility(View.GONE);
        }
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
        Notification.Builder builder = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                ? new Notification.Builder(this, "deeptrace_guard")
                : new Notification.Builder(this);
        return builder.setContentTitle("DeepTrace Call & Live Guardian")
                .setContentText("Bong bóng giám sát Deepfake Zalo/FaceTime đang hoạt động")
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .build();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (floatingView != null) windowManager.removeView(floatingView);
        if (virtualDisplay != null) virtualDisplay.release();
        if (imageReader != null) imageReader.close();
        if (mediaProjection != null) mediaProjection.stop();
    }
}
