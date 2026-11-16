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
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

public class DeepTraceOverlayService extends Service {
    private WindowManager windowManager;
    private View floatingView;
    private TextView bubbleIcon;
    private TextView verdictText;

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
        createNotificationChannel();
        startForeground(1001, buildNotification());

        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
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
        params.y = 350;

        floatingView = LayoutInflater.from(this).inflate(R.layout.overlay_bubble_layout, null);
        bubbleIcon = floatingView.findViewById(R.id.bubble_icon);
        verdictText = floatingView.findViewById(R.id.verdict_badge);

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
                            // User tapped the floating shield
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
        if (isScanning) return;

        // Check clipboard for copied TikTok / YouTube / video link
        String targetUrl = "";
        try {
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null && clipboard.hasPrimaryClip() && clipboard.getPrimaryClip().getItemCount() > 0) {
                ClipData.Item item = clipboard.getPrimaryClip().getItemAt(0);
                CharSequence text = item != null ? item.getText() : null;
                if (text != null) {
                    String clipStr = text.toString().trim();
                    if (clipStr.startsWith("http://") || clipStr.startsWith("https://")) {
                        targetUrl = clipStr;
                    }
                }
            }
        } catch (Exception ignored) {}

        if (!targetUrl.isEmpty()) {
            triggerForensicScanUrl(targetUrl);
        } else {
            showToastOnMain("🛡️ DeepTrace: Nhấn Chia sẻ -> Sao chép liên kết trên TikTok, rồi chạm lại vào Khiên để quét!");
            try {
                Intent appIntent = new Intent(this, MainActivity.class);
                appIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
                startActivity(appIntent);
            } catch (Exception ignored) {}
        }
    }

    private void triggerForensicScanUrl(String url) {
        isScanning = true;
        verdictText.setVisibility(View.VISIBLE);
        verdictText.setText("🛡️ DeepTrace: Đang kiểm định luồng...");
        verdictText.setBackgroundColor(Color.parseColor("#0A84FF"));

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
                                                verdictText.setText(String.format("🚨 DEEPFAKE! (Rủi ro: %.1f%%)\nMã: %s", finalRisk * 100, finalCertId));
                                                verdictText.setBackgroundColor(0xFFDC2626);
                                            } else {
                                                verdictText.setText(String.format("✅ CHÂN THẬT (Tin cậy: %.1f%%)\nMã: %s", (1.0 - finalRisk) * 100, finalCertId));
                                                verdictText.setBackgroundColor(0xFF16A34A);
                                            }
                                        });
                                        success = true;
                                        break;
                                    }
                                }
                            }
                        }
                    } catch (Exception ignored) {
                    }
                }

                if (!success) {
                    showToastOnMain("Không thể giám định liên kết này qua Cổng Pháp Y.");
                    new Handler(Looper.getMainLooper()).post(this::resetBadge);
                }
            } catch (Exception e) {
                showToastOnMain("Lỗi: " + e.getMessage());
                new Handler(Looper.getMainLooper()).post(this::resetBadge);
            } finally {
                isScanning = false;
                new Handler(Looper.getMainLooper()).postDelayed(this::resetBadge, 9000);
            }
        }).start();
    }

    private void resetBadge() {
        if (verdictText != null) {
            verdictText.setVisibility(View.GONE);
        }
    }

    private void showToastOnMain(String msg) {
        new Handler(Looper.getMainLooper()).post(() -> Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_LONG).show());
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
        if (floatingView != null && windowManager != null) {
            try {
                windowManager.removeView(floatingView);
            } catch (Exception ignored) {}
        }
    }
}
