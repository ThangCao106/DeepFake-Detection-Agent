# DeepTrace Mobile Add-in: Real-time Live Stream & Call Guardian (Android & iOS)

Cung cấp giải pháp giám định Deepfake thời gian thực cho **Cuộc gọi Video (FaceTime, Zalo, Messenger)** và **Luồng Live Stream (TikTok Live, Facebook Live, YouTube Live)** trên thiết bị di động.

---

## 1. Cấu trúc thư mục mã nguồn Mobile

```text
mobile_app/
├── android_overlay_service/               # Giải pháp Android Native (Bong bóng nổi)
│   ├── DeepTraceOverlayService.java      # Service chạy nền vẽ Bong bóng nổi + MediaProjection chụp frame
│   └── overlay_bubble_layout.xml         # Giao diện Floating Widget hiển thị đè lên Zalo / Messenger
├── ios_share_extension/                   # Giải pháp iOS Action & Share Extension
│   └── DeepTraceShareViewController.swift # Nhận luồng Live/Video từ nút Share của TikTok, YT, Zalo
└── flutter_app/                           # Ứng dụng Hybrid đa nền tảng Flutter (Android & iOS)
    └── lib/main.dart                     # Giao diện giám định Live Stream & Call Guardian thời gian thực
```

---

## 2. Cơ chế hoạt động trên Android (Bong bóng nổi - Floating Bubble)

1. **Vấn đề trên Android**: App Zalo, Messenger chạy trong môi trường bảo vệ riêng, app khác không thể đọc trực tiếp camera/micro của Zalo.
2. **Giải pháp DeepTrace**:
   - `DeepTraceOverlayService.java` sử dụng quyền `SYSTEM_ALERT_WINDOW` (Hiển thị trên các ứng dụng khác) và `MediaProjectionManager`.
   - Một bong bóng biểu tượng khiên `🛡️ DeepTrace` luôn nổi ở góc màn hình trong suốt cuộc gọi.
   - Khi có dấu hiệu nghi ngờ lừa đảo, người dùng **chạm vào bong bóng**.
   - Service tự động chụp 1-3 khung hình cuộc gọi, mã hóa base64 và gửi đến `https://deeptrace.io.vn/api/analyze` với `routing_mode: "call"`.
   - Ngay lập tức (trong 1.5s), banner màu đỏ `🚨 CẢNH BÁO GIẢ MẠO` hoặc xanh `✅ CUỘC GỌI AN TOÀN` hiển thị trực tiếp trên màn hình cuộc gọi kèm mã chứng thư `CERT-DT-CALL-XXXXXXXX`.

---

## 3. Cơ chế hoạt động trên iOS (Action & Share Extension)

1. **Vấn đề trên iOS**: Apple Sandbox cách ly hoàn toàn các ứng dụng, không cho phép vẽ bong bóng nổi can thiệp vào cuộc gọi FaceTime / Zalo.
2. **Giải pháp DeepTrace**:
   - Sử dụng **iOS Share Extension** (`DeepTraceShareViewController.swift`):
     - Khi xem **TikTok Live / YouTube Shorts / Facebook Live**: Bấm **Share ➔ DeepTrace**.
     - Khi đang trong **Cuộc gọi FaceTime / Zalo nghi vấn**: Chụp ảnh màn hình (Screenshot) ➔ Bấm **Share ➔ DeepTrace**.
   - Extension tự động gửi dữ liệu về Cổng pháp y `https://deeptrace.io.vn` và trả về kết quả giám định pháp y cùng chứng thư số số hóa ngay trên màn hình iOS.

---

## 4. Ứng dụng độc lập Flutter (`flutter_app`)

- Viết bằng Dart/Flutter, tương thích cả Android và iOS.
- Cho phép dán trực tiếp đường link phòng Live (TikTok, YouTube, Facebook) hoặc Call Web Link để hệ thống tự động bóc tách luồng streaming và hiển thị kết quả phân tích vi mô.

