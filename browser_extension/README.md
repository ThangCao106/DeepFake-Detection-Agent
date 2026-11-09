# 🛡️ Deepfake Forensic Shield - Chrome/Edge Extension (Manifest V3)
> **Client-Server Hybrid In-The-Wild Social Media Deepfake Detection Extension**  
> *Đề tài Luận án Tiến sĩ: Khung Giám Định Pháp Y Đa Phương Thức & Chứng Thư Số EU AI Act*

---

## 📌 1. Giới Thiệu (Overview)

**Deepfake Forensic Shield** là tiện ích mở rộng trình duyệt (Chromium Extension: Google Chrome, Microsoft Edge, Brave, Opera) hoạt động theo mô hình **Client-Server Hybrid**:
- **Phía Client (Trình duyệt)**: Đóng vai trò bộ trích xuất luồng nhẹ (*Headless Edge Media Sniffer*), tự động phát hiện video và hình ảnh trên các mạng xã hội (**TikTok**, **Instagram Reels**, **Threads**, **YouTube Shorts**, **Facebook**, **X/Twitter**).
- **Phía Server (Cổng Pháp Y Số - detection.dtpsoft.vn)**: Nhận khung hình hoặc liên kết gốc, kích hoạt toàn bộ chu trình giám định chuyên sâu gồm:
  - Phân tích gai phổ tần số 2D-FFT và sóng con 2D-DWT Haar Wavelet ($E_{\text{high}}$)
  - Phân tích vân cảm biến quang học máy ảnh PRNU
  - Kiểm định đối xứng phản xạ giác mạc mắt ($\Delta\theta$)
  - Tranh biện đối kháng giữa 3 mô hình nền tảng (Triad Council: Gemini + Muse + Claude)
  - Cấp phát **Chứng thư Giám định Pháp y Số (`CERT-DF-...`)** có thể tra cứu công khai qua Google Search.

---

## 🚀 2. Hướng Dẫn Cài Đặt (Load Unpacked Extension)

1. Mở trình duyệt Chrome hoặc Microsoft Edge.
2. Truy cập vào trang quản lý tiện ích:
   - Chrome: `chrome://extensions/`
   - Edge: `edge://extensions/`
3. Bật chế độ nhà phát triển (**Developer mode**) ở góc trên bên phải.
4. Nhấn nút **Tải tiện ích đã giải nén (Load unpacked)**.
5. Chọn thư mục:
   ```
   D:\DATA\TC\Py_Ex\NCS\DeepFake-Detection-Agent\browser_extension
   ```
6. Tiện ích **Deepfake Forensic Shield** sẽ xuất hiện trên thanh công cụ với biểu tượng chiếc khiên 🛡️.

---

## 💡 3. Cách Sử Dụng Trên Mạng Xã Hội

1. **Cách 1: Nút Bấm Nổi Trực Tiếp (Floating Button)**:
   - Khi duyệt video trên TikTok, Instagram Reels hoặc Threads, di chuột vào khung video.
   - Nút nổi **`🛡️ Scan AI`** sẽ xuất hiện ở góc trên bên phải của video.
   - Bấm vào nút để kích hoạt giám định ngay lập tức.
2. **Cách 2: Menu Chuột Phải (Context Menu)**:
   - Click chuột phải vào bất kỳ video hoặc hình ảnh nào trên trang web.
   - Chọn **"🛡️ Giám định Deepfake Phương Tiện Này..."**.
3. **Xem Kết Quả & Chứng Thư Số**:
   - Thẻ kết quả pháp y số sẽ hiển thị trực tiếp đè lên khung video:
     - Nhãn phán quyết: `🚨 DEEPFAKE DETECTED` (Đỏ) hoặc `✅ AUTHENTIC MEDIA` (Xanh)
     - Chỉ số rủi ro tổng hợp $Risk_{\text{composite}}$ và Độ tin cậy (%)
     - Mã chứng chỉ số `CERT-DF-XXXXXXXX`
   - Bấm **"📜 Tra Cứu Chứng Thư Số ↗"** để mở trang thẩm định công khai trên cổng `detection.dtpsoft.vn`.

---

## ⚙️ 4. Cấu Hình Cổng Kết Nối Gateway

Mở popup tiện ích từ thanh công cụ để chọn cổng kết nối:
- **Cloud Gateway (Mặc định)**: `https://detection.dtpsoft.vn` (Phục vụ kiểm thử trực tiếp trên môi trường production)
- **Local Development**: `http://localhost:5000` (Phục vụ dev và kiểm thử nội bộ trên máy cá nhân)
