# 📥 Dropzone: Thư Mục Kéo Thả Mẫu Fake Đối Kháng (Consumer AI)

Thư mục này phục vụ cho việc tiếp nhận các mẫu Deepfake / Generative AI độc lạ được tạo trực tiếp từ giao diện web của các công cụ hàng đầu:
- **Canva Magic Media**
- **Meta AI Imagine**
- **Google Gemini (Imagen 3)**
- **ChatGPT Plus (DALL-E 3 / GPT-4o)**
- **Midjourney v6 / FLUX.1 Pro**

---

### 📌 Hướng dẫn sử dụng:
1. Khi bạn tạo được các hình ảnh fake đối kháng cao cấp (hoán đổi khuôn mặt người Việt, ánh sáng phức tạp, đeo kính, khẩu trang, v.v.), hãy lưu tệp dưới định dạng `.jpg` hoặc `.png`.
2. Kéo-thả trực tiếp tệp ảnh vào thư mục này:
   `DeepFake-Detection-Agent/crawler/dropzone_consumer_ai/`
3. Pipeline tự động [crawler/daily_pentest_pipeline.py](file:///d:/DATA/TC/Py_Ex/NCS/DeepFake-Detection-Agent/crawler/daily_pentest_pipeline.py) khi chạy cuối ngày sẽ:
   - Ưu tiên nạp ngay các mẫu này vào nhóm **`03_manipulated_openweight/`**.
   - Đánh giá khả năng phòng thủ của hệ thống detector trong **Job Pentest EOD**.
   - Đóng gói cùng $1,000$ ảnh, $100$ video, $100$ audio thật lên **Hugging Face Datasets Hub** (`ThangCao106/Threads-DFBench-Vietnamese`).
   - Đẩy sang **Google Colab** để tinh chỉnh thích ứng mô hình ViT (LoRA Continual Fine-Tuning).
