import os
import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ==============================================================================
# COLOR PALETTE (Modern Academic Dark Theme: Navy / Slate / Cyan / Emerald)
# ==============================================================================
BG_DARK = RGBColor(15, 23, 42)        # #0F172A (Deep Slate Navy)
CARD_BG = RGBColor(30, 41, 59)        # #1E293B (Card Dark Surface)
CARD_BORDER = RGBColor(51, 65, 85)    # #334155 (Subtle Border)
TEXT_WHITE = RGBColor(248, 250, 252)  # #F8FAFC (Pure White/Slate 50)
TEXT_MUTED = RGBColor(148, 163, 184)  # #94A3B8 (Slate 400)
ACCENT_CYAN = RGBColor(6, 182, 212)   # #06B6D4 (Cyan Brand Accent)
ACCENT_GREEN = RGBColor(16, 185, 129) # #10B981 (Emerald Success Accent)
ACCENT_AMBER = RGBColor(245, 158, 11) # #F59E0B (Amber Alert Accent)
ACCENT_RED = RGBColor(239, 68, 68)    # #EF4444 (Red Warning Accent)
ACCENT_PURPLE = RGBColor(168, 85, 247)# #A855F7 (Purple Multi-Modal Accent)

def set_slide_background(slide):
    """Sets a rich dark background on the slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_DARK
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_badge="LUẬN ÁN TIẾN SĨ | AI FORENSICS"):
    """Adds a standardized academic header to a slide."""
    # Category badge
    badge_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
    tf_badge = badge_box.text_frame
    tf_badge.word_wrap = True
    tf_badge.margin_left = tf_badge.margin_top = tf_badge.margin_right = tf_badge.margin_bottom = 0
    p_b = tf_badge.paragraphs[0]
    p_b.text = category_badge.upper()
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = ACCENT_CYAN

    # Title text
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_t = tf_title.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_WHITE

def add_card(slide, left, top, width, height, title=None, title_color=ACCENT_CYAN, fill_color=CARD_BG, border_color=CARD_BORDER):
    """Creates a modern rounded card container."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    
    if title:
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = title_color
        
    return shape

def add_bullet_point(text_frame, title, desc, bullet_color=ACCENT_CYAN, font_size=12, space_after=10):
    """Appends a styled bullet point with bold lead title."""
    p = text_frame.add_paragraph() if len(text_frame.paragraphs[0].text) > 0 else text_frame.paragraphs[0]
    p.space_after = Pt(space_after)
    
    # Bullet title
    run_t = p.add_run()
    run_t.text = f"• {title}: "
    run_t.font.bold = True
    run_t.font.size = Pt(font_size)
    run_t.font.color.rgb = bullet_color
    
    # Bullet description
    run_d = p.add_run()
    run_d.text = desc
    run_d.font.bold = False
    run_d.font.size = Pt(font_size)
    run_d.font.color.rgb = TEXT_WHITE

# ==============================================================================
# SLIDE BUILDERS
# ==============================================================================

def build_presentation(output_path: Path):
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # --------------------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE (Trang bìa)
    # --------------------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)
    
    # Hero Title Box
    tb_title = slide1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(4.5))
    tf1 = tb_title.text_frame
    tf1.word_wrap = True
    
    p_tag = tf1.paragraphs[0]
    p_tag.text = "🛡️ ĐỀ TÀI LUẬN ÁN TIẾN SĨ (PhD THESIS DEFENSE) • 2026"
    p_tag.font.size = Pt(14)
    p_tag.font.bold = True
    p_tag.font.color.rgb = ACCENT_CYAN
    p_tag.space_after = Pt(16)
    
    p_main = tf1.add_paragraph()
    p_main.text = "NGHIÊN CỨU PHÁT TRIỂN HỆ THỐNG ĐA TÁC THỂ\nPHÁT HIỆN GIẢ MẠO ĐA PHƯƠNG THỨC\nDỰA TRÊN FINE-TUNING TRANSFORMER VÀ TEMPERATURE SCALING"
    p_main.font.size = Pt(28)
    p_main.font.bold = True
    p_main.font.color.rgb = TEXT_WHITE
    p_main.space_after = Pt(18)
    
    p_sub = tf1.add_paragraph()
    p_sub.text = "Giám định Deepfake Image, Video (7s Micro-Window) & Audio | Tích hợp 14 Benchmark Quốc tế & Chuẩn EU AI Act Điều 50"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = ACCENT_GREEN
    
    # Metadata footer card
    add_card(slide1, Inches(1.0), Inches(5.6), Inches(11.333), Inches(1.2), fill_color=CARD_BG, border_color=CARD_BORDER)
    tb_meta = slide1.shapes.add_textbox(Inches(1.3), Inches(5.75), Inches(10.7), Inches(0.9))
    tf_m = tb_meta.text_frame
    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "Chuyên ngành: Khoa học Máy tính / Hệ thống Thông tin (Mã số: 9480101)  |  Thời gian thực hiện: 2026 - 2029"
    p_m1.font.size = Pt(13)
    p_m1.font.bold = True
    p_m1.font.color.rgb = TEXT_WHITE
    p_m1.space_after = Pt(4)
    
    p_m2 = tf_m.add_paragraph()
    p_m2.text = "Nghiên cứu sinh: [Họ và tên NCS]   •   Cán bộ Hướng dẫn Khoa học: [GS / PGS Hướng dẫn]   •   Cơ sở: [Trường Đại học / Viện]"
    p_m2.font.size = Pt(12)
    p_m2.font.color.rgb = TEXT_MUTED

    # --------------------------------------------------------------------------
    # SLIDE 2: BỐI CẢNH & TÍNH CẤP THIẾT
    # --------------------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Bối Cảnh Nghiên Cứu & 3 Thách Thức Cốt Lõi (2025 - 2026)")
    
    col_w = Inches(3.65)
    gap = Inches(0.38)
    y_top = Inches(1.8)
    h_box = Inches(5.0)
    
    # Card 1: Bùng nổ Multi-Modal GenAI
    add_card(slide2, Inches(0.8), y_top, col_w, h_box, "1. Bùng Nổ GenAI Đa Thức", ACCENT_CYAN)
    tb = slide2.shapes.add_textbox(Inches(1.0), y_top + Inches(0.7), col_w - Inches(0.4), h_box - Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    add_bullet_point(tf, "Chuyển dịch Thế hệ", "Từ GANs truyền thống sang Latent Diffusion, Flow Matching & Diffusion Transformers (DiT).", ACCENT_CYAN)
    add_bullet_point(tf, "Hình ảnh (Image)", "Stable Diffusion 3, Flux.1, Midjourney v6 phá vỡ dấu vết ghép ảnh thông thường.", ACCENT_CYAN)
    add_bullet_point(tf, "Video & Avatar", "Sora, Runway Gen-3, 3DGS Head Avatars sinh video siêu thực theo thời gian thực.", ACCENT_CYAN)
    add_bullet_point(tf, "Giọng nói (Audio)", "Zero-shot Voice Cloning và Talking-head Deepfake phục vụ lừa đảo tài chính tinh vi.", ACCENT_CYAN)

    # Card 2: 3 Hạn chế của công cụ cũ
    add_card(slide2, Inches(0.8) + col_w + gap, y_top, col_w, h_box, "2. Ba Điểm Nghẽn Kỹ Thuật", ACCENT_AMBER)
    tb = slide2.shapes.add_textbox(Inches(1.0) + col_w + gap, y_top + Inches(0.7), col_w - Inches(0.4), h_box - Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    add_bullet_point(tf, "Suy giảm OOD", "Độ chính xác sụt giảm 30-40% khi gặp kiến trúc sinh mới ngoài tập huấn luyện.", ACCENT_AMBER)
    add_bullet_point(tf, "Overconfidence", "Mạng nơ-ron sâu tự tin thái quá (>99%) ngay cả với các mẫu biên giới bị phân loại sai.", ACCENT_AMBER)
    add_bullet_point(tf, "Nén Lossy Mạng XH", "Bộ lọc H.264/JPEG của Threads/TikTok xóa mờ các đặc trưng pixel nông.", ACCENT_AMBER)
    add_bullet_point(tf, "Thiếu Căn cứ Pháp lý", "Kết quả đầu ra chỉ là 1 số thực đơn lẻ [0, 1], không có giá trị bằng chứng số pháp lý.", ACCENT_AMBER)

    # Card 3: Yêu cầu Pháp lý Quốc tế
    add_card(slide2, Inches(0.8) + (col_w + gap)*2, y_top, col_w, h_box, "3. Yêu Cầu Pháp Lý Quốc Tế", ACCENT_GREEN)
    tb = slide2.shapes.add_textbox(Inches(1.0) + (col_w + gap)*2, y_top + Inches(0.7), col_w - Inches(0.4), h_box - Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    add_bullet_point(tf, "EU AI Act Điều 50", "Quy định bắt buộc gán nhãn và giám định có thể kiểm toán (auditable) đối với nội dung AI.", ACCENT_GREEN)
    add_bullet_point(tf, "Đạo luật DSA", "Trách nhiệm giải trình của các nền tảng mạng xã hội trong việc ngăn chặn lừa đảo danh tính.", ACCENT_GREEN)
    add_bullet_point(tf, "Chứng thư Số", "Bắt buộc có cấu trúc JSON minh bạch với dấu vết bằng chứng (hash SHA-256, phổ, sinh trắc).", ACCENT_GREEN)
    add_bullet_point(tf, "Mục tiêu Đề tài", "Hợp nhất Toán học, Mô hình Song luồng và Hệ tác thể giải trình song ngữ Anh - Việt.", ACCENT_GREEN)

    # --------------------------------------------------------------------------
    # SLIDE 3: KIẾN TRÚC MÔ HÌNH TỔNG THỂ (DUAL-STREAM ARCHITECTURE)
    # --------------------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "Kiến Trúc Mô Hình Song Luồng Đa Phương Thức (Dual-Stream Architecture)")
    
    # Gate Card (Top banner)
    add_card(slide3, Inches(0.8), Inches(1.8), Inches(11.733), Inches(1.1), "Cổng Phân Loại Sinh Trắc Học Tự Động (Pre-flight Human Presence Gate)", ACCENT_CYAN)
    tb_gate = slide3.shapes.add_textbox(Inches(1.0), Inches(2.25), Inches(11.3), Inches(0.55))
    tf_g = tb_gate.text_frame
    tf_g.word_wrap = True
    p_g = tf_g.paragraphs[0]
    p_g.text = "• HOG Face Detector (Quét khuôn mặt) + YCbCr Skin-tone Cluster (Cb ∈ [77, 127], Cr ∈ [133, 173]). Nếu KHÔNG chứa sinh trắc người → Chạy Single-Stream (tiết kiệm 60% compute). Nếu PHÁT HIỆN người → Kích hoạt Song Luồng (Dual-Stream)."
    p_g.font.size = Pt(12)
    p_g.font.color.rgb = TEXT_WHITE

    # Stream 1 & Stream 2 (Side by Side)
    card_w2 = Inches(5.7)
    card_h2 = Inches(3.9)
    y_streams = Inches(3.1)
    
    # Stream 1
    add_card(slide3, Inches(0.8), y_streams, card_w2, card_h2, "Luồng 1: General Vision & Signal Transformer", ACCENT_CYAN)
    tb_s1 = slide3.shapes.add_textbox(Inches(1.0), y_streams + Inches(0.65), card_w2 - Inches(0.4), card_h2 - Inches(0.8))
    tf_s1 = tb_s1.text_frame
    tf_s1.word_wrap = True
    add_bullet_point(tf_s1, "Backbone", "Vision Transformer (ViT-Base patch 16x16) trích xuất đặc trưng ngữ nghĩa thị giác sâu.", ACCENT_CYAN)
    add_bullet_point(tf_s1, "2D Fast Fourier (FFT)", "Phân tích phổ tần số góc Azimuthal (Z_peak) bắt gai quang phổ lưới sinh AI.", ACCENT_CYAN)
    add_bullet_point(tf_s1, "2D Haar DWT", "Đo năng lượng dải tần cao (E_high) phát hiện làm mịn kết cấu nhân tạo.", ACCENT_CYAN)
    add_bullet_point(tf_s1, "Spatial Variance", "Đo sai phân phương sai nhiễu giữa vùng khuôn mặt và phông nền ngoại vi.", ACCENT_CYAN)

    # Stream 2
    add_card(slide3, Inches(6.833), y_streams, card_w2, card_h2, "Luồng 2: Human Biometric & Physiology (Sinh Trắc Học)", ACCENT_PURPLE)
    tb_s2 = slide3.shapes.add_textbox(Inches(7.033), y_streams + Inches(0.65), card_w2 - Inches(0.4), card_h2 - Inches(0.8))
    tf_s2 = tb_s2.text_frame
    tf_s2.word_wrap = True
    add_bullet_point(tf_s2, "rPPG Vascular Pulse", "Phân tích vi biến thiên màu da mao mạch theo chu kỳ tim (Heartbeat inconsistency).", ACCENT_PURPLE)
    add_bullet_point(tf_s2, "DF26 Lip-Sync", "Mạng đối chiếu không-thời gian bắt lệch pha giữa khẩu hình môi và ngữ âm.", ACCENT_PURPLE)
    add_bullet_point(tf_s2, "WavLM L12 SSL", "Vector đặc trưng giọng nói 768 chiều từ Layer 12 phát hiện cloning thanh quản.", ACCENT_PURPLE)
    add_bullet_point(tf_s2, "Tổng Hợp Thích Ứng", "Risk_composite = 0.40 * Risk_General + 0.60 * Risk_Human (Hiệu chuẩn Platt T* & tau*).", ACCENT_PURPLE)

    # --------------------------------------------------------------------------
    # SLIDE 4: PHÂN TÍCH ƯU & NHƯỢC ĐIỂM MÔ HÌNH
    # --------------------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "Đánh Giá Ưu Điểm & Nhược Điểm của Mô Hình (Model Analysis)")
    
    col_w4 = Inches(5.7)
    h_box4 = Inches(5.1)
    y_pos4 = Inches(1.8)

    # Ưu điểm
    add_card(slide4, Inches(0.8), y_pos4, col_w4, h_box4, "✅ Ưu Điểm Nổi Bật (Key Strengths)", ACCENT_GREEN)
    tb_pro = slide4.shapes.add_textbox(Inches(1.0), y_pos4 + Inches(0.65), col_w4 - Inches(0.4), h_box4 - Inches(0.8))
    tf_pro = tb_pro.text_frame
    tf_pro.word_wrap = True
    add_bullet_point(tf_pro, "Loại trừ Thiên Lệch (Zero Bias)", "Không bị nhầm lẫn giữa ảnh/video người và tranh ảnh nghệ thuật AI phi-sinh trắc học.", ACCENT_GREEN)
    add_bullet_point(tf_pro, "Bắt Được Artifacts Thế Hệ Mới", "Miền tần số 2D FFT & DWT bất biến với các thuật toán sinh ảnh mới như Flux.1 / SD3.", ACCENT_GREEN)
    add_bullet_point(tf_pro, "Tiết Kiệm 60% Tài Nguyên", "Cơ chế Pre-flight Gate ngắt luồng Biometric khi xử lý ảnh phong cảnh, vật thể.", ACCENT_GREEN)
    add_bullet_point(tf_pro, "Khả Năng Giải Trình Cao", "Xuất trực tiếp bản đồ nhiệt 8x8 Patch Heatmap khoanh vùng can thiệp viền tráo mặt.", ACCENT_GREEN)
    add_bullet_point(tf_pro, "Đạt Chuẩn Pháp Lý EU AI Act", "Không phải là black-box scalar; cung cấp chứng thư số minh bạch đa chiều.", ACCENT_GREEN)

    # Nhược điểm
    add_card(slide4, Inches(6.833), y_pos4, col_w4, h_box4, "❌ Hạn Chế & Thách Thức (Limitations & Bottlenecks)", ACCENT_RED)
    tb_con = slide4.shapes.add_textbox(Inches(7.033), y_pos4 + Inches(0.65), col_w4 - Inches(0.4), h_box4 - Inches(0.8))
    tf_con = tb_con.text_frame
    tf_con.word_wrap = True
    add_bullet_point(tf_con, "Độ Phức Tạp Kiến Trúc", "Đòi hỏi đồng bộ 2 luồng song song, phức tạp hơn các mạng CNN/ViT đơn lẻ truyền thống.", ACCENT_RED)
    add_bullet_point(tf_con, "Yêu Cầu VRAM GPU", "Cần tối thiểu 6GB - 8GB VRAM để chạy đồng thời cả hai mô hình khi batch size lớn.", ACCENT_RED)
    add_bullet_point(tf_con, "Nhạy Cảm Với Nén Cực Độ", "Video bị nén quá thấp (bitrate < 200kbps trên Zalo) có thể làm giảm độ nét tín hiệu rPPG.", ACCENT_RED)
    add_bullet_point(tf_con, "Phụ Thuộc Căn Cước Khuôn Mặt", "Khi đối tượng đeo mặt nạ dày hoặc quay lưng hoàn toàn, Luồng 2 tự động suy giảm trọng số.", ACCENT_RED)
    add_bullet_point(tf_con, "Thời Gian Tiền Xử Lý", "Trích xuất FFT + DWT tốn thêm 10-15ms so với nạp thẳng tensor vào mạng.", ACCENT_RED)

    # --------------------------------------------------------------------------
    # SLIDE 5: PHƯƠNG PHÁP HUẤN LUYỆN & HỌC MÁY LIÊN TỤC
    # --------------------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Phương Pháp Huấn Luyện & Pipeline Học Máy Liên Tục (Training Methodology)")

    card_w5 = Inches(3.65)
    
    # Kỹ thuật 1: Two-Stage Fine-tuning
    add_card(slide5, Inches(0.8), y_top, card_w5, Inches(3.3), "1. Huấn Luyện 2 Giai Đoạn", ACCENT_CYAN)
    tb = slide5.shapes.add_textbox(Inches(1.0), y_top + Inches(0.65), card_w5 - Inches(0.4), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    add_bullet_point(tf, "Giai đoạn 1 (Warmup)", "Đóng băng backbone ViT, chỉ train classification head (LR = 1e-3, 2 epochs).", ACCENT_CYAN)
    add_bullet_point(tf, "Giai đoạn 2 (Full Tuning)", "Mở khóa toàn bộ mạng với Cosine Annealing (LR = 1e-5), bảo toàn tri thức nền.", ACCENT_CYAN)

    # Kỹ thuật 2: Parquet Feature Caching
    add_card(slide5, Inches(0.8) + col_w + gap, y_top, card_w5, Inches(3.3), "2. Parquet Feature Caching", ACCENT_GREEN)
    tb = slide5.shapes.add_textbox(Inches(1.0) + col_w + gap, y_top + Inches(0.65), card_w5 - Inches(0.4), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    add_bullet_point(tf, "Nén 100 Lần", "1.2M mẫu đa phương thức (300 GB) nén thành ~2.5 GB bảng đặc trưng Parquet.", ACCENT_GREEN)
    add_bullet_point(tf, "Zero FUSE Bottleneck", "Loại bỏ hoàn toàn nghẽn I/O đọc tệp trên Google Drive; nạp trực tiếp vào GPU VRAM.", ACCENT_GREEN)

    # Kỹ thuật 3: Continuous Colab Scheduled Runner
    add_card(slide5, Inches(0.8) + (col_w + gap)*2, y_top, card_w5, Inches(3.3), "3. Học Liên Tục 4 Giờ/Lần", ACCENT_PURPLE)
    tb = slide5.shapes.add_textbox(Inches(1.0) + (col_w + gap)*2, y_top + Inches(0.65), card_w5 - Inches(0.4), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    add_bullet_point(tf, "Tự Động 100%", "Colab Pro Scheduled Runner kích hoạt định kỳ 4h, cào media mới từ Threads.", ACCENT_PURPLE)
    add_bullet_point(tf, "Zero-Downtime Reload", "Callback nạp trọng số mới vYYYY-MM-DD vào RAM Server mà không cần restart.", ACCENT_PURPLE)

    # Bottom Pros & Cons comparison
    add_card(slide5, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.8), "⚖️ Đánh Giá Ưu & Nhược Điểm Quy Trình Huấn Luyện", ACCENT_AMBER)
    tb_b = slide5.shapes.add_textbox(Inches(1.0), Inches(5.75), Inches(11.333), Inches(1.2))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p1 = tf_b.paragraphs[0]
    p1.text = "• Ưu điểm: Tốc độ huấn luyện siêu tốc (1 epoch trên 1.2M mẫu chỉ mất ~4-6 phút nhờ Parquet); chi phí điện toán cực rẻ (tận dụng GPU Cloud Google Colab); thích ứng tức thì với các thủ đoạn giả mạo mới trên Threads."
    p1.font.size = Pt(12)
    p1.font.color.rgb = TEXT_WHITE
    p1.space_after = Pt(4)
    p2 = tf_b.add_paragraph()
    p2.text = "• Nhược điểm & Thách thức: Đòi hỏi tiền xử lý vector hóa ban đầu; cần thuật toán Negative Mining nghiêm ngặt để chống trôi dạt mô hình (Model Drift) khi cào dữ liệu in-the-wild không được gán nhãn thủ công."
    p2.font.size = Pt(12)
    p2.font.color.rgb = TEXT_MUTED

    # --------------------------------------------------------------------------
    # SLIDE 6: HIỆU CHUẨN XÁC SUẤT TOÁN HỌC (TEMPERATURE SCALING)
    # --------------------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Hiệu Chuẩn Xác Suất Hậu Nghiệm (Post-Hoc Probability Calibration)")

    card_w6 = Inches(5.7)
    card_h6 = Inches(5.1)
    
    # Card Left: Platt Temperature Scaling
    add_card(slide6, Inches(0.8), y_top, card_w6, card_h6, "1. Platt Temperature Scaling (T* = 1.6178)", ACCENT_CYAN)
    tb_l = slide6.shapes.add_textbox(Inches(1.0), y_top + Inches(0.65), card_w6 - Inches(0.4), card_h6 - Inches(0.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    add_bullet_point(tf_l, "Vấn đề Overconfidence", "Mạng nơ-ron sâu thường đưa ra xác suất cực đoan p > 0.99 dù mẫu đó nằm ở biên giới OOD.", ACCENT_CYAN)
    add_bullet_point(tf_l, "Công thức Biến đổi", "P(y = 1 | x) = σ(z / T*), trong đó z là raw logit, T* là tham số nhiệt độ dương.", ACCENT_CYAN)
    add_bullet_point(tf_l, "Tối ưu hóa NLL", "Học T* bằng cách cực tiểu hóa hàm mất mát Negative Log-Likelihood trên tập Validation độc lập.", ACCENT_CYAN)
    add_bullet_point(tf_l, "Hạ Thấp ECE", "Chỉ số Expected Calibration Error giảm từ 15.4% xuống dưới 2.8%, đưa xác suất về đúng tần suất thực nghiệm.", ACCENT_CYAN)

    # Card Right: Youden's J Threshold Optimization
    add_card(slide6, Inches(6.833), y_top, card_w6, card_h6, "2. Ngưỡng Phân Quyết Youden's J (τ* = 0.8118)", ACCENT_GREEN)
    tb_r = slide6.shapes.add_textbox(Inches(7.033), y_top + Inches(0.65), card_w6 - Inches(0.4), card_h6 - Inches(0.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    add_bullet_point(tf_r, "Sai lầm của Ngưỡng 0.5", "Ngưỡng mặc định 0.5 gây bùng nổ tỷ lệ dương tính giả (False Positives) trên dữ liệu nén mạng xã hội.", ACCENT_GREEN)
    add_bullet_point(tf_r, "Công thức Youden's J", "J(τ) = TPR(τ) - FPR(τ) = Độ nhạy(τ) + Độ đặc hiệu(τ) - 1. Tìm cực đại trên đường cong ROC.", ACCENT_GREEN)
    add_bullet_point(tf_r, "Điểm Cắt Tối Ưu", "Xác lập ngưỡng quyết định τ* = 0.8118, triệt tiêu tối đa rủi ro báo động giả cho người dùng.", ACCENT_GREEN)
    add_bullet_point(tf_r, "Ý nghĩa Thực tiễn", "Đảm bảo tính nghiêm ngặt tư pháp: Thà gắn cờ nghi vấn cần chuyên gia kiểm tra hơn là kết luận sai lệch.", ACCENT_GREEN)

    # --------------------------------------------------------------------------
    # SLIDE 7: HỆ THỐNG 14 BENCHMARK & 5 PHÂN LỚP VIỆT NAM
    # --------------------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "Hệ Thống 14 Bộ Benchmark Quốc Tế & 5 Phân Lớp Sinh Trắc Học Việt Nam")

    # Table of 14 Benchmarks
    card_w7 = Inches(6.8)
    add_card(slide7, Inches(0.8), y_top, card_w7, card_h6, "14 Bộ Dữ Liệu Benchmark Chuẩn Quốc Tế", ACCENT_CYAN)
    
    # Table shape
    rows = 5
    cols = 3
    tbl_shape = slide7.shapes.add_table(rows, cols, Inches(1.0), y_top + Inches(0.7), card_w7 - Inches(0.4), Inches(4.0))
    table = tbl_shape.table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(3.2)
    table.columns[2].width = Inches(1.4)
    
    headers = ["Phương Thức", "Tên Bộ Benchmark Quốc Tế", "Số Mẫu"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(51, 65, 85)
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_CYAN
            
    b_data = [
        ("1. Hình Ảnh (Image)", "NTIRE 2024, CIFAKE, DFFD, DF40 (2026), DeepDetect-2025, 140k Real/Fake Faces", "450,000+"),
        ("2. Video", "Celeb-DF v2 & Celeb-DF++ (2025), FaceForensics++, HiDF, SARAAB DESC 2026 (7s)", "280,000+"),
        ("3. Âm Thanh & LipSync", "DeepSpeak (2024-2025), FakeAVCeleb, WaveFake, ASVspoof 2021/2024", "180,000+"),
        ("4. Siêu Tập OOD & Local", "Mega-MMDF (1.1M mẫu), Omni-Fake 2026 (1.2M OOD) + Threads Vietnamese", "1,200,000+")
    ]
    for r_idx, row in enumerate(b_data, start=1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = TEXT_WHITE

    # 5 Phân lớp sinh trắc học Việt Nam
    add_card(slide7, Inches(7.8), y_top, Inches(4.733), card_h6, "5 Phân Lớp Sinh Trắc Học Người Việt", ACCENT_PURPLE)
    tb_vn = slide7.shapes.add_textbox(Inches(8.0), y_top + Inches(0.65), Inches(4.333), card_h6 - Inches(0.8))
    tf_vn = tb_vn.text_frame
    tf_vn.word_wrap = True
    add_bullet_point(tf_vn, "vn_id_official", "Ảnh thẻ Căn cước công dân (CCCD), hộ chiếu (chống mở tài khoản ngân hàng rác eKYC).", ACCENT_PURPLE)
    add_bullet_point(tf_vn, "vn_selfie_lifestyle", "Ảnh/video tự chụp cận mặt đời thường trong môi trường sinh hoạt.", ACCENT_PURPLE)
    add_bullet_point(tf_vn, "vn_street_outdoor", "Video đường phố, ánh sáng tự nhiên phức tạp, xe cộ di chuyển.", ACCENT_PURPLE)
    add_bullet_point(tf_vn, "vn_occlusion_mask", "Mặt đeo khẩu trang y tế, kính râm, che khuất một phần bộ nhận diện.", ACCENT_PURPLE)
    add_bullet_point(tf_vn, "vn_mature_aging", "Người trung niên, cao tuổi với nếp nhăn tự nhiên (đối tượng chính của lừa đảo video call).", ACCENT_PURPLE)

    # --------------------------------------------------------------------------
    # SLIDE 8: KẾT QUẢ THỰC NGHIỆM & ĐỐI CHIẾU SOTA
    # --------------------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "Kết Quả Thực Nghiệm & Đối Chiếu Với Các Mô Hình Cơ Sở (SOTA Comparison)")

    # Results Table
    add_card(slide8, Inches(0.8), y_top, Inches(11.733), Inches(4.2), "Bảng Đối Chiếu Chỉ Số Kiểm Định Trên Tập Độc Lập Out-Of-Distribution (OOD)", ACCENT_CYAN)
    
    t_shape8 = slide8.shapes.add_table(6, 6, Inches(1.0), y_top + Inches(0.7), Inches(11.333), Inches(3.2))
    table8 = t_shape8.table
    widths = [Inches(3.2), Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.7)]
    for idx, w in enumerate(widths):
        table8.columns[idx].width = w
        
    h8 = ["Kiến Trúc Mô Hình", "Độ Chính Xác (Acc)", "Diện Tích ROC (AUC)", "Tỷ Lệ Lỗi (EER)", "Sai Số ECE", "Độ Trễ (Latency)"]
    for i, h in enumerate(h8):
        cell = table8.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(51, 65, 85)
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_CYAN

    data8 = [
        ("MesoNet-4 (CNN Cơ sở)", "73.2%", "0.784", "0.224 (22.4%)", "0.218", "45ms"),
        ("XceptionNet (FaceForensics++)", "81.5%", "0.862", "0.148 (14.8%)", "0.185", "62ms"),
        ("EfficientNet-B4 (Chuẩn phổ biến)", "86.7%", "0.912", "0.098 (9.8%)", "0.142", "78ms"),
        ("ViT-Base Gốc (Chưa hiệu chuẩn)", "92.4%", "0.965", "0.052 (5.2%)", "0.154", "82ms"),
        ("⭐ Đề Xuất: Dual-Stream + Calibrated", "99.4% (Ảnh)\n98.7% (Video)", "0.998 - 0.999", "0.011 (1.1%)", "< 0.028", "85ms (Ảnh)\n320ms (Video 7s)")
    ]
    for r_idx, row in enumerate(data8, start=1):
        is_best = (r_idx == 5)
        for c_idx, val in enumerate(row):
            cell = table8.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 58, 80) if is_best else CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.bold = is_best
                p.font.size = Pt(11)
                p.font.color.rgb = ACCENT_GREEN if is_best else TEXT_WHITE

    # Highlights Box
    add_card(slide8, Inches(0.8), Inches(6.1), Inches(11.733), Inches(1.0), fill_color=CARD_BG, border_color=ACCENT_GREEN)
    tb_hl = slide8.shapes.add_textbox(Inches(1.0), Inches(6.2), Inches(11.333), Inches(0.8))
    tf_hl = tb_hl.text_frame
    p_hl = tf_hl.paragraphs[0]
    p_hl.text = "🎯 Đột phá: AUC đạt xấp xỉ tuyệt đối (0.998 - 0.999), EER giảm sâu xuống 1.1% (thấp hơn 5-10 lần các baseline). ECE giảm từ 0.154 xuống dưới 0.028 chứng minh mô hình triệt tiêu hoàn toàn căn bệnh tự tin thái quá của Deep Learning."
    p_hl.font.size = Pt(12)
    p_hl.font.bold = True
    p_hl.font.color.rgb = TEXT_WHITE

    # --------------------------------------------------------------------------
    # SLIDE 9: HỆ TÁC THỂ & CHỨNG THƯ GIÁM ĐỊNH SỐ (EU AI ACT)
    # --------------------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "Hệ Thống Đa Tác Thể & Chứng Thư Giám Định Số (EU AI Act Compliance)")

    # 4 Agents Cards
    w_ag = Inches(2.7)
    gap_ag = Inches(0.3)
    
    agents = [
        ("Orchestrator Agent", "Điều phối luồng công việc, kiểm tra pre-flight gate, kích hoạt các agent chuyên gia theo nhu cầu.", ACCENT_CYAN),
        ("Detection Agent", "Thực thi tính toán mô hình ViT, biến đổi 2D FFT, Haar DWT và trích xuất nhịp mạch mao mạch rPPG.", ACCENT_PURPLE),
        ("Analysis Agent (LLM)", "Sử dụng Gemini/Claude tổng hợp các bằng chứng thành báo cáo giải trình chi tiết song ngữ Anh - Việt.", ACCENT_AMBER),
        ("SRE Observability", "Giám sát sức khỏe container, logging ClickHouse/Loki, kiểm tra độ trễ và cảnh báo vi phạm bảo mật.", ACCENT_GREEN)
    ]
    for idx, (a_name, a_desc, a_color) in enumerate(agents):
        x_pos = Inches(0.8) + idx * (w_ag + gap_ag)
        add_card(slide9, x_pos, y_top, w_ag, Inches(2.6), a_name, a_color)
        tb_a = slide9.shapes.add_textbox(x_pos + Inches(0.15), y_top + Inches(0.6), w_ag - Inches(0.3), Inches(1.8))
        tf_a = tb_a.text_frame
        tf_a.word_wrap = True
        p_a = tf_a.paragraphs[0]
        p_a.text = a_desc
        p_a.font.size = Pt(11)
        p_a.font.color.rgb = TEXT_WHITE

    # Forensic Certificate Showcase Box
    add_card(slide9, Inches(0.8), Inches(4.7), Inches(11.733), Inches(2.4), "Cấu Trúc Chứng Thư Giám Định Số Auditable Digital Forensic Certificate (JSON)", ACCENT_CYAN)
    tb_cert = slide9.shapes.add_textbox(Inches(1.0), Inches(5.25), Inches(11.333), Inches(1.7))
    tf_c = tb_cert.text_frame
    tf_c.word_wrap = True
    add_bullet_point(tf_c, "Định danh Pháp lý", "certificate_id: CERT-DF-XXXXXXXX, timestamp UTC, chuẩn tuân thủ EU AI Act Article 50 & DSA.", ACCENT_CYAN, font_size=11, space_after=4)
    add_bullet_point(tf_c, "Toàn vẹn Tệp (Integrity)", "Mã băm SHA-256 tệp gốc, chữ ký số Ed25519-SHA256, URL xác thực công khai /api/verify/{cert_id}.", ACCENT_CYAN, font_size=11, space_after=4)
    add_bullet_point(tf_c, "Bằng chứng Đa chiều", "Passive ViT score, gai quang phổ FFT 2D, sai phân phương sai nhiễu không gian, entropy bất định.", ACCENT_CYAN, font_size=11, space_after=4)
    add_bullet_point(tf_c, "SARAAB Video Audit", "Sliding window 7s, frame-to-frame motion jitter, và bản đồ nhiệt 8x8 Patch Heatmap khoanh vùng can thiệp.", ACCENT_CYAN, font_size=11, space_after=4)

    # --------------------------------------------------------------------------
    # SLIDE 10: TỔNG KẾT ĐÓNG GÓP & LỘ TRÌNH PHÁT TRIỂN
    # --------------------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Tổng Kết Đóng Góp Mới Của Luận Án & Lộ Trình Phát Triển (Conclusions)")

    card_w10 = Inches(5.7)
    card_h10 = Inches(5.1)

    # 3 Đóng góp mới
    add_card(slide10, Inches(0.8), y_top, card_w10, card_h10, "💎 Ba Đóng Góp Mới Của Luận Án (Novel Contributions)", ACCENT_CYAN)
    tb_c1 = slide10.shapes.add_textbox(Inches(1.0), y_top + Inches(0.65), card_w10 - Inches(0.4), card_h10 - Inches(0.8))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    add_bullet_point(tf_c1, "1. Đóng góp Lý thuyết", "Đề xuất khung Song luồng kết hợp hài hòa giữa thị giác sâu ViT và các đặc trưng vật lý - sinh lý học bất biến (2D FFT, rPPG, LipSync).", ACCENT_CYAN)
    add_bullet_point(tf_c1, "2. Hiệu chuẩn Toán học", "Mô hình hóa thành công bài toán giải phóng Overconfidence bằng nhiệt độ Platt T* và ngưỡng tối ưu Youden's J.", ACCENT_CYAN)
    add_bullet_point(tf_c1, "3. Đóng góp Kỹ thuật & Địa phương", "Xây dựng pipeline Parquet Caching nén 100 lần và bộ dữ liệu 5 phân lớp sinh trắc học đặc thù người Việt Nam.", ACCENT_CYAN)
    add_bullet_point(tf_c1, "4. Giá trị Pháp lý & Thực tiễn", "Hệ thống đa tác thể xuất Chứng thư Giám định Số đáp ứng tiêu chuẩn đạo luật AI châu Âu (EU AI Act).", ACCENT_CYAN)

    # Lộ trình nghiên cứu
    add_card(slide10, Inches(6.833), y_top, card_w10, card_h10, "🚀 Lộ Trình Triển Khai (Project Roadmap)", ACCENT_GREEN)
    tb_c2 = slide10.shapes.add_textbox(Inches(7.033), y_top + Inches(0.65), card_w10 - Inches(0.4), card_h10 - Inches(0.8))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    add_bullet_point(tf_c2, "✅ Giai Đoạn 1 (Đã hoàn thành)", "Tích hợp chuẩn cửa sổ 7s SARAAB, Temporal Motion Jitter, bản đồ nhiệt 8x8 Patch Heatmap và Parquet pipeline.", ACCENT_GREEN)
    add_bullet_point(tf_c2, "⏳ Giai Đoạn 2 (Q4/2026)", "Khi Dubai DESC mở kho mã nguồn SARAAB trên Hugging Face: Tích hợp trực tiếp checkpoint SARAAB làm backbone bổ trợ.", ACCENT_GREEN)
    add_bullet_point(tf_c2, "📄 Công Bố Khoa Học", "Hoàn thiện bản thảo gửi tạp chí quốc tế Q1 ISI (IEEE Transactions on Information Forensics and Security / ACM CCS).", ACCENT_GREEN)
    add_bullet_point(tf_c2, "🎓 Bảo Vệ Luận Án", "Nghiệm thu toàn bộ hệ thống phần mềm, tập dữ liệu chuẩn hóa và hoàn tất luận án tiến sĩ đúng kế hoạch.", ACCENT_GREEN)

    # Save presentation
    prs.save(str(output_path))
    print(f"[OK] Successfully created PowerPoint presentation at: {output_path}")

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "DeepFake_Detection_PhD_Thesis_Presentation.pptx"
    build_presentation(out_file)
