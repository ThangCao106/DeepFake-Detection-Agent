# scripts/generate_report_figures.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure typography for academic Vietnamese rendering
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def create_figure_1_tam_giac_hoi_tu(output_path):
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background subtle border
    outer_box = patches.FancyBboxPatch((1, 1), 98, 98, boxstyle="round,pad=1,rounding_size=2",
                                        facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.5)
    ax.add_patch(outer_box)

    # Title
    ax.text(50, 95, "MÔ HÌNH TAM GIÁC HỘI TỤ TRONG GIÁM ĐỊNH DEEPFAKE",
            fontsize=15, fontweight='bold', ha='center', va='center', color='#0F172A')
    ax.text(50, 91.5, "Sự Kết Hợp Giữa Khung Pháp Lý Bạch Diện (XAI) và Dấu Vết Vật Lý - Sinh Trắc Học",
            fontsize=10.5, fontstyle='italic', ha='center', va='center', color='#475569')

    # 1. TOP BOX: OPTION C (KHUNG KIẾN TRÚC CHỦ ĐẠO)
    top_box = patches.FancyBboxPatch((20, 61), 60, 26, boxstyle="round,pad=0.8,rounding_size=2",
                                     facecolor='#EEF2FF', edgecolor='#4338CA', linewidth=2.2, zorder=3)
    ax.add_patch(top_box)
    ax.text(50, 83.5, "[KIẾN TRÚC CHỦ ĐẠO: OPTION C]",
            fontsize=11.5, fontweight='bold', ha='center', va='center', color='#312E81', zorder=4)
    ax.text(50, 79.5, "Giám Định Bạch Diện (White-Box XAI) & Chứng Thư Số Pháp Lý",
            fontsize=11, fontweight='bold', ha='center', va='center', color='#1E1B4B', zorder=4)
    
    top_desc = (
        "• Tiêu chuẩn minh bạch: Tuân thủ nghiêm ngặt EU AI Act Article 50 (XAI).\n"
        "• Toàn vẹn chứng cứ số: Khóa băm SHA-256 + Chữ ký số Ed25519 chống can thiệp.\n"
        "• Ma trận chứng cứ 6 chiều: Tự động hóa bản dịch giải trình ngôn ngữ tự nhiên.\n"
        "• Thẩm tra công khai: Truy xuất chứng thư trực tuyến và Google Indexing API."
    )
    ax.text(50, 70, top_desc, fontsize=9.2, ha='center', va='center', color='#1E293B', zorder=4, linespacing=1.4)

    # 2. BOTTOM-LEFT BOX: OPTION A (SENSOR PRNU RESIDUAL)
    bl_box = patches.FancyBboxPatch((4, 10), 42, 28, boxstyle="round,pad=0.8,rounding_size=2",
                                    facecolor='#ECFDF5', edgecolor='#059669', linewidth=2, zorder=3)
    ax.add_patch(bl_box)
    ax.text(25, 34.5, "[BỔ SUNG 1: OPTION A]",
            fontsize=10.5, fontweight='bold', ha='center', va='center', color='#065F46', zorder=4)
    ax.text(25, 30.5, "Dấu Vết Cảm Biến Sensor PRNU",
            fontsize=10.5, fontweight='bold', ha='center', va='center', color='#047857', zorder=4)
    
    bl_desc = (
        "• Tích hợp: Luồng 1 (Thị giác Tổng quát).\n"
        "• Cơ chế: Nhiễu dư phần cứng W = I - F(I)\n"
        "  và hệ số Kurtosis phân phối hạt.\n"
        "• Đột phá: Blind PRNU không cần máy ảnh gốc.\n"
        "• Bắt trọn: Ranh giới ghép dán FaceSwap\n"
        "  và sự triệt tiêu tần số của Diffusion."
    )
    ax.text(25, 20, bl_desc, fontsize=8.8, ha='center', va='center', color='#064E3B', zorder=4, linespacing=1.35)

    # 3. BOTTOM-RIGHT BOX: OPTION B (CORNEAL REFLECTION SYMMETRY)
    br_box = patches.FancyBboxPatch((54, 10), 42, 28, boxstyle="round,pad=0.8,rounding_size=2",
                                     facecolor='#FAF5FF', edgecolor='#7C3AED', linewidth=2, zorder=3)
    ax.add_patch(br_box)
    ax.text(75, 34.5, "[BỔ SUNG 2: OPTION B]",
            fontsize=10.5, fontweight='bold', ha='center', va='center', color='#581C87', zorder=4)
    ax.text(75, 30.5, "Bất Đối Xứng Giác Mạc Mắt",
            fontsize=10.5, fontweight='bold', ha='center', va='center', color='#6D28D9', zorder=4)
    
    br_desc = (
        "• Tích hợp: Luồng 2 (Sinh trắc Học sâu).\n"
        "• Cơ chế: Quang học mắt (Gương cầu lồi),\n"
        "  vector lệch tâm khoảng cách Δ giữa 2 mắt.\n"
        "• Đột phá: Hộp giải phẫu Ocular tự động.\n"
        "• Bắt trọn: Đốm sáng giả lập StyleGAN\n"
        "  và vector phản xạ môi trường lệch pha."
    )
    ax.text(75, 20, br_desc, fontsize=8.8, ha='center', va='center', color='#4C1D95', zorder=4, linespacing=1.35)

    # Central Badge
    center_circle = patches.FancyBboxPatch((40, 43), 20, 10, boxstyle="round,pad=0.5,rounding_size=1.5",
                                           facecolor='#FFFFFF', edgecolor='#0F172A', linewidth=1.5, zorder=5)
    ax.add_patch(center_circle)
    ax.text(50, 49, "HỢP NHẤT TỐI ƯU", fontsize=9, fontweight='bold', ha='center', va='center', color='#0F172A', zorder=6)
    ax.text(50, 45, "α = 0.60 | τ* = 0.8118", fontsize=8.5, fontweight='bold', ha='center', va='center', color='#2563EB', zorder=6)

    # Connecting Arrows (Double-headed / annotated arrows)
    # Arrow Top to Bottom-Left
    ax.annotate("", xy=(30, 39), xytext=(40, 61),
                arrowprops=dict(arrowstyle="<->", color="#4338CA", lw=2, mutation_scale=15), zorder=2)
    ax.text(26, 52, "Dấu Vết Vật Lý\nPhần Cứng", fontsize=8.5, fontweight='bold', ha='center', va='center',
            color='#1E40AF', bbox=dict(boxstyle="round,pad=0.2", facecolor='#EFF6FF', edgecolor='none'))

    # Arrow Top to Bottom-Right
    ax.annotate("", xy=(70, 39), xytext=(60, 61),
                arrowprops=dict(arrowstyle="<->", color="#7C3AED", lw=2, mutation_scale=15), zorder=2)
    ax.text(74, 52, "Sinh Trắc &\nQuang Hình Học", fontsize=8.5, fontweight='bold', ha='center', va='center',
            color='#6D28D9', bbox=dict(boxstyle="round,pad=0.2", facecolor='#F3E8FF', edgecolor='none'))

    # Arrow Bottom-Left to Bottom-Right
    ax.annotate("", xy=(53, 24), xytext=(47, 24),
                arrowprops=dict(arrowstyle="<->", color="#059669", lw=2, mutation_scale=15), zorder=2)
    ax.text(50, 20.5, "Kiểm Thử Loại Trừ\nAblation Study", fontsize=8, fontweight='bold', ha='center', va='center',
            color='#065F46', bbox=dict(boxstyle="round,pad=0.2", facecolor='#ECFDF5', edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[success] Generated Figure 1 at: {output_path}")

def create_figure_2_kien_truc_tong_the(output_path):
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Outer border
    outer = patches.FancyBboxPatch((1, 1), 118, 98, boxstyle="round,pad=1,rounding_size=2",
                                   facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.5)
    ax.add_patch(outer)

    # Main Title
    ax.text(60, 95.5, "SƠ ĐỒ KIẾN TRÚC TỔNG THỂ HỆ THỐNG GIÁM ĐỊNH PHÁP Y ĐA LUỒNG 6D",
            fontsize=15, fontweight='bold', ha='center', va='center', color='#0F172A')
    ax.text(60, 92, "Kiến Trúc Thích Ứng (Adaptive Dual-Stream), Hiệu Chuẩn Platt ECE & Chứng Thư Số EU AI Act",
            fontsize=10, fontstyle='italic', ha='center', va='center', color='#475569')

    # STEP 1: INPUT MEDIA BOX
    in_box = patches.FancyBboxPatch((3, 44), 16, 32, boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor='#F1F5F9', edgecolor='#475569', linewidth=1.8)
    ax.add_patch(in_box)
    ax.text(11, 71, "MEDIA ĐẦU VÀO", fontsize=9.5, fontweight='bold', ha='center', va='center', color='#0F172A')
    ax.text(11, 67, "(Ảnh / Video / Audio)", fontsize=8, ha='center', va='center', color='#64748B')
    ax.text(11, 58, "Mã Băm Bất Biến:\nSHA-256 Digest\nChống Can Thiệp", fontsize=8, ha='center', va='center', color='#334155', linespacing=1.3)
    ax.text(11, 48, "Phân giải đa tỉ lệ\n& Đồng bộ âm sắc", fontsize=7.5, ha='center', va='center', color='#64748B')

    # ARROW 1 -> GATING
    ax.annotate("", xy=(23, 60), xytext=(19.5, 60),
                arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2, mutation_scale=14))

    # STEP 2: HUMAN PRESENCE GATE
    gate_box = patches.FancyBboxPatch((23.5, 42), 18, 36, boxstyle="round,pad=0.5,rounding_size=1.5",
                                      facecolor='#FEF3C7', edgecolor='#D97706', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(32.5, 73, "BỘ LỌC CỔNG", fontsize=10, fontweight='bold', ha='center', va='center', color='#92400E')
    ax.text(32.5, 69, "(Human Presence Gate)", fontsize=8, ha='center', va='center', color='#B45309')
    
    gate_text = (
        "1. Dò tìm khuôn mặt:\n   HOG / CNN Face\n"
        "2. Phân đoạn sắc tố da:\n   Không gian YCbCr\n   (Cb in [77, 127],\n    Cr in [133, 173])\n"
        "3. Tự định tuyến luồng"
    )
    ax.text(32.5, 54, gate_text, fontsize=7.8, ha='center', va='center', color='#78350F', linespacing=1.25)

    # BRANCHING ARROWS
    # Arrow to Stream 1 (Top)
    ax.annotate("", xy=(46, 73), xytext=(42, 65),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2, mutation_scale=14))
    ax.text(40, 72, "Mọi phương tiện\n(General)", fontsize=7.5, fontweight='bold', color='#1D4ED8')

    # Arrow to Stream 2 (Bottom)
    ax.annotate("", xy=(46, 32), xytext=(42, 53),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2, mutation_scale=14))
    ax.text(39, 40, "Phát hiện có mặt\nngười (Human)", fontsize=7.5, fontweight='bold', color='#6D28D9')

    # STREAM 1 BOX (TOP)
    s1_box = patches.FancyBboxPatch((46.5, 57), 35, 31, boxstyle="round,pad=0.6,rounding_size=1.5",
                                    facecolor='#EFF6FF', edgecolor='#2563EB', linewidth=2)
    ax.add_patch(s1_box)
    ax.text(64, 84, "LUỒNG 1: THỊ GIÁC TỔNG QUÁT (GENERAL STREAM)", fontsize=9.5, fontweight='bold', ha='center', color='#1E40AF')
    s1_items = (
        "• Dim 1: Vision Transformer (ViT) Feature Vector (55%)\n"
        "• Dim 2: Lưới chu kỳ miền tần số 2D FFT Anomaly (15%)\n"
        "• Dim 3: Dấu vân cảm biến phần cứng Blind PRNU (15%)\n"
        "• Dim 4: Bất biến phương sai nhiễu không gian (10%)\n"
        "• Dim 6a: Bản đồ nhiệt thao túng SARAAB Heatmap (5%)"
    )
    ax.text(64, 71.5, s1_items, fontsize=8, ha='center', va='center', color='#1E3A8A', linespacing=1.35)
    ax.text(64, 61, "Điểm rủi ro Luồng 1: Risk_general in [0.0, 1.0]", fontsize=8.5, fontweight='bold', ha='center', color='#1D4ED8')

    # STREAM 2 BOX (BOTTOM)
    s2_box = patches.FancyBboxPatch((46.5, 14), 35, 35, boxstyle="round,pad=0.6,rounding_size=1.5",
                                    facecolor='#F5F3FF', edgecolor='#7C3AED', linewidth=2)
    ax.add_patch(s2_box)
    ax.text(64, 45, "LUỒNG 2: SINH TRẮC CON NGƯỜI (HUMAN SPECIALIST)", fontsize=9.5, fontweight='bold', ha='center', color='#5B21B6')
    s2_items = (
        "• Dim 5: Bất đối xứng phản xạ quang học giác mạc mắt\n"
        "• S2a: Vi cấu trúc da biến đổi sóng con 2D DWT WAVIE\n"
        "• S2b: Tính nhất quán nhịp sinh học mạch máu rPPG\n"
        "• S2c: Quỹ đạo giọng nói SSL HuBERT & LipSync DF26\n"
        "• Hiệu chuẩn Platt Scaling (T* = 1.6178) & Youden (τ* = 0.8118)"
    )
    ax.text(64, 32, s2_items, fontsize=8, ha='center', va='center', color='#4C1D95', linespacing=1.35)
    ax.text(64, 18, "Điểm rủi ro Luồng 2: Risk_human (ECE < 0.020)", fontsize=8.5, fontweight='bold', ha='center', color='#6D28D9')

    # ARROWS FROM STREAMS TO FUSION
    ax.annotate("", xy=(85.5, 60), xytext=(82, 70),
                arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2, mutation_scale=14))
    ax.annotate("", xy=(85.5, 45), xytext=(82, 33),
                arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2, mutation_scale=14))

    # STEP 3: FUSION & 6D MATRIX BOX
    fus_box = patches.FancyBboxPatch((86, 36), 31, 38, boxstyle="round,pad=0.6,rounding_size=1.5",
                                     facecolor='#ECFDF5', edgecolor='#059669', linewidth=2.2)
    ax.add_patch(fus_box)
    ax.text(101.5, 70.5, "HỢP NHẤT ĐA LUỒNG & MA TRẬN 6D", fontsize=9.5, fontweight='bold', ha='center', color='#065F46')
    
    fus_text = (
        "Hợp nhất Trọng số Tối ưu (Ablation):\n"
        "• Có người: Risk_comp = 0.40·R_gen + 0.60·R_hum\n"
        "• Phi nhân tạo: Risk_comp = 1.00·R_gen\n\n"
        "Ma Trận Chứng Cứ 6 Chiều (6D Matrix):\n"
        "[1] ViT  [2] FFT  [3] PRNU\n"
        "[4] Không gian  [5] Giác mạc  [6] SARAAB"
    )
    ax.text(101.5, 55.5, fus_text, fontsize=8, ha='center', va='center', color='#064E3B', linespacing=1.25)
    ax.text(101.5, 40.5, "Phán Quyết Hậu Kiểm: τ* = 0.8118", fontsize=8.5, fontweight='bold', ha='center', color='#047857')

    # ARROW TO OUTPUT
    ax.annotate("", xy=(101.5, 26), xytext=(101.5, 36),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2, mutation_scale=14))

    # STEP 4: AUDITABLE CERTIFICATE BOX (BOTTOM RIGHT)
    cert_box = patches.FancyBboxPatch((86, 6), 31, 19, boxstyle="round,pad=0.5,rounding_size=1.5",
                                      facecolor='#0F172A', edgecolor='#334155', linewidth=1.5)
    ax.add_patch(cert_box)
    ax.text(101.5, 21.5, "CHỨNG THƯ PHÁP LÝ SỐ (CERT-DF)", fontsize=9.5, fontweight='bold', ha='center', color='#F8FAFC')
    cert_text = (
        "• Chuẩn mực: EU AI Act Art. 50 (White-box XAI)\n"
        "• Khóa bảo mật: Ed25519-SHA256 Signature\n"
        "• Thẩm tra: Google Indexing API & REST Endpoint"
    )
    ax.text(101.5, 12.5, cert_text, fontsize=7.8, ha='center', va='center', color='#E2E8F0', linespacing=1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[success] Generated Figure 2 at: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "figures"))
    os.makedirs(base_dir, exist_ok=True)

    fig1_path = os.path.join(base_dir, "fig1_tam_giac_hoi_tu.png")
    fig2_path = os.path.join(base_dir, "fig2_kien_truc_tong_the_6d.png")

    print("Generating High-Resolution Architectural Figures (300 DPI)...")
    create_figure_1_tam_giac_hoi_tu(fig1_path)
    create_figure_2_kien_truc_tong_the(fig2_path)
    print("All figures successfully created!")
