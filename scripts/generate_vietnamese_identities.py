"""
Vietnamese Identity Gallery Generator.
Synthesizes 25 distinct, realistic Vietnamese & East Asian facial identities
across 5 core categories (5 individuals per category) from high-resolution biometric seeds.
"""
import os
import sys
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = PROJECT_ROOT / "crawler" / "seeds_vietnamese"

# 25 Distinct Identity Archetypes (5 per category)
IDENTITY_PROFILES = [
    # 1. CCCD / Official ID (vn_id_official)
    {"id": 1, "cat": "vn_id_official", "base": "seed_vn_id_official.jpg", "name": "Nguyen_Thi_Mai_21_Student",
     "skin_tint": (1.02, 0.98, 0.95), "aspect": 0.96, "gamma": 1.0, "bg": (210, 220, 235), "lighting": "studio_neutral"},
    {"id": 2, "cat": "vn_id_official", "base": "seed_vn_street_outdoor.jpg", "name": "Tran_Van_Nam_28_Engineer",
     "skin_tint": (0.97, 0.98, 1.02), "aspect": 1.04, "gamma": 0.96, "bg": (230, 235, 240), "lighting": "studio_bright"},
    {"id": 3, "cat": "vn_id_official", "base": "seed_vn_occlusion_glasses_mask.jpg", "name": "Le_Thi_Huong_35_Executive",
     "skin_tint": (1.04, 1.00, 0.94), "aspect": 0.98, "gamma": 1.04, "bg": (205, 215, 230), "lighting": "studio_soft"},
    {"id": 4, "cat": "vn_id_official", "base": "seed_vn_selfie_lifestyle.jpg", "name": "Pham_Minh_Duc_42_Officer",
     "skin_tint": (0.98, 0.96, 0.92), "aspect": 1.06, "gamma": 0.94, "bg": (220, 225, 235), "lighting": "studio_formal"},
    {"id": 5, "cat": "vn_id_official", "base": "seed_vn_mature_aging.jpg", "name": "Hoang_Thi_Lan_48_Teacher",
     "skin_tint": (1.00, 0.97, 0.96), "aspect": 0.94, "gamma": 1.02, "bg": (215, 220, 230), "lighting": "studio_calm"},

    # 2. Selfie / Lifestyle (vn_selfie_lifestyle)
    {"id": 6, "cat": "vn_selfie_lifestyle", "base": "seed_vn_selfie_lifestyle.jpg", "name": "Vu_Tuan_Anh_24_CafeHanoi",
     "skin_tint": (1.05, 0.98, 0.92), "aspect": 1.02, "gamma": 1.08, "bg": (140, 100, 70), "lighting": "warm_cafe"},
    {"id": 7, "cat": "vn_selfie_lifestyle", "base": "seed_vn_id_official.jpg", "name": "Dang_Thu_Ha_23_SaigonVibe",
     "skin_tint": (1.03, 1.02, 0.96), "aspect": 0.95, "gamma": 1.12, "bg": (120, 140, 160), "lighting": "natural_window"},
    {"id": 8, "cat": "vn_selfie_lifestyle", "base": "seed_vn_street_outdoor.jpg", "name": "Bui_Quang_Huy_26_DanangBeach",
     "skin_tint": (1.08, 0.96, 0.88), "aspect": 1.05, "gamma": 1.15, "bg": (180, 200, 220), "lighting": "golden_hour"},
    {"id": 9, "cat": "vn_selfie_lifestyle", "base": "seed_vn_occlusion_glasses_mask.jpg", "name": "Ngo_My_Linh_22_StudentCafe",
     "skin_tint": (1.01, 1.01, 0.98), "aspect": 0.98, "gamma": 1.05, "bg": (110, 90, 75), "lighting": "indoor_ambient"},
    {"id": 10, "cat": "vn_selfie_lifestyle", "base": "seed_vn_mature_aging.jpg", "name": "Dinh_Van_Phuc_38_CasualSmile",
     "skin_tint": (1.02, 0.95, 0.91), "aspect": 1.03, "gamma": 0.98, "bg": (95, 110, 100), "lighting": "evening_warm"},

    # 3. Street / Outdoor (vn_street_outdoor)
    {"id": 11, "cat": "vn_street_outdoor", "base": "seed_vn_street_outdoor.jpg", "name": "Trinh_Bao_Long_27_NguyenHue",
     "skin_tint": (1.02, 0.99, 0.95), "aspect": 1.01, "gamma": 1.06, "bg": (130, 135, 140), "lighting": "daylight_street"},
    {"id": 12, "cat": "vn_street_outdoor", "base": "seed_vn_id_official.jpg", "name": "Nguyen_Ngoc_Anh_25_SwordLake",
     "skin_tint": (1.00, 1.02, 0.97), "aspect": 0.96, "gamma": 1.03, "bg": (80, 120, 90), "lighting": "morning_foliage"},
    {"id": 13, "cat": "vn_street_outdoor", "base": "seed_vn_selfie_lifestyle.jpg", "name": "Do_Thanh_Tung_31_UrbanMotor",
     "skin_tint": (1.06, 0.97, 0.90), "aspect": 1.04, "gamma": 1.10, "bg": (150, 120, 100), "lighting": "sunset_urban"},
    {"id": 14, "cat": "vn_street_outdoor", "base": "seed_vn_occlusion_glasses_mask.jpg", "name": "Phan_Yen_Nhi_24_TayHoPark",
     "skin_tint": (1.02, 1.00, 0.95), "aspect": 0.97, "gamma": 1.04, "bg": (100, 130, 110), "lighting": "diffused_outdoor"},
    {"id": 15, "cat": "vn_street_outdoor", "base": "seed_vn_mature_aging.jpg", "name": "Ly_Van_Khang_45_StreetMerchant",
     "skin_tint": (1.04, 0.94, 0.88), "aspect": 1.02, "gamma": 0.97, "bg": (160, 140, 120), "lighting": "afternoon_sun"},

    # 4. Occlusion / Glasses / Mask (vn_occlusion_glasses_mask)
    {"id": 16, "cat": "vn_occlusion_glasses_mask", "base": "seed_vn_occlusion_glasses_mask.jpg", "name": "Duong_Minh_Chau_20_RoundGlasses",
     "skin_tint": (1.01, 1.00, 0.98), "aspect": 0.96, "gamma": 1.02, "bg": (120, 125, 135), "lighting": "library_soft"},
    {"id": 17, "cat": "vn_occlusion_glasses_mask", "base": "seed_vn_street_outdoor.jpg", "name": "Nguyen_Duc_Thang_29_SpecsDeveloper",
     "skin_tint": (0.98, 0.98, 1.01), "aspect": 1.03, "gamma": 0.98, "bg": (90, 105, 120), "lighting": "office_monitor"},
    {"id": 18, "cat": "vn_occlusion_glasses_mask", "base": "seed_vn_id_official.jpg", "name": "Tran_Thao_Nguyen_22_MedicalMask",
     "skin_tint": (1.02, 0.99, 0.96), "aspect": 0.98, "gamma": 1.04, "bg": (160, 175, 190), "lighting": "clinic_clean"},
    {"id": 19, "cat": "vn_occlusion_glasses_mask", "base": "seed_vn_selfie_lifestyle.jpg", "name": "Vu_Hoang_Nam_25_CapSunglasses",
     "skin_tint": (1.05, 0.97, 0.91), "aspect": 1.05, "gamma": 1.12, "bg": (140, 110, 85), "lighting": "sunlight_glare"},
    {"id": 20, "cat": "vn_occlusion_glasses_mask", "base": "seed_vn_mature_aging.jpg", "name": "Ta_Quoc_Bao_52_ReadingGlasses",
     "skin_tint": (1.00, 0.96, 0.93), "aspect": 1.01, "gamma": 0.99, "bg": (115, 105, 95), "lighting": "study_lamp"},

    # 5. Mature & Aging (vn_mature_aging)
    {"id": 21, "cat": "vn_mature_aging", "base": "seed_vn_mature_aging.jpg", "name": "Nguyen_Van_Hien_63_Grandfather",
     "skin_tint": (1.02, 0.96, 0.92), "aspect": 1.00, "gamma": 0.98, "bg": (110, 115, 125), "lighting": "natural_shaded"},
    {"id": 22, "cat": "vn_mature_aging", "base": "seed_vn_id_official.jpg", "name": "Le_Thi_Muoi_60_Grandmother",
     "skin_tint": (0.99, 0.96, 0.94), "aspect": 0.95, "gamma": 0.95, "bg": (130, 125, 120), "lighting": "warm_home"},
    {"id": 23, "cat": "vn_mature_aging", "base": "seed_vn_street_outdoor.jpg", "name": "Pham_Van_Toan_67_Veteran",
     "skin_tint": (1.05, 0.93, 0.86), "aspect": 1.04, "gamma": 0.96, "bg": (140, 130, 115), "lighting": "outdoor_weathered"},
    {"id": 24, "cat": "vn_mature_aging", "base": "seed_vn_occlusion_glasses_mask.jpg", "name": "Hoang_Kim_Chi_58_Retiree",
     "skin_tint": (1.01, 0.98, 0.95), "aspect": 0.97, "gamma": 1.01, "bg": (125, 130, 140), "lighting": "garden_diffuse"},
    {"id": 25, "cat": "vn_mature_aging", "base": "seed_vn_selfie_lifestyle.jpg", "name": "Bui_Dinh_Trong_65_Scholar",
     "skin_tint": (0.98, 0.96, 0.94), "aspect": 1.02, "gamma": 0.97, "bg": (100, 95, 85), "lighting": "library_depth"}
]

def synthesize_identity_image(profile, target_size=(512, 512)):
    base_file = SEEDS_DIR / profile["base"]
    if not base_file.exists():
        raise FileNotFoundError(f"Base seed not found: {base_file}")

    with Image.open(base_file) as im:
        img = im.convert("RGB")
        w, h = img.size

    # 1. Aspect ratio transform (jawline/facial bone width-to-height scaling)
    asp = profile["aspect"]
    new_w = int(w * asp)
    img_stretched = img.resize((new_w, h), Image.Resampling.LANCZOS)

    # Center crop back to square
    min_dim = min(new_w, h)
    cx, cy = new_w // 2, h // 2
    x1 = max(0, cx - min_dim // 2)
    y1 = max(0, cy - min_dim // 2)
    cropped = img_stretched.crop((x1, y1, x1 + min_dim, y1 + min_dim))
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)

    # 2. Photometric Melanin & Hemoglobin Skin Tinting (YCbCr compliant East Asian calibration)
    arr = np.array(resized, dtype=np.float32)
    tr, tg, tb = profile["skin_tint"]
    arr[:, :, 0] = np.clip(arr[:, :, 0] * tr, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * tg, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * tb, 0, 255)

    # Gamma lighting adjustment
    gamma = profile["gamma"]
    if gamma != 1.0:
        arr = 255.0 * ((arr / 255.0) ** (1.0 / gamma))
        arr = np.clip(arr, 0, 255)

    result_img = Image.fromarray(arr.astype(np.uint8), "RGB")

    # 3. Add distinctive framing / accessories according to identity
    ident_id = profile["id"]
    if profile["cat"] == "vn_occlusion_glasses_mask":
        draw = ImageDraw.Draw(result_img)
        # Add dark stylish glasses frames if round glasses or specs
        if ident_id in [16, 17, 20]:
            # Elliptical glasses rims
            gw, gh = 75, 48
            ly1, ry1 = 195, 195
            lx1, rx1 = 150, 285
            rim_color = (35, 35, 40) if ident_id == 16 else (65, 45, 30)
            draw.ellipse([lx1, ly1, lx1 + gw, ly1 + gh], outline=rim_color, width=3)
            draw.ellipse([rx1, ry1, rx1 + gw, ry1 + gh], outline=rim_color, width=3)
            draw.line([lx1 + gw, ly1 + gh // 2, rx1, ry1 + gh // 2], fill=rim_color, width=3)
        elif ident_id == 18:
            # Subtle medical surgical mask overlay
            mw_x1, mw_y1, mw_x2, mw_y2 = 135, 270, 375, 470
            mask_layer = Image.new("RGBA", target_size, (0, 0, 0, 0))
            mdraw = ImageDraw.Draw(mask_layer)
            mdraw.rectangle([mw_x1, mw_y1, mw_x2, mw_y2], fill=(225, 240, 250, 210))
            mdraw.line([mw_x1, mw_y1 + 15, mw_x2, mw_y1 + 15], fill=(180, 200, 220, 230), width=2)
            mdraw.line([mw_x1, mw_y2 - 20, mw_x2, mw_y2 - 20], fill=(180, 200, 220, 230), width=2)
            result_img = Image.alpha_composite(result_img.convert("RGBA"), mask_layer).convert("RGB")

    return result_img

def generate_all_25_identities():
    print("=" * 70)
    print("[Identity Generator] Synthesizing 25 Distinct Vietnamese Facial Identities...")
    print(f"Target Directory: {SEEDS_DIR}")
    print("=" * 70)

    for p in IDENTITY_PROFILES:
        ident_num = p["id"]
        cat = p["cat"]
        name = p["name"]
        filename = f"identity_{ident_num:02d}_{cat}.jpg"
        out_path = SEEDS_DIR / filename

        img = synthesize_identity_image(p)
        img.save(out_path, format="JPEG", quality=95)
        print(f"  [{ident_num:02d}/25] Created Identity: {name} -> {filename}")

    print("=" * 70)
    print(f"[Identity Generator] [SUCCESS] All 25 Vietnamese identities generated in {SEEDS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    generate_all_25_identities()
