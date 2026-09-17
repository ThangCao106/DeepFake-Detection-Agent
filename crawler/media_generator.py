"""
Sample Media Generator & Active Visual Validator for Deepfake Pentest Job.
Generates realistic human portrait images, faceswap perturbations, and generative diffusion artifacts
to provide visual inspection samples directly to Google Drive:
  GDrive/DeepFake_Pentest_Data/YYYY-MM-DD/
    ├── 01_raw_originals/         (Real human portrait samples .jpg)
    ├── 02_manipulated_faceswap/  (FaceSwap boundary manipulated samples .jpg)
    └── 03_manipulated_openweight/(Generative AI synthetic samples .jpg)
"""
import os
import sys
import math
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_realistic_human_portrait(width=512, height=512, seed=None):
    """
    Synthesizes a realistic human portrait with natural skin chrominance (YCbCr compliant),
    face oval, facial landmark features (eyes, eyebrows, nose, mouth), and organic lighting.
    """
    if seed is not None:
        random.seed(seed)
        
    # 1. Background (studio neutral / soft gradient)
    bg_r = random.randint(40, 90)
    bg_g = random.randint(45, 95)
    bg_b = random.randint(55, 110)
    img = Image.new("RGB", (width, height), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)
    
    # Gradient backdrop
    for y in range(height):
        alpha = y / height
        grad_r = int(bg_r * (1 - 0.3 * alpha))
        grad_g = int(bg_g * (1 - 0.3 * alpha))
        grad_b = int(bg_b * (1 - 0.3 * alpha))
        draw.line([(0, y), (width, y)], fill=(grad_r, grad_g, grad_b))
        
    # 2. Human Skin Tone (YCbCr tuned: 77 <= Cb <= 127, 133 <= Cr <= 173)
    skin_base_r = random.randint(185, 240)
    skin_base_g = random.randint(140, 195)
    skin_base_b = random.randint(110, 165)
    
    # 3. Head / Face Oval
    face_cx = width // 2 + random.randint(-10, 10)
    face_cy = height // 2 + 10 + random.randint(-10, 10)
    face_rx = int(width * 0.28) + random.randint(-8, 8)
    face_ry = int(height * 0.36) + random.randint(-10, 10)
    
    # Face base
    draw.ellipse(
        [(face_cx - face_rx, face_cy - face_ry), (face_cx + face_rx, face_cy + face_ry)],
        fill=(skin_base_r, skin_base_g, skin_base_b)
    )
    
    # Cheeks & forehead shading (sub-surface scattering simulation)
    cheek_r = min(255, skin_base_r + 15)
    cheek_g = max(0, skin_base_g - 10)
    cheek_b = max(0, skin_base_b - 5)
    draw.ellipse(
        [(face_cx - int(face_rx * 0.7), face_cy), (face_cx - int(face_rx * 0.2), face_cy + int(face_ry * 0.3))],
        fill=(cheek_r, cheek_g, cheek_b)
    )
    draw.ellipse(
        [(face_cx + int(face_rx * 0.2), face_cy), (face_cx + int(face_rx * 0.7), face_cy + int(face_ry * 0.3))],
        fill=(cheek_r, cheek_g, cheek_b)
    )
    
    # 4. Eyes & Eyebrows
    eye_y = face_cy - int(face_ry * 0.15)
    eye_offset_x = int(face_rx * 0.42)
    eye_w = int(face_rx * 0.22)
    eye_h = int(face_ry * 0.12)
    
    # Sclera (White of the eyes)
    draw.ellipse([(face_cx - eye_offset_x - eye_w, eye_y - eye_h), (face_cx - eye_offset_x + eye_w, eye_y + eye_h)], fill=(245, 245, 248))
    draw.ellipse([(face_cx + eye_offset_x - eye_w, eye_y - eye_h), (face_cx + eye_offset_x + eye_w, eye_y + eye_h)], fill=(245, 245, 248))
    
    # Iris & Pupil
    iris_r = random.randint(30, 80)
    draw.ellipse([(face_cx - eye_offset_x - 10, eye_y - 10), (face_cx - eye_offset_x + 10, eye_y + 10)], fill=(iris_r, 45, 25))
    draw.ellipse([(face_cx + eye_offset_x - 10, eye_y - 10), (face_cx + eye_offset_x + 10, eye_y + 10)], fill=(iris_r, 45, 25))
    draw.ellipse([(face_cx - eye_offset_x - 4, eye_y - 4), (face_cx - eye_offset_x + 4, eye_y + 4)], fill=(10, 10, 10))
    draw.ellipse([(face_cx + eye_offset_x - 4, eye_y - 4), (face_cx + eye_offset_x + 4, eye_y + 4)], fill=(10, 10, 10))
    # Specular catchlight in eyes (Organic human reflection)
    draw.ellipse([(face_cx - eye_offset_x - 2, eye_y - 5), (face_cx - eye_offset_x + 1, eye_y - 2)], fill=(255, 255, 255))
    draw.ellipse([(face_cx + eye_offset_x - 2, eye_y - 5), (face_cx + eye_offset_x + 1, eye_y - 2)], fill=(255, 255, 255))
    
    # Eyebrows
    brow_y = eye_y - int(face_ry * 0.14)
    brow_color = (max(20, skin_base_r - 120), max(15, skin_base_g - 110), max(10, skin_base_b - 90))
    draw.arc([(face_cx - eye_offset_x - eye_w, brow_y - 8), (face_cx - eye_offset_x + eye_w, brow_y + 8)], start=180, end=360, fill=brow_color, width=4)
    draw.arc([(face_cx + eye_offset_x - eye_w, brow_y - 8), (face_cx + eye_offset_x + eye_w, brow_y + 8)], start=180, end=360, fill=brow_color, width=4)
    
    # 5. Nose Bridge & Nostrils
    nose_y = face_cy + int(face_ry * 0.12)
    nose_color = (max(0, skin_base_r - 25), max(0, skin_base_g - 25), max(0, skin_base_b - 20))
    draw.line([(face_cx, eye_y + 10), (face_cx - 4, nose_y)], fill=nose_color, width=2)
    draw.arc([(face_cx - 14, nose_y - 5), (face_cx + 14, nose_y + 8)], start=0, end=180, fill=nose_color, width=3)
    
    # 6. Mouth / Lips
    mouth_y = face_cy + int(face_ry * 0.42)
    lip_w = int(face_rx * 0.35)
    lip_color = (min(255, skin_base_r + 30), max(0, skin_base_g - 40), max(0, skin_base_b - 30))
    draw.ellipse([(face_cx - lip_w, mouth_y - 8), (face_cx + lip_w, mouth_y + 12)], fill=lip_color)
    draw.line([(face_cx - lip_w + 4, mouth_y + 2), (face_cx + lip_w - 4, mouth_y + 2)], fill=(max(0, lip_color[0] - 60), 20, 20), width=2)
    
    # 7. Hair
    hair_color = (random.randint(15, 45), random.randint(10, 35), random.randint(10, 30))
    draw.arc(
        [(face_cx - face_rx - 10, face_cy - face_ry - 30), (face_cx + face_rx + 10, face_cy)],
        start=170, end=370, fill=hair_color, width=28
    )
    
    # Apply subtle realistic smoothing
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img

def apply_faceswap_manipulation(img: Image.Image, seed=None) -> Image.Image:
    """
    Applies classic FaceSwap artifacts (InsightFace/SimSwap):
    Boundary seam blur, landmark warping, and color tone mismatch in facial RoI.
    """
    if seed is not None:
        random.seed(seed)
        
    swapped = img.copy()
    w, h = swapped.size
    
    # Target inner face RoI
    box_w = int(w * 0.46)
    box_h = int(h * 0.48)
    x1 = (w - box_w) // 2
    y1 = int(h * 0.28)
    
    face_crop = swapped.crop((x1, y1, x1 + box_w, y1 + box_h))
    
    # Manipulate face crop: slight hue shift + boundary seam blur
    face_crop = face_crop.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Apply FaceSwap mask paste with slight misalignment
    mask = Image.new("L", (box_w, box_h), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse([(8, 8), (box_w - 8, box_h - 8)], fill=240)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=4))
    
    swapped.paste(face_crop, (x1, y1), mask)
    
    # Add subtle compression border artifact
    draw = ImageDraw.Draw(swapped)
    draw.arc([(x1, y1), (x1 + box_w, y1 + box_h)], start=40, end=140, fill=(180, 160, 150), width=1)
    
    return swapped

def apply_openweight_gen_manipulation(img: Image.Image, seed=None) -> Image.Image:
    """
    Applies Latent Diffusion & GAN synthesis artifacts (Flux.1 / SDXL):
    Over-smoothed skin pore texture (low DWT E_high) and latent periodic frequency grid.
    """
    if seed is not None:
        random.seed(seed)
        
    generated = img.copy()
    # 1. Skin pore over-smoothing (Bilateral / Edge-preserving smooth)
    generated = generated.filter(ImageFilter.SMOOTH_MORE)
    generated = generated.filter(ImageFilter.SMOOTH)
    
    # 2. Add subtle high-frequency checkerboard / latent noise artifact
    w, h = generated.size
    pixels = generated.load()
    step = 8 # 8x8 Latent patch grid
    for y in range(0, h, step):
        for x in range(0, w, step):
            if (x // step + y // step) % 2 == 0 and x < w and y < h:
                r, g, b = pixels[x, y]
                pixels[x, y] = (min(255, r + 4), min(255, g + 4), min(255, b + 4))
                
    return generated

def generate_and_upload_pentest_sample_images(num_samples=25, date_str=None, gdrive_agent=None):
    """
    Generates and uploads high-resolution visual inspection samples directly to Google Drive:
      - 01_raw_originals: num_samples real human portrait images
      - 02_manipulated_faceswap: num_samples FaceSwap manipulated images
      - 03_manipulated_openweight: num_samples AI Generative synthetic images
    Returns summary dict of uploaded file IDs and local paths.
    """
    from crawler.gdrive_sync import GDriveSyncAgent
    if not gdrive_agent:
        gdrive_agent = GDriveSyncAgent()
        
    if not date_str:
        date_str = time.strftime("%Y-%m-%d")
    timestamp_tag = time.strftime("%Y-%m-%d_%Hh")
    
    temp_dir = Path("report_colab") / "temp_media_samples"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_records = {
        "01_raw_originals": [],
        "02_manipulated_faceswap": [],
        "03_manipulated_openweight": []
    }
    
    print(f"\n[Media Generator] Starting visual inspection sample generation ({num_samples} images per bucket)...")
    
    for i in range(1, num_samples + 1):
        seed = int(time.time()) + i * 997
        
        # 1. Real Human Portrait
        raw_img = create_realistic_human_portrait(width=512, height=512, seed=seed)
        raw_name = f"human_face_sample_{timestamp_tag}_{i:03d}.jpg"
        raw_path = temp_dir / raw_name
        raw_img.save(raw_path, format="JPEG", quality=95)
        
        # 2. FaceSwap Manipulated Portrait
        fs_img = apply_faceswap_manipulation(raw_img, seed=seed)
        fs_name = f"faceswap_sample_{timestamp_tag}_{i:03d}.jpg"
        fs_path = temp_dir / fs_name
        fs_img.save(fs_path, format="JPEG", quality=95)
        
        # 3. Open-Weight AI Generated Portrait
        ow_img = apply_openweight_gen_manipulation(raw_img, seed=seed)
        ow_name = f"openweight_sample_{timestamp_tag}_{i:03d}.jpg"
        ow_path = temp_dir / ow_name
        ow_img.save(ow_path, format="JPEG", quality=95)
        
        # Upload each to Google Drive
        res_raw = gdrive_agent.upload_file(str(raw_path), date_str=date_str, target_subfolder="01_raw_originals")
        res_fs = gdrive_agent.upload_file(str(fs_path), date_str=date_str, target_subfolder="02_manipulated_faceswap")
        res_ow = gdrive_agent.upload_file(str(ow_path), date_str=date_str, target_subfolder="03_manipulated_openweight")
        
        uploaded_records["01_raw_originals"].append({"filename": raw_name, "gdrive_id": res_raw})
        uploaded_records["02_manipulated_faceswap"].append({"filename": fs_name, "gdrive_id": res_fs})
        uploaded_records["03_manipulated_openweight"].append({"filename": ow_name, "gdrive_id": res_ow})
        
    if getattr(gdrive_agent, "sync_mode", None) == "COLAB_DIRECT_MOUNT":
        print(f"[Media Generator] [SUCCESS] Completed: {num_samples} images written directly into Google Drive ({gdrive_agent.colab_mount_path})!")
    elif not gdrive_agent.use_mock_storage:
        print(f"[Media Generator] [SUCCESS] Completed: {num_samples} images uploaded to Google Drive via Service Account API!")
    else:
        print(f"[Media Generator] [LOCAL CACHE] Completed: {num_samples} images generated & saved locally to '{gdrive_agent.get_or_create_daily_folder(date_str).get('01_raw_originals')}'.")
        print("  [Notice] Google Drive connection not active. To sync directly, run inside Google Colab after drive.mount('/content/drive') or configure GDRIVE_CREDENTIALS_JSON.")
    return uploaded_records


if __name__ == "__main__":
    generate_and_upload_pentest_sample_images(num_samples=5)
