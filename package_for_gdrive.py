"""
Utility script to package the DeepFake-Detection-Agent codebase into a clean, lightweight zip file
specifically for Option C deployment on Google Drive (/content/drive/MyDrive/Thread_Data/).
Excludes large binaries, virtual environments, and git cache.
"""
import os
import sys
import zipfile
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent

EXCLUDE_DIRS = {
    ".git", "venv", ".venv", "env", "__pycache__", ".pytest_cache",
    ".cache", "colab_datasets", "benchmark_data", "uploads"
}

EXCLUDE_EXTENSIONS = {
    ".pt", ".pth", ".bin", ".safetensors", ".h5", ".onnx", ".db"
}

EXCLUDE_FILES = {
    "ResembleAI-Q1-Deepfake-Threats.pdf", "DeepFake-Detection-Agent.zip"
}

def package_codebase(output_zip_path=None):
    if not output_zip_path:
        output_zip_path = PROJECT_ROOT / "DeepFake-Detection-Agent.zip"
        
    output_zip_path = Path(output_zip_path)
    print(f"📦 Packaging codebase into: {output_zip_path}...")
    
    file_count = 0
    total_uncompressed_bytes = 0
    
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            rel_root = Path(root).relative_to(PROJECT_ROOT)
            
            for f in files:
                if f in EXCLUDE_FILES or Path(f).suffix in EXCLUDE_EXTENSIONS or f.startswith("~$") or f.startswith(".~"):
                    continue
                file_path = Path(root) / f
                arcname = (rel_root / f).as_posix()
                
                zipf.write(file_path, arcname=arcname)
                file_count += 1
                total_uncompressed_bytes += file_path.stat().st_size
                
    zip_size_mb = round(output_zip_path.stat().st_size / (1024 * 1024), 2)
    uncomp_mb = round(total_uncompressed_bytes / (1024 * 1024), 2)
    
    print("=" * 65)
    print(f"🎉 Codebase successfully packaged for Option C (Google Drive)!")
    print(f"  📄 Total Files: {file_count}")
    print(f"  📊 Uncompressed Size: {uncomp_mb} MB")
    print(f"  🗜️ Compressed Zip Size: {zip_size_mb} MB")
    print(f"  💾 Output Path: {output_zip_path}")
    print("=" * 65)
    print("\n👉 CÁCH SỬ DỤNG CHO OPTION C:")
    print("  1. Copy tệp 'DeepFake-Detection-Agent.zip' vào Google Drive tại:")
    print("     MyDrive/Thread_Data/DeepFake-Detection-Agent.zip")
    print("     (hoặc giải nén thành thư mục MyDrive/Thread_Data/DeepFake-Detection-Agent/)")
    print("  2. Chạy Google Colab Scheduled Runner, Colab sẽ tự động giải nén và chạy 100% không cần GitHub!")
    print("=" * 65 + "\n")
    return output_zip_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Package codebase for Google Drive")
    parser.add_argument("--output", "-o", type=str, default=None, help="Target output zip file path")
    args = parser.parse_args()
    
    package_codebase(output_zip_path=args.output)
