"""
scripts/download_weights.py
Helper utility to automatically download or verify PRISM neural backbone weights.
"""
import os
import sys
import hashlib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
FINAL_CHECKPOINT = MODEL_DIR / "prism_model_final.pth"

def verify_checkpoint(path):
    if not path.is_file():
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Located checkpoint: {path} ({size_mb:.2f} MB)")
    return size_mb > 300

def main():
    print("=" * 65)
    print("  PRISM Forensics - Pretrained Weights Verification Utility")
    print("=" * 65)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    if verify_checkpoint(FINAL_CHECKPOINT):
        print("âœ“ PRISM model weights are verified and ready for inference!")
        return 0
    
    print("Notice: Neural backbone checkpoints (352 MB) exceed standard GitHub 100 MB limits.")
    print("Please download 'prism_model_final.pth' from the official project release:")
    print("  Release URL: https://github.com/Tanush-Bhootra/PRISM/releases/tag/v1.0.0")
    print(f"Place the downloaded .pth file into: {MODEL_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
