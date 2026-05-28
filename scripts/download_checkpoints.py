"""Pull the pretrained weights for inference from the project Google Drive.

Run from the repository root:
    python scripts/download_checkpoints.py
"""

import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    sys.exit("pip install gdown   # required for downloading checkpoints")


ROOT = Path(__file__).resolve().parents[1]

# (google_drive_id, relative_path)
WEIGHTS = [
    # face animator + audio2{pose,gaze,exp} + gaze estimator (required)
    ("17eMdl3zTgYhg7JlYCaRgJKVmp3TOAZQ9", "checkpoints/FaceAnimator_GoHD.pt"),
    ("1QVamgnISBRjFdJfUZVeq1hnzK45eLGak", "checkpoints/audio2pose.pt"),
    ("1P7QZepvq3s9PKM6LLVYVauXyTrZFunzL", "checkpoints/audio2gaze.pt"),
    ("1kp27iFDpabZiWCIJVxOyc6Xf-vjGY4ub", "checkpoints/audio2exp.pth"),
    ("16CkA_WBleBmXIp3hQOJSgYBkaJ9SYF1E", "checkpoints/L2CSNet_gaze360.pkl"),
    # face parsing model used inside the renderer
    ("1XIQEzJvLPDlJQimgz4MrkFOkwc3OA-0s", "face_parsing/79999_iter.pth"),
    # Deep3DFaceRecon weights (face reconstruction model used by CoeffDetector)
    # Default checkpoints_dir is ./Deep3DFaceRecon_pytorch/checkpoints and the
    # default model name is "model_name" — keep this path or override --name.
    ("1DFASeWXn1IwgsI4CNzALtDNdfAWeuE9m",
     "Deep3DFaceRecon_pytorch/checkpoints/model_name/epoch_20.pth"),
]


def main():
    for fid, rel in WEIGHTS:
        out = ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            print(f"  already have {rel}")
            continue
        print(f"  downloading {rel}")
        gdown.download(id=fid, output=str(out), quiet=False)
    print("\nDone. See README.md for BFM files (separate license).")


if __name__ == "__main__":
    main()
