# Pretrained Checkpoints

Run `python scripts/download_checkpoints.py` from the repository root to
populate this directory automatically. The expected layout is:

```
checkpoints/
├── FaceAnimator_GoHD.pt        # main animator (warp + render)
├── audio2pose.pt               # diffusion head-pose generator
├── audio2gaze.pt               # diffusion gaze generator
├── audio2exp.pth               # audio-conditioned expression generator
└── L2CSNet_gaze360.pkl         # L2CS gaze estimator (pretrained on Gaze360)

# The Deep3DFaceRecon weights live separately because of how its option
# parser builds the checkpoint path:
Deep3DFaceRecon_pytorch/checkpoints/model_name/epoch_20.pth
```

You can also download the files manually from the original GoHD release on
Google Drive:

  <https://drive.google.com/drive/folders/1S2RxB8pUsO-lM4iRi6rO7EPdDpM85h0k>

(Files live under `checkpoints/` and `Deep3DFaceRecon_pytorch/checkpoints/`.)

The face-parsing model `face_parsing/79999_iter.pth` is downloaded by the
same helper script — it has to live under `face_parsing/`, not here.
