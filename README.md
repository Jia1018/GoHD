# GoHD

Official inference code for **GoHD: Gaze-oriented and Highly Disentangled
Portrait Animation with Rhythmic Poses and Realistic Expressions**
(AAAI 2025).

Given a single portrait image and a driving audio clip, GoHD generates a
talking-head video with synchronised lips, rhythmic head pose, natural eye
movements and expressive but identity-preserving motion. Short reference
videos can optionally be used as the source of head pose, gaze, or
expression motion.

> ⚠️ This repository contains **inference code only**. The training pipeline
> is not released.

---

## 1. Installation

We tested on Ubuntu 20.04 with CUDA 11.8 and Python 3.9 / 3.10.

```bash
# 1) Create an environment (conda or venv, your choice)
conda create -n gohd python=3.10 -y
conda activate gohd

# 2) Install PyTorch matching your CUDA version.
#    Example for CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 3) System libraries (ffmpeg is required by moviepy + imageio)
sudo apt-get install -y ffmpeg cmake build-essential

# 4) Python dependencies
pip install -r requirements.txt
```

The `dlib` wheel sometimes needs `cmake` and `build-essential`. If you hit a
compile error, install `cmake` first.

### Pretrained weights

Place the model weights under `checkpoints/` and the Basel Face Model files
under `BFM/`.

```bash
# Pull the GoHD checkpoints from the project Google Drive.
pip install gdown
python scripts/download_checkpoints.py
```

For the BFM files (released under a research-only licence), follow the
instructions in [`BFM/README.md`](BFM/README.md).

After both steps the layout should be:

```
GoHD/
├── checkpoints/
│   ├── FaceAnimator_GoHD.pt
│   ├── audio2pose.pt
│   ├── audio2gaze.pt
│   ├── audio2exp.pth
│   └── L2CSNet_gaze360.pkl
├── Deep3DFaceRecon_pytorch/checkpoints/model_name/
│   └── epoch_20.pth
├── face_parsing/
│   └── 79999_iter.pth
└── BFM/
    ├── 01_MorphableModel.mat
    ├── BFM_model_front.mat
    └── ...
```

---

## 2. Quick start

A ready-to-run example using bundled assets:

```bash
bash scripts/run.sh
```

This is equivalent to:

```bash
python inference.py \
    --source_path        examples/source/anne_crop.jpg \
    --audio_path         examples/audios/sing.wav \
    --source_pose_video  examples/driving_videos/15_WRA_SteveDaines_000.mp4 \
    --source_gaze_video  examples/driving_videos/0_WRA_RoyBlunt_000.mp4 \
    --source_exp_video   examples/driving_videos/15_WRA_SteveDaines_000.mp4 \
    --facemodel_path     checkpoints/FaceAnimator_GoHD.pt \
    --diffmodel_path     checkpoints/audio2pose.pt \
    --gazemodel_path     checkpoints/audio2gaze.pt \
    --expmodel_path      checkpoints/audio2exp.pth \
    --save_folder        results
```

The result is written to `results/<source-stem>_<audio-stem>_voice.mp4`.

### Examples shipped with the repo

| Folder                          | Items                                                       |
| ------------------------------- | ----------------------------------------------------------- |
| `examples/source/`              | portrait images (real face crops + paintings)               |
| `examples/audios/`              | short speech / singing clips                                |
| `examples/driving_videos/`      | reference videos for head pose / gaze / expression          |

Try mixing and matching them. For instance:

```bash
python inference.py \
    --source_path       examples/source/audrey_crop.jpg \
    --audio_path        examples/audios/monalisa.mp3 \
    --source_pose_video examples/driving_videos/0_WRA_RoyBlunt_000.mp4 \
    --facemodel_path    checkpoints/FaceAnimator_GoHD.pt \
    --diffmodel_path    checkpoints/audio2pose.pt \
    --gazemodel_path    checkpoints/audio2gaze.pt \
    --expmodel_path     checkpoints/audio2exp.pth
```

---

## 3. CLI overview

Run `python inference.py --help` for the full list. The most useful options:

| Flag                                                  | Default            | Meaning                                                                                  |
| ----------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------- |
| `--source_path`                                       | *required*         | Portrait image (any resolution; will be cropped + resized to 256 × 256).                 |
| `--audio_path`                                        | *required*         | Driving audio (`.wav` or `.mp3`).                                                        |
| `--source_pose_video`                                 | `None`             | Reference video used to extract head-pose trajectory.                                    |
| `--source_gaze_video`                                 | `None`             | Reference video for gaze trajectory.                                                     |
| `--source_exp_video`                                  | `None`             | Reference video for expression motion.                                                   |
| `--use_generated_pose / _exp / _gaze`                 | `False`            | If `True`, sample the corresponding modality from audio with the diffusion / LSTM heads. |
| `--freeze_gaze`                                       | `0`                | `0`: forward-looking, `1`: look left, `2`: look right, `3`: free gaze.                   |
| `--pose_scale` / `--exp_scale`                        | `1.2` / `1.5`      | Multiply driving-pose / expression magnitudes.                                           |
| `--crop_input_img`                                    | `True`             | Run a dlib face detector to crop the source portrait before generation.                  |
| `--max_inference_length`                              | `16`               | Frames processed per diffusion window. Increase for longer videos.                       |
| `--save_folder`                                       | `./results`        | Output directory.                                                                        |

---

## 4. Repository layout

```
GoHD/
├── inference.py                 # entry point (was run_demo_withaudio_*.py)
├── audio.py / params.py / ...   # small support modules
├── networks/                    # face animator + audio2{pose,gaze,exp}
├── face_parsing/                # BiSeNet face parsing
├── sync_batchnorm/              # synced BN for multi-GPU inference
├── Deep3DFaceRecon_pytorch/     # 3DMM coefficient detector
├── BFM/                         # Basel Face Model assets (you supply)
├── checkpoints/                 # pretrained weights (you download)
├── examples/                    # sample sources / audios / driving videos
└── scripts/
    ├── run.sh
    └── download_checkpoints.py
```

The original drop included training scripts, dataset preprocessing, and
extensive ablation utilities; this release strips everything outside the
inference path so you can reproduce the demo results with the fewest moving
parts.

---

## 5. Citation

```bibtex
@inproceedings{zhou2025gohd,
    title     = {GoHD: Gaze-oriented and Highly Disentangled Portrait Animation
                 with Rhythmic Poses and Realistic Expressions},
    author    = {Zhou, Ziqi and Quan, Weize and Shi, Hailin and Li, Wei and
                 Wang, Lili and Yan, Dong-Ming},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence
                 (AAAI)},
    year      = {2025}
}
```

## 6. Acknowledgements

GoHD builds on the work of:

* [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
  for the 3DMM coefficient detector.
* [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) for gaze estimation
  (Gaze360).
* [LIA](https://github.com/wyhsirius/LIA) for the latent-image-animator
  baseline.

The bundled `Deep3DFaceRecon_pytorch/` is a slightly modified copy; refer to
its `LICENSE` for redistribution terms.
