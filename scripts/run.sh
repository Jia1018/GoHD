#!/usr/bin/env bash
# Single-image talking-head generation driven by an audio clip,
# borrowing head pose / gaze from short reference videos.
set -e

cd "$(dirname "$0")/.."

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
    --generate_exp_from_latent 1 \
    --freeze_gaze        0 \
    --save_folder        results \
    --max_inference_length 5000
