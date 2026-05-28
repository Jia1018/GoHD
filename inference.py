"""GoHD inference: audio-driven talking-head generation from a single portrait.

Usage example (see scripts/run.sh):
    python inference.py \
        --source_path examples/source/anne_crop.jpg \
        --audio_path  examples/audios/sing.wav \
        --source_pose_video examples/driving_videos/0_WRA_RoyBlunt_000.mp4 \
        --diffmodel_path checkpoints/audio2pose.pt \
        --gazemodel_path checkpoints/audio2gaze.pt \
        --expmodel_path  checkpoints/audio2exp.pth \
        --facemodel_path checkpoints/FaceAnimator_GoHD.pt \
        --save_folder    results
"""

import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import dlib
import imageio
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import face_alignment
from face_detection import RetinaFace

import audio
from config import AudioConfig
from merge import merge
from params import params as base_params

from networks.audio2exp import SimpleWrapperV2
from networks.generator import Generator, Generator_Semantic
from networks.models.nn import LDA
from networks.utils import draw_annotation_box, draw_gaze

from Deep3DFaceRecon_pytorch.coeff_detector import CoeffDetector, getArch
from Deep3DFaceRecon_pytorch.extract_kp_videos import KeypointExtractor
from Deep3DFaceRecon_pytorch.options.inference_options import InferenceOptions


# ---------------------------------------------------------------------------
# Audio helpers (inlined from the original dataset.py)
# ---------------------------------------------------------------------------

def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode="constant", constant_values=0)
    return wav


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps
    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)
    return audio_length, num_frames


def read_audio(wavpath):
    """Return (per-frame syncnet mel chunks, full mel spectrogram)."""
    hop_size = 160
    fps = 25
    sample_rate = hop_size * fps * 4  # 4 mel bins per frame
    au = AudioConfig.AudioConfig(num_frames_per_clip=5, hop_size=hop_size,
                                 frame_rate=fps, sample_rate=sample_rate)
    wav = au.read_audio(wavpath)
    spectrogram = au.audio_to_spectrogram(wav)

    wav = audio.load_wav(wavpath, sample_rate)
    wav_length, num_frames = parse_audio_length(len(wav), sample_rate, fps)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spectrogram = orig_mel.copy()

    indiv_mels = []
    for i in range(num_frames):
        start_frame_num = i - 2
        start_idx = int(80.0 * (start_frame_num / float(fps)))
        end_idx = start_idx + 16
        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
        indiv_mels.append(spectrogram[seq, :].T)
    return np.asarray(indiv_mels), spectrogram


# ---------------------------------------------------------------------------
# Gaze helpers
# ---------------------------------------------------------------------------

_GAZE_TRANSFORMS = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_gaze_params(model, detector, img):
    bbox, _, score = detector(img)[0]
    assert score > 0.95, "low-confidence face detection"
    lx, ly, rx, ry = (int(v) for v in bbox)
    img = img[ly:ry, lx:rx, :]
    img = cv2.resize(img, (224, 224))
    img_t = _GAZE_TRANSFORMS(Image.fromarray(img))
    img_t = Variable(img_t).cuda().unsqueeze(0)

    softmax = nn.Softmax(dim=1)
    gaze_pitch, gaze_yaw = model(img_t)
    pitch_predicted = softmax(gaze_pitch)
    yaw_predicted = softmax(gaze_yaw)

    idx_tensor = torch.FloatTensor(list(range(90))).cuda()
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0
    return pitch_predicted, yaw_predicted


# ---------------------------------------------------------------------------
# Diffusion samplers
# ---------------------------------------------------------------------------

_MODELS = {}  # path -> loaded model


def diff_synthesize_pose(model_dir, ctrl, global_cond, gaze=False):
    base_params.in_channels = 2 if gaze else 6
    base_params.out_channels = 2 if gaze else 6
    if global_cond is not None:
        base_params.style_dim = global_cond.shape[1]

    if model_dir not in _MODELS:
        ckpt_path = f"{model_dir}/weights.pt" if os.path.exists(f"{model_dir}/weights.pt") else model_dir
        checkpoint = torch.load(ckpt_path)
        model = LDA(base_params, "tisa").to(ctrl.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        _MODELS[model_dir] = model
    model = _MODELS[model_dir]

    with torch.no_grad():
        noise_schedule = torch.linspace(1e-4, 0.05, 150).to(ctrl.device)
        training_noise_schedule = noise_schedule
        inference_noise_schedule = training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = torch.cumprod(talpha, dim=0)
        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / \
                              (talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                    T.append(t + twiddle)
                    break

        if ctrl.dim() == 2:
            ctrl = ctrl.unsqueeze(0)
            global_cond = global_cond.unsqueeze(0)
        poses = torch.randn(ctrl.shape[0], ctrl.shape[1], base_params.out_channels,
                            device=ctrl.device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            poses = c1 * (poses - c2 * model(poses, ctrl, global_cond,
                                             T[n].unsqueeze(-1)).squeeze(1))
            if n > 0:
                noise = torch.randn_like(poses)
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                poses += sigma * noise
        poses = torch.clamp(poses, -1.0, 1.0)
    return poses.squeeze(0)


def get_exp_from_audio(audio_chunks, model_path, exp0, length, device=torch.device("cuda")):
    model = SimpleWrapperV2(blink_control=0).to(device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict_G"])
    model.eval()

    x = {
        "zb": None,
        "exp0": exp0.unsqueeze(0).to(device),
        "audio": audio_chunks.unsqueeze(0).to(device),
    }
    return model(x)


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------

def frame2audio_indexs(frame_inds, num_bins_per_frame):
    return frame_inds * num_bins_per_frame


def load_spectrogram(audio_ind, spectrogram, num_bins_per_frame):
    num_audio_bins = num_bins_per_frame
    mel_shape = spectrogram.shape
    if (audio_ind + num_audio_bins) <= mel_shape[0] and audio_ind >= 0:
        chunk = np.array(spectrogram[audio_ind:audio_ind + num_audio_bins, :]).astype("float32")
    elif audio_ind > 0:
        chunk = np.array(spectrogram[audio_ind:audio_ind + num_audio_bins, :]).astype("float32")
    else:
        chunk = np.zeros((num_audio_bins, mel_shape[1])).astype("float32")
    return torch.from_numpy(chunk).unsqueeze(0)


def _append_seq(buf, new):
    if buf is None:
        return new.clone()
    return torch.cat((buf[:-1], new), 0)


def seq_smoothing(seq):
    for i in range(2, len(seq) - 2):
        seq[i] = torch.mean(seq[i - 2:i + 3], dim=0)
    return seq


def concat_source_sequence(input_seq, target_length):
    num_repeats = target_length // len(input_seq)
    chunks = []
    for i in range(num_repeats):
        chunks.append(torch.flip(input_seq, [0]) if i % 2 == 1 else input_seq)
    return torch.cat(chunks, dim=0)[:target_length]


def get_target_seqs(args, num_bins_per_frame, gaze0, pose0, exp0,
                    pose_target, exp_target, gaze_target):
    aud_syncnet_target, aud_target = read_audio(args.audio_path)
    aud_target = np.concatenate([aud_target, aud_target], axis=0)

    base_params.frame_rate = 25
    target_length = len(aud_syncnet_target)
    if pose_target is not None and target_length > len(pose_target):
        pose_target = concat_source_sequence(pose_target, target_length)
    if exp_target is not None and target_length > len(exp_target):
        exp_target = concat_source_sequence(exp_target, target_length)
    if gaze_target is not None and target_length > len(gaze_target):
        gaze_target = concat_source_sequence(gaze_target, target_length)

    target_frame_inds = [
        np.arange(i, min(i + args.max_inference_length, target_length))
        for i in range(0, target_length, args.max_inference_length - 1)
    ]

    all_pose_target = pose0.unsqueeze(0)
    all_exp_target = None if args.use_generated_exp else exp0.unsqueeze(0)
    all_gaze_target = gaze0.unsqueeze(0)

    for frame_inds in target_frame_inds:
        infer_length = len(frame_inds)
        audio_inds = frame2audio_indexs(frame_inds, num_bins_per_frame)
        aud_target_ = torch.cat(
            [load_spectrogram(i, aud_target, num_bins_per_frame) for i in audio_inds],
            dim=0,
        )
        seq_len = aud_target_.shape[0]

        if pose_target is not None and not args.use_generated_pose:
            pose_target_ = pose_target[frame_inds]
        else:
            pose_target_ = torch.zeros((len(frame_inds), 6))
        if exp_target is not None and not args.use_generated_exp:
            exp_target_ = exp_target[frame_inds]
        else:
            exp_target_ = torch.zeros((len(frame_inds), 64))
        if gaze_target is not None and not args.use_generated_gaze:
            gaze_target_ = gaze_target[frame_inds]
        else:
            gaze_target_ = torch.zeros((len(frame_inds), 2))

        base_params.seq_len = seq_len

        if args.use_generated_pose:
            mel_frs = aud_target_.view(seq_len, -1).cuda()[1:]
            if args.pose_withpose0:
                global_cond = pose0.unsqueeze(0).repeat(mel_frs.shape[0], 1).cuda()
            else:
                global_cond = None
            target_3dmm = diff_synthesize_pose(args.diffmodel_path, mel_frs, global_cond)
            pose_target_ = target_3dmm[:, :6]
            if args.use_residual_headpose:
                pose_target_ += pose0.unsqueeze(0)
            pose0 = pose_target_[-1]
        else:
            pose_target_ = pose_target_[1:]

        if args.use_generated_exp:
            aud_syncnet_target_ = torch.tensor(
                aud_syncnet_target[frame_inds], dtype=torch.float, requires_grad=False
            ).cuda()
            pred = get_exp_from_audio(aud_syncnet_target_, args.expmodel_path,
                                      exp0, infer_length)
            if args.use_residual_exp:
                exp_target_ = pred["exp_pred"].squeeze(0)
            else:
                exp_target_ = pred["exp_motion_pred"].squeeze(0)
            exp_target_[0] = (exp0 + exp_target_[0]) / 2
            exp0 = exp_target_[-1]
        else:
            exp_target_ = exp_target_[1:]

        if args.use_generated_gaze:
            mel_frs = aud_target_.view(seq_len, -1).cuda()[1:]
            global_cond = gaze0.unsqueeze(0).repeat(mel_frs.shape[0], 1).cuda()
            target_3dmm = diff_synthesize_pose(args.gazemodel_path, mel_frs,
                                               global_cond, gaze=True)
            gaze_target_ = target_3dmm[:, :2]
            if args.use_residual_gaze:
                gaze_target_ += gaze0.unsqueeze(0)
            gaze0 = gaze_target_[-1]
        else:
            gaze_target_ = gaze_target_[1:]

        if args.generate_from0:
            exp0 = torch.zeros(64).cuda()

        all_pose_target = _append_seq(all_pose_target, pose_target_)
        if all_exp_target is None:
            all_exp_target = exp_target_[:-1]
        else:
            all_exp_target = _append_seq(all_exp_target, exp_target_)
        all_gaze_target = _append_seq(all_gaze_target, gaze_target_)

    if args.pose_smoothing:
        all_pose_target = seq_smoothing(all_pose_target)

    pose_frames = []
    for i in range(all_pose_target.shape[0]):
        pose_frame = torch.tensor(draw_annotation_box(all_pose_target[i].cpu().numpy()),
                                  dtype=torch.float, requires_grad=False).unsqueeze(0)
        pose_frame = (pose_frame / 255.0 - 0.5) * 2
        pose_frames.append(pose_frame)

    gaze_frames = []
    for i in range(all_gaze_target.shape[0]):
        gaze_frame = torch.tensor(draw_gaze(all_gaze_target[i].cpu().numpy()),
                                  dtype=torch.float, requires_grad=False).unsqueeze(0)
        gaze_frame = (gaze_frame / 255.0 - 0.5) * 2
        gaze_frames.append(gaze_frame)

    return (
        torch.stack(pose_frames).unsqueeze(0),
        all_pose_target.unsqueeze(0),
        all_exp_target.unsqueeze(0),
        all_gaze_target.unsqueeze(0),
        Path(args.audio_path).stem,
        args.audio_path,
    )


# ---------------------------------------------------------------------------
# Image / video preprocessing
# ---------------------------------------------------------------------------

def load_image(filename, size, crop, reverse=False):
    img = Image.open(filename).convert("RGB")
    img = np.asarray(img)
    if crop is not None:
        lx, ly, rx, ry = crop
        img = img[int(ly):int(ry), int(lx):int(rx), :]
        save_path = filename.replace(filename[-4:], "_crop" + filename[-4:])
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("Cropped input img saved to", save_path)
    if reverse:
        img = img[:, :, ::-1]
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1))
    return img / 255.0


def img_preprocessing(img_path, size, crop=None, reverse=False):
    img = load_image(img_path, size, crop, reverse)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return (img - 0.5) * 2.0


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit="sec")
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]["video_fps"]
    return (vid / 255.0 - 0.5) * 2.0, fps


def save_video(vid_target_recon, save_path, fps, reverse=False):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    if reverse:
        vid = torch.flip(vid, [-1])
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type("torch.ByteTensor")
    print("Frame rate:", fps)
    torchvision.io.write_video(save_path, vid[0], fps=fps)


def compute_aspect_preserved_bbox(bbox, increase_area, h, w):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
    width_increase = max(increase_area,
                         ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area,
                          ((1 + 2 * increase_area) * width - height) / (2 * height))
    left_t = int(left - width_increase * width)
    top_t = int(top - height_increase * height)
    right_t = int(right + width_increase * width)
    bot_t = int(bot + height_increase * height)
    left_oob = -min(0, left_t)
    right_oob = right - min(right_t, w)
    top_oob = -min(0, top_t)
    bot_oob = bot - min(bot_t, h)
    if max(left_oob, right_oob, top_oob, bot_oob) > 0:
        max_w = max(left_oob, right_oob)
        max_h = max(top_oob, bot_oob)
        if max_w > max_h:
            return left_t + max_w, top_t + max_w, right_t - max_w, bot_t - max_w
        return left_t + max_h, top_t + max_h, right_t - max_h, bot_t - max_h
    return (left_t, top_t, right_t, bot_t)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

class Demo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.params = base_params
        self.semantic_radius = 13
        model_path = args.facemodel_path

        # gaze estimator (L2CS) + retinaface for face detection
        self.gaze_model = getArch("ResNet50", 90)
        print("Loading gaze model:", "checkpoints/L2CSNet_gaze360.pkl")
        self.gaze_model.load_state_dict(torch.load("checkpoints/L2CSNet_gaze360.pkl"))
        self.gaze_model.cuda().eval()
        self.detector = RetinaFace(gpu_id=0)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D)

        # 3DMM coefficient detector + landmark extractor
        opt = InferenceOptions().parse()
        self.coeff_detector = CoeffDetector(opt)
        self.kp_extractor = KeypointExtractor()

        # Face animator
        print("Loading face animator:", model_path)
        if args.transfer_type != "vid2vid":
            in_channel, out_channel = 3, 3
            sem_dim = 72 if args.gen_with_gaze else 70
            self.gen = Generator_Semantic(
                args.size, sem_dim, args.latent_dim_style, args.latent_dim_motion,
                args.channel_multiplier, args.use_sft, in_channel, out_channel,
            ).cuda()
            weight = torch.load(model_path, map_location=lambda s, l: s)["gen_pose"]
        else:
            self.gen = Generator(args.size, args.latent_dim_style,
                                 args.latent_dim_motion, args.channel_multiplier).cuda()
            weight = torch.load(model_path, map_location=lambda s, l: s)["gen"]
        self.gen.load_state_dict(weight)
        self.gen.eval()

        # Source image -> initial pose / expression / gaze
        self.save_path = args.save_folder
        os.makedirs(self.save_path, exist_ok=True)
        num_bins_per_frame = int(self.params.sample_rate / self.params.hop_samples
                                 / self.params.frame_rate)

        im = Image.open(args.source_path).convert("RGB")
        img_np = np.asarray(im)
        pitch, yaw = get_gaze_params(self.gaze_model, self.detector, img_np)
        lm = self.kp_extractor.extract_keypoint(im)
        crop = self.crop_src_image(img_np) if args.crop_input_img else None

        predicted = self.coeff_detector(im, lm)["coeff_3dmm"][0]
        angles = torch.tensor(predicted[224:227], dtype=torch.float32).cuda()
        translation = torch.tensor(predicted[254:257], dtype=torch.float32).cuda()
        self.gaze0 = torch.tensor(np.array([pitch, yaw]), dtype=torch.float32).cuda()
        print("Initial gaze:", self.gaze0)
        self.pose0 = torch.cat((angles, translation), dim=-1)
        self.exp0 = torch.tensor(predicted[80:144], dtype=torch.float32).cuda()
        exp_start = torch.zeros(64).cuda() if args.generate_from0 else self.exp0

        # Optional driving videos -> pose / exp / gaze targets
        if args.driving_coeffs_file:
            assert args.use_generated_gaze is False
            driving_dict = loadmat(args.driving_coeffs_file)
            ref_semantic_target = torch.tensor(
                driving_dict["coeff_3dmm"][:, :70], dtype=torch.float32
            ).cuda()
            ref_pose_target = ref_semantic_target[:, 64:]
            ref_exp_target = ref_semantic_target[:, :64]

        pose_target = self.get_source_pose(args) \
            if args.source_pose_video and not args.use_generated_pose else None
        exp_target = self.get_source_exp(args) \
            if args.source_exp_video and not args.use_generated_exp else None
        if exp_target is not None:
            exp_target = torch.zeros_like(exp_target)
        gaze_target = self.get_source_gaze(args) \
            if args.source_gaze_video and not args.use_generated_gaze else None

        if args.transfer_type != "vid2vid":
            (self.pose_target, self.posevector_target, self.exp_target,
             self.gaze_target, self.driving_name, self.audio_path) = get_target_seqs(
                args, num_bins_per_frame, self.gaze0, self.pose0, exp_start,
                pose_target, exp_target, gaze_target,
            )
            if args.driving_coeffs_file:
                self.posevector_target = ref_pose_target[:self.pose_target.shape[1]].unsqueeze(0)
                self.exp_target = ref_exp_target[:self.exp_target.shape[1]].unsqueeze(0)
            self.pose_target = self.pose_target.cuda()
            self.posevector_target = (self.posevector_target * args.pose_scale).cuda()
            self.exp_target = (self.exp_target * args.exp_scale).cuda()
            self.gaze_target = self.gaze_target.cuda()

            if args.freeze_gaze == 0:  # forward / center
                fixed = torch.tensor([0.0, 0.0])
            elif args.freeze_gaze == 1:  # look "left"
                fixed = torch.tensor([0.2, 0.0])
            elif args.freeze_gaze == 2:  # look "right"
                fixed = torch.tensor([-0.7, 0.0])
            else:
                fixed = None
            if fixed is not None:
                self.gaze_target = fixed.cuda().view(1, 1, 2).repeat(
                    1, self.gaze_target.shape[1], 1
                )

            if args.gen_with_gaze:
                self.semantic_target = torch.cat(
                    (self.exp_target, self.posevector_target, self.gaze_target), dim=-1
                )
            else:
                self.semantic_target = torch.cat(
                    (self.exp_target, self.posevector_target), dim=-1
                )
        else:
            self.vid_target, self.fps = vid_preprocessing(args.driving_path)
            self.vid_target = self.vid_target.cuda()

        self.img_source = img_preprocessing(args.source_path, args.size, crop,
                                            args.reverse_img).cuda()
        self.crop = crop

    # ---------- driving-video readers ----------

    def get_source_pose(self, args):
        vid = imageio.get_reader(args.source_pose_video)
        poses = []
        for frame in vid:
            frame = Image.fromarray(frame).convert("RGB")
            lm = self.kp_extractor.extract_keypoint(frame)
            predicted = self.coeff_detector(frame, lm)["coeff_3dmm"][0]
            angles = torch.tensor(predicted[224:227], dtype=torch.float32).cuda()
            translation = torch.tensor(predicted[254:257], dtype=torch.float32).cuda()
            poses.append(torch.cat((angles, translation), dim=-1))
        return torch.stack(poses, dim=0)

    def get_source_gaze(self, args):
        vid = imageio.get_reader(args.source_gaze_video)
        gazes = []
        for frame in vid:
            frame = Image.fromarray(frame).convert("RGB")
            pitch, yaw = get_gaze_params(self.gaze_model, self.detector, np.asarray(frame))
            gazes.append([pitch, yaw])
        return torch.tensor(gazes, dtype=torch.float32).cuda()

    def get_source_exp(self, args):
        vid = imageio.get_reader(args.source_exp_video)
        exps = []
        for frame in vid:
            frame = Image.fromarray(frame).convert("RGB")
            lm = self.kp_extractor.extract_keypoint(frame)
            predicted = self.coeff_detector(frame, lm)["coeff_3dmm"][0]
            exps.append(torch.tensor(predicted[80:144], dtype=torch.float32).cuda())
        return torch.stack(exps, dim=0)

    # ---------- preprocessing helpers ----------

    def crop_src_image(self, frame, increase_ratio=0.4):
        detector = dlib.get_frontal_face_detector()
        img = np.array(frame)
        faces = detector(img, 0)
        if not faces:
            raise ValueError("No face detected in the input image")
        bbox = [faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()]
        l = bbox[3] - bbox[1]
        bbox[1] = max(0, bbox[1] - l * 0.1)
        bbox[3] = min(img.shape[0], bbox[3] - l * 0.1)
        return compute_aspect_preserved_bbox(
            tuple(bbox), increase_ratio, img.shape[0], img.shape[1]
        )

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index - self.semantic_radius, index + self.semantic_radius + 1))
        return [min(max(i, 0), num_frames - 1) for i in seq]

    # ---------- main loop ----------

    def run(self):
        print("==> running")
        args = self.args
        with torch.no_grad():
            vid_target_recon = []
            seq_len = self.pose_target.size(1)

            if args.transfer_type != "vid2vid":
                if args.semantic_mapping:
                    source_semantics = torch.cat(
                        (self.exp0, self.pose0, self.gaze0), dim=-1
                    ).unsqueeze(0).unsqueeze(-1).repeat(
                        1, 1, self.semantic_radius * 2 + 1
                    )
                    self.gen.enc.enc_motion(self.img_source, source_semantics)
                else:
                    target_info = {
                        "id_img": self.img_source,
                        "pose": self.pose_target[:, 0],
                        "exp": self.exp_target[:, 0],
                    }
                    if args.gen_with_gaze:
                        target_info["gaze"] = self.gaze_target[:, 0]
                    self.gen.enc.enc_motion(target_info)

                for i in tqdm(range(seq_len)):
                    index = self.obtain_seq_index(i, seq_len)
                    semantics = self.semantic_target[:, index].transpose(-1, -2)
                    target_info = {
                        "id_img": self.img_source,
                        "pose": self.pose_target[:, i],
                        "exp": self.exp_target[:, i],
                        "semantics": semantics,
                    }
                    if args.gen_with_gaze:
                        target_info["gaze"] = self.gaze_target[:, i]
                    if i == 0:
                        self.gen.enc.enc_motion(self.img_source, semantics)
                    pyramid_recons = self.gen(self.img_source, target_info)
                    vid_target_recon.append(pyramid_recons[0].unsqueeze(2))
            else:
                h_start = None if args.datatype == "ted" \
                    else self.gen.enc.enc_motion(self.vid_target[:, 0, :, :, :])
                for i in tqdm(range(self.vid_target.size(1))):
                    img_target = self.vid_target[:, i, :, :, :]
                    img_recon = self.gen(self.img_source, img_target, h_start)
                    vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            stem = Path(args.source_path).stem + "_" + str(self.driving_name)
            mp4_path = os.path.join(self.save_path, stem + ".mp4")
            voiced_path = mp4_path.replace(".mp4", "_voice.mp4")
            save_video(vid_target_recon, mp4_path, self.params.frame_rate, args.reverse_img)
            merge(mp4_path, self.audio_path, voiced_path, 25)
            os.remove(mp4_path)
            print("Saved to:", voiced_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in {"1", "true", "yes", "y", "t"}


def build_argparser():
    p = argparse.ArgumentParser(description="GoHD inference")
    # I/O
    p.add_argument("--source_path", type=str, required=True,
                   help="path to a single portrait image")
    p.add_argument("--audio_path", type=str, required=True,
                   help="path to a driving audio (.wav / .mp3)")
    p.add_argument("--source_pose_video", type=str, default=None,
                   help="optional driving video for head pose")
    p.add_argument("--source_exp_video", type=str, default=None,
                   help="optional driving video for expression")
    p.add_argument("--source_gaze_video", type=str, default=None,
                   help="optional driving video for gaze")
    p.add_argument("--driving_coeffs_file", type=str, default="",
                   help="optional .mat file with precomputed 3DMM coefficients")
    p.add_argument("--save_folder", type=str, default="./results")
    # Model paths
    p.add_argument("--facemodel_path", type=str, required=True,
                   help="checkpoints/FaceAnimator_GoHD.pt")
    p.add_argument("--diffmodel_path", type=str, default="",
                   help="checkpoints/audio2pose.pt")
    p.add_argument("--gazemodel_path", type=str, default="",
                   help="checkpoints/audio2gaze.pt")
    p.add_argument("--expmodel_path", type=str, default="",
                   help="checkpoints/audio2exp.pth")
    # Inference mode
    p.add_argument("--datatype", type=str, default="HDTF",
                   choices=["vox", "taichi", "ted", "vox2", "HDTF"])
    p.add_argument("--transfer_type", type=str, default="aud2vid",
                   choices=["aud2vid", "vid2vid"])
    # Generation toggles
    p.add_argument("--use_generated_pose", type=_str2bool, default=False)
    p.add_argument("--use_generated_exp", type=_str2bool, default=False)
    p.add_argument("--use_generated_gaze", type=_str2bool, default=False)
    p.add_argument("--gen_with_gaze", type=_str2bool, default=True)
    p.add_argument("--pose_smoothing", type=_str2bool, default=True)
    p.add_argument("--pose_withpose0", type=_str2bool, default=True)
    p.add_argument("--use_residual_headpose", type=_str2bool, default=True)
    p.add_argument("--use_residual_exp", type=_str2bool, default=False)
    p.add_argument("--use_residual_gaze", type=_str2bool, default=False)
    p.add_argument("--semantic_mapping", type=_str2bool, default=True)
    p.add_argument("--use_sft", type=_str2bool, default=True)
    p.add_argument("--generate_from0", type=_str2bool, default=False)
    p.add_argument("--generate_exp_from_latent", type=int, default=1)
    p.add_argument("--freeze_gaze", type=int, default=0,
                   help="0: center, 1: left, 2: right, 3: free")
    p.add_argument("--crop_input_img", type=_str2bool, default=True)
    p.add_argument("--reverse_img", type=_str2bool, default=False)
    # Scaling / sampling
    p.add_argument("--pose_scale", type=float, default=1.2)
    p.add_argument("--exp_scale", type=float, default=1.5)
    p.add_argument("--max_inference_length", type=int, default=16)
    # Model dims
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--channel_multiplier", type=int, default=1)
    p.add_argument("--latent_dim_style", type=int, default=512)
    p.add_argument("--latent_dim_motion", type=int, default=20)
    # 3DMM (consumed by Deep3DFaceRecon InferenceOptions)
    p.add_argument("--bfm_folder", type=str, default="BFM")
    p.add_argument("--bfm_model", type=str, default="BFM_model_front.mat")
    p.add_argument("--focal", type=float, default=1015.0)
    p.add_argument("--center", type=float, default=128.0)
    p.add_argument("--camera_d", type=float, default=10.0)
    return p


if __name__ == "__main__":
    # parse_known_args because Deep3DFaceRecon also reads sys.argv to set up
    # its own option namespace (gpu_ids, checkpoints_dir, model name, ...).
    args, _ = build_argparser().parse_known_args()
    print(args)
    Demo(args).run()
