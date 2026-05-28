# Copyright 2020 LMNT, Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


params = AttrDict(
    batch_size=8,
    learning_rate=0.0006,
    max_grad_norm=None,

    frame_rate=25,
    sample_rate=16000,
    n_mels=320,
    n_fft=1280,
    hop_samples=160,
    crop_mel_frames=100,

    in_channels=6,
    out_channels=6,
    in_out_channels=6,
    residual_layers=20,
    residual_channels=256,
    embedding_dim=512,
    style_dim=18,
    dilation_cycle=[0, 1, 2],
    dilation_cycle_length=10,
    unconditional=False,
    n_noise_schedule=150,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    audio_len=22050 * 5,
    pose_seq_len=150,
    pose_seq_step=100,

    bfm_folder='BFM',
    bfm_model='BFM_model_front.mat',
    focal=1015.,
    center=128.,
    camera_d=10.,
)
