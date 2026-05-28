import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

# class SyncNet_color(nn.Module):
#     def __init__(self):
#         super(SyncNet_color, self).__init__()

#         self.face_encoder = nn.Sequential(
#             Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

#             Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#             Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#             Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#             Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#             Conv2d(512, 512, kernel_size=2, stride=1, padding=0),)
#             #Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

#         self.audio_encoder = nn.Sequential(
#             Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
#             Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(32, 64, kernel_size=3, stride=(1, 4), padding=1),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(64, 128, kernel_size=3, stride=3, padding=2),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

#             Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#             Conv2d(512, 512, kernel_size=2, stride=1, padding=0),)

#     def forward(self, audio_sequences, face_sequences): 

#         face_embedding = self.face_encoder(face_sequences)
#         audio_embedding = self.audio_encoder(audio_sequences)

#         audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
#         face_embedding = face_embedding.view(face_embedding.size(0), -1)

#         audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
#         face_embedding = F.normalize(face_embedding, p=2, dim=1)


#         return audio_embedding, face_embedding

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding
    
class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    
class SyncNet(nn.Module):
    def __init__(self, lm_dim=90):
        super(SyncNet, self).__init__()

        self.mouth_encoder = nn.Sequential(
            Conv1d(lm_dim, 96, kernel_size=3, stride=1, padding=1),

            Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = SyncNet_color().audio_encoder
        #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        # wav2lip_state_dict = torch.load('checkpoints/lipsync_expert.pth', map_location='cpu')['state_dict']
        # state_dict = self.audio_encoder.state_dict()

        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         state_dict[k.replace('audio_encoder.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)
        # self.audio_encoder.eval()
        # for param in self.audio_encoder.parameters():
        #     param.requires_grad = False

    def forward(self, audio_sequences, lms_sequences): 
        lms_sequences = lms_sequences.transpose(1,2)
        lms_embedding = self.mouth_encoder(lms_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        lms_embedding = lms_embedding.view(lms_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        lms_embedding = F.normalize(lms_embedding, p=2, dim=1)

        return audio_embedding, lms_embedding
