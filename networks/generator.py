from torch import nn
from .encoder import Encoder, PoseAudEncoder, Encoder_with_Semantics
from .styledecoder import Synthesis


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats = self.enc(img_source, img_drive, h_start)
        img_recon = self.dec(wa, alpha, feats)

        return img_recon

class Generator_withPose(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, use_sft=False, withAudio=True, withExp=True, ExpEmbed=True, ExpSmooth=True, withGaze=True, blur_kernel=[1, 3, 3, 1]):
        super(Generator_withPose, self).__init__()

        # encoder
        self.enc = PoseAudEncoder(size, style_dim, motion_dim, withAudio, withExp, ExpEmbed, ExpSmooth, withGaze)
        self.dec = Synthesis(size, style_dim, motion_dim, use_sft, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, None, feat)

        return img

    def forward(self, img_source, target_info, h_start=None, source_info=None, input_em=None):
        wa, alpha, ref_alpha, feats = self.enc(img_source, target_info, h_start, source_info, input_em)
        img_recon = self.dec(wa, alpha, ref_alpha, feats)

        return img_recon
    

class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super(MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        return out  


# class Generator_Semantic(nn.Module):
#     def __init__(self, size, semantic_dim=70, style_dim=512, motion_dim=20, channel_multiplier=1, use_sft=False, blur_kernel=[1, 3, 3, 1]):
#         super(Generator_Semantic, self).__init__()

#         # encoder
#         self.enc = Encoder(size, style_dim, motion_dim)
#         self.dec = Synthesis(size, style_dim, motion_dim, use_sft, blur_kernel, channel_multiplier)
        
#         self.semantic_enc = MappingNet(semantic_dim, 512, 3)

#     def get_direction(self):
#         return self.dec.direction(None)

#     def synthesis(self, wa, alpha, feat):
#         img = self.dec(wa, alpha, None, feat)

#         return img

#     def forward(self, img_source, target_info, switch=False, h_start=None):
#         if target_info is not None:
#             img_drive = target_info['target']
#             semantics = target_info['semantics']
#             p = self.semantic_enc(semantics).squeeze(-1)
#         else:
#             img_drive = None
#             p = None
#         wa, alpha, feats = self.enc(img_source, img_drive, h_start)
#         img_recon, latent_loss = self.dec(wa, alpha, None, feats, p, switch)

#         return img_recon, latent_loss
    
# class Generator_Semantic(nn.Module):
#     def __init__(self, size, semantic_size, style_dim=512, motion_dim=20, channel_multiplier=1, use_sft=False, blur_kernel=[1, 3, 3, 1]):
#         super(Generator_Semantic, self).__init__()

#         # encoder
#         self.enc = Encoder_with_Semantics(size, semantic_size, style_dim, motion_dim)
#         self.dec = Synthesis(size, style_dim, motion_dim, use_sft, blur_kernel, channel_multiplier)

#     def get_direction(self):
#         return self.dec.direction(None)

#     def synthesis(self, wa, alpha, feat):
#         img = self.dec(wa, alpha, feat)

#         return img

#     def forward(self, img_source, target_info, h_start=None, h_source_motion=None):
#         if target_info is not None and 'semantics' in target_info:
#             semantics = target_info['semantics']
#         else:
#             semantics = None
#         wa, alpha, feats = self.enc(img_source, semantics, h_start, h_source_motion)
#         img_recon, latent_loss = self.dec(wa, alpha, None, feats, None, switch=False)

#         return img_recon, latent_loss

class Generator_Semantic(nn.Module):
    def __init__(self, size, semantic_size, style_dim=512, motion_dim=20, channel_multiplier=1, use_sft=False, in_channel=3, out_channel=3, blur_kernel=[1, 3, 3, 1]):
        super(Generator_Semantic, self).__init__()

        # encoder
        self.enc = Encoder_with_Semantics(size, semantic_size, style_dim, motion_dim, in_channel)
        self.dec = Synthesis(size, style_dim, motion_dim, out_channel, use_sft, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, target_info, h_start=None):
        if target_info is not None:
            semantics = target_info['semantics']
        else:
            semantics = None
        wa, alpha, feats, _ = self.enc(img_source, semantics, h_start)
        img_recon = self.dec(wa, alpha, feats)

        return img_recon