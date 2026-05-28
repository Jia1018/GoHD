import os

import torch
import torch.nn as nn
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
from networks.resnet34 import resnet34
import numpy as np
from networks.syncnet import SyncNet_color
from networks.core.generator import CrossAttGenerator, StyleEncoder, _reset_parameters
from networks.core.dynamic_fc_decoder import DynamicFCDecoderLayer, DynamicFCDecoder
from networks.core.transformer import (
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)

class MyResNet34(nn.Module):
    def __init__(self,embedding_dim,input_channel = 3):
        super(MyResNet34, self).__init__()
        self.resnet = resnet34(norm_layer = BatchNorm2d,num_classes=embedding_dim,input_channel = input_channel)
    def forward(self, x):
        return self.resnet(x)

# class audio2expLSTM(nn.Module):
#     def __init__(self, blink_control=0):
#         super(audio2expLSTM,self).__init__()

#         self.em_audio = MyResNet34(256, 1)
#         #self.em_img = MyResNet34(256, 3)
#         self.em_exp0 = nn.Sequential(
#                        nn.Linear(64, 128),
#                        nn.LeakyReLU(),
#                        #BatchNorm1d(128),
#                        nn.Linear(128, 256)
#                        #nn.LeakyReLU()
#                        #BatchNorm1d(256)
#                     )

#         self.lstm = nn.LSTM(512+blink_control,256,num_layers=2,bias=True,batch_first=True)
#         #self.output = nn.Linear(256,64)
#         self.output = nn.Sequential(
#                        nn.Linear(256, 128),
#                        nn.LeakyReLU(),
#                        #BatchNorm1d(128),
#                        nn.Linear(128, 64)
#                        #BatchNorm1d(64)
#                     )
#         self.blink_control = blink_control

#     def forward(self,x):
#         self.lstm.flatten_parameters()
#         exp_em = self.em_exp0(x['exp0'])
#         result = [self.output(exp_em).unsqueeze(1)]
#         #result = []
#         bs,seqlen,_,_ = x["audio"].shape
#         zero_state = torch.zeros((2,bs,256),requires_grad=True).to(exp_em.device)
#         cur_state = (zero_state,zero_state)
#         audio = x["audio"].reshape(-1, 1, 20, 80)
#         audio_em = self.em_audio(audio).reshape(bs, seqlen, 256)
#         for i in range(seqlen):
#             #print(audio_em.shape, exp_em.shape)
#             if self.blink_control:
#                 exp_em,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],exp_em.unsqueeze(1),x["zb"][:, i:i+1]),dim=2),cur_state)
#             else:
#                 exp_em,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],exp_em.unsqueeze(1)),dim=2),cur_state)
#             exp_em = exp_em.reshape(-1, 256)
#             result.append(self.output(exp_em).unsqueeze(1))
#         res = torch.cat(result,dim=1)
#         return res
    
# class audio2expLSTM(nn.Module):
#     def __init__(self, blink_control=0, in_out_channel=64):
#         super(audio2expLSTM, self).__init__()

#         # self.em_audio = MyResNet34(256, 1)
#         self.em_audio = SyncNet_color().audio_encoder
#         #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
#         wav2lip_state_dict = torch.load('checkpoints/lipsync_expert.pth', map_location='cpu')['state_dict']
#         state_dict = self.em_audio.state_dict()

#         for k,v in wav2lip_state_dict.items():
#             if 'audio_encoder' in k:
#                 state_dict[k.replace('audio_encoder.', '')] = v
#         self.em_audio.load_state_dict(state_dict)
#         self.em_audio.eval()
#         for param in self.em_audio.parameters():
#             param.requires_grad = False
        
#         #self.em_img = MyResNet34(256, 3)
#         self.em_exp0 = nn.Sequential(
#                        nn.Linear(in_out_channel, 128),
#                        nn.LeakyReLU(),
#                        #BatchNorm1d(128),
#                        nn.Linear(128, 512),
#                     #    nn.LeakyReLU(),
#                     #    #BatchNorm1d(256)
#                     #    nn.Linear(256, 512)
#                     )

#         self.lstm = nn.LSTM(1024+blink_control,512,num_layers=2,bias=True,batch_first=True)
#         #self.output = nn.Linear(256,64)
#         self.output = nn.Sequential(
#                     #    nn.Linear(512, 256),
#                     #    nn.LeakyReLU(),
#                        nn.Linear(512, 128),
#                        nn.LeakyReLU(),
#                        #BatchNorm1d(128),
#                        nn.Linear(128, in_out_channel)
#                        #BatchNorm1d(64)
#                     )
#         self.blink_control = blink_control

#     def forward(self,x,separate=False):
#         self.lstm.flatten_parameters()
#         if separate:
#             exp_em = self.em_exp0(x['exp0_lip'])
#         else:
#             exp_em = self.em_exp0(x['exp0'])
#         #result = [self.output(exp_em).unsqueeze(1)]
#         result = []
#         residuals = []
#         bs, seqlen, _, _ = x["audio_syncnet"].shape
#         zero_state = torch.zeros((2,bs,512),requires_grad=True).to(exp_em.device)
#         cur_state = (zero_state,zero_state)
#         # audio = x["audio"].reshape(-1, 1, 20, 80)
#         audio = x['audio_syncnet'].reshape(-1, 1, 80, 16)
#         audio_em = self.em_audio(audio).reshape(bs, seqlen, 512)
#         x['audio'] = audio_em[:, 1:]
#         for i in range(1, seqlen):
#             #print(audio_em.shape, exp_em.shape)
#             if self.blink_control:
#                 exp_em,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],exp_em.unsqueeze(1),x["zb"][:, i:i+1]),dim=2),cur_state)
#             else:
#                 exp_em,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],exp_em.unsqueeze(1)),dim=2),cur_state)
#             exp_em = exp_em.reshape(-1, 512)
#             exp_res = self.output(exp_em)
#             if separate:
#                 out_exp = exp_res + x['exp0_lip']
#             else:
#                 out_exp = exp_res + x['exp0']
#             residuals.append(exp_res.unsqueeze(1))
#             result.append(out_exp.unsqueeze(1))
#         if separate:
#             x['exp_lip_pred'] = torch.cat(result, dim=1)
#             x['exp_lip_motion_pred'] = torch.cat(residuals, dim=1)
#         else:
#             x['exp_pred'] = torch.cat(result, dim=1)
#             x['exp_motion_pred'] = torch.cat(residuals, dim=1)
#         return x

    
class audio2expLSTM(nn.Module):
    def __init__(self, blink_control=0, generation=1, finetune=0, in_out_channel=64):
        super(audio2expLSTM, self).__init__()

        # self.em_audio = MyResNet34(256, 1)
        self.em_audio = SyncNet_color().audio_encoder
        #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        wav2lip_state_dict = torch.load('checkpoints/lipsync_expert.pth', map_location='cpu')['state_dict']
        state_dict = self.em_audio.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                state_dict[k.replace('audio_encoder.', '')] = v
        self.em_audio.load_state_dict(state_dict)
        self.em_audio.eval()
        for param in self.em_audio.parameters():
            param.requires_grad = False
        
        #self.em_img = MyResNet34(256, 3)
        self.em_exp0 = nn.Sequential(
                       nn.Linear(in_out_channel, 128),
                       nn.LeakyReLU(),
                       #BatchNorm1d(128),
                       nn.Linear(128, 512),
                    #    nn.LeakyReLU(),
                    #    #BatchNorm1d(256)
                    #    nn.Linear(256, 512)
                    )

        self.lstm = nn.LSTM(1024+blink_control+512*generation,512,num_layers=2,bias=True,batch_first=True)
        #self.output = nn.Linear(256,64)
        self.output = nn.Sequential(
                    #    nn.Linear(512, 256),
                    #    nn.LeakyReLU(),
                       nn.Linear(512, 128),
                       nn.LeakyReLU(),
                       #BatchNorm1d(128),
                       nn.Linear(128, in_out_channel)
                       #BatchNorm1d(64)
                    )
        self.blink_control = blink_control
        self.generation = generation
        
        # if finetune:
        #     self.finetune = nn.Sequential(
        #                     nn.Linear(in_out_channel, 512),
        #                     nn.LeakyReLU(),
        #                     nn.Linear(512, 64)
        #                     )
        if finetune:
            self.em_exp0.eval()
            self.lstm.eval()
            self.output.eval()
            for param in self.em_exp0.parameters():
                param.requires_grad = False
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.output.parameters():
                param.requires_grad = False
            self.finetune = nn.Sequential(
                            nn.Linear(64, 512),
                            nn.LeakyReLU(),
                            # BatchNorm1d(512),
                            nn.Linear(512, 64)
                            )

    def forward(self,x,separate=False):
        self.lstm.flatten_parameters()
        if separate:
            exp_em0 = self.em_exp0(x['exp0_lip'])
        else:
            exp_em0 = self.em_exp0(x['exp0'])
        #result = [self.output(exp_em).unsqueeze(1)]
        result = []
        residuals = []
        bs, seqlen, _, _ = x["audio"].shape
        zero_state = torch.zeros((2,bs,512),requires_grad=True).to(exp_em0.device)
        cur_state = (zero_state,zero_state)
        # audio = x["audio"].reshape(-1, 1, 20, 80)
        audio = x['audio'].reshape(-1, 1, 80, 16)
        audio_em = self.em_audio(audio).reshape(bs, seqlen, 512)#[:, 1:]
        x['audio_em'] = audio_em
        # if x['zb'] is not None:
        #     x['zb'] = x['zb'][:, 1:]
        exp_em = exp_em0
        # exp_em = torch.randn_like(exp_em0).to(exp_em0.device)
        for i in range(0, seqlen):
            #print(audio_em.shape, exp_em.shape)
            cat_feature = torch.cat((audio_em[:,i:i+1], exp_em.unsqueeze(1)),dim=2)
            # cat_feature = torch.cat((audio_em[:,i:i+1], exp_em.unsqueeze(1), exp_em0.unsqueeze(1)),dim=2)
            if self.blink_control:
                cat_feature = torch.cat((cat_feature, x["zb"][:, i:i+1]), dim=2)
            if self.generation:
                z = torch.randn(bs, 1, 512).to(cat_feature.device)
                cat_feature = torch.cat((cat_feature, z), dim=2)
            exp_em,cur_state = self.lstm(cat_feature, cur_state)
            exp_em = exp_em.reshape(-1, 512) #+ exp_em0
            if hasattr(self, 'finetune'):
                tune_scale = self.finetune(exp_em)
                exp_em = exp_em + tune_scale
            exp_res = self.output(exp_em)
            # if hasattr(self, 'finetune'):
            #     tune_scale = self.finetune(exp_res).clamp(0, 1) + 1
            #     exp_res = exp_res * tune_scale
            if separate:
                out_exp = exp_res + x['exp0_lip']
            else:
                out_exp = exp_res + x['exp0']
            residuals.append(exp_res.unsqueeze(1))
            result.append(out_exp.unsqueeze(1))
        if separate:
            x['exp_lip_pred'] = torch.cat(result, dim=1)
            x['exp_lip_motion_pred'] = torch.cat(residuals, dim=1)
        else:
            x['exp_pred'] = torch.cat(result, dim=1)
            x['exp_motion_pred'] = torch.cat(residuals, dim=1)
        return x
    
class audio2exp_CrossAttention(nn.Module):
    def __init__(self, blink_control=0, generation=1, in_out_channel=64):
        super(audio2exp_CrossAttention, self).__init__()

        # self.em_audio = MyResNet34(256, 1)
        self.em_audio = SyncNet_color().audio_encoder
        #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        # wav2lip_state_dict = torch.load('ckpt/lipsync_expert.pth', map_location='cpu')['state_dict']
        wav2lip_state_dict = torch.load('../LIA-main/syncnet_ckpt/lms3d_mouthdiagonalandupdowndist_vox2_withbrown/checkpoint_step000078000.pth', map_location='cpu')['state_dict']
        state_dict = self.em_audio.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                state_dict[k.replace('audio_encoder.', '').replace('module.', '')] = v
        self.em_audio.load_state_dict(state_dict)
        self.em_audio.eval()
        for param in self.em_audio.parameters():
            param.requires_grad = False
        
        #self.em_img = MyResNet34(256, 3)
        self.em_exp0 = nn.Sequential(
                       nn.Linear(in_out_channel, 128),
                       nn.LeakyReLU(),
                    #    BatchNorm1d(128),
                       nn.Linear(128, 512),
                    #    nn.LeakyReLU(),
                    #    BatchNorm1d(512)
                    #    nn.Linear(256, 512)
                    )

        self.lstm = nn.LSTM(1536+blink_control+512*generation,512,num_layers=2,bias=True,batch_first=True)
        self.direct_mapping = nn.Linear(1024+blink_control, 64)
        # self.bn = BatchNorm1d(512)
        #self.output = nn.Linear(256,64)
        self.output = nn.Sequential(
                    #    nn.Linear(512, 256),
                    #    nn.LeakyReLU(),
                       nn.Linear(512, 128),
                       nn.LeakyReLU(),
                       #BatchNorm1d(128),
                       nn.Linear(128, in_out_channel)
                       #BatchNorm1d(64)
                    )
        self.blink_control = blink_control
        self.generation = generation
        
        # self.cross_att = CrossAttGenerator(input_dim=in_out_channel, output_dim=in_out_channel)

    def forward(self,x):
        self.lstm.flatten_parameters()
        exp_em0 = self.em_exp0(x['exp0'])
        result_seq = []
        result_indiv = []
        residuals = []
        bs, seqlen, _, _ = x["audio"].shape
        zero_state = torch.zeros((2,bs,512),requires_grad=True).to(exp_em0.device)
        cur_state = (zero_state, zero_state)
        # audio = x["audio"].reshape(-1, 1, 20, 80)
        audio = x['audio'].reshape(-1, 1, 80, 16)
        audio_em = self.em_audio(audio).reshape(bs, seqlen, 512)#[:, 1:]
        x['audio_em'] = audio_em
        if x['zb'] is not None:
            x['zb'] = x['zb']#[:, 1:]
        exp_em = exp_em0
        # exp_em = torch.randn_like(exp_em0).to(exp_em0.device)
        for i in range(0, seqlen):
            #print(audio_em.shape, exp_em.shape)
            # exp_em_ = exp_em
            cat_feature = torch.cat((audio_em[:,i:i+1], exp_em.unsqueeze(1), exp_em0.unsqueeze(1)),dim=2)
            direct_feature = torch.cat((audio_em[:,i], exp_em0),dim=-1)
            if self.blink_control:
                cat_feature = torch.cat((cat_feature, x["zb"][:, i:i+1]), dim=2)
                direct_feature = torch.cat((direct_feature, x["zb"][:, i]), dim=-1)
            if self.generation:
                z = torch.randn(bs, 1, 512).to(cat_feature.device)
                cat_feature = torch.cat((cat_feature, z), dim=2)
            exp_em, cur_state = self.lstm(cat_feature, cur_state)
            exp_em = exp_em.reshape(-1, 512) + exp_em0
            exp_res = self.output(exp_em)
            out_exp = exp_res + x['exp0']
            residuals.append(exp_res.unsqueeze(1))
            result_seq.append(out_exp.unsqueeze(1))
            result_indiv.append(self.direct_mapping(direct_feature).unsqueeze(1))
        x['exp_pred'] = torch.cat(result_seq, dim=1)
        x['exp_motion_pred'] = torch.cat(residuals, dim=1)
        x['exp_indiv_pred'] = torch.cat(result_indiv, dim=1)
        # x['exps_integrate'] = self.cross_att(x['exps_indiv_pred'], x['exps_motion_pred'])
        
        return x
    
class SimpleWrapperV2(nn.Module):
    def __init__(self, blink_control=0, in_out_channel=64) -> None:
        super().__init__()

        self.audio_encoder = SyncNet_color().audio_encoder
        # Optionally initialise the audio encoder from a wav2lip syncnet
        # checkpoint. The inference checkpoint (audio2exp.pth) already
        # carries trained weights for this submodule, so skipping the
        # pretrain step is harmless.
        wav2lip_path = "checkpoints/lipsync_expert.pth"
        if os.path.exists(wav2lip_path):
            wav2lip_state_dict = torch.load(wav2lip_path, map_location="cpu")["state_dict"]
            state_dict = self.audio_encoder.state_dict()
            for k, v in wav2lip_state_dict.items():
                if "audio_encoder" in k:
                    state_dict[k.replace("audio_encoder.", "").replace("module.", "")] = v
            self.audio_encoder.load_state_dict(state_dict)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.mapping1 = nn.Linear(512 + in_out_channel + 1 * blink_control, in_out_channel)
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, data):
        audio = data['audio']
        bs, seqlen, _, _ = audio.shape
        ref = data['exp0'].unsqueeze(1).repeat(1, seqlen, 1)
        x = self.audio_encoder(audio.reshape(-1, 1, 80, 16)).view(bs*seqlen, -1)
        ref_reshape = ref.reshape(bs*seqlen, -1)
        if data['zb'] is not None:
            ratio = data['zb']
            ratio = ratio.reshape(bs*seqlen, -1)
            y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1)) 
        else:
            y = self.mapping1(torch.cat([x, ref_reshape], dim=1)) 
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        data['exp_motion_pred'] = out
        return data
    
class SimpleWrapperV3(nn.Module):
    def __init__(self, blink_control=0, brow_control=0, in_out_channel=64) -> None:
        super().__init__()

        self.audio_encoder = SyncNet_color().audio_encoder
        #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        # wav2lip_state_dict = torch.load('/data/zqzhou/diffwave/ckpt/lipsync_expert.pth', map_location='cpu')['state_dict']
        # state_dict = self.audio_encoder.state_dict()

        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         state_dict[k.replace('audio_encoder.', '').replace('module.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)

        self.mapping1 = nn.Linear(512+in_out_channel+1*blink_control+20*brow_control, in_out_channel)
        #self.mapping2 = nn.Linear(30, 64)
        #nn.init.constant_(self.mapping1.weight, 0.)
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, data):
        audio = data['audio']
        bs, seqlen, _, _ = audio.shape
        ref = data['exp0'].unsqueeze(1).repeat(1, seqlen, 1)
        x = self.audio_encoder(audio.reshape(-1, 1, 80, 16)).view(bs*seqlen, -1)
        ref_reshape = ref.reshape(bs*seqlen, -1)
        if data['zb'] is not None and data['bc'] is not None:
            ratio = data['zb']
            ratio = ratio.reshape(bs*seqlen, -1)
            brow_distance = data['bc']
            brow_distance = brow_distance.reshape(bs*seqlen, -1)
            y = self.mapping1(torch.cat([x, ref_reshape, ratio, brow_distance], dim=1)) 
        elif data['zb'] is not None:
            ratio = data['zb']
            ratio = ratio.reshape(bs*seqlen, -1)
            y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1))
        elif data['bc'] is not None:
            brow_distance = data['bc']
            brow_distance = brow_distance.reshape(bs*seqlen, -1)
            y = self.mapping1(torch.cat([x, ref_reshape, brow_distance], dim=1))
        else:
            y = self.mapping1(torch.cat([x, ref_reshape], dim=1)) 
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        data['exp_motion_pred'] = out
        return data
    
class TwoStageGenerator(nn.Module):
    def __init__(self, blink_control=0, brow_control=0, in_out_channel=64, mid_channel=21, generation=1) -> None:
        super().__init__()

        self.simplewarpper = SimpleWrapperV3(blink_control, brow_control, in_out_channel)
        #### load the pre-trained simplewarpper
        # warpper_state_dict = torch.load('ckpt/HDTF_regression_seqlen30_lipread_wav2lip_browandeyecontrol/audio2exp_7.pth', map_location='cpu')['state_dict_G']
        # # self.simplewarpper.load_state_dict(warpper_state_dict)
        # new_state_dict = OrderedDict()
        # for key, value in warpper_state_dict.items():
        #     new_state_dict[key.replace('module.', '')] = value
        # self.simplewarpper.load_state_dict(new_state_dict)
        # self.simplewarpper.eval()
        # for param in self.simplewarpper.parameters():
        #     param.requires_grad = False

        self.control_encoder = nn.Sequential(
                                nn.Linear(mid_channel, 128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 512),
                                )

        self.lstm = nn.LSTM(1024+in_out_channel+512*generation,512,num_layers=2,bias=True,batch_first=True)
        self.output = nn.Sequential(
                       nn.Linear(512, 128),
                       nn.LeakyReLU(),
                       nn.Linear(128, mid_channel)
                    )
        self.generation = generation

    def forward(self, data):
        audio = data['audio']
        bs, seqlen, _, _ = audio.shape
        ref = data['exp0'].unsqueeze(1)
        audio_em = self.simplewarpper.audio_encoder(audio.reshape(-1, 1, 80, 16)).reshape(bs, seqlen, -1)
        ref_reshape = ref.repeat(1, seqlen, 1).reshape(bs*seqlen, -1)
        assert data['zb'] is not None and data['bc'] is not None
        eye_ratio0 = data['zb'][:, 0]
        brow_control0 = data['bc'][:, 0]
        control0 = torch.cat([eye_ratio0, brow_control0], dim=-1)
        control_em0 = self.control_encoder(control0).unsqueeze(1)
        control_seq = [control0.unsqueeze(1)]
        zero_state = torch.zeros((2,bs,512),requires_grad=True).to(control_em0.device)
        cur_state = (zero_state, zero_state)
        for i in range(1, seqlen):
            #print(audio_em.shape, exp_em.shape)
            # exp_em_ = exp_em
            cat_feature = torch.cat((audio_em[:,i:i+1], control_em0, ref), dim=-1)
            if self.generation:
                z = torch.randn(bs, 1, 512).to(cat_feature.device)
                cat_feature = torch.cat((cat_feature, z), dim=-1)
            control_em, cur_state = self.lstm(cat_feature, cur_state)
            control_em = control_em + control_em0
            out_control = self.output(control_em.reshape(-1, 512))
            control_seq.append(out_control.unsqueeze(1))
        control_seq = torch.cat(control_seq, dim=1)
        data['control_seq'] = control_seq
        y = self.simplewarpper.mapping1(torch.cat([audio_em.view(bs*seqlen, -1), ref_reshape, control_seq.view(bs*seqlen, -1)], dim=-1)) 
        out = y.reshape(bs, seqlen, -1) #+ ref # resudial
        data['exp_motion_pred'] = out
        return data
    
class TwoStageTransGenerator(nn.Module):
    def __init__(self, blink_control=0, brow_control=0, in_out_channel=64, mid_channel=21, generation=1,
                d_model=512,  
                in_exp0_channel=64,
                nhead=8,
                num_decoder_layers=3,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=False,
                pos_embed_len=1000) -> None:
        super().__init__()

        self.simplewarpper = SimpleWrapperV3(blink_control, brow_control, in_out_channel)
        # #### load the pre-trained simplewarpper
        # warpper_state_dict = torch.load('ckpt/HDTF_regression_seqlen30_lipread_wav2lip_browandeyecontrol/audio2exp_9.pth', map_location='cpu')['state_dict_G']
        # # self.simplewarpper.load_state_dict(warpper_state_dict)
        # new_state_dict = OrderedDict()
        # for key, value in warpper_state_dict.items():
        #     new_state_dict[key.replace('module.', '')] = value
        # self.simplewarpper.load_state_dict(new_state_dict)
        # self.simplewarpper.eval()
        # for param in self.simplewarpper.parameters():
        #     param.requires_grad = False

        self.control_encoder = nn.Sequential(
                                nn.Linear(mid_channel, 128),
                                nn.LeakyReLU(),
                                nn.Linear(128, d_model),
                                )

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        _reset_parameters(self.decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.output = nn.Sequential(
                       nn.Linear(d_model, 128),
                       nn.LeakyReLU(),
                       nn.Linear(128, mid_channel)
                    )
        self.generation = generation

    def forward(self, data):
        audio = data['audio']
        bs, seqlen, _, _ = audio.shape
        ref = data['exp0'].unsqueeze(1)
        audio_em = self.simplewarpper.audio_encoder(audio.reshape(-1, 1, 80, 16)).reshape(bs, seqlen, -1)
        ref_reshape = ref.repeat(1, seqlen, 1).reshape(bs*seqlen, -1)
        assert data['zb'] is not None and data['bc'] is not None
        eye_ratio0 = data['zb'][:, 0]
        brow_control0 = data['bc'][:, 0]
        control0 = torch.cat([eye_ratio0, brow_control0], dim=-1)
        control_em0 = self.control_encoder(control0).unsqueeze(1).repeat(1, seqlen, 1)

        tgt = torch.randn_like(control_em0)
        pos_embed = self.pos_embed(seqlen)
        # pos_embed = pos_embed.permute(1, 0, 2)
        control_feat = self.decoder(tgt, audio_em, pos=pos_embed, query_pos=control_em0)[0]
        control_seq = self.output(control_feat + control_em0)
        data['control_seq'] = control_seq
        y = self.simplewarpper.mapping1(torch.cat([audio_em.view(bs*seqlen, -1), ref_reshape, control_seq.view(bs*seqlen, -1)], dim=-1)) 
        out = y.reshape(bs, seqlen, -1) #+ ref # resudial
        data['exp_motion_pred'] = out
        return data


class StyleClassifier(nn.Module):
    def __init__(self, in_channel=21, class_num=112, d_model=512,
                nhead=8,
                num_encoder_layers=3,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                pos_embed_len=500) -> None:
        super().__init__()
        self.audio_encoder = SyncNet_color().audio_encoder
        #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        # wav2lip_state_dict = torch.load('ckpt/lipsync_expert.pth', map_location='cpu')['state_dict']
        # state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                state_dict[k.replace('audio_encoder.', '').replace('module.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)

        self.style_encoder = StyleEncoder(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            pos_embed_len,
            in_channel,
            condition=True,
            cond_dim=512
        )

        # mlps 
        self.output_layer = nn.Sequential(
                            nn.Linear(d_model*2, d_model),
                            BatchNorm1d(d_model),
                            nn.LeakyReLU(),
                            nn.Linear(d_model, class_num)
                            )
        # self.output_layer = nn.Linear(d_model*2, class_num)

    def forward(self, data):
        audio = data['audio']
        bs, seqlen, _, _ = audio.shape
        cond = self.audio_encoder(audio.reshape(-1, 1, 80, 16)).reshape(bs, seqlen, -1)
        x = torch.cat((data['bc'], data['zb']), dim=-1)
        x = self.style_encoder(x, cond)
        return self.output_layer(x), cond

class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

class StyleExpGenerator(nn.Module):
    def __init__(self, in_channel=21, class_num=112, d_model=512, out_channel=64, 
                in_exp0_channel=64,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=False,
                pos_embed_len=500) -> None:
        super().__init__()

        self.encoder = StyleClassifier(in_channel, class_num, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, normalize_before, pos_embed_len)
        # classifier_state_dict = torch.load('/data/zqzhou/diffwave/ckpt/style_classifier_seq300_lr1e-4_mlps/audio2exp_46.pth', map_location='cpu')['state_dict']
        # state_dict = self.encoder.state_dict()
        # for k,v in classifier_state_dict.items():
        #     if 'module' in k:
        #         state_dict[k.replace('module.', '')] = v
        # self.encoder.load_state_dict(state_dict)
        # # self.encoder.load_state_dict(classifier_state_dict)
        # self.encoder.eval()
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.exp0_encoder = nn.Sequential(
                       nn.Linear(in_exp0_channel, 128),
                       nn.LeakyReLU(),
                       nn.Linear(128, d_model),
                    )

        self.softmax = nn.Softmax(dim=1)
        self.direction = Direction(class_num)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        _reset_parameters(self.decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, out_channel),
        )

    def forward(self, x):
        """

        Args:
            x (dict): audio: (B, num_frames, 80, 16)
                      bc: (B, num_frames, 20)
                      zb: (B, num_frames, 1)
            style_label: (B, 112)

        Returns:
            face3d: (B, num_frames, 64)
        """
        style_label, content = self.encoder(x)
        B, T, _ = content.shape
        # style label fusion
        if x['label'] is not None:
            random_indices = torch.randint(2, size=(B,)).to(style_label.device)
            style_fusion = torch.where(random_indices.unsqueeze(1).bool(), style_label, x['label'])
        else:
            style_fusion = style_label
        style_weight = self.softmax(style_fusion)
        # (B, class_num)
        cond = self.exp0_encoder(x['exp0'])
        style_code = self.direction(style_weight)
        # (B, C)
        style = style_code.unsqueeze(1).repeat(1, T, 1)
        cond = cond.unsqueeze(1).repeat(1, T, 1)
        # (B, T, C)

        tgt = torch.randn_like(style) + cond
        pos_embed = self.pos_embed(T)
        face3d_feat = self.decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        # (B, T, C)
        face3d = self.tail_fc(face3d_feat)
        # (B, T, C_exp)
        x['exp_motion_pred'] = face3d
        x['style_label'] = style_fusion
        return x
    
class RandExpGenerator(nn.Module):
    def __init__(self, d_model=512, out_channel=64, 
                in_exp0_channel=64,
                nhead=8,
                num_decoder_layers=3,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=False,
                pos_embed_len=1000) -> None:
        super().__init__()

        self.audio_encoder = SyncNet_color().audio_encoder
        #### load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        # wav2lip_state_dict = torch.load('ckpt/lipsync_expert.pth', map_location='cpu')['state_dict']
        # state_dict = self.audio_encoder.state_dict()

        # for k,v in wav2lip_state_dict.items():
        #     if 'audio_encoder' in k:
        #         state_dict[k.replace('audio_encoder.', '').replace('module.', '')] = v
        # self.audio_encoder.load_state_dict(state_dict)
        # self.audio_encoder.eval()
        # for param in self.audio_encoder.parameters():
        #     param.requires_grad = False

        self.exp0_encoder = nn.Sequential(
                       nn.Linear(in_exp0_channel, 128),
                       nn.LeakyReLU(),
                       nn.Linear(128, d_model),
                    )

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        _reset_parameters(self.decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, out_channel),
        )

    def forward(self, x):
        """

        Args:
            x (dict): audio: (B, num_frames, 80, 16)
                      exp0: (B, 64)

        Returns:
            face3d: (B, num_frames, 64)
        """
        audio = x['audio']
        B, T, _, _ = audio.shape
        content = self.audio_encoder(audio.reshape(-1, 1, 80, 16)).reshape(B, T, -1)
        cond = self.exp0_encoder(x['exp0'])
        # (B, C)
        cond = cond.unsqueeze(1).repeat(1, T, 1)
        # content = content.permute(1, 0, 2)
        # (B, T, C)

        tgt = torch.randn_like(cond)
        pos_embed = self.pos_embed(T)
        # pos_embed = pos_embed.permute(1, 0, 2)
        face3d_feat = self.decoder(tgt, content, pos=pos_embed, query_pos=cond)[0]
        # (B, T, C)
        # face3d_feat = face3d_feat.permute(1, 0, 2)
        # (B, T, C)
        face3d = self.tail_fc(face3d_feat)
        # (B, T, C_exp)
        x['exp_motion_pred'] = face3d
        return x
