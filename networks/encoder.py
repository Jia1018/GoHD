import math
import torch
from torch import nn
from torch.nn import functional as F
from .utils import AntiAliasInterpolation2d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), ]

    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1, )

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):

        return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
                                  bias=bias and not activate))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class EncoderApp(nn.Module):
    def __init__(self, size, in_channel, w_dim=512):
        super(EncoderApp, self).__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }

        self.w_dim = w_dim
        log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(in_channel, channels[size], 1))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs.append(EqualConv2d(in_channel, self.w_dim, 4, padding=0, bias=False))

    def forward(self, x):

        res = []
        h = x
        for conv in self.convs:
            h = conv(h)
            res.append(h)

        return res[-1].squeeze(-1).squeeze(-1), res[::-1][2:]

class Encoder_with_Semantics(nn.Module):
    def __init__(self, size, semantic_size, dim=512, dim_motion=20, in_channel=3):
        super(Encoder_with_Semantics, self).__init__()

        # appearance netmork
        self.net_app = EncoderApp(size, in_channel, dim)

        # semantics mapping
        self.semantics_enc = MappingNet(semantic_size, dim, 3)

        # motion network
        fc = [EqualLinear(dim*2, dim)]
        for i in range(3):
            fc.append(EqualLinear(dim, dim))

        fc.append(EqualLinear(dim, dim_motion))
        self.fc = nn.Sequential(*fc)

    def enc_app(self, x):

        h_source = self.net_app(x)

        return h_source

    def enc_motion(self, x, semantic):

        h, _ = self.net_app(x)
        h_ = self.semantics_enc(semantic).squeeze(-1)
        h_motion = self.fc(torch.cat((h, h_), dim=-1))

        return h_motion

    def forward(self, input_source, input_semantic, h_start=None, h_motion_source=None):

        if input_semantic is not None:

            h_source, feats = self.net_app(input_source)
            h_target = self.semantics_enc(input_semantic).squeeze(-1)

            h_motion_target = self.fc(torch.cat((h_source, h_target), dim=-1))

            if h_start is not None and h_motion_source is not None:
                h_motion = [h_motion_target, h_motion_source, h_start]
            else:
                h_motion = [h_motion_target]

            return h_source, h_motion, feats, h_target
        else:
            h_source, feats = self.net_app(input_source)

            return h_source, None, feats, None

class Encoder(nn.Module):
    def __init__(self, size, dim=512, dim_motion=20):
        super(Encoder, self).__init__()

        # appearance netmork
        self.net_app = EncoderApp(size, 3, dim)

        # motion network
        fc = [EqualLinear(dim, dim)]
        for i in range(3):
            fc.append(EqualLinear(dim, dim))

        fc.append(EqualLinear(dim, dim_motion))
        self.fc = nn.Sequential(*fc)

    def enc_app(self, x):

        h_source = self.net_app(x)

        return h_source

    def enc_motion(self, x):

        h, _ = self.net_app(x)
        h_motion = self.fc(h)

        return h_motion

    def forward(self, input_source, input_target, h_start=None):

        if input_target is not None:

            h_source, feats = self.net_app(input_source)
            h_target, _ = self.net_app(input_target)

            h_motion_target = self.fc(h_target)

            if h_start is not None:
                h_motion_source = self.fc(h_source)
                h_motion = [h_motion_target, h_motion_source, h_start]
            else:
                h_motion = [h_motion_target]

            return h_source, h_motion, feats
        else:
            h_source, feats = self.net_app(input_source)

            return h_source, None, feats
        
#embed_fn, input_ch = get_embedder(10)
#embed_fn_normals, input_ch_normals = get_embedder(4)
        
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        
        # 如果包含原始位置
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)  # 把一个不对数据做出改变的匿名函数添加到列表中
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  # 得到 [2^0, 2^1, ... ,2^(L-1)] 参考论文 5.1 中的公式
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)  # 得到 [2^0, 2^(L-1)] 的等差数列，列表中有 L 个元素
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))  # sin(x * 2^n)  参考位置编码公式
                out_dim += d  # 每使用子编码公式一次就要把输出维度加 3，因为每个待编码的位置维度是 3
                    
        self.embed_fns = embed_fns  # 相当于是一个编码公式列表
        self.out_dim = out_dim
        
    def embed(self, inputs):
    # 对各个输入进行编码，给定一个输入，使用编码列表中的公式分别对他编码
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
def get_embedder(multires, i=0):

    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,  # 如果为真，最终的编码结果包含原始坐标
                'input_dims' : 64,  # 输入给编码器的数据的维度
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,  # 即论文中 5.1 节位置编码公式中的 L 
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, emo=embedder_obj : emo.embed(x)  # embed 现在相当于一个编码器，具体的编码公式与论文中的一致。
    return embed, embedder_obj.out_dim

class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super( MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        #print(input_3dmm.shape)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            #print(out.shape)
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        return out   
    

class AudioPoseEmbed(nn.Module):
    def __init__(self, withAudio=True, withLandmarks=True, ExpEmbed=True):
        super(AudioPoseEmbed,self).__init__()
        self.withLandmarks = withLandmarks
        self.ExpEmbed = ExpEmbed
        self.pad = 0
        self.down_id = AntiAliasInterpolation2d(3,0.25)
        self.down_pose = AntiAliasInterpolation2d(1,0.25)
        if ExpEmbed:
            self.PE, self.em_dim = get_embedder(10)
            self.exp_embedder = nn.Sequential(nn.ConvTranspose2d(1, 8, (1, 4), stride=(1, 3), padding=(0, 0)),
                                       BatchNorm2d(8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(8, 2, (1, 1), stride=(1, 1), padding=(0, 0)))
        if withLandmarks:
            self.down_lm = AntiAliasInterpolation2d(1,0.25)

        if withAudio:
            self.embedding = nn.Sequential(nn.ConvTranspose2d(1, 8, (13, 15), stride=(1, 1), padding=(0, 31)),
                                       BatchNorm2d(8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(8, 2, (13, 13), stride=(1, 1), padding=(6, 6)))
        else:
            self.embedding = None

    def forward(self, x):

        id_feature = self.down_id(x["id_img"]) #[B,3,64,64]
        pose_feature = self.down_pose(x["pose"]) #[B,1,64,64]
        
        if self.embedding is not None:
        
            bs,_,c_dim = x["audio"].shape #[B,4*5,80]

            audio_embedding = self.embedding(x["audio"].reshape(-1,1,20,c_dim)) #[B,2,32,32]
            audio_embedding = F.interpolate(audio_embedding,scale_factor=2).reshape(bs,2,64,64) #[B,2,64,64]

            embeddings = torch.cat([audio_embedding, id_feature, pose_feature],dim=1) #[B,6,64,64]
        
        elif self.withLandmarks:
            lm_feature = self.down_lm(x["lms"]) #[B,1,64,64]
            embeddings = torch.cat([id_feature, pose_feature, lm_feature],dim=1) #[B,5,64,64]
            
        elif self.ExpEmbed:
            bs, _ = x["exp"].shape
            exp_embeddings = self.PE(x["exp"])
            #print(exp_embeddings.shape)
            exp_embeddings = exp_embeddings.reshape(bs, 64, -1).unsqueeze(1) #[B,1,64,11]
            exp_feature = self.exp_embedder(exp_embeddings) #[B,2,64,64]
            embeddings = torch.cat([id_feature, pose_feature, exp_feature],dim=1) #[B,6,64,64]
            
        else:
            embeddings = torch.cat([id_feature, pose_feature],dim=1) #[B,4,64,64]

        return embeddings
    
    
class PoseAudEncoder(nn.Module):
    def __init__(self, size, dim=512, dim_motion=20, withAudio=True, withExp=True, ExpEmbed=True, ExpSmooth=True, withGaze=True):
        super(PoseAudEncoder, self).__init__()

        # appearance netmork
        self.net_app = EncoderApp(size, 3, dim)
        # target encoding
        self.embed = AudioPoseEmbed(withAudio, not withExp, ExpEmbed)
        if withAudio:
            input_channel = 6
        elif withExp: # w/ exp w/o audio
            if ExpEmbed:
                input_channel = 6
            else:
                input_channel = 4
        else: # w/ landmarks w/o exp and audio
            input_channel = 5
        self.target_enc = EncoderApp(64, input_channel, dim)
        # gaze encoding
        if withGaze:
            self.fc0 = EqualLinear(dim+2, dim)
        # exp encoding
        if withExp and not ExpEmbed:
            if not ExpSmooth:
                self.exp_encode = EqualLinear(64, dim)
            else:
                self.exp_encode = MappingNet(64, dim, 1)
            fc = [EqualLinear(dim*2, dim)]
        else:
            self.exp_encode = None
            fc = [EqualLinear(dim, dim)]

        # motion network
        for i in range(3):
            fc.append(EqualLinear(dim, dim))

        fc.append(EqualLinear(dim, dim_motion))
        self.fc = nn.Sequential(*fc)

    def enc_app(self, x):

        h_source = self.net_app(x)

        return h_source

    def enc_motion(self, x):
        
        h = self.embed(x)
        h, _ = self.target_enc(h)
        
        if self.exp_encode is not None:
            exp = x['exp']
            exp = self.exp_encode(exp).view(h.shape)
            
            h_motion = self.fc(torch.cat((h, exp), dim=-1))
        else:
            h_motion = self.fc(h)

        return h_motion

    def forward(self, input_source, input_target, h_start=None, source_info=None, input_em=None):

        if input_target is not None:

            h_source, feats = self.net_app(input_source)
            # for feat in feats:
            #     print("feat", feat.shape)
            if input_em is None:
                h_target = self.embed(input_target)
            else:
                h_target = input_em
            #print(h_target.shape)
            h_target, _ = self.target_enc(h_target)
            
            if 'gaze' in input_target and hasattr(self, 'fc0'):
                gaze = input_target['gaze']
                #print(h_target.shape, gaze.shape)
                h_target = self.fc0(torch.cat((h_target, gaze), dim=-1))

            if self.exp_encode is not None:
                exp = input_target['exp']
                exp = self.exp_encode(exp).view(h_target.shape)
                h_motion_target = self.fc(torch.cat((h_target, exp), dim=-1))
            else:
                h_motion_target = self.fc(h_target)
            
            if 'target' in input_target:
                h_ref, _ = self.net_app(input_target['target'])
                h_motion_ref = self.fc(h_ref)
            else:
                h_motion_ref = None
                
            if h_start is not None and source_info is not None:
                h_motion_source = self.enc_motion(source_info)
                h_motion = [h_motion_target, h_motion_source, h_start]
                if h_motion_ref is not None:
                    ref_motion = [h_motion_ref, h_motion_source, h_start]
                else:
                    ref_motion = None
            else:
                h_motion = [h_motion_target]
                if h_motion_ref is not None:
                    ref_motion = [h_motion_ref]
                else:
                    ref_motion = None

            return h_source, h_motion, ref_motion, feats
        else:
            h_source, feats = self.net_app(input_source)

            return h_source, None, None, feats
        
