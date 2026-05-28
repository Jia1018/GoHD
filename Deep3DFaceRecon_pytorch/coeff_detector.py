import os
import glob
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision

from Deep3DFaceRecon_pytorch.models import create_model
from Deep3DFaceRecon_pytorch.models.networks import L2CS
from Deep3DFaceRecon_pytorch.util.preprocess import align_img
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d


class CoeffDetector(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = 'cuda'
        self.model.parallelize()
        self.model.eval()

        self.lm3d_std = load_lm3d(opt.bfm_folder) 

    def forward(self, img, lm):
        
        img, trans_params = self.image_transform(img, lm)

        data_input = {                
                'imgs': img[None],
                }        
        self.model.set_input(data_input)  
        self.model.test()
        pred_coeff = {key:self.model.pred_coeffs_dict[key].cpu().numpy() for key in self.model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
            pred_coeff['id'], 
            pred_coeff['exp'], 
            pred_coeff['tex'], 
            pred_coeff['angle'],
            pred_coeff['gamma'],
            pred_coeff['trans'],
            trans_params[None],
            ], 1)
        
        return {'coeff_3dmm':pred_coeff, 
                'crop_img': Image.fromarray((img.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8))}

    def image_transform(self, images, lm):
        """
        param:
            images:          -- PIL image 
            lm:              -- numpy array
        """
        W,H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, trans_params        

def get_data_path(root, keypoint_root):
    filenames = list()
    keypoint_filenames = list()

    IMAGE_EXTENSIONS_LOWERCASE = {'jpg', 'png', 'jpeg', 'webp'}
    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS_LOWERCASE.union({f.upper() for f in IMAGE_EXTENSIONS_LOWERCASE})
    extensions = IMAGE_EXTENSIONS

    for ext in extensions:
        filenames += glob.glob(f'{root}/*.{ext}', recursive=True)
    filenames = sorted(filenames)
    for filename in filenames:
        name = os.path.splitext(os.path.basename(filename))[0]
        keypoint_filenames.append(
            os.path.join(keypoint_root, name + '.txt')
        )
    return filenames, keypoint_filenames

def get_landmark_bbox(lm, scale=1):
    bbox = []
    for _i, box_id in enumerate([[0, 68]]): # the first bbox crops the mouth area, the second bbox crops the whole face
        box_lm = lm[:, box_id[0]:box_id[1]]
        ly, ry = torch.min(box_lm[:, :, 0], dim=1)[0], torch.max(box_lm[:, :, 0], dim=1)[0]
        lx, rx = torch.min(box_lm[:, :, 1], dim=1)[0], torch.max(box_lm[:, :, 1], dim=1)[0]  # shape: [b]
        lx, rx, ly, ry = (lx * scale).long(), (rx * scale).long(), (ly * scale).long(), (ry * scale).long()
        lx, rx, ly, ry = lx, rx, ly, ry
        lx, rx, ly, ry = lx.unsqueeze(1), rx.unsqueeze(1), ly.unsqueeze(1), ry.unsqueeze(1)
        bbox.append(torch.cat([lx, rx, ly, ry], dim=1))
    return bbox

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

def get_gaze_params(model, img):
    img = cv2.resize(img, (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = transformations(im_pil)
    #print(img.shape)
    img  = Variable(img).cuda()
    img  = img.unsqueeze(0) 
    softmax = nn.Softmax(dim=1)
                    
    # gaze prediction
    gaze_pitch, gaze_yaw = model(img)
                    
    pitch_predicted = softmax(gaze_pitch)
    yaw_predicted = softmax(gaze_yaw)
                    
    # Get continuous predictions in degrees.
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi/180.0
    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi/180.0
    #print(pitch_predicted, yaw_predicted)
    
    return pitch_predicted, yaw_predicted

def vis_landmark(img, shape, linewidth=2):
    height, width, _ = img.shape
    if isinstance(shape, torch.Tensor):
        shape = shape.cpu().numpy()
    shape = shape.astype('int32')
    linewidth = linewidth * (height // 256)
    radius = (height // 256)
    def draw_curve(idx_list, color=(255, 255, 255), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth, cv2.LINE_AA)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth, cv2.LINE_AA)
    draw_curve(list(range(0, 16)))  # jaw
    draw_curve(list(range(17, 21)))  # eye brow
    draw_curve(list(range(22, 26)))
    draw_curve(list(range(27, 35)))  # nose
    draw_curve(list(range(36, 41)), loop=True)  # eyes
    draw_curve(list(range(42, 47)), loop=True)
    draw_curve(list(range(48, 59)), loop=True)  # mouth
    draw_curve(list(range(60, 67)), loop=True)
    for i in range(68):
        img = cv2.circle(img, (shape[i, 0], shape[i, 1]), radius, (255, 255, 255), -1)
    
    return img

