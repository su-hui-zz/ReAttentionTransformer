import os
import sys
import datetime
import pprint
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, str_gpus, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams import evaluate_cls_loc

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F

import argparse
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.vgg import vgg16_cam
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
from urllib.request import urlretrieve
import pdb

def mparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default = './configs/CUB/deit_trt_fuse_base_patch16_224_0.6.yaml', type=str)
    parser.add_argument('--pth_file', default = './ckpt_save/CUB/deit_trt_fuse_base_patch16_224_TOKENTHR0.6_BS128_0.912/ckpt/model_best_top1_loc.pth', type=str) 

    parser.add_argument('--img_pths', default = ['./figures/CUB/Bewick_Wren_0121_184765.png','./figures/CUB/California_Gull_0014_40880.png',], nargs='+')
    #parser.add_argument('--save_dir', default = 'save_res', type=str, help = 'save dir')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    args = parser.parse_args()
    return args

def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer  
    if 'pos_' in cfg.MODEL.ARCH:
        model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=True,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            select_num = cfg.MODEL.SELECT_NUM,
            score_num = cfg.MODEL.SCORE_NUM,
            # num_classes = 5089
        )
    elif 'trt' in cfg.MODEL.ARCH:
        model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=False,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            thresh = cfg.MODEL.THRESH
        )  
    elif 'vgg' in cfg.MODEL.ARCH:
        model = vgg16_cam(
            pretrained=True,
            num_classes=cfg.DATA.NUM_CLASSES,
        )
    else:
        model = create_deit_model(
            cfg.MODEL.ARCH,
            pretrained=True,
            num_classes=cfg.DATA.NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            # select_num = 50,
            # score_num = 25,
            # num_classes = 5089
        )
    #print(model)

    if args.pth_file:
        checkpoint = torch.load(args.pth_file, map_location='cpu')['state_dict']
        check_dict = {}
        for k, v in checkpoint.items():
            check_dict[k.replace('module.','')] = v
        model.load_state_dict(check_dict)
        print('load pretrained model.')
    
    model = model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    args = mparse()
    config_file = args.config_file
    cfg_from_file(config_file)
    cfg.BASIC.ROOT_DIR = './'

    model = creat_model(cfg, args)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img_path in args.img_pths:
        im = Image.open(img_path).convert('RGB')
        x = transform(im)
        device = torch.device("cuda")
        x = x.to(device)

        with torch.no_grad():
            x_logits, resmap = model(x.unsqueeze(0),return_cam=True)

        ### cls probs
        x_probs = F.softmax(x_logits, dim=-1)
        pred_cls_id = x_probs.argmax()

        ### map
        #cam_pred = tscams[0,pred_cls_id,:,:].detach().cpu().numpy()
        cam_pred = resmap[0,pred_cls_id,:,:]
        cam_pred = (cam_pred - cam_pred.min())/(cam_pred.max() - cam_pred.min())
        cam_pred = np.array(cam_pred.detach().cpu())

        cam_pred = cv2.resize(cam_pred, (x.size()[1], x.size()[2]))
        cam_pred = np.uint8(255*cam_pred)
        #cam_pred = cv2.applyColorMap(cam_pred, cv2.COLORMAP_JET)

        save_pth = os.path.join(os.path.dirname(img_path), os.path.basename(img_path).split('.')[0]+'_res.png')
        cv2.imwrite(save_pth, cam_pred)


 