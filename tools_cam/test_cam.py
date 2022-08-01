# ----------------------------------------------------------------------------------------------------------
# TS-CAM
# Copyright (c) Learning and Machine Perception Lab (LAMP), SECE, University of Chinese Academy of Science.
# ----------------------------------------------------------------------------------------------------------
import os
import sys
import datetime
import pprint
import numpy as np

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader,\
    AverageMeter, accuracy, list2acc, adjust_learning_rate
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_cls_loc
from cams_MaxBoxAccV2_deit import evaluate_maxbox_acc, BoxEvaluator
from models import *
from models.vgg import vgg16_cam
import pdb

import torch
from torch.utils.tensorboard import SummaryWriter

from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer

import cv2

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
    print(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        pretrained_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(pretrained_dict)
        
        #model.load_state_dict(checkpoint['model'])
        print('load pretrained model.')

    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)

    print('Preparing networks done!')
    return device, model, cls_criterion


#############################
### gt-known and top-acc ####
#############################
def main_gt_top(args):
    #args = update_config()

    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    #prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, cls_criterion = creat_model(cfg, args)

    update_val_step = 0
    update_val_step, _, cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known, wrong_details = \
        val_loc_one_epoch(val_loader, model, device, cls_criterion, writer, cfg, update_val_step)
    print('Cls@1:{0:.3f}\tCls@5:{1:.3f}\n'
          'Loc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}\n'.format( cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known))
    print('wrong_details:{} {} {} {} {} {}'.format(wrong_details[0], wrong_details[1], wrong_details[2],
                                                   wrong_details[3], wrong_details[4], wrong_details[5]))
    print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))


def val_loc_one_epoch(val_loader, model, device, criterion, writer, cfg, update_val_step):

    losses = AverageMeter()

    cls_top1 = []
    cls_top5 = []
    loc_top1 = []
    loc_top5 = []
    loc_gt_known = []
    top1_loc_right = []
    top1_loc_cls = []
    top1_loc_mins = []
    top1_loc_part = []
    top1_loc_more = []
    top1_loc_wrong = []

    with torch.no_grad():
        model.eval()
        for i, (input, target, bbox, image_names) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input = input.to(device)

            cls_logits, cams = model(input, return_cam=True)
            loss = criterion(cls_logits, target)

            prec1, prec5 = accuracy(cls_logits.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            writer.add_scalar('loss_iter/val', loss.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top1', prec1.item(), update_val_step)
            writer.add_scalar('acc_iter/val_top5', prec5.item(), update_val_step)

            cls_top1_b, cls_top5_b, loc_top1_b, loc_top5_b, loc_gt_known_b, top1_loc_right_b, \
                top1_loc_cls_b,top1_loc_mins_b, top1_loc_part_b, top1_loc_more_b, top1_loc_wrong_b = \
                    evaluate_cls_loc(input, target, bbox, cls_logits, cams, image_names, cfg, 0)
            cls_top1.extend(cls_top1_b)
            cls_top5.extend(cls_top5_b)
            loc_top1.extend(loc_top1_b)
            loc_top5.extend(loc_top5_b)
            top1_loc_right.extend(top1_loc_right_b)
            top1_loc_cls.extend(top1_loc_cls_b)
            top1_loc_mins.extend(top1_loc_mins_b)
            top1_loc_more.extend(top1_loc_more_b)
            top1_loc_part.extend(top1_loc_part_b)
            top1_loc_wrong.extend(top1_loc_wrong_b)

            loc_gt_known.extend(loc_gt_known_b)

            if i % cfg.BASIC.DISP_FREQ == 0 or i == len(val_loader)-1:
                print('Val Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    0, i+1, len(val_loader), loss=losses))
                print('Cls@1:{0:.3f}\tCls@5:{1:.3f}\n'
                      'Loc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}\n'.format(
                    list2acc(cls_top1), list2acc(cls_top5),
                    list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known)))
        wrong_details = []
        wrong_details.append(np.array(top1_loc_right).sum())
        wrong_details.append(np.array(top1_loc_cls).sum())
        wrong_details.append(np.array(top1_loc_mins).sum())
        wrong_details.append(np.array(top1_loc_part).sum())
        wrong_details.append(np.array(top1_loc_more).sum())
        wrong_details.append(np.array(top1_loc_wrong).sum())
    return update_val_step, losses.avg, list2acc(cls_top1), list2acc(cls_top5), \
           list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known), wrong_details



#############################
######## maxboxaccv2 ########
#############################
def main_maxboxaccv2(args):
    #args = update_config()

    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join('ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    #prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, cls_criterion = creat_model(cfg, args)

    max_box_acc = val_maxboxaccv_one_epoch(val_loader, model, device, cls_criterion, writer, cfg)


def val_maxboxaccv_one_epoch(val_loader, model, device, criterion, writer, cfg):


    evaluator = BoxEvaluator(dataset_name=cfg.DATA.DATASET, cam_threshold_list=[cfg.MODEL.CAM_THR], 
                            iou_threshold_list=[30,50,70], multi_contour_eval=True)

    with torch.no_grad():
        model.eval()
        for i, (input, target, bbox, image_names) in enumerate(val_loader):
            if i % cfg.BASIC.DISP_FREQ == 0:
                print(i,'/', len(val_loader))
            target = target.to(device)
            input = input.to(device)
            cls_logits, cams = model(input, return_cam=True)
            evaluate_maxbox_acc(input, target, bbox, evaluator, cams, cfg)
    max_box_acc = evaluator.compute()
    max_box_acc = np.average(max_box_acc)   
    #pdb.set_trace()
    print('MaxboxAcc2: ',max_box_acc)
    return max_box_acc


if __name__ == "__main__":
    args = update_config()
    
    #main_gt_top()
    #main_maxboxaccv2()
    eval('main_'+cfg.TEST.METRICS)(args)  #TEST.METRICS 
    
