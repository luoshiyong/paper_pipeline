import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from dataset.dataset import Dataset
import matplotlib.pyplot as plt
from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score
import utilities.losses as losses
from utilities.utils import str2bool, count_params
import pandas as pd
from net import Unet, unet
from utilities.lr_policy import PolyLR

# from  net.small_ecanet import PraNet121
# 换模型需要修改的地方
arch_names = list(Unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')

    # 换模型需要修改的地方
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    # 换数据集需要修改的地方
    parser.add_argument('--dataset', default="LiTS",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    # 换模型需要修改的地方
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 30)')

    # 换模型需要修改的地方
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--deepsupervision', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--checkpoint', default='models/LiTS_Unet_lym/2022-09-03-18-27-42/epoch186-0.9679-0.7708_model.pth',
                        help='image file extension')
    parser.add_argument('--model_name', default="unet",
                        help='dataset name')
    parser.add_argument('--save_path', default="/home/luosy/TTNet/pred/tmp/",
                        help='dataset name')
    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def validate(args, val_loader, model):
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target, filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
            else:
                output = model(input)
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)[0]
                dice_2 = dice_coef(output, target)[1]
                output = torch.sigmoid(output)
                pred_liver = output.detach().cpu().numpy()[0][0]
                pred_tumor = output.detach().cpu().numpy()[0][1]
                pred_liver[pred_liver > 0.5] = 133
                pred_liver[pred_liver <= 0.5] = 0
                pred_tumor[pred_tumor > 0.5] = 133
                pred_tumor[pred_tumor <= 0.5] = 0
                pred_liver[pred_tumor > 1] = 255
                im = Image.fromarray(np.uint8(pred_liver))
                im.convert('L').save(args.save_path + filename[0] + '.jpg')

            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def main():
    args = parse_args()
    # args.dataset = "datasets"


    # Data loading code
    val_img_paths = glob('/home/luosy/data3382/val_image/*')
    val_mask_paths = glob('/home/luosy/data3382/val_mask/*')
    print("val_num:%s" % str(len(val_img_paths)))

    # create model
    # 换模型需要修改的地方
    print("=> creating model %s" % args.arch)
    if args.model_name=='unet':
        model = unet.U_Net(args)
    else:
        model = unet.U_Net(args)
    model = model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    print(count_params(model))




    val_dataset = Dataset(args, val_img_paths, val_mask_paths, val=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,90,150], gamma=0.3)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[38], gamma=0.3)
    # scheduler = PolyLR(optimizer,max_iters = 200,power=0.8)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)


    best_loss = 100
    best_iou = 0
    trigger = 0
    first_time = time.time()
    lr_list = []


    val_log = validate(args, val_loader, model)

    print('val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'% (val_log['iou'], val_log['dice_1'], val_log['dice_2']))

    end_time = time.time()
    print("time:", (end_time - first_time) / 60)
    torch.cuda.empty_cache()



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    main()
