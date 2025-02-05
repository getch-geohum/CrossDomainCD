import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
from utils.metrics import eval_metrics, AverageMeter
from utils.htmlwriter import HTML
from matplotlib import pyplot as plt
from utils.helpers import DeNormalize
import time

def get_imgid_list(Dataset_Path, split, i):
    file_list  = os.path.join(Dataset_Path, 'list', split +".txt")
    filelist   = np.loadtxt(file_list, dtype=str)
    if filelist.ndim == 2:
        return filelist[:, 0]
    image_id = filelist[i].split("/")[-1].split(".")[0]
    return image_id

def multi_scale_predict(model, image_A, image_B, scales, num_classes, flip=False):
    H, W        = (image_A.size(2), image_A.size(3))
    upsize      = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample    = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w= upsize[0] - H, upsize[1] - W
    image_A     = F.pad(image_A, pad=(0, pad_w, 0, pad_h), mode='reflect')
    image_B     = F.pad(image_B, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image_A.shape[2], image_A.shape[3]))

    for scale in scales:
        scaled_img_A = F.interpolate(image_A, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_img_B = F.interpolate(image_B, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(A_l=scaled_img_A, B_l=scaled_img_B))
        
        if flip:
            fliped_img_A = scaled_img_A.flip(-1)
            fliped_img_B = scaled_img_B.flip(-1)
            fliped_predictions  = upsample(model(A_l=fliped_img_A, B_l=fliped_img_B))
            scaled_prediction   = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)
    
    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main(data_fold, exp_ext, weight=None):
    args = parse_arguments(data_fold, exp_ext=exp_ext)
    print(args)

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [1.0,1.25]

    # data_name = "Nguyile"
    config['train_unsupervised']["data_dir"] = config['train_unsupervised']["data_dir"].format(data_fold)
    config['name'] = config['name'].format(data_fold, exp_ext)
    config["experim_name"] = config["experim_name"].format(data_fold, exp_ext)

    # DATA LOADER
    config['val_loader']["batch_size"]  = 1
    config['val_loader']["num_workers"] = 1
    config['val_loader']["split"]       = "val"  # given the training and testing images are the same
    config['val_loader']["shuffle"]     = False
    config['val_loader']['data_dir']    = args.Dataset_Path
    loader = dataloaders.CDDataset(config['val_loader'])
    num_classes = 2
    palette = get_voc_pallete(num_classes)

    # MODEL
    config['model']['supervised'] = False if weight is None else True
    config['model']['semi'] = False if weight is not None else True
    model = models.Consistency_ResNet50_CD(num_classes=num_classes, conf=config['model'], testing=True)
    print(f'\n{model}\n')
    if weight is not None:
        checkpoint = torch.load(weight)
    else:
        checkpoint = torch.load(args.model)
    model = torch.nn.DataParallel(model)
    try:
        print("Loading the state dictionery...")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()

    if args.save and not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    #Set HTML
    web_dir = f'{args.outpath}/{config["experim_name"]}'
    txt_dir = f'{args.outpath}/{config["experim_name"]}/reults.txt'
    html_results = HTML(web_dir=web_dir, exp_name=config['experim_name']+"--Test--",
                            save_name=config['experim_name'], config=config)

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    for index, data in enumerate(tbar):
        image_A, image_B, label = data
        image_id = get_imgid_list(Dataset_Path=args.Dataset_Path, split=config['val_loader']["split"], i=index)
        image_A = image_A.cuda()
        image_B = image_B.cuda()
        label   = label.cuda()
        
        #PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image_A, image_B, scales, num_classes)
        
        prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)

        #Calculate metrics
        output = torch.from_numpy(output).cuda()
        label[label>=1] = 1
        output = torch.unsqueeze(output, 0)
        label  = torch.unsqueeze(label, 0)
        correct, labeled, inter, union  = eval_metrics(output, label, num_classes)
        total_inter, total_union        = total_inter+inter, total_union+union
        total_correct, total_label      = total_correct+correct, total_label+labeled
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        tbar.set_description('Test Results | PixelAcc: {:.4f}, IoU(no-change): {:.4f}, IoU(change): {:.4f} |'.format(pixAcc, IoU[0], IoU[1]))

        #SAVE RESULTS
        prediction_im = colorize_mask(prediction, palette)
        prediction_im.save(f'{args.outpath}/{config["experim_name"]}/{image_id}.png')
    
    #Printing average metrics on test-data
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                   "IoU_change": IoU[1],
                   "IoU_noChange": IoU[0]} # "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3)))
    print(seg_metrics)
    log = {     
                'val_loss': 0.0,
                **seg_metrics
            }
    html_results.add_results(epoch=1, seg_resuts=log)
    html_results.save()

    with open(txt_dir, 'w') as f:
        print(seg_metrics, file=f)

def parse_arguments(fold, exp_ext):
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='/home/getch/meta/SemiCD/configs/config-sup_open_camp.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '--model', default=f'/home/getch/ssl/CHANGE-EXPERIMENT/LEVIR-CD_reproduce/SemiCD_WUHAN_sup_100_{fold}_100_unsup_{exp_ext}/best_model.pth', type=str,
                        help='Path to the trained .pth model')
    parser.add_argument( '--save', action='store_true', help='Save images')
    parser.add_argument('--Dataset_Path', default=f"/home/getch/ssl/LEVIR_WIHAN_256/{fold}", type=str,
                        help='Path to dataset LEVIR-CD')
    parser.add_argument('--outpath', default=f"/home/getch/ssl/SemiCD/outputs_SUP_OPEN_CAMP_100_{exp_ext}", type=str,
                        help='Path to dataset test dataset')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    for data_fold in ["Nduta", "Minawao", "Nguyile", "kuletirkidi"]: # "Rakuba_sudan",
        print("Processing data fold: ", data_fold)
        unsupervised = True
        if unsupervised:
            dicts = {"LEVIR":"/home/getch/ssl/CHANGE-EXPERIMENT/LEVIR-CD_reproduce/SemiCD_sup_100/best_model.pth",
                     "WUHAN":"/home/getch/ssl/CHANGE-EXPERIMENT/LEVIR-CD_reproduce/WUHAN_sup_100/best_model.pth",
                     "EGY":"/home/getch/ssl/CHANGE-EXPERIMENT/LEVIR-CD_reproduce/EGY_BCD_sup_100/best_model.pth"}
            for key in dicts.keys():
                print("GOT weight: ", key)
                main(data_fold=data_fold, exp_ext=key, weight=dicts[key])
        else:
            for expext in ["nOT", "OT"]:
                main(data_fold=data_fold, exp_ext=expext, weight=None)

# /home/getch/ssl/LEVIR_WIHAN_256/LEVIR-CD256
# /home/getch/ssl/LEVIR_WIHAN_256/WHU-CD-256
