import os
import json
import argparse
import torch
import dataloaders
import models
import math
from utils import Logger
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, domain_loss=False):
    torch.manual_seed(42)
    train_logger = Logger()
    
    # DATA LOADERS
    config['train_supervised']['percnt_lbl'] = config["sup_percent"]
    config['train_unsupervised']['percnt_lbl'] = config["unsup_percent"]
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    supervised_loader = dataloaders.CDDataset(config['train_supervised'])
    unsupervised_loader = dataloaders.CDDataset(config['train_unsupervised'])
    val_loader = dataloaders.CDDataset(config['val_loader'])
    iter_per_epoch = len(unsupervised_loader)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    elif config['model']['sup_loss'] == 'FL':
        alpha = get_alpha(supervised_loader) # calculare class occurences
        print(alpha)
        sup_loss = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2.0, smooth = 1e-5)
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                                num_classes=val_loader.dataset.num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                        rampup_ends=rampup_ends)

    model = models.Consistency_ResNet50_CD(num_classes=val_loader.dataset.num_classes,
                                           conf=config['model'],
                                           sup_loss=sup_loss,
                                           cons_w_unsup=cons_w_unsup,
                                           weakly_loss_w=config['weakly_loss_w'],
                                           use_weak_lables=config['use_weak_lables'],
                                           domain_loss=domain_loss)
   
    print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_EGY-sup_Minawao-unsup.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    
    for dloss in [False, True]:
        if dloss:
            exp_ext = "OT"
        else:
            exp_ext = "nOT"
        for data_name in ["Nduta", "Minawao", "Nguyile", "kuletirkidi"]: # "Nduta","Rakuba_sudan"
            print("==============================================================================")
            print(f"=================Processing {data_name} =====================================")

            config = json.load(open(args.config))
            config['train_unsupervised']["data_dir"] = config['train_unsupervised']["data_dir"].format(data_name)
            config['name'] = config['name'].format(data_name, exp_ext)
            config["experim_name"] = config["experim_name"].format(data_name, exp_ext)

            #torch.backends.cudnn.benchmark = True
            main(config, args.resume, domain_loss=dloss)
            print("==============================================================================")
