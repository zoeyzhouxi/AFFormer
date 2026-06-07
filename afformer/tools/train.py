# -*- coding: utf-8 -*-
# Author: Xi Zhou <xi.zhou@jcu.edu.au>, Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import afformer.hypes_yaml.yaml_utils as yaml_utils
from afformer.tools import train_utils
from afformer.data_utils.datasets import build_dataset
import glob
from icecream import ic
import tqdm
import wandb
from datetime import datetime

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--run_test', default=True, help='Run test after training.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, mode='train')
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False, mode='validate')

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    lowest_val_epoch = -1
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, steps_per_epoch=len(train_loader), init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, steps_per_epoch=len(train_loader))

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    lowest_val_loss = float('inf')
    patience = hypes['train_params']['patience']
    trials = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        train_loss = 0
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            train_loss += final_loss.item()
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
            # back-propagation
            final_loss.backward()
            optimizer.step()

            # torch.cuda.empty_cache()
        train_loss = train_loss / len(train_loader)
        writer.add_scalar('Train_Loss', train_loss, epoch)

        
        valid_ave_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                if batch_data is None:
                    continue
                model.eval()

                batch_data = train_utils.to_device(batch_data, device)
                batch_data['ego']['epoch'] = epoch
                ouput_dict = model(batch_data['ego'])

                final_loss = criterion(ouput_dict,
                                        batch_data['ego']['label_dict'])
                valid_ave_loss.append(final_loss.item())

        valid_ave_loss = statistics.mean(valid_ave_loss)
        print('At epoch %d, the validation loss is %f' % (epoch,
                                                            valid_ave_loss))
        writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)


        # lowest val loss
        if valid_ave_loss < lowest_val_loss:
            lowest_val_loss = valid_ave_loss
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))
            if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                'net_epoch%d.pth' % (lowest_val_epoch))):
                os.remove(os.path.join(saved_path,
                                'net_epoch%d.pth' % (lowest_val_epoch)))
            lowest_val_epoch = epoch + 1
            trials = 0
        else:
            trials += 1
            if trials > patience:
                print('Early stopping')
                break

     
        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    
    if opt.run_test:
        fusion_method = opt.fusion_method
        cmd = f"python afformer/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
