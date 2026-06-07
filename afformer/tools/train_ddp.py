import argparse
import os
import statistics
import glob
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
import importlib
import afformer.hypes_yaml.yaml_utils as yaml_utils
from afformer.tools import train_utils
from afformer.data_utils.datasets import build_dataset
from afformer.tools import multi_gpu_utils
import tqdm
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}
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
    multi_gpu_utils.init_distributed_mode(opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, mode='train')
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False, mode='validate')

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    sample_length = len(train_loader)
    print('Creating Model')
    model = train_utils.create_model(hypes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    lowest_val_epoch = -1
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, steps_per_epoch=sample_length, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, steps_per_epoch=sample_length)

    # record training
    writer = SummaryWriter(saved_path)

    # for knowledge distillation
    if "kd_flag" in hypes.keys():
        kd_flag = True
        teacher_model_name = hypes['kd_flag']['teacher_model'] # point_pillar_disconet_teacher
        teacher_model_config = hypes['kd_flag']['teacher_model_config']
        teacher_checkpoint_path = hypes['kd_flag']['teacher_path']

        # import the model
        model_filename = "afformer.models." + teacher_model_name
        model_lib = importlib.import_module(model_filename)
        teacher_model_class = None
        target_model_name = teacher_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                teacher_model_class = cls
        
        teacher_model = teacher_model_class(teacher_model_config)
        teacher_model.load_state_dict(torch.load(teacher_checkpoint_path, weights_only=False), strict=False)
        
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
            teacher_model.to(device)

        teacher_model.eval()
    else:
        kd_flag = False

    # define the loss
    criterion = train_utils.create_loss(hypes)
    # record lowest validation loss checkpoint.
    lowest_val_loss = float('inf')
    epoches = hypes['train_params']['epoches']
    patience = hypes['train_params']['patience']
    trials = 0
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    print('Training start')
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=sample_length, leave=True)
        if opt.distributed:
            sampler_train.set_epoch(epoch)

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

            if kd_flag:
                teacher_output_dict = teacher_model(batch_data['ego'])
                ouput_dict.update(teacher_output_dict)

            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, sample_length, writer, pbar=pbar2)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, sample_length, writer, pbar=pbar2, suffix="_single")

            pbar2.update(1)

            # back-propagation
            final_loss.backward()
            optimizer.step()

        
        valid_ave_loss = []
        model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                if batch_data is None:
                    continue
                batch_data = train_utils.to_device(batch_data, device)
                batch_data['ego']['epoch'] = epoch
                ouput_dict = model(batch_data['ego'])
                if kd_flag:
                    teacher_output_dict = teacher_model(batch_data['ego'])
                    ouput_dict.update(teacher_output_dict)

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
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
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