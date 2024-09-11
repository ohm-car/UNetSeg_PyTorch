import argparse
import logging
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torchsummary
import datetime

from eval_multiloss import eval_net
from unet.unet_model_xB import UNet

from torch.utils.tensorboard import SummaryWriter
# from utils.pascalVOC_multiloss import PascalVOCDataset
from utils.petsReconDataset_multiloss_pl import PetsReconDataset
from utils.percLoss import percLoss
from torch.utils.data import DataLoader, random_split

# root_dir = Path().resolve().parent
# dir_img = os.path.join(root_dir, 'Datasets/petsData/images/')
# dir_mask = os.path.join(root_dir, 'Datasets/petsData/annotations/trimaps/')
# tm = datetime.datetime.now()
# dir_checkpoint = 'checkpoints/multiloss/{:02d}-{:02d}/{:02d}-{:02d}-{:02d}/'.format(tm.month, tm.day, tm.hour, tm.minute, tm.second)

root_dir = None
dir_img = None
dir_mask = None
tm = datetime.datetime.now()
dir_checkpoint = None


def train_net(args,
              net,
              device,
              save_freq,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              regularizer=None,
              regularizer_weight=0.1):

    root_dir = args.rd
    print(root_dir, type(root_dir))
    dir_img = os.path.join(root_dir, 'Datasets/petsData/images/')
    print(dir_img, type(dir_img))
    dir_mask = os.path.join(root_dir, 'Datasets/petsData/annotations/trimaps/')
    print(dir_mask, type(dir_mask))
    tm = datetime.datetime.now()
    dir_checkpoint = 'checkpoints/multiloss/{:02d}-{:02d}/{:02d}-{:02d}-{:02d}/'.format(tm.month, tm.day, tm.hour, tm.minute, tm.second)

    dataset = PetsReconDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    # print(type(dataset),type(train),type(train.dataset))
    # print("Train IDs:", train.dataset.ids)
    # print("Val IDs:", val.dataset.ids)
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers= max(min(2, cpu_count()), 1), pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=max(min(2, cpu_count()), 1), pin_memory=True, drop_last=True)
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers = 2)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers = 2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:             {epochs}
        Batch size:         {batch_size}
        Learning rate:      {lr}
        Training size:      {n_train}
        Validation size:    {n_val}
        Checkpoints:        {save_cp}
        Device:             {device.type}
        Images scaling:     {img_scale}
        Regularizer:        {regularizer}
        Regularizer Weight: {regularizer_weight}
        Mask Sampling:      {args.sp}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.L1Loss()
    else:
        criterion = nn.L1Loss()

    weight_recon_loss, weight_percLoss = 1, 5

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # print("Started Batch Training!\n")
                imgs = batch['image']
                recon_img = batch['reconstructed_image']
                imgs_percs = batch['mask_perc']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                recon_img = recon_img.to(device=device, dtype=torch.float32)
                imgs_percs = imgs_percs.to(device=device, dtype=torch.float32)

                pred_recon_img, pred_mask = net(imgs)
                # pred_recon_img = torch.argmax(pred_recon_img, dim=1)
                # print("Masks Pred shape:", pred_recon_img.shape, "True Masks shape:", recon_img.shape)
                pcLossCriterion = percLoss(threshold_prob = 0.9, regularizer = regularizer, regularizer_weight = regularizer_weight, sampler = args.sp)
                # pcLossCriterion = nn.L1Loss()

                loss = weight_recon_loss * criterion(pred_recon_img, recon_img)
                # print(torch.squeeze(pred_mask).shape)
                # print(torch.mean(torch.squeeze(pred_mask), (1,2)).shape, imgs_percs)
                pcLoss = weight_percLoss * pcLossCriterion(pred_mask, imgs_percs)
                total_loss = loss + pcLoss
                epoch_loss += loss.item() + pcLoss.item()
                writer.add_scalar('Loss/train', total_loss.item(), global_step)

                pbar.set_postfix(**{'percLoss (batch)': pcLoss.item(), 'reconstruction loss': loss.item(),'total loss (batch)': total_loss.item()})

                optimizer.zero_grad()
                total_loss.backward()
                # pcLoss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # print("Finished Batch Training!\n")
                # print(global_step, n_train, batch_size)
                if global_step % (n_train // (1 * batch_size) + 1) == 0:
                # if global_step % (n_train // (100 * batch_size) + 1) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device, regularizer)
                    # scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation L1 loss: Total: {}, Mask: {}, Recon: {}'.format(val_score[0], val_score[1], val_score[2]))
                        writer.add_scalar('Loss/test', val_score[0], global_step)
                    else:
                        logging.info('Validation L1 loss: Total: {}, Mask: {}, Recon: {}'.format(val_score[0], val_score[1], val_score[2]))
                        writer.add_scalar('Loss/test', val_score[0], global_step)
                        writer.add_scalar('Recon/test', val_score[1], global_step)
                        writer.add_scalar('Perc/test', val_score[2], global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', recon_img, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(pred_recon_img) > 0.5, global_step)

        if save_cp:
            try:
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if (epoch + 1) % save_freq == 0:
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-sf', '--saveFreq', metavar='SF', type=int, default=1,
                        help='Save every sf epochs', dest='saveFreq')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-ms', '--manualSeed', metavar='MS', type=int, default=0,
                        help='Manual Seed for reproducability', dest='manual_seed')
    parser.add_argument('-r', '--regularization', metavar='R', type=str, default=None,
                        help='Regularizer', dest='reg')
    parser.add_argument('-rw', '--regularizer-weight', metavar='RW', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='rw')
    parser.add_argument('-sp', '--sampling', metavar='SP', type=str, nargs='?', default=None,
                        help='Whether to use the differentiable sampler to sample masks from probability values', dest='sp')
    parser.add_argument('-c', '--numClasses', metavar='C', type=int, default=1,
                        help='Number of classes in the dataset. If 1 or 2, use 1. Else use the number of classes.', dest='classes')
    parser.add_argument('-rd', '--rootDir', metavar='RD', type=str, default=Path().resolve().parent,
                        help='Root Directory for dataset', dest='rd')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logging.info(f'Using device {device}')
    logging.info(f'CPU workers available: {cpu_count()}')

    torch.manual_seed(args.manual_seed)
    logging.info(f'Set seed for reproducability: {args.manual_seed}')


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    torchsummary.summary(net, input_size=(3, 160, 160))
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(args=args,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  save_freq = args.saveFreq,
                  regularizer = args.reg,
                  regularizer_weight = args.rw)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
