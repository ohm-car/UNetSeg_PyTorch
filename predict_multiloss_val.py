import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.petsReconDataset_multiloss import PetsReconDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(PetsReconDataset.preprocess(full_img, scale_factor, isImage=True))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_im, pred_mask = net(img)

        print('ReconIM shape:', pred_im.shape)
        print('Mask shape:', pred_mask.shape)

        # if net.n_classes > 1:
        #     im_probs = F.softmax(output, dim=1)
        # else:
        #     im_probs = torch.sigmoid(output)

        im_probs = pred_im.squeeze(0)
        mask_probs = pred_mask.squeeze(0)


        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        im_probs = tf(im_probs.cpu())
        mask_probs = tf(mask_probs.cpu())
        full_im = im_probs.squeeze().cpu().numpy()
        print('mask_probs shape:', mask_probs.shape)
        full_mask = mask_probs.squeeze().cpu().numpy()

    return full_im, full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=False)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('-ms', '--manualSeed', metavar='MS', type=int, default=0,
                        help='Manual Seed for reproducability', dest='manual_seed')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            # print(pathsplit)
            out_files.append(("{}_OUT{}".format(pathsplit[0], pathsplit[1]), "{}_OUT_M{}".format(pathsplit[0], pathsplit[1])))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


# def mask_to_image(mask):
#     return Image.fromarray((mask * 255).astype(np.uint8))

def imrecon_to_image(img):
    
    # print(mask, mask * 255)
    img = img.transpose((1,2,0))
    # print(mask.shape)
    return Image.fromarray((img * 255).astype(np.uint8), 'RGB')

def mask_to_image(mask):
    
    #Change this function to get either actual probabilities or the final image; by setting a threshold probability.
    # print(mask.shape, type(mask))
    # mask = mask.transpose((1,2,0))
    # thres_mask = mask > 0.3
    # print(mask.shape, type(mask))
    mask = np.clip(mask, 0, 1)
    return Image.fromarray((mask * 255).astype(np.uint8), 'L')
    # return Image.fromarray((thres_mask * 240).astype(np.uint8), 'L')

def get_dataloaders(dataset, val_percent, batch_size):

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    # print(type(dataset),type(train),type(train.dataset))
    # print("Train IDs:", train.dataset.ids)
    # print("Val IDs:", val.dataset.ids)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    return train_loader, val_loader

def get_datasets(dataset, val_percent):

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    return train, val

def get_image_filenames(dataset):

    return dataset.get_filenames()

def mask_median(mask):
    median = torch.median(mask)
    return median

def rounded_mask(mask):
    mask = torch.round(mask)
    return mask


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    val_percent = 0.1
    batch_size = 2

    torch.manual_seed(args.manual_seed)

    #Code to get train and val dataloaders. Use PetsDataset from utils

    root_dir = Path().resolve().parent
    print("Root dir: ", root_dir)
    dir_img = os.path.join(root_dir, 'data/images/')
    dir_mask = os.path.join(root_dir, 'data/annotations/trimaps/')

    tm = datetime.now()
    dir_result = os.path.join(root_dir, 'Validation_Runs/{:02d}-{:02d}'.format(tm.month, tm.day))
    print("Result dir:", dir_result)
    os.makedirs(dir_result, exist_ok = True)

    petsDataset = PetsReconDataset(dir_img, dir_mask, args.scale)
    # print(petsDataset.__len__())

    train_loader, val_loader = get_dataloaders(petsDataset, val_percent, batch_size)

    # train_dataset, val_dataset = get_datasets(petsDataset, val_percent)

    net = UNet(n_channels=3, n_classes=1)
    logging.info("Loading model {}".format(args.model))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    #Create new csv

    f = open(dir_result + str('/AA_percentages.csv'), 'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(['Image ID', 'Actual perc', 'Val Predicted perc', 'Val Rounded perc', 'Median'])

    percs_gt = petsDataset.getPercsDict('')

    # for fname in val_dataset.get_filenames():

    #     net.eval()
    #     im_file = glob(self.dir_img + fname + '.*')
    #     print(str(im_file))
    #     assert len(im_file) == 1, \
    #         f'Either no image or multiple images found for the ID {idx}: {img_file}'
    #     img = Image.open(im_file[0])
    #     rec_im, mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)

    #     #Write predictions

    #     if not args.no_save:
    #         out_fn = '{}/{}'.format(dir_result, fname)
    #         result_im = imrecon_to_image(rec_im)
    #         result_mask = mask_to_image(mask)
    #         result_im.save('{}_RI.png'.format(out_fn))
    #         result_mask.save('{}_M.png'.format(out_fn))

    #         key = '{}.png'.format(fname)

    #         pred_perc = torch.mean(torch.squeeze(mask), (1,2))
    #         pred_perc = torch.unsqueeze(pred_perc, 1)
    #         csvwriter.writerow([str(fname), str(percs_gt[key]), str(pred_perc)])

    #         logging.info("Mask saved to {}".format(out_files[i]))


    for batch in val_loader:
        net.eval()
        imgs = batch['image']
        imgs = imgs.to(device=device, dtype=torch.float32)
        for i in range(batch_size):
            with torch.no_grad():

                # print(imgs.shape, imgs[i].shape)
                img = torch.unsqueeze(imgs[i], 0)
                # print(img.shape)
                # print(type(imgs[i]))
                # img = petsDataset.preprocess(imgs[i], args.scale, isImage=True)
                pred_recon_img, pred_mask = net(img)

                # print('ReconIM shape:', pred_recon_img.shape)
                # print('Mask shape:', pred_mask.shape)

                # if net.n_classes > 1:
                #     im_probs = F.softmax(output, dim=1)
                # else:
                #     im_probs = torch.sigmoid(output)

                # im_probs = pred_recon_img.squeeze(0)
                # mask_probs = pred_mask.squeeze(0)


                # tf = transforms.Compose(
                #     [
                #         transforms.ToPILImage(),
                #         transforms.Resize(imgs.size[1]),
                #         transforms.ToTensor()
                #     ]
                # )

                # im_probs = tf(im_probs.cpu())
                # mask_probs = tf(mask_probs.cpu())
                # full_im = im_probs.squeeze().cpu().numpy()
                # # print('mask_probs shape:', mask_probs.shape)
                # full_mask = mask_probs.squeeze().cpu().numpy()

            fname = batch['image_ID'][i]
            # print(fname)
            out_fn = '{}/{}'.format(dir_result, fname)
            # print(out_fn)
            # print(pred_mask.cpu().numpy().shape)
            result_im = imrecon_to_image(pred_recon_img.squeeze().cpu().numpy())
            result_mask = mask_to_image(pred_mask.squeeze().cpu().numpy())
            result_median = mask_median(pred_mask)
            result_rounded = rounded_mask(pred_mask)
            result_rounded = mask_to_image(result_rounded.squeeze().cpu().numpy())

            result_im.save('{}_RI.png'.format(out_fn))
            result_mask.save('{}_M.png'.format(out_fn))
            result_rounded.save('{}_RM.png'.format(out_fn))

            key = '{}.png'.format(fname)
            print(key)
            # print(list(percs_gt.items())[:3])

            pred_perc = torch.mean(torch.squeeze(pred_mask, 0), (1,2))
            pred_perc = torch.unsqueeze(pred_perc, 1)

            pred_perc_rounded = torch.mean(torch.squeeze(result_rounded, 0), (1,2))
            pred_perc_rounded = torch.unsqueeze(pred_perc_rounded, 1)
            csvwriter.writerow([str(fname), str(percs_gt[key]), str(pred_perc.item()), 
                str(pred_perc_rounded.item()), str(result_median.item())])

            logging.info("Mask saved to {}_M.png".format(out_fn))


    print("Validation Run Complete")
    f.close()


    # out_files = get_output_filenames(args)

    # net = UNet(n_channels=3, n_classes=1)

    # logging.info("Loading model {}".format(args.model))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
    # logging.info(f'Using device {device}')
    # net.to(device=device)
    # net.load_state_dict(torch.load(args.model, map_location=device))

    # logging.info("Model loaded !")

    # for i, fn in enumerate(in_files):
    #     logging.info("\nPredicting image {} ...".format(fn))

    #     img = Image.open(fn)

    #     rec_im, mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)

    #     # print(type(mask), mask.shape)
    #     # mask = np.argmax(mask, axis = 0)
    #     # print(mask.shape)

    #     # Code to print true percentage and predicted percentage

    #     if not args.no_save:
    #         out_fn = out_files[i]
    #         result_im = imrecon_to_image(rec_im)
    #         result_mask = mask_to_image(mask)
    #         result_im.save(out_files[i][0])
    #         result_mask.save(out_files[i][1])

    #         logging.info("Mask saved to {}".format(out_files[i]))

    #     if args.viz:
    #         logging.info("Visualizing results for image {}, close to continue ...".format(fn))
    #         plot_img_and_mask(img, mask)
