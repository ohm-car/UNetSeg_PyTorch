import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.petsReconSegDataset import PetsReconSegDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(PetsReconSegDataset.preprocessIM(full_img, scale_factor, isImage=True))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_recon_img, pred_mask = net(img)
        # pred_recon_img = net(img)

        print('ReconIM shape:', pred_recon_img.shape)
        print('Mask shape:', pred_mask.shape)

        # if net.n_classes > 1:
        #     im_probs = F.softmax(output, dim=1)
        # else:
        #     im_probs = torch.sigmoid(output)

        im_probs = pred_recon_img.squeeze(0)
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
        print('im_probs shape:', im_probs.shape)
        full_mask = mask_probs.squeeze().cpu().numpy()

    return full_im, full_mask
    # return full_im


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

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

    print(out_files)
    return out_files


# def mask_to_image(mask):
#     return Image.fromarray((mask * 255).astype(np.uint8))

def imrecon_to_image(recon_im):
    
    # print(recon_im, recon_im * 255)
    recon_im = recon_im.transpose((1,2,0))
    # print(recon_im.shape)
    return Image.fromarray((recon_im * 255).astype(np.uint8), 'RGB')

def mask_to_image(mask):
    
    print(mask.shape, type(mask))

    mask = np.argmax(mask, axis = 0)
    # mask = mask.transpose((1,2,0))
    # print(mask.shape, type(mask))
    return Image.fromarray((mask * 64).astype(np.uint8), 'L')


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=3)

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

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        rec_im, mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # rec_im = predict_img(net=net,
        #                    full_img=img,
        #                    scale_factor=args.scale,
        #                    out_threshold=args.mask_threshold,
        #                    device=device)

        # print(type(mask), mask.shape)
        # mask = np.argmax(mask, axis = 0)
        print(mask.shape)
        print(rec_im.shape)
        if not args.no_save:
            out_fn = out_files[i]
            result_im = imrecon_to_image(rec_im)
            result_mask = mask_to_image(mask)
            result_im.save(out_files[i][0])
            result_mask.save(out_files[i][1])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
