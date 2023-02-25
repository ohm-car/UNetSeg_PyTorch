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
from utils.dataset import BasicDataset

import cv2

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=False)
    parser.add_argument('--videoPath', '-vp', metavar='INPUT', nargs='+',
                        help='filename/path of video', required=True)
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
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def getVideo(vidpath):

    cap = cv2.VideoCapture(vidpath)

    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # out = cv2.VideoWriter('data/videos_croppedFrames/demo_031.mp4', fourcc, 25, (500, 280))

    vid = []
    ctr = 0
    while(True):

        ret, frame = cap.read()
        if not ret:
            break

        if ret:
            vid.append(Image.fromarray(np.uint8(frame)))

        if ctr%10 == 0:
            print(ctr)
        ctr += 1
    cap.release()
    print("Success reading video:", len(vid))
    return vid

def postprocess(img, mask):

    vimg = np.asarray(img, np.uint8)
    vmask = np.asarray(mask, np.uint8)
    vmask = np.expand_dims(vmask, axis = 2)
    vmaskf = np.append(vmask, vmask, axis = 2)
    vmaskf = np.append(vmaskf, vmask, axis = 2)
    vmaskf = vmaskf * [0, 0, 255]
    # print(vmaskf)
    # print(vimg)
    alpha = 0.6

    # print(vmaskf.shape, type(vmaskf), alpha, vimg.shape)

    vop = np.zeros(shape = vimg.shape, dtype=np.uint8)

    cv2.addWeighted(np.asarray(vmaskf, np.uint8), alpha, vimg, 1 - alpha, 0, vop)

    return vop


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    videoPath = args.videoPath[0]
    print(videoPath, type(videoPath))
    # out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    # for i, fn in enumerate(in_files):
    #     logging.info("\nPredicting image {} ...".format(fn))

    #     img = Image.open(fn)

    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)

    #     if not args.no_save:
    #         out_fn = out_files[i]
    #         result = mask_to_image(mask)
    #         result.save(out_files[i])

    #         logging.info("Mask saved to {}".format(out_files[i]))

    #     if args.viz:
    #         logging.info("Visualizing results for image {}, close to continue ...".format(fn))
    #         plot_img_and_mask(img, mask)


    vid = getVideo(videoPath)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    outname = videoPath.split('/')[-1]
    print(os.path.abspath('../data/UNetPreds/' + outname))
    out = cv2.VideoWriter('../data/UNetPreds/' + outname, fourcc, 25, (500, 280))

    ctr = 0
    for frame in vid:

        mask = predict_img(net=net,
                           full_img=frame,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        # print(mask)

        fmask = postprocess(frame, mask)
        # print(fmask.shape)
        out.write(fmask)
        ctr += 1
        if(ctr%10 == 0):
            print(ctr)

    out.release()


