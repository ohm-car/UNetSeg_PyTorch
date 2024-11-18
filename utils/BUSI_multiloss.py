import os
import sys
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import PIL
from PIL import Image, ImageOps
from torch.nn.functional import one_hot
import torchvision.transforms as transforms
import csv
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import square, erosion, binary_erosion
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

"""A custom dataset loader object. This dataset returns the same labels as the input"""

class BUSIDataset(Dataset):
    def __init__(self, root_dir, threshold = 50, im_res = 224, scale=1, preload = False):

        self.main_dir = os.path.join(root_dir, 'Datasets/Dataset_BUSI_with_GT/')
        # self.imgs_dir = os.path.join(root_dir, 'Datasets/VOCdevkit/VOC2012/JPEGImages/')
        # self.masks_dir = os.path.join(root_dir, 'Datasets/VOCdevkit/VOC2012/SegmentationClass/')
        self.file_list = self.get_filenames(self.main_dir)
        # print(self.file_list)

        self.num_classes = 1 + 1 #+1 for background
        self.im_res = (im_res, im_res)
        self.scale = scale
        self.threshold = threshold

        self.preload = preload
        self.transform = transforms.Compose([transforms.PILToTensor()])
        if self.preload:
            self.images, self.masks, self.eroded_masks, self.percs = self.load_data()
            logging.info(f'Loaded dataset with {len(self.file_list)} examples')

        # transform = transforms.Compose([transforms.PILToTensor()])
        # self.percsDict = self.getPercsDict(percs_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #             if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.file_list)} examples')

    def load_data(self):

        images, masks, eroded_masks, percs = list(), list(), list(), list()

        for filename in self.file_list:
            # print(filename)
            img = self.load_image(filename)
            mask = self.load_image_masks(filename)
            eroded_mask = self.eroded_image_masks(filename) if self.threshold != 0 else mask
            perc = self.get_perc(mask)

            images.append(img)
            masks.append(mask)
            eroded_masks.append(eroded_mask)
            percs.append(perc)

        return images, masks, eroded_masks, percs

    def load_image(self, filename):

        img_file = glob(self.main_dir + filename + '.*')
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {i}: {img_file}'
        T = Image.open(img_file[0])
        if T.mode != 'RGB':
            T = T.convert(mode = 'RGB')
        T = self.preprocess(T, self.transform)
        return T

    def load_image_masks(self, filename):

        mask_file = glob(self.main_dir + filename + '_mask' + '*')
        mask = torch.unsqueeze(torch.zeros(self.im_res), dim=0)
        for mf in mask_file:
            #Mask as torch Tensor
            M = Image.open(mf)
            if M.mode != '1':
                M = M.convert(mode = '1')
            M = self.preprocess_mask(M, self.transform)
            mask += M
        mask = torch.clamp(mask, max = 1.0)
        return mask
        # return masks

    def eroded_image_masks(self, filename):

        mask_file = glob(self.main_dir + filename + '_mask' + '*')
        # eroded_mask = np.expand_dims(np.zeros(self.im_res), axis=0)
        eroded_mask = torch.unsqueeze(torch.zeros(self.im_res), dim=0)
        for mf in mask_file:
            #Mask as np array
            M = imread(mf, as_gray = True)
            #resize np mask
            M = resize(M, self.im_res)
            e_M = self.eroded_mask(M)
            eroded_mask += e_M

        return eroded_mask

    def eroded_mask(self, mask):

        # o_pixels = np.sum(mask)
        e_mask = mask
        pixels = np.sum(e_mask)
        if self.threshold < 1.0:
            threshold = max(30, int(pixels * self.threshold))
        else:
            threshold = self.threshold
        # print(threshold, pixels)
        while pixels >= threshold:
            e_mask_t = erosion(e_mask, np.ones((3,3)))
            pixels = np.sum(e_mask_t)
            
            if pixels != 0:
                e_mask = e_mask_t
            
        # print(pixels)
        
        return torch.from_numpy(e_mask)
        # return e_mask


    def __len__(self):
        return len(self.file_list)

    # def getPercsDict(self, percs_dir):
        
    #     x = dict()
    #     f = open('utils/percs.csv', 'r')
    #     reader = csv.reader(f)

    #     for row in reader:
    #         x[row[0].split('/')[-1]] = float(row[1])
    #     # print(x)
    #     return x

    # def get_percs(self, mask):

    #     percs = list()

    #     for i in range(self.num_classes):

    #         percs.append(torch.mean((mask == i) * 1.0))

    #     return torch.tensor(percs)

    def get_perc(self, mask):

        perc = torch.mean((mask == 1) * 1.0)

        return torch.unsqueeze(perc, 0)

    def get_filenames(self, path):

        file_list = list()

        for i in os.listdir(os.path.join(path, 'benign')):
            fname = i.split('.')[0]
            if fname[-1] == ')':
                file_list.append(f'benign/{fname}')

        for i in os.listdir(os.path.join(path, 'malignant')):
            fname = i.split('.')[0]
            if fname[-1] == ')':
                file_list.append(f'malignant/{fname}')

        return file_list

    def preprocess_mask(self, pil_mask, transform):

        pil_mask = pil_mask.resize(self.im_res)

        imgM = transform(pil_mask)
        # imgM -= 1
        # imgM = (imgM > 0) * 1
        return imgM

    def preprocess(self, pil_img, transform):
        w, h = pil_img.size

        pil_img = pil_img.resize(self.im_res)

        imgT = transform(pil_img)

        # imgT = imgT.permute(2, 0, 1)
        imgT = imgT / 255

        return imgT

    def gen_partial_mask(self, mask, sq_to_center = 4):


        partial_mask = np.expand_dims(partial_mask, axis=0)
        return torch.tensor(partial_mask)


    # @classmethod
    # def preprocess(cls, pil_img, scale, isImage):
    #     w, h = pil_img.size
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small'
    #     pil_img = pil_img.resize((160, 160))

    #     img_nd = np.array(pil_img)

    #     if len(img_nd.shape) == 2:
    #         img_nd -= 1
    #         # img_nd = np.expand_dims(img_nd, axis=2)

    #     # if not isImage:
    #         # img_nd = cls.onehot_initialization(cls, img_nd)
    #         # img_nd -= 1
    #         # img_nd = (np.arange(img_nd.max()+1) == img_nd[...,None]).astype(int)
    #         # print(img_nd.shape)

    #     # HWC to CHW
        
    #     if isImage:
    #         img_trans = img_nd.transpose((2, 0, 1))
    #         if img_trans.max() > 1:
    #             img_trans = img_trans / 255
    #     else:
    #         img_trans = img_nd

    #     return img_trans

    # def processMask(self, pilmask):

    #     mask = np.asarray(pilmask)

    #     mask = np.sum(mask, axis = 2)
    #     mask = mask == 765

    #     return Image.fromarray(np.uint8(mask))

    # def all_idx(self, idx, axis):
    #     grid = np.ogrid[tuple(map(slice, idx.shape))]
    #     grid.insert(axis, idx)
    #     return tuple(grid)

    # def onehot_initialization(self, a):
    #     ncols = a.max()+1
    #     out = np.zeros(a.shape + (ncols,), dtype=int)
    #     out[self.all_idx(a, axis=2)] = 1
    #     return out

    def __getitem__(self, i):
        
        if self.preload:

            idx = self.file_list[i]
            T = self.images[i]
            M = self.masks[i]
            Mc = self.eroded_masks[i]
            P = self.percs[i]

            return {
                'image_ID': idx,
                'image': T,
                'reconstructed_image': T,
                'mask': M,
                'comp_mask': Mc,
                'mask_perc': P
            }
        else:

            idx = self.file_list[i]
            T = self.load_image(idx)
            M = self.load_image_masks(idx)
            Mc = self.eroded_image_masks(idx)
            P = self.get_perc(M)

            return {
            'image_ID': idx,
            'image': T,
            'reconstructed_image': T,
            'mask': M,
            'comp_mask': Mc,
            'mask_perc': P
        }

    # def __getitem__(self, i):
    #     idx = self.file_list[i]
    #     # print(self.imgs_dir, self.masks_dir, self.mask_suffix)

    #     img_file = glob(self.main_dir + idx + '.*')

    #     mask_file = glob(self.main_dir + idx + '_mask' + '.*')

    #     assert len(img_file) == 1, \
    #         f'Either no image or multiple images found for the ID {i}: {img_file}'

    #     assert len(mask_file) == 1, \
    #         f'Either no mask or multiple masks found for the ID {i}: {mask_file}'

    #     T = Image.open(img_file[0])
    #     if T.mode != 'RGB':
    #         T = T.convert(mode = 'RGB')

    #     T = self.preprocess(T, self.transform)
    #     # print("Image tensor type ", T)


    #     M = Image.open(mask_file[0])
    #     # print(M.mode)
    #     # assert M.mode == 'L' or M.mode == '1', \
    #     #     f'Error with file {mask_file}'

    #     M = self.preprocess_mask(M, self.transform)
    #     # print("Mask tensor type ", M)
    #     # print("Mask shape: ", M.shape)

    #     # M = M - ((M == 255) * 255)
    #     assert torch.max(M) == 1.0 and torch.min(M) == 0,\
    #         f'Check mask file'

    #     P = self.get_perc(M)
    #     # print(P)

    #     # Mp = M1 + M3
    #     # print(M.shape, torch.squeeze(M).shape, M.dtype)
    #     # Mp = torch.permute(one_hot(torch.squeeze(M), num_classes = self.num_classes), (2, 0, 1))
    #     # Mp = torch.permute(torch.squeeze(M), (2, 0, 1))
    #     # Mp = torch.squeeze(M)
    #     Mc = self.gen_partial_mask(M)

    #     assert Mc.size() == M.size(), \
    #         f'Shapes mismatch {Mc.size()} != {M.size()}'

    #     return {
    #         'image_ID': idx,
    #         'image': T,
    #         'reconstructed_image': T,
    #         'mask': M,
    #         'comp_mask': Mc,
    #         'mask_perc': P
    #     }
