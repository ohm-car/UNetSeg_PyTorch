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
import skimage
from skimage.morphology import square, erosion, binary_erosion
from skimage.color import rgb2gray
from skimage import io
from skimage.transform import resize
"""A custom dataset loader object. This dataset returns the same labels as the input"""
"""This dataset class preloads the entire dataset into the GPU memory (line 33, load_data() function). Not the best practice, but this is useful in
conditions where fetching batches from the disk becomes the bottleneck rather than the training process itself."""

class PetsDataset(Dataset):
    def __init__(self, root_dir, threshold = 50, im_res = 224, scale=1, preload = False):
        
        self.main_dir = os.path.join(root_dir, 'Datasets/petsData/')
        self.imgs_dir = os.path.join(self.main_dir, 'images/')
        self.masks_dir = os.path.join(self.main_dir, 'annotations/trimaps/')

        self.scale = scale
        self.num_classes = 1 + 1 #+1 for background
        self.percsDict = self.getPercsDict()
        self.im_res = (im_res, im_res)
        self.im_pad = (im_res - 1, im_res - 1)
        self.threshold = threshold
        self.preload = preload
        print(self.im_res, self.im_pad)

        self.transform = transforms.Compose([transforms.PILToTensor()])
        # print(self.percsDict)
        # self.images = self.load_images(imgs_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.file_list = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]

        if self.preload:
            self.images, self.masks, self.eroded_masks, self.percs = self.load_data()
            logging.info(f'Loaded dataset with {len(self.file_list)} examples')

        logging.info(f'Creating dataset with {len(self.file_list)} examples')

    def __len__(self):
        return len(self.file_list)

    # EDIT THE BELOW load_data() FUNCTION TO THE USE CASE. THE NEXT FUNCTION IS TO BE DEPRECATED
    def load_data(self):

        images, masks, eroded_masks, percs = list(), list(), list(), list()

        for filename in self.file_list:
            # print(filename)
            img = self.load_image(filename)
            mask = self.load_image_masks(filename)
            eroded_mask = self.eroded_image_masks(filename) if self.threshold != 0 else mask
            perc = self.get_perc(filename)

            images.append(img)
            masks.append(mask)
            eroded_masks.append(eroded_mask)
            percs.append(perc)

        return images, masks, eroded_masks, percs

    def load_image(self, filename):

        img_file = glob(self.imgs_dir + filename + '.*')
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {i}: {img_file}'
        T = Image.open(img_file[0])
        if T.mode != 'RGB':
            T = T.convert(mode = 'RGB')
        T = self.preprocess(T, self.transform)
        return T

    def load_image_masks(self, filename):

        mask_file = glob(self.masks_dir + filename + '.*')
        assert len(mask_file) == 1, \
                f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        M = Image.open(mask_file[0])

        assert M.mode == 'L' or M.mode == '1', \
                f'Error with file {mask_file}'

        M = self.preprocess_mask(M, self.transform)

        M1 = ((M == 1) * 1).float()
        M2 = ((M == 2) * 1).float()
        M3 = ((M == 3) * 1).float()

        mask = M1 + M3

        # mask = torch.clamp(mask, max = 1.0)
        return mask

    def eroded_image_masks(self, filename):

        # TODO -- Erosion is currently NOT WORKING as expected. Fix that first
        # Issues are with the resize function - need to inspect and resize appropriately
        # so that erosion will work as expected

        mask_file = glob(self.masks_dir + filename + '.*')
        assert len(mask_file) == 1, \
                f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # eroded_mask = np.expand_dims(np.zeros(self.im_res), axis=0)
        # eroded_mask = torch.unsqueeze(torch.zeros(self.im_res), dim=0)
        #Mask as np array
        # M = imread(mask_file[0], as_gray = True)
        M = io.imread(mask_file[0])


        # assert 1 in M, \
        #     f'Failed for file:{mask_file}'
        # assert 2 in M,\
        #     f'Failed for file:{mask_file}'
        # assert 3 in M,\
        #     f'Failed for file:{mask_file}'

        #resize np mask

        M1 = ((M == 1) * 1.0)
        M2 = ((M == 2) * 1.0)
        M3 = ((M == 3) * 1.0)

        Mp = M1 + M3

        if 1 not in Mp:
            print(mask_file, "Check 1")
            print(0 in Mp, 1 in Mp, 2 in Mp, 3 in Mp)

        # Mp = resize(Mp, self.im_res)
        Mp = resize(Mp, self.im_pad)
        Mp = np.pad(Mp, ((1,1),(1,1)), 'constant', constant_values=(0))

        # assert 0 in Mp
        # assert 1 in Mp
        # assert 2 not in Mp
        # assert 3 not in Mp

        if 1 not in Mp:
            print(mask_file, "Check 2")
            print(0 in Mp, 1 in Mp, 2 in Mp, 3 in Mp)

        # M1 = ((M == 1) * 1.0)
        # M2 = ((M == 2) * 1.0)
        # M3 = ((M == 3) * 1.0)

        #Sanity Checks on lines

        eroded_mask = self.eroded_mask(Mp)

        return eroded_mask

    def eroded_mask(self, mask):
        
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

    def getPercsDict(self):
        
        x = dict()
        print("Calling PercsDict")
        f = open('utils/percs.csv', 'r')
        reader = csv.reader(f)

        for row in reader:
            x[row[0].split('/')[-1].split('.')[0]] = float(row[2])
        # print(x)
        return x

    def get_perc(self, filename):
        return torch.Tensor([self.percsDict[filename]])

    def get_filenames(self):
        return self.file_list

    
    def preprocess(self, pil_img, transform):
        w, h = pil_img.size

        pil_img = pil_img.resize(self.im_res)

        imgT = transform(pil_img)

        # imgT = imgT.permute(2, 0, 1)
        imgT = imgT / 255

        return imgT

    def preprocess_mask(self, pil_mask, transform):

        pil_mask = pil_mask.resize(self.im_res)

        imgM = transform(pil_mask)
        # imgM -= 1
        # imgM = (imgM > 0) * 1
        return imgM.float()

    # def load_images(self, imgs_dir):

    #     temp_images = dict()

    #     transform = transforms.Compose([transforms.PILToTensor()])

    #     print('Loading Dataset')

    #     # for i in listdir(imgs_dir):
    #     for i in self.file_list:

    #         img_file = glob(self.imgs_dir + i + '.*')

    #         assert len(img_file) == 1, \
    #             f'Either no image or multiple images found for the ID {idx}: {img_file}'

    #         T = Image.open(img_file[0])
    #         if T.mode != 'RGB':
    #             T = T.convert(mode = 'RGB')

    #         T = self.preprocess(T, transform)
    #         # print(i)
    #         temp_images[i] = T

    #     print('Loaded Dataset')

    #     return temp_images

    # def load_data(self, imgs_dir, masks_dir):

    #     temp_images = list()
    #     temp_masks = list()
    #     temp_percs = list()

    #     transform = transforms.Compose([transforms.PILToTensor()])

    #     print('Loading Dataset')

    #     # for i in listdir(imgs_dir):
    #     for i in self.file_list:

    #         img_file = glob(self.imgs_dir + i + '.*')
    #         mask_file = glob(self.masks_dir + i + '.*')

    #         assert len(img_file) == 1, \
    #             f'Either no image or multiple images found for the ID {idx}: {img_file}'

    #         assert len(mask_file) == 1, \
    #             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

    #         T = Image.open(img_file[0])
    #         if T.mode != 'RGB':
    #             T = T.convert(mode = 'RGB')

    #         T = self.preprocess(T, transform)
    #         # print("Image tensor type ", T)


    #         M = Image.open(mask_file[0])
    #         # print(M.mode)
    #         assert M.mode == 'L' or M.mode == '1', \
    #             f'Error with file {mask_file}'

    #         M = self.preprocess_mask(M, transform)
    #         # print("Mask tensor type ", M)

    #         M1 = ((M == 1) * 1).float()
    #         M2 = ((M == 2) * 1).float()
    #         M3 = ((M == 3) * 1).float()

    #         Mp = M1 + M3

    #         # print(i)
    #         temp_images.append(T)
    #         temp_masks.append(Mp)
    #         temp_percs.append(torch.Tensor([self.percsDict[i]]))
    #         # print(i, torch.mean(torch.Tensor(M1)), torch.mean(torch.Tensor(M2)), torch.mean(torch.Tensor(M3)), torch.Tensor([self.percsDict[i]]))

    #     print('Loaded Dataset')

    #     return temp_images, temp_masks, temp_percs

    # def __getitem__(self, i):

    #     idx = self.file_list[i]
    #     # print(self.imgs_dir, self.masks_dir, self.mask_suffix)
    #     mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
    #     # print(mask_file[0])
    #     img_file = glob(self.imgs_dir + idx + '.*')

    #     # mask = Image.open(mask_file[0])
    #     mask_percF = mask_file[0].split('/')[-1]
    #     # print(mask.size, mask.mode)
    #     # mask = self.processMask(mask)

    #     img = self.images[idx]
    #     # mask = self.preprocess(mask, self.scale, isImage = False)
    #     maskPerc = self.percsDict[mask_percF]

    #     return {
    #         'image_ID': img_file[0] + idx,
    #         'image': img,
    #         'reconstructed_image': img,
    #         'mask_perc': torch.Tensor([maskPerc])
    #     }

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
            P = self.get_perc(idx)

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
    #     mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
    #     # print(mask_file[0])
    #     img_file = glob(self.imgs_dir + idx + '.*')

    #     assert len(mask_file) == 1, \
    #         f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    #     assert len(img_file) == 1, \
    #         f'Either no image or multiple images found for the ID {idx}: {img_file}'
    #     mask = Image.open(mask_file[0])
    #     mask_percF = mask_file[0].split('/')[-1]
    #     # print(mask.size, mask.mode)
    #     # mask = self.processMask(mask)
    #     img = Image.open(img_file[0])
    #     if img.mode != 'RGB':
    #         # print(img_file, img.mode)
    #         img = img.convert(mode = 'RGB')
    #         # print(img_file, img.mode)
    #     # print(img.size)
    #     # print(type(mask))

    #     assert img.size == mask.size, \
    #         f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

    #     img = self.preprocess(img)
    #     # mask = self.preprocess(mask, self.scale, isImage = False)
    #     maskPerc = self.percsDict[mask_percF]

    #     return {
    #         'image_ID': img_file[0] + idx,
    #         # 'image': torch.from_numpy(img).type(torch.FloatTensor),
    #         'image': img,
    #         # 'reconstructed_image': torch.from_numpy(img).type(torch.FloatTensor),
    #         'reconstructed_image': img,
    #         'mask_perc': torch.Tensor([maskPerc])
    #     }
