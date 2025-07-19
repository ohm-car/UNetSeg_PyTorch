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
import torch.nn.functional as F
import torchvision.transforms as transforms
import csv
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import square, erosion, binary_erosion
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
import cv2
import random

"""A custom dataset loader object. This dataset returns the same labels as the input"""

class KiTS_Dataset(Dataset):

    # Class Labels:
    # 0: Background
    # 1: Kidney
    # 2: Tumor
    # 3: Cyst
    # Considering Background, Kidney, and Tumor for this expt, so 3 classes.
    num_classes = 3

    def __init__(self, root_dir, file_list_path = None, threshold = 100, im_res = 512, scale=1, preload = False):

        self.main_dir = os.path.join(root_dir, 'Datasets/KiTS23_DL')
        random.seed(0)
        # self.imgs_dir = os.path.join(root_dir, 'Datasets/VOCdevkit/VOC2012/JPEGImages/')
        # self.masks_dir = os.path.join(root_dir, 'Datasets/VOCdevkit/VOC2012/SegmentationClass/')
        if file_list_path:
            tp_path = os.path.join(root_dir, 'Datasets/KiTS23_DL/KiTS_multiloss', file_list_path)
            self.file_list = self.get_filenames_from_file(tp_path)
        else:
            raise Exception("Variable file_list_path required.")
        # print("File List: ", self.file_list)

        # Class Labels:
        # 0: Background
        # 1: Kidney
        # 2: Tumor
        # 3: Cyst
        # Considering Background, Kidney, and Tumor for this expt, so 3 classes.

        # self.num_classes = 30
        # print("Classes: ", KiTS_Dataset.num_classes, self.num_classes)
        self.im_res = (im_res, im_res)  
        self.scale = scale
        self.threshold = threshold

        self.resized_files = list()

        self.preload = preload
        self.transform = transforms.Compose([transforms.PILToTensor()])
        if self.preload:
            self.images, self.masks, self.eroded_masks, self.percs = self.load_data()
            logging.info(f'Loaded dataset with {len(self.file_list)} examples')
            print(self.resized_files if len (self.resized_files) < 100 else len(self.resized_files))

        # transform = transforms.Compose([transforms.PILToTensor()])
        # self.percsDict = self.getPercsDict(percs_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #             if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.file_list)} examples')

    def np_one_hot(self, arr):

        return np.eye(self.num_classes)[arr]

    def resize_img(self, img):
        img = cv2.resize(img, self.im_res, interpolation = cv2.INTER_LINEAR)
        return img

    def resize_mask(self, mask):
        mask = cv2.resize(mask, self.im_res, interpolation = cv2.INTER_LINEAR)
        return mask

    def load_data(self):

        images, masks, eroded_masks, percs = list(), list(), list(), list()

        for filename in self.file_list:
            img = self.load_image(filename)
            np_mask = self.load_image_mask_numpy(filename)
            mask = self.load_image_mask(filename)
            eroded_mask = self.eroded_mask(np_mask) if self.threshold != 0 else mask
            perc = self.get_perc(mask)

            images.append(img)
            masks.append(mask)
            eroded_masks.append(eroded_mask)
            percs.append(perc)

        return images, masks, eroded_masks, percs

    def load_image(self, filename):

        img_file = glob(os.path.join(self.main_dir, 'images', filename + '*'))
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {filename}: {img_file}'
        T = np.load(img_file[0])

        # Check image size
        if T.shape != self.im_res:
            T = self.resize_img(T)
            self.resized_files.append(filename)

        # Rescaling to get this between 0 and 1
        T = T + 1024
        T = T / 4095

        T = torch.from_numpy(T)
        T = torch.unsqueeze(T, dim=0)
        return T

    def load_image_mask_numpy(self, filename):

        mask_file = glob(os.path.join(self.main_dir, 'gt_masks', filename + '*'))
        assert len(mask_file) == 1, \
            f'Either no image or multiple images found for the ID {filename}: {img_file}'
            #Mask as torch Tensor
        M = np.load(mask_file[0])

        # Check image size
        if M.shape != self.im_res:
            M = self.resize_mask(M)

        # Set edges to background class (ie 0) to ensure erosion works

        M[0,:] = 0
        M[-1,:] = 0
        M[:,0] = 0
        M[:,-1] = 0

        # Replace cysts annotations with kidney
        M[M == 3] = 1

        return M

    def load_image_mask(self, filename):

        iM = self.preprocess_mask(self.load_image_mask_numpy(filename))
        iM = iM.permute(2,0,1)

        return iM

    def eroded_mask(self, np_mask):

        np_mask_oh = self.np_one_hot(np_mask)

        # Remove background
        np_mask_oh = np_mask_oh[:,:,1:]

        # Since a tumor pixel is also a kidney pixel, add the tumor ones to the kidney
        np_mask_oh[:,:,0] = np_mask_oh[:,:,0] + np_mask_oh[:,:,1]

        er_mask = np.zeros(np_mask_oh.shape)

        #TODO: Code to erode the loaded mask

        for i in range(self.num_classes - 1):
            e_mask = np_mask_oh[:,:,i]
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
            er_mask[:,:,i] = e_mask
            
        eM = torch.from_numpy(er_mask)
        eM = eM.permute(2,0,1)
        return eM
        # return e_mask


    def __len__(self):
        return len(self.file_list)

    def get_perc(self, mask):

        perc = torch.mean(mask.float(), (1,2))

        return perc

    def get_all_slices_of_case(self, case_id):

        tp_slices = glob(os.path.join(self.main_dir, 'images', case_id) + '*')

        slices = list()

        for tp_slice in tp_slices:
            slices.append(tp_slice.split('/')[-1])

        return slices

    def get_filenames_from_file(self, path):

        # print("In function get_filenames_from_file")

        file_list = list()

        f = open(path)
        temp_fl = f.read().split(',')

        # Add code to get all slices from the given case number
        for k in range(len(temp_fl)):
            # subfiles = self.get_all_slices_of_case(temp_fl[k])
            subfiles = self.get_all_slices_of_case(temp_fl[k].replace("'", "").strip())
            for subfile in subfiles:
                file_list.append(subfile)

            # file_list.append(temp_fl[k].replace("'", "").strip())

        random.shuffle(file_list)
        return file_list

    def preprocess_mask(self, np_mask):

        # preProcess loaded segmentation mask as per the task, and return a torch tensor

        imgM = np_mask

        imgM = torch.from_numpy(imgM).long()
        imgM = F.one_hot(imgM, num_classes = self.num_classes)

        # Remove background
        imgM = imgM[:,:,1:]

        # Since a tumor pixel is also a kidney pixel, add the tumor ones to the kidney
        imgM[:,:,0] = imgM[:,:,0] + imgM[:,:,1]

        return imgM

    def preprocess(self, np_img):
        w, h = pil_img.size

        pil_img = pil_img.resize(self.im_res)

        imgT = transform(pil_img)

        # imgT = imgT.permute(2, 0, 1)
        imgT = imgT / 255

        return imgT

    def gen_partial_mask(self, mask, sq_to_center = 4):


        partial_mask = np.expand_dims(partial_mask, axis=0)
        return torch.tensor(partial_mask)


    def __getitem__(self, i):
        
        if self.preload:

            idx = self.file_list[i]
            T = self.images[i]
            M = self.masks[i]
            Mc = self.eroded_masks[i]
            P = self.percs[i]

            # print(T.shape, M.shape, Mc.shape, P.shape)

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
            M = self.load_image_mask(idx)
            _ = self.load_image_mask_numpy(idx)
            Mc = self.eroded_mask(_)
            P = self.get_perc(M)

            # print(T.shape, M.shape, Mc.shape, P.shape)

            return {
            'image_ID': idx,
            'image': T,
            'reconstructed_image': T,
            'mask': M,
            'comp_mask': Mc,
            'mask_perc': P
        }