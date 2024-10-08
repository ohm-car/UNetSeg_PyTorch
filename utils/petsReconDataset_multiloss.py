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

"""A custom dataset loader object. This dataset returns the same labels as the input"""

class PetsReconDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, percs_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.percsDict = self.getPercsDict(percs_dir)
        # print(self.percsDict)
        # self.images = self.load_images(imgs_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.images, self.percs = self.load_data(imgs_dir)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def getPercsDict(self, percs_dir):
        
        x = dict()
        print("Calling PercsDict")
        f = open('utils/percs.csv', 'r')
        reader = csv.reader(f)

        for row in reader:
            x[row[0].split('/')[-1].split('.')[0]] = float(row[1])
        # print(x)
        return x

    def get_filenames(self):
        return self.ids


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

    
    def preprocess(self, pil_img, transform):
        w, h = pil_img.size

        pil_img = pil_img.resize((160, 160))

        imgT = transform(pil_img)

        # imgT = imgT.permute(2, 0, 1)
        imgT = imgT / 255

        return imgT

    # def load_images(self, imgs_dir):

    #     temp_images = dict()

    #     transform = transforms.Compose([transforms.PILToTensor()])

    #     print('Loading Dataset')

    #     # for i in listdir(imgs_dir):
    #     for i in self.ids:

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

    def load_data(self, imgs_dir):

        temp_images = list()
        temp_percs = list()

        transform = transforms.Compose([transforms.PILToTensor()])

        print('Loading Dataset')

        # for i in listdir(imgs_dir):
        for i in self.ids:

            img_file = glob(self.imgs_dir + i + '.*')

            assert len(img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'

            T = Image.open(img_file[0])
            if T.mode != 'RGB':
                T = T.convert(mode = 'RGB')

            T = self.preprocess(T, transform)
            # print(i)
            temp_images.append(T)
            temp_percs.append(torch.Tensor([self.percsDict[i]]))

        print('Loaded Dataset')

        return temp_images, temp_percs

    # def __getitem__(self, i):

    #     idx = self.ids[i]
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

        return {
            # 'image_ID': img_file[0] + idx,
            'image': self.images[i],
            'reconstructed_image': self.images[i],
            'mask_perc': self.percs[i]
        }

    # def __getitem__(self, i):
    #     idx = self.ids[i]
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
