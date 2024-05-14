from os.path import splitext
import os
import sys
from pathlib import Path
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import PIL
from PIL import Image, ImageOps
from torch.nn.functional import one_hot
import csv
import time
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms
from PIL import features

# print(features.pilinfo(supported_formats = True))

nas_dir = Path().resolve().parent
# nas_dir = Path().resolve().parent
# nas_dir_img = os.path.join(nas_dir, 'Datasets/petsData/images/')
# nas_dir_img = os.path.join(nas_dir, 'data/images/')
nas_dir_img = os.path.join(nas_dir, 'Datasets/VOCdevkit/VOC2012/')
nas_dir_img = os.path.join(nas_dir_img, 'SegmentationClass/')
scratch_dir = sys.argv[1]
# scratch_dir_img = os.path.join(scratch_dir, 'Datasets/petsData/images/')
scratch_dir_img = os.path.join(scratch_dir, 'data/images/')

transform = transforms.Compose([ 
    transforms.PILToTensor() 
])

# def preprocess(pil_img, scale, isImage):
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

def preprocess(pil_img):
    # w, h = pil_img.size

    imgT = pil_img.resize((160, 160))

    imgT = transform(pil_img)

    # imgT = imgT.permute(2, 0, 1)
    # imgT = imgT / 255

    return imgT


imlist = list()

nas_imgs = os.listdir(nas_dir_img)
print(str(len(nas_imgs)) + " Images found in NAS storage")
nas_time = time.time()
for i in nas_imgs:
	# print(i)
	T = Image.open(nas_dir_img + i)
	# if T.mode != 'RGB':
	# 	T = T.convert(mode = 'RGB')
		# T = transform(T)
	T = preprocess(T)
	T = T - ((T == 255) * 255)
	# print(T.shape)
	# T1 = torch.permute(one_hot(torch.squeeze(T), num_classes = 21), (2, 0, 1))
	imlist.append(T)
nas_time = time.time() - nas_time
print("NAS: Time taken to read " + str(len(nas_imgs)) + " images is " + str(nas_time) + " seconds.")

nas_time = time.time()
for i in imlist:
	# print(i)
	
	# if T.mode != 'RGB':
	# 	T = T.convert(mode = 'RGB')
		# T = transform(T)
	
	
	# print(T.shape)
	T1 = torch.permute(one_hot(torch.squeeze(i), num_classes = 21), (2, 0, 1))
nas_time = time.time() - nas_time
print("NAS: Time taken to read " + str(len(nas_imgs)) + " images is " + str(nas_time) + " seconds.")

# nas_imgs = os.listdir(nas_dir_img)
# print(str(len(nas_imgs)) + " Images found in NAS storage")
# nas_time = time.time()
# for i in nas_imgs:
# 	# print(i)
# 	try:
# 		T = read_image(nas_dir_img + i, ImageReadMode.RGB)
# 		# print(type(T))
# 	except Exception:
# 		pass
# 		# print(i, e)

# nas_time = time.time() - nas_time
# print("NAS: Time taken to read " + str(len(nas_imgs)) + " images is " + str(nas_time) + " seconds.")

# scatch_imgs = os.listdir(scratch_dir_img)
# print(str(len(scatch_imgs)) + " Images found in Scratch storage")
# scratch_time = time.time()
# for i in scatch_imgs:
# 	T = Image.open(scratch_dir_img + i)
# 	if T.mode != 'RGB':
# 		T = T.convert(mode = 'RGB')
# 	# T = preprocess(T, 1, True)
# scratch_time = time.time() - scratch_time
# print("Scratch: Time taken to read " + str(len(scatch_imgs)) + " images is " + str(scratch_time) + " seconds.")