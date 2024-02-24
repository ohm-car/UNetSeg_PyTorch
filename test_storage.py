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

print(sys.argv, '\n')

nas_dir = Path().resolve().parent
nas_dir_img = os.path.join(nas_dir, 'Datasets/petsData/images/')

scratch_dir = sys.argv[1]
scratch_dir_img = os.path.join(scratch_dir, 'Datasets/petsData/images/')

nas_imgs = os.listdir(nas_dir_img)
print(str(len(nas_imgs)) + " Images found in NAS storage")
nas_time = time.time()
for i in nas_imgs:
	T = Image.open(nas_dir_img + i)
nas_time = time.time() - nas_time
print("NAS: Time taken to read " + str(len(nas_imgs)) + " images is " + str(nas_time) + " seconds.")

scatch_imgs = os.listdir(scratch_dir_img)
print(str(len(scatch_imgs)) + " Images found in Scratch storage")
scratch_time = time.time()
for i in scatch_imgs:
	T = Image.open(scratch_dir_img + i)
scratch_time = time.time() - scratch_time
print("Scratch: Time taken to read " + str(len(scatch_imgs)) + " images is " + str(scratch_time) + " seconds.")