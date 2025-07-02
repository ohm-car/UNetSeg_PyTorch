#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import nibabel as nib
import PIL
import os
import sys
import matplotlib.pyplot as plt
import re
from pathlib import Path


# In[ ]:


drnm = os.path.dirname
cases_path = Path(os.getcwd()).parent.parent / 'KiTS23/kits23/dataset'
# cases_path = Path(z).parent.parent / 'KiTS23/KiTS_subset'
print(cases_path)


# In[ ]:


cases = os.listdir(cases_path)
try:
    cases.remove('kits23.json')
except Exception as e:
    print(e)


# In[ ]:


def get_slice_idx(instance):
    
    nib_img = nib.load(os.path.join(case_path, 'instances', instance))
    img = nib_img.get_fdata()
    slice_idx = np.argmax(np.sum(img, axis=(1,2)))
    return slice_idx


# In[ ]:


save_path = os.path.join(drnm(os.getcwd()), 'Datasets/KiTS23_DL')
print(save_path)


# In[ ]:


for case in cases:
    print(case)
    case_path = os.path.join(cases_path, case)
    t_list = os.listdir(os.path.join(case_path, 'instances'))
    r_list = list()
    
    for j in range(len(t_list)):
        t_ = t_list[j]
        if t_[-8] in ['2', '3']:
            r_list.append(t_)
    
    for e in r_list:
        t_list.remove(e)

    slice_indices = set()
    
    for instance in t_list:
        slice_idx = get_slice_idx(instance)
        slice_indices.add(slice_idx)
    
    img_f = nib.load(os.path.join(case_path, 'imaging.nii.gz'))
    imaging = img_f.get_fdata()
    imaging_dtype = img_f.header.get_data_dtype()
    seg_f = nib.load(os.path.join(case_path, 'segmentation.nii.gz'))
    segmentation = seg_f.get_fdata()
    segmentation_dtype = seg_f.header.get_data_dtype()
    
    for s in slice_indices:
        
        image = imaging[s]
        mask = segmentation[s]
        image_ogdt = image.astype(imaging_dtype)
        mask_ogdt = mask.astype(segmentation_dtype)
        
        np.save(os.path.join(save_path, 'images', f'{case}_slice_{s}.npy'), image_ogdt)
        np.save(os.path.join(save_path, 'gt_masks', f'{case}_slice_{s}.npy'), mask_ogdt)


# In[ ]:


z = os.getcwd()
print(z)


# In[ ]:


drnm = os.path.dirname
print(drnm(drnm(z)), z)

