import numpy as np
import csv
from PIL import Image

path = ''

gt_c = Image.open(path)

gt_n = np.array(gt_c)

gt_n[gt_n == 255] = 0

#Replace all animals to idx 3
gt_n[(gt_n == 3) | (gt_n == 8) | (gt_n == 10) | (gt_n == 12) | (gt_n == 13) | (gt_n == 17)] = 3
#Replace all vehicles to idx 2
gt_n[(gt_n == 1) | (gt_n == 2) | (gt_n == 4) | (gt_n == 6) | (gt_n == 7) | (gt_n == 14)] = 2
#Replace all persons to idx 1
gt_n[gt_n == 15] = 1
#Rest are all 0
gt_n[gt_n > 3] = 0

