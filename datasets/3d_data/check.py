import matplotlib.pyplot as plt
import os
import pydicom
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
img = sitk.GetArrayFromImage(sitk.ReadImage(r"G:\datasets\nodule_patches64\0\UCT202212280133_nodule.nii.gz"))
print(img.shape)
# plt.imshow(img[101], cmap='gray')
plt.show()