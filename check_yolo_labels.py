import cv2
import numpy as np
import glob
from utils.plots

img_folder= '/cluster/work/cvl/himeva/datasets/fastmri_yolo/train/images'

all_images= glob.glob(img_folder)

selected_image = all_images[0]
