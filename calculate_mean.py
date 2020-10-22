import os, glob, cv2
import numpy as np
from tqdm import tqdm

OUTPUT_DATA_DIR = '/media/shan/Data/lb-dataset/segmentation/LBFREESPACE/images/training'

mean_list = []

for filename in tqdm(glob.glob(os.path.join(OUTPUT_DATA_DIR,'*.jpg'))):
    image = cv2.imread(filename)
    flatten = image.reshape((-1,3))
    mean = np.mean(flatten, axis=0)
    mean_list.append(mean)

mean_list = np.array(mean_list)
mean = np.mean(mean_list, axis=0)
print(mean)