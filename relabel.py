import os, glob
import numpy as np
import cv2
import ntpath
from tqdm import tqdm

class Relabel:

    def __init__(self):
        self.ADE20K_DATA_DIR = './data/ADEChallengeData2016/'
        self.ADE20K_OUT_DIR = './data/LB_ADE20K/'

        self.ANNOTATIONS_TRAINING_DIR = os.path.join(self.ADE20K_DATA_DIR, 'annotations/training')
        self.ANNOTATIONS_VALIDATION_DIR = os.path.join(self.ADE20K_DATA_DIR, 'annotations/validation')

        self.OUT_TRAINING_DIR = os.path.join(self.ADE20K_OUT_DIR, 'annotations/training')
        self.OUT_VALIDATION_DIR = os.path.join(self.ADE20K_OUT_DIR, 'annotations/validation')

        self.map_list = {0: [1,9,15,33,43,44,145], 1: [4,7,12,14,29,30,53,55], 2: [6], 3: [13]}

    def remap_single(self, input, output):
        img = cv2.imread(input, 0)
        mask_list = []
        for label in self.map_list.values():
            mask_list.append(np.isin(img,label))
        
        result = np.full(img.shape,4) # By default is label '4'
        for i, mask in enumerate(mask_list):
            result[mask == 1] = i
        cv2.imwrite(output, result)

    def run(self):

        if not os.path.exists(self.OUT_TRAINING_DIR):
            os.makedirs(self.OUT_TRAINING_DIR)
        
        if not os.path.exists(self.OUT_VALIDATION_DIR):
            os.makedirs(self.OUT_VALIDATION_DIR)

        # Remap training images
        print("Relabeling training images")
        for filename in tqdm(glob.glob(os.path.join(self.ANNOTATIONS_TRAINING_DIR,'*.png'))):
            output = os.path.join(self.OUT_TRAINING_DIR, ntpath.basename(filename))
            self.remap_single(filename, output)

        # Remap validation images
        print("Relabeling validation images")
        for filename in tqdm(glob.glob(os.path.join(self.ANNOTATIONS_VALIDATION_DIR,'*.png'))):
            output = os.path.join(self.OUT_VALIDATION_DIR, ntpath.basename(filename))
            self.remap_single(filename, output)

if __name__ == "__main__":
    relabel = Relabel()
    relabel.run()