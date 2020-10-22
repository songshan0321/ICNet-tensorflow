import os, glob
import numpy as np
import cv2
import ntpath
from tqdm import tqdm
import argparse
from shutil import copyfile

from utils.visualize import decode_labels

class Relabel:

    def __init__(self, remaining_class = 1):
        self.ADE20K_DATA_DIR = './data/ADEChallengeData2016/' # Input dateset
        self.ADE20K_OUT_DIR = './data/ADEFreeSpace/' # Output dateset

        self.ANNOTATIONS_TRAINING_DIR = os.path.join(self.INPUT_DATA_DIR, 'annotations/training')
        self.ANNOTATIONS_VALIDATION_DIR = os.path.join(self.INPUT_DATA_DIR, 'annotations/validation')

        self.IMAGES_TRAINING_DIR = os.path.join(self.INPUT_DATA_DIR, 'images/training')
        self.IMAGES_VALIDATION_DIR = os.path.join(self.INPUT_DATA_DIR, 'images/validation')

        self.OUT_ANNOTATIONS_TRAINING_DIR = os.path.join(self.OUTPUT_DATA_DIR, 'annotations/training')
        self.OUT_ANNOTATIONS_VALIDATION_DIR = os.path.join(self.OUTPUT_DATA_DIR, 'annotations/validation')

        self.OUT_VISUALIZATIONS_TRAINING_DIR = os.path.join(self.OUTPUT_DATA_DIR, 'visualizations/training')
        self.OUT_VISUALIZATIONS_VALIDATION_DIR = os.path.join(self.OUTPUT_DATA_DIR, 'visualizations/validation')

        self.OUT_IMAGES_TRAINING_DIR = os.path.join(self.OUTPUT_DATA_DIR, 'images/training')
        self.OUT_IMAGES_VALIDATION_DIR = os.path.join(self.OUTPUT_DATA_DIR, 'images/validation')

        # Define relabel mapping here, e.g. {0:[1,2], 1:[3,4]}
        # 4: flooring, 12: pavement, 29: carpet
        self.map_list = {0: [4, 12, 29]}

        self.remaining_class = len(self.map_list)

        self.train_list = []
        self.val_list = []

    def remap_single(self, image):
        """
        Remap a single segmentation image
        image  : Input image
        """

        # Only relabel image contains any interested label in self.map_list
        contain = False
        for key, val in self.map_list.items():
            for label in val:
                if label in image:
                    contain = True
        
        if not contain:
            return None, None

        mask_list = {}
        for key, val in self.map_list.items():
            mask_list[key] = np.isin(image,val)
        
        annotation = np.full(image.shape, self.remaining_class)
        for key, val in mask_list.items():
            annotation[val == 1] = key

        # visualization = decode_labels(annotation, image.shape, len(self.map_list))[0]
        
        return annotation, None

    def run(self):
        """
        Takes in segmented images of Training and Validation set and output a new dataset
        """

        if not os.path.exists(self.OUT_ANNOTATIONS_TRAINING_DIR):
            os.makedirs(self.OUT_ANNOTATIONS_TRAINING_DIR)
        
        if not os.path.exists(self.OUT_ANNOTATIONS_VALIDATION_DIR):
            os.makedirs(self.OUT_ANNOTATIONS_VALIDATION_DIR)

        # Remap training images
        print("Relabeling training images")
        for filename in tqdm(glob.glob(os.path.join(self.ANNOTATIONS_TRAINING_DIR,'*.png'))):
            input_image_file = os.path.join(self.IMAGES_TRAINING_DIR, ntpath.basename(filename)[:-4] + '.jpg')
            output_image_file = os.path.join(self.OUT_IMAGES_TRAINING_DIR, ntpath.basename(filename)[:-4] + '.jpg')
            output_annotation_file = os.path.join(self.OUT_ANNOTATIONS_TRAINING_DIR, ntpath.basename(filename))
            output_visualization_file = os.path.join(self.OUT_VISUALIZATIONS_TRAINING_DIR, ntpath.basename(filename))

            annotation = cv2.imread(filename, 0)
            relabeled_annotation, visualization = self.remap_single(annotation)
            if relabeled_annotation is not None:
                cv2.imwrite(output_annotation_file, relabeled_annotation)
                # cv2.imwrite(output_visualization_file, visualization)
                copyfile(input_image_file, output_image_file)
                line = 'images/training/{} annotations/training/{}\n'.format(ntpath.basename(filename)[:-4]+'.jpg', ntpath.basename(filename))
                self.train_list.append(line)
        
        self.train_list.sort()
        f = open(os.path.join(self.OUTPUT_DATA_DIR, 'train_list.txt'), 'w')
        for line in self.train_list:
            f.write(line)
        f.close()

        # Remap validation images
        print("Relabeling validation images")
        for filename in tqdm(glob.glob(os.path.join(self.ANNOTATIONS_VALIDATION_DIR,'*.png'))):
            input_image_file = os.path.join(self.IMAGES_VALIDATION_DIR, ntpath.basename(filename)[:-4] + '.jpg')
            output_image_file = os.path.join(self.OUT_IMAGES_VALIDATION_DIR, ntpath.basename(filename)[:-4] + '.jpg')
            output_annotation_file = os.path.join(self.OUT_ANNOTATIONS_VALIDATION_DIR, ntpath.basename(filename))
            output_visualization_file = os.path.join(self.OUT_VISUALIZATIONS_VALIDATION_DIR, ntpath.basename(filename))

            annotation = cv2.imread(filename, 0)
            relabeled_annotation, visualization = self.remap_single(annotation)
            if relabeled_annotation is not None:
                cv2.imwrite(output_annotation_file, relabeled_annotation)
                # cv2.imwrite(output_visualization_file, visualization)
                copyfile(input_image_file, output_image_file)
                line = 'images/validation/{} annotations/validation/{}\n'.format(ntpath.basename(filename)[:-4]+'.jpg', ntpath.basename(filename))
                self.val_list.append(line)

        self.val_list.sort()
        f = open(os.path.join(self.OUTPUT_DATA_DIR, 'val_list.txt'), 'w')
        for line in self.val_list:
            f.write(line)
        f.close()

if __name__ == "__main__":

    relabel = Relabel()
    relabel.run()