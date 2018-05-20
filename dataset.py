import re
import random
import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
from glob import glob


class Dataset():

    def __init__(self, data_folder, image_shape):
        self.__data_folder = data_folder
        self.__image_shape = image_shape
        self.__h = 32 * (self.__image_shape[0] // 32)
        self.__w = 32 * (self.__image_shape[1] // 32)

    def __filter_labels(self, label_image):
        label_image[label_image == 6] = 7
        label_image[np.isin(label_image, [7,10], invert=True)] = 0
        label_image[490:,:,:][label_image[490:,:,:] != 7] = 0
        return label_image

    def get_batches(self, batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(self.__data_folder, 'CameraRGB', '*.png'))
        label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(self.__data_folder, 'CameraSeg', '*.png'))}
        background_color = np.array([0, 0, 0])
        road_color = np.array([7, 0, 0])
        vehicle_color = np.array([10, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imread(image_file)
                gt_image = scipy.misc.imread(gt_image_file)
                gt_image = self.__filter_labels(gt_image)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_road = np.all(gt_image == road_color, axis=2)
                gt_road = gt_road.reshape(*gt_road.shape, 1)
                gt_vehicle = np.all(gt_image == vehicle_color, axis=2)
                gt_vehicle = gt_vehicle.reshape(*gt_vehicle.shape, 1)
                gt_image = np.concatenate((gt_bg, gt_road, gt_vehicle), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images)[:,:self.__h,:self.__w,:], np.array(gt_images)[:,:self.__h,:self.__w,:]

