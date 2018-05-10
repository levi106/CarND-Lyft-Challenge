import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def filter_labels(gt_image):
        gt_image[gt_image == 6] = 7
        gt_image[np.isin(gt_image, [7,10], invert=True)] = 0
        gt_image[490:,:,:][gt_image[490:,:,:] != 7] = 0
        return gt_image

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(data_folder, 'CameraSeg', '*.png'))}
        background_color = np.array([0, 0, 0])
        road_color = np.array([7, 0, 0])
        vehicle_color = np.array([10, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                #image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                image = scipy.misc.imread(image_file)[:image_shape[0],:image_shape[1],:]
                #gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                gt_image = scipy.misc.imread(gt_image_file)[:image_shape[0],:image_shape[1],:]
                gt_image = filter_labels(gt_image)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_road = np.all(gt_image == road_color, axis=2)
                gt_road = gt_road.reshape(*gt_road.shape, 1)
                gt_vehicle = np.all(gt_image == vehicle_color, axis=2)
                gt_vehicle = gt_vehicle.reshape(*gt_vehicle.shape, 1)
                gt_image = np.concatenate((gt_bg, gt_road, gt_vehicle), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'CameraRGB', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        road_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        road_segmentation = (road_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        road_mask = np.dot(road_segmentation, np.array([[7, 0, 0]]))
        car_softmax = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
        car_segmentation = (car_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        car_mask = np.dot(car_segmentation, np.array([[10, 0, 0]]))
        mask = np.maximum(road_mask, car_mask)
        mask = scipy.misc.toimage(mask, mode="RGB")
        #street_im = scipy.misc.toimage(image)
        #street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(mask)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'Test'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
