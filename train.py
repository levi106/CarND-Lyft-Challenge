import tensorflow as tf
import warnings
from distutils.version import LooseVersion
from fcn import FCN
from dataset import Dataset

def main():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer. YOu are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    num_classes = 3
    image_shape = (600, 800)
    max_epochs = 20
    data_dir = './data/Train'

    dataset = Dataset(data_dir, image_shape)
    fcn = FCN(num_classes)
    fcn.train(dataset, epochs=max_epochs)
    fcn.save('./frozen_fcn_model.pb')

if __name__ == '__main__':
    main()

