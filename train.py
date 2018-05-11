import os.path
import tensorflow as tf
import helper
import warnings
import project_tests as tests
from distutils.version import LooseVersion
from fcn import FCN

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer. YOu are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def run():
    num_classes = 3
    #image_shape = (600, 800)
    image_shape = (576, 800)
    data_dir = '../data'
    runs_dir = './runs'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'Train'), image_shape)
    
    fcn = FCN(num_classes)
    fcn.train(get_batches_fn, model_path='./fcn_model.ckpt')

    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    run()

