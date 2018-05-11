import os.path
import tensorflow as tf
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


class FCN():
    def __init__(self, num_classes):
        self.__num_classes = num_classes
        self.__model = {}


    def train(self, get_batches_fn, epochs=10, batch_size=10, learning_rate=0.001, keep_prob=0.85, model_path=None):
        params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'keep_prob': keep_prob
        }

        with tf.Session() as sess:
            vgg_path = self.__maybe_download_pretrained_vgg()
            self.__model = self.__placeholder(self.__model, self.__num_classes)
            self.__model = self.__load_vgg(self.__model, sess, vgg_path)
            self.__model = self.__layers(self.__model, self.__num_classes)
            self.__model = self.__optimize(self.__model, self.__num_classes)

            self.__train_nn(self.__model, sess, get_batches_fn, params)

            if model_path != None:
                saver = tf.train.Saver()
                saver.save(sess, model_path)


    def __maybe_download_pretrained_vgg(self, data_dir='./'):
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

        return vgg_path


    def __placeholder(self, model, num_classes):
        """
        Create Placeholders.
        """
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        model['current_label'] = current_label
        model['learning_rate'] = learning_rate

        return model

    def __load_vgg(self, model, sess, vgg_path):
        """
        Load Pretrained VGG Model into TensorFlow.
        """
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
        image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

        model['image_input'] = image_input
        model['keep_prob'] = keep_prob
        model['layer3_out'] = layer3_out
        model['layer4_out'] = layer4_out
        model['layer7_out'] = layer7_out

        return model


    def __layers(self, model, num_classes)
        """
        Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
        """
        layer7 = tf.layers.conv2d(model['vgg_layer7_out'], num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        layer4 = tf.layers.conv2d(model['vgg_layer4_out'], num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        layer3 = tf.layers.conv2d(model['vgg_layer3_out'], num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        
        output = tf.layers.conv2d_transpose(layer7, num_classes, 4, 2, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        output = tf.add(output, layer4)

        output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        output = tf.add(output, layer3)

        output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        model['layers_output'] = output

        return model


    def __optimize(self, model, num_classes):
        """
        Build the TensorFLow loss and optimizer operations.
        """
        logits = tf.reshape(model['layers_output'], (-1, num_classes))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=model['correct_label']))
        cross_entropy_loss += sum(reg_losses)
        train_op = tf.train.AdamOptimizer(model['learning_rate']).minimize(cross_entropy_loss)

        model['logits'] = logits
        model['train_op'] = train_op
        model['cross_entropy_loss'] = cross_entropy_loss

        return model


    def __train_nn(self, model, sess, get_batches_fn, params):
        """
        Train neural network and print out the loss during training.
        """
        i = 0
        epochs = params['epochs']
        batch_size = params['batch_size']
        keep_prob = params['keep_prob']
        lr = param['learning_rate']

        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch_xs, batch_ys in get_batches_fn(batch_size):
                feed_dict = {
                    model['input_image']: batch_xs,
                    model['correct_label']: batch_ys,
                    model['keep_prob']: keep_prob,
                    model['learning_rate']: lr
                }
                sess.run(model['train_op'], feed_dict=feed_dict)

                if i % 10 == 0:
                    loss = sess.run(model['cross_entropy_loss'], feed_dict=feed_dict)
                    print("Epoch:{} Iteration:{} Loss:{}".format(epoch, i, loss))
                i += 1

