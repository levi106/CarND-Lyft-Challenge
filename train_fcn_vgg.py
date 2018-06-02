import tensorflow as tf
from utils import maybe_download_pretrained_vgg
from dataset import Dataset

num_classes = 3
lr = 0.001
kp = 0.85
batch_size = 10
epochs = 20
#epochs = 1
base_dir = './fcn'
graph_file = 'saved_model.pb'
image_shape = (600, 800)
data_dir = './data/Train'

def train(dataset):
    vgg_path = maybe_download_pretrained_vgg()

    with tf.Session() as sess:

        correct_label = tf.placeholder(tf.float32, [None,None,None,num_classes])
        learning_rate = tf.placeholder(tf.float32)

        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        image_input = sess.graph.get_tensor_by_name('image_input:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        layer3_out = sess.graph.get_tensor_by_name('layer3_out:0')
        layer4_out = sess.graph.get_tensor_by_name('layer4_out:0')
        layer7_out = sess.graph.get_tensor_by_name('layer7_out:0')

        layer7 = tf.layers.conv2d(layer7_out, num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        layer4 = tf.layers.conv2d(layer4_out, num_classes, 1, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        layer3 = tf.layers.conv2d(layer3_out, num_classes, 1, padding='same',
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

        logits = tf.reshape(output, (-1, num_classes), name='logits')
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
        cross_entropy_loss += sum(reg_loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

        sess.run(tf.global_variables_initializer())

        i = 0
        for epoch in range(epochs):
            for batch_xs, batch_ys in dataset.get_batches(batch_size):
                feed_dict = {
                    image_input: batch_xs,
                    correct_label: batch_ys,
                    keep_prob: kp,
                    learning_rate: lr
                }
                sess.run(train_op, feed_dict=feed_dict)

                if i % 10 == 0:
                    loss = sess.run(cross_entropy_loss, feed_dict=feed_dict)
                    print("Epoch:{} Iteration:{} Loss:{}".format(epoch, i, loss))
                i += 1

        g = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['logits'])
        tf.train.write_graph(g, base_dir, graph_file, as_text=False)

def main():
    dataset = Dataset(data_dir, image_shape)
    train(dataset)

if __name__ == '__main__':
    main()

