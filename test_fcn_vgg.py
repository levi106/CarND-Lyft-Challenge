import os.path
import scipy.misc
import numpy as np
import tensorflow as tf

graph_file = './fcn/saved_model.pb'
image_shape = (600,800)
output_dir = './runs'

def test(image_files):
    h = 32 * (image_shape[0] // 32)
    w = 32 * (image_shape[1] // 32)
    zero = np.zeros((image_shape[0]-h, image_shape[1], 1), dtype=np.int8)

    with open(graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        image_input = sess.graph.get_tensor_by_name('image_input:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        logits = sess.graph.get_tensor_by_name('logits:0')

        for image_file in image_files:
            print('{}'.format(image_file))
            image = scipy.misc.imread(image_file)
            z = sess.run(
                [tf.nn.softmax(logits)],
                {
                    keep_prob: 1.0,
                    image_input: [np.array(image)[:h,:w,:]]
                }
            )

            road_softmax = z[0][:,1].reshape(h,w)
            road_segmentation = (road_softmax > 0.5).reshape(h,w,1)
            road_mask = np.dot(road_segmentation, np.array([[7,0,0]]))

            car_softmax = z[0][:,2].reshape(h,w)
            car_segmentation = (car_softmax > 0.5).reshape(h,w,1)
            car_mask = np.dot(car_segmentation, np.array([[10,0,0]]))

            mask = np.maximum(road_mask, car_mask)
            mask = np.concatenate((mask,zero), axis=0)
            mask = scipy.misc.toimage(mask, mode="RGB")

            name = os.path.basename(image_file)
            scipy.misc.imsave(os.path.join(output_dir, name), mask)

def main():
    image_files = [
        './data/Train/CameraRGB/53.png',
        './data/Train/CameraRGB/102.png',
        '/data/Train/CameraRGB/382.png',
        './data/Train/CameraRGB/720.png',
        './data/Train/CameraRGB/999.png'
    ]
    test(image_files)

if __name__ == '__main__':
    main()

