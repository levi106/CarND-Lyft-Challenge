import sys, skvideo.io, json, base64
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO, StringIO
from fcn import FCN

graph_file = './fcn/saved_model.pb'
image_shape = (600,800)
h = 32 * (image_shape[0] // 32)
w = 32 * (image_shape[1] // 32)
file = sys.argv[-1]

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

zero = np.zeros((image_shape[0]-h, image_shape[1]), dtype=np.int8)

with open(graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

video = skvideo.io.vread(file)


frame = 1
answer_key = {}

for rgb_frame in video:
    with tf.Session() as sess:
    
        image_input = sess.graph.get_tensor_by_name('image_input:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        logits = sess.graph.get_tensor_by_name('logits:0')

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {
                keep_prob: 1.0,
                image_input: [np.array(rgb_frame)[:h,:w,:]]
            }
        )
        road_softmax = im_softmax[0][:, 1].reshape(h, w)
        road_softmax = np.concatenate((road_softmax,zero), axis=0)
        binary_road_result = (road_softmax > 0.5).reshape(image_shape[0], image_shape[1]).astype('uint8')
        car_softmax = im_softmax[0][:, 2].reshape(h, w)
        car_softmax = np.concatenate((car_softmax,zero), axis=0)
        binary_car_result = (car_softmax > 0.5).reshape(image_shape[0], image_shape[1]).astype('uint8')
        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        frame += 1

print(json.dumps(answer_key))
 
