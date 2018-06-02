import sys, skvideo.io, json, base64
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO, StringIO
import sys
import time

def stopwatch(msg, value):
    d = (((value/365)/24)/60)
    days = int(d)
    h = (d-days)*365
    hours = int(h)
    m = (h - hours)*24
    minutes = int(m)
    s = (m - minutes)*60
    seconds = int(s)
    sys.stderr.write('[{}] {};{}:{}:{}\n'.format(msg,days,hours,minutes,seconds))

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

start = time.time()
with open(graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
end = time.time()
stopwatch('load graph', end - start)
start = end
    
video = skvideo.io.vread(file)
end = time.time()
stopwatch('read video', end - start)
start = end

frame = 1
answer_key = {}

with tf.Session() as sess:
    image_input = sess.graph.get_tensor_by_name('image_input:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    logits = sess.graph.get_tensor_by_name('logits:0')
    batch_size = 25
    zero = np.zeros((batch_size, image_shape[0]-h, image_shape[1]), dtype=np.int8)
    for batch_i in range(0, len(video), batch_size):
        rgb_frames = video[batch_i:batch_i+batch_size]
        len_frames = len(rgb_frames)
        start = time.time()
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {
                keep_prob: 1.0,
                image_input: np.array(rgb_frames)[:,:h,:w,:]
            }
        )
        end = time.time()
        stopwatch('session run', end - start)
        start = end
        road_softmax = im_softmax[0][:, 1].reshape(len_frames, h, w)
        road_softmax = np.concatenate((road_softmax,zero[:len_frames,:,:]), axis=1)
        binary_road_result = (road_softmax > 0.5).reshape(len_frames, image_shape[0], image_shape[1]).astype('uint8')
        car_softmax = im_softmax[0][:, 2].reshape(len_frames, h, w)
        car_softmax = np.concatenate((car_softmax,zero[:len_frames,:,:]), axis=1)
        binary_car_result = (car_softmax > 0.5).reshape(len_frames, image_shape[0], image_shape[1]).astype('uint8')
        for i in range(len_frames):
            answer_key[frame] = [encode(binary_car_result[i]), encode(binary_road_result[i])]
            frame += 1
        end = time.time()
        stopwatch('post process', end - start)

print(json.dumps(answer_key))
 
