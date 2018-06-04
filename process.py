import sys, skvideo.io, json, base64
import numpy as np
import tensorflow as tf
from io import BytesIO, StringIO
import cv2

def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode('utf-8')

file = sys.argv[-1]
graph_file = '../fcn/saved_model.pb'
image_shape = (600,800)
h = 32 * (image_shape[0] // 32)
w = 32 * (image_shape[1] // 32)
frame = 1
batch_size = 20
answer_key = {}
zero = np.zeros((batch_size, 3, image_shape[0]-h, image_shape[1]), dtype=np.int8)

with tf.gfile.GFile(graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
import sys
sys.stderr.write("{}\n".format(file))
video = skvideo.io.vread(file)

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")
    
    for batch_i in range(0, len(video), batch_size):
        rgb_frames = video[batch_i:batch_i+batch_size]
        len_frames = len(rgb_frames)
        rgb_frames = np.array(rgb_frames)[:,:h,:w,:]
        
        with tf.Session(graph=graph) as sess:
            softmax = sess.graph.get_tensor_by_name('softmax:0')
            im_softmax = sess.run(
                [softmax],
                {
                    'keep_prob:0': 1.0,
                    'image_input:0': rgb_frames
                }
            )[0]
        
        im_softmax = im_softmax.reshape(len_frames,h,w,3).transpose((0,3,1,2))
        im_softmax = np.concatenate((im_softmax,zero[:len_frames,:,:,:]),axis=2)
        
        for i in range(len_frames):
            result = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
            result[im_softmax[i][1] > 0.6] = 1
            result[im_softmax[i][2] > 0.15] = 2
            binary_road_result = (result == 1).astype('uint8')
            binary_car_result = (result == 2).astype('uint8')
            answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
            frame += 1

print(json.dumps(answer_key))
