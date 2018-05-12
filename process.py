import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from fcn import FCN

file = sys.argv[-1]

def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

fcn = FCN(3)
fcn.load('./frozen_fcn_model.pb')
video = skvideo.io.vread(file)

answer_key = {}

frame = 1

for rgb_frame in video:
    im_softmax = fcn.process(rgb_frame)
    road_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    binary_road_result = (road_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1).astype('uint8')
    car_softmax = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
    binary_car_result = (car_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1).astype('uint8')
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    frame += 1

print(json.dumps(answer_key))
 
