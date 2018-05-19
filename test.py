import os.path
import scipy.misc
import numpy as np
from fcn import FCN

def process_file(fcn, image_file, image_shape):
    image = scipy.misc.imread(image_file)
    im_softmax = fcn.process(image)

    road_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    road_segmentation = (road_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    road_mask = np.dot(road_segmentation, np.array([[7, 0, 0]]))
    car_softmax = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
    car_segmentation = (car_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    car_mask = np.dot(car_segmentation, np.array([[10, 0, 0]]))
    mask = np.maximum(road_mask, car_mask)
    mask = scipy.misc.toimage(mask, mode="RGB")

    return os.path.basename(image_file), np.array(mask)

def main():
    num_classes = 3
    image_files = [
        './data/Train/CameraRGB/53.png',
        './data/Train/CameraRGB/102.png',
        './data/Train/CameraRGB/382.png',
        './data/Train/CameraRGB/720.png',
        './data/Train/CameraRGB/999.png'
    ]
    output_dir = './runs'
    fcn = FCN(num_classes)
    fcn.load('./frozen_fcn_model.pb')
    for image_file in image_files:
        print('{}'.format(image_file))
        name, image = process_file(fcn, image_file, (600, 800))
        scipy.misc.imsave(os.path.join(output_dir, name), image)
    

if __name__ == "__main__":
    main()

