import os
import shutil

if not os.path.exists('../data/Test'):
    os.makedirs('../data/Test/CameraRGB')
    os.makedirs('../data/Test/CameraSeg')
    
img_files = [str(filename) + '.png' for filename in range(950,1000)]
for filename in img_files:
    rgb = '../data/Train/CameraRGB/' + filename
    seg = '../data/Train/CameraSeg/' + filename
    if os.path.isfile(rgb):
        shutil.move(rgb, '../data/Test/CameraRGB/')
    if os.path.isfile(seg):
        shutil.move(seg, '../data/Test/CameraSeg/')

