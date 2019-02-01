import os, glob, cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc as sp


def pack_into_tensor(file):
    img_array = np.array([sp.imresize(sp.imread(i), (375, 1242)) for i in file])
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, 3)
        print("new dimension: ", img_array.ndim)
    return img_array


def read_data(path_raw, path_label):
    '''
    Read files from kitti folder and return tensors as inputs
    '''
    # get files
    data_input = glob.glob(path_raw)
    data_input.sort()
    label_input = glob.glob(path_label)
    label_input.sort()

    # put file into tensor
    image_input_tensor = pack_into_tensor(data_input)
    print(image_input_tensor.shape)
    label_input_tensor = pack_into_tensor(label_input)
    print(label_input_tensor.shape)
    return image_input_tensor, label_input_tensor


# test code
if __name__ == '__main__':
    img_show = sp.imread("/home/jlyw1017/datasets/kitti/training/image_2/000000_10.png")
    img_show2 = sp.imread("/home/jlyw1017/datasets/kitti/training/semantic/000000_10.png")
    img_show3 = sp.imread("/home/jlyw1017/datasets/kitti/training/semantic_rgb/000000_10.png")
    print(type(img_show))
    print(img_show.shape)


