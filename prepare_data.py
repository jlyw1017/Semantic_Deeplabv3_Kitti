import os, glob, cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc as sp
import progressbar

def pack_into_tensor(file_input, h, w, type = 'lanczos'):
    img_array_tensor = np.array([sp.imresize(sp.imread(i), (h, w), interp=type) for i in file_input])
    if img_array_tensor.ndim == 3:
        img_array_tensor = np.eye(35)[img_array_tensor]
    print("Dimension: ", img_array_tensor.shape)
    return img_array_tensor


def read_data(path_raw, path_label, h, w):
    '''
    Read files from kitti folder and return tensors as inputs
    '''
    # get files
    data_input = glob.glob(path_raw)
    print("Glob image finished")
    data_input.sort()
    print("Sort image finished")
    label_input = glob.glob(path_label)
    print("Glob label finished")
    label_input.sort()
    print("Sort label finished")

    # put file into tensor
    image_input_tensor = pack_into_tensor(data_input, h, w)
    label_input_tensor = pack_into_tensor(label_input, h, w, 'nearest')

    print(image_input_tensor.shape, label_input)


    return image_input_tensor, label_input_tensor


# test code
if __name__ == '__main__':
    img_show = sp.imresize(sp.imread("/home/jlyw1017/datasets/kitti/training/image_2/000000_10.png"), (250, 828))
    img_show2 = sp.imresize(sp.imread("/home/jlyw1017/datasets/kitti/training/semantic/000000_10.png"), (250, 828), interp='nearest')
    img_show3 = sp.imresize(sp.imread("/home/jlyw1017/datasets/kitti/training/semantic_rgb/000000_10.png"), (250, 828), interp='nearest')
    plt.imshow(img_show)
#    print(img_show)
    plt.show()
    plt.imshow(img_show2)
    print(img_show2)
    plt.show()
    plt.imshow(img_show3)
    plt.show()

#    print(type(img_show))
#    print(img_show.shape)


