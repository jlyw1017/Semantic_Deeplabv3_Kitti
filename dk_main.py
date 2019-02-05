import sys, os, keras
sys.path.append('./keras_deeplab_v3_plus')
sys.path.append('./Keras-segmentation-deeplab-v3.1')

import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import scipy.misc as sp
from keras.callbacks import TensorBoard
from keras import backend as Kb

# from model import Deeplabv3, relu6, BilinearUpsampling
from deeplabv3p import Deeplabv3
from prepare_data import read_data
from keras.utils import plot_model

path_image = os.path.join('/home/jlyw1017/datasets/kitti', "training/image_2", "000*_10.png")
path_label = os.path.join('/home/jlyw1017/datasets/kitti', "training/semantic", "000*_10.png")

path_image_cityscape = os.path.join('/home/jlyw1017/datasets/cityscapes/leftImg8bit', "train/*", "*_leftImg8bit.png")
path_label_cityscape_coarse = os.path.join('/home/jlyw1017/datasets/cityscapes/gtCoarse', "train/*", "*_gtCoarse_labelIds.png")
path_label_cityscape_fine = os.path.join('/home/jlyw1017/datasets/cityscapes/gtFine', "train/*", "*_gtFine_labelIds.png")

path_test = os.path.join('/home/jlyw1017/datasets/kitti', "testing/image", "*_10.png")
path_test_label = os.path.join('/home/jlyw1017/datasets/kitti', "testing/label", "*_10.png")

inference_path = os.path.join('/home/jlyw1017/datasets/kitti', "inference/image_2", "*_10.png")

#input_tensor, label_tensor = read_data(path_image, path_label, 250, 828)
test_tensor, test_tensor_label = read_data(path_test, path_test_label, 250, 828)

inference_tensor, test_tensor_label = read_data(inference_path, path_test_label, 250, 828)


deeplab_model = Deeplabv3(weights=None , input_tensor=None, infer = True, input_shape=(250, 828, 3),
                          classes=35, backbone='mobilenetv2', OS=8, alpha=1.)
#deeplab_model = Deeplabv3(input_shape=(375, 1242, 3), classes=35, backbone='mobilenetv2', )
opter = keras.optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)
deeplab_model.compile(optimizer=opter, loss='sparse_categorical_crossentropy')
#plot_model(deeplab_model, show_shapes=True, to_file='model.png')
deeplab_model.load_weights('/home/jlyw1017/Semantic_Deeplabv3_Kitti/8_gtCoarse_300x600_gtFine_300x600_kitti_250x828.h5')

#deeplab_model.fit(input_tensor, label_tensor, validation_split=0.05, shuffle=True, nb_epoch=20, batch_size=4,
#                 callbacks=[TensorBoard(log_dir='/home/jlyw1017/Semantic_Deeplabv3_Kitti/mytensorboard', write_images= 1)])
#deeplab_model.save_weights('/home/jlyw1017/Semantic_Deeplabv3_Kitti/8_gtCoarse_300x600_gtFine_300x600_kitti_250x828.h5')
loss_and_metrics = deeplab_model.evaluate(test_tensor, test_tensor_label, batch_size=1)
print(loss_and_metrics)

count = 0
for image in test_tensor:
    count += 1
    test_input = np.array([image])
    res = deeplab_model.predict(test_input)
    labels = np.argmax(res.squeeze(), -1)
#    plt.figure(res)
#    plt.savefig(res)
    plt.imshow(labels)
    plt.show()

'''
print('Predict finshed')
print(res.shape)
la = res.squeeze()
print(la.shape)
print(la[0].shape)
labels = np.argmax(la, -1)
print(labels.shape)
plt.imshow(labels)
plt.show()
'''
