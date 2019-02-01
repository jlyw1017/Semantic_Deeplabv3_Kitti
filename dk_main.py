import sys, os, keras
sys.path.append('./keras_deeplab_v3_plus')
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard
from model import Deeplabv3, relu6, BilinearUpsampling
from prepare_data import read_data

path_image = os.path.join('/home/jlyw1017/datasets/kitti', "training/image_2", "*_10.png")
path_label = os.path.join('/home/jlyw1017/datasets/kitti', "training/semantic", "*_10.png")

input_tensor, label_tensor = read_data(path_image, path_label)
deeplab_model = Deeplabv3(input_shape=(375, 1242,3), classes=35)
opter = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
deeplab_model.compile(optimizer=opter, loss='categorical_crossentropy')
deeplab_model.fit(input_tensor, label_tensor, validation_split=0.1, shuffle=True, nb_epoch=10, batch_size=64,
                 callbacks=[TensorBoard(log_dir='/home/jlyw1017/Semantic_Deeplabv3_Kitti/mytensorboard', write_images= 1)])

#deeplab_model = load_model('example.h5',custom_objects={'relu6':relu6,'BilinearUpsampling':BilinearUpsampling })

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
