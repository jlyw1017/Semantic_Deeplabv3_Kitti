import sys
sys.path.append('./keras-deeplab-v3-plus')
import numpy as np
from matplotlib import pyplot as plt
import cv2
from model import Deeplabv3, relu6, BilinearUpsampling


deeplab_model = Deeplabv3(input_shape=(375,1242,3), classes=10)
deeplab_model = load_model('example.h5',custom_objects={'relu6':relu6,'BilinearUpsampling':BilinearUpsampling })
#deeplab_model.fit()
img = plt.imread("/home/yawei/datasets/Kitti Semantic Instance Segmentation Evaluation/training/image_2/000000_10.png")
res = deeplab_model.predict(np.expand_dims(img,0))
print('Predict finshed')
print(res.shape)
la = res.squeeze()
print(la.shape)
print(la[0].shape)
labels = np.argmax(la, -1)
print(labels.shape)
plt.imshow(labels)
plt.show()

