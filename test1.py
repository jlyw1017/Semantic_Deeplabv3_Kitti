from keras import backend as kb
import numpy as np
from tensorflow import Print
import tensorflow as tf
from sklearn import preprocessing
x = np.array([[1,2], [3, -1]])
print(x.shape)
y = np.eye(4)[x]
print(y)
print(y.shape)
