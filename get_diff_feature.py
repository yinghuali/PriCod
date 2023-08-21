import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_x = 'data/cifar10_x.pkl'
path_y = 'data/cifar10_y.pkl'
num_classes = 10

x = pickle.load(open(path_x, 'rb'))
y = pickle.load(open(path_y, 'rb'))
x = x.astype('float32')
x /= 255.0

original_model = load_model('./models/orginal_models/cifa10_vgg_20.h5')
ori_probabilities = original_model.predict(x)

print(ori_probabilities.shape)
print('=======')
print(ori_probabilities[0])

# (60000, 10)
# =======
# [2.2958200e-07 1.9639522e-06 1.7921941e-04 5.3656002e-04 2.7390945e-04
#  3.7106907e-04 9.9861610e-01 1.1841866e-05 8.3397390e-07 8.3228824e-06]




