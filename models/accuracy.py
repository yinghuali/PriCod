import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path_pre_vec = './onDevice_out_vec/fashionMnist_lenet5_3_tflite_vec.pkl'
path_y = '../data/fashionMnist_y.pkl'
pre_vec = pickle.load(open(path_pre_vec, 'rb'))
y = pickle.load(open(path_y, 'rb'))
pre_vec_train, pre_vec_test, y_train, y_test = train_test_split(pre_vec, y, test_size=0.3, random_state=0)

y_train = np.array([i[0] for i in y_train])
y_test = np.array([i[0] for i in y_test])

pre_y_train = pre_vec_train.argmax(axis=1)
pre_y_test = pre_vec_test.argmax(axis=1)

print(y_train.shape)
print(pre_y_train.shape)
acc_train = str(accuracy_score(y_train, pre_y_train))[:5]
acc_test = str(accuracy_score(y_test, pre_y_test))[:5]
print('train:', acc_train, 'test:', acc_test)



