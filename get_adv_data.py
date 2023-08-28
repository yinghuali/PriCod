from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
import os
import argparse
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ap = argparse.ArgumentParser()
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--num_classes", type=int)
ap.add_argument("--epochs", type=int)
ap.add_argument("--batch_size", type=int)
ap.add_argument("--path_save", type=str)
args = ap.parse_args()

path_x = args.path_x
path_y = args.path_y
num_classes = args.num_classes
epochs = args.epochs
batch_size = args.batch_size
path_save = args.path_save

# python get_adv_data.py --path_x './data/cifar10_x.pkl' --path_y './data/cifar10_y.pkl' --num_classes 10 --epochs 80 --batch_size 64 --path_save './advdata/cifar10/'




def cnn():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, name='dense_1'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def get_train_model(model, x_train, y_train, batch_size, nb_epochs, save_model_path):
    classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, validation_data=(x_test, y_test), shuffle=True)
    model.save(save_model_path)
    return classifier


def fsgm_x_adv(classifier, x):
    attack = FastGradientMethod(estimator=classifier, eps=0.05)  # eps 值越大，攻击越大，一般eps=0.1
    x_adv = attack.generate(x=x)
    output = open(path_save+'fsgm_x_adv.pkl', 'wb')
    pickle.dump(x_adv, output)


def patch_x_adv(classifier, x):
    attack = AdversarialPatch(classifier=classifier)
    x_adv = attack.apply_patch(x=x, scale=0.3)
    output = open(path_save+'patch_x_adv.pkl', 'wb')
    pickle.dump(x_adv, output)


def gda_x_adv(classifier, x, y):   # 一般慢
    attack = GeoDA(estimator=classifier)
    x_adv = attack.generate(x=x, y=y)
    output = open(path_save+'gda_x_adv.pkl', 'wb')
    pickle.dump(x_adv, output)


def bim_x_adv(classifier, x): # 一般慢
    attack = BasicIterativeMethod(estimator=classifier)
    x_adv = attack.generate(x=x)
    output = open(path_save+'bim_x_adv.pkl', 'wb')
    pickle.dump(x_adv, output)


def pgd_x_adv(classifier, x): #一般慢
    attack = ProjectedGradientDescent(estimator=classifier)
    x_adv = attack.generate(x=x)
    output = open(path_save+'pgd_x_adv.pkl', 'wb')
    pickle.dump(x_adv, output)


def newf_x_adv(classifier, x): # 一般慢
    attack = NewtonFool(classifier=classifier)
    x_adv = attack.generate(x=x)
    output = open(path_save+'newf_x_adv.pkl', 'wb')
    pickle.dump(x_adv, output)


def smm_x_adv(classifier, x): # 一般慢
    attack = SaliencyMapMethod(classifier=classifier)
    x_adv = attack.generate(x=x)
    output = open(path_save+'smm_x_adv.pkl', 'wb')
    pickle.dump(x_adv, output)


if __name__ == '__main__':
    model = cnn()
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    x = x.astype('float32')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train /= 255.0
    x_test /= 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    classifier = get_train_model(model, x_train, y_train, batch_size, epochs, 'cnn.h5')
    fsgm_x_adv(classifier, x)
    patch_x_adv(classifier, x)
    gda_x_adv(classifier, x, y)
    bim_x_adv(classifier, x)
    pgd_x_adv(classifier, x)
    newf_x_adv(classifier, x)
    smm_x_adv(classifier, x)

