import os
import argparse
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import MaxPooling2D, Input, Activation, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

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

def NIN(x_train, num_classes):
    img = Input(shape=x_train.shape[1:])
    weight_decay = 1e-6
    def NiNBlock(kernel, mlps, strides):
        def inner(x):
            l = Conv2D(mlps[0], kernel, padding='same', strides=strides, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.01))(x)
            l = BatchNormalization()(l)
            l = Activation('relu')(l)
            for size in mlps[1:]:
                l = Conv2D(size, 1, padding='same', strides=[1,1], kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.05))(l)
                l = BatchNormalization()(l)
                l = Activation('relu')(l)
            return l
        return inner
    l1 = NiNBlock(5, [192, 160, 96], [1,1])(img)
    l1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l1)
    l1 = Dropout(0.5)(l1)

    l2 = NiNBlock(5, [192, 192, 192], [1,1])(l1)
    l2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l2)
    l2 = Dropout(0.5)(l2)

    l3 = NiNBlock(3, [192, 192, num_classes], [1,1])(l2)

    l4 = GlobalAveragePooling2D()(l3)
    l4 = Activation('softmax')(l4)

    model = Model(inputs=img, outputs=l4, name='NIN-none')
    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def main():
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    x = x.astype('float32')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train /= 255.0
    x_test /= 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    model = NIN(x_train, num_classes=num_classes)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    model.save(path_save)


if __name__ == '__main__':
    main()