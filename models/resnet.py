import os
import argparse
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model
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


def ResNet50(x_train, num_classes):
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=x_train.shape[1:],
        pooling=None,
        classes=num_classes
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

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

    model = ResNet50(x_train, num_classes=num_classes)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    model.save(path_save)


if __name__ == '__main__':
    main()
