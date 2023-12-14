import tensorflow as tf
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path_model", type=str)
ap.add_argument("--path_save", type=str)
args = ap.parse_args()

path_model = args.path_model
path_save = args.path_save


# def get_tflite(path_model, path_save):
#     model = tf.keras.models.load_model(path_model)
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_types = [tf.uint8]
#     tflite_quant_model = converter.convert()
#     open(path_save, "wb").write(tflite_quant_model)


def get_tflite(path_model, path_save):
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file=path_model)
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open(path_save, "wb").write(tflite_model)


# def get_tflite(path_model, path_save):
#     converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file=path_model, input_shapes={'input_1': (1, 32, 32, 3)})
#     converter.post_training_quantize = True
#     tflite_model = converter.convert()
#     open(path_save, "wb").write(tflite_model)


def main():
    get_tflite(path_model, path_save)


if __name__ == '__main__':
    main()

