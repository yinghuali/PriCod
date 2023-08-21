import tensorflow as tf
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path_model", type=str)
ap.add_argument("--path_save_tflite", type=str)
args = ap.parse_args()

path_model = args.path_model
path_save_tflite = args.path_save_tflite


def get_tflite(path_model, path_save):
    model = tf.keras.models.load_model(path_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.uint8]
    tflite_quant_model = converter.convert()
    open(path_save, "wb").write(tflite_quant_model)


def main():
    get_tflite(path_model, path_save_tflite)


if __name__ == '__main__':
    main()
