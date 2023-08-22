import tensorflow as tf
import coremltools
import argparse
from coremltools.models.neural_network import quantization_utils

ap = argparse.ArgumentParser()
ap.add_argument("--path_model", type=str)
ap.add_argument("--path_save", type=str)
args = ap.parse_args()

path_model = args.path_model
path_save = args.path_save


def get_coreml(path_model, save_path):
    model = tf.keras.models.load_model(path_model)
    coreml_model = coremltools.convert(model)
    coreml_quan = quantization_utils.quantize_weights(coreml_model, nbits=8)
    coreml_quan.save(save_path)


def main():
    get_coreml(path_model, path_save)


if __name__ == '__main__':
    main()


