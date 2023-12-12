import numpy as np
import os
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_model = './original_models/imagenet_inceptionv3.h5'
data_path = '/raid/yinghua/PriCod/data/imagenet'
path_original_pre_save = '/raid/yinghua/PriCod/data/pkl_data/InceptionV3_pre.pkl'
path_label_save = '/raid/yinghua/PriCod/data/pkl_data/label.pkl'


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.JPEG'):
                    path_list.append(file_absolute_path)
    return path_list


def main():

    path_list = get_path(data_path)
    path_list = sorted(path_list)

    model = InceptionV3(weights='imagenet')
    # model.save(path_save)

    pre_list = []
    label_list = []
    i = 0
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds_vec = model.predict(x)
        pre = list(preds_vec[0])
        pre_list.append(pre)

        label = img_path.split('/')[-1].split('_')[0]
        label_list.append(label)

        print(i)
        i+=1

    pre_np = np.array(pre_list)
    label_np = np.array(label_list)

    pickle.dump(pre_np, open(path_original_pre_save, 'wb'), protocol=4)
    pickle.dump(label_np, open(path_label_save, 'wb'), protocol=4)


if __name__ == '__main__':
    main()


# pre = decode_predictions(preds_vec, top=1) # [[('n09421951', 'sandbar', 0.9377819)]]
# y = pre[0][0][0]
# probability = pre[0][0][2]
# print(y)
# print(probability)
