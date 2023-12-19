import pickle
import json
import argparse
from diff_feature import get_all_feature
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from lightgbm import LGBMClassifier
ap = argparse.ArgumentParser()

ap.add_argument("--path_original_out_vec", type=str)
ap.add_argument("--path_onDevice_out_vec", type=str)
ap.add_argument("--path_embedding_vec", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--path_save_res", type=str)
args = ap.parse_args()


path_original_out_vec = args.path_original_out_vec
path_onDevice_out_vec = args.path_onDevice_out_vec
path_embedding_vec = args.path_embedding_vec
path_y = args.path_y
path_save_res = args.path_save_res

# python main.py --path_original_out_vec './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_vgg_20_tflite.json'


original_out_vec = pickle.load(open(path_original_out_vec, 'rb'))
onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))
embedding_vec = pickle.load(open(path_embedding_vec, 'rb'))


def main():
    y = pickle.load(open(path_y, 'rb'))
    if y.shape == (y.size, ):
        y = y
    else:
        y = np.array([i[0] for i in y])


    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)

    distance_feature = get_all_feature(original_out_vec, onDevice_out_vec)

    target_train_pre, target_test_pre, train_y, test_y = train_test_split(onDevice_pre_y, y, test_size=0.3, random_state=0)
    onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    distance_feature_train, distance_feature_test, _, _ = train_test_split(distance_feature, y, test_size=0.3, random_state=0)
    embedding_vec_feature_train, embedding_vec_feature_test, _, _ = train_test_split(embedding_vec, y, test_size=0.3, random_state=0)

    dic = {}
    model = LGBMClassifier()
    model.fit(distance_feature_train, miss_train_label)
    y_concat_all = model.predict_proba(distance_feature_test)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    distance_apfd = apfd(idx_miss_test_list, rank_idx)
    dic['distance_apfd'] = distance_apfd

    model = LGBMClassifier()
    model.fit(embedding_vec_feature_train, miss_train_label)
    y_concat_all = model.predict_proba(embedding_vec_feature_test)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    embedding_apfd = apfd(idx_miss_test_list, rank_idx)
    dic['embedding_apfd'] = embedding_apfd


    json.dump(dic, open(path_save_res, 'w'), sort_keys=False, indent=4)



if __name__ == '__main__':
    main()
