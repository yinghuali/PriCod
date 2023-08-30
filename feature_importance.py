import pickle
import json
import argparse
from diff_feature import get_all_feature
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
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


def main():
    original_out_vec = pickle.load(open(path_original_out_vec, 'rb'))
    onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))
    embedding_vec = pickle.load(open(path_embedding_vec, 'rb'))

    y = pickle.load(open(path_y, 'rb'))
    y = np.array([i[0] for i in y])
    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)

    distance_feature = get_all_feature(original_out_vec, onDevice_out_vec)
    # pca = PCA(n_components=100)
    pca = PCA(n_components=300) # vgg16
    # pca = PCA(n_components=700)
    new_embedding_vec = pca.fit_transform(embedding_vec)

    concat_all_feature = np.hstack((distance_feature, new_embedding_vec))

    # invalid_rows = np.isnan(concat_all_feature).any(axis=1) | np.isinf(concat_all_feature).any(axis=1)
    # concat_all_feature = concat_all_feature[~invalid_rows]
    # y= y[~invalid_rows]
    # onDevice_pre_y = onDevice_pre_y[~invalid_rows]
    # onDevice_out_vec = onDevice_out_vec[~invalid_rows]


    target_train_pre, target_test_pre, train_y, test_y = train_test_split(onDevice_pre_y, y, test_size=0.3, random_state=0)
    concat_train_all_feature, concat_test_all_feature, _, _ = train_test_split(concat_all_feature, y, test_size=0.3, random_state=0)
    onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    model = XGBClassifier()
    model.fit(concat_train_all_feature, miss_train_label)
    importance = model.get_booster().get_score(importance_type='cover')
    dic = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    json.dump(dic, open(path_save_res, 'w'), indent=4)



if __name__ == '__main__':
    main()
