import pickle
import json
import datetime
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


original_out_vec = pickle.load(open(path_original_out_vec, 'rb'))
onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))
embedding_vec = pickle.load(open(path_embedding_vec, 'rb'))


def main():
    y = pickle.load(open(path_y, 'rb'))
    if y.shape == (y.size, ):
        y = y
    else:
        y = np.array([i[0] for i in y])

    # uncertainty_feature
    uncertainty_feature = get_uncertainty_feature(onDevice_out_vec)

    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)
    distance_feature = get_all_feature(original_out_vec, onDevice_out_vec)
    concat_all_feature = np.hstack((distance_feature, uncertainty_feature, embedding_vec))

    target_train_pre, target_test_pre, train_y, test_y = train_test_split(onDevice_pre_y, y, test_size=0.3, random_state=0)
    concat_train_all_feature, concat_test_all_feature, _, _ = train_test_split(concat_all_feature, y, test_size=0.3, random_state=0)
    onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    model = LGBMClassifier(n_estimators=300)
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    model_apfd = apfd(idx_miss_test_list, rank_idx)
    print(model_apfd)


if __name__ == '__main__':
    main()
