import pickle
from get_rank_idx import *
from sklearn.model_selection import train_test_split
from diff_feature import get_all_feature
from uncertaity_feature import get_uncertainty_feature
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import preprocessing


path_original_out_vec = './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl'
path_onDevice_out_vec = './models/onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl'
path_y = './data/cifar10_y.pkl'

original_out_vec = pickle.load(open(path_original_out_vec, 'rb'))
onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))
y = pickle.load(open(path_y, 'rb'))
y = np.array([i[0] for i in y])

original_pre_y = original_out_vec.argmax(axis=1)
onDevice_pre_y = onDevice_out_vec.argmax(axis=1)

distance_feature = get_all_feature(original_out_vec, onDevice_out_vec)
uncertainty_feature = get_uncertainty_feature(onDevice_out_vec)
concat_all_feature = np.hstack((distance_feature, uncertainty_feature, onDevice_out_vec))  # xgb_apfd 0.7676975790895062
# concat_all_feature = np.hstack((distance_feature, onDevice_out_vec))  # xgb_apfd 0.7673290774498456
# concat_all_feature = onDevice_out_vec # xgb_apfd 0.767958767361111
# concat_all_feature = uncertainty_feature # xgb_apfd 0.7302643952546296
# concat_all_feature = distance_feature # xgb_apfd


target_train_pre, target_test_pre, train_y, test_y = train_test_split(onDevice_pre_y, y, test_size=0.3, random_state=0)
concat_train_all_feature, concat_test_all_feature, _, _ = train_test_split(concat_all_feature, y, test_size=0.3, random_state=0)
onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)
distance_feature_train, distance_feature_test, _, _ = train_test_split(distance_feature, y, test_size=0.3, random_state=0)
uncertainty_feature_train, uncertainty_feature_test, _, _ = train_test_split(uncertainty_feature, y, test_size=0.3, random_state=0)

miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)


model = XGBClassifier()
model.fit(concat_train_all_feature, miss_train_label)
y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
xgb_rank_idx = y_concat_all.argsort()[::-1].copy()
xgb_apfd = apfd(idx_miss_test_list, xgb_rank_idx)

model = XGBClassifier()
model.fit(onDevice_out_vec_train, miss_train_label)
y_concat_all = model.predict_proba(onDevice_out_vec_test)[:, 1]
xgb_rank_idx = y_concat_all.argsort()[::-1].copy()
xgb_apfd_outvec = apfd(idx_miss_test_list, xgb_rank_idx)


deepGini_rank_idx = DeepGini_rank_idx(onDevice_out_vec_test)
vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(onDevice_out_vec_test)
pcs_rank_idx = PCS_rank_idx(onDevice_out_vec_test)
entropy_rank_idx = Entropy_rank_idx(onDevice_out_vec_test)
random_rank_idx = Random_rank_idx(onDevice_out_vec_test)

random_apfd = apfd(idx_miss_test_list, random_rank_idx)
deepGini_apfd = apfd(idx_miss_test_list, deepGini_rank_idx)
vanillasoftmax_apfd = apfd(idx_miss_test_list, vanillasoftmax_rank_idx)
pcs_apfd = apfd(idx_miss_test_list, pcs_rank_idx)
entropy_apfd = apfd(idx_miss_test_list, entropy_rank_idx)

dic = {
    'random_apfd': random_apfd,
    'deepGini_apfd': deepGini_apfd,
    'vanillasoftmax_apfd': vanillasoftmax_apfd,
    'pcs_apfd': pcs_apfd,
    'entropy_apfd': entropy_apfd,
    'xgb_apfd': xgb_apfd,
    'xgb_apfd_outvec': xgb_apfd_outvec,
}

print('random_apfd', random_apfd)
print('deepGini_apfd', deepGini_apfd)
print('vanillasoftmax_apfd', vanillasoftmax_apfd)
print('pcs_apfd', pcs_apfd)
print('entropy_apfd', entropy_apfd)
print('xgb_apfd', xgb_apfd)
print('xgb_apfd_outvec', xgb_apfd_outvec)
