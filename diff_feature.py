import numpy as np
from scipy.stats import spearmanr
from scipy.stats import wasserstein_distance


def get_kill_feature(original_out_vec, onDevice_out_vec):
    """
    Prediction diferenc
    """
    kill_feaure = []
    original_pre_y = original_out_vec.argmax(axis=1)
    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)
    for i in range(len(original_pre_y)):
        if original_pre_y[i] != onDevice_pre_y[i]:
            kill_feaure.append(1)
        else:
            kill_feaure.append(0)
    kill_feaure = np.array(kill_feaure)
    return kill_feaure


def get_confidence_diff_feature(original_out_vec, onDevice_out_vec):
    """
    Confidence difference
    """
    confidence_feaure = []
    original_pre_y = original_out_vec.argmax(axis=1)
    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)
    for i in range(len(original_pre_y)):
        confidence_diff = abs(original_out_vec[i][original_pre_y[i]] - onDevice_out_vec[i][onDevice_pre_y[i]])
        confidence_feaure.append(confidence_diff)
    confidence_feaure = np.array(confidence_feaure)
    return confidence_feaure


def euclidean_distance(original_out_vec, onDevice_out_vec):
    """
    Euclidean Distance
    """
    distance_feaure = []
    for i in range(len(original_out_vec)):
        distance = np.linalg.norm(original_out_vec[i] - onDevice_out_vec[i])
        distance_feaure.append(distance)
    distance_feaure = np.array(distance_feaure)
    return distance_feaure


def cosine_similarity(original_out_vec, onDevice_out_vec):
    """
    Cosine Similarity
    """
    cosine_similarity_feaure = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cosine_similarity_feaure.append(distance)
    cosine_similarity_feaure = np.array(cosine_similarity_feaure)
    return cosine_similarity_feaure


def manhattan_distance(original_out_vec, onDevice_out_vec):
    """
    Manhattan Distance
    """
    manhattan_distance_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.sum(np.abs(np.array(vector1) - np.array(vector2)))
        manhattan_distance_feature.append(distance)
    manhattan_distance_feature = np.array(manhattan_distance_feature)
    return manhattan_distance_feature


def chebyshev_distance(original_out_vec, onDevice_out_vec):
    """
    Chebyshev Distanc
    """
    chebyshev_distance_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.max(np.abs(np.array(vector1) - np.array(vector2)))
        chebyshev_distance_feature.append(distance)
    chebyshev_distance_feature = np.array(chebyshev_distance_feature)
    return chebyshev_distance_feature


def pearson_correlation_coefficient(original_out_vec, onDevice_out_vec):
    """
    Pearson Correlation Coefficient
    """
    pearson_correlation_coefficient_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.corrcoef(vector1, vector2)[0, 1]
        pearson_correlation_coefficient_feature.append(distance)
    pearson_correlation_coefficient_feature = np.array(pearson_correlation_coefficient_feature)
    return pearson_correlation_coefficient_feature


# 时间较长
def spearman_rank_correlation_coefficient(original_out_vec, onDevice_out_vec):
    """
    Spearman's Rank Correlation Coefficient
    """
    spearman_rank_correlation_coefficient_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance, _ = spearmanr(vector1, vector2)
        spearman_rank_correlation_coefficient_feature.append(distance)
    spearman_rank_correlation_coefficient_feature = np.array(spearman_rank_correlation_coefficient_feature)
    return spearman_rank_correlation_coefficient_feature


def sum_squared_differences(original_out_vec, onDevice_out_vec):
    """
    Sum of Squared Differences
    """
    sum_squared_differences_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.sum((vector1 - vector2) ** 2)
        sum_squared_differences_feature.append(distance)
    sum_squared_differences_feature = np.array(sum_squared_differences_feature)
    return sum_squared_differences_feature


def kullback_leibler_divergence(original_out_vec, onDevice_out_vec):
    """
    Kullback-Leibler Divergence
    """
    kullback_leibler_divergence_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.sum(vector1 * np.log(vector1 / vector2))
        kullback_leibler_divergence_feature.append(distance)
    kullback_leibler_divergence_feature = np.array(kullback_leibler_divergence_feature)
    return kullback_leibler_divergence_feature


def bhattacharyya_distance(original_out_vec, onDevice_out_vec):
    """
    Bhattacharyya
    """
    Bhattacharyya_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance =-np.log(np.sum(np.sqrt(vector1 * vector2)))
        Bhattacharyya_feature.append(distance)
    Bhattacharyya_feature = np.array(Bhattacharyya_feature)
    return Bhattacharyya_feature


def hellinger_distance(original_out_vec, onDevice_out_vec):
    """
    Hellinger
    """
    Hellinger_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.linalg.norm(np.sqrt(vector1) - np.sqrt(vector2)) / np.sqrt(2)
        Hellinger_feature.append(distance)
    Hellinger_feature = np.array(Hellinger_feature)
    return Hellinger_feature


def wasserstein(original_out_vec, onDevice_out_vec):
    """
    wasserstein distance
    """
    wasserstein_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = wasserstein_distance(vector1, vector2)
        wasserstein_feature.append(distance)
    wasserstein_feature = np.array(wasserstein_feature)
    return wasserstein_feature


def mse_distance(original_out_vec, onDevice_out_vec):
    """
    Mean Squared Error (MSE) distance
    """
    mse_distance_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.mean((vector1 - vector2) ** 2)
        mse_distance_feature.append(distance)
    mse_distance_feature = np.array(mse_distance_feature)
    return mse_distance_feature


def mad_distance(original_out_vec, onDevice_out_vec):
    """
    Mean Absolute Difference (MAD) distance
    """
    mad_distance_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.mean(np.abs(vector1 - vector2))
        mad_distance_feature.append(distance)
    mad_distance_feature = np.array(mad_distance_feature)
    return mad_distance_feature


def relative_entropy(original_out_vec, onDevice_out_vec):
    """
    Relative Entropy
    """
    relative_entropy_feature = []
    for i in range(len(original_out_vec)):
        vector1 = original_out_vec[i]
        vector2 = onDevice_out_vec[i]
        distance = np.sum(vector1 * np.log2(vector2))
        relative_entropy_feature.append(distance)
    relative_entropy_feature = np.array(relative_entropy_feature)
    return relative_entropy_feature


def diff_out_vec(original_out_vec, onDevice_out_vec):
    diff_vec = onDevice_out_vec-original_out_vec
    return diff_vec


def get_all_feature(original_out_vec, onDevice_out_vec):
    diff_vec = diff_out_vec(original_out_vec, onDevice_out_vec)

    all_distance_feature = [
        get_kill_feature(original_out_vec, onDevice_out_vec),
        get_confidence_diff_feature(original_out_vec, onDevice_out_vec),
        euclidean_distance(original_out_vec, onDevice_out_vec),
        cosine_similarity(original_out_vec, onDevice_out_vec),
        manhattan_distance(original_out_vec, onDevice_out_vec),
        mse_distance(original_out_vec, onDevice_out_vec),
        mad_distance(original_out_vec, onDevice_out_vec),
        relative_entropy(original_out_vec, onDevice_out_vec),
        pearson_correlation_coefficient(original_out_vec, onDevice_out_vec),

        # chebyshev_distance(original_out_vec, onDevice_out_vec),
        # sum_squared_differences(original_out_vec, onDevice_out_vec),
        # kullback_leibler_divergence(original_out_vec, onDevice_out_vec),
        # bhattacharyya_distance(original_out_vec, onDevice_out_vec),
        # hellinger_distance(original_out_vec, onDevice_out_vec),
        # wasserstein(original_out_vec, onDevice_out_vec),

        # spearman_rank_correlation_coefficient(original_out_vec, onDevice_out_vec), # 时间较长

    ]

    all_distance_feature = np.array(all_distance_feature)
    all_distance_feature = all_distance_feature.T

    all_distance_feature = np.hstack((all_distance_feature, diff_vec))

    return all_distance_feature

