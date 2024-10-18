import numpy as np
import scipy.sparse
import torch

def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T) / (a_norm * b_norm.T)


def tfidf1_advanced(count_mat):
    if not scipy.sparse.issparse(count_mat):
        count_mat = scipy.sparse.coo_matrix(count_mat)
    nfreqs = count_mat.multiply(1.0 / count_mat.sum(axis=0))
    tfidf_mat = nfreqs.multiply(np.log(1 + 1.0 * count_mat.shape[1] / count_mat.sum(axis=1)).reshape(-1, 1)).tocoo()
    return tfidf_mat


def match_next_same_label(feature_1, labels_1, feature_2, labels_2):
    matched_feature_2 = np.zeros_like(feature_2)
    for i in range(len(labels_1)):
        current_label = labels_1[i]
        for j in range(i + 1, len(labels_2)):
            if labels_2[j] == current_label:
                matched_feature_2[i] = feature_2[j]
                break
        else:
            matched_feature_2[i] = feature_2[i]
    return matched_feature_2