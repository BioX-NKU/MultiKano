import numpy as np
import scipy.sparse
import torch
from sklearn.preprocessing import LabelEncoder
import anndata as ad
import scanpy as sc
import episcanpy as epi
from .utils import *

def data_processing(train_data_rna, test_data_rna, train_data_atac, test_data_atac):
    # 合并训练和测试数据，进行统一预处理
    adatas_r_all = [train_data_rna, test_data_rna]
    adatas_a_all = [train_data_atac, test_data_atac]

    # RNA数据预处理
    adata_r_all = ad.concat(adatas_r_all, label="index")
    sc.pp.normalize_total(adata_r_all, target_sum=1e4)
    sc.pp.log1p(adata_r_all)
    sc.pp.highly_variable_genes(adata_r_all, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_r_all = adata_r_all[:, adata_r_all.var.highly_variable]

    # ATAC数据预处理
    adata_a_all = ad.concat(adatas_a_all, label="index")
    epi.pp.binarize(adata_a_all)
    epi.pp.filter_features(adata_a_all, min_cells=np.ceil(0.06 * adata_a_all.shape[0]))
    tfidf_res = tfidf1_advanced(adata_a_all.X.T).T
    adata_a_all.X = tfidf_res.copy()

    # 将稀疏矩阵转换为csr格式，支持按行索引
    adata_a_all_csr = adata_a_all.X.tocsr()

    # 分离训练集和测试集
    rna_f_train = adata_r_all[adata_r_all.obs['index'] == '0'].X.toarray()
    rna_f_test = adata_r_all[adata_r_all.obs['index'] == '1'].X.toarray()

    # 使用csr_matrix进行索引
    atac_f_train = adata_a_all_csr[adata_a_all.obs['index'] == '0'].toarray()
    atac_f_test = adata_a_all_csr[adata_a_all.obs['index'] == '1'].toarray()

    # 标签编码
    label_encoder = LabelEncoder()
    rna_l_train = label_encoder.fit_transform(adata_r_all[adata_r_all.obs['index'] == '0'].obs['cell_type'])
    rna_l_test = label_encoder.transform(adata_r_all[adata_r_all.obs['index'] == '1'].obs['cell_type'])

    # 数据扩增（通过两轮拼接）
    atac_f_train_matched_round_1 = np.hstack([rna_f_train, atac_f_train])
    atac_f_train_matched_round_2 = np.hstack(
        [rna_f_train, match_next_same_label(atac_f_train, rna_l_train, atac_f_train, rna_l_train)])

    # 样本量翻倍
    train_features = np.vstack([atac_f_train_matched_round_1, atac_f_train_matched_round_2])

    # 拼接训练集RNA和ATAC特征
    test_features = np.hstack([rna_f_test, atac_f_test])

    # 将数据转为PyTorch张量
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    train_labels_tensor = torch.tensor(np.hstack([rna_l_train, rna_l_train]), dtype=torch.long)  # 样本标签翻倍
    test_labels_tensor = torch.tensor(rna_l_test, dtype=torch.long)

    # 将两轮的特征分开
    first_round_features_tensor = torch.tensor(atac_f_train_matched_round_1, dtype=torch.float32)
    second_round_features_tensor = torch.tensor(atac_f_train_matched_round_2, dtype=torch.float32)
    num_classes = len(np.unique(rna_l_train))

    # 返回处理后的训练集和测试集
    return train_features_tensor, train_labels_tensor, test_features_tensor, test_labels_tensor, first_round_features_tensor, second_round_features_tensor, num_classes