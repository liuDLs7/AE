import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def neighbor_ave_gpu(A, pad):
    if pad == 0:
        return torch.from_numpy(A).float().cuda()
    ll = pad * 2 + 1
    conv_filter = torch.ones(1, 1, ll, ll).cuda()
    B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().cuda(), conv_filter, padding=pad * 2)
    return B[0, 0, pad:-pad, pad:-pad] / float(ll * ll)


"""
pad = 2
A = np.random.rand(8,8)
ll = pad * 2 + 1
conv_filter = torch.ones(1, 1, ll, ll).cuda()
B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().cuda(), conv_filter, padding=pad * 2)
C = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().cuda(), conv_filter, padding=pad)
print((B[0, 0, pad:-pad, pad:-pad] / float(ll * ll)).shape)
print((B[0, 0, pad:-pad, pad:-pad] / float(ll * ll)))
print((C[0, 0, :, :] / float(ll * ll)).shape)
print((C[0, 0, :, :] / float(ll * ll)))
"""


def random_walk_gpu(A, rp):
    ngene, _ = A.shape
    A = A - torch.diag(torch.diag(A))
    A = A + torch.diag(torch.sum(A, 0) == 0).float()

    P = torch.div(A, torch.sum(A, 0))
    Q = torch.eye(ngene).cuda()
    I = torch.eye(ngene).cuda()
    for i in range(30):
        Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
        delta = torch.norm(Q - Q_new, 2)
        Q = Q_new
        if delta < 1e-6:
            break
    return Q


def impute_gpu(args):
    cell, c, ngene, pad, rp = args
    D = np.loadtxt(cell + '_chr' + c + '.txt')
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
    A = np.log2(A + A.T + 1)
    A = neighbor_ave_gpu(A, pad)
    if rp == -1:
        Q = A[:]
    else:
        Q = random_walk_gpu(A, rp)
    return Q.reshape(ngene * ngene)


# def hicluster_gpu(network, chromsize, nc, res=1000000, pad=1, rp=0.5, prct=20, ndim=20):
#     #             细胞类型， 染色体长度，n类细胞，分辨率   ，  填充，  重启概率，取前百分比 ，降维数
#     matrix = []
#     # 将所有细胞的每条染色体信息分别PCA，最后合并
#     for i, c in enumerate(chromsize):
#         ngene = int(chromsize[c] / res) + 1
#         start_time = time.time()
#         # reshape 成 m*(n*n) 的 size
#         Q_concat = torch.zeros(len(network), ngene * ngene).float().cuda()
#         for j, cell in enumerate(network):
#             Q_concat[j] = impute_gpu([cell, c, ngene, pad, rp])
#         Q_concat = Q_concat.cpu().numpy()
#         if prct > -1:
#             thres = np.percentile(Q_concat, 100 - prct, axis=1)
#         Q_concat = (Q_concat > thres[:, None])
#         end_time = time.time()
#         print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
#         # TODO: why ?
#         ndim = int(min(Q_concat.shape) * 0.2) - 1
#         # U, S, V = torch.svd(Q_concat, some=True)
#         # R_reduce = torch.mm(U[:, :ndim], torch.diag(S[:ndim])).cuda().numpy()
#         pca = PCA(n_components=ndim)
#         R_reduce = pca.fit_transform(Q_concat)
#         matrix.append(R_reduce)
#         print(c)
#     matrix = np.concatenate(matrix, axis=1)
#     pca = PCA(n_components=min(matrix.shape) - 1)
#     matrix_reduce = pca.fit_transform(matrix)
#     kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, :ndim])
#     return kmeans.labels_, matrix_reduce
# # TODO: where is whitening matrix? embedding?

# 重写
def hicluster_gpu(network, ngenes, nc, pad=1, rp=0.5, prct=20, ndim=20, is_X=False):
    #             细胞类型， 染色体长度，n类细胞，分辨率   ，  填充，  重启概率，取前百分比 ，降维数
    matrix = []
    # 将所有细胞的每条染色体信息分别PCA，最后合并
    for c, ngene in enumerate(ngenes):
        # print('ndim = ' + str(ndim))
        c = 'X' if is_X and c == len(ngenes) - 1 else str(c + 1)
        start_time = time.time()
        # reshape 成 m*(n*n) 的 size
        Q_concat = torch.zeros(len(network), ngene * ngene).float().cuda()
        for j, cell in enumerate(network):
            Q_concat[j] = impute_gpu([cell, c, ngene, pad, rp])
        Q_concat = Q_concat.cpu().numpy()
        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
        Q_concat = (Q_concat > thres[:, None])
        end_time = time.time()
        print('Load and impute chromosome', c, 'take', end_time - start_time, 'seconds')
        # TODO: why ?
        ndim = int(min(Q_concat.shape) * 0.2) - 1
        print(Q_concat.shape)
        # U, S, V = torch.svd(Q_concat, some=True)
        # R_reduce = torch.mm(U[:, :ndim], torch.diag(S[:ndim])).cuda().numpy()
        pca = PCA(n_components=ndim)
        R_reduce = pca.fit_transform(Q_concat)
        matrix.append(R_reduce)
        print(c)
    matrix = np.concatenate(matrix, axis=1)
    pca = PCA(n_components=min(matrix.shape) - 1)
    matrix_reduce = pca.fit_transform(matrix)
    print('ndim = ' + str(ndim))
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, :ndim])
    return kmeans.labels_, matrix_reduce


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


if __name__ == '__main__':
    ngenes = [
      250,
      244,
      198,
      192,
      181,
      171,
      160,
      147,
      142,
      136,
      135,
      134,
      116,
      108,
      103,
      91,
      82,
      79,
      60,
      63,
      49,
      52,
      155
   ]
    root_dir = 'Original_Datas/Ramani'
    label_dirs = get_subdirectories(root_dir)
    str2dig = {}
    x = []
    y = []
    network = []
    nc = 4
    is_X = True
    chr_num = 23

    for i, label_name in enumerate(label_dirs):
        str2dig[label_name] = i

    print(str2dig)

    for label_dir in label_dirs:
        sub_path = os.path.join(root_dir, label_dir)
        files = os.listdir(sub_path)
        file_num = 0
        for file in files:
            file_num += 1
        cell_num = int(file_num / chr_num)
        for i in range(1, cell_num + 1):
            cell_path = os.path.join(sub_path, 'cell_' + str(i))
            network.append(cell_path)
            y.append(str2dig[label_dir])

    # print(len(network))
    # print(len(y))
    y = np.array(y)
    # print(y)
    # for i in range(nc):
    #     print(np.count_nonzero(y == i))

    cluster_labels, matrix_reduced = hicluster_gpu(network=network, ngenes=ngenes, nc=nc, pad=1, rp=0.5
                                                   , prct=20, ndim=20, is_X=is_X)

    # print(len(cluster_labels))
    # print(cluster_labels)
    # for i in range(nc):
    #     print(np.count_nonzero(cluster_labels == i))

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # 计算调整兰德指数和归一化互信息
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
