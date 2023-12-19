import datetime
import json
import re
import numpy as np
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
from tqdm import tqdm


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def get_max_chr_len(root_dir, chr_num):
    # 获取每条染色体的最大长度用于对齐
    max_len = [0] * chr_num
    subdirectories = get_subdirectories(root_dir)

    for subdirectory in subdirectories:
        # 文件夹路径
        folder_path = os.path.join(root_dir, subdirectory)
        # 获取文件夹下的所有文件
        files = os.listdir(folder_path)

        for file_name in files:
            # 根据文件名获取染色体信息
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
            cell_number = int(match.group(1))
            chromosome_number = int(match.group(2)) if match.group(2) != 'X' else chr_num

            file_path = os.path.join(root_dir, subdirectory, file_name)
            # 打开文件
            with open(file_path, 'r') as file:
                # 读取第一行
                first_line = file.readline().strip()
                # 转换为整数
                ngene = int(float(first_line)) + 1
                max_len[chromosome_number - 1] = max(ngene, max_len[chromosome_number - 1])

    return max_len


def neighbor_ave_gpu(A, pad):
    if pad == 0:
        return torch.from_numpy(A).float().cuda()
    ll = pad * 2 + 1
    conv_filter = torch.ones(1, 1, ll, ll).cuda()
    B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().cuda(), conv_filter, padding=pad * 2)
    return B[0, 0, pad:-pad, pad:-pad] / float(ll * ll)


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


def impute_gpu(ngene, pad, rp, file_path):
    D = np.loadtxt(file_path)
    A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape=(ngene, ngene)).toarray()
    A = np.log2(A + A.T + 1)
    A = neighbor_ave_gpu(A, pad)
    if rp == -1:
        Q = A[:]
    else:
        Q = random_walk_gpu(A, rp)
    return Q


def normalize_by_chr(ngene, pad, rp, file_path, mode='None'):
    # not to conv when pad == 0, not to random_walk when rp == -1
    Q = impute_gpu(ngene, pad, rp, file_path)
    assert mode in ['None', 'chr_max', 'chr_sum'], \
        print('normalize_mode should in [\'None\', \'chr_max\', \'chr_sum\']')
    if mode == 'None':
        param = 1.0
    elif mode == 'chr_max':
        param = torch.max(Q)
    elif mode == 'chr_sum':
        param = torch.sum(Q)
    else:
        assert 0, exit(2)
    return Q / param


def myflatten(A, process_pattern: str = 'row', m: int = -1):
    if process_pattern == 'row':
        # 按行取
        # 获取上三角矩阵的索引
        if m != -1:
            A = A[:m, :]
        # 拉伸为一维向量
        indices = np.triu_indices_from(A)

        B = A[indices].flatten()

    elif process_pattern == 'diag':
        # 按对角线取，只取靠近主对角线的m条（含主对角线)
        if m != -1:
            upper_diags = [np.diagonal(A, offset=i) for i in range(0, m)]
        else:
            upper_diags = [np.diagonal(A, offset=i) for i in range(0, A.shape[0])]

        B = np.concatenate(upper_diags)

    else:
        assert 0, print('error!')

    return B


def main():
    # **********************************调参部分*******************************************
    root_dir = '../../Original_Datas/4DN'
    chr_num = 23
    target_dir = '../../vectors/4DN/diag3'
    processed_dir = '../../Processed_Datas/4DN/4DN_processed'

    pad = 0
    rp = -1
    mode = 'chr_max'
    process_pattern = 'diag'
    m = 3
    notes = 'flatten the whole matrix to vector, not just the upper_triangle'
    # ************************************************************************************

    sub_dirs = get_subdirectories(root_dir)

    chr_lens = get_max_chr_len(processed_dir, chr_num=chr_num)
    print(chr_lens)
    # chr_lens = [250, 244, 198, 192, 181, 171, 160, 147, 142, 136, 135, 134, 116, 108, 103, 91, 82, 79, 60, 63, 49, 52,
    #             155]

    start = time.time()

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        target_sub_dir = os.path.join(target_dir, sub_dir)
        file_names = os.listdir(sub_path)

        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).txt', file_name)
            cell_number = int(match.group(1))
            chromosome_number = int(match.group(2)) if match.group(2) != 'X' else chr_num
            ngene = chr_lens[chromosome_number - 1]
            M = normalize_by_chr(ngene=ngene, pad=pad, rp=rp, file_path=file_path, mode=mode)
            M = M.cpu().numpy()
            M_vector = myflatten(M, process_pattern=process_pattern, m=m)
            # M_vector = M.flatten()

            # 将处理后的数据写入文件
            # 可以先用这个创建文件夹
            os.makedirs(target_sub_dir, exist_ok=True)
            target_file = os.path.join(target_sub_dir, file_name[:-4] + '.npy')
            np.save(target_file, M_vector)

        print(sub_dir + ' has been processed!')

    print('use time: ' + str(time.time() - start))

    data_info = {
        'root_dir': root_dir,
        'processed_dir': processed_dir,
        'target_dir': target_dir,
        'chr_num': chr_num,
        'pad': pad,
        'rp': rp,
        'mode': mode,
        'process_pattern': process_pattern,
        'm': m,
        'chr_lens': chr_lens,
        'last_update': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'notes':notes
    }

    data_info_file_path = os.path.join(target_dir, 'data_info.json')

    with open(data_info_file_path, 'w') as json_file:
        json.dump(data_info, json_file, indent=3)


if __name__ == '__main__':
    main()
