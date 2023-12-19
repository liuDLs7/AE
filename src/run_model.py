import torch
from nets.net1 import Autoencoder
# from model_test import Autoencoder
from dataset import MyDataset
import numpy as np
from torch.utils.data import DataLoader


def sccByDiag(x1, x2):
    # convert each diagonal to one row of a csr_matrix in order to compute
    # diagonal-wise correlation between m1 and m2
    m1D = x1
    m2D = x2
    nSamplesD = np.count_nonzero(m1D + m2D)
    rowSumM1D = m1D.sum()
    rowSumM2D = m2D.sum()
    # ignore zero-division warnings because the corresponding elements in the
    # output don't contribute to the SCC scores
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = np.multiply(m1D, m2D).sum() - rowSumM1D * rowSumM2D / nSamplesD
        rhoD = cov / np.sqrt(
            (np.multiply(m1D, m1D).sum() - np.square(rowSumM1D) / nSamplesD) *
            (np.multiply(m2D, m2D).sum() - np.square(rowSumM2D) / nSamplesD))
        wsD = nSamplesD * nSamplesD
        # Convert NaN and Inf resulting from div by 0 to zeros.
        # posinf and neginf added to fix behavior seen in 4DN datasets
        # 4DNFIOQLTI9G and DNFIH7MQHOR at 5kb where inf would be reported
        # as an SCC score
        wsNan2Zero = np.nan_to_num(wsD, copy=True, posinf=0.0, neginf=0.0)
        rhoNan2Zero = np.nan_to_num(rhoD, copy=True, posinf=0.0, neginf=0.0)

    return rhoNan2Zero.astype(np.float32) * wsNan2Zero.astype(np.float32) / wsNan2Zero.sum()


def pearson_similarity(arr1, arr2):
    correlation_matrix = np.corrcoef(arr1, arr2)
    return correlation_matrix[0, 1]


def cosine_similarity(arr1, arr2):
    dot_product = np.dot(arr1, arr2)
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)
    return dot_product / (norm_arr1 * norm_arr2)


def euclidean_distance(arr1, arr2):
    distance = np.linalg.norm(arr1 - arr2)
    return 1 / (1 + distance)


# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)

# 加载数据位置
# root_dir = 'contact_626_vector'
root_dir = 'contact_626_diagvector3'

# model_path = 'autoencoder.pth'
model_path = '75.pth'

# 加载数据集

test_dataset = MyDataset(root_dir=root_dir, is_shuffle=False, is_mask=True,
                         random_mask=False, update_mask=True, is_train=False, mask_rate=0.1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

data_size = test_dataset.datasize

# 创建模型实例
model = Autoencoder(data_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 查看模型每层的参数值
# for i, (name, param) in enumerate(model.named_parameters()):
#     print(f"Layer {i + 1}: {name} - {param.data.numpy().shape}")
#     print(param.data.numpy())

# print(model.parameters()
# exit(0)

total = 0.0
sum_similarity = 0.0
cycles = 99
i = 0
with torch.no_grad():
    for i, test_data in enumerate(test_loader):

        # file_name = list(test_dataset.datas.keys())[i]

        datas, _ = test_data
        if isinstance(datas, list):
            # 此时datas是由[original_datas,masked_datas]组成
            original_datas = datas[0]
            masked_datas = datas[1]
        else:
            original_datas = datas
            masked_datas = datas

        original_datas = original_datas.view(original_datas.size(0), -1).to(device)
        masked_datas = masked_datas.view(masked_datas.size(0), -1).to(device)

        reconstructed_datas = model(masked_datas).to(device)

        # print(file_name)
        # print(np.load(file_name))
        # print(original_datas)
        print(reconstructed_datas[:10])

        sum_similarity += pearson_similarity(original_datas, reconstructed_datas)

        total += 1

        i += 1
        if i > cycles:
            break
        # print(reconstructed_datas)

        # print(datas[:20])
        # print(reconstructed_datas[:20])
        # print(datas.shape)
        # datas = datas.flatten().numpy()
        # reconstructed_datas = reconstructed_datas.flatten().numpy()
        # # print(sccByDiag(datas,reconstructed_datas))

print('average similarity = ', sum_similarity / total)
print(total)
