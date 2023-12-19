from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dataset import MyDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from aenets.net1 import AutoEncoder
import os


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def run_on_model(root_dir, model_path, nc, ndim):
    # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(device)

    # 加载数据集

    test_dataset = MyDataset(root_dir=root_dir, is_shuffle=False, is_mask=False,
                             random_mask=False, update_mask=False, is_train=False, mask_rate=0.1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    data_size = test_dataset.datasize

    # 创建模型实例
    # model = AutoEncoder(data_size).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    label_dirs = get_subdirectories(root_dir)
    str2dig = {}
    x = []
    y = []

    for i, label_name in enumerate(label_dirs):
        str2dig[label_name] = i

    print(str2dig)

    total = 0.0
    sum_similarity = 0.0
    cycles = 9999
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):

            file_name = list(test_dataset.datas.keys())[i]

            datas, flag = test_data
            if isinstance(datas, list):
                # 此时datas是由[original_datas,masked_datas]组成
                original_datas = datas[0]
                masked_datas = datas[1]
            else:
                original_datas = datas
                masked_datas = datas

            original_datas = original_datas.view(original_datas.size(0), -1).to(device)
            # masked_datas = masked_datas.view(masked_datas.size(0), -1).to(device)

            embedding = model.encoder(original_datas).to(device)
            reconstructed_datas = model.decoder(embedding).to(device)

            # x.append(original_datas)
            # (ARI): 0.7321816897176677
            # (NMI): 0.816458083762365

            # x.append(reconstructed_datas)
            # (ARI): 0.02345865826960453
            # (NMI): 0.049204928920091616

            # x.append(embedding)
            # (ARI): 0.09403542950656404
            # (NMI): 0.07817979017876982

            x.append(np.copy(embedding.numpy()))
            # print(file_name)
            # print(flag[0])
            y.append(str2dig[flag[0]])

            i += 1
            if i > cycles:
                break

    # 假设 X_train 是训练后的样本数据，y_train 是样本标签
    # 这里假设 X_train 是一个二维的特征数据
    # 假设我们选择将数据映射到二维空间进行观察
    X_train = np.concatenate(x, axis=0)
    y_train = np.array(y)

    print(X_train.shape)
    print(y_train.shape)

    pca = PCA(n_components=min(X_train.shape) - 1)
    matrix_reduce = pca.fit_transform(X_train)
    kmeans = KMeans(n_clusters=nc, n_init=200).fit(matrix_reduce[:, 1:ndim])

    print(matrix_reduce.shape)
    print(kmeans.labels_.shape)

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # 计算调整兰德指数和归一化互信息
    ari = adjusted_rand_score(y_train, kmeans.labels_)
    nmi = normalized_mutual_info_score(y_train, kmeans.labels_)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)


def run_original_data(root_dir):
    label_dirs = get_subdirectories(root_dir)
    str2dig = {}
    x = []
    y = []

    for i, label_name in enumerate(label_dirs):
        str2dig[label_name] = i

    print(str2dig)

    for label_dir in label_dirs:
        sub_path = os.path.join(root_dir, label_dir)
        files = os.listdir(sub_path)
        for file in files:
            file_path = os.path.join(sub_path, file)
            tmp = np.load(file_path)
            x.append(tmp)
            y.append(str2dig[label_dir])

    X_train = np.array(x)
    y_train = np.array(y)

    print(X_train.shape)
    print(y_train.shape)

    pca = PCA(n_components=min(X_train.shape) - 1)
    matrix_reduce = pca.fit_transform(X_train)
    kmeans = KMeans(n_clusters=4, n_init=200).fit(matrix_reduce[:, 1:(20 + 1)])

    print(matrix_reduce.shape)
    print(kmeans.labels_.shape)

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # 计算调整兰德指数和归一化互信息
    ari = adjusted_rand_score(y_train, kmeans.labels_)
    nmi = normalized_mutual_info_score(y_train, kmeans.labels_)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)


if __name__ == '__main__':
    # 加载数据位置
    root_dir = '../vectors/Ramani/diag3'
    model_path = '../models/Ramani_diag3_1000epochs.pth'
    nc = 5
    ndim = 20
    run_on_model(root_dir, model_path, nc, ndim)
