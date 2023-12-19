import torch
import torch.nn as nn
import torch.optim as optim
from ..aenets.net1 import Autoencoder
from torch.utils.data import DataLoader
from dataset import MyDataset
# from model_test import Autoencoder
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

# 加载数据位置
# root_dir = 'contact_626_vector'
root_dir = '../Processed_Datas/Collombet/Collombet_diagvector3'

model_path = '../models/Ramani_diag3_1000epochs.pth'

# 加载数据集
train_dataset = MyDataset(root_dir=root_dir, is_shuffle=True, is_mask=True,
                          random_mask=True, update_mask=True, is_train=True, mask_rate=0.1)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

data_size = train_dataset.datasize

# 是否使用训练过的模型继续训练
is_pretrained = False

# 创建模型实例并将其移动到GPU上
model = Autoencoder(data_size)
if is_pretrained:
    model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    # train_dataset.gen_mask_time = 0.0
    # train_dataset.read_dic_time = 0.0
    print(f"start Epoch [{epoch + 1}/{num_epochs}]")
    start = time.time()
    running_loss = 0.0  # 用于累积整个训练集上的损失值
    for train_data in train_loader:
        datas, _ = train_data
        if isinstance(datas, list):
            # 此时datas是由[original_datas,masked_datas]组成
            original_datas = datas[0]
            masked_datas = datas[1]
        else:
            original_datas = datas
            masked_datas = datas

        original_datas = original_datas.view(original_datas.size(0), -1).to(device)
        masked_datas = masked_datas.view(masked_datas.size(0), -1).to(device)

        reconstructed_datas = model(masked_datas)
        loss = criterion(reconstructed_datas, original_datas)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * original_datas.size(0)  # 累积损失值

    epoch_loss = running_loss / len(train_loader.dataset)  # 计算整个训练集上的平均损失值

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss*1e4:.4f}")
    print('use time: ' + str(time.time() - start))
    # print('read_dic_time: ' + str(train_dataset.read_dic_time))
    # print('gen_mask_time: ' + str(train_dataset.gen_mask_time))
    # 保存模型
    print('saving model...')
    torch.save(model.state_dict(), model_path)
    print('model saved!')
