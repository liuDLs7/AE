import os
import time
from torch.utils.data import Dataset
import numpy as np
import random
import json


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def generate_random_numbers(k: int, m1: int, m2: int):
    # 随机生成k个值，范围在(m1,m2)中
    if m1 > m2 or k > m2 - m1 + 1:
        raise ValueError("an error raised in generating random mask index!")

    numbers = random.sample(range(m1, m2), k)
    numbers.sort()
    return numbers


class MyDataset(Dataset):

    def __init__(self,
                 root_dir: str = 'contact_626_diagvector3',
                 is_mask: bool = False,
                 update_mask: bool = False,
                 random_mask: bool = False,  # True表示对每个细胞都随机覆盖
                 mask_rate: float = 0.1,
                 is_train: bool = True,
                 is_shuffle: bool = False):

        self.is_mask = is_mask
        self.random_mask = random_mask
        self.mask_rate = mask_rate

        self.root_dir = root_dir
        sub_dirs = get_subdirectories(root_dir)
        self.datas = {}

        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))
            label = sub_dir
            for file_name in file_names:
                file_path = os.path.join(root_dir, sub_dir, file_name)
                self.datas[file_path] = label

        if is_shuffle:
            # 获取字典的键值对列表
            items = list(self.datas.items())
            # 打乱列表顺序
            random.shuffle(items)
            # 将打乱顺序后的列表转换回字典
            self.datas = dict(items)

        self.datasize = self.get_datasize()
        if is_train:
            self.mask_file = os.path.join(root_dir, 'train_mask_index.json')
        else:
            self.mask_file = os.path.join(root_dir, 'test_mask_index.json')

        # 生成掩码位置索引
        if is_mask and update_mask:
            print('updating '+self.mask_file)
            if not self.random_mask:
                # 生成一组index作为mask索引，所有细胞共用该索引组，考虑存储该索引，在计算相似度时，可以适当提高这部分权值
                self.global_mask_index = generate_random_numbers(int(self.mask_rate * self.datasize), 0, self.datasize)
                with open(self.mask_file, 'w') as file:
                    json.dump(self.global_mask_index, file)
            else:
                # 为每一个细胞生成一组掩码位置索引，并保存
                self.masks_index = {}
                for sub_dir in sub_dirs:
                    file_names = os.listdir(os.path.join(root_dir, sub_dir))
                    for file_name in file_names:
                        file_path = os.path.join(root_dir, sub_dir, file_name)
                        mask_index = generate_random_numbers(int(self.mask_rate * self.datasize), 0, self.datasize)
                        self.masks_index[file_path] = mask_index

                with open(self.mask_file, 'w') as file:
                    json.dump(self.masks_index, file)
                print(self.mask_file+' is updated!')

        self.read_dic_time = 0.0
        self.gen_mask_time = 0.0

    def __getitem__(self, idx):
        data_item_path = list(self.datas.keys())[idx]
        # data = torch.from_numpy(np.load(data_item_path)).float()
        data = np.load(data_item_path).astype(np.float32)
        data = data/max(data)
        label = list(self.datas.values())[idx]

        if self.is_mask:
            # 同时保存掩码前后的数据，掩码后的数据作为输入，掩码前的数据用于计算loss
            combined_data = [data]

            if self.random_mask:
                # 随机生成一组索引，作为mask位置，考虑存储该索引，在计算相似度时，可以适当提高这部分权值
                # io操作太费时间了
                # start_time = time.time()
                # with open(self.mask_file, 'r') as file:
                #     mask_index = json.load(file)[data_item_path]
                # self.read_dic_time += time.time() - start_time
                mask_index = self.masks_index[data_item_path]
                start_time = time.time()
                masked_data = np.copy(data)
                masked_data[mask_index] = 0
                combined_data.append(masked_data)
                self.gen_mask_time += time.time() - start_time
            else:
                # 使用__init__中生成的索引

                # with open(self.mask_file, 'r') as file:
                #     mask_index = json.load(file)
                mask_index = self.global_mask_index
                masked_data = np.copy(data)
                masked_data[mask_index] = 0
                combined_data.append(masked_data)

            return combined_data, label

        else:
            return data, label

    def __len__(self):
        return len(self.datas)

    def get_datasize(self):
        return len(np.load(list(self.datas.keys())[0]))


if __name__ == '__main__':
    t = MyDataset(is_shuffle=False,is_mask=True,update_mask=False,random_mask=False)
    data, _ = t[0]
    print(data[0][:50])
    print(data[1][:50])
    # print(data[:50])
    pass
