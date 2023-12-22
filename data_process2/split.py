import re
import os
import shutil
import random


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def main():
    # **********************************调参部分*******************************************
    root_dir = '../../vectors/Ramani/diag5'
    train_dir = '../../vectors/Ramani/diag5_train'
    test_dir = '../../vectors/Ramani/diag5_test'

    chr_num = 23
    is_random = True
    train_rate = 0.75
    # ************************************************************************************

    sub_dirs = get_subdirectories(root_dir)
    data_info_path = os.path.join(root_dir, 'data_info.json')

    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        train_sub_dir = os.path.join(train_dir, sub_dir)
        test_sub_dir = os.path.join(test_dir, sub_dir)

        os.makedirs(train_sub_dir, exist_ok=True)
        os.makedirs(test_sub_dir, exist_ok=True)

        file_names = os.listdir(sub_path)
        cell_num = int(len(file_names) / chr_num)
        train_size = int(cell_num * train_rate)

        # 生成 1 到 n 的整数列表
        numbers = list(range(1, cell_num + 1))

        # 随机打乱整数列表
        if is_random:
            random.shuffle(numbers)

        # 将前 count 个整数放入 A 列表，剩余的放入 B 列表
        train_list = numbers[:train_size]
        test_list = numbers[train_size:]

        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            match = re.search(r'cell_(\d+)_chr([0-9XY]+).npy', file_name)
            cell_number = int(match.group(1))
            chromosome_number = int(match.group(2)) if match.group(2) != 'X' else chr_num
            if cell_number in train_list:
                shutil.copy(file_path, train_sub_dir)
            else:
                shutil.copy(file_path, test_sub_dir)

        print(sub_dir + ' has been processed!')

    shutil.copy(data_info_path, train_dir)
    shutil.copy(data_info_path, test_dir)


if __name__ == '__main__':
    main()
