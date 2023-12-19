import os
import time


def get_gnene(args, tdir):
    doc, cellt, celln, c = args
    os.makedirs(os.path.join(tdir, cellt), exist_ok=True)
    spath = os.path.join(doc, cellt, 'cell_' + celln + '_chr' + c + '.txt')
    tpath = os.path.join(tdir, cellt, 'cell_' + celln + '_chr' + c + '.txt')

    # 读取输入文件
    with open(spath, 'r') as file:
        lines = file.readlines()

    # 提取前两列数据并找出最大值
    max_value = float('-inf')  # 初始化最大值为负无穷
    for line in lines:
        values = line.split()
        if len(values) >= 2:
            col1, col2 = float(values[0]), float(values[1])
            max_value = max(max_value, col1, col2)

    # 将最大值写入输出文件的第一行
    lines.insert(0, str(int(max_value)) + '\n')

    # 写入输出文件
    with open(tpath, 'w') as file:
        file.writelines(lines)


def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


def write_with_gnene(root_dir: str, target_dir: str, chr_num: int, is_X: bool, is_Y: bool = False):
    # 获取一级子文件夹名
    subdirectories = get_subdirectories(root_dir)

    cell_counts = []

    for subdirectory in subdirectories:
        # 文件夹路径
        folder_path = os.path.join(root_dir, subdirectory)
        # 获取文件夹下的所有文件
        files = os.listdir(folder_path)
        # 统计文件数量
        chr_count = 0
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                chr_count += 1
        cell_counts.append(int(chr_count / chr_num))

    for i, cell_type in enumerate(subdirectories):
        for celln in range(1, cell_counts[i] + 1):
            # 考虑X,Y染色体对编号的影响
            if is_X and is_Y:
                tmp = chr_num-2
            elif is_X or is_Y:
                tmp = chr_num-1
            else:
                tmp = chr_num
            for chrn in range(1, tmp + 1):
                args = [root_dir, cell_type, str(celln), str(chrn)]
                get_gnene(args, target_dir)
            if is_X:
                args = [root_dir, cell_type, str(celln), 'X']
                get_gnene(args, target_dir)
            if is_Y:
                args = [root_dir, cell_type, str(celln), 'Y']
                get_gnene(args, target_dir)


class FileProcess:
    def __init__(self,
                 root_dir: str,
                 target_dir: str,
                 chr_num: int,
                 is_X: bool,
                 is_Y: bool = False):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.chr_num = chr_num
        self.is_X = is_X
        self.is_Y = is_Y

    def process_file(self):
        write_with_gnene(self.root_dir, self.target_dir, self.chr_num, self.is_X, self.is_Y)


if __name__ == '__main__':

    # root = 'Flyamer'
    # tdir = 'Flyamer_processed'
    # chr_num = 20
    # is_X = True
    # is_Y = False
    #
    # start_time = time.time()
    # write_with_gnene(root, tdir, chr_num, is_X, is_Y)
    # print('process use time: ' + str(time.time() - start_time))
    pass