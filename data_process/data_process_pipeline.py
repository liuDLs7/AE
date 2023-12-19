from file_process import FileProcess
from matrix2vector import Matrix2Arr1D
import time


def data_process(args):
    root_dir, target_dir1, target_dir2, chr_num, is_X, is_Y, is_write, m, process_pattern, norm_mode, chr_len = args

    print('file_processing start!')

    time1 = time.time()

    file_process = FileProcess(root_dir=root_dir, target_dir=target_dir1,
                               chr_num=chr_num, is_X=is_X, is_Y=is_Y)
    file_process.process_file()

    time2 = time.time()

    print('file_processing completed!')
    print('use time: ' + str(time2-time1))
    print('matrix_flattening start!')

    matrix_flattener = Matrix2Arr1D(root_dir=target_dir1, target_dir=target_dir2, is_write=True, m=m, chr_num=chr_num,
                                    process_pattern=process_pattern, norm_mode=norm_mode)
    matrix_flattener.flatten()

    time3 = time.time()

    print('matrix_flattening completed!')
    print('use time: ' + str(time3-time2))


if __name__ == '__main__':
    # 需要设定的参数
    root_dir = '../Original_Datas/Lee'
    target_dir1 = '../Processed_Datas/Lee/Lee_processed'
    target_dir2 = '../Processed_Datas/Lee/Lee_diagvector3'
    chr_num = 23    # 总染色体数(含X,Y)
    is_X = False
    is_Y = False
    is_write = True
    m = 3
    process_pattern = 'diag'
    norm_mode = 'None'
    chr_len = []    # 默认不使用该参数
    args = [root_dir, target_dir1, target_dir2, chr_num, is_X, is_Y, is_write, m, process_pattern, norm_mode, chr_len]
    data_process(args)


