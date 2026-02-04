import numpy as np
import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from VecMulUnit import tree_vec_mul_128_group4, tree_vec_mul_128_group8, tree_vec_mul_128_group2
from utils import do_conv_torch, compare_result, load_data, tpu_conv_first_channel


def profile_layer_sparsity(data, weights, out_channels, in_channels, kernel_size, stride, padding):
    '''
    do convolution using different tree_vec_mul_128_group, and profile the sparsity of operations
    '''
    assert padding == 1, "current only support padding = 1"
    assert stride == 1, "current only support stride = 1"

    ref_result = do_conv_torch(data, weights, out_channels, in_channels, kernel_size, stride, padding)
    index_channel = 0
    

    

    # group size = 8
    print("=============================group size = 8======================================")
    
    tree_ver_mul_group8 = tree_vec_mul_128_group8()
    index_channel, tpu_result, skip_recordings, weights_update_time, total_do_compute_count = tpu_conv_first_channel(data, weights, in_channels, 1, padding, tree_ver_mul_group8,index_channel)
    max_diff, mean_diff, max_relative_diff = compare_result(tpu_result, ref_result[0,index_channel])
    print("============compare with reference result=============")
    print(f"group size = 8, max_diff: {max_diff}, mean_diff: {mean_diff}, max_relative_diff: {max_relative_diff}")
    print(f"skip_recordings: {skip_recordings}")
    print(f"total_do_compute_count: {total_do_compute_count}")
    print(f"weights_update_time: {weights_update_time}")
    print(f"level1 (128 -> 16)    skipped operation rate: {skip_recordings[0] / total_do_compute_count / 16}")
    print(f"level2 (16 -> 2)      skipped operation rate: {skip_recordings[1] / total_do_compute_count / 2}")
    print("\n")



    print("=============================group size = 4======================================")

    # group size = 4
    tree_ver_mul_unit = tree_vec_mul_128_group4()
    index_channel, tpu_result, skip_recordings, weights_update_time, total_do_compute_count = tpu_conv_first_channel(data, weights, in_channels, 1, padding, tree_ver_mul_unit,index_channel)
    max_diff, mean_diff, max_relative_diff = compare_result(tpu_result, ref_result[0,index_channel])
    print("============compare with reference result=============")
    print(f"group size = 4, max_diff: {max_diff}, mean_diff: {mean_diff}, max_relative_diff: {max_relative_diff}")
    print(f"skip_recordings: {skip_recordings}")
    print(f"total_do_compute_count: {total_do_compute_count}")
    print(f"weights_update_time: {weights_update_time}")
    print(f"level1 (128 -> 32)    skipped operation rate: {skip_recordings[0] / total_do_compute_count / 32}")
    print(f"level2 (32 -> 8)      skipped operation rate: {skip_recordings[1] / total_do_compute_count / 8}")
    print(f"level3 (8 -> 2)       skipped operation rate: {skip_recordings[2] / total_do_compute_count / 2}")
    print("\n")


    print("=============================group size = 2======================================")

    # group size = 2
    tree_ver_mul_unit = tree_vec_mul_128_group2()
    index_channel, tpu_result, skip_recordings, weights_update_time, total_do_compute_count = tpu_conv_first_channel(data, weights, in_channels, 1, padding, tree_ver_mul_unit,index_channel)
    max_diff, mean_diff, max_relative_diff = compare_result(tpu_result, ref_result[0,index_channel])
    print("============compare with reference result=============")
    print(f"group size = 2, max_diff: {max_diff}, mean_diff: {mean_diff}, max_relative_diff: {max_relative_diff}")
    print(f"skip_recordings: {skip_recordings}")
    print(f"total_do_compute_count: {total_do_compute_count}")
    print(f"weights_update_time: {weights_update_time}")
    print(f"level1 (128 -> 64)    skipped operation rate: {skip_recordings[0] / total_do_compute_count / 64}")
    print(f"level2 (64 -> 32)     skipped operation rate: {skip_recordings[1] / total_do_compute_count / 32}")
    print(f"level3 (32 -> 16)     skipped operation rate: {skip_recordings[2] / total_do_compute_count / 16}")
    print(f"level4 (16 -> 8)      skipped operation rate: {skip_recordings[3] / total_do_compute_count / 8}")
    print(f"level5 (8 -> 4)       skipped operation rate: {skip_recordings[4] / total_do_compute_count / 4}")
    print(f"level6 (4 -> 2)       skipped operation rate: {skip_recordings[5] / total_do_compute_count / 2}")
    print("\n")
    print("=============================profile sparsity done======================================")


if __name__ == "__main__":
    print("=============================profile sparsity start======================================")

    in_channels = 256
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 1
    height = 96
    width = 160

    #in_channels = 512
    #out_channels = 128
    #kernel_size = 3
    #stride = 1
    #padding = 1
    #height = 48
    #width = 80

    # load data
    data_path = './extract_data/conv/input_001.txt'
    #'./extract_data/step13_conv/input_001.txt'
    #'./extract_data/step13_conv/input_001.txt'
    data = load_data(data_path, (in_channels, height, width))
    

    weight_Path = './extract_data/conv/weights.txt'
    #'./extract_data/step13_conv/weights.txt'
    #'./extract_data/step13_conv/weights.txt'
    weights = load_data(weight_Path, (out_channels, in_channels, kernel_size, kernel_size))
    # mapping weights to -1, 0, 1
    weights = np.where(weights > 0, 1, np.where(weights < 0, -1, 0))

    # analyze the sparsity of weights
    print("=============================analyze the sparsity of weights start======================================")
    print(f"weights shape: {weights.shape}")
    print(f"sparsity of weights: {np.sum(weights == 0) / np.prod(weights.shape)}")
    print("=============================analyze the sparsity of weights done======================================")
    print("\n")
    # analyze the sparsity of data
    print("=============================analyze the sparsity of data start======================================")
    print(f"data shape: {data.shape}")
    print(f"sparsity of data: {np.sum(data == 0) / np.prod(data.shape)}")
    print(f"equal to 1 of data: {np.sum(data == 1) / np.prod(data.shape)}")
    print(f"equal to 2 of data: {np.sum(data == 2) / np.prod(data.shape)}")
    print(f"equal to 3 of data: {np.sum(data == 3) / np.prod(data.shape)}")
    print("=============================analyze the sparsity of data done======================================")
    print("\n")

    profile_layer_sparsity(data, weights, 64, 256, 3, 1, 1)