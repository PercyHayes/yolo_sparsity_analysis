import numpy as np
import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from VecMulUnit import tree_vec_mul_128_group4, tree_vec_mul_128_group8, tree_vec_mul_128_group2



def do_conv_torch(data, weights, out_channels, in_channels, kernel_size, stride, padding):
    '''
    do convolution with torch, used for reference
    '''
    data = torch.from_numpy(data).unsqueeze(0)
    weights = torch.from_numpy(weights)
    
    result = F.conv2d(data, weights, bias=None, stride=stride, padding=padding)
    return result.numpy()

def compare_result(tpu_result, ref_result):
    assert tpu_result.shape == ref_result.shape, f"result shape mismatch, tpu_result.shape: {tpu_result.shape}, ref_result.shape: {ref_result.shape}"
    diff = np.abs(tpu_result - ref_result)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    relative_diff = diff / (np.abs(ref_result) + 1e-8)
    max_relative_diff = np.max(relative_diff)
    return max_diff, mean_diff, max_relative_diff

def load_data(txt_path, expected_shape):
    '''
    load data from txt file, then reshape to expected shape
    '''
    data = np.loadtxt(txt_path)
    assert data.size == np.prod(expected_shape), "data size mismatch"
    data = data.reshape(expected_shape)
    return data


def tpu_conv(data, weights_vec, o_height, o_width, padding, tree_ver_mul_unit, group_size) -> Tuple[np.ndarray, np.ndarray, int]:
    '''
    do convolution using tree_vec_mul_128_group, and return the result and the sparsity of operations
    '''
    assert padding == 1, "current only support padding = 1"
    padding_height = data.shape[1] + 2 * padding
    padding_width = data.shape[2] + 2 * padding
    in_channel = data.shape[0]
    in_channel_group = in_channel // 128
    padding_data = np.pad(data, ((0,0), (padding,padding), (padding,padding)))

    # prepare data
    padding_data = padding_data.transpose(1, 2, 0) # in_height, in_width, in_channel -> in_height, in_width, in_channel_group
    padding_data = padding_data.reshape(padding_height, padding_width, in_channel_group, 128)

    # prepare weights
    kernel_size = weights_vec.shape[1]
    weights_vec = weights_vec.transpose(1,2,0) # in_channel, kernel_size, kernel_size -> kernel_size, kernel_size, in_channel
    weights_vec = weights_vec.reshape(kernel_size, kernel_size, in_channel_group, 128)
    
    # result
    output = np.zeros((o_height, o_width),dtype=np.int16)
    # skip recordings
    skip_recordings = np.zeros(((np.log2(128)-1) //np.log2(group_size)).astype(int))
    
    total_do_compute_count = 0
    # print(f"kernel_size: {kernel_size}, in_channel_group: {in_channel_group}, group_size: {group_size}, o_height: {o_height}, o_width: {o_width}")
    for k_row in range(kernel_size):
        for k_col in range(kernel_size):
            for in_ch_group in range(in_channel_group):
                tree_ver_mul_unit.update_weights(weights_vec[k_row,k_col,in_ch_group])
                for o_row in range(o_height):
                    for o_col in range(o_width):
                        input = padding_data[o_row+k_row,o_col+k_col,in_ch_group,:]
                        result, skipped_operations = tree_ver_mul_unit.do_compute(input)
                        output[o_row,o_col] += result
                        skipped_operations = np.array(skipped_operations)
                        skip_recordings += skipped_operations
                        total_do_compute_count += 1
    return output, skip_recordings, total_do_compute_count





def profile_sparsity(data, weights, out_channels, in_channels, kernel_size, stride, padding):
    '''
    do convolution using different tree_vec_mul_128_group, and profile the sparsity of operations
    '''
    assert padding == 1, "current only support padding = 1"
    assert stride == 1, "current only support stride = 1"

    ref_result = do_conv_torch(data, weights, out_channels, in_channels, kernel_size, stride, padding)
    ref_result = ref_result[0,0].copy()

    weight_vec = weights[0]

    # group size = 8
    print("=============================group size = 8======================================")
    
    tree_ver_mul_group8 = tree_vec_mul_128_group8()
    tpu_result, skip_recordings, total_do_compute_count = tpu_conv(data, weight_vec, ref_result.shape[0], ref_result.shape[1], padding, tree_ver_mul_group8, 8)
    max_diff, mean_diff, max_relative_diff = compare_result(tpu_result, ref_result)
    print("============compare with reference result=============")
    print(f"group size = 8, max_diff: {max_diff}, mean_diff: {mean_diff}, max_relative_diff: {max_relative_diff}")
    print(f"skip_recordings: {skip_recordings}")
    print(f"total_do_compute_count: {total_do_compute_count}")
    print(f"level1 (128 -> 16)    skipped operation rate: {skip_recordings[0] / total_do_compute_count / 16}")
    print(f"level2 (16 -> 2)      skipped operation rate: {skip_recordings[1] / total_do_compute_count / 2}")
    print("\n")



    print("=============================group size = 4======================================")

    # group size = 4
    tree_ver_mul_unit = tree_vec_mul_128_group4()
    tpu_result, skip_recordings, total_do_compute_count = tpu_conv(data, weight_vec, ref_result.shape[0], ref_result.shape[1], padding, tree_ver_mul_unit, 4)
    max_diff, mean_diff, max_relative_diff = compare_result(tpu_result, ref_result)
    print("============compare with reference result=============")
    print(f"group size = 4, max_diff: {max_diff}, mean_diff: {mean_diff}, max_relative_diff: {max_relative_diff}")
    print(f"skip_recordings: {skip_recordings}")
    print(f"total_do_compute_count: {total_do_compute_count}")
    print(f"level1 (128 -> 32)    skipped operation rate: {skip_recordings[0] / total_do_compute_count / 32}")
    print(f"level2 (32 -> 8)      skipped operation rate: {skip_recordings[1] / total_do_compute_count / 8}")
    print(f"level3 (8 -> 2)       skipped operation rate: {skip_recordings[2] / total_do_compute_count / 2}")
    print("\n")


    print("=============================group size = 2======================================")

    # group size = 2
    tree_ver_mul_unit = tree_vec_mul_128_group2()
    tpu_result, skip_recordings, total_do_compute_count = tpu_conv(data, weight_vec, ref_result.shape[0], ref_result.shape[1], padding, tree_ver_mul_unit, 2)
    max_diff, mean_diff, max_relative_diff = compare_result(tpu_result, ref_result)
    print("============compare with reference result=============")
    print(f"group size = 2, max_diff: {max_diff}, mean_diff: {mean_diff}, max_relative_diff: {max_relative_diff}")
    print(f"skip_recordings: {skip_recordings}")
    print(f"total_do_compute_count: {total_do_compute_count}")
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
    data_path = './extract_data/step13_conv/input_001.txt'
    #'/9950backfile/liguoqi/gsw/HD/SpikeYolo/refs/SpikeYolo_TWN/inference_data_TWN_bn_fused/model/model/4_SNN-Block1/0/step12-13_conv2/step13_conv/input_001.txt'
    #'./extract_data/step13_conv/input_001.txt'
    data = load_data(data_path, (in_channels, height, width))
    

    weight_Path = './extract_data/step13_conv/weights.txt'
    #'/9950backfile/liguoqi/gsw/HD/SpikeYolo/refs/SpikeYolo_TWN/inference_data_TWN_bn_fused/model/model/4_SNN-Block1/0/step12-13_conv2/step13_conv/weights.txt'
    #'./extract_data/step13_conv/weights.txt'
    weights = load_data(weight_Path, (out_channels, in_channels, kernel_size, kernel_size))

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

    profile_sparsity(data, weights, 64, 256, 3, 1, 1)