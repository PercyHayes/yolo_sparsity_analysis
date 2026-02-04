import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def weights_sparsity(weights):
    '''
    calculate the sparsity of weights
    '''
    return np.sum(weights == 0) / np.prod(weights.shape)

def data_sparsity(data):
    '''
    calculate the sparsity of data
    '''
    return np.sum(data == 0) / np.prod(data.shape)

def do_conv_torch(data, weights, out_channels, in_channels, kernel_size, stride, padding):
    '''
    do convolution with torch, used for reference
    '''
    data = torch.from_numpy(data).unsqueeze(0).float()
    weights = torch.from_numpy(weights).float()
    
    result = F.conv2d(data, weights, bias=None, stride=stride, padding=padding)
    return result.numpy()

def compare_result(tpu_result, ref_result):
    assert tpu_result.shape == ref_result.shape, f"result shape mismatch, tpu_result.shape: {tpu_result.shape}, ref_result.shape: {ref_result.shape}"
    #print(f"tpu_result: {tpu_result[0:4,0:4]}")
    #print(f"ref_result: {ref_result[0:4,0:4]}")
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

def tpu_conv_first_channel(data, weights, in_channels, stride, padding, tree_adder,index_channel=None):
    '''
    do convolution using tree_vec_mul_128_group4_adder2, and return the result and the sparsity of operations
    '''
    assert in_channels >= 128, f"in_channels must be greater than or equal to 128, but got {in_channels}"
    padding_height = data.shape[1] + 2 * padding
    padding_width = data.shape[2] + 2 * padding
    in_channel = data.shape[0]
    in_channel_group = in_channel // 128
    

    # prepare data
    padding_data = np.pad(data, ((0,0), (padding,padding), (padding,padding)))
    padding_data = padding_data.transpose(1, 2, 0) # in_height, in_width, in_channel -> in_height, in_width, in_channel_group
    padding_data = padding_data.reshape(padding_height, padding_width, in_channel_group, 128)
    
    # prepare weights
    # randomly choose one channel of weights
    if index_channel is None:
        index_channel = np.random.randint(0, weights.shape[0])
    
    weights_vec = weights[index_channel]
    kernel_size = weights_vec.shape[1]
    weights_vec = weights_vec.transpose(1,2,0) # in_channel, kernel_size, kernel_size -> kernel_size, kernel_size, in_channel
    weights_vec = weights_vec.reshape(kernel_size, kernel_size, in_channel_group, 128)

    # calculate output shape
    out_height = (padding_height - kernel_size) // stride + 1
    out_width = (padding_width - kernel_size) // stride + 1

    output = np.zeros((out_height, out_width),dtype=np.int16)
    skip_recordings = np.zeros(tree_adder.skip_recordings_len)
    
    weights_update_time = 0
    do_compute_time = 0
    for k_row in range(kernel_size):
        for k_col in range(kernel_size):
            for in_ch_group in range(in_channel_group):
                tree_adder.update_weights(weights_vec[k_row,k_col,in_ch_group])
                weights_update_time += 1
                for o_row in range(out_height):
                    for o_col in range(out_width):
                        input = padding_data[o_row * stride + k_row,o_col * stride + k_col,in_ch_group,:]
                        result, skipped_operations = tree_adder.do_compute(input)
                        output[o_row,o_col] += result
                        do_compute_time += 1
                        skip_recordings += np.array(skipped_operations)

    return index_channel, output, skip_recordings, weights_update_time, do_compute_time