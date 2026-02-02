import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class naive_vec_mul_128:
    def __init__(self):
        self.weights = np.zeros((128,1),dtype=np.int16)

    def update_weights(self, weights):
        self.weights = weights.clip(-1,1)

    def do_compute(self, input):
        skipped_operations = []
        result = np.sum(self.weights * input)
        return result, skipped_operations


class tree_vec_mul_128_group4:
    def __init__(self):
        self.group_size = 4
        self.weights = np.zeros((128 // self.group_size, self.group_size),dtype=np.int16)
        

    def update_weights(self, weights):
        self.weights = weights.clip(-1,1)
        self.weights = self.weights.reshape(-1, self.group_size)
        
    def do_compute(self, input):
        assert input.shape[0] == self.weights.shape[0] * self.group_size, "input shape mismatch"
        skipped_operations = []

        level1_input = input.reshape(-1, self.group_size) # [32, 4]
        zero_count = np.sum(level1_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level1_sum = np.sum(self.weights * level1_input, axis=1)

        level2_input = level1_sum.reshape(-1, self.group_size) # [8, 4]
        zero_count = np.sum(level2_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level2_sum = np.sum(level2_input,axis=1) # [8]
        

        level3_input = level2_sum.reshape(-1, self.group_size) # [2, 4]
        zero_count = np.sum(level3_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level3_sum = np.sum(level3_input,axis=1) # [2]

        level4 = np.sum(level3_sum)
        return level4, skipped_operations

            

      

def load_data(txt_path, expected_shape):
    '''
    load data from txt file, then reshape to expected shape
    '''
    data = np.loadtxt(txt_path)
    assert data.size == np.prod(expected_shape), "data size mismatch"
    data = data.reshape(expected_shape)
    return data

def do_conv_torch(data, weights, out_channels, in_channels, kernel_size, stride, padding):
    data = torch.from_numpy(data).unsqueeze(0)
    weights = torch.from_numpy(weights)
    
    result = F.conv2d(data, weights, bias=None, stride=stride, padding=padding)
    return result.numpy()

def tpu_conv(data, weights_vec):
    padding_data = np.pad(data, ((0,0), (1,1), (1,1)))
    padding_data = padding_data.transpose(1, 2, 0)
    padding_data = padding_data.reshape(98,162,2,128)
    
    weights_vec = weights_vec.transpose(1,2,0) # 256, 3,3 -> 3,3,256
    weights_vec = weights_vec.reshape(3,3,2,128)

    tree_vec_mul = tree_vec_mul_128_group4()

    output = np.zeros((96,160),dtype=np.int16)
    skipped_operation_count = np.zeros(3)
    total_do_compute_count = 0
    for k_row in range(3):
        for k_col in range(3):
            for in_channel_group in range(2):
                tree_vec_mul.update_weights(weights_vec[k_row,k_col,in_channel_group])
                for o_row in range(96):
                    for o_col in range(160):
                        input = padding_data[o_row+k_row,o_col+k_col,in_channel_group,:]
                        result, skipped_operations = tree_vec_mul.do_compute(input)
                        output[o_row,o_col] += result
                        skipped_operations = np.array(skipped_operations)
                        skipped_operation_count += skipped_operations
                        total_do_compute_count += 1
    return output, skipped_operation_count, total_do_compute_count


def compare_result(tpu_result, ref_result):
    assert tpu_result.shape == ref_result.shape, "result shape mismatch"
    diff = np.abs(tpu_result - ref_result)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    relative_diff = diff / (np.abs(ref_result) + 1e-8)
    max_relative_diff = np.max(relative_diff)
    return max_diff, mean_diff, max_relative_diff, relative_diff

if __name__ == "__main__":
    data_path = '/9950backfile/liguoqi/gsw/HD/SpikeYolo/refs/SpikeYolo_TWN/inference_data_TWN_bn_fused/model/model/2_SNN-Block-1/step12-13_conv2/step13_conv/input_001.txt'
    data = load_data(data_path, (256,96,160))

    weight_Path = '/9950backfile/liguoqi/gsw/HD/SpikeYolo/refs/SpikeYolo_TWN/inference_data_TWN_bn_fused/model/model/2_SNN-Block-1/step12-13_conv2/step13_conv/weights.txt'
    weights = load_data(weight_Path, (64,256,3,3))

    torch_result = do_conv_torch(data, weights, 64, 256, 3, 1, 1)
    ref_result = torch_result[0,0].copy()

    weight_vec = weights[0]

    tpu_result, skipped_operation_count, total_do_compute_count = tpu_conv(data, weight_vec)

    #compare result
    print("===================================================================")
    max_diff, mean_diff, max_relative_diff, relative_diff = compare_result(tpu_result, ref_result)
    print(f"max_diff: {max_diff}, mean_diff: {mean_diff}, max_relative_diff: {max_relative_diff}")
    print(f"relative_diff: {relative_diff}")
    print("===================================================================")
    print(f"skipped_operation_count: {skipped_operation_count}")
    print(f"total_do_compute_count: {total_do_compute_count}")
    print("===================================================================")
    print(f"level1_skipped_operation_rate: {skipped_operation_count[0] / total_do_compute_count / 32}")
    print(f"level2_skipped_operation_rate: {skipped_operation_count[1] / total_do_compute_count / 8}")
    print(f"level3_skipped_operation_rate: {skipped_operation_count[2] / total_do_compute_count / 2}")
    print("===================================================================")
    

    
    

    


    

    



    
    
    