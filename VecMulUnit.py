import numpy as np

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
    '''
    128 -> 32 -> 8 -> 2 -> 1
    '''
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

class tree_vec_mul_128_group8:
    '''
    128 -> 16 -> 2 -> 1
    '''
    def __init__(self):
        self.group_size = 8
        self.weights = np.zeros((128 // self.group_size, self.group_size),dtype=np.int16)
        
    def update_weights(self, weights):
        self.weights = weights.clip(-1,1)
        self.weights = self.weights.reshape(-1, self.group_size)
        
    def do_compute(self, input):
        assert input.shape[0] == self.weights.shape[0] * self.group_size, "input shape mismatch"
        skipped_operations = []

        level1_input = input.reshape(-1, self.group_size) # [16, 8]
        zero_count = np.sum(level1_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level1_sum = np.sum(self.weights * level1_input, axis=1)
        
        level2_input = level1_sum.reshape(-1, self.group_size) # [2, 8]
        zero_count = np.sum(level2_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level2_sum = np.sum(level2_input,axis=1) # [2]
        
        level3_sum = np.sum(level2_sum)
        return level3_sum, skipped_operations

class tree_vec_mul_128_group2:
    '''
    128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    '''
    def __init__(self):
        self.group_size = 2
        self.weights = np.zeros((128 // self.group_size, self.group_size),dtype=np.int16)
        
    def update_weights(self, weights):
        self.weights = weights.clip(-1,1)
        self.weights = self.weights.reshape(-1, self.group_size)

    def do_compute(self, input):
        assert input.shape[0] == self.weights.shape[0] * self.group_size, "input shape mismatch"
        skipped_operations = []

        level1_input = input.reshape(-1, self.group_size) # [64, 2]
        zero_count = np.sum(level1_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level1_sum = np.sum(self.weights * level1_input, axis=1)
        
        level2_input = level1_sum.reshape(-1, self.group_size) # [32, 2]
        zero_count = np.sum(level2_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level2_sum = np.sum(level2_input,axis=1) # [32]
        
        level3_input = level2_sum.reshape(-1, self.group_size) # [16, 2]
        zero_count = np.sum(level3_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level3_sum = np.sum(level3_input,axis=1) # [16]
        
        level4_input = level3_sum.reshape(-1, self.group_size) # [8, 2]
        zero_count = np.sum(level4_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level4_sum = np.sum(level4_input,axis=1) # [8]
        
        level5_input = level4_sum.reshape(-1, self.group_size) # [4, 2]
        zero_count = np.sum(level5_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level5_sum = np.sum(level5_input,axis=1) # [4]
        
        level6_input = level5_sum.reshape(-1, self.group_size) # [2, 2]
        zero_count = np.sum(level6_input==0,axis=1)
        skipped_operations.append(np.sum(zero_count==self.group_size))
        level6_sum = np.sum(level6_input,axis=1) # [2]
        
        level7_sum = np.sum(level6_sum)
        return level7_sum, skipped_operations