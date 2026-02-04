import numpy as np
import os
from typing import List, Dict, Tuple
import csv
import pandas as pd
from utils import do_conv_torch, compare_result, load_data, tpu_conv_first_channel, data_sparsity, weights_sparsity
from VecMulUnit import tree_vec_mul_128_group4_adder2


def find_all_conv_layers(base_path: str) -> List[str]:
    """
    查找所有包含weights.txt的卷积层目录
    
    Args:
        base_path: 基础路径
        
    Returns:
        所有卷积层目录路径列表
    """
    conv_dirs = []
    for root, dirs, files in os.walk(base_path):
        if 'weights.txt' in files:
            conv_dirs.append(root)
    return sorted(conv_dirs)


def read_conv_params(conv_dir: str) -> Dict:
    """
    读取卷积层的参数
    
    Args:
        conv_dir: 卷积层目录路径
        
    Returns:
        包含卷积参数的字典
    """
    params = {}
    
    # 读取权重形状
    weights_shape_file = os.path.join(conv_dir, 'weights_shape.txt')
    if os.path.exists(weights_shape_file):
        with open(weights_shape_file, 'r') as f:
            lines = f.readlines()
            params['out_channels'] = int(lines[0].strip())
            params['in_channels'] = int(lines[1].strip())
            params['kernel_h'] = int(lines[2].strip())
            params['kernel_w'] = int(lines[3].strip())
    else:
        # 如果没有weights_shape.txt，尝试从weights.txt推断
        weights_file = os.path.join(conv_dir, 'weights.txt')
        if os.path.exists(weights_file):
            weights = np.loadtxt(weights_file)
            # 需要其他信息来推断形状，这里先返回None
            return None
    
    # 读取kernel_size
    kernel_size_file = os.path.join(conv_dir, 'kernel_size.txt')
    if os.path.exists(kernel_size_file):
        with open(kernel_size_file, 'r') as f:
            params['kernel_size'] = int(f.readline().strip())
    else:
        # 使用weights_shape中的kernel_size（假设kernel_h == kernel_w）
        if 'kernel_h' in params:
            params['kernel_size'] = params['kernel_h']
            # 验证kernel_h和kernel_w是否相同
            if 'kernel_w' in params and params['kernel_h'] != params['kernel_w']:
                print(f"警告: kernel_h ({params['kernel_h']}) != kernel_w ({params['kernel_w']})")
    
    # 读取stride
    stride_file = os.path.join(conv_dir, 'stride.txt')
    if os.path.exists(stride_file):
        with open(stride_file, 'r') as f:
            params['stride'] = int(f.readline().strip())
    else:
        params['stride'] = 1  # 默认值
    
    # 读取padding
    padding_file = os.path.join(conv_dir, 'padding.txt')
    if os.path.exists(padding_file):
        with open(padding_file, 'r') as f:
            params['padding'] = int(f.readline().strip())
    else:
        params['padding'] = 0  # 默认值
    
    # 读取输入形状
    input_shape_file = os.path.join(conv_dir, 'input_shape_001.txt')
    if os.path.exists(input_shape_file):
        with open(input_shape_file, 'r') as f:
            lines = f.readlines()
            params['batch'] = int(lines[0].strip())
            params['input_channels'] = int(lines[1].strip())
            params['input_height'] = int(lines[2].strip())
            params['input_width'] = int(lines[3].strip())
    
    return params


def check_conv_valid(params: Dict) -> bool:
    """
    检查卷积层是否满足条件：输入通道大于128且可以被128整除
    
    Args:
        params: 卷积参数字典
        
    Returns:
        是否满足条件
    """
    if params is None:
        return False
    in_channels = params.get('in_channels', 0)
    return in_channels >= 128 and in_channels % 128 == 0


def process_single_conv_layer(conv_dir: str, params: Dict) -> Dict:
    """
    处理单个卷积层：加载数据，进行计算，比较结果，记录稀疏度
    
    Args:
        conv_dir: 卷积层目录路径
        params: 卷积参数字典
        
    Returns:
        包含结果的字典
    """
    result = {
        'layer_path': conv_dir,
        'in_channels': params['in_channels'],
        'out_channels': params['out_channels'],
        'kernel_size': params['kernel_size'],
        'stride': params['stride'],
        'padding': params['padding'],
    }
    
    try:
        # 加载输入数据
        input_file = os.path.join(conv_dir, 'input_001.txt')
        if not os.path.exists(input_file):
            result['error'] = 'input_001.txt not found'
            return result
        
        input_shape = (params['input_channels'], params['input_height'], params['input_width'])
        data = load_data(input_file, input_shape)
        
        # 加载权重
        weights_file = os.path.join(conv_dir, 'weights.txt')
        weights_shape = (params['out_channels'], params['in_channels'], 
                        params['kernel_size'], params['kernel_size'])
        weights = load_data(weights_file, weights_shape)
        # 映射权重到 -1, 0, 1
        weights = np.where(weights > 0, 1, np.where(weights < 0, -1, 0))
        
        # 使用PyTorch计算参考结果
        ref_result = do_conv_torch(data, weights, params['out_channels'], 
                                  params['in_channels'], params['kernel_size'],
                                  params['stride'], params['padding'])
        #ref_result = ref_result[0, 0].copy()  # 取第一个batch和第一个输出通道
        
        # 使用TPU模块计算（随机选择一个输出通道）
        tree_adder = tree_vec_mul_128_group4_adder2()
        index_channel, tpu_result, skip_recordings, weights_update_time, total_do_compute_count = \
            tpu_conv_first_channel(data, weights, params['in_channels'], 
                                   params['stride'], params['padding'], tree_adder)
        
        # 严格比较结果（必须完全相等）
        if not np.array_equal(tpu_result, ref_result[0,index_channel]):
            result['error'] = 'Results do not match exactly'
            return result
        
        result['match'] = True
        result['index_channel'] = int(index_channel)
        
        # 计算输入稀疏度
        input_sparsity = data_sparsity(data)
        #np.sum(data == 0) / np.prod(data.shape) if data.size > 0 else 0.0
        result['input_sparsity'] = float(input_sparsity)
        
        # 计算权重稀疏度（使用选中的输出通道的权重）
        selected_weights = weights[index_channel]
        weight_sparsity = weights_sparsity(selected_weights)
        result['weight_sparsity'] = float(weight_sparsity)
        
        # 每个level的计算单元数（根据tree_vec_mul_128_group4_adder2的结构）
        # Level 1: 128 -> 32 (32个计算单元)
        # Level 2: 32 -> 16 (16个计算单元)
        # Level 3: 16 -> 8 (8个计算单元)
        # Level 4: 8 -> 4 (4个计算单元)
        # Level 5: 4 -> 2 (2个计算单元)
        # Level 6: 2 -> 1 (1个计算单元)
        level_units = [32, 16, 8, 4, 2, 1]
        
        # 计算各层的稀疏度（跳过次数 / (总计算次数 * 该层的计算单元数)）
        result['level_sparsity'] = []
        result['skip_recordings'] = skip_recordings.tolist()
        result['total_do_compute_count'] = int(total_do_compute_count)
        result['weights_update_time'] = int(weights_update_time)
        
        if total_do_compute_count > 0:
            for i, (skip_count, units) in enumerate(zip(skip_recordings, level_units)):
                # 该level的总计算次数 = total_do_compute_count * units
                level_total_compute = total_do_compute_count * units
                level_sparsity = skip_count / level_total_compute if level_total_compute > 0 else 0.0
                result['level_sparsity'].append(float(level_sparsity))
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def profile_model_sparsity(base_path: str, output_csv: str = None):
    """
    分析模型中所有满足条件的卷积层的稀疏度
    
    Args:
        base_path: 模型数据的基础路径
        output_csv: 输出CSV文件路径（可选）
    """
    print("=" * 80)
    print("开始分析模型稀疏度")
    print("=" * 80)
    
    # 查找所有卷积层
    print(f"\n正在查找所有卷积层目录...")
    conv_dirs = find_all_conv_layers(base_path)
    print(f"找到 {len(conv_dirs)} 个卷积层目录")
    
    # 处理每个卷积层
    results = []
    valid_count = 0
    
    for conv_dir in conv_dirs:
        print(f"\n处理: {conv_dir}")
        
        # 读取参数
        params = read_conv_params(conv_dir)
        if params is None:
            print(f"  跳过: 无法读取参数")
            continue
        
        # 检查是否满足条件
        if not check_conv_valid(params):
            print(f"  跳过: in_channels={params['in_channels']} 不满足条件 (需要 >= 128 且能被128整除)")
            continue
        
        print(f"  满足条件: in_channels={params['in_channels']}, out_channels={params['out_channels']}, "
              f"kernel_size={params['kernel_size']}, stride={params['stride']}, padding={params['padding']}")
        
        # 处理卷积层
        result = process_single_conv_layer(conv_dir, params)
        results.append(result)
        valid_count += 1
        
        if 'error' in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  结果匹配: {result['match']}, 选择的输出通道: {result.get('index_channel', 'N/A')}")
            print(f"  输入稀疏度: {result.get('input_sparsity', 0):.4f}, 权重稀疏度: {result.get('weight_sparsity', 0):.4f}")
            print(f"  总计算次数: {result['total_do_compute_count']}, "
                  f"权重加载次数: {result['weights_update_time']}")
            if 'level_sparsity' in result:
                print(f"  各层稀疏度: {[f'{s:.4f}' for s in result['level_sparsity']]}")
    
    print(f"\n" + "=" * 80)
    print(f"处理完成: 共处理 {valid_count} 个满足条件的卷积层")
    print("=" * 80)
    
    # 生成表格
    if len(results) == 0:
        print("没有满足条件的卷积层")
        return
    
    # Level名称映射
    level_names = ['128->32', '32->16', '16->8', '8->4', '4->2', '2->1']
    level_units = [32, 16, 8, 4, 2, 1]
    
    # 准备表格数据
    table_data = []
    for result in results:
        # 从路径提取layer name（相对于base_path的相对路径）
        layer_path = result['layer_path']
        if layer_path.startswith(base_path):
            layer_name = os.path.relpath(layer_path, base_path)
        else:
            layer_name = layer_path
        
        row = {
            'Layer name': layer_name,
            'In channels': result['in_channels'],
            'kernel_size': result['kernel_size'],
        }
        
        if 'error' not in result:
            row['index_channel'] = result.get('index_channel', '')
            row['input_sparsity'] = result.get('input_sparsity', 0.0)
            row['weight_sparsity'] = result.get('weight_sparsity', 0.0)
            
            skip_recordings = result.get('skip_recordings', [])
            total_do_compute_count = result['total_do_compute_count']
            level_sparsity_list = result.get('level_sparsity', [])
            
            # 添加各层的skip counts和skip rate
            for i, (level_name, units) in enumerate(zip(level_names, level_units)):
                if i < len(skip_recordings):
                    skip_count = int(skip_recordings[i])
                    skip_rate = level_sparsity_list[i] if i < len(level_sparsity_list) else 0.0
                    row[f'Level({level_name}) skip counts'] = skip_count
                    row[f'Level({level_name}) skip rate'] = skip_rate
                else:
                    row[f'Level({level_name}) skip counts'] = 0
                    row[f'Level({level_name}) skip rate'] = 0.0
            
            row['compute count'] = total_do_compute_count
            row['weights_update_time'] = result['weights_update_time']
        else:
            row['Error'] = result['error']
        
        table_data.append(row)
    
    # 添加平均值行
    if len(table_data) > 0 and 'error' not in table_data[0]:
        avg_row = {
            'Layer name': 'average',
            'In channels': '',
            'kernel_size': '',
            'index_channel': '',
            'input_sparsity': '',
            'weight_sparsity': '',
        }
        
        # 计算各level的skip counts总和和跳过率
        total_compute_count = 0
        total_weights_update_time = 0
        input_sparsity_list = []
        weight_sparsity_list = []
        
        # 计算所有层的compute count和weights_update_time的总和，以及收集稀疏度
        for row in table_data:
            if 'Error' not in row:
                total_compute_count += row.get('compute count', 0)
                total_weights_update_time += row.get('weights_update_time', 0)
                if 'input_sparsity' in row:
                    input_sparsity_list.append(row['input_sparsity'])
                if 'weight_sparsity' in row:
                    weight_sparsity_list.append(row['weight_sparsity'])
        
        # 计算平均稀疏度
        if input_sparsity_list:
            avg_row['input_sparsity'] = np.mean(input_sparsity_list)
        if weight_sparsity_list:
            avg_row['weight_sparsity'] = np.mean(weight_sparsity_list)
        
        for level_name, units in zip(level_names, level_units):
            skip_counts_sum = 0
            
            # 计算所有层的skip counts总和
            for row in table_data:
                if 'Error' not in row:
                    skip_counts_key = f'Level({level_name}) skip counts'
                    if skip_counts_key in row:
                        skip_counts_sum += row[skip_counts_key]
            
            avg_row[f'Level({level_name}) skip counts'] = skip_counts_sum
            
            # 计算跳过率 = skip_counts_sum / (total_compute_count * units)
            if total_compute_count > 0:
                level_total_compute = total_compute_count * units
                avg_row[f'Level({level_name}) skip rate'] = skip_counts_sum / level_total_compute if level_total_compute > 0 else 0.0
            else:
                avg_row[f'Level({level_name}) skip rate'] = 0.0
        
        avg_row['compute count'] = total_compute_count
        avg_row['weights_update_time'] = total_weights_update_time
        
        table_data.append(avg_row)
    
    # 计算平均稀疏度（排除average行）
    if len(table_data) > 0:
        data_rows = [row for row in table_data if row.get('Layer name') != 'average']
        if data_rows:
            level_names = ['128->32', '32->16', '16->8', '8->4', '4->2', '2->1']
            print("\n各层平均稀疏度:")
            for level_name in level_names:
                skip_rate_key = f'Level({level_name}) skip rate'
                skip_rates = [row.get(skip_rate_key, 0) for row in data_rows if skip_rate_key in row]
                if skip_rates:
                    avg_level_sparsity = np.mean(skip_rates)
                    print(f"  {skip_rate_key}: {avg_level_sparsity:.4f}")
    
    # 显示表格
    print("\n" + "=" * 80)
    print("结果表格:")
    print("=" * 80)
    
    df = pd.DataFrame(table_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.to_string(index=False))
    
    # 保存到CSV（按照指定顺序）
    if output_csv and len(table_data) > 0:
        # 定义列的顺序
        level_names = ['128->32', '32->16', '16->8', '8->4', '4->2', '2->1']
        columns = ['Layer name', 'In channels', 'kernel_size', 'index_channel', 'input_sparsity', 'weight_sparsity']
        
        # 添加各level的skip counts和skip rate
        for level_name in level_names:
            columns.append(f'Level({level_name}) skip counts')
            columns.append(f'Level({level_name}) skip rate')
        
        # 添加compute count和weights_update_time
        columns.extend(['compute count', 'weights_update_time'])
        
        # 确保所有列都存在（对于有Error的行）
        all_columns = set()
        for row in table_data:
            all_columns.update(row.keys())
        
        # 添加Error列（如果存在）
        if 'Error' in all_columns:
            columns.append('Error')
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in table_data:
                # 确保所有列都有值
                complete_row = {col: row.get(col, '') for col in columns}
                writer.writerow(complete_row)
        print(f"\n结果已保存到: {output_csv}")
    
    return table_data


if __name__ == "__main__":
    # 设置模型数据路径
    base_path = '/9950backfile/liguoqi/gsw/HD/SpikeYolo/refs/inference_data_TWN_bn_fused/model/model'
    output_csv = './sparsity_profile_results.csv'
    
    profile_model_sparsity(base_path, output_csv)
    
    