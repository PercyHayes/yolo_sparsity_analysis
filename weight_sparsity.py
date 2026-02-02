import numpy as np
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple


def load_weights(txt_path):
    '''
    load ternary weights from txt file, then return parameter num and sparsity
    '''
    txt_file = os.path.join(txt_path, "weights.txt")
    shape_file = os.path.join(txt_path, "weights_shape.txt")
    
    # 加载权重
    weights = np.loadtxt(txt_file)
    
    # 尝试加载形状文件（如果存在）
    shape = None
    if os.path.exists(shape_file):
        shape = np.loadtxt(shape_file)
        if shape.ndim == 0:  # 如果是标量，转换为数组
            shape = np.array([int(shape)])
        else:
            shape = shape.astype(int)
        expected_num = np.prod(shape)
        num_parameters = weights.size
        if num_parameters != expected_num:
            print(f"Warning: num_parameters ({num_parameters}) != expected_num ({expected_num}) for {txt_path}")
    else:
        num_parameters = weights.size
    
    # 统计各种权重的数量
    zero_count = np.sum(weights == 0)
    one_count = np.sum(weights == 1)
    negative_one_count = np.sum(weights == -1)
    
    # 计算占比
    sparsity = zero_count / num_parameters if num_parameters > 0 else 0.0
    nonzero_count = one_count + negative_one_count
    nonzero_ratio = nonzero_count / num_parameters if num_parameters > 0 else 0.0
    one_ratio = one_count / num_parameters if num_parameters > 0 else 0.0
    negative_one_ratio = negative_one_count / num_parameters if num_parameters > 0 else 0.0
    
    return {
        'num_parameters': num_parameters,
        'sparsity': sparsity,
        'zero_count': zero_count,
        'one_count': one_count,
        'negative_one_count': negative_one_count,
        'nonzero_count': nonzero_count,
        'zero_ratio': sparsity,
        'nonzero_ratio': nonzero_ratio,
        'one_ratio': one_ratio,
        'negative_one_ratio': negative_one_ratio,
        'shape': shape
    }


def find_all_weights_files(root_path: str) -> List[Tuple[str, str]]:
    """
    递归查找指定路径下所有的weights.txt文件
    
    Args:
        root_path: 根目录路径
    
    Returns:
        List of (layer_path, layer_name) tuples
    """
    weights_files = []
    root_path = Path(root_path)
    
    # 递归查找所有weights.txt文件
    for weights_file in root_path.rglob("weights.txt"):
        # 获取包含weights.txt的目录路径
        layer_path = str(weights_file.parent)
        # 生成层名称（相对于root_path的路径）
        try:
            layer_name = str(weights_file.parent.relative_to(root_path))
        except ValueError:
            # 如果无法计算相对路径，使用绝对路径
            layer_name = layer_path
        
        weights_files.append((layer_path, layer_name))
    
    return weights_files


def analyze_all_layers(root_path: str, output_csv: str = None) -> List[Dict]:
    """
    分析指定路径下所有层的权重稀疏度
    
    Args:
        root_path: 根目录路径
        output_csv: 输出CSV文件路径，如果为None则不保存
    
    Returns:
        List of dictionaries containing statistics for each layer
    """
    # 查找所有权重文件
    weights_files = find_all_weights_files(root_path)
    print(f"找到 {len(weights_files)} 个权重文件")
    
    results = []
    
    # 分析每个层
    for layer_path, layer_name in weights_files:
        try:
            stats = load_weights(layer_path)
            #stats['layer_path'] = layer_path
            stats['layer_name'] = layer_name
            results.append(stats)
            print(f"处理完成: {layer_name}")
        except Exception as e:
            print(f"处理失败 {layer_name}: {e}")
            continue
    
    # 保存为CSV
    if output_csv and results:
        save_to_csv(results, output_csv)
        print(f"\n结果已保存到: {output_csv}")
    
    return results


def save_to_csv(results: List[Dict], output_csv: str):
    """
    将统计结果保存为CSV文件
    
    Args:
        results: 统计结果列表
        output_csv: 输出CSV文件路径
    """
    if not results:
        print("没有数据可保存")
        return
    
    # 定义CSV列
    fieldnames = [
        'layer_name',
        'num_parameters',
        'zero_count',
        'one_count',
        'negative_one_count',
        'nonzero_count',
        'zero_ratio',
        'nonzero_ratio',
        'one_ratio',
        'negative_one_ratio',
        'sparsity'
    ]
    
    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # 只写入CSV需要的字段
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)


if __name__ == "__main__":
    # 指定路径
    root_path = "/9950backfile/liguoqi/gsw/HD/SpikeYolo/refs/SpikeYolo_TWN/inference_data_TWN_bn_fused/model/model"
    output_csv = "/9950backfile/liguoqi/gsw/HD/SpikeYolo/hardware_sim/weight_sparsity_stats.csv"
    
    print("="*80)
    print("权重稀疏度统计分析")
    print("="*80)
    print(f"搜索路径: {root_path}")
    print(f"输出文件: {output_csv}")
    print("="*80)
    
    # 分析所有层
    results = analyze_all_layers(root_path, output_csv)
    
    # 打印汇总信息
    if results:
        print("\n" + "="*80)
        print("汇总统计")
        print("="*80)
        total_params = sum(r['num_parameters'] for r in results)
        total_zeros = sum(r['zero_count'] for r in results)
        total_ones = sum(r['one_count'] for r in results)
        total_neg_ones = sum(r['negative_one_count'] for r in results)
        
        print(f"总层数: {len(results)}")
        print(f"总参数量: {total_params:,}")
        print(f"总零值数量: {total_zeros:,} ({total_zeros/total_params*100:.2f}%)")
        print(f"总非零值数量: {total_ones + total_neg_ones:,} ({(total_ones + total_neg_ones)/total_params*100:.2f}%)")
        print(f"  其中 +1: {total_ones:,} ({total_ones/total_params*100:.2f}%)")
        print(f"  其中 -1: {total_neg_ones:,} ({total_neg_ones/total_params*100:.2f}%)")
        print("="*80)