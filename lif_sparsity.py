import numpy as np
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple

def load_lifoutputs(txt_path):
    '''
    load integer outputs (0, 1, 2, 3) from txt file, then return parameter num and sparsity
    '''
    txt_file = os.path.join(txt_path, "output.txt")
    shape_file = os.path.join(txt_path, "output_shape.txt")
    
    # 加载权重
    outputs = np.loadtxt(txt_file)
    
    # 尝试加载形状文件（如果存在）
    shape = None
    if os.path.exists(shape_file):
        shape = np.loadtxt(shape_file)
        if shape.ndim == 0:  # 如果是标量，转换为数组
            shape = np.array([int(shape)])
        else:
            shape = shape.astype(int)
        expected_num = np.prod(shape)
        num_parameters = outputs.size
        if num_parameters != expected_num:
            print(f"Warning: num_parameters ({num_parameters}) != expected_num ({expected_num}) for {txt_path}")
    else:
        num_parameters = outputs.size
    
    # 统计各种权重的数量
    zero_count = np.sum(outputs == 0)
    one_count = np.sum(outputs == 1)
    two_count = np.sum(outputs == 2)
    three_count = np.sum(outputs == 3)
    
    # 计算占比
    sparsity = zero_count / num_parameters if num_parameters > 0 else 0.0
    nonzero_count = one_count + two_count + three_count
    nonzero_ratio = nonzero_count / num_parameters if num_parameters > 0 else 0.0
    one_ratio = one_count / num_parameters if num_parameters > 0 else 0.0
    two_ratio = two_count / num_parameters if num_parameters > 0 else 0.0
    three_ratio = three_count / num_parameters if num_parameters > 0 else 0.0
    
    return {
        'num_parameters': num_parameters,
        'sparsity': sparsity,
        'zero_count': zero_count,
        'one_count': one_count,
        'two_count': two_count,
        'three_count': three_count,
        'nonzero_count': nonzero_count,
        'zero_ratio': sparsity,
        'nonzero_ratio': nonzero_ratio,
        'one_ratio': one_ratio,
        'two_ratio': two_ratio,
        'three_ratio': three_ratio,
        'shape': shape
    }

def find_all_lif_dirs(root_path: str) -> List[Tuple[str, str]]:
    """
    递归查找指定路径下所有包含lif的目录（这些目录应该有output.txt和output_shape.txt）
    
    Args:
        root_path: 根目录路径
    
    Returns:
        List of (layer_path, layer_name) tuples
    """
    lif_dirs = []
    root_path = Path(root_path)
    
    # 递归查找所有包含"lif"的目录
    for lif_dir in root_path.rglob("*lif*"):
        if lif_dir.is_dir():
            # 检查目录中是否有output.txt和output_shape.txt
            output_file = lif_dir / "output.txt"
            output_shape_file = lif_dir / "output_shape.txt"
            
            if output_file.exists() and output_shape_file.exists():
                layer_path = str(lif_dir)
                # 生成层名称（相对于root_path的路径）
                try:
                    layer_name = str(lif_dir.relative_to(root_path))
                except ValueError:
                    # 如果无法计算相对路径，使用绝对路径
                    layer_name = layer_path
                
                lif_dirs.append((layer_path, layer_name))
    
    return lif_dirs


def analyze_all_lif_layers(root_path: str, output_csv: str = None) -> List[Dict]:
    """
    分析指定路径下所有LIF层的激活稀疏度
    
    Args:
        root_path: 根目录路径
        output_csv: 输出CSV文件路径，如果为None则不保存
    
    Returns:
        List of dictionaries containing statistics for each layer
    """
    # 查找所有LIF目录
    lif_dirs = find_all_lif_dirs(root_path)
    print(f"找到 {len(lif_dirs)} 个LIF层")
    
    results = []
    
    # 分析每个层
    for layer_path, layer_name in lif_dirs:
        try:
            stats = load_lifoutputs(layer_path)
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
        'two_count',
        'three_count',
        'nonzero_count',
        'zero_ratio',
        'nonzero_ratio',
        'one_ratio',
        'two_ratio',
        'three_ratio',
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
    output_csv = "/9950backfile/liguoqi/gsw/HD/SpikeYolo/hardware_sim/lif_sparsity_stats.csv"
    
    print("="*80)
    print("LIF神经网络激活稀疏度统计分析")
    print("="*80)
    print(f"搜索路径: {root_path}")
    print(f"输出文件: {output_csv}")
    print("="*80)
    
    # 分析所有LIF层
    results = analyze_all_lif_layers(root_path, output_csv)
    
    # 打印汇总信息
    if results:
        print("\n" + "="*80)
        print("汇总统计")
        print("="*80)
        total_params = sum(r['num_parameters'] for r in results)
        total_zeros = sum(r['zero_count'] for r in results)
        total_ones = sum(r['one_count'] for r in results)
        total_twos = sum(r['two_count'] for r in results)
        total_threes = sum(r['three_count'] for r in results)
        total_nonzeros = sum(r['nonzero_count'] for r in results)
        
        print(f"总LIF层数: {len(results)}")
        print(f"总激活值数量: {total_params:,}")
        print(f"总零值数量: {total_zeros:,} ({total_zeros/total_params*100:.2f}%)")
        print(f"总非零值数量: {total_nonzeros:,} ({total_nonzeros/total_params*100:.2f}%)")
        print(f"  其中 1: {total_ones:,} ({total_ones/total_params*100:.2f}%)")
        print(f"  其中 2: {total_twos:,} ({total_twos/total_params*100:.2f}%)")
        print(f"  其中 3: {total_threes:,} ({total_threes/total_params*100:.2f}%)")
        print("="*80)
