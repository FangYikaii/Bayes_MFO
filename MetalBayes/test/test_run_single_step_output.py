#!/usr/bin/env python3
"""
测试 run_single_step 的输出结果

展示 run_single_step 返回的字典结构和内容，并将结果保存到文件
"""

import os
import json
import pytest
import torch
import numpy as np
from datetime import datetime

from conftest import TraceAwareKGOptimizer, OptimizerManager


def test_run_single_step_output_structure(optimizer_and_param_space, temp_dirs):
    """测试 run_single_step 返回结果的结构和内容，并保存到文件"""
    optimizer, param_space = optimizer_and_param_space
    
    # 创建输出文件路径（保存到测试目录，方便查看）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"run_single_step_output_{timestamp}.txt")
    json_file = os.path.join(output_dir, f"run_single_step_results_{timestamp}.json")
    
    # 用于保存 JSON 格式的结果
    results_data = {
        'test_timestamp': timestamp,
        'phase': optimizer.phase,
        'param_names': optimizer.param_names,
        'iterations': []
    }
    
    # 打开文件用于写入文本输出
    with open(output_file, 'w', encoding='utf-8') as f:
        def write_output(text):
            """同时写入文件和打印到控制台"""
            print(text)
            f.write(text + '\n')
        
        # 第一次迭代（初始样本）
        result1 = optimizer.run_single_step(simulation_flag=True)
        
        write_output("\n" + "="*60)
        write_output("第一次迭代（初始样本）返回结果：")
        write_output("="*60)
        write_output(f"返回类型: {type(result1)}")
        write_output(f"返回键: {list(result1.keys())}")
        write_output("")
        
        # 详细展示每个字段
        write_output("1. iteration (迭代次数):")
        write_output(f"   类型: {type(result1['iteration'])}")
        write_output(f"   值: {result1['iteration']}")
        write_output(f"   说明: 第一次迭代返回 0")
        write_output("")
        
        write_output("2. candidates (候选样本):")
        write_output(f"   类型: {type(result1['candidates'])}")
        write_output(f"   形状: {result1['candidates'].shape}")
        write_output(f"   说明: 包含 {result1['candidates'].shape[0]} 个候选参数组合，每个有 {result1['candidates'].shape[1]} 个参数")
        write_output(f"   所有样本:")
        candidates1_np = result1['candidates'].cpu().numpy()
        for i in range(result1['candidates'].shape[0]):
            write_output(f"     样本 {i}: {candidates1_np[i]}")
        write_output("")
        
        write_output("3. hypervolume (超体积):")
        write_output(f"   类型: {type(result1['hypervolume'])}")
        write_output(f"   值: {result1['hypervolume']:.6f}")
        write_output(f"   说明: 当前帕累托前沿的超体积值，用于评估优化性能")
        write_output("")
        
        # 保存第一次迭代结果到 JSON
        results_data['iterations'].append({
            'iteration': int(result1['iteration']),
            'candidates': candidates1_np.tolist(),
            'hypervolume': float(result1['hypervolume']),
            'sample_count': int(optimizer.X.shape[0])
        })
        
        # 第二次迭代（贝叶斯优化生成的候选点）
        result2 = optimizer.run_single_step(simulation_flag=True)
        
        write_output("\n" + "="*60)
        write_output("第二次迭代（贝叶斯优化）返回结果：")
        write_output("="*60)
        write_output(f"返回类型: {type(result2)}")
        write_output(f"返回键: {list(result2.keys())}")
        write_output("")
        
        write_output("1. iteration (迭代次数):")
        write_output(f"   类型: {type(result2['iteration'])}")
        write_output(f"   值: {result2['iteration']}")
        write_output(f"   说明: 第二次迭代返回 1（递增）")
        write_output("")
        
        write_output("2. candidates (候选样本):")
        write_output(f"   类型: {type(result2['candidates'])}")
        write_output(f"   形状: {result2['candidates'].shape}")
        write_output(f"   说明: 包含 {result2['candidates'].shape[0]} 个候选参数组合（batch_size），每个有 {result2['candidates'].shape[1]} 个参数")
        write_output(f"   所有样本:")
        candidates2_np = result2['candidates'].cpu().numpy()
        for i in range(result2['candidates'].shape[0]):
            write_output(f"     样本 {i}: {candidates2_np[i]}")
        write_output("")
        
        write_output("3. hypervolume (超体积):")
        write_output(f"   类型: {type(result2['hypervolume'])}")
        write_output(f"   值: {result2['hypervolume']:.6f}")
        write_output(f"   说明: 更新后的超体积值（应该 >= 第一次迭代的值）")
        write_output("")
        
        # 保存第二次迭代结果到 JSON
        results_data['iterations'].append({
            'iteration': int(result2['iteration']),
            'candidates': candidates2_np.tolist(),
            'hypervolume': float(result2['hypervolume']),
            'sample_count': int(optimizer.X.shape[0])
        })
        
        # 对比两次迭代
        write_output("\n" + "="*60)
        write_output("迭代对比：")
        write_output("="*60)
        write_output(f"迭代次数: {result1['iteration']} -> {result2['iteration']}")
        write_output(f"候选样本数: {result1['candidates'].shape[0]} -> {result2['candidates'].shape[0]}")
        write_output(f"超体积变化: {result1['hypervolume']:.6f} -> {result2['hypervolume']:.6f}")
        write_output(f"超体积变化量: {result2['hypervolume'] - result1['hypervolume']:.6f}")
        write_output("")
        
        # 保存对比信息
        results_data['comparison'] = {
            'iteration_change': f"{result1['iteration']} -> {result2['iteration']}",
            'candidate_count_change': f"{result1['candidates'].shape[0]} -> {result2['candidates'].shape[0]}",
            'hypervolume_change': f"{result1['hypervolume']:.6f} -> {result2['hypervolume']:.6f}",
            'hypervolume_delta': float(result2['hypervolume'] - result1['hypervolume'])
        }
        
        write_output("✓ 所有验证通过！")
        write_output(f"\n结果已保存到:")
        write_output(f"  文本文件: {output_file}")
        write_output(f"  JSON文件: {json_file}")
    
    # 保存 JSON 格式的结果
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 验证返回值的结构
    assert isinstance(result1, dict), "返回值应该是字典"
    assert 'iteration' in result1, "应包含 'iteration' 键"
    assert 'candidates' in result1, "应包含 'candidates' 键"
    assert 'hypervolume' in result1, "应包含 'hypervolume' 键"
    
    assert isinstance(result1['iteration'], int), "iteration 应该是整数"
    assert isinstance(result1['candidates'], torch.Tensor), "candidates 应该是 torch.Tensor"
    assert isinstance(result1['hypervolume'], (float, int)), "hypervolume 应该是数值"
    
    # 验证迭代次数递增
    assert result2['iteration'] == result1['iteration'] + 1, "迭代次数应该递增"
    
    # 验证候选样本数量
    assert result1['candidates'].shape[0] == optimizer.n_init, "第一次迭代应该返回 n_init 个样本"
    assert result2['candidates'].shape[0] == optimizer.batch_size, "后续迭代应该返回 batch_size 个样本"
    
    print(f"\n✓ 结果已保存到文件:")
    print(f"  文本: {output_file}")
    print(f"  JSON: {json_file}")


def test_multiple_iterations_output(optimizer_and_param_space, temp_dirs):
    """测试多次迭代的输出结果，并保存到文件"""
    optimizer, param_space = optimizer_and_param_space
    
    # 创建输出文件路径（保存到测试目录，方便查看）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"multiple_iterations_output_{timestamp}.txt")
    json_file = os.path.join(output_dir, f"multiple_iterations_results_{timestamp}.json")
    
    results = []
    results_data = {
        'test_timestamp': timestamp,
        'phase': optimizer.phase,
        'param_names': optimizer.param_names,
        'iterations': []
    }
    
    # 打开文件用于写入
    with open(output_file, 'w', encoding='utf-8') as f:
        def write_output(text):
            print(text)
            f.write(text + '\n')
        
        write_output("="*60)
        write_output("多次迭代测试结果")
        write_output("="*60)
        write_output(f"测试时间: {timestamp}")
        write_output(f"阶段: {optimizer.phase}")
        write_output(f"参数名称: {optimizer.param_names}")
        write_output("")
        
        # 运行5次迭代
        for i in range(5):
            result = optimizer.run_single_step(simulation_flag=True)
            results.append(result)
            
            candidates_np = result['candidates'].cpu().numpy()
            
            write_output(f"\n迭代 {i+1} (iteration={result['iteration']}):")
            write_output(f"  candidates shape: {result['candidates'].shape}")
            write_output(f"  hypervolume: {result['hypervolume']:.6f}")
            write_output(f"  total samples in X: {optimizer.X.shape[0]}")
            write_output(f"  total samples in Y: {optimizer.Y.shape[0]}")
            write_output(f"  候选样本:")
            for j, candidate in enumerate(candidates_np):
                write_output(f"    样本 {j}: {candidate}")
            
            # 保存到 JSON
            results_data['iterations'].append({
                'iteration': int(result['iteration']),
                'candidates': candidates_np.tolist(),
                'hypervolume': float(result['hypervolume']),
                'sample_count_x': int(optimizer.X.shape[0]),
                'sample_count_y': int(optimizer.Y.shape[0])
            })
        
        write_output("\n" + "="*60)
        write_output("总结:")
        write_output("="*60)
        write_output(f"总迭代次数: {len(results)}")
        write_output(f"最终样本数: {optimizer.X.shape[0]}")
        write_output(f"最终超体积: {results[-1]['hypervolume']:.6f}")
        write_output(f"\n结果已保存到:")
        write_output(f"  文本文件: {output_file}")
        write_output(f"  JSON文件: {json_file}")
    
    # 保存 JSON 格式的结果
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 验证迭代次数递增
    for i in range(len(results)):
        assert results[i]['iteration'] == i, f"迭代 {i} 应该返回 iteration={i}"
    
    # 验证超体积非递减
    for i in range(1, len(results)):
        assert results[i]['hypervolume'] >= results[i-1]['hypervolume'] - 1e-6, \
            f"超体积不应该减少: {results[i-1]['hypervolume']} -> {results[i]['hypervolume']}"
    
    print(f"\n✓ 结果已保存到文件:")
    print(f"  文本: {output_file}")
    print(f"  JSON: {json_file}")


@pytest.mark.parametrize('phase', ['phase_1_oxide', 'phase_1_organic', 'phase_2'])
def test_all_phases_output(optimizer_manager, temp_dirs, seed, device, phase):
    """测试所有阶段的 run_single_step 输出结果，并保存到文件"""
    param_space = optimizer_manager._get_phase_parameter_space(phase)
    optimizer = TraceAwareKGOptimizer(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device,
        phase=phase,
        param_space=param_space
    )
    
    # 创建输出文件路径（保存到测试目录，方便查看）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_name = phase.replace('_', '-')
    
    # 第一个阶段运行10轮，其他阶段运行2轮
    n_iterations = 10 if phase == 'phase_1_oxide' else 2
    
    output_file = os.path.join(output_dir, f"run_single_step_{phase_name}_{n_iterations}iter_{timestamp}.txt")
    json_file = os.path.join(output_dir, f"run_single_step_{phase_name}_{n_iterations}iter_{timestamp}.json")
    
    # 用于保存 JSON 格式的结果
    results_data = {
        'test_timestamp': timestamp,
        'phase': phase,
        'param_names': optimizer.param_names,
        'n_init': int(optimizer.n_init),
        'batch_size': int(optimizer.batch_size),
        'n_iterations': n_iterations,
        'iterations': []
    }
    
    # 打开文件用于写入文本输出
    with open(output_file, 'w', encoding='utf-8') as f:
        def write_output(text):
            """同时写入文件和打印到控制台"""
            print(text)
            f.write(text + '\n')
        
        write_output("="*60)
        write_output(f"阶段: {phase}")
        write_output(f"测试时间: {timestamp}")
        write_output(f"迭代次数: {n_iterations} 轮")
        write_output("="*60)
        write_output(f"参数名称: {optimizer.param_names}")
        write_output(f"参数数量: {len(optimizer.param_names)}")
        write_output(f"初始样本数 (n_init): {optimizer.n_init}")
        write_output(f"批次大小 (batch_size): {optimizer.batch_size}")
        write_output("")
        
        # 运行多次迭代
        all_results = []
        hypervolumes = []
        
        for iter_num in range(n_iterations):
            result = optimizer.run_single_step(simulation_flag=True)
            all_results.append(result)
            hypervolumes.append(result['hypervolume'])
            
            candidates_np = result['candidates'].cpu().numpy()
            
            write_output("\n" + "="*60)
            if iter_num == 0:
                write_output(f"第 {iter_num+1} 次迭代（初始样本）返回结果：")
            else:
                write_output(f"第 {iter_num+1} 次迭代（贝叶斯优化）返回结果：")
            write_output("="*60)
            write_output(f"iteration: {result['iteration']}")
            write_output(f"candidates shape: {result['candidates'].shape}")
            write_output(f"hypervolume: {result['hypervolume']:.6f}")
            write_output(f"当前总样本数: {optimizer.X.shape[0]}")
            
            if iter_num < 2 or iter_num == n_iterations - 1:  # 详细显示前2次和最后一次
                write_output(f"候选样本:")
                for i in range(min(3, candidates_np.shape[0])):  # 最多显示3个样本
                    param_str = ", ".join([f"{optimizer.param_names[j]}: {candidates_np[i][j]:.2f}" 
                                          for j in range(len(optimizer.param_names))])
                    write_output(f"  样本 {i}: [{param_str}]")
                if candidates_np.shape[0] > 3:
                    write_output(f"  ... (还有 {candidates_np.shape[0] - 3} 个样本)")
            
            # 保存到 JSON
            results_data['iterations'].append({
                'iteration': int(result['iteration']),
                'candidates': candidates_np.tolist(),
                'hypervolume': float(result['hypervolume']),
                'sample_count': int(optimizer.X.shape[0])
            })
        
        # 总结信息
        write_output("\n" + "="*60)
        write_output("迭代总结：")
        write_output("="*60)
        write_output(f"总迭代次数: {len(all_results)}")
        write_output(f"最终样本数: {optimizer.X.shape[0]}")
        write_output(f"最终超体积: {all_results[-1]['hypervolume']:.6f}")
        write_output(f"超体积变化: {all_results[0]['hypervolume']:.6f} -> {all_results[-1]['hypervolume']:.6f}")
        write_output(f"超体积增量: {all_results[-1]['hypervolume'] - all_results[0]['hypervolume']:.6f}")
        write_output("")
        write_output("每次迭代的超体积:")
        for i, hv in enumerate(hypervolumes):
            write_output(f"  迭代 {i}: {hv:.6f}")
        write_output("")
        
        # 保存总结信息
        results_data['summary'] = {
            'total_iterations': len(all_results),
            'final_sample_count': int(optimizer.X.shape[0]),
            'final_hypervolume': float(all_results[-1]['hypervolume']),
            'initial_hypervolume': float(all_results[0]['hypervolume']),
            'hypervolume_delta': float(all_results[-1]['hypervolume'] - all_results[0]['hypervolume']),
            'hypervolumes': [float(hv) for hv in hypervolumes]
        }
        
        write_output("✓ 测试完成！")
        write_output(f"\n结果已保存到:")
        write_output(f"  文本文件: {output_file}")
        write_output(f"  JSON文件: {json_file}")
    
    # 保存 JSON 格式的结果
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 验证返回值的结构
    assert len(all_results) == n_iterations, f"应该运行 {n_iterations} 次迭代"
    assert all_results[0]['iteration'] == 0, "第一次迭代应该返回 iteration=0"
    assert all_results[-1]['iteration'] == n_iterations - 1, f"最后一次迭代应该返回 iteration={n_iterations-1}"
    assert all_results[0]['candidates'].shape[0] == optimizer.n_init, "第一次迭代应该返回 n_init 个样本"
    
    # 验证超体积非递减
    for i in range(1, len(hypervolumes)):
        assert hypervolumes[i] >= hypervolumes[i-1] - 1e-6, \
            f"超体积不应该减少: {hypervolumes[i-1]} -> {hypervolumes[i]}"
    
    print(f"\n✓ [{phase}] {n_iterations}轮迭代结果已保存到文件:")
    print(f"  文本: {output_file}")
    print(f"  JSON: {json_file}")


if __name__ == '__main__':
    # 直接运行测试以查看输出
    pytest.main([__file__, '-v', '-s'])

