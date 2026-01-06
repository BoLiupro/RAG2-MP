#!/usr/bin/env python3
"""
测试脚本：筛选并测试非静止轨迹样本
此脚本专门从测试集中筛选出那些在最后几步发生了位置移动的样本进行预测测试。
打印完整的 Prompt、LLM 回答、Gravity 候选、RAG 摘要以及最终预测结果。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import numpy as np
import random
from tqdm import tqdm
from model.Predictor import MobilityPredictor
from dataset.dataset import load_datasets
from util.utils import calculate_grid_distance

def is_moving_trajectory(observation, lookback_len=3, min_dist_km=0.1, city='beijing'):
    """
    判断轨迹是否为"非静止"
    检查最后 lookback_len 个点中是否存在移动超过 min_dist_km 的情况
    或者当前点与倒数第二个点是否位置不同
    """
    if len(observation) < 2:
        return False
        
    # 1. 检查最后两个点是否相同
    last_loc = observation[-1]['location_id']
    prev_loc = observation[-2]['location_id']
    
    if last_loc != prev_loc:
        return True
        
    # 2. 如果最后两个点相同，检查最近一段轨迹是否有显著移动
    coords = []
    # 获取最后几个点的坐标
    # 注意：这里需要Predictor或utils中有坐标转换逻辑，或者直接用location_id判断
    # 简单起见，我们只检查location_id变化
    
    unique_locs = set()
    check_len = min(len(observation), lookback_len)
    for i in range(check_len):
        unique_locs.add(observation[-(i+1)]['location_id'])
        
    return len(unique_locs) > 1

def run_moving_samples_test():
    """运行非静止样本测试"""
    
    # 1. 加载配置
    config_path = "/workspace/China_Journal/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"{'='*80}")
    print(f"非静止轨迹预测测试 - 城市: {config['data']['city']}")
    print(f"{'='*80}")
    
    # 2. 加载数据集
    print("加载数据集...")
    _, _, test_dataset = load_datasets(
        city=config['data']['city'],
        data_dir=config['data']['data_dir'],
        obs_len=config['data']['obs_len'],
        pred_len=config['data']['pred_len']
    )
    print(f"测试集包含 {len(test_dataset)} 个样本")
    
    # 3. 筛选非静止样本
    print("筛选非静止样本...")
    moving_samples = []
    for idx, sample in enumerate(test_dataset):
        if is_moving_trajectory(sample['observation'], lookback_len=4, city=config['data']['city']):
            # 还可以增加规则：Ground Truth 也发生变化
            last_obs_loc = sample['observation'][-1]['location_id']
            if sample['next_location'] != last_obs_loc:
                moving_samples.append((idx, sample))
    
    print(f"筛选出 {len(moving_samples)} 个非静止且发生下步移动的样本")
    
    if not moving_samples:
        print("未找到符合条件的样本，退出测试")
        return

    # 4. 初始化模型
    print("\n初始化模型...")
    # 确保使用 verbose=True 以打印详细信息
    try:
        prediction_time_interval = config['model']['prediction']['time_interval']
    except KeyError:
        prediction_time_interval = "1 hour"

    predictor = MobilityPredictor(
        llm_model_name=config['model']['llm']['model_name'],
        llm_model_path=config['model']['llm']['model_path'],
        city=config['data']['city'],
        top_k_predictions=config['model']['prediction']['top_k_predictions'],
        rag_top_m_samples=config['model']['rag']['top_m_samples'],
        gravity_top_n_candidates=config['model']['gravity']['top_n_candidates'],
        gravity_weight=config['model']['gravity']['weight'],
        gravity_radius=config['model']['gravity']['radius'],
        use_quantization=config['model']['llm']['use_quantization'],
        verbose=True,  # 开启冗余输出
        temperature=config['model']['llm'].get('temperature', 0.2),
        top_p=config['model']['llm'].get('top_p', 0.8),
        top_k=config['model']['llm'].get('top_k', 40),
        do_sample=config['model']['llm'].get('do_sample', True),
        max_new_tokens=config['model']['llm'].get('max_new_tokens', 256),
        prediction_time_interval=prediction_time_interval
    )
    
    # 5. 随机抽取样本进行测试
    num_test_samples = 5
    selected_indices = random.sample(range(len(moving_samples)), min(num_test_samples, len(moving_samples)))
    selected_samples = [moving_samples[i] for i in selected_indices]
    
    print(f"\n开始测试 {len(selected_samples)} 个样本...")
    
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    
    for i, (original_idx, sample) in enumerate(selected_samples):
        print(f"\n\n{'#'*80}")
        print(f"测试样本 {i+1}/{len(selected_samples)} (原始索引: {original_idx})")
        print(f"{'#'*80}")
        
        observation = sample['observation']
        ground_truth = sample['next_location']
        
        print(f"观察轨迹长度: {len(observation)}")
        print(f"当前位置: Grid {observation[-1]['location_id']}")
        print(f"真实下一位置: Grid {ground_truth}")
        
        # 运行预测
        # 注意：Predictor内部的 verbose=True 会自动打印 prompt 和中间结果
        # 但是 Predictor.predict 方法会抑制一部分输出，除非 print_prompt=True
        predictions, _, _ = predictor.predict(
            observation_trajectory=observation,
            ground_truth=ground_truth,
            print_prompt=True
        )
        
        # 打印预测结果
        print(f"\n{'-'*40}")
        print(f"预测结果 (Top {len(predictions)}):")
        print(f"{'-'*40}")
        
        top1_pred = predictions[0][0] if predictions else -1
        is_correct = (top1_pred == ground_truth)
        
        # 检查 Ground Truth 是否在预测中
        rank = -1
        for r, (pred_loc, conf) in enumerate(predictions):
            marker = "✓" if pred_loc == ground_truth else " "
            print(f"Rank {r+1}: Grid {pred_loc} (Confidence: {conf:.4f}) {marker}")
            if pred_loc == ground_truth:
                rank = r + 1
        
        if rank == 1:
            correct_top1 += 1
            print("\n结果: Top-1 预测正确! 🎉")
        elif rank > 0:
            if rank <= 5: correct_top5 += 1
            if rank <= 10: correct_top10 += 1
            print(f"\n结果: 在 Top-{rank} 中找到正确答案。")
        else:
            print(f"\n结果: 预测错误。真实值 Grid {ground_truth} 未在前 {len(predictions)} 个预测中。")
            
    print(f"\n\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}")
    print(f"测试样本数: {len(selected_samples)}")
    print(f"Top-1 准确率: {correct_top1}/{len(selected_samples)} ({correct_top1/len(selected_samples)*100:.1f}%)")
    print(f"Top-5 准确率: {correct_top5}/{len(selected_samples)} ({correct_top5/len(selected_samples)*100:.1f}%)")
    print(f"Top-10 准确率: {correct_top10}/{len(selected_samples)} ({correct_top10/len(selected_samples)*100:.1f}%)")

if __name__ == "__main__":
    run_moving_samples_test()
