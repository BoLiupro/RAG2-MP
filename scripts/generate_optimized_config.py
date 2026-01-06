#!/usr/bin/env python3
"""
优化配置生成器
根据测试结果自动生成优化后的配置文件
"""

import yaml
import argparse
import os
from datetime import datetime

def generate_optimized_config(base_config_path, optimization_level="medium"):
    """
    生成优化后的配置文件
    
    Args:
        base_config_path: 基础配置文件路径
        optimization_level: 优化级别 (low, medium, high)
    """
    
    # 读取基础配置
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"生成优化配置 - 优化级别: {optimization_level}")
    print("="*60)
    
    # 根据优化级别调整参数
    if optimization_level == "low":
        # 轻微优化 - 主要改进格式控制
        config['model']['llm']['temperature'] = 0.5
        config['model']['llm']['top_p'] = 0.85
        config['detailed_debug']['num_samples'] = 10
        
        print("✓ 降低temperature到0.5（提高格式遵守度）")
        print("✓ 调整top_p到0.85")
        
    elif optimization_level == "medium":
        # 中度优化 - 平衡性能和质量
        config['model']['llm']['temperature'] = 0.3
        config['model']['llm']['top_p'] = 0.8
        config['model']['llm']['top_k'] = 40
        config['model']['llm']['max_new_tokens'] = 128
        config['model']['rag']['top_m_samples'] = 3
        config['model']['gravity']['top_n_candidates'] = 5
        config['detailed_debug']['num_samples'] = 20
        
        print("✓ 降低temperature到0.3（更确定的输出）")
        print("✓ 调整top_p到0.8")
        print("✓ 调整top_k到40")
        print("✓ 减少max_new_tokens到128（加快生成）")
        print("✓ RAG检索样本数改为3（减少噪音）")
        print("✓ 重力模型候选数增加到5（更多选择）")
        print("✓ 测试样本数增加到20")
        
    elif optimization_level == "high":
        # 激进优化 - 最大化格式遵守和速度
        config['model']['llm']['temperature'] = 0.1
        config['model']['llm']['top_p'] = 0.7
        config['model']['llm']['top_k'] = 30
        config['model']['llm']['max_new_tokens'] = 64
        config['model']['llm']['do_sample'] = False  # 使用greedy decoding
        config['model']['rag']['top_m_samples'] = 2
        config['model']['gravity']['top_n_candidates'] = 7
        config['model']['gravity']['radius'] = 3  # 减小搜索半径
        config['detailed_debug']['num_samples'] = 50
        
        print("✓ 降低temperature到0.1（接近确定性）")
        print("✓ 禁用sampling，使用greedy decoding")
        print("✓ 大幅减少max_new_tokens到64")
        print("✓ RAG仅检索2个最相似样本")
        print("✓ 重力模型候选数增加到7")
        print("✓ 减小搜索半径到3（更集中）")
        print("✓ 测试样本数增加到50")
    
    # 添加优化标记
    config['_optimization_info'] = {
        'level': optimization_level,
        'generated_at': datetime.now().isoformat(),
        'base_config': base_config_path
    }
    
    # 生成输出文件名
    output_path = base_config_path.replace('.yaml', f'_optimized_{optimization_level}.yaml')
    
    # 保存优化后的配置
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("="*60)
    print(f"✓ 优化配置已保存到: {output_path}")
    print(f"\n使用方法:")
    print(f"  ./scripts/detailed_train.sh {output_path}")
    
    return output_path


def generate_fewshot_prompt_template():
    """生成Few-shot learning的prompt模板"""
    
    template = '''"""
Few-shot Prompt Template for Mobility Prediction
使用此模板可以提高LLM的格式遵守度和预测准确度
"""

def build_fewshot_prediction_prompt(
    observation_trajectory,
    rag_summary,
    candidates_by_category,
    current_location
):
    """
    构建包含few-shot示例的预测prompt
    """
    
    prompt = """I will show you examples of correct predictions first.

EXAMPLE 1:
Current Location: Grid 465 (Transportation Facilities)
Suggested Categories: Shopping & Consumer Goods, Life Services
Candidates: Grid 465, 545, 546, 464, 466
Correct Output: {"Grid": 545}

EXAMPLE 2:
Current Location: Grid 1176 (Transportation Facilities)
Suggested Categories: Companies & Enterprises, Residential
Candidates: Grid 1176, 1216, 1136, 1177
Correct Output: {"Grid": 1216}

EXAMPLE 3:
Current Location: Grid 975 (Transportation Facilities)
Suggested Categories: Shopping & Consumer Goods, Life Services
Candidates: Grid 975, 1014, 974, 976
Correct Output: {"Grid": 1014}

Now it's your turn. Follow the EXACT same format.

CRITICAL RULES:
1. Output ONLY a JSON object: {"Grid": <number>}
2. NO explanations, NO additional text
3. Choose from candidate Grid IDs only
4. Consider the suggested categories from similar mobility patterns

"""
    
    # Add current prediction task
    prompt += f"CURRENT PREDICTION TASK:\\n"
    prompt += f"Current Location: Grid {current_location}\\n"
    
    # Add RAG summary (extract suggested categories)
    import json
    try:
        rag_data = json.loads(rag_summary)
        suggested_categories = rag_data.get('category_of_next_location', 'Unknown')
        prompt += f"Suggested Categories: {suggested_categories}\\n"
    except:
        prompt += f"Suggested Categories: See RAG summary below\\n"
    
    # Add candidates list
    all_candidates = set([current_location])
    for category, cands in candidates_by_category.items():
        for grid_id, score in cands:
            all_candidates.add(grid_id)
    
    candidates_str = ', '.join([f"Grid {g}" for g in sorted(list(all_candidates))[:15]])
    prompt += f"Candidates: {candidates_str}\\n\\n"
    
    # Add detailed trajectory and candidates
    prompt += "## Detailed Information\\n\\n"
    prompt += "### Current Trajectory\\n"
    # ... add formatted trajectory ...
    
    prompt += "\\n### RAG Summary\\n"
    prompt += f"{rag_summary}\\n\\n"
    
    prompt += "### Candidate Locations\\n"
    # ... add formatted candidates ...
    
    prompt += "\\n\\nYOUR OUTPUT (JSON only):\\n"
    
    return prompt
"""

# 保存模板
template_path = '/workspace/China_Journal/model/fewshot_prompt_template.py'
with open(template_path, 'w') as f:
    f.write(template)

print(f"✓ Few-shot prompt模板已保存到: {template_path}")
print(f"\n如何使用:")
print(f"1. 在Predictor.py中导入此模板")
print(f"2. 替换_build_final_prediction_prompt方法")
print(f"3. 重新测试以验证改进效果")

return template_path


def main():
    parser = argparse.ArgumentParser(description='生成优化的配置文件')
    parser.add_argument('--base-config', 
                       default='/workspace/China_Journal/config/config.yaml',
                       help='基础配置文件路径')
    parser.add_argument('--level', 
                       choices=['low', 'medium', 'high'],
                       default='medium',
                       help='优化级别: low, medium, high')
    parser.add_argument('--generate-fewshot', 
                       action='store_true',
                       help='生成Few-shot prompt模板')
    
    args = parser.parse_args()
    
    # 生成优化配置
    optimized_path = generate_optimized_config(args.base_config, args.level)
    
    print("\\n" + "="*60)
    print("建议的后续步骤:")
    print("="*60)
    print("1. 使用优化配置运行测试:")
    print(f"   ./scripts/detailed_train.sh {optimized_path}")
    print("")
    print("2. 比较结果:")
    print("   cat output/logs/detailed_test_results.json | python -m json.tool")
    print("")
    print("3. 如果结果改善，可以替换原配置:")
    print(f"   cp {optimized_path} {args.base_config}")
    
    # 生成Few-shot模板（如果请求）
    if args.generate_fewshot:
        print("\\n" + "="*60)
        print("生成Few-shot Prompt模板...")
        print("="*60)
        generate_fewshot_prompt_template()


if __name__ == '__main__':
    main()
'''
