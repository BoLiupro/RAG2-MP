<div align="center">

# RAG²-MP
### 基于检索增强生成与引力模型的用户移动行为预测

[![Venue](https://img.shields.io/badge/计算机学报-2026-blue)](https://cjc.ict.ac.cn/)
[![Status](https://img.shields.io/badge/Status-Accepted-success)](https://tong89.github.io/tongli.github.io/)

**Bo Liu · Tong Li · Zhu Xiao · Zhuo Tang · Kenli Li**

*《计算机学报》, 2026 — 已录用*

</div>

---

## 项目简介

**RAG²-MP** 是一个面向用户移动行为预测的语义推理框架，将 **Retrieval-Augmented Generation (RAG)** 与改进的 **Gravity Model** 结合，并由大语言模型整合历史群体行为模式、空间吸引力与距离约束，完成最终位置预测。

<p align="center">
  <img src="sources/framework.png" width="92%" alt="RAG2-MP framework" />
</p>

## 方法亮点

- **检索增强模块**：构建历史轨迹知识库并检索相似群体行为模式，为预测提供可解释的经验知识。
- **引力搜索模块**：建模 POI 分布、空间吸引力与地理距离衰减，为候选位置提供显式空间先验。
- **LLM 决策模块**：融合检索结果、空间先验和上下文信息，进行语义推理与最终预测。
- **跨城市验证**：在多个真实世界数据集上测试泛化能力。

## 论文信息

论文 **“A Mobility Prediction Method Based on Retrieval-Augmented Generation and Gravity Model”** 已于 2026 年被《计算机学报》录用。

> 正式卷期、页码和 DOI 将以期刊最终出版信息为准。

## 数据集

本项目涉及北京、深圳、上海等真实世界移动行为数据。使用数据时请遵循原始数据提供方的许可与隐私要求。

- **Beijing / CoPB**: https://github.com/tsinghua-fib-lab/CoPB
- **Shenzhen private-car trajectories**: https://github.com/HunanUniversityZhuXiao/PrivateCarTrajectoryData
- **Shanghai app/mobility data**: https://fi.ee.tsinghua.edu.cn/appusage/

## 环境配置

```bash
conda env create -f environment.yml
conda activate rag2-mp
```

## 运行流程

### 1. 构建检索数据库与空间先验

```bash
python util/fit_gravity_weight.py
python util/build_rag_database.py
```

### 2. 训练 / 测试

```bash
python trainer/trainer.py --config config/config.yaml
```

如需输出更详细的推理日志：

```bash
python trainer/detailed_trainer.py --config config/config.yaml
```

## Citation

```bibtex
@article{liu2026rag2mp,
  title   = {A Mobility Prediction Method Based on Retrieval-Augmented Generation and Gravity Model},
  author  = {Liu, Bo and Li, Tong and Xiao, Zhu and Tang, Zhuo and Li, Kenli},
  journal = {Chinese Journal of Computers},
  year    = {2026},
  note    = {Accepted}
}
```

## Contact

如有问题或学术交流，欢迎提交 Issue，或联系 **Bo Liu**：`liubo317@hnu.edu.cn`。
