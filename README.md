<div align="center">

# RAG²-MP
### 基于检索增强生成与引力模型的用户移动行为预测

**中文 | [English](README_EN.md)**

[![Venue](https://img.shields.io/badge/计算机学报-2026-blue)](https://cjc.ict.ac.cn/)
[![Status](https://img.shields.io/badge/Status-Accepted-success)](https://tong89.github.io/tongli.github.io/)

**Bo Liu · Tong Li · Zhu Xiao · Zhuo Tang · Kenli Li**

*《计算机学报》, 2026 — 已录用*

</div>

---

## 项目简介

**RAG²-MP** 是一个面向用户移动行为预测的语义推理框架，将 **Retrieval-Augmented Generation (RAG)** 与改进的 **Gravity Model** 结合，并由大语言模型整合历史群体行为模式、空间吸引力、地理距离以及当前上下文信息，完成最终位置预测。

<p align="center">
  <img src="sources/framework.png" width="92%" alt="RAG2-MP framework" />
</p>

## 摘要

移动行为预测对城市交通调度、位置推荐服务与应急管理具有重要意义。现有基于大语言模型（LLMs）的移动行为预测方法虽然在语义推理上表现出色，但其受限于仅依赖内部静态参数记忆，难以显式地捕捉历史数据中的统计特征与移动模式，同时缺乏空间感知能力，无法有效量化物理空间的距离约束与区域吸引力，导致预测结果在地理层面往往缺乏合理性。针对上述局限，本文提出一种融合检索增强生成（RAG）与改进引力模型的移动行为预测框架。具体而言，该框架设计了检索增强模块，通过构建历史轨迹知识库并实时检索相似的群体行为模式，将隐式的移动规律转化为显式的结构化知识，使模型能够摆脱静态参数限制，动态利用实证数据进行推理；同时引入改进的引力搜索模块，显式建模 POI 分布与地理距离的非线性衰减关系，为预测提供可解释的物理约束与候选先验；最后利用大语言模型作为决策中枢整合多源信息进行语义预测。本文在三个大规模真实世界数据集上进行了广泛实验。结果表明，所提方法在 Acc@1、Acc@5 和 MRR@5 指标上相比现有的先进基线模型平均提升了 **10.02%、11.32% 和 8.91%**，验证了该方法的有效性与鲁棒性。进一步分析表明，该框架具备较强的跨城市泛化能力，且通过扩展知识库规模能够进一步增强模型性能。

## 方法亮点

- **检索增强模块**：构建历史轨迹知识库并检索相似群体行为模式，将隐式移动规律转化为显式经验知识。
- **引力搜索模块**：建模 POI 分布、空间吸引力与地理距离的非线性衰减，为候选位置提供可解释的空间先验。
- **LLM 决策模块**：融合检索结果、空间先验与上下文信息，执行语义推理并输出最终预测。
- **跨城市验证**：在多个真实世界数据集上评估模型的迁移与泛化能力。

## 论文信息

论文 **“A Mobility Prediction Method Based on Retrieval-Augmented Generation and Gravity Model”** 已于 **2026 年 9 月 1 日**被《计算机学报》录用。

- **期刊：**《计算机学报》 / Chinese Journal of Computers
- **状态：** Accepted
- **期刊主页：** https://cjc.ict.ac.cn/

> 正式卷期、页码、DOI 以及全文下载链接将以期刊最终出版信息为准。

## 数据集

本项目涉及北京、深圳、上海等真实世界移动行为数据。使用数据时请遵循原始数据提供方的许可与隐私要求。

- **Beijing / CoPB:** https://github.com/tsinghua-fib-lab/CoPB
- **Shenzhen private-car trajectories:** https://github.com/HunanUniversityZhuXiao/PrivateCarTrajectoryData
- **Shanghai app/mobility data:** https://fi.ee.tsinghua.edu.cn/appusage/

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
