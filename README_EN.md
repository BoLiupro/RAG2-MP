<div align="center">

# RAG²-MP
### A Mobility Prediction Method Based on Retrieval-Augmented Generation and Gravity Model

**[中文](README.md) | English**

[![Venue](https://img.shields.io/badge/Chinese%20Journal%20of%20Computers-2026-blue)](https://cjc.ict.ac.cn/)
[![Status](https://img.shields.io/badge/Status-Accepted-success)](https://tong89.github.io/tongli.github.io/)

**Bo Liu · Tong Li · Zhu Xiao · Zhuo Tang · Kenli Li**

*Chinese Journal of Computers, 2026 — Accepted*

</div>

---

## Overview

**RAG²-MP** is an LLM-based mobility prediction framework that combines **Retrieval-Augmented Generation (RAG)** with an improved **Gravity Model**. The framework equips an LLM with explicit historical mobility knowledge and physically interpretable spatial priors, allowing it to reason jointly over collective mobility patterns, regional attractiveness, geographic distance, and current context when predicting a user's next location.

<p align="center">
  <img src="sources/framework.png" width="92%" alt="RAG2-MP framework" />
</p>

## Abstract

Human mobility prediction plays an important role in urban traffic management, location-based recommendation, and emergency response. Existing mobility prediction approaches based on large language models exhibit strong semantic reasoning capabilities, but they largely depend on knowledge stored implicitly in static model parameters. As a result, they have difficulty explicitly exploiting statistical regularities and mobility patterns contained in historical observations. They also lack explicit spatial awareness, making it difficult to quantify geographic distance constraints and regional attractiveness, which can lead to geographically implausible predictions. To address these limitations, we propose a mobility prediction framework that integrates **Retrieval-Augmented Generation (RAG)** with an improved gravity model. First, a retrieval-augmented module constructs a historical trajectory knowledge base and dynamically retrieves similar collective mobility patterns, converting implicit mobility regularities into explicit structured knowledge that can be used during inference. Second, an improved gravity-search module explicitly models the nonlinear relationship between POI distributions and geographic distance decay, providing interpretable physical constraints and candidate-location priors. Finally, an LLM serves as the reasoning and decision center to integrate these heterogeneous information sources and generate the final semantic prediction. Extensive experiments on three large-scale real-world datasets show that RAG²-MP improves **Acc@1, Acc@5, and MRR@5 by 10.02%, 11.32%, and 8.91% on average**, respectively, compared with strong existing baselines. Further analyses demonstrate robust cross-city generalization and show that enlarging the retrieval knowledge base can further improve prediction performance.

## Highlights

- **Retrieval-Augmented Mobility Knowledge** — constructs a trajectory knowledge base and retrieves behaviorally similar collective mobility patterns at inference time.
- **Improved Gravity Search** — models POI-based attractiveness and nonlinear geographic-distance decay to provide interpretable spatial priors.
- **LLM-based Semantic Decision Making** — integrates retrieved evidence, spatial constraints, and contextual information for final mobility prediction.
- **Cross-city Generalization** — evaluated on multiple real-world urban mobility datasets.

## Paper Information

The paper **“A Mobility Prediction Method Based on Retrieval-Augmented Generation and Gravity Model”** was accepted by the **Chinese Journal of Computers** on **September 1, 2026**.

- **Journal:** Chinese Journal of Computers
- **Status:** Accepted
- **Journal website:** https://cjc.ict.ac.cn/

> Volume, issue, page numbers, DOI, and the final full-text download link will be updated after formal publication.

## Datasets

The project uses real-world mobility datasets covering Beijing, Shenzhen, and Shanghai. Please follow the licenses and privacy requirements of the original data providers.

- **Beijing / CoPB:** https://github.com/tsinghua-fib-lab/CoPB
- **Shenzhen private-car trajectories:** https://github.com/HunanUniversityZhuXiao/PrivateCarTrajectoryData
- **Shanghai app/mobility data:** https://fi.ee.tsinghua.edu.cn/appusage/

## Environment

```bash
conda env create -f environment.yml
conda activate rag2-mp
```

## Running the Project

### 1. Build the retrieval database and spatial priors

```bash
python util/fit_gravity_weight.py
python util/build_rag_database.py
```

### 2. Train / evaluate

```bash
python trainer/trainer.py --config config/config.yaml
```

For more detailed inference logs:

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

For questions, academic discussions, or collaboration, please open an issue or contact **Bo Liu** at `liubo317@hnu.edu.cn`.
