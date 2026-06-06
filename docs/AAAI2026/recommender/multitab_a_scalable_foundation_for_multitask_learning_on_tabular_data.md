---
title: >-
  [Paper Note] MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data
description: >-
  [AAAI 2026][Recommender Systems][Multitask Learning] This paper proposes MultiTab-Net — the first multitask Transformer architecture for tabular data — which alleviates task competition via a multitask masked attention m…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "Multitask Learning"
  - "Transformer"
  - "Tabular Data"
  - "Masked Attention"
  - "Synthetic Data Benchmark"
date: 2026-05-08
content_hash: d8d4bad60c794b4b
---

# MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data

**Conference**: AAAI 2026
**arXiv**: [2511.09970](https://arxiv.org/abs/2511.09970)  
**Code**: [Armanfard-Lab/MultiTab](https://github.com/Armanfard-Lab/MultiTab)  
**Area**: Recommender Systems / Tabular Data
**Keywords**: Multitask Learning, Transformer, Tabular Data, Masked Attention, Synthetic Data Benchmark

## TL;DR

This paper proposes MultiTab-Net — the first multitask Transformer architecture for tabular data — which alleviates task competition via a multitask masked attention mechanism, and substantially outperforms existing MLP-based multitask models and single-task Transformer models across datasets from recommendation, census, and physics domains.

## Background & Motivation

### The Gap Between Tabular Data and Multitask Learning

Tabular data is the most abundant data type in the world, widely used in finance, healthcare, and e-commerce. Many real-world scenarios naturally require simultaneous prediction of multiple related targets:

**Healthcare**: The same patient record can be used to predict risks of both diabetes and hypertension.

**E-commerce**: Predicting not only clicks but also add-to-cart and purchase events.

**Recommender Systems**: Jointly optimizing CTR and CVR.

**Multitask Learning (MTL)** leverages inter-task correlations through shared representations to improve generalization and efficiency. However, existing tabular MTL work suffers from two limitations:

**Narrow scope**: Prior work mainly focuses on large-scale recommender systems (MMoE, PLE, STEM), with insufficient exploration of broader tabular scenarios.

**Backbone limitations**: Nearly all methods are MLP-based, which lacks explicit feature interaction modeling and scales poorly in data-rich settings.

### Why Transformers?

- MLPs implicitly learn feature interactions through fully connected layers, lacking explicit interaction modeling.
- Transformers dynamically model feature-to-feature and sample-to-sample dependencies via self-attention.
- The advantages of Transformers in large-data regimes have been extensively validated in NLP and CV.
- Tabular Transformers (FT-T, SAINT) have demonstrated strong capabilities for modeling inter-feature and inter-sample relationships.

### Core Challenge: Task Competition

Extending Transformers to multitask settings introduces task tokens that alter the structure of the attention matrix. Unconstrained interaction among task tokens may cause the **seesaw phenomenon** — dominant tasks monopolize shared capacity, degrading overall performance.

## Method

### Overall Architecture

MultiTab-Net builds upon SAINT, with two core innovations:
1. **Multi-token design**: Each task is assigned a dedicated task token, rather than sharing a single CLS token.
2. **Multitask masked attention**: Task token interactions are selectively restricted in inter-feature attention to alleviate task competition.

Input processing pipeline:
- $d$ feature tokens + $t$ task tokens → each projected to vectors of dimension $e$ via embedding networks.
- Concatenated to form $x\in\mathbb{R}^{(d+t)\times e}$ → passed through $N$ encoder blocks.
- Each encoder block consists of: Inter-Feature Attention + Inter-Sample Attention + FFN.
- Final $t$ task tokens are independently processed by task-specific MLPs to produce predictions.

### Key Designs

#### 1. Multitask Masked Attention

**Function**: Selectively suppresses interactions between certain tokens in inter-feature attention.

**Mechanism**: A mask $M_A$ is added to the pre-activation attention scores of attention matrix $A_i$:

$$A_i = \text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_k}} + M_A\right)$$

$M_A$ assigns $-\infty$ to masked positions, which become zero after softmax.

**Three candidate masking strategies**:

| Strategy | Description | Effect |
|----------|-------------|--------|
| F↛T | Feature-to-task attention is masked | Unstable; sometimes degrades performance |
| T↛T | Cross-task token attention is masked | **Best**; consistently optimal |
| F↛T & T↛T | Combination of both | Sub-optimal |

**Design Motivation**: T↛T masking prevents task tokens from directly influencing each other, thereby alleviating task competition. However, preserving T→F (task tokens attending to features) is necessary, as tasks must extract information from features. Masking F→T yields unstable results, suggesting that feature tokens also need to be aware of task context.

#### 2. Multi-Token vs. Single-Token

**Function**: Assigns each task an independent learnable task token, rather than a shared BERT-style CLS token.

**Core advantage** (confirmed by ablation):
- With 2 tasks, the gap between single- and multi-token is small.
- With 8 tasks (Higgs dataset), multi-token + T↛T masking achieves $\Delta_m = 1.23\%$, while the best single-token configuration yields $\Delta_m = -6.35\%$.
- A single shared token cannot adequately capture task-specific information across multiple tasks.

#### 3. MultiTab-Bench: Synthetic Multitask Dataset Generator

**Function**: Generates synthetic multitask tabular data with controllable task correlation, task complexity, and number of tasks.

**Mechanism**: Weight matrices are constructed via eigendecomposition. Given a desired correlation matrix $\mathbf{P}$, the eigendecomposition $\mathbf{P}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ is performed, and the weight matrix $\mathbf{W}=\mathbf{Q}\boldsymbol{\Lambda}^{1/2}\mathbf{U}^T$ is constructed, where $\mathbf{U}$ consists of orthonormal vectors. It can be shown that the cosine similarity between weight vectors of different tasks equals exactly $P_{ij}$.

Label generation: $y_i=\sum_{k=1}^{d_i}(\mathbf{w}_i^T\mathbf{x})^k+\epsilon_i$

**Advantages over MMoE's synthetic data**:
- Supports an **arbitrary number** of tasks (MMoE is limited to 2).
- Supports different **polynomial degrees** per task (controlling relative difficulty).
- Supports **task-specific noise**.

### Loss & Training

- Binary/multi-class cross-entropy for classification tasks; RMSE for regression tasks.
- Adam optimizer with weight decay.
- Early stopping: monitors average AUC for classification tasks and average EV for regression tasks.
- All results are averaged over 5 random seeds.

## Key Experimental Results

### Main Results

| Model | AliExpress $\Delta_m$↑ | ACS Income $\Delta_m$↑ | Higgs $\Delta_m$↑ |
|-------|----------------------|----------------------|-------------------|
| STL (baseline) | 0.0000 | 0.0000 | 0.0000 |
| MTL | 0.1129 | 0.0612 | -0.6531 |
| MMoE | 0.0873 | 0.0893 | -0.3525 |
| PLE | 0.2778 | 0.0892 | -0.0314 |
| STEM | 0.1763 | 0.0725 | 0.0571 |
| SAINT | 0.1146 | 0.0948 | -1.6514 |
| **MultiTab-Net** | **0.5512** | **0.1064** | **1.2337** |

MultiTab-Net achieves the highest multitask gain across all datasets. Notably, on the 8-task Higgs dataset, most baselines yield negative $\Delta_m$ (i.e., MTL hurts performance relative to STL), whereas MultiTab-Net achieves a positive gain of 1.23%.

### Ablation Study

| Configuration | AliExpress $\Delta_m$ | ACS Income $\Delta_m$ | Higgs $\Delta_m$ |
|---------------|----------------------|----------------------|-------------------|
| Single-token, no mask | 0.2669 | 0.0893 | -6.3491 |
| Multi-token, no mask | 0.2579 | 0.0783 | 1.1182 |
| Multi-token, F↛T | 0.3698 | 0.0951 | 0.9626 |
| Multi-token, T↛T | **0.5512** | **0.1064** | **1.2337** |
| Multi-token, F↛T & T↛T | 0.2975 | 0.1007 | 1.0197 |

**Key Finding**: T↛T masking is consistently optimal across all datasets.

### Computational Efficiency

| Model | AliExpress (Params/FLOPs M) | ACS Income | Higgs |
|-------|-----------------------------|------------|-------|
| SAINT (recent single-task Transformer) | 3.62/9.70 | 0.49/1.35 | 5.50/15.02 |
| STEM (latest MTL model) | 1.55/3.11 | 0.69/1.29 | 1.25/2.51 |
| **MultiTab-Net** | **1.80/4.85** | **0.28/0.77** | **0.70/1.90** |

Compared to SAINT, MultiTab-Net achieves approximately 2× and 8× efficiency improvements on ACS Income and Higgs, respectively — roughly proportional to the number of tasks.

### Key Findings

1. The advantage of the multi-token design grows with the number of tasks (negligible at 2 tasks; substantial at 8 tasks).
2. Synthetic data experiments confirm MultiTab-Net's consistent superiority across varying task correlations, complexities, and task counts.
3. Under non-uniform task complexity settings, MultiTab-Net's advantage becomes even more pronounced.

## Highlights & Insights

1. **Filling a gap**: MultiTab-Net is the first multitask Transformer for tabular data, bringing the benefits of attention mechanisms to the intersection of MTL and tabular learning.
2. **Simple yet effective masking**: The intuition behind T↛T masking is straightforward — prevent tasks from interfering with each other — and it introduces virtually no additional computational overhead.
3. **Practical value of MultiTab-Bench**: Supporting arbitrary task counts, adjustable correlations, and difficulty levels, it provides a standardized synthetic benchmark for MTL research.
4. **Inspired by STEM**: STEM constrains cross-task updates during backpropagation via stop-gradient; MultiTab-Net achieves a more direct form of task isolation at the attention level during the forward pass.

## Limitations & Future Work

1. **Limited dataset scale and diversity**: Only 3 public datasets are evaluated, with task types primarily restricted to classification and regression; tasks such as ranking are not explored.
2. **Unfair comparison with XGBoost**: XGBoost has limited native multitask support; the multioutput variant is only applicable when all tasks share the same output type.
3. **Static masking strategy**: T↛T masking uniformly suppresses all cross-task interactions, without considering that information sharing between certain task pairs may be beneficial.
4. **Dynamic masking not explored**: Adaptive masking learned from task correlations remains an open direction.
5. **Scalability unverified**: Performance with more than 8 tasks remains unknown.

## Related Work & Insights

- **MMoE (Ma et al. 2018)**: Multi-gate mixture-of-experts architecture; pioneered the tabular MTL direction.
- **PLE (Tang et al. 2020)**: Shared and task-specific experts; mitigates the seesaw phenomenon.
- **STEM (Su et al. 2024)**: Stop-gradient constraints; directly inspired the masked attention design in MultiTab-Net.
- **SAINT (Somepalli et al. 2021)**: Inter-sample attention; serves as the architectural foundation of MultiTab-Net.
- Insight: **Task isolation at the attention level** may represent a generalizable paradigm for multitask learning, worth exploring in CV and NLP settings.

## Rating

- Novelty: ⭐⭐⭐⭐ (First multitask Transformer for tabular data; masking design is simple but effective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (3 public datasets + synthetic data; greater diversity would strengthen the work)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; mathematical derivation in the MultiTab-Bench section is rigorous)
- Value: ⭐⭐⭐⭐ (Fills an important gap; open-source code; synthetic benchmark has independent value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[AAAI 2026\] Probabilistic Hash Embeddings for Online Learning of Categorical Features](probabilistic_hash_embeddings_for_online_learning_of_categorical_features.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](../../ICLR2026/recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](../../ACL2026/recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ACL 2026\] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](../../ACL2026/recommender/from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)

</div>

<!-- RELATED:END -->
