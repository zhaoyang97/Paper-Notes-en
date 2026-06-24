---
title: >-
  [Paper Note] TabFlex: Scaling Tabular Learning to Millions with Linear Attention
description: >-
  [ICML2025][LLM (Other)][Linear Attention] Replaces the softmax attention in TabPFN with linear attention to scale the in-context learning (ICL) method for tabular classification from small datasets to millions of samples, achieving a over 2× speedup with no performance degradation.
tags:
  - "ICML2025"
  - "LLM (Other)"
  - "Linear Attention"
  - "In-Context Learning"
  - "TabPFN"
  - "Tabular Classification"
  - "Scalability"
date: 2026-05-08
content_hash: 1eaf6acf1acf6c04
---

# TabFlex: Scaling Tabular Learning to Millions with Linear Attention

**Conference**: ICML2025  
**arXiv**: [2506.05584](https://arxiv.org/abs/2506.05584)  
**Code**: [microsoft/ticl](https://github.com/microsoft/ticl)  
**Area**: Tabular Learning  
**Keywords**: Linear Attention, In-Context Learning, TabPFN, Tabular Classification, Scalability

## TL;DR

Replaces the softmax attention in TabPFN with linear attention to scale the in-context learning (ICL) method for tabular classification from small datasets to millions of samples, achieving a over 2× speedup with no performance degradation.

## Background & Motivation

TabPFN leverages the in-context learning (ICL) capability of Transformers for tabular classification. By feeding all training and test samples as a single prompt in a single forward pass, it achieves outstanding performance on small-scale datasets. However, the $O(n^2)$ complexity of softmax attention severely limits its scalability:

- **Limitations of sample size**: TabPFN only supports approximately 3000 training samples; larger datasets must be truncated.
- **Limitations of feature count**: It only processes up to 100 features and 10 classes.
- **Real-world gap**: Real-world tabular data in domains such as recommendation systems, finance, and healthcare often contain hundreds of thousands to millions of rows.

Core Problem: **How can TabPFN be scaled to large-scale, high-dimensional tabular data while maintaining the training-free advantages of ICL?**

## Method

### 1. Analysis of Attention Mechanism Selection

The authors systematically compare two types of alternative structures:

**Issues with State Space Models (SSM/Mamba)**: SSMs are inherently causal models, where the output depends only on previous tokens. Experiments indicate that causal attention performs poorly in ICL—as the number of training samples increases, the accuracy of causal models first rises and then falls, whereas non-causal models show continuous improvement. The training loss of Mamba is significantly higher than that of TabPFN, and its test AUC is much lower.

**Advantages of Linear Attention**: Non-causal linear attention maintains comparable performance to softmax attention while significantly reducing computational overhead.

### 2. Core Formulas of Linear Attention

Standard softmax attention output:

$$\mathbf{a}_i = \frac{\sum_{j=1}^{n} \exp(\mathbf{q}_i^\top \mathbf{k}_j) \cdot \mathbf{v}_j}{\sum_{j=1}^{n} \exp(\mathbf{q}_i^\top \mathbf{k}_j)}$$

Linear attention replaces $\exp(\mathbf{q}_i^\top \mathbf{k}_j)$ with $\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)$:

$$\mathbf{a}_i = \frac{\phi(\mathbf{q}_i)^\top \sum_{j=1}^{n} \phi(\mathbf{k}_j) \cdot \mathbf{v}_j}{\phi(\mathbf{q}_i)^\top \sum_{j=1}^{n} \phi(\mathbf{k}_j)}$$

where $\phi: \mathbb{R}^d \to \mathbb{R}^d$ is a feature mapping function (e.g., $\text{elu}(\cdot)+1$). The two summation terms $\sum_{j=1}^{n} \phi(\mathbf{k}_j) \cdot \mathbf{v}_j$ and $\sum_{j=1}^{n} \phi(\mathbf{k}_j)$ can be precomputed once, reducing the computational complexity of each position from $O(n)$ to $O(1)$, and the total computation from $O(n^2d)$ to $O(nd^2)$.

### 3. Theoretical Guarantees for HBM Efficiency

**Theorem 1**: For any element-wise kernel feature mapping, the HBM access, HBM memory, and FLOPS of non-causal linear attention are $O(ND)$, $O(ND)$, and $O(ND^2)$ respectively, aligning with the optimized FlashLinearAttention. Therefore, direct implementation in PyTorch is sufficient to achieve linear-level HBM efficiency without requiring custom CUDA kernels.

### 4. Three-Sub-Model Architecture of TabFlex

| Sub-model | Prompt Length | Number of Features | Number of Classes | Scenario |
|--------|------------|--------|--------|---------|
| TabFlex-S100 | 1152 | 100 | 10 | Small-scale low-dimensional datasets (n<3K, d≤100) |
| TabFlex-L100 | 50K | 100 | 10 | Large-scale low-dimensional datasets (n≥3K, d≤100) |
| TabFlex-H1K | 50K | 1000 | 100 | Large-scale high-dimensional datasets (d>100) |

**Condition-Based Selection Strategy** (Algorithm 1):

1. If $n \geq 3K$ and $d \leq 100$ $\to$ TabFlex-L100
2. If $d > 100$ or ($d/n \geq 0.2$ and $n \geq 3K$) $\to$ TabFlex-H1K (features exceeding 1000 are first reduced using random projection)
3. Otherwise $\to$ TabFlex-S100

## Key Experimental Results

### Performance Ranking on 57 Small Datasets (≤1250 samples)

| Algorithm | Type | Median AUC | Mean AUC | Median Time / 1k Samples (s) |
|------|------|---------|---------|-------------------|
| **TabPFN** | TF | 0.97 | 0.90 | 0.82 |
| **TabFlex** | TF | **0.96** | **0.89** | **0.29** |
| CatBoost | GBDT | 0.95 | 0.89 | 2.59 |
| RandomForest| Classical | 0.92 | 0.86 | 0.45 |
| XGBoost | GBDT | 0.91 | 0.86 | 0.49 |

The performance of TabFlex is almost on par with TabPFN, while achieving a speedup of over **2×**.

### Highlights on Large-Scale High-Dimensional Datasets

| Dataset | Sample Size | Feature Count | TabPFN AUC | TabFlex AUC | TabFlex Time (s) |
|--------|--------|--------|-----------|------------|---------------|
| poker-hand | 1,025,009 | 10 | 0.72 | **0.84** | **4.88** |
| albert | 425,240 | 78 | 0.69 | **0.70** | 13.46 |
| airlines | 539,383 | 7 | 0.63 | **0.64** | 4.20 |
| nomao | 34,465 | 118 | 0.76 | **0.99** | 5.34 |

On poker-hand (1M+ samples), TabFlex completes classification in only **4.88 seconds**, whereas the 5th fastest baseline requires **504 seconds**.

### TabZilla Hard Benchmark

Among 36 hard datasets, only TabFlex, TabPFN, and XGBoost successfully ran across all datasets. TabFlex is faster and performs better than TabPFN, and is faster than XGBoost with a minor performance gap.

## Highlights & Insights

1. **Causality is a bottleneck for ICL**: For the first time, experiments clearly demonstrate the systematic disadvantage of causal attention/SSMs in tabular ICL—non-causal mechanisms are required to fully exploit the permutation invariance of all training samples.
2. **Natural fit between linear attention and ICL**: The precomputation of global statistics in linear attention perfectly complies with the semantics that "all training samples are equivalent" in tabular data, with almost no performance loss.
3. **Extreme efficiency**: Processes 1 million samples in under 5 seconds, which is two orders of magnitude faster than traditional GBDT methods.
4. **Engineering simplicity**: Theoretical proofs show that a native PyTorch implementation already achieves optimal HBM efficiency, eliminating the need for custom CUDA kernels.
5. **Condition-based selection among three models**: Simple rules are sufficient to cover global scenarios ranging from small-scale to million-scale datasets, and from low-dimensional to high-dimensional spaces.

## Limitations & Future Work

1. **Coarse support for regression tasks**: Currently, continuous values are converted into classification tasks via binning discretization, which performs significantly worse than XGBoost on 18 regression datasets.
2. **Performance gap in high-dimensional settings**: When the feature dimension exceeds 800, the accuracy-speed trade-off of XGBoost is superior to that of TabFlex.
3. **Reliance on synthetic data pre-training**: The model is pre-trained offline on synthetic prior data, so distribution shifts in real-world data may impact generalization.
4. **Non-smooth transition between sub-models**: The decision boundary uses hard thresholds, which may cause datasets near the boundary to select sub-optimal sub-models.
5. **Unexplored linear attention variants**: Advanced linear attention mechanisms such as RetNet and GLA could potentially yield further improvements.

## Rating

- Novelty: ⭐⭐⭐ — The core innovation (replacing softmax with linear attention) is relatively direct, but the causal analysis and the three-sub-model strategy offer strong engineering insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation involving 115 OpenML datasets, 25 baselines, covering both classification and regression, with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with tight integration between theory and experiments, and high information-density charts.
- Value: ⭐⭐⭐⭐ — Significantly advances the practicality of tabular ICL, carrying direct application value for large-scale industrial tabular tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](../../NeurIPS2025/llm_nlp/in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)
- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](../../NeurIPS2025/llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[NeurIPS 2025\] Composing Linear Layers from Irreducibles](../../NeurIPS2025/llm_nlp/composing_linear_layers_from_irreducibles.md)
- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](../../NeurIPS2025/llm_nlp/linear_transformers_implicitly_discover_unified_numerical_algorithms.md)
- [\[CVPR 2025\] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](../../CVPR2025/llm_nlp/exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)

</div>

<!-- RELATED:END -->
