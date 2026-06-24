---
title: >-
  [Paper Note] Retraining-Free Merging of Sparse MoE via Hierarchical Clustering
description: >-
  [ICML 2025][LLM Efficiency][Sparse Mixture-of-Experts] Proposes HC-SMoE, a retraining-free expert merging framework based on hierarchical clustering of expert outputs. It achieves efficient compression of SMoE models through output similarity metrics and hierarchical clustering, reducing expert parameters by 25%-50% on Qwen and Mixtral while maintaining superior performance.
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Sparse Mixture-of-Experts"
  - "Expert Merging"
  - "Hierarchical Clustering"
  - "Model Compression"
  - "Retraining-Free"
date: 2026-05-08
content_hash: 7b163ec0be1bb075
---

# Retraining-Free Merging of Sparse MoE via Hierarchical Clustering

**Conference**: ICML 2025  
**arXiv**: [2410.08589](https://arxiv.org/abs/2410.08589)  
**Code**: [GitHub](https://github.com/wazenmai/HC-SMoE)  
**Area**: LLM Efficiency  
**Keywords**: Sparse Mixture-of-Experts, Expert Merging, Hierarchical Clustering, Model Compression, Retraining-Free

## TL;DR

Proposes HC-SMoE, a retraining-free expert merging framework based on hierarchical clustering of expert outputs. It achieves efficient compression of SMoE models through output similarity metrics and hierarchical clustering, reducing expert parameters by 25%-50% on Qwen and Mixtral while maintaining superior performance.

## Background & Motivation

Sparse Mixture-of-Experts (SMoE) models achieve efficient parameter utilization through sparse activation mechanisms—each input token activates only a subset of experts, thereby scaling model capacity without increasing inference costs. However, the **total parameter size** of SMoE models remains massive (requiring all expert parameters to be loaded into memory), posing a severe deployment bottleneck in resource-constrained environments.

Prior research has revealed the severity of the **expert redundancy** problem in SMoE:

**Liu et al. (2023)** identified high representational similarity among experts

**Lu et al. (2024)** provided further empirical support
3. These redundancies indicate substantial room for optimization

Existing expert reduction methods possess clear limitations:

- **Expert Pruning** (TSEP, O-prune, S-prune): Directly removing experts causes irreversible loss of learned representations.
- **M-SMoE Expert Merging**: Relies on router logits for grouping, which is sensitive to task data, and using activation frequency to determine the number of retained experts per layer performs poorly in task-agnostic settings.
- **ZipIt Model Merging**: Feature correlation computation is computationally expensive and unsuitable for large-scale expert merging.

**Key Insight**: Rather than discarding expert parameters (pruning), it is better to merge functionally similar experts to preserve more knowledge. Using expert outputs rather than router logits as the similarity metric can better capture the functional equivalence of experts.

## Method

### Overall Architecture

HC-SMoE is a two-stage framework: **Grouping** $\rightarrow$ **Merging**, which is retraining-free, task-agnostic, and scalable.

Overall pipeline:
1. Collect the average output vector of each expert on a calibration dataset $\mathcal{D}_{cal}$ (e.g., C4).
2. Perform hierarchical clustering on the experts of each layer based on the cosine similarity of their output vectors.
3. Merge experts within the same cluster into a new expert using active frequency weighting.
4. Keep the router network unchanged, redirecting inputs originally routed to any expert within the same cluster to the newly merged expert.

The framework adopts a **static grouping strategy**: maintaining a fixed number of $r$ experts per layer after merging, aligned with O-prune to facilitate a fair comparison.

### Key Designs

#### 1. Output-Based Expert Similarity Metric

This is the core design innovation of HC-SMoE. For expert $E_j$, its representative vector is defined as:

$$o_j := \mathbb{E}_{x \sim \mathcal{D}_{cal}}[E_j(x)] = \frac{1}{T}\sum_{x \in \mathcal{D}_{cal}}^{T} E_j(x)$$

where $T$ is the total number of tokens in the calibration dataset.

**Why not use router logits?**
- Router logits $R(x)$ reflect input-dependent dispatch preferences rather than the intrinsic functionality of experts.
- Router logits possess task-specific bias, hindering generalization.
- Parameter space comparison (such as concatenating flattened weights) fails in high-dimensional spaces.

**Why use expert outputs?**
- Output similarity is highly correlated with functional equivalence (Li et al., 2016; Stoica et al., 2024).
- It simultaneously captures contextual input information and the transformations learned by the experts.
- Its effectiveness is verified by L2 error experiments in the Appendix.

#### 2. Hierarchical Clustering Algorithm

HC-SMoE adopts **agglomerative hierarchical clustering**, with the core steps:

1. **Initialization**: Each expert starts as an individual cluster.
2. **Iterative Merging**: In each step, the two most similar clusters are merged.
3. **Termination Condition**: Stop when the number of clusters is reduced to the target number $r$.

The distance between clusters is calculated using **average linkage**, which is the average similarity of all expert pairs between two clusters.

**Compared to M-SMoE's one-pass grouping**, the advantages of hierarchical clustering include:
- **Iterative Comparison**: Re-evaluates all cluster distances after each merging step, ensuring global optimality.
- **Cluster Diversity**: Maintains better inter-cluster diversity and intra-cluster similarity.
- **Theoretical Guarantees**: Hierarchical clustering provides provable guarantees on clustering quality.
- **Insensitive to Initialization**: Unlike methods like K-means that depend on initialization, it yields deterministic results.

#### 3. Frequency-Weighted Merging Strategy

When merging experts in the same cluster, HC-SMoE uses a **weighted average based on activation frequency**:

For a cluster $C_i = \{E_0^i, E_1^i, \ldots, E_{|C_i|}^i\}$, the merged new expert weight is:

$$W_{\text{merged}}^i = \sum_{j=0}^{|C_i|} \frac{f_j}{\sum_k f_k} \cdot W_j^i$$

where $f_j$ is the activation frequency of expert $E_j$ on the calibration data.

**Design Intuition**: High-frequency experts contribute more, so their parameters should receive larger weights during merging. Note that frequency is only used for computing merging weights, not for clustering and grouping—which is a key difference from M-SMoE.

### Loss & Training

HC-SMoE is a **completely retraining-free** method. It involves no loss function optimization or gradient updates. The entire pipeline only requires:

1. **A single forward pass** through the calibration dataset to collect expert outputs.
2. **Hierarchical clustering** computation (time complexity $O(n^2 \log n)$, where $n$ is the number of experts).
3. **Parameter weighted averaging** to complete the merging.

This makes HC-SMoE highly efficient for actual deployment—compared to methods that require fine-tuning (e.g., TSEP, M-SMoE), HC-SMoE incurs zero GPU training overhead.

## Key Experimental Results

### Main Results

Evaluated on 8 zero-shot language tasks (LM-Harness benchmark).

**Qwen1.5-MoE-A2.7B-Chat Model:**

| Compression Rate | Method | Average Accuracy | vs. Strongest Baseline |
|:---:|:---:|:---:|:---:|
| 0% | Original Model | ~56% | - |
| 25% | S-prune | Lower | - |
| 25% | HC-SMoE | Optimal | **+6.95%** |
| 37.5% | Strongest Baseline | Significant Drop | - |
| 37.5% | HC-SMoE | Optimal | **+2.14%** |
| 50% | Baselines | Massive Drop | - |
| 50% | HC-SMoE | Optimal | Significant Lead |

**Mixtral 8×7B Model:**

| Method | Type | Task-Agnostic | Retraining-Free | Performance |
|:---:|:---:|:---:|:---:|:---:|
| O-prune | Pruning | ✓ | ✓ | Baseline |
| S-prune | Pruning | ✓ | ✓ | Baseline |
| F-prune | Pruning | ✓ | ✓ | Baseline |
| M-SMoE | Merging | ✗ | ✗ | Below Baseline |
| HC-SMoE | Merging | ✓ | ✓ | **Universally Best** |

### Ablation Study

| Configuration | Key Metric | Description |
|:---:|:---:|:---|
| Router logits clustering | Lower accuracy | M-SMoE style, task-sensitive |
| Parameter space clustering | Lower accuracy | High-dimensional space distance metric fails |
| **Expert output clustering** | **Highest accuracy** | Ours (HC-SMoE), captures functional equivalence |
| K-means clustering | Unstable | Sensitive to initialization |
| One-pass grouping | Lower quality | M-SMoE style |
| **Hierarchical clustering** | **Optimal & Stable** | Ours (HC-SMoE), deterministic results |
| Uniform weight merging | Acceptable | Simple average |
| **Frequency-weighted merging** | **Optimal** | Ours (HC-SMoE) method |

### Key Findings

1. **Output similarity is superior to router logits**: Expert outputs directly reflect functional equivalence, whereas router logits merely represent input-dependent dispatch preferences, leading to a significant difference in clustering performance.
2. **Hierarchical clustering outperforms one-pass and K-means**: Iterative merging guarantees globally optimal clustering quality and delivers highly deterministic results.
3. **Clustering quality is crucial for merging performance**: High-quality clustering paired with a simple merging strategy yielded excellent results, whereas poor clustering is difficult to mitigate even with complex merging strategies.
4. **Merging is superior to pruning**: Under the same compression rate, merging preserves far more knowledge, yielding significantly better performance than directly dropping experts.
5. **Strong generalization across datasets**: The clustering results remained stable across different calibration datasets, validating the task-agnostic property.
6. **M-SMoE fails in task-agnostic settings**: Grouping and retention strategies dependent on frequency perform poorly in zero-shot settings.

## Highlights & Insights

1. **Simple yet effective concept**: Deconstructs the expert merging problem into three steps ("metric -> clustering -> merging"), providing clear design justifications and comparative experiments for each step.
2. **Insight on Outputs vs. Routers**: Points out that router logits are inherently input-dependent routing preferences, which do not equal physical expert functional similarity. This observation offers broad inspiration for subsequent MoE research.
3. **Balancing theory and practice**: Provides theoretical analysis of hierarchical clustering quality, backed by thorough experimental validation.
4. **Engineering-friendly**: Requires absolutely zero retraining, needing only a single forward pass and hierarchical clustering, making its computational overhead during deployment extremely low.
5. **Method versatility**: Proves effective across two different scales of MoE architectures, Qwen (60 experts) and Mixtral (8 experts), demonstrating strong cross-model generalization.

## Limitations & Future Work

1. **Evaluated only on language tasks**: All experiments were based on zero-shot language understanding tasks, lacking validation on generative tasks (e.g., summarization, translation).
2. **Static grouping strategy**: Retaining the same number of experts per layer ignores the varying redundancy across different layers. Dynamic allocation could achieve a better compression-performance trade-off.
3. **Sensitivity to calibration datasets**: Although the paper claims to be task-agnostic, the choice of calibration dataset could still influence the representativeness of expert outputs.
4. **Simple merging strategy**: Only frequency-weighted averaging was used; more complex merging options (such as TIES-Merging, DARE, etc.) were not explored.
5. **Lack of combination with methods like quantization**: The integration of HC-SMoE with complementary compression techniques, such as weight quantization and knowledge distillation, remains unexplored.
6. **Scalability of hierarchical clustering**: As the number of experts scales up dramatically (e.g., future designs with hundreds/thousands of experts), the $O(n^2)$ clustering complexity might become a bottleneck.

## Related Work & Insights

- **M-SMoE (Li et al., 2024)**: The most direct prior work, which proposed an expert merging framework but relied on router logits and frequency information. HC-SMoE improves upon it in three dimensions: metrics, clustering, and merging.
- **ZipIt (Stoica et al., 2024)**: Work in the field of model merging that utilizes feature correlations to merge models trained on different tasks, inspiring the perspective that expert merging can be viewed as a multi-model merging problem.
- **O-prune / S-prune (Lu et al., 2024; He et al., 2024)**: Pruning methods that confirm the existence of expert redundancy, though direct removal causes knowledge loss.
- **Inspirations for MoE compression**: The paradigm of output representation vectors combined with hierarchical clustering may generalize to other MoE variants (e.g., Soft-MoE, Expert Choice).

## Rating

| Dimension | Score | Description |
|:---:|:---:|:---|
| Novelty | 7/10 | The combination of hierarchical clustering and output metrics is novel, though individual components are not entirely new |
| Technical Depth | 7/10 | Solid theoretical analysis, but the method itself is relatively intuitive |
| Experimental Thoroughness | 8/10 | Covers multiple models, compression rates, and ablations, yet lacks validation on generative tasks |
| Value | 9/10 | Retraining-free + task-agnostic + open-source code, indicating outstanding engineering practicality |
| Writing Quality | 8/10 | Clear structure with solid justification of motivations |
| **Overall Score** | **7.8/10** | A solid engineering-oriented paper with outstanding practical value |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Expert Merging in Sparse Mixture of Experts with Nash Bargaining](../../ICLR2026/llm_efficiency/expert_merging_in_sparse_mixture_of_experts_with_nash_bargaining.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ICLR 2026\] Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs](../../ICLR2026/llm_efficiency/tactic_adaptive_sparse_attention_with_clustering_and_distribution_fitting_for_lo.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[NeurIPS 2025\] MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE](../../NeurIPS2025/llm_efficiency/moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)

</div>

<!-- RELATED:END -->
