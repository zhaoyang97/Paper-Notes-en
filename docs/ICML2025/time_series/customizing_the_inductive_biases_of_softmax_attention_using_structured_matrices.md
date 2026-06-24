---
title: >-
  [Paper Note] Customizing the Inductive Biases of Softmax Attention using Structured Matrices
description: >-
  [ICML 2025][Time Series][Structured matrices] This paper proposes replacing the low-rank scoring function in softmax attention with efficient structured matrices (BTT and MLR). This both addresses the low-rank bottleneck of standard attention and introduces a distance-dependent computational bias through MLR, yielding improvements in in-context regression, language modeling, and long-range time series forecasting.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Structured matrices"
  - "Attention mechanism"
  - "Inductive bias"
  - "Multi-level low-rank matrices"
  - "Time series forecasting"
date: 2026-05-08
content_hash: 4880e17f7dc79f2c
---

# Customizing the Inductive Biases of Softmax Attention using Structured Matrices

**Conference**: ICML 2025  
**arXiv**: [2509.07963](https://arxiv.org/abs/2509.07963)  
**Code**: [YilunKuang/structured-attention](https://github.com/YilunKuang/structured-attention)  
**Area**: Time Series Analysis  
**Keywords**: Structured matrices, Attention mechanism, Inductive bias, Multi-level low-rank matrices, Time series forecasting

## TL;DR

This paper proposes replacing the low-rank scoring function in softmax attention with efficient structured matrices (BTT and MLR). This both addresses the low-rank bottleneck of standard attention and introduces a distance-dependent computational bias through MLR, yielding improvements in in-context regression, language modeling, and long-range time series forecasting.

## Background & Motivation

The core of the standard Transformer is the attention mechanism, whose scoring function projects the input into low-dimensional query and key spaces followed by a dot-product. This design suffers from two fundamental limitations:

**Low-Rank Bottleneck**: The head dimension $r$ is much smaller than the embedding dimension $D$ ($D = H \cdot r$), making $\mathbf{W}_Q \mathbf{W}_K^\top$ a low-rank matrix of rank $r$. For intrinsically high-dimensional inputs (e.g., high-dimensional regression tasks), a large amount of information is lost during the projection process. Amsel et al. (2024) proved that unless $r \gtrsim d_{\text{input}}$, attention cannot even approximately solve the nearest-neighbor problem on high-dimensional spheres.

**Lack of Distance-Dependent Computational Bias**: Standard attention applies the same scoring function to all token pairs in a sequence, failing to distinguish between local and global interactions. However, real-world data such as natural language exhibits prominent locality—words within the same paragraph are closely coupled, whereas distant words are rarely directly related. Although sparse schemes like Sliding Window Attention (SWA) can save computation, they are typically brittle and often degrade performance.

Core Insight: The scoring function of attention is essentially a bilinear form $s(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{M} \mathbf{x}'$, where the structure of matrix $\mathbf{M}$ determines the inductive bias of the attention mechanism. By replacing the low-rank matrix $\mathbf{W}_Q \mathbf{W}_K^\top$ with a suitable **structured matrix**, one can precisely customize the inductive biases to suit different tasks.

## Method

### Overall Architecture

Starting from the bilinear form of the scoring function, this paper proposes a unified framework to customize the inductive biases of attention using structured matrices. The core idea is to replace the underlying matrix in the attention scoring matrix $\mathbf{S}_{j,j'} = s(\mathbf{x}_j, \mathbf{x}_{j'})$ from low-rank to other families of structured matrices, achieving two objectives:

- **Goal 1**: Resolve the low-rank bottleneck—using high-rank yet parameter- and computation-efficient structured matrices (BTT, MLR) as the core matrix of the bilinear form.
- **Goal 2**: Introduce distance-dependent computational allocation—imposing a hierarchical structure on the scoring matrix $\mathbf{S}$ itself using the MLR structure.

These two applications target the two Ry-mentioned problems separately, yet are both based on the same mathematical tool: the family of structured matrices.

### Key Designs

#### 1. Structured Matrix Families

This paper examines four types of $D \times D$ structured matrices:

| Structure | Formulation | Parameter Count | Rank |
|------|---------|--------|-----|
| Dense | $\mathbf{W}$ | $D^2$ | $D$ |
| Low Rank | $\mathbf{L}\mathbf{R}^\top$ | $2Dr$ | $r$ |
| MLR (Multi-Level Low-Rank) | $\sum_{l=1}^L \bigoplus_{k=1}^{p_l} \mathbf{L}_{l,k}\mathbf{R}_{l,k}^\top$ | $2D\sum_l r_l$ | $\sum_l r_l p_l$ |
| BTT (Block Tensor Train) | $\mathbf{P}_L(\bigoplus \mathbf{L}_{k'}) \mathbf{P}_R(\bigoplus \mathbf{R}_k^\top)$ | $2D^{3/2}s$ | $D$ |

Key point: Both BTT and MLR can achieve high rank or even full rank while maintaining parameter counts and computational costs significantly smaller than $D^2$.

#### 2. Structured Bilinear Scoring Function (Resolving Low-Rank Bottleneck)

Replacing the standard attention scoring function $s(\mathbf{x}_j, \mathbf{x}_{j'}) = \mathbf{x}_j^\top \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{x}_{j'}$ with:

**MLR Scoring Function**:
$$s_{\text{MLR}}(\mathbf{x}_j, \mathbf{x}_{j'}) = \mathbf{x}_j^\top \left(\sum_{l=1}^L \bigoplus_{k=1}^{2^{l-1}} \mathbf{L}_{l,k}\mathbf{R}_{l,k}^\top \right) \mathbf{x}_{j'}$$

**BTT Scoring Function**:
$$s_{\text{BTT}}(\mathbf{x}_j, \mathbf{x}_{j'}) = \mathbf{x}_j^\top \left(\mathbf{P}_L \bigoplus_{k'=1}^{b} \mathbf{L}_{k'} \mathbf{P}_R \bigoplus_{k=1}^{c} \mathbf{R}_k^\top \right) \mathbf{x}_{j'}$$

BTT requires only $O(D^{3/2})$ parameters and FLOPs to achieve full rank $D$ when $a=b=c=d=\sqrt{D}$. By setting $p_l = 2^{l-1}$ and $\sum_l r_l = r$, MLR achieves a high rank while matching the efficiency of standard attention.

#### 3. MLR Attention: Distance-Dependent Computational Bias

This is the most innovative design of this work. Instead of structuring the underlying parameters of the scoring matrix, the MLR structure is imposed directly onto the **scoring matrix $\mathbf{S}$ itself**:

$$\mathbf{S} = \sum_{l=1}^L \bigoplus_{k=1}^{p_l} \mathbf{Q}_{l,k} \mathbf{K}_{l,k}^\top$$

where $\mathbf{Q}_{l,k}$ and $\mathbf{K}_{l,k}$ are the slices of the query/key matrices at level $l$, block $k$, respectively. Intuitive meaning:

- **Level 1** ($p_1 = 1$): A global low-rank component shared by all token pairs, with rank $r_1$.
- **Level 2** ($p_2 = 2$): The sequence is split into 2 blocks; token pairs within the same block obtain an additional scoring component with rank $r_2$.
- **Level $l$** ($p_l = 2^{l-1}$): The sequence is split into $2^{l-1}$ blocks; token pairs within the same block receive an additional component with rank $r_l$.
- The closer the token pairs, the higher the cumulative rank of their scoring function, leading to richer information interactions.

This hierarchical structure brings two major practical advantages:

**Computational Savings**: Forming $\mathbf{S}$ in standard attention requires $T^2 r$ FLOPs, whereas MLR attention requires only $T^2 \sum_{l=1}^L \frac{r_l}{2^{l-1}}$ FLOPs. For instance, with an 8-level MLR and $r_l = r/8$, it can save about $4\times$ computation.

**KV Cache Compression**: For autoregressive generation, level $l$ only needs to retain the key of the last block, $\mathbf{K}_{l,p_l}$, resulting in a total cache size of $T \sum_{l=1}^L \frac{r_l}{2^{l-1}}$, which also enables approximately $4\times$ compression.

#### 4. MLBTC Unified Framework

This paper proposes Multi-Level Block Tensor Contraction (MLBTC) as a unified framework:

$$\text{MLBTC}(\mathbf{L}, \mathbf{R}) = \sum_{l=1}^L \alpha_l \mathbf{P}_L \bigoplus_{k'=1}^{p'_l} \mathbf{L}_{l,k'} \mathbf{P}_R \bigoplus_{k=1}^{p_l} \mathbf{R}_{l,k}^\top$$

By setting different parameters, MLBTC can degenerate into:
- MLR ($\mathbf{P}_L = \mathbf{P}_R = \mathbf{I}$)
- BTT (retaining only one level)
- Monarch, Butterfly, Kronecker, Low Rank, etc.

This theoretical unification provides a foundation for exploring more structured attention variants in the future.

### Loss & Training

- **In-Context Regression Task**: Trained a 6-layer Transformer with 8 attention heads using Mean Squared Error (MSE) loss.
- **Language Modeling**: Trained on the FineWeb dataset using standard cross-entropy loss.
- **Time Series Forecasting**: Evaluated on standard long-range forecasting benchmarks using MSE loss.
- **Stability Trick**: The paper mentions feature learning techniques (Section 3.5) used to stabilize the training of structured matrices.
- All structured attention variants are compatible with Grouped-Query Attention (GQA).

## Key Experimental Results

### Main Results

**Experiment 1: In-Context Regression**

| Method | $d=128$ Performance | $d=64$ Performance | Parameter Efficiency |
|------|---------|---------|---------|
| Standard Attention (8-head, $r < d$) | High regression error | High regression error | Requires $r \geq d$ to learn |
| Bilinear BTT | Learned with small width | Learned with small width | Full rank, $O(D^{3/2})$ parameters |
| Bilinear MLR | Learned with small width | Learned with small width | High rank, equivalent parameters to standard |
| Standard Attention (1-head, full rank) | Learnable but parameters are large | Learnable but parameters are large | Requires large model width |

Key finding: BTT and MLR outperform standard attention under **any fixed computational budget** because they break the low-rank bottleneck.

**Experiment 2: Language Modeling Scaling Laws**

| Method | Scaling Law Performance | Comparison with Standard Attention | Comparison with SWA |
|------|---------|---------|---------|
| Standard Attention | Baseline | — | Better than SWA |
| Sliding Window Attention (SWA) | Worse than standard | Inferior to standard | —|
| MLR Attention | **Better than standard** | Superior scaling | Significantly outperforms SWA |

MLR attention demonstrates better scaling law trends in language modeling than both standard attention and SWA.

**Experiment 3: Long-Range Time Series Forecasting**

The hierarchical structure of MLR attention is naturally suited for multi-scale temporal dependency patterns in time series. The paper reports promising results.

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| $L=1$ (degenerate to standard attention) | Baseline | No distance bias |
| $L=2$ (2-level MLR) | Improved | Introduces local/global distinction |
| $L \leq 8$ (multi-level MLR) | Best | Hierarchical distance-dependent computation |
| Rank allocation $r_1 \vert r_2 \vert \cdots \vert r_L$ | Affects accuracy/efficiency trade-off | Allocation of ranks across levels is a key hyperparameter |
| Head dimension $r$ vs input dimension $d$ | Standard attention fails when $r < d$ | Validates the low-rank bottleneck |
| BTT $s=1$ vs $s=2$ | Larger $s$ is stronger but more expensive | Parameter-expressivity trade-off |

### Key Findings

1. **The low-rank bottleneck of standard attention is real**: In the in-context regression task with $d_{\text{input}} = 128$, the 8-head attention completely fails to learn when $r < d$, whereas BTT/MLR successfully learn in much smaller models.
2. **The scaling law of MLR attention is superior to standard attention**: It demonstrates for the first time the scaling advantage of structured attention in language modeling.
3. **Distance-dependent computational bias is "non-brittle"**: Unlike the hard truncation of SWA, MLR achieves a flexible local-global balance through hierarchical rank allocation without sacrificing accuracy.
4. **KV cache compression is free**: MLR attention naturally supports approximately $4\times$ KV cache compression during inference.

## Highlights & Insights

1. **Novel Perspective**: Revisiting attention from the perspective of bilinear transformations rather than query/key projections, bridging structured matrix theory with the design of attention mechanisms.
2. **Unified MLBTC Framework**: For the first time, structures like MLR, BTT, Monarch, Butterfly, and Kronecker matrices are unified under a single framework, providing a clear theoretical map for future research.
3. **Two Birds with One Stone**: The exact same mathematical tool (structured matrices) simultaneously addresses two seemingly unrelated problems: the low-rank bottleneck and the lack of locality bias.
4. **Compatibility with GQA**: MLR/BTT attention can be directly integrated into modern LLM architectures using Grouped-Query Attention.
5. **Practical Value**: KV cache compression and FLOP savings provide direct engineering benefits in LLM inference.

## Limitations & Future Work

1. **Insufficient Time Series Experiments**: The paper only mentions that MLR attention achieves "promising results" on long-range forecasting, lacking detailed comparisons with SOTA time series models such as PatchTST and iTransformer.
2. **Lack of Large-Scale LLM Validation**: The scaling law experiments are conducted on models of limited scale; validation on models with 7B+ parameters is still required.
3. **Undemonstrated MLBTC Framework**: Although the unified framework is proposed, its systematic validation is left for future work, with only the MLR and BTT special cases validated so far.
4. **Fixed Hierarchical Structure**: The geometric partitioning $p_l = 2^{l-1}$ may not be optimal; the paper mentions the possibility of dynamic partitioning (by paragraphs/documents) but does not implement it.
5. **FlashAttention Compatibility**: It remains unclear whether any structured scoring matrices can be integrated with highly efficient implementations like FlashAttention.
6. **Sequence Length Constraints**: Currently, the sequence length is required to be $T > \max_l p_l$, necessitating more flexible designs for extremely long sequence scenarios.

## Related Work & Insights

- **Monarch/Butterfly Matrices (Dao et al., 2022/2020)**: BTT generalizes Monarch matrices; this paper introduces them to attention scoring.
- **Linear Attention (Katharopoulos et al., 2020)**: Another line of research on efficient attention, which often comes at the cost of accuracy.
- **Longformer (Beltagy et al., 2020)**: An early attempt to combine sliding window and global attention, for which MLR attention serves as a more elegant alternative.
- **SSM/Mamba (Gu & Dao, 2024)**: SSMs inherently possess distance-decay bias; MLR attention achieves a similar effect within the Transformer framework.
- **In-Context Learning (Garg et al., 2022)**: A crucial benchmark task source for the experiments in this paper.
- **Insights for Time Series**: The multi-scale hierarchical structure of MLR attention is naturally aligned with the multi-frequency/multi-periodic patterns in time series, showing potential to replace standard attention in time series Transformers.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4.5 | The intersecting perspective of structured matrices × attention is highly novel |
| Theoretical Depth | 4.5 | The unified MLBTC framework makes solid theoretical contributions |
| Experimental Thoroughness | 3.5 | In-context regression and language modeling experiments are solid, but time series experiments are limited |
| Practical Value | 4.0 | KV cache compression and FLOP savings have direct engineering significance |
| Writing Quality | 4.5 | Clear structure and rigorous mathematical derivation |
| **Overall** | **4.2** | An elegant theoretical work with a novel perspective; the experiments could be further strengthened |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICML 2025\] TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation](transpl_vq-code_transition_matrices_for_pseudo-labeling_of_time_series_unsupervi.md)
- [\[ICLR 2026\] Understanding the Implicit Biases of Design Choices for Time Series Foundation Models](../../ICLR2026/time_series/understanding_the_implicit_biases_of_design_choices_for_time_series_foundation_m.md)
- [\[CVPR 2025\] FLAVC: Learned Video Compression with Feature Level Attention](../../CVPR2025/time_series/flavc_learned_video_compression_with_feature_level_attention.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](../../NeurIPS2025/time_series/less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)

</div>

<!-- RELATED:END -->
