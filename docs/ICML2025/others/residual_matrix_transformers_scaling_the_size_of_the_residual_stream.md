---
title: >-
  [Paper Note] Residual Matrix Transformers: Scaling the Size of the Residual Stream
description: >-
  [ICML2025][Residual Stream] Replaces the residual stream vector of the Transformer with an outer-product memory matrix, allowing the size of the residual stream to be scaled independently of the model parameter count and FLOPS, saving 58% FLOPS, 25% parameters, and 41% training tokens for the same loss.
tags:
  - "ICML2025"
  - "Residual Stream"
  - "Outer Product Memory"
  - "Parameter Efficiency"
  - "Computational Efficiency"
  - "Variance Propagation"
date: 2026-05-08
content_hash: 4432034215d38235
---

# Residual Matrix Transformers: Scaling the Size of the Residual Stream

**Conference**: ICML2025  
**arXiv**: [2506.22696](https://arxiv.org/abs/2506.22696)  
**Code**: [bmac3/residual-matrix-transformer](https://github.com/bmac3/residual-matrix-transformer)  
**Area**: Transformer Architecture  
**Keywords**: Residual Stream, Outer Product Memory, Parameter Efficiency, Computational Efficiency, Variance Propagation

## TL;DR
Replaces the residual stream vector of the Transformer with an outer-product memory matrix, allowing the size of the residual stream to be scaled independently of the model parameter count and FLOPS, saving 58% FLOPS, 25% parameters, and 41% training tokens for the same loss.

## Background & Motivation

- **The residual stream serves as the core communication channel of the Transformer**: Elhage et al. (2021) noted that the residual stream acts as a "memory bus" where layers read and write features. The dimension $D$ of the residual stream determines the number of features that can be stored (i.e., its bandwidth).
- **Scaling the residual stream is highly expensive**: In a standard Transformer, increasing $D$ linearly increases the size of all parameter matrices, leading to a synchronous growth in parameter count and FLOPS. For example, doubling the residual stream size doubles parameters and approximately doubles the FLOPS.
- **Analogy to the MoE approach**: Mixture of Experts (Fedus et al., 2022) enables scaling model size independently of computation (yielding a 7x efficiency gain) via a "sparse parameter axis". This paper proposes a similar decoupled scaling along a new axis: the "residual stream size".
- **Core Problem**: Can the capacity of the residual stream be expanded to improve performance without increasing computation and parameters?

## Method

### Core Idea: Replacing the Residual Stream Vector with an Outer-Product Memory Matrix

In a standard Transformer, the residual stream of each token is a $D$-dimensional vector $\mathbf{x} \in \mathbb{R}^D$. RMT replaces it with an **outer-product memory matrix** $\mathbf{M} \in \mathbb{R}^{D_k \times D_v}$, where the "size" of the residual stream is $D_k \times D_v$, and $D_v$ corresponds to the attention head dimension $D_h$ of the standard Transformer.

The construction and retrieval of the outer-product memory matrix follow classical associative memory (Kohonen, 1972; Anderson, 1972):

$$\mathbf{M} = \text{Norm}\left(\sum_{p=1}^{N} \mathbf{q}^{(p)} \otimes \mathbf{x}^{(p)}\right)$$

During retrieval, data vectors are retrieved via tensor contraction with a key vector:

$$\mathbf{x}^{(r)} \approx \mathbf{q}^{(r)} \cdot_1 \mathbf{M}$$

### Modification of RMT Layers

**Embedding Layer**: The initial residual matrix is constructed using the sum of outer products of $R$ key vectors $\mathbf{w}_E^{(h)} \in \mathbb{R}^{D_k}$ and their corresponding embedding matrices $\mathbf{W}_E^{(h)} \in \mathbb{R}^{D_v \times V}$:

$$\text{E}(\mathbf{S}) = \sum_{h=1}^{R} \mathbf{w}_E^{(h)} \otimes \mathbf{W}_E^{(h)} \mathbf{S}$$

**Attention Layer**: The QKV projection matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D_h \times D}$ in standard Transformers are replaced with key vectors $\mathbf{r}_Q, \mathbf{r}_K, \mathbf{r}_V \in \mathbb{R}^{D_k}$. Features are retrieved from the residual matrix via tensor contraction:

$$\mathbf{Q}^{(h)} = \mathbf{r}_Q^{(h)} \cdot_1 \mathbf{X}, \quad \mathbf{K}^{(h)} = \mathbf{r}_K^{(h)} \cdot_1 \mathbf{X}, \quad \mathbf{V}^{(h)} = \mathbf{r}_V^{(h)} \cdot_1 \mathbf{X}$$

The output is written back to the residual matrix using the outer product of the key vector $\mathbf{w}_O^{(h)}$:

$$\text{MHA}(\mathbf{X}) = \sum_{h=1}^{R} \mathbf{w}_O^{(h)} \otimes \text{SHA}(\mathbf{Q}^{(h)}, \mathbf{K}^{(h)}, \mathbf{V}^{(h)})$$

**FeedForward Layer**: The core FFN operation $\tilde{\text{FF}}(\mathbf{X}) = \mathbf{W}_2 \text{GeLU}(\mathbf{W}_1 \mathbf{X})$ remains unchanged (as the FFN weights store factual knowledge), with only key vector adapters added at the input/output ends for reading and writing back.

**Unembedding Layer**: Data is retrieved using key vectors and then multiplied by unembedding weights to obtain logits.

### Key Designs

| Component | Transformer Parameters | RMT Parameters | Change |
|------|-----------------|---------|------|
| QKV Projection | $\mathbf{W} \in \mathbb{R}^{D_h \times D}$ | $\mathbf{r} \in \mathbb{R}^{D_k}$ (key vector) | Matrix $\rightarrow$ Vector |
| Output Projection | $\mathbf{W}_O \in \mathbb{R}^{D \times D_h}$ | $\mathbf{w}_O \in \mathbb{R}^{D_k}$ (key vector) | Matrix $\rightarrow$ Vector |
| FFN Weights | $\mathbf{W}_1, \mathbf{W}_2$ | Unchanged + adapter | Retained to store knowledge |

## Key Experimental Results

### Efficiency Comparison (Core Results)

| Metric | RMT vs. Transformer | Description |
|------|---------------------|------|
| FLOPS Savings | **58%** | Computation required to reach the same loss |
| Parameter Savings | **25%** | Parameter count required to reach the same loss |
| Training Token Savings | **41%** | Data volume required to reach the same loss |
| Downstream Evaluation | **Outperforms Transformer** | Specific evaluation tasks are not detailed |

### Resource Scaling Characteristics

- Transformer: Residual stream size ×2 → parameters ×2, FLOPS ≈ ×2
- RMT: Residual stream size ×2 → parameters and FLOPS **almost unchanged** (approximate constant)
- This is because RMT replaces the $D_h \times D$ matrix with $D_k$-dimensional key vectors.

### Theoretical Analysis of Variance Propagation (GPT2-medium Configuration)

| Layer | Operation | Model | Forward Variance Ratio $\sigma^2_{out}/\sigma^2_{in}$ | Backward Variance Ratio $\sigma^2_{g_{in}}/\sigma^2_{g_{out}}$ |
|----|------|------|----|----|
| Attn | Storage | RMT | 0.4 | 1.6 |
| Attn | Storage | Transformer | 1.0 | 1.0 |
| Attn | Retrieval | **RMT** | **1.14** | 0.86 |
| Attn | Retrieval | Transformer | 0.5 | 1.5 |
| FF | Storage/Retrieval | **RMT** | **1.0** | **1.0** |
| FF | Storage/Retrieval | Transformer | 0.4/1.6 | 1.6/0.4 |

**Interpretation**: In the FFN layer, both the forward and backward variance ratios of RMT are at the ideal value of 1.0, whereas the Transformer suffers from significant variance decay/amplification issues. RMT is also closer to 1.0 in Attention retrieval.

## Highlights & Insights

1. **New Scaling Dimension**: Following the sparse parameter axis of MoE, RMT introduces the "residual stream size" as a new scaling axis independent of computation and parameters, presenting a simple and powerful concept.
2. **Return of Classical Associative Memory**: Combining the 1970s outer-product memory (Kohonen/Anderson) with modern Transformers, replacing matrix multiplication with vector retrieval, represents an elegant parameter dimensionality reduction.
3. **Design Insight on Retaining FFN Unchanged**: Based on the understanding that FFNs store factual knowledge (Geva et al., 2021; Meng et al., 2022), selectively retaining FFN weights instead of replacing them entirely reflects a deep understanding of the Transformer's internal mechanism.
4. **Dual Validation via Theory and Experiment**: The paper provides both closed-form analysis of variance propagation (Table 1-2) and empirical results on actual training efficiency.

## Limitations & Future Work

1. **Small Experimental Scale**: The theoretical analysis is based on GPT2-medium (~350M), and scalability has not yet been verified at the 7B+ scale.
2. **Only Verified on Language Models**: Lack of experiments on other modalities such as multimodal or vision tasks.
3. **Storage Overhead of the Residual Matrix**: Although parameters and FLOPS are reduced, the intermediate activation for each token is transformed from a $D$-dimensional vector to a $D_k \times D_v$ matrix, which may increase memory consumption during inference.
4. **Compatibility with Existing Optimization Techniques**: The compatibility with mainstream inference optimizations such as GQA, FlashAttention, and KV cache compression is not discussed.
5. **Cache File Truncation**: Detailed results in the experimental section (§4) of the paper are not fully covered; the 58%/25%/41% figures are sourced from the abstract.

## Related Work & Insights

- **Mixture of Experts** (Fedus et al., 2022): Inspired the idea of "scaling a specific axis independently."
- **Outer-Product Associative Memory** (Kohonen, 1972; Anderson, 1972): The theoretical foundation of the core mechanism in RMT.
- **Residual Stream as a Memory Bus** (Elhage et al., 2021): Motivated the research on residual stream capacity.
- **FFNs Store Factual Knowledge** (Geva et al., 2021; Meng et al., 2022): Guided the design of retaining full weights in the FFN layers.

## Rating
- Novelty: ⭐⭐⭐⭐ — Replacing the residual stream with outer-product memory is a novel and theoretically grounded idea.
- Experimental Thoroughness: ⭐⭐⭐ — The efficiency gains are significant, but the scale is small, lacking large-scale model validation.
- Writing Quality: ⭐⭐⭐⭐ — The theoretical derivation is clear, and the architecture description is detailed.
- Value: ⭐⭐⭐⭐ — If validated at scale, this will be an important direction for Transformer architecture improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](../../ACL2025/others/rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)
- [\[ICML 2025\] Softmax is not Enough (for Sharp Size Generalisation)](softmax_is_not_enough_for_sharp_size_generalisation.md)
- [\[ICCV 2025\] HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding](../../ICCV2025/others/hytip_hybrid_temporal_information_propagation_for_masked_conditional_residual_vi.md)

</div>

<!-- RELATED:END -->
