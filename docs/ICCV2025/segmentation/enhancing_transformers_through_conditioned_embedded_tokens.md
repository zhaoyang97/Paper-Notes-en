---
title: >-
  [Paper Note] Enhancing Transformers Through Conditioned Embedded Tokens
description: >-
  [ICCV 2025][Segmentation][Transformer] This paper identifies an inherent ill-conditioning problem in the self-attention matrices of Transformers. Through theoretical analysis…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Transformer"
  - "condition number"
  - "self-attention"
  - "embedded tokens"
  - "optimization stability"
date: 2026-05-08
content_hash: 1cfcd22a920e4941
---

# Enhancing Transformers Through Conditioned Embedded Tokens

**Conference**: ICCV 2025
**arXiv**: [2505.12789](https://arxiv.org/abs/2505.12789)  
**Code**: Unavailable  
**Area**: Image Segmentation / General Transformer Improvement
**Keywords**: Transformer, condition number, self-attention, embedded tokens, optimization stability

## TL;DR

This paper identifies an inherent ill-conditioning problem in the self-attention matrices of Transformers. Through theoretical analysis, it establishes a direct relationship between the condition number of the self-attention matrix and that of the embedded token matrix, and proposes Conditioned Embedded Tokens — an SVD-based correction term applied to the embedding matrix — achieving consistent performance improvements across image classification, object detection, instance segmentation, and NLP tasks.

## Background & Motivation

### Problem Definition

The core of Transformers is the self-attention mechanism, which models global dependencies via $\mathbf{A}(X) = \text{softmax}(XW_QW_K^TX^T)XW_V$. The **condition number** of a matrix (the ratio of its largest to smallest singular value) is a critical indicator of optimization difficulty: a larger condition number leads to more unstable gradient optimization.

### Limitations of Prior Work

1. Condition number studies in feed-forward networks are relatively mature (weight matrix preconditioning, NTK condition improvement), but the **condition number problem in self-attention** has received almost no attention.
2. Existing optimization improvements for Transformers (e.g., skip connections, multi-head attention) improve conditioning only indirectly, lacking a systematic theoretical framework.
3. In practice, the condition number of the embedded token matrix is often extremely large, causing gradient instability.

### Core Idea

The upper bound on the condition number of the self-attention matrix is directly related to the condition number of the embedded token matrix $X$ (scaling as $\kappa(X)^3$ for linear attention). By applying SVD decomposition to $X$ and adding a correction term $C$ such that $\kappa(X+C) \leq 2$, the ill-conditioning of self-attention can be substantially reduced.

## Method

### Overall Architecture

A correction matrix $C \in \mathbb{R}^{N \times d}$ is added to the embedded token matrix $X = [Ex_1 \cdots Ex_N]^T \in \mathbb{R}^{N \times d}$ at the first Transformer layer, such that the condition number of $X+C$ is greatly reduced. The correction term $C$ is computed via SVD decomposition of $X$, and the corrected matrix is fed into the first Transformer layer.

### Key Designs

#### 1. Condition Number Analysis of Self-Attention

- **Function**: Establishes theoretical upper bounds on the condition number of the self-attention matrix.
- **Core Result** (Proposition 4.2):
    - Linear attention: $\kappa(\mathbf{LA}(X)) \leq \kappa(W_Q) \cdot \kappa(W_K) \cdot \kappa(W_V) \cdot \kappa(X)^3$
    - Softmax attention: $\kappa(\mathbf{A}(X)) \leq \kappa(\text{softmax}(XW_QW_K^TX^T)) \cdot \kappa(X) \cdot \kappa(W_V)$
- **Design Motivation**: $\kappa(X)$ is typically very large in practice and constitutes the primary bottleneck; reducing $\kappa(X)$ simultaneously reduces the condition number of the entire self-attention operation.

#### 2. Conditioned Embedded Tokens

- **Function**: Constructs a correction matrix $C$ such that $\kappa(X+C) \leq 2$.
- **Core Theorem** (Theorem 4.4): For any embedding matrix with $\kappa(X) > 2$, there exists a $C$ such that $\kappa(X+C) \leq 2$.
- **Mechanism**: SVD decomposition of $X$ is performed, and the optimal correction term is constructed by adjusting the singular values. The correction is deterministic and introduces no additional learnable parameters.
- **Design Motivation**: Even though the upper bound is an approximation, experiments demonstrate a strong correlation between reduced condition numbers and improved performance.

#### 3. Cross-Layer Propagation Effect

- **Function**: Validates that the conditioning improvement at the first layer propagates to subsequent layers.
- **Core Finding**: Although the theoretical guarantee applies only to the first layer, experiments show that the average self-attention condition number is significantly reduced across all layers.
- **Design Motivation**: The output of the first layer serves as input to the second, giving the conditioning improvement a cascading effect.

### Loss & Training

No additional loss functions are introduced. The method serves as a drop-in replacement for the existing embedding layer, with all original training configurations unchanged.

## Key Experimental Results

### Main Results

**ImageNet-1k Image Classification (Top-1 %)**

| Model | Baseline | +Conditioned | Gain |
|-------|----------|-------------|------|
| ViT-Base | 80.3 | 81.3 | +1.0 |
| DeiT-Base | 81.6 | 82.5 | +0.9 |
| Swin-Base | 83.1 | 83.9 | +0.8 |
| XCiT-Medium | 82.2 | 82.9 | +0.7 |
| DaViT-Base | 83.6 | 84.6 | +1.0 |

**COCO Object Detection and Instance Segmentation (Mask R-CNN, AP)**

| Model | AP_box | AP50_box | AP_mask | AP50_mask |
|-------|--------|---------|---------|----------|
| XCiT-S Baseline | 44.9 | 66.1 | 40.1 | 63.1 |
| XCiT-S +Cond. | **45.7** | **66.6** | **40.4** | **63.5** |
| XCiT-M Baseline | 45.7 | 66.8 | 40.8 | 63.6 |
| XCiT-M +Cond. | **46.2** | **67.4** | **41.4** | **63.8** |

**GLUE Benchmark (Crammed BERT, Accuracy)**

| Task | MNLI | SST-2 | RTE | QNLI | QQP | MRPC | CoLA | GLUE Avg. |
|------|------|-------|-----|------|-----|------|------|-----------|
| Baseline | 83.8 | 92.3 | 55.1 | 90.1 | 87.3 | 85.0 | 48.9 | 78.6 |
| +Cond. | **84.2** | **92.5** | **55.6** | **91.1** | **87.4** | **86.3** | **53.7** | **79.7** |

### Ablation Study

**Condition Number Comparison Across Transformer Architectures (ViT-B, averaged over full training)**

| Metric | Baseline | +Conditioned |
|--------|----------|-------------|
| Embedded token $\kappa(X)$ | ~$10^3$ order | ~$10^1$ order |
| Layer-1 attention $\kappa$ | Significantly higher | Significantly reduced |
| All-layer attention $\kappa$ (avg.) | Higher | Significantly reduced |

**GPT-2 Validation Loss (TinyStories)**

| Model | Val. Loss↓ |
|-------|-----------|
| GPT-2 Baseline | 2.41 |
| GPT-2 +Conditioned | **2.36** |

**Nyströmformer Long-Sequence Benchmark (LRA, Accuracy %)**

| Task | ListOps | Text | Retrieval | Image | Pathfinder |
|------|---------|------|-----------|-------|------------|
| Baseline | 37.1 | 63.8 | 79.8 | 39.9 | 72.9 |
| +Cond. | **37.9** | **64.9** | **80.9** | **40.1** | **73.3** |

### Key Findings

- Conditioning improvements are consistently effective across all tested architectures (ViT, DeiT, Swin, XCiT, DaViT, BERT, GPT-2, Nyströmformer).
- The method benefits not only standard self-attention but also advanced attention mechanisms including shifted windows, cross-covariance, and Nyström approximations.
- The conditioning improvement at the first layer cascades to all subsequent layers.
- The method introduces zero additional parameters and no additional loss terms, functioning as a pure drop-in replacement.

## Highlights & Insights

1. **Effective integration of theory and practice**: The condition number analysis provides clear theoretical motivation; although a complete proof from condition number to optimization convergence is yet to be established, experimental results consistently support the conclusions.
2. **Strong generality**: The same method is effective across CV, NLP, and long-sequence modeling, and can be directly embedded into various modern Transformer architectures.
3. **Simple implementation**: Only SVD decomposition of the embedding matrix and addition of the correction term are required, with no modifications to training configurations or introduction of hyperparameters.
4. **Reveals a previously overlooked optimization bottleneck**: The condition number of embedded tokens frequently reaches the order of $10^3$, a problem that has received almost no prior attention.

## Limitations & Future Work

1. **Missing final theoretical step**: The complete chain from condition number improvement → NTK improvement → accelerated optimization convergence has not been formally proven.
2. **The condition number upper bound for softmax attention requires additional assumptions** (the conditional assumption in Eq. 14).
3. **SVD computational overhead**: SVD decomposition of the embedding matrix is required at every forward pass, which may introduce non-trivial overhead in large-scale models.
4. **Theoretical guarantees limited to the first layer**: Although experiments show benefits across multiple layers, theoretical analysis for subsequent layers is absent.
5. **Effects on very deep Transformers (e.g., LLMs) have not been sufficiently validated.**

## Related Work & Insights

- Weight conditioning [Saratchandran et al., 2025] applies preconditioning to feed-forward network weight matrices; the present work extends this idea to the attention mechanism.
- NTK condition number analysis [Liu et al., 2022] establishes the relationship between condition numbers and gradient descent convergence, but is restricted to feed-forward networks.
- Skip connections have been shown to improve the condition number of attention blocks [Ji et al., 2025]; the proposed method is complementary to this line of work.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic analysis of condition numbers in self-attention with a theoretically motivated correction method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 5 CV models, 2 language models, 1 long-sequence model, and 4 tasks; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and experiments are well organized.
- **Value**: ⭐⭐⭐⭐ — High practical value as a drop-in improvement, though SVD overhead may limit applicability to large models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [\[ICCV 2025\] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP](know_no_better_a_data-driven_approach_for_enhancing_negation_awareness_in_clip.md)

</div>

<!-- RELATED:END -->
