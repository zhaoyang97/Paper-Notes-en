---
title: >-
  [Paper Note] Rectifying Magnitude Neglect in Linear Attention
description: >-
  [ICCV 2025][Signal & Communication][Linear Attention] This paper identifies that Linear Attention completely discards Query magnitude information, causing a significant deviation of attention score distributions from Softmax Attention. It proposes Magnitude-Aware Linear Attention (MALA), which restores magnitude awareness by introducing a scaling factor $\beta$ and an offset term $\gamma$, achieving comprehensive improvements over existing methods across classification…
tags:
  - "ICCV 2025"
  - "Signal & Communication"
  - "Linear Attention"
  - "Magnitude-Aware"
  - "Vision Transformer"
  - "Attention Score Distribution"
  - "Linear Complexity"
date: 2026-05-08
content_hash: 2fcab350583bf135
---

# Rectifying Magnitude Neglect in Linear Attention

**Conference**: ICCV 2025
**arXiv**: [2507.00698](https://arxiv.org/abs/2507.00698)  
**Code**: [https://github.com/qhfan/MALA](https://github.com/qhfan/MALA)  
**Area**: Signal Communication
**Keywords**: Linear Attention, Magnitude-Aware, Vision Transformer, Attention Score Distribution, Linear Complexity

## TL;DR

This paper identifies that Linear Attention completely discards Query magnitude information, causing a significant deviation of attention score distributions from Softmax Attention. It proposes Magnitude-Aware Linear Attention (MALA), which restores magnitude awareness by introducing a scaling factor $\beta$ and an offset term $\gamma$, achieving comprehensive improvements over existing methods across classification, detection, segmentation, NLP, speech recognition, and image generation tasks.

## Background & Motivation

- **Background**: The quadratic complexity $O(N^2)$ of Softmax Attention limits Vision Transformers in high-resolution visual tasks. Linear Attention reduces complexity to $O(N)$ via kernel function approximation, but suffers from notable performance degradation.
- **Limitations of Prior Work**: Existing improvements (e.g., EfficientViT adding convolutions for local compensation, Flatten Transformer's focused linear attention) are largely heuristic patch-up strategies.
- **Key Challenge**: This paper analyzes the fundamental cause of the performance gap at the **mathematical formulation level**: in the Linear Attention formulation, the Query magnitude $\|\phi(Q_i)\|$ cancels out in the numerator and denominator (Eq. 4), leaving only the directional component $\vec{\alpha_i}$. This means:
    - Softmax Attention: increasing Query magnitude → sharper attention distribution (high-score Keys receive more attention)
    - Linear Attention: attention distribution remains unchanged regardless of Query magnitude variation
- **Goal**: This finding explains the long-standing issue of overly smooth attention scores and weak local perception in Linear Attention.

## Method

### Overall Architecture

MALA modifies the normalization scheme of Linear Attention: replacing division-based normalization with additive normalization, and introducing a scaling factor $\beta$ and an offset term $\gamma$ that are functions of the $\phi(Q_i)$ magnitude, enabling attention scores to dynamically adapt to Query magnitude.

### Key Designs

1. **Formal Proof of Magnitude Neglect**:

    - Substituting $\phi(Q_i) = \|\phi(Q_i)\| \vec{\alpha_i}$ into the Linear Attention formula shows the magnitude term cancels in numerator and denominator.
    - Empirical validation: replacing $Q$ with $Q/\|Q\|$ in DeiT-T's Softmax Attention (discarding magnitude) drops accuracy from 72.2% to 70.0%, approaching Linear Attention's 69.8%.
    - Attention score visualization also converges to the smooth distribution characteristic of Linear Attention.

2. **MALA Formulation**:

    - Attention score: $\text{Attn}(Q_i, K_j) = \beta \cdot \phi(Q_i)\phi(K_j)^T - \gamma$
    - Scaling factor: $\beta = 1 + \frac{1}{\phi(Q_i)\sum_m \phi(K_m)^T}$ (negatively correlated with Query magnitude)
    - Offset term: $\gamma = \frac{\phi(Q_i)\sum_m \phi(K_m)^T}{N}$ (positively correlated with Query magnitude)
    - Normalization preserved: $\sum_j \text{Attn}(Q_i, K_j) = 1$
    - **Core property**: when $\|\phi(Q_i)\|$ increases by a factor of $a$, the attention ratio between high-score and low-score Keys increases ($p_m > p$), consistent with the behavior of Softmax Attention.

3. **Difference in Magnitude Growth Rates (Key Insight)**:

    - In Softmax Attention, the ratio $p$ grows **exponentially** with the scaling factor $a$ ($p^a$) → overly sharp attention.
    - In MALA, the ratio $p$ grows **fractionally** with $a$ (more moderate) → more balanced distribution.
    - Visualization confirms: Softmax focuses too locally, Linear Attention is overly smooth, and MALA achieves a well-balanced distribution.
    - **Linear complexity preserved**: $Y_i = \beta \phi(Q_i)\sum_j \phi(K_j)^T V_j - \gamma \sum_j V_j$, where $K^TV$ can still be computed first and then interacted with $Q$.

### Loss & Training

- A MAViT (Magnitude-Aware Vision Transformer) model family T/S/B/L is constructed.
- Image classification: trained from scratch on ImageNet-1K for 300 epochs, with maximum stochastic depth rates of 0.1/0.15/0.4/0.55.
- Detection/Segmentation: standard COCO/ADE20K configurations using RetinaNet/Mask R-CNN/Cascade Mask R-CNN/SemanticFPN/UperNet.

## Key Experimental Results

### Main Results

ImageNet-1K classification accuracy comparison (key scales):

| Model | Type | Params(M) | FLOPs(G) | Top-1(%) |
|-------|------|-----------|----------|----------|
| RMT-S | Trans | 27 | 4.5 | 84.1 |
| SECViT-S | Trans | 27 | 4.6 | 84.3 |
| RAVLT-S | Linear | 26 | 4.6 | 84.4 |
| **MAViT-S** | **Linear** | **27** | **4.6** | **84.7** |
| RMT-B | Trans | 54 | 9.7 | 85.0 |
| RAVLT-B | Linear | 48 | 9.9 | 85.5 |
| **MAViT-B** | **Linear** | **50** | **9.9** | **85.7** |
| RMT-L | Trans | 95 | 18.2 | 85.5 |
| **MAViT-L** | **Linear** | **98** | **16.1** | **86.0** |

COCO detection (Cascade Mask R-CNN 3×+MS): MAViT-B achieves 55.5 $AP^b$ / 48.0 $AP^m$, surpassing the larger CSwin-B.

### Ablation Study

Comparison of Linear Attention variants (DeiT-T/Swin-T/Swin-S settings, replacing only the attention mechanism):

| Linear Attention Type | DeiT-T | Swin-T | Swin-S |
|----------------------|--------|--------|--------|
| Hydra Attn | 68.3 | 80.7 | — |
| Enhanced Linear Attn | 72.9 | 81.8 | — |
| Focused Linear Attn | 74.1 | 82.1 | 83.5 |
| InLine Attn | 74.5 | 82.4 | 83.6 |
| **MALA** | **75.1** | **83.7** | **85.3** |

Ablation of $\beta$ and $\gamma$ (MAViT-T): removing $\beta$ drops accuracy to 52.3%; removing $\gamma$ causes NaN; replacing both with learnable parameters drops accuracy to 71.7%.

Kernel function insensitivity: ELU+1, ReLU, and exp are nearly equivalent (82.9 vs. 82.8 vs. 82.9).

### Key Findings

- MALA outperforms Softmax Attention on **all tested tasks** while maintaining linear complexity.
- NLP (0.3B model / 15B tokens): MALA is competitive with Transformer and Mamba on LMB/PIQA/Hella/Wino benchmarks.
- Speech recognition (Conformer replacement): WER improves from Softmax's 2.7/6.3 to MALA's 2.4/5.3.
- Image generation (DiT framework): FID improves from 68.40 to 49.62, with throughput of 5.6 imgs/s (fastest).
- Clear efficiency advantage in high-resolution inference: MAViT significantly outperforms Softmax-based models in 512×2048 semantic segmentation.
- **No negative or zero attention scores observed** in experiments (theoretically possible but does not occur in practice).

## Highlights & Insights

- **Precise analytical starting point**: Rather than heuristically adding convolutions for compensation, this paper mathematically identifies the fundamental flaw of Linear Attention (magnitude cancellation), and the solution directly addresses the identified problem.
- **Minimal yet effective design of $\beta$ and $\gamma$**: Only two analytically derived hyperparameters (non-learnable); ablations demonstrate their indispensability.
- The difference in growth rates is a profound insight: the exponential growth in Softmax leads to overly sharp attention, while the fractional growth in MALA yields a more reasonable distribution.
- Cross-domain validation (vision/NLP/speech/generation) demonstrates that this is a fundamental improvement rather than a task-specific trick.

## Limitations & Future Work

- The introduction of $\beta$ and $\gamma$ incurs additional per-token scalar computation, which increases constant overhead without affecting asymptotic complexity.
- The theoretical analysis of negative attention scores is insufficiently rigorous (only claimed to be absent empirically).
- No thorough comparison with Mamba/SSM-based methods at large-scale NLP settings.
- Performance of Linear Attention on ultra-long sequences (e.g., video, genomics) remains unexplored.

## Related Work & Insights

- Flatten Transformer (ICCV 2023) and InLine Attention (NeurIPS 2024) are the most direct comparison baselines.
- MILA (NeurIPS 2024) is inspired by Mamba to redesign the macro-architecture of Linear Attention, but does not address the magnitude problem.
- RMT (CVPR 2024) is prior work from the same group using a Retentive mechanism; MALA provides a more fundamental improvement.
- The analysis of Softmax being overly sharp versus Linear Attention being overly smooth offers broadly applicable guidance for designing new attention mechanisms.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Both the problem identification and the solution are exceptionally elegant, grounded in mathematical fundamentals.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across 7 tasks with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, though some formula typesetting could be improved.
- **Value**: ⭐⭐⭐⭐⭐ A fundamental improvement to Linear Attention with broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Breaking the Low-Rank Dilemma of Linear Attention](../../CVPR2025/signal_comm/breaking_the_low-rank_dilemma_of_linear_attention.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](../../ICML2025/signal_comm/fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [\[CVPR 2025\] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](../../CVPR2025/signal_comm/abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[ICCV 2025\] Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors](generalizable_non-line-of-sight_imaging_with_learnable_physical_priors.md)

</div>

<!-- RELATED:END -->
