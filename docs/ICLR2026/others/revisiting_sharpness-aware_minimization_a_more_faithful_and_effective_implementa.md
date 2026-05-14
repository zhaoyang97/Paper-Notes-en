---
title: >-
  [Paper Note] Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation
description: >-
  This paper proposes a new intuitive interpretation of SAM's underlying mechanism — that the gradient at the perturbed point approximates the direction toward the local maximum — and reveals its imprecision as well as th…
tags:

date: 2026-05-08
content_hash: fe724d1fcbbab6b6
---

# Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2603.10048](https://arxiv.org/abs/2603.10048)
- **Code**: [https://github.com/Cccjl219/XSAM](https://github.com/Cccjl219/XSAM)
- **Area**: Others
- **Keywords**: sharpness-aware minimization, SAM, optimization, generalization, flat minima

## TL;DR
This paper proposes a new intuitive interpretation of SAM's underlying mechanism — that the gradient at the perturbed point approximates the direction toward the local maximum — and reveals its imprecision as well as the multi-step degradation problem. It then introduces XSAM, which achieves more faithful and effective sharpness-aware minimization by explicitly estimating the direction of the maximum.

## Background & Motivation
- **SAM** promotes flat minima and better generalization by minimizing the maximum loss within a $\rho$-neighborhood, but its practical implementation applies the gradient computed at the perturbed point back to the current parameters — the reason why this "misaligned gradient" works has lacked an intuitive explanation.
- **Common misconception**: The gradient computed at the estimated maximum point does not directly minimize the maximum loss within the neighborhood — the key lies in the discrepancy between where the gradient is computed and where it is applied.
- **Puzzle with multi-step SAM**: More steps should theoretically yield a better approximation of the maximum, yet empirically multi-step SAM performs worse rather than better.

## Method

### Core Insight (Discovered via Visualization)
1. **Better approximation** (Figure 1a): The single-step perturbed gradient $g_1@\vartheta_0$ approximates the direction from the current parameters to the neighborhood maximum better than the local gradient $g_0$.
2. **Imprecision**: The approximation is often inaccurate and varies considerably throughout training.
3. **Multi-step degradation** (Figure 1b): $g_k@\vartheta_0$ may point toward the maximum direction less accurately than $g_1@\vartheta_0$.

### Theoretical Confirmation
**Proposition 1**: Under second-order approximation, for sufficiently large $\rho_m$:
1. $L(\vartheta_0 + \rho_m \frac{g_1}{\|g_1\|}) > L(\vartheta_0 + \rho_m \frac{g_0}{\|g_0\|})$ (the SAM gradient does approximate the maximum direction better)
2. There exists $\alpha$ such that $g_\alpha = \alpha g_1 + (1-\alpha) g_0$ outperforms $g_1$ (the SAM gradient is still not optimal)

### XSAM Method
XSAM explicitly searches for the maximum direction within the 2D hyperplane spanned by $v_0$ (the direction from the current parameters to the perturbed point) and $v_1$ (the direction of the perturbed gradient):

$$v_0 = \frac{\vartheta_k - \vartheta_0}{\|\vartheta_k - \vartheta_0\|}, \quad v_1 = \frac{g_k}{\|g_k\|}$$

Candidate directions are generated via spherical linear interpolation:
$$v(\alpha) = \frac{\sin((1-\alpha)\psi)}{\sin(\psi)} v_0 + \frac{\sin(\alpha\psi)}{\sin(\psi)} v_1$$

The optimal $\alpha^*$ is found explicitly:
$$\alpha^* = \arg\max_{\alpha \in [0, a]} L(\vartheta_0 + \rho_m \cdot v(\alpha))$$

Parameter update:
$$\theta_{t+1} = \theta_t - \eta_t \cdot v(\alpha^*) \cdot \|g_k\|$$

### Key Design Advantages
1. The search space includes the known highest-loss point (the direction pointed to by $v_1$).
2. Both single-step and multi-step settings are handled in a unified manner.
3. $\alpha^*$ changes slowly during training (Figure 2), requiring updates only once per epoch → negligible computational overhead.

### Computational Overhead
Each $\alpha^*$ update requires 20–40 forward passes, performed at the first iteration of each epoch, resulting in less than 3% additional computation overall.

## Key Experimental Results

### Main Results: Single-Step Setting on Classification Tasks

| Dataset/Model | SGD | SAM | GSAM | WSAM | **XSAM** |
|-----------|-----|-----|------|------|---------|
| CIFAR-10/ResNet-18 | 95.3 | 96.0 | 96.0 | 96.1 | **96.3** |
| CIFAR-100/ResNet-18 | 78.0 | 79.5 | 79.8 | 79.8 | **80.3** |
| CIFAR-100/DenseNet-121 | 79.5 | 81.0 | 81.2 | 81.2 | **81.6** |
| Tiny-ImageNet/ResNet-18 | 64.5 | 66.0 | 66.2 | 66.3 | **66.8** |

> XSAM consistently outperforms SAM and its variants across all model–dataset combinations.

### Ablation Study: Multi-Step Setting

| Method | 1-step | 2-step | 5-step | 10-step |
|-----|-----|-----|-----|------|
| SAM | 79.5 | 79.2 | 78.8 | 78.3 |
| **XSAM** | **80.3** | **80.5** | **80.6** | **80.7** |

> SAM performance degrades as the number of steps increases, while XSAM improves consistently — validating both the multi-step degradation phenomenon and XSAM's remedy.

### Training Time Comparison (hours / 200 epochs)

| Model/Dataset | SAM | XSAM | Extra Overhead |
|-----------|-----|------|---------|
| VGG-11/CIFAR-10 | 0.93 | 0.96 | +3.2% |
| ResNet-18/CIFAR-100 | 2.40 | 2.43 | +1.3% |
| DenseNet-121/CIFAR-100 | 8.05 | 8.07 | +0.2% |

> XSAM introduces negligible additional computation time.

### Key Findings
1. The SAM gradient approximates the maximum direction better than the SGD gradient, but remains imprecise.
2. Multi-step SAM degrades because the directional information of $g_k$ becomes distorted far from $\vartheta_0$.
3. $\alpha^*$ is stable throughout training, making epoch-wise updates sufficient.
4. Combining XSAM with ASAM can yield further performance gains.

## Highlights & Insights
- **Filling an intuitive gap**: This work is the first to provide an intuitive and visual explanation of why SAM's "misaligned gradient" is effective.
- **Resolving the multi-step puzzle**: It elegantly explains a phenomenon that has puzzled the community — why more ascent steps do not lead to better performance.
- **Minimal overhead improvement**: Only 20–40 forward passes per epoch are needed, with less than 3% additional cost.
- **Unified framework**: A single improvement scheme that covers both single-step and multi-step SAM.

## Limitations & Future Work
- The search is confined to a 2D hyperplane, potentially missing the true maximum direction in high-dimensional space.
- The assumption that the maximum lies on the neighborhood boundary may not hold for complex loss landscapes.
- The hyperparameter $\rho_m$ is introduced and carries a different meaning from SAM's $\rho$.
- Effectiveness on very large-scale models (e.g., LLMs) remains unvalidated.

## Related Work & Insights
- **SAM variants**: ASAM (Kwon et al., 2021) employs adaptive perturbations; GSAM (Zhuang et al., 2022) uses the orthogonal component of the local gradient.
- **WSAM** (Yue et al., 2023) and Zhao et al. (2022a) also use linear combinations of $g_0$ and $g_1$, but with fixed weights.
- **SAM theory**: Wen et al. (2023) and Bartlett et al. (2023) investigate implicit bias.
- **Multi-step SAM**: Originally proposed in Foret et al. (2020) but shown to be ineffective in practice.

## Rating
- Novelty: ⭐⭐⭐⭐ — New intuitive explanation + multi-step degradation analysis + unified method
- Theoretical Depth: ⭐⭐⭐⭐ — Theoretical confirmation under second-order approximation, combining intuitive and formal analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple models and datasets, multi-step ablation, and computational overhead analysis
- Writing Quality: ⭐⭐⭐⭐ — Plug-and-play replacement for SAM with almost no additional overhead

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICLR 2026\] A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)
- [\[ICLR 2026\] Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning](oversmoothing_oversquashing_heterophily_long-range_and_more_demystifying_common_.md)

</div>

<!-- RELATED:END -->
