---
title: >-
  [Paper Note] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training
description: >-
  [CVPR 2026][sparse training] This paper proposes ZO-SAM, which replaces the backpropagation in SAM's perturbation step with zeroth-order gradient estimation…
tags:
  - "CVPR 2026"
  - "sparse training"
  - "SAM"
  - "zeroth-order optimization"
  - "gradient variance"
  - "flat minima"
date: 2026-05-08
content_hash: 8c2224d0db423094
---

# ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training

**Conference**: CVPR 2026
**arXiv**: [2603.13115](https://arxiv.org/abs/2603.13115)
**Code**: None
**Area**: Other
**Keywords**: sparse training, SAM, zeroth-order optimization, gradient variance, flat minima

## TL;DR

This paper proposes ZO-SAM, which replaces the backpropagation in SAM's perturbation step with zeroth-order gradient estimation, reducing SAM's computational overhead from two backward passes to one. This makes SAM practical for sparse training for the first time, achieving consistent improvements of 0.38%–2.54% over all mainstream sparse training methods on CIFAR-10/100 and ImageNet-1K.

## Background & Motivation

**Background**: Sparse neural networks significantly reduce parameter count and computational cost by retaining only a small fraction of active weights. Mainstream approaches are categorized into static methods (LTH, SNIP, GraSP) and dynamic methods (SET, DSR, RigL, MEST).

**Limitations of Prior Work**:
   - Gradient signals in sparse training are noisy and chaotic — after extensive pruning, the remaining parameters bear a disproportionate burden, and gradient variance increases sharply with sparsity.
   - High sparsity leads to narrower and steeper loss landscapes, resulting in inefficient and meandering optimization trajectories.
   - SAM can guide models toward flat minima to mitigate these issues, but its **dual-backpropagation overhead** directly contradicts the computational efficiency goals of sparse training.

**Key Challenge**: The generalization benefit of SAM versus its doubled computational cost is a fundamental tension, particularly acute in sparse training, which is inherently motivated by computational savings.

**Key Insight**: The perturbation step of SAM does not require high gradient accuracy — it only needs to determine the perturbation direction — making it amenable to coarse zeroth-order approximation instead of exact gradients.

**Core Idea**: Apply random gradient estimation (RGE) in SAM's perturbation step while retaining the first-order exact gradient in the update step, reducing the number of backward passes from two to one.

## Method

### Overall Architecture

ZO-SAM preserves SAM's two-step structure but modifies the first step:

1. **Perturbation step (zeroth-order)**: Estimates the gradient direction via RGE without backpropagation:
   $$\epsilon = \rho \frac{\hat{\nabla}\mathcal{L}(\theta)}{\|\hat{\nabla}\mathcal{L}(\theta)\|}$$
2. **Update step (first-order)**: Updates parameters using exact first-order gradients computed at the perturbed point:
   $$\theta \leftarrow \theta - \eta \nabla\mathcal{L}(\theta^*(\epsilon))$$

### Key Designs

1. **Random Gradient Estimation (RGE) as a Substitute for Backpropagation**:

    - Function: Estimates the gradient direction via forward passes during the perturbation step, eliminating the first backward pass.
    - Core formula:
    $\hat{\nabla}\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^m \frac{\mathcal{L}(\theta + \delta u_i) - \mathcal{L}(\theta - \delta u_i)}{2\delta} u_i$
      where $u_i \sim \mathcal{N}(0, I)$, $\delta$ is a small step size, and $m \ll d$ is the number of samples.
    - Design Motivation: The perturbation step only requires identifying an approximate worst-case direction; exact gradients are unnecessary. RGE requires only $2m$ forward passes (with small $m$), far less than a full backward pass. The random directional sampling also provides smoother landscape exploration.
    - Why RGE over CGE: CGE requires $d$ evaluations (where $d$ is the parameter dimension, on the order of millions), making it infeasible. RGE's random direction sampling additionally yields smoother landscape exploration.

2. **Retention of First-Order Exact Update**:

    - Function: Computes exact gradients via standard backpropagation at the perturbed parameter point $\theta^*(\epsilon) = \theta + \epsilon$.
    - Design Motivation: The parameter update step demands high-precision gradients to ensure training stability and convergence; approximation is not acceptable here.

3. **Compatibility with Sparse Training Methods**:

    - Function: ZO-SAM serves as a drop-in optimizer replacement for SGD and can be combined with any sparse training method.
    - Validated in combination with 7 methods: LTH, SNIP, GraSP (static) + SET, DSR, RigL, MEST (dynamic).
    - Design Motivation: As a general optimization framework, ZO-SAM does not alter the sparse structure search logic.

### Loss & Training

Standard classification loss (cross-entropy) is used; ZO-SAM modifies only the optimizer. The hyperparameter $\rho$ (neighborhood size) inherits SAM's default value, while $\delta$ (zeroth-order step size) and $m$ (number of samples) are newly introduced hyperparameters.

## Key Experimental Results

### Main Results — ResNet-32 on CIFAR-10/100 (90%/95%/98% Sparsity)

| Method | CIFAR-10 90% | CIFAR-10 98% | CIFAR-100 90% | CIFAR-100 98% |
|------|-------------|-------------|--------------|--------------|
| RigL | 93.07 | 89.00 | 70.34 | 64.07 |
| **RigL+ZO-SAM** | **93.66**(+0.59) | **90.61**(+1.61) | **72.88**(+2.54) | **65.17**(+1.10) |
| MEST | 92.56 | 89.22 | 70.44 | 64.59 |
| **MEST+ZO-SAM** | **93.50**(+0.94) | **91.53**(+2.31) | **72.20**(+1.76) | **66.01**(+1.42) |

### Transformer on ImageNet-1K

| Model | Sparsity | Method | Accuracy (%) | Gain |
|------|--------|------|-------------|------|
| DeiT-Small | 70% | RigL | 77.99 | - |
| DeiT-Small | 70% | **RigL+ZO-SAM** | **79.16** | +1.17 |
| DeiT-Tiny | 50% | SViTE | 70.18 | - |
| DeiT-Tiny | 50% | **SNIP+ZO-SAM** | **71.32** | +1.14 |

### Convergence Speed Comparison

| Method | Epochs to reach 90% accuracy (CIFAR-10, sp=0.9) |
|------|--------------------------------------|
| SGD | 104 |
| ESAM | 75 |
| LookSAM(k=5) | 79 |
| GSAM | 84 |
| **ZO-SAM** | **70** |

### Key Findings
- **Higher sparsity yields greater gains from ZO-SAM**: Improvements are most pronounced at 98% sparsity (MEST+ZO-SAM achieves +2.31% on CIFAR-10), as gradient variance issues are more severe at higher sparsity levels.
- ZO-SAM transforms the loss landscape from a narrow, deep basin to a wide, flat basin, as confirmed by visualization.
- Gradient variance is substantially reduced: at 90% sparsity, ZO-SAM's gradient variance is approximately one-third that of SGD.
- Convergence speed is roughly 30 epochs faster than SGD and comparable to efficient SAM variants such as ESAM.
- Effectiveness generalizes to Transformers (DeiT), not limited to CNNs.
- ZO-SAM demonstrates greater robustness on CIFAR-10-C distribution shift benchmarks.

## Highlights & Insights
- **Precise diagnosis of the core problem in sparse training**: Rather than broadly attributing difficulty to sparsity, the paper pinpoints "high gradient variance" as the specific root cause and addresses it directly.
- **The hybrid zeroth-order/first-order strategy is elegant**: The perturbation step does not require directional precision (zeroth-order suffices), while the update step demands exact gradients (first-order retained). This selective allocation of computational resources is a principled design insight worth emulating.
- **Plug-and-play generality**: Consistent improvements across 7 sparse training methods × 3 sparsity levels × 2 datasets, without modifying the underlying sparse methods.
- **First practical deployment of SAM in sparse training**: Prior to this work, SAM's doubled computational overhead rendered it impractical for sparse training; ZO-SAM removes this barrier.

## Limitations & Future Work
- The approximation noise introduced by RGE may accumulate in very high-dimensional settings; further validation on large-scale models is needed.
- The selection of the zeroth-order sample count $m$ lacks an adaptive strategy and currently requires manual tuning.
- Evaluation is limited to $\ell_\infty$ unstructured sparsity; structured sparsity (e.g., channel pruning) is not addressed.
- Full ImageNet-1K experiments cover only DeiT-Tiny/Small; the behavior of larger models (e.g., ViT-Large) remains unknown.
- Combinations with more efficient SAM variants (ESAM, LookSAM) are unexplored — whether a "ZO-ESAM" could yield further speedups is an open question.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of zeroth-order estimation with SAM is conceptually straightforward, yet the design decision to apply it selectively to the perturbation step reflects careful reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 7 methods × 3 sparsity levels × multiple datasets and architectures.
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis (gradient variance, loss landscape visualization) is well executed.
- Value: ⭐⭐⭐⭐ Makes SAM genuinely usable in sparse training, with high practical engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[ICLR 2026\] Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](../../ICLR2026/others/revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](../../NeurIPS2025/others/sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)

</div>

<!-- RELATED:END -->
