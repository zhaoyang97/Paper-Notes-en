---
title: >-
  [Paper Note] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training
description: >-
  [CVPR2025][Sparse Training] ZO-SAM strategically integrates zero-order optimization into the perturbation step of SAM, achieving the flat-minimum advantages of SAM with only a single backward pass, thereby halving the computational overhead while enhancing accuracy and robustness in sparse training scenarios.
tags:
  - "CVPR2025"
  - "Sparse Training"
  - "Zero-Order Optimization"
  - "Sharpness-Aware Minimization"
  - "Gradient Variance"
  - "Model Compression"
date: 2026-05-08
content_hash: 0301f89c04fd00d7
---

# ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training

**Conference**: CVPR2025  
**arXiv**: [2603.13115](https://arxiv.org/abs/2603.13115)  
**Code**: Pending confirmation  
**Area**: Others  
**Keywords**: Sparse Training, Zero-Order Optimization, Sharpness-Aware Minimization, Gradient Variance, Model Compression

## TL;DR

ZO-SAM strategically integrates zero-order optimization into the perturbation step of SAM, achieving the flat-minimum advantages of SAM with only a single backward pass, thereby halving the computational overhead while enhancing accuracy and robustness in sparse training scenarios.

## Background & Motivation

- Although sparse neural networks reduce computational and memory overhead by significantly cutting down parameter size, existing sparse training methods face the **highly noisy and chaotic gradient signals** key challenge at high sparsity levels, which severely hinders convergence and generalization.
- SAM (Sharpness-Aware Minimization) reduces gradient variance and improves generalization by guiding models toward flat minima, but it requires **two full backward passes** per step, doubling the computational overhead.
- In sparse training scenarios, computational efficiency itself is a core demand, making the doubled overhead of SAM directly undermine its feasibility.
- **Core Problem**: Can the redundant computation of double backward passes be eliminated while retaining the flat-minimum advantage of SAM?

## Method

### ZO-SAM Framework Design

The core idea of ZO-SAM is to **selectively** use zero-order/first-order gradients during the two steps of SAM:

1. **Perturbation Step (Zero-Order Substitute)**: Uses Random Gradient Estimation (RGE) instead of backward propagation to calculate the perturbation direction.
    - $\hat{\nabla}\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^{m}\frac{\mathcal{L}(\theta+\delta u_i)-\mathcal{L}(\theta-\delta u_i)}{2\delta}u_i$
    - Requires only forward passes (function evaluation) without backward propagation.
    - The purpose of the perturbation step is to determine the perturbation direction, which has a lower requirement for gradient accuracy and tolerates approximation errors.

2. **Gradient Update Step (Retaining First-Order Accuracy)**: Performs a full backward propagation at the perturbed parameter point $\theta+\epsilon$.
    - $\theta \leftarrow \theta - \eta\nabla\mathcal{L}(\theta+\epsilon)$
    - The update step directly determines the quality of parameter optimization, requiring precise gradients.

### Rationality of Design Choices

- Choosing RGE instead of CGE (Coordinate Gradient Estimation): RGE requires only $m \ll d$ function evaluations whereas CGE requires $d$ evaluations, which is infeasible for large-scale models.
- The random direction sampling of RGE offers a more global exploration of the loss landscape, assisting in avoiding sharp minima.
- Placing the zero-order estimation in the first step (perturbation) rather than the second step (update): the perturbation step is error-tolerant, whereas the update step demands high accuracy.

### Computational Overhead Analysis

- Standard SAM: requires 2 full backward passes per step $\rightarrow$ overhead is approximately twice that of SGD.
- ZO-SAM: the first step only requires $2m$ forward passes (function evaluations), and the second step requires 1 backward pass $\rightarrow$ total overhead is around $1 + 2m \times (\text{forward/backward ratio})$ of SGD.
- When $m$ is small (typically $m=1$ in the paper), the overhead of ZO-SAM is substantially lower than SAM, with measured throughput being approximately 1.6 times that of SAM.

## Key Experimental Results

### ResNet-32 on CIFAR-10/100 (Sparsity 90%/95%/98%)
- ZO-SAM consistently improves accuracy across all 7 sparse training methods.
- CIFAR-10 improvement range: 0.38%–2.31% (ResNet-32).
- CIFAR-100 improvement range: 0.45%–2.54% (ResNet-32).
- The maximum improvement occurs in RigL + ZO-SAM: +2.54% at 90% sparsity on CIFAR-100.

### DeiT on ImageNet-1K
- DeiT-Tiny 50% sparsity: maximum improvement of 1.14% (SViTE $\rightarrow$ SNIP + ZO-SAM).
- DeiT-Small 70% sparsity: maximum improvement of 1.17% (RigL + ZO-SAM).

### Efficiency Comparison with SAM Variants (ResNet-32, 90% Sparsity, MEST)

| Method | CIFAR-10 | Throughput (img/s) | Relative Efficiency |
|------|----------|--------------|---------|
| SAM | 93.77% | 2704 | 47.67% |
| GSAM | 93.72% | 2701 | 47.60% |
| ZO-SAM | 93.50% | 4349 | **76.67%** |

### Robustness (CIFAR-10-C, SNIP, 90%)
- ZO-SAM achieves a 3.10% accuracy improvement on corruption datasets, with the smallest $\Delta$ and the strongest robustness.

### Convergence Speed
- Epochs to reach 90% accuracy: ZO-SAM 70 epochs vs SGD 104 epochs (88 vs 131 under 95% sparsity).
- Convergence speed outperforms all SAM variants (ESAM 75/92, LookSAM 79/94, GSAM 84/113).

### Loss Landscape Visualization
- Without ZO-SAM, the high-sparsity loss landscape exhibits a narrow and steep basin.
- With the introduction of ZO-SAM, the loss landscape transitions to a wide and flat basin, indicating improved gradient stability.

### Feature Map Comparison
- Clearer and more concentrated feature activations are observed across shallow (3), intermediate (17), and deep (31) layers of ResNet-32.
- Feature maps of baseline methods display scattered or blurry patterns, showing high gradient variance.

## Highlights & Insights

1. **Exquisite Hybrid Strategy**: Rather than simply replacing all gradients with zero-order ones, it selectively applies them based on the distinct accuracy demands of the two SAM steps, achieving a balance between efficiency and quality.
2. **Plug-and-Play**: Can be seamlessly integrated with any sparse training method (static/dynamic) to yield consistent improvements, without modifying the baseline training flows.
3. **Thorough Multi-Dimensional Validation**: Extensively covers CNNs/Transformers, multiple datasets, various sparsities, loss landscape visualizations, convergence rates, feature maps, and robustness.
4. **High Practical Value**: Achieving a throughput of 76.67% of SGD (far exceeding SAM's 47.67%), making it highly practical in resource-constrained scenarios.

## Limitations & Future Work

1. Accuracy is slightly lower than full SAM (~0.27% on CIFAR-10), sacrificing a minor margin of accuracy for efficiency.
2. The selection of hyperparameters $m$ (number of RGE perturbation directions) and $\delta$ (perturbation step size) lacks in-depth analysis and sensitivity experiments.
3. Evaluated only on classification tasks, without extending to downstream tasks like detection or segmentation.
4. Theoretical analysis (convergence guarantees) is relatively weak, lacking a formal convergence proof of zero-order estimation under the SAM framework.
5. The approximation quality of RGE might degenerate in ultra-high-dimensional parameter spaces, and the upper limit of model scale remains undiscussed.
6. Comparison with other zero-order optimization methods (e.g., ZO-SGD, ZO-Adam) integrated into the SAM framework is missing.
7. Experiments cover only vision classification backbones, without verifying generalizability to other fields such as NLP or multimodal learning.

## Rating

- Novelty: ⭐⭐⭐ (Combining zero-order optimization with SAM is intuitive and reasonable, but not highly unexpected)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive coverage across various methods, architectures, datasets, and metrics)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and smooth logical flow)
- Value: ⭐⭐⭐⭐ (Holds direct application value in practical sparse training)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)
- [\[ICLR 2026\] Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](../../ICLR2026/others/revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)
- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](../../NeurIPS2025/others/sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)

</div>

<!-- RELATED:END -->
