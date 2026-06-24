---
title: >-
  [Paper Note] Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework
description: >-
  [ECCV2024][Optimization][tensor completion] A multi-objective tensor recovery framework (MOTC) based on learnable tensor nuclear norm is proposed. By introducing learnable unitary matrices in place of fixed transforms, this approach addresses the performance degradation of t-SVD methods on non-smooth tensor data, while effectively exploiting the low-rankness of tensors across all dimensions through multi-objective optimization.
tags:
  - "ECCV2024"
  - "Optimization"
  - "tensor completion"
  - "tensor SVD"
  - "multi-objective optimization"
  - "learnable tensor nuclear norm"
  - "non-smooth tensor recovery"
date: 2026-05-08
content_hash: f295fe364d7b0066
---

# Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework

**Conference**: ECCV2024  
**arXiv**: [2311.13958](https://arxiv.org/abs/2311.13958)  
**Code**: [jzheng20/MOTC](https://github.com/jzheng20/MOTC)  
**Area**: Optimization  
**Keywords**: tensor completion, tensor SVD, multi-objective optimization, learnable tensor nuclear norm, non-smooth tensor recovery

## TL;DR

A multi-objective tensor recovery framework (MOTC) based on learnable tensor nuclear norm is proposed. By introducing learnable unitary matrices in place of fixed transforms, this approach addresses the performance degradation of t-SVD methods on non-smooth tensor data, while effectively exploiting the low-rankness of tensors across all dimensions through multi-objective optimization.

## Background & Motivation

Tensor recovery methods based on tensor singular value decomposition (t-SVD) have recently achieved significant success in processing visual data, such as color images and videos. The core mechanism of these methods is to apply a fixed reversible transform (such as DFT or DCT) along a specific dimension of the tensor and analyze the low-rank structures of the transformed frontal slices.

However, existing t-SVD methods suffer from severe performance degradation when dealing with tensor data characterized by **non-smooth variations**. This issue primarily manifests in two scenarios:

1. **Unordered image sequences**: In classification tasks, the order of samples is typically random. Slice Permutation Variability (SPV) significantly affects t-SVD performance.
2. **Tensor data with rapid content variations**: Examples include videos with high frame-to-frame dynamics and image sequences composed of different scenes.

The root cause of these limitations is that fixed reversible transforms (DFT/DCT) make t-SVD highly sensitive to the order and non-smooth variations of tensor slices. Furthermore, when extending t-SVD to high-order tensors, traditional weighted sum methods (such as WSTNN) introduce $h(h-1)/2$ weight parameters, which are notoriously difficult to tune.

## Core Problem

1. How can t-SVD methods remain effective when dealing with non-smooth or unordered tensor data?
2. How can t-SVD be efficiently extended to high-order tensors while capturing correlations across different dimensions without relying on a large number of weight parameters?

## Method

### 1. Learnable Tensor Nuclear Norm

Traditional t-SVD methods assume that a tensor can be decomposed as $\mathcal{M} = \mathcal{Z} \times_{k_3} \hat{U}_{k_3}^T \cdots \times_{k_h} \hat{U}_{k_h}^T$, where the transform matrix $\hat{U}$ is predefined (e.g., DFT matrix). When slices are unordered, information collapses into high-frequency slices, causing the low-rank assumption to fail.

The core innovation of this work is to replace fixed transforms with **learnable unitary matrices** $\{U_{k_n}\}$, proposing a tensor completion model with learnable tensor rank (TC-SL):

$$\min_{\mathcal{X}, U_{k_n}} \text{rank}_{[k_1,k_2]}(\mathcal{X} \times_{k_{s+1}} U_{k_{s+1}} \cdots \times_{k_h} U_{k_h}) \quad \text{s.t. } \Psi_{\mathbb{I}}(\mathcal{M}) = \Psi_{\mathbb{I}}(\mathcal{X}), \; U_{k_n}^T U_{k_n} = I$$

Since the tensor rank function is discrete (NP-hard), it is relaxed using the tensor nuclear norm as the tightest convex envelope, yielding the final TC-SL model:

$$\min_{\mathcal{X}, \tilde{\mathcal{U}}} \|\mathcal{X}\|_{*, \tilde{\mathcal{U}}}^{[k_1,k_2]} \quad \text{s.t. } \Psi_{\mathbb{I}}(\mathcal{M}) = \Psi_{\mathbb{I}}(\mathcal{X})$$

### 2. Multi-objective Tensor Recovery Framework (MOTC)

TC-SL depends on a specific choice of $(k_1, k_2)$ and only considers the low-rankness of a single mode. To simultaneously exploit correlations across all dimensions, a multi-objective optimization model is proposed:

$$\min_{\mathcal{X}, \tilde{\mathcal{U}}_{(k_1,k_2)}} \left[\|\mathcal{X}\|_{*, \tilde{\mathcal{U}}_{(k_1,k_2)}}^{[k_1,k_2]}\right]_{1 \leq k_1 < k_2 \leq h} \quad \text{s.t. } \Psi_{\mathbb{I}}(\mathcal{M}) = \Psi_{\mathbb{I}}(\mathcal{X})$$

Each objective function evaluates the low-rankness of a different mode pair $(k_1, k_2)$, without introducing extra weight parameters.

### 3. Alternating Proximal Multiplier Method (APMM)

An Alternating Proximal Multiplier Method (APMM) is developed to solve TC-SL:

- **Update $\mathcal{Z}$**: Solved via the tensor singular value thresholding (t-SVT) operator.
- **Update $U_{k_n}$**: Computed in closed-form as $U_{k_n} = UV^T$ via SVD of $\mu \mathcal{A}_{(k_n)}\mathcal{B}_{(k_n)}^T + \eta U_{k_n}^{(t)}$.
- **Update $\mathcal{E}$**: Projecting onto the complementary set yields a closed-form solution.

The computational complexity per iteration is $\mathcal{O}((h-s)(h+I_{(1)}-1)I_{(1)} \prod I_k + h I_{(1)} \prod I_k)$.

### 4. Solving MOTC

Two major steps are executed alternately:
- Use APMM to learn the optimal transform $\hat{\mathcal{U}}_{(k_1,k_2)}$ corresponding to each $(k_1, k_2)$ pair.
- Use NSGA-II (a multi-objective genetic algorithm) to solve the multi-objective optimization problem, treating the average of the Pareto front individuals as the final output.

Both steps can be parallelized for execution.

## Key Experimental Results

### Image Inpainting — Image Sequences from Different Scenes (BSD Dataset)

| Sampling Rate | TNN-DCT | WSTNN | TC-SL | **MOTC** |
|--------|---------|-------|-------|----------|
| 0.3    | 23.25   | 25.75 | 26.32 | **27.53** |
| 0.5    | 27.25   | 31.07 | 31.55 | **33.25** |
| 0.7    | 32.04   | 37.11 | 38.33 | **39.97** |
| Average   | 27.51   | 31.31 | 32.06 | **33.58** |

### Image Inpainting — Randomly Permuted Sequences (Sampling Rate 0.3)

| Dataset   | WSTNN | HTNN-DCT | TC-SL | **MOTC** |
|----------|-------|----------|-------|----------|
| CIFAR10  | 25.02 | 22.12    | 24.63 | **26.46** |
| CIFAR100 | 24.76 | 21.71    | 24.50 | **26.16** |
| LFW      | 34.33 | 30.15    | 31.57 | **35.67** |
| GTF      | 26.66 | 22.11    | 32.10 | **33.56** |
| Average     | 27.69 | 24.02    | 28.20 | **30.46** |

### Video Inpainting — Rapidly Changing Frames (HMDB51, Sampling Rate 0.3)

| Method | TNN-DCT | WSTNN | HTNN-DCT | TC-SL | **MOTC** |
|------|---------|-------|----------|-------|----------|
| Average PSNR | 30.01 | 35.28 | 27.71 | 36.82 | **38.74** |

In video inpainting, MOTC outperforms the third-best method by **more than 3.5 dB**, and achieves improvements of up to **5-10 dB** on challenging videos. Compared to methods that only consider single-dimension low-rankness, TC-SL improves the results by about **6.5 dB**.

## Highlights & Insights

1. **Precise Problem Definition**: The paper clearly identifies the fundamental limitations of t-SVD on non-smooth data (SPV and non-smooth variation) and presents a unified solution.
2. **Replacing Fixed Transforms with Learnable Transforms**: The introduction of learnable unitary matrices is a natural and elegant design. It mitigates the SPV problem while adapting to the intrinsic structure of the data.
3. **Multi-objective Framework Avoiding Weight Tuning**: Substituting weighted sum formulations with multi-objective optimization eliminates the need to configure $h(h-1)/2$ weight parameters.
4. **Convergence Guarantee**: Mathematically guarantees that APMM converges to KKT points with a rate of $\mathcal{O}(1/\eta^{(t)})$.
5. **Significant Empirical Improvements**: MOTC demonstrates a gain of 3.5+ dB in video inpainting, with up to 5-10 dB improvement on complex samples.

## Limitations & Future Work

1. **High Computational Cost**: Running APMM for every $(k_1, k_2)$ mode pair combined with NSGA-II iterations introduces a high computational overhead, making scalability to large datasets challenging.
2. **Heuristic Nature of NSGA-II**: The multi-objective optimization stage relies heavily on genetic algorithms, and taking the average of the Pareto front lacks strong theoretical optimality guarantees.
3. **Task-Specific Validation**: Although the framework is claimed to be adaptable to denoising and clustering, experiments are limited to tensor completion.
4. **Initialization of Unitary Matrices**: The choice of initial transforms (e.g., DFT/DCT) may affect convergence quality, but sensitivity to initialization is not analyzed.
5. **Lack of Comparison with Deep Learning Methods**: Performance comparisons against deep learning-based image/video restoration approaches are missing.

## Related Work & Insights

| Method | Core Idea | Handles Non-smooth | High-order Tensor | Weight Tuning |
|------|---------|-----------|---------|---------|
| TNN-DCT/DFT | Fixed transform + 3D t-SVD | No | No | None |
| WSTNN | Weighted TNN on all mode unfoldings | No | Yes | Requires $h(h-1)/2$ |
| HTNN-DCT | High-order t-SVD + Fixed DCT | No | Yes (single dim) | None |
| TC-SL (Ours) | Learnable unitary matrix + t-SVD | **Yes** | Yes (single dim) | None |
| MOTC (Ours) | Multi-objective + Learnable nuclear norm | **Yes** | **Yes (multi-dim)** | **None** |

A direct comparison with HTNN-DCT highlights the value of learnable transforms: TC-SL averages a gain of more than 4+ dB on randomly permuted data. Building upon this, MOTC exploits cross-dimensional correlations to yield an additional 2+ dB increase.

## Insights & Implications

- **The concept of learnable transforms** can be extended to other tensor methods relying on fixed transforms (e.g., tensor denoising, tensor decomposition) by replacing fixed transforms with learnable unitary matrices and optimizing with analogous proximal schemes.
- **Using multi-objective optimization to bypass weighted sums** represents a useful design paradigm for optimization tasks requiring multiple regularization terms, circumventing tedious parameter searches.
- The proposed method holds direct application value for video understanding scenarios with discontinuous frame sequences (e.g., video summarization and completion after keyframe extraction).

## Rating
- Novelty: ⭐⭐⭐⭐ — The integration of learnable tensor nuclear norm with a multi-objective framework is both novel and logical.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple scenarios and datasets with substantial gains, though comparisons with deep learning baselines are absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations with well-presented motivations.
- Value: ⭐⭐⭐⭐ — Provides an effective solution for adapting t-SVD methods to non-smooth scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Power of Small Initialization in Noisy Low-Tubal-Rank Tensor Recovery](../../ICLR2026/optimization/the_power_of_small_initialization_in_noisy_low-tubal-rank_tensor_recovery.md)
- [\[ICLR 2026\] Hierarchical Multi-Stage Recovery Framework for Kronecker Compressed Sensing](../../ICLR2026/optimization/hierarchical_multi-stage_recovery_framework_for_kronecker_compressed_sensing.md)
- [\[ICLR 2026\] Combinatorial Bandit Bayesian Optimization for Tensor Outputs](../../ICLR2026/optimization/combinatorial_bandit_bayesian_optimization_for_tensor_outputs.md)
- [\[ICLR 2026\] In-Context Multi-Objective Optimization](../../ICLR2026/optimization/in-context_multi-objective_optimization.md)
- [\[ICML 2026\] Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization](../../ICML2026/optimization/accelerated_multiple_wasserstein_gradient_flows_for_multi-objective_distribution.md)

</div>

<!-- RELATED:END -->
