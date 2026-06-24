---
title: >-
  [Paper Note] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry
description: >-
  This work explains why Lottery Ticket Hypothesis (LTH) masks cannot transfer to new initializations from the perspective of weight symmetry, and proposes to achieve sparse training by aligning LTH masks with the optimization basins of new initializations via permutation matching.
tags:

date: 2026-05-08
content_hash: 43d1f5c5d7024696
---

# Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry

## Meta Information
- **Conference**: ICML 2025
- **arXiv**: [2505.05143](https://arxiv.org/abs/2505.05143)
- **Code**: GitHub (Public)
- **Area**: Others
- **Keywords**: Lottery Ticket Hypothesis, Weight Symmetry, Permutation Matching, Sparse Training, Linear Mode Connectivity

## TL;DR
This work explains why Lottery Ticket Hypothesis (LTH) masks cannot transfer to new initializations from the perspective of weight symmetry, and proposes to achieve sparse training by aligning LTH masks with the optimization basins of new initializations via permutation matching.

## Background & Motivation

### Background

**Background**: LTH: Dense networks contain sparse subnetworks that, when trained with the original initialization, can match the performance of the dense network.

### Limitations of Prior Work

**Limitations of Prior Work**: Core Problem: LTH masks are tied to the initialization from which they were found and cannot be transferred to a new random initialization.

### Key Challenge

**Key Challenge**: Weight Symmetry: Neural networks exhibit permutation invariance, where swapping two neurons in the same layer does not alter functionality.

### Core Idea

**Core Idea**: Hypothesis: Mask transfer fails because the optimization basin of the LTH mask is not aligned with that of the new initialization.

## Method

### 1. Core Hypothesis
Models trained from different random initializations converge to the same basin (modulo permutation); therefore, LTH masks require corresponding permutations to align with new initializations.

### 2. Permutation Matching
Activation matching (Ainsworth et al., 2023) is used to find the permutation mapping $\pi$:
$$\pi_l = \arg\min_\pi \|Z_l^B - \pi Z_l^A\| = \arg\max_\pi \langle \pi, Z^B(Z^A)^\top \rangle_F$$
The linear assignment problem is solved via the Hungarian algorithm.

### 3. Mask Alignment
- Train models A and B to convergence.
- Find the permutation $\pi$ using activation matching to align $\pi(w_A^{t=T})$ with $w_B^{t=T}$.
- Permute the LTH mask $m_A$ to $\pi(m_A)$.
- Start sparse training with $\pi(m_A)$ starting from $w_B^{t=k}$ (rewinding point).

### 4. Variance Collapse Repair
The REPAIR method is used to correct the variance collapse of activation statistics in interpolated networks, validating the linear mode connectivity.

## Experiments

### Main Results

| Dataset/Model | Sparsity | naive vs permuted Gap |
|------------|--------|---------------------|
| ResNet20/CIFAR-10 | 90% | permuted is significantly better than naive |
| ResNet20/CIFAR-100 | 90% | Consistent advantage |
| VGG11/CIFAR-10 | 90% | permuted is close to LTH |
| ResNet50/ImageNet | 95% | permuted outperforms naive by about 2% |

### Width Effect
Wider models lead to more accurate permutation matching, reducing the gap between permuted and LTH (progressively narrowing from width=1 to 16).

### Diversity Analysis (Table 1)

| Method | Test Accuracy | Ensemble Accuracy | Disagreement | KL | JS |
|------|---------|---------|--------|-----|-----|
| LTH | 91.15% | 91.43% | 0.035 | 0.038 | 0.011 |
| Permuted | 89.38% | 91.75% | 0.107 | 0.273 | 0.091 |

The Permuted method exhibits significantly higher functional diversity than LTH, which in turn leads to better ensemble performance.

## Highlights & Insights
- Novel Insight: Understanding the non-transferability of LTH masks from the perspective of weight symmetry.
- Corrected the conclusion of Paul et al. (2023) that LTH and dense solutions do not reside in the same basin (they are connected after accounting for variance collapse).
- The permuted models exhibit much higher functional diversity than LTH, which is beneficial for ensembling.
- The method is simple, requiring only standard activation matching and mask permutation.

## Limitations & Future Work
- Requires training two dense models to obtain the permutation mapping, which increases computational cost.
- Permutation matching is an NP-hard problem; greedy solutions are not precise enough on ImageNet.
- Current hardware cannot efficiently exploit unstructured sparsity.
- Model pruning may introduce algorithmic bias (Hooker et al., 2020).

## Rating
⭐⭐⭐⭐ Explains the limitations of LTH from a symmetry perspective with strong insight. The experiments comprehensively demonstrate the effectiveness and diversity advantages of mask alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](../../NeurIPS2025/others/sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[ICML 2026\] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate](../../ICML2026/others/theoretical_analysis_of_sparse_optimization_with_reparameterization_weight_decay.md)
- [\[CVPR 2025\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2025/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)

</div>

<!-- RELATED:END -->
