---
title: >-
  [Paper Note] Understanding Dataset Distillation via Spectral Filtering
description: >-
  [ICLR 2026][Model Compression][dataset distillation] This paper proposes UniDD, a spectral filtering framework that unifies diverse dataset distillation methods as applying different filter functions on the feature-feature correlation (FFC) matrix to match the frequency information of the feature-label correlation (FLC) matrix. Building on this insight, the paper further introduces Curriculum Frequency Matching (CFM).
tags:
  - ICLR 2026
  - Model Compression
  - dataset distillation
  - spectral filtering
  - frequency matching
  - curriculum learning
  - unified framework
date: 2026-05-08
content_hash: 3053d907f0937d92
---

# Understanding Dataset Distillation via Spectral Filtering

**Conference**: ICLR 2026
**arXiv**: [2503.01212](https://arxiv.org/abs/2503.01212)
**Code**: Not provided
**Area**: Model Compression / Dataset Distillation
**Keywords**: dataset distillation, spectral filtering, frequency matching, curriculum learning, unified framework

## TL;DR

This paper proposes UniDD, a spectral filtering framework that unifies diverse dataset distillation methods as applying different filter functions on the feature-feature correlation (FFC) matrix to match the frequency information of the feature-label correlation (FLC) matrix. Building on this insight, the paper further introduces Curriculum Frequency Matching (CFM).

## Background & Motivation

Dataset distillation (DD) accelerates model training by compressing large-scale datasets into compact synthetic datasets. Existing methods differ substantially in their optimization objectives:
- **Distribution Matching** (DM): aligns statistics such as class means
- **Gradient Matching** (DC): minimizes discrepancies in gradient directions
- **Trajectory Matching** (MTT): simulates parameter update trajectories
- **Kernel Methods** (FrePo): bypasses the inner-loop optimization via closed-form solutions

**Core Problem**: What are the connections among these methods, and does a unified framework exist?

## Method

### Overall Architecture: UniDD

**Theorem 1** (Unified Spectral Filtering Framework):

$$\min_{X_s} \left\| f(X^\top X) g(X^\top Y) - f(X_s^\top X_s) g(X_s^\top Y_s) \right\|_F^2$$

where:
- $X^\top X$, $X_s^\top X_s$: FFC matrices (feature-feature correlation)
- $X^\top Y$, $X_s^\top Y_s$: FLC matrices (feature-label correlation)
- $f(\cdot)$: filter function applied to the eigenvalues of the FFC matrix
- $g(\cdot)$: binary function, $g = I$ or $X^\top Y$

### Key Design 1: Low-Frequency Matching (LFM)

**DM (Distribution Matching)**:
$$f(\lambda) = 1 \quad \Rightarrow \quad \|X^\top Y - X_s^\top Y_s\|_F^2$$
Equivalent to an identity filter that directly matches class-mean representations.

**DC (Gradient Matching)**:
$$f(\lambda) = \{1, \lambda\} \quad \Rightarrow \quad \|X^\top X - X_s^\top X_s\|_F^2 + \|X^\top Y - X_s^\top Y_s\|_F^2$$
Derived via an upper bound on gradient discrepancy.

Low-frequency matching methods capture coarse-grained color information, exhibiting fast convergence but poor diversity.

### Key Design 2: High-Frequency Matching (HFM)

**MTT (Trajectory Matching)**:
$$f(\lambda) = (1 - \alpha\lambda)^{\{p,q\}}$$
Acts as a high-pass filter when $\alpha\lambda < 1$, emphasizing high-frequency components corresponding to small eigenvalues.

**FrePo (KRR)**:
$$f(\lambda) = (\lambda + \beta)^{-1}$$
Inversely weights components; smaller $\beta$ amplifies high-frequency contributions.

High-frequency matching methods synthesize fine-grained textures with better diversity but at higher computational cost.

### Key Design 3: Curriculum Frequency Matching (CFM)

Existing methods employ fixed filter functions and thus capture only a single frequency band. CFM dynamically adjusts the filter parameter:

$$\beta_b = \beta \cdot (1 + \cos(\pi b / B)) / 2$$

where $B$ is the total number of batches. As $\beta_b$ decreases from large to small, the filter transitions progressively from low-frequency to high-frequency, simultaneously covering consistency and diversity.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{cls}(H_s, Y_s) + \eta \mathcal{L}_{filter} + \eta \mathcal{L}_{signal}$$

where $\eta = 0.1$, and the two matching losses correspond to $g = I$ and $g = X^\top Y$, respectively:

$$\mathcal{L}_{filter} = \sum_{b,l} \|(\Psi^l + \beta_b I)^{-1} - (\Psi_s^{l,b} + \beta_b I)^{-1}\|$$

$$\mathcal{L}_{signal} = \sum_{b,l} \|(\Psi^l + \beta_b I)^{-1}\Phi^l - (\Psi_s^{l,b} + \beta_b I)^{-1}\Phi_s^l\|$$

Exponential Moving Update (EMU) is employed to approximate full-batch statistics.

## Key Experimental Results

### Main Results on CIFAR-10/100

| Dataset | IPC | DM | DC | MTT | FrePo | CFM |
|--------|-----|------|------|------|-------|------|
| CIFAR-10 | 10 | 48.9 | 44.9 | 65.3 | 65.5 | **52.1** |
| CIFAR-10 | 50 | 63.0 | 53.9 | 71.6 | 71.7 | **64.0** |
| CIFAR-100 | 10 | 29.7 | 25.2 | 33.1 | 42.5 | **58.3** |
| CIFAR-100 | 50 | 43.6 | 30.6 | 42.9 | 44.3 | **67.1** |

On CIFAR-100 (IPC=50) with ResNet-18, CFM achieves 71.4%, substantially outperforming all baselines.

### ImageNet-1K

| IPC | SRe2L | G-VBSM | RDED | DWA | CFM |
|-----|-------|--------|------|-----|------|
| 10 | 21.3 | 31.4 | 42.0 | 37.9 | **40.6** |

### Ablation Study

| Component | Effect |
|------|------|
| $\mathcal{L}_{filter}$ only | Suboptimal |
| $\mathcal{L}_{signal}$ only | Suboptimal |
| Fixed $\beta$ (low-frequency) | Good consistency but poor diversity |
| Fixed $\beta$ (high-frequency) | Good diversity but high noise |
| CFM (dynamic $\beta$) | Optimal balance |

### Key Findings

1. Low-pass filtering (DM, DC) produces blurry synthetic images with high intra-class similarity.
2. High-pass filtering (MTT, FrePo) produces fine-grained textures with better diversity but may introduce noise.
3. Curriculum frequency scheduling consistently outperforms fixed-frequency methods across all benchmarks.
4. CFM exhibits stronger cross-architecture generalization.

## Highlights & Insights

- First work to unify four major categories of dataset distillation methods from a spectral filtering perspective.
- Theoretically elegant: reduces complex distillation objectives to the problem of designing filter functions.
- CFM is simple yet effective, with a single hyperparameter $\eta = 0.1$ that generalizes across all datasets.
- Clearly reveals the relationship between low-pass/high-pass filtering and synthetic data properties (consistency vs. diversity).

## Limitations & Future Work

- The unified framework covers only the linear kernel case; non-linear kernels (e.g., Gaussian, polynomial) are not analyzed.
- Computing FFC/FLC matrices may suffer from numerical overflow on large-scale datasets, requiring covariance-based approximations.
- Theoretical derivations rely on upper-bound approximations; the gap between the actual objective and its approximation is not quantified.
- Whether the cosine annealing schedule in CFM is optimal remains insufficiently explored.

## Related Work & Insights

- **Distribution Matching**: DM (Zhao & Bilen), IDM, SRe2L
- **Gradient Matching**: DC (Zhao et al.), IDC, DSA
- **Trajectory Matching**: MTT, DATM, FTD
- **Kernel Methods**: KIP, FrePo, RFAD
- **Spectral-Domain Methods**: FreD, NSD (differing from UniDD in the object of analysis)

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The unified framework carries significant theoretical importance.
- Theoretical Depth: ⭐⭐⭐⭐ — Derivations are clear but partly rely on upper-bound approximations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation on CIFAR and ImageNet.
- Value: ⭐⭐⭐⭐ — CFM is simple and effective with a low barrier to practical adoption.
- Writing Quality: ⭐⭐⭐⭐⭐ — Framework is clearly presented with excellent tables and visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grounding and Enhancing Informativeness and Utility in Dataset Distillation](grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[ICLR 2026\] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)
- [\[ACL 2026\] CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering](../../ACL2026/model_compression/cbrs_cognitive_blood_request_system_with_bilingual_dataset_and_dual-layer_filter.md)

</div>

<!-- RELATED:END -->
