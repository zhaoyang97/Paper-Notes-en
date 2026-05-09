---
title: >-
  [Paper Note] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation
description: >-
  [ICLR 2026][Model Compression][Dataset distillation] This paper proposes RD3 (Rectified Decoupled Dataset Distillation), systematically demonstrating that performance discrepancies among existing decoupled dataset distillation methods stem primarily from inconsistent post-evaluation settings rather than differences in distillation quality. By establishing a unified and fair evaluation framework, the reported 27.3% performance gap is corrected to 6.7%.
tags:
  - ICLR 2026
  - Model Compression
  - Dataset distillation
  - decoupled distillation
  - fair evaluation
  - post-evaluation protocol
  - synthetic data
date: 2026-05-08
content_hash: 1af035e2fef4e1ea
---

# Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation

**Conference**: ICLR 2026
**arXiv**: [2509.19743](https://arxiv.org/abs/2509.19743)
**Code**: [GitHub](https://github.com/ndhg1213/RD3)
**Area**: Model Compression / Dataset Distillation
**Keywords**: Dataset distillation, decoupled distillation, fair evaluation, post-evaluation protocol, synthetic data

## TL;DR

This paper proposes RD3 (Rectified Decoupled Dataset Distillation), systematically demonstrating that performance discrepancies among existing decoupled dataset distillation methods stem primarily from inconsistent post-evaluation settings rather than differences in distillation quality. By establishing a unified and fair evaluation framework, the reported 27.3% performance gap is corrected to 6.7%.

## Background & Motivation

Dataset distillation aims to generate compact synthetic datasets such that models trained on them achieve performance comparable to training on the full dataset. Recent decoupled distillation methods (e.g., SRe2L) have significantly extended scalability to large-scale datasets such as ImageNet-1K by separating teacher pre-training from synthetic data generation.

**Core Problem**: Existing decoupled distillation methods suffer from severe evaluation inconsistencies:
- CDA employs a smaller batch size; RDED uses a smoothed learning rate and stronger data augmentation.
- G-VBSM and EDC utilize multi-teacher mixed soft labels.
- The majority of the reported 27.3% performance gap is attributable to evaluation discrepancies rather than distillation quality.

This work presents the first systematic investigation of this issue, revealing the problem of "spurious performance gains."

## Method

### Overall Architecture

RD3 establishes a standardized evaluation protocol along three dimensions: **target dataset, compression ratio, and cross-architecture generalization**.

### 1. Unified Post-Evaluation Settings

**Training Epochs**: Unified to 400 epochs (as opposed to method-specific choices such as 300), eliminating bias introduced by differences in convergence speed.

**Batch Size**: Unified to 50 (as opposed to 1024 for SRe2L or 128 for CDA), yielding approximately 10% performance improvement.

**Smoothed Learning Rate Schedule**: Adam optimizer with an initial learning rate of 0.001 and cosine annealing schedule; smoothing factor $\zeta = 1$ (ResNet-18) or $\zeta = 2$ (other architectures).

**Data Augmentation**: RDED's augmentation strategy (CutMix + Random Resized Crop + Random Horizontal Flip) is adopted uniformly, eliminating augmentation-induced variance.

### 2. Systematic Evaluation Across Three Paradigms

Existing methods are categorized by their generation mechanism:
- **Optimization-based** (SRe2L, CDA, G-VBSM, DWA, EDC): pixel-level optimization of synthetic data via pre-trained classifiers.
- **Generation-based** (Minimax, D4M): fine-tuning generative models or optimizing vision-text embeddings.
- **Selection-based** (RDED): cropping class-relevant visual regions.

### 3. Soft Label Matching

A single pre-trained ResNet-18 with KL divergence is used uniformly to generate soft labels, formulated as:

$$\theta_{\mathcal{S}}^{t+1} = \arg\min_{\theta \in \Theta} L_{KL}(f_{\theta_\mathcal{T}}(\mathcal{A}(\mathcal{S})), f_{\theta_\mathcal{S}^t}(\mathcal{A}(\mathcal{S})))$$

## Key Experimental Results

### Main Results: Pre- and Post-Rectification Comparison on ImageNet-1K (IPC=10, ResNet-18)

| Method | Before Rectification | After Rectification | Change |
|--------|----------------------|---------------------|--------|
| SRe2L | 21.3 | 40.2 | +18.9↑ |
| CDA | 33.5 | 41.2 | +7.7↑ |
| G-VBSM | 31.4 | 41.5 | +10.1↑ |
| DWA | 37.9 | 42.5 | +4.6↑ |
| EDC | 48.6 | 46.9 | -1.5↓ |
| Minimax | 44.3 | 45.9 | +1.6↑ |
| D4M | 27.9 | 45.4 | +17.5↑ |
| RDED | 42.0 | 46.3 | +4.3↑ |

**Key Finding**: After rectification, the performance gap among methods narrows from 27.3% to 6.7%, demonstrating that most reported gains originate from evaluation settings rather than distillation quality.

### Cross-Dataset Comprehensive Comparison (Rectified, IPC=50)

| Dataset | SRe2L | CDA | DWA | EDC | Minimax | D4M | RDED |
|---------|-------|-----|-----|-----|---------|-----|------|
| CIFAR-10 | 53.9 | 54.5 | 59.9 | **64.8** | — | 61.9 | 63.3 |
| CIFAR-100 | 54.4 | 56.2 | 62.1 | **65.2** | — | 64.3 | 64.1 |
| TinyImageNet | 52.5 | 53.0 | 54.2 | 57.1 | 54.4 | 53.8 | **58.7** |
| ImageNet-1K | 55.2 | 56.7 | 57.7 | **60.1** | 60.4 | 60.2 | 58.9 |

Under unified settings, EDC (optimization-based) achieves the best overall performance, though the gaps among methods are substantially reduced.

## Highlights & Insights

1. **Exposing a critical field-wide issue**: This is the first systematic demonstration that inconsistent evaluation settings constitute the primary confounding factor in decoupled distillation comparisons.
2. **Practical calibration contribution**: Unifying three key settings—small batch size (50), smoothed learning rate, and RDED augmentation—suffices to eliminate most reported performance discrepancies.
3. **Methodological significance**: Provides a fair and reproducible benchmark for future research, preventing "leaderboard manipulation via evaluation tuning."
4. **Impact of simple techniques**: Real-data initialization for optimization-based methods is found to yield substantial performance improvements.

## Limitations & Future Work

- The work primarily focuses on the post-evaluation stage without deeply analyzing intrinsic quality differences among synthetic datasets.
- The unified settings may obscure certain methods' advantages in specific scenarios.
- Systematic comparison along additional evaluation dimensions such as computational cost is not considered.
- Generation-based methods (e.g., Minimax) are not applicable to small datasets (CIFAR-10/100).

## Related Work & Insights

- **Bi-level distillation**: DC, DM, MTT — effective at small scales but lack scalability.
- **Decoupled distillation**: SRe2L as the pioneering work, followed by refinements such as CDA and EDC.
- **Soft label matching**: Epoch-wise soft labels have become standard practice in large-scale distillation.
- **Dataset distillation surveys**: Focus on methodological progress but lack unified evaluation.

## Rating

| Dimension | Score | Remarks |
|-----------|-------|---------|
| Novelty | ⭐⭐⭐⭐ | Reveals an important and overlooked evaluation consistency issue |
| Practicality | ⭐⭐⭐⭐⭐ | Provides fair evaluation standards for the community, directly promoting field standardization |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 6 datasets, 8 methods, comprehensive coverage of IPC 1–100 |
| Writing Quality | ⭐⭐⭐⭐ | Clear argumentation with persuasive figures and tables |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](understanding_dataset_distillation_via_spectral_filtering.md)
- [\[ICLR 2026\] Grounding and Enhancing Informativeness and Utility in Dataset Distillation](grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)

<!-- RELATED:END -->
