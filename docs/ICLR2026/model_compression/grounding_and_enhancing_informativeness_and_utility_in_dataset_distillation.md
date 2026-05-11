---
title: >-
  [Paper Note] Grounding and Enhancing Informativeness and Utility in Dataset Distillation
description: >-
  [ICLR 2026][Model Compression][Dataset Distillation] This paper proposes InfoUtil, a framework that maximizes sample informativeness via game-theoretic Shapley Values (to identify the most critical patches) and maximizes…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Dataset Distillation"
  - "Shapley Value"
  - "Gradient Norm"
  - "Informativeness"
  - "Utility"
date: 2026-05-08
content_hash: 20e6995c5441723b
---

# Grounding and Enhancing Informativeness and Utility in Dataset Distillation

**Conference**: ICLR 2026
**arXiv**: [2601.21296](https://arxiv.org/abs/2601.21296)
**Code**: None
**Area**: Dataset Distillation
**Keywords**: Dataset Distillation, Shapley Value, Gradient Norm, Informativeness, Utility

## TL;DR
This paper proposes InfoUtil, a framework that maximizes sample informativeness via game-theoretic Shapley Values (to identify the most critical patches) and maximizes sample utility via gradient norms (to select the most training-valuable samples), achieving a 6.1% improvement over the previous SOTA on ImageNet-1K.

## Background & Motivation

**Background**: Dataset distillation aims to synthesize compact datasets from large ones such that models trained on them approximate the performance of those trained on the full data. Mainstream approaches fall into two categories: matching-based methods (e.g., gradient matching, trajectory matching) and knowledge distillation-based methods. The latter (e.g., RDED) achieve superior performance but lack theoretical justification.

**Limitations of Prior Work**: Matching-based methods struggle to balance efficiency and performance (e.g., trajectory matching requires 4× A100 GPUs); knowledge distillation-based methods such as RDED rely on random cropping and heuristic scoring to select patches, lacking principled theoretical guarantees — randomly selected patches frequently miss semantically critical regions.

**Key Challenge**: How to simultaneously address two problems within a theoretically interpretable framework: (1) which regions within each sample are most informative, and (2) which samples are most valuable for training (Utility)?

**Goal**: To establish a theoretical foundation for dataset distillation, formally define optimal distillation, and design algorithms accordingly.

**Key Insight**: The paper introduces two concepts — Informativeness (at the patch level, measuring information content) and Utility (at the sample level, measuring training value) — mathematically defines optimal distillation, employs Shapley Values for informativeness attribution, and uses gradient norms for utility estimation.

**Core Idea**: Selecting the most informative patches via Shapley Values and the most valuable samples via gradient norms yields theoretically grounded optimal distillation.

## Method

### Overall Architecture
A two-stage pipeline: Step 1 applies Shapley Values to identify the most informative patches per sample for compression, yielding a compressed dataset $\mathcal{D}'$; Step 2 evaluates the training value of each compressed sample via gradient norms and selects the top-$m$ samples to form the distilled dataset $\tilde{\mathcal{D}}$. The selected samples are then reconstructed to full resolution and annotated with soft labels.

### Key Designs

1. **Game-theoretic Informativeness Maximization**:

    - Function: Applies Shapley Value attribution to identify the most important patches within each image.
    - Mechanism: The image is modeled as a cooperative game where each patch is a player: $\phi_f(x^{(i)}) = \frac{1}{d}\sum_{s:s_i=0}\binom{d-1}{\mathbf{1}^\top s}(f(x\circ(s+e_i)) - f(x\circ s))$. KernelShap is used for efficient estimation. Patches with the highest Shapley values are retained.
    - Design Motivation: Shapley Values are the unique attribution method satisfying four axioms simultaneously — linearity, dummy, symmetry, and efficiency — providing the strongest theoretical foundation.

2. **Principled Utility Maximization**:

    - Function: Evaluates the training importance of each sample via gradient norms and selects the top-$m$ samples.
    - Mechanism: Theorem 1 proves that the utility function $\mathcal{U}$ is upper-bounded by the gradient norm: $\mathcal{U}(x_i, y_i; f_{\theta^{(t)}}) \leq c\|\nabla_{\theta^{(t)}}\ell_t(f_{\theta^{(t)}}(x_i), y_i)\|$.
    - Design Motivation: Directly computing utility requires counterfactual experiments for each sample, which is computationally prohibitive. The gradient norm serves as a tractable upper bound — larger values indicate greater influence on training.

3. **Diversity Control**:

    - Function: Introduces diversity in patch selection by adding random noise to Shapley attribution heatmaps.
    - Mechanism: $\phi + \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, \sigma^2)$.
    - Design Motivation: Pure Shapley attribution may consistently select the same regions; injecting noise encourages different informative regions to be selected across samples.

### Loss & Training
- Shapley Values are estimated using KernelShap to avoid $2^{16}$ inference calls.
- Gradient norms are computed using intermediate checkpoints of the teacher model.
- Images are compressed to 1/4 resolution; four compressed images are tiled into one full-resolution image.
- Soft labels are obtained from intermediate checkpoints of the teacher model.

## Key Experimental Results

### Main Results
ResNet-18, IPC=50 (50 images per class):

| Dataset | Method | Top-1 Acc | Gain |
|--------|------|-----------|------|
| ImageNet-1K | RDED (Prev. SOTA) | Baseline | — |
| ImageNet-1K | **InfoUtil** | Baseline+6.1% | +6.1% |
| ImageNet-100 | RDED | Baseline | — |
| ImageNet-100 | **InfoUtil** | Baseline+16% | +16% |
| CIFAR-10 IPC50 | RDED | 62.1 | — |
| CIFAR-10 IPC50 | **InfoUtil** | **71.0** | +8.9% |

### Ablation Study

| Configuration | ImageNet-1K Acc | Notes |
|------|----------------|------|
| Full InfoUtil | Best | Shapley + gradient norm |
| Random patch (w/o Shapley) | Significant drop | Informativeness selection is critical |
| Random sample selection (w/o gradient norm) | Drop | Utility ranking is valuable |
| No diversity noise | Slight drop | Diversity is beneficial |

### Key Findings
- Patches selected by Shapley Values align with semantically critical regions (e.g., animal heads rather than backgrounds), whereas RDED's random cropping frequently captures irrelevant backgrounds.
- Gradient norm serves as a simple yet effective utility metric — samples with high gradient norms are empirically more important for training.
- Substantial gains persist on large-scale datasets such as ImageNet-1K (+6.1%), demonstrating scalability.
- Cross-architecture generalization: distilled data obtained with ResNet-18 yields significant improvements when evaluated on ResNet-101.

## Highlights & Insights
- **Rigorous theoretical foundation**: The entire pipeline — from defining optimal distillation (Definition 4) via Informativeness and Utility to approximating them with Shapley Values and gradient norms — is theoretically grounded rather than heuristic.
- **Shapley Value-based image attribution for dataset distillation** is introduced for the first time and substantially outperforms random cropping. This idea is broadly applicable to scenarios requiring identification of the most important regions.
- **The theorem establishing utility as an upper bound of the gradient norm** (Theorem 1) provides an intuitive and computationally tractable surrogate — samples with large gradients exert greater influence on training dynamics and should be prioritized.

## Limitations & Future Work
- Shapley Value computation, even with KernelShap, incurs non-trivial overhead; scalability to datasets exceeding 1M images remains to be validated.
- The compression ratio of 1/4 resolution is fixed; adaptive compression rates may yield further improvements.
- Gradient norms are computed at a single checkpoint; ensemble evaluation across multiple checkpoints may provide more robust utility estimates.
- Validation is limited to classification tasks; dense prediction tasks such as detection and segmentation remain unexplored.

## Related Work & Insights
- **vs. RDED**: RDED relies on random cropping and heuristic scoring, whereas InfoUtil employs Shapley Values and gradient norms with stronger theoretical backing, achieving 6.1%–16% performance gains.
- **vs. SRe2L**: SRe2L is another knowledge distillation-based method; InfoUtil substantially outperforms it across all IPC settings.
- **vs. matching-based methods (MTT/DATM)**: Matching-based methods are competitive on small datasets but do not scale to large ones; InfoUtil performs well across both regimes.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying Shapley Values to patch selection in dataset distillation is novel, with a complete theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 7 datasets, 3 architectures, multiple IPC settings, and cross-architecture generalization.
- Writing Quality: ⭐⭐⭐⭐ Theoretical definitions are clear and theorem proofs are complete.
- Value: ⭐⭐⭐⭐ Establishes a theoretically principled new paradigm for dataset distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](understanding_dataset_distillation_via_spectral_filtering.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[ICLR 2026\] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)
- [\[ICCV 2025\] Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)

</div>

<!-- RELATED:END -->
