---
title: >-
  [Paper Note] DP-GenG: Differentially Private Dataset Distillation Guided by DP-Generated Data
description: >-
  [AAAI2026][Model Compression][differential privacy] This paper proposes DP-GenG, a framework that leverages differentially private generated data (DP-generated data) to guide three stages of dataset distillation — initia…
tags:
  - "AAAI2026"
  - "Model Compression"
  - "differential privacy"
  - "Dataset Distillation"
  - "DP-Generated Data"
  - "Feature Matching"
  - "Privacy Budget Allocation"
date: 2026-05-08
content_hash: 6c7f3ce83ce2379a
---

# DP-GenG: Differentially Private Dataset Distillation Guided by DP-Generated Data

**Conference**: AAAI2026  
**arXiv**: [2511.09876](https://arxiv.org/abs/2511.09876)  
**Code**: [shuoshiss/DP-GENG](https://github.com/shuoshiss/DP-GENG)  
**Area**: Model Compression  
**Keywords**: differential privacy, Dataset Distillation, DP-Generated Data, Feature Matching, Privacy Budget Allocation

## TL;DR

This paper proposes DP-GenG, a framework that leverages differentially private generated data (DP-generated data) to guide three stages of dataset distillation — initialization, feature matching, and expert calibration — significantly improving the utility and privacy protection of the distilled dataset under a limited privacy budget.

## Background & Motivation

Dataset Distillation (DD) compresses a large dataset into a small one while preserving model training performance. Although the distilled dataset is compact, recent studies have shown that standard DD methods lack formal privacy guarantees and may still leak sensitive information from the original data, making them vulnerable to Membership Inference Attacks (MIA).

Existing differentially private dataset distillation (DP-DD) methods (e.g., PSG, NDPDC) inject Gaussian noise during the distillation process to provide privacy guarantees, but suffer from two key limitations:

1. **Insufficient realism (L1)**: Due to the lack of direct access to natural data, distilled samples exhibit poor visual and semantic coherence. Random Gaussian noise initialization is typically used, resulting in low-quality distilled datasets.
2. **Excessive noise (L2)**: Under a limited privacy budget, a large amount of noise must be injected, further degrading dataset quality and utility. Training multiple feature extractors requires splitting the privacy budget, leaving each extractor with a smaller share.

The authors observe that recent DP synthetic data generation techniques (e.g., PE, PrivImage) can produce synthetic data closely resembling the original data distribution, and by the post-processing property of DP, these data can be freely used in downstream computation without incurring additional privacy cost. This observation motivates the core idea of the paper.

## Core Problem

1. How can DP-generated data be used to guide the distillation process and address the realism and noise issues in existing DP-DD methods?
2. How can the utility of the distilled dataset be maximized under a limited privacy budget?

## Method

DP-GenG consists of three core components, all centered around DP-generated data:

### 1. DP Data Generation

Existing DP image synthesis methods (e.g., PE or PrivImage) are used to generate a large volume of synthetic data from the original private dataset. These methods inject Gaussian noise at different stages — input-level, model-level, or output-level — to ensure privacy. The resulting synthetic dataset inherits the privacy guarantees of the generation process and, by the DP post-processing theorem, can be freely used in subsequent computations without additional privacy cost. This paper adopts $\mu$-GDP (Gaussian Differential Privacy) as the privacy accounting framework, which provides tighter privacy bounds than RDP.

### 2. DP Feature Matching

Feature matching is the core algorithmic component of distillation and consists of three sub-steps:

- **DP-generated data initialization**: The distilled dataset is initialized with DP-generated data rather than Gaussian noise. Representative samples are selected from the generated data via strategies such as k-means clustering, and a parameterization technique is used to embed multiple DP synthetic images into a single image to maximize information utilization. This directly addresses L1.
- **Training feature extractors on DP-generated data**: Multiple feature extractors are trained on DP-generated data, avoiding the privacy cost of training on the private dataset. By the post-processing property, these feature extractors consume no additional privacy budget, addressing L2.
- **Feature matching with DP noise injection**: The original private dataset is used for feature matching, with Gaussian noise injected during the process. Features are first clipped to bound the sensitivity, followed by noise addition. Using the subsampling theorem of GDP, the noise magnitude can be reduced proportionally to the sampling probability $p$.

### 3. DP Expert Guidance

DP noise may cause the feature representations of distilled samples to deviate from their original class. To address this, an expert model is introduced as a calibrator:

- The model is first pre-trained on DP-generated data, then fine-tuned on the original private data using DP-SGD.
- For each distilled sample, reference points are sampled from DP-generated data of the same class.
- A KL divergence loss aligns the soft-label distributions of distilled samples and reference points, while a cross-entropy loss preserves class-label consistency.
- Since reference points are drawn from DP-generated data, no additional privacy cost is incurred.

### 4. Privacy Budget Allocation Strategy

The three components jointly consume the privacy budget: generation $\mu_G$, feature matching $\mu_F$, and expert training $\mu_E$. The total privacy parameter is computed via the GDP composition lemma: $\mu_{total} = \sqrt{\mu_G^2 + \mu_F^2 + \mu_E^2}$.

Allocation strategy: the noise levels for the generator and expert model are determined first (via binary search to reach target FID and target accuracy), and the noise level for feature matching $\sigma_F$ is then derived from the remaining privacy budget. The final privacy guarantee is reported by converting from GDP to $(\epsilon, \delta)$-DP.

## Key Experimental Results

Evaluation is conducted on CIFAR-10, CIFAR-100, and CelebA using ConvNet as the default backbone:

| Dataset | IPC | ε=1 DP-GenG | ε=1 NDPDC | ε=10 DP-GenG | ε=10 NDPDC |
|---------|-----|-------------|-----------|--------------|------------|
| CIFAR-10 | 50 | **56.9%** | 42.6% | **65.5%** | 53.9% |
| CIFAR-100 | 50 | **25.9%** | 11.5% | **32.3%** | 19.2% |
| CelebA | 50 | **82.1%** | 80.4% | **85.7%** | 82.3% |

- At ε=10, DP-GenG on CIFAR-10 (65.5%) approaches the non-private DM (64.0%) and surpasses DM at IPC=10.
- MIA experiments: DP-GenG achieves TPR@0.1%FPR comparable to PSG/NDPDC (approximately 0.10–0.14), far below the non-DP DM (0.82), confirming effective privacy protection.

Ablation study (CIFAR-10, IPC=50, ε=10):

- DP initialization only: 48.7%
- DP feature matching only: 53.2%
- DP initialization + DP feature matching: 60.8%
- Full DP-GenG (all three components): **65.5%**

## Highlights & Insights

1. **Clever exploitation of the DP post-processing property**: All operations performed on DP-generated data — initialization, training feature extractors, and sampling reference points — consume no additional privacy budget, which is the key to efficient privacy budget utilization.
2. **Systematic framework**: DP-generated data is integrated across all three stages of the distillation pipeline, with each stage addressing a specific problem.
3. **Unified theory and practice**: A complete privacy analysis and budget allocation strategy are provided, and the GDP framework yields tighter privacy bounds than RDP.
4. **Comprehensive evaluation**: Experiments are conducted on more challenging datasets (CIFAR-100), with MIA evaluation and multi-architecture generalization experiments included.

## Limitations & Future Work

1. **Dependence on DP generator quality**: The performance ceiling of the framework is constrained by the quality of the DP data generator (PE/PrivImage); if the generator performs poorly on complex datasets, overall performance will be limited.
2. **Restricted to the image domain**: The framework is validated only on image classification tasks; applicability to other modalities such as tabular data and text remains unexplored.
3. **Computational overhead**: A DP generator must first be trained to produce a large volume of synthetic data, followed by training of multiple feature extractors and an expert model, resulting in higher computational costs than direct distillation.
4. **Privacy budget allocation**: The selection of target FID and target accuracy still requires manual tuning; an automated allocation strategy is lacking.

## Related Work & Insights

| Method | Type | Privacy Framework | Initialization | Feature Extractor | Noise Calibration |
|--------|------|-------------------|----------------|-------------------|-------------------|
| PSG | Gradient matching DP-DD | RDP | Gaussian noise | None | None |
| NDPDC | Distribution matching DP-DD | RDP | Gaussian noise | Random initialization | None |
| DP-GenG | Feature matching DP-DD | GDP | DP-generated data | Trained on DP-generated data | Expert model |

Compared to DP synthetic data generation methods (PE, PrivImage): the latter directly generate large volumes of synthetic data as output, whereas DP-GenG uses generated data to assist the distillation process, yielding a more compact and information-rich distilled dataset under the same storage budget.

**Broader implications**:
- The DP post-processing property is a powerful yet often underestimated tool — once a DP-compliant intermediate product is available, all subsequent operations are "free." This principle can be generalized to other privacy-preserving learning scenarios.
- The "generate-then-distill" two-stage pipeline can be analogized to federated learning: DP mechanisms could first aggregate a global synthetic dataset, which then guides local distillation.
- The expert model calibration of distributional drift is analogous to the role of the teacher model in knowledge distillation; lighter-weight alternatives are worth exploring.

## Rating
- Novelty: 8/10 — Systematically integrating DP-generated data across the full DD pipeline is a novel combinatorial contribution.
- Experimental Thoroughness: 8/10 — Experiments cover increasingly challenging datasets with comprehensive ablation and MIA evaluation, though larger-scale datasets are absent.
- Writing Quality: 8/10 — Problem motivation is clear and the framework is described systematically.
- Value: 8/10 — Establishes a new paradigm for privacy-preserving dataset distillation with strong practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[NeurIPS 2025\] DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment](../../NeurIPS2025/model_compression/dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)
- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)

</div>

<!-- RELATED:END -->
