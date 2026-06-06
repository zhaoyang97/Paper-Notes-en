---
title: >-
  [Paper Note] BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal Learning] BALM proposes a model-agnostic plug-and-play framework to address multimodal learning under **Imbalanced Missing Rates (IMR)**. It introduces a Feature Calibration Module…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multimodal Learning"
  - "Missing Modality"
  - "Imbalanced Missing Rate"
  - "Gradient Rebalancing"
  - "Feature Calibration"
date: 2026-05-08
content_hash: a368e6715941528f
---

# BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates

**Conference**: CVPR 2026
**arXiv**: [2603.19718](https://arxiv.org/abs/2603.19718)  
**Code**: [https://github.com/np4s/BALM_CVPR2026.git](https://github.com/np4s/BALM_CVPR2026.git)  
**Area**: Multimodal Learning / VLM
**Keywords**: Multimodal Learning, Missing Modality, Imbalanced Missing Rate, Gradient Rebalancing, Feature Calibration

## TL;DR
BALM proposes a model-agnostic plug-and-play framework to address multimodal learning under **Imbalanced Missing Rates (IMR)**. It introduces a Feature Calibration Module (FCM) to align representations across different missing patterns, and a Gradient Rebalancing Module (GRM) to balance the optimization dynamics of each modality from both distributional and spatial perspectives. The framework consistently improves the robustness of various backbone networks across multiple multimodal sentiment recognition benchmarks.

## Background & Motivation

**Background**: Multimodal learning (audio + visual + text) has achieved remarkable progress, yet sensor failures, recording noise, or data acquisition costs frequently result in **partial or complete modality absence** in practice.

**Imbalance Induced by Missing Modalities**:
- **Shared Missing Rate (SMR)** assumption: all modalities are dropped with equal probability — an unrealistic simplification.
- **Imbalanced Missing Rate (IMR)**: different modalities are absent with different probabilities (e.g., audio missing 70% of the time while text is missing only 30%) — this reflects realistic deployment conditions.

**Dual Challenges Introduced by IMR**:
- **(1) Representational Imbalance**: Heterogeneous missing patterns distort the feature distributions of individual modalities, hindering consistent cross-modal fusion.
- **(2) Learning Imbalance**: Gradients are dominated by more frequently available modalities, causing optimization bias and underfitting of less available modalities.
- Gradient contribution is proportional to availability: $\frac{\partial \mathcal{L}}{\partial \theta_m} \propto (1 - r_m)$

**Blind Spots of Existing Methods**:
- Alignment methods (e.g., contrastive learning) and generative methods (e.g., missing modality reconstruction) typically assume SMR or train on complete data.
- Modality imbalance methods (resampling, gradient modulation) generally assume all modalities are fully available.
- Few methods **simultaneously address missingness and imbalance**.

**Core Idea**: Perform rebalancing at both the representation level (feature calibration) and the optimization level (gradient rebalancing) as plug-and-play modules integrated into any existing backbone.

## Method

### Overall Architecture
BALM consists of two complementary modules inserted into the encoding and optimization processes of the backbone:
- **FCM**: Applied before the encoder — calibrates feature representations of each modality.
- **GRM**: Applied during backpropagation — modulates gradients for each modality.

### Key Designs

1. **Feature Calibration Module (FCM)**:

    - **Objective**: Re-calibrate unimodal features distorted by uneven exposure, leveraging global context.
    - **Steps**:
        - Compute a global descriptor per modality: $x_{glob}^m = \frac{\sum_i \tilde{x}_i^m}{\varepsilon + \sum_i e_i^m}$ (averaged over available samples only).
        - Concatenate and extract cross-modal context via an FC layer: $x_{global} = \text{ReLU}(\mathbf{F}_{global}([x_{glob}^{m_1}, ..., x_{glob}^{m_M}]))$
        - Project to modality-specific calibration weights: $w_{cal}^m = \mathbf{F}_{cal}^m(x_{global})$
        - Gated calibration: $\hat{x}_i^m = (1 + \sigma(w_{cal}^m)) \odot \tilde{x}_i^m$
    - **Design Motivation**: Rather than reconstructing missing modalities, FCM leverages global statistics from all available modalities to correct distributional shift within each modality.

2. **Gradient Rebalancing Module (GRM)**:

    - **Auxiliary Structure**: A lightweight unimodal prediction head (two FC layers) is added per modality, with dimensions aligned to the main prediction head.

   **(a) Distribution-driven Modulation**:
    - KL divergence measures the distance between each modality's prediction and the multimodal reference: $\mathcal{D}_{KL}^m = \sum_i \text{KL}(\hat{y}_i^m \| \hat{y}_i)$
    - Relative learning progress is computed: $\Delta_{KL}^m = \mathcal{D}_{KL}^{m^{(t-1)}} - \mathcal{D}_{KL}^{m^{(t)}}$
    - Faster-learning modalities receive reduced gradients; slower-learning modalities receive amplified gradients:
    $$\mu^m = \rho \frac{\sum_{m' \neq m} \Delta_{KL}^{m'}}{\sum_{m'} \Delta_{KL}^{m'}}$$
    - Update rule: $\theta_{(t+1)}^{\phi^m} = \theta_{(t)}^{\phi^m} - \alpha \mu^m \frac{\partial \mathcal{L}}{\partial \theta_{(t)}^{\phi^m}}$

   **(b) Spatial-driven Modulation**:
    - Prediction head gradients approximate the overall model gradient $\nabla_{pred}$ and per-modality gradient $\nabla_{pred}^m$.
    - Directional conflicts between each modality's gradient and the overall gradient are detected.
    - Conflicting components are removed via projection to ensure gradient direction consistency across modalities.
    - **Design Motivation**: Even when gradient magnitudes are balanced, directional conflicts can cause mutual cancellation; spatial modulation ensures coordinated gradient directions.

### Loss & Training
- Main task loss: $\mathcal{L}_{task}$ (cross-entropy or L1, depending on the backbone).
- FCM introduces no additional loss — it only modifies input features.
- GRM introduces no additional loss — it only modifies gradients during backpropagation.
- Fully plug-and-play; the backbone architecture remains unchanged.

## Key Experimental Results

### Main Results (IEMOCAP, 6 IMR configurations, $(r_A, r_L, r_V)$)

| Method | (0.3,0.5,0.7) Acc | (0.5,0.3,0.7) Acc | (0.7,0.3,0.5) Acc | Avg. Stability |
|------|-------------------|-------------------|-------------------|----------|
| MMIN | 55.97 | 56.40 | 55.87 | Consistent but low performance |
| SDR-GNN | 59.46 | 58.53 | 55.14 | High variance |
| Mi-CGA | 58.84 | 60.50 | 58.84 | Moderate |
| MoMKE | 55.39 | 60.18 | 58.26 | High variance |

(BALM improves both performance and stability of each backbone as a plugin; see detailed results in the paper.)

### Ablation Study

| Configuration | Description |
|------|------|
| FCM only | Alleviates representational imbalance; improves feature quality for weaker modalities. |
| GRM only | Balances optimization dynamics; prevents dominant modalities from overfitting. |
| FCM + GRM (full) | Complementary effects yield best performance and stability. |

### Key Findings
- Conventional missing modality methods are unstable under IMR (e.g., SDR-GNN accuracy fluctuates by 4.3% across IMR configurations).
- BALM consistently enhances diverse backbone types (standard MER models, imbalance-specific models, and missing modality methods) as a plug-in.
- Distribution-driven and spatial-driven modulation are complementary — using either alone is inferior to combining both.
- FCM yields more pronounced gains for low-availability modalities; GRM effectively suppresses overfitting in high-availability modalities.

## Highlights & Insights
- The **formalization of IMR** is notably clear — the derivation from SMR to IMR, and the quantification of imbalance degree $\Delta_{IMR}$ and gradient bias, are well-motivated.
- The **plug-and-play design** offers strong practical utility — no modification to the backbone architecture is required, only two lightweight modules need to be inserted.
- **Dual-dimension rebalancing** (representation + optimization) simultaneously addresses both root causes of the problem.
- Replacing missing modality reconstruction with global context calibration is an elegant design choice — rather than "generating" missing information, the framework "corrects" existing information.

## Limitations & Future Work
- Experiments are primarily conducted on multimodal emotion recognition (MER); generalization to other multimodal tasks (e.g., VQA, video understanding) requires further validation.
- FCM's global descriptors are batch-level statistics, which may be unstable when batch sizes are small.
- GRM's distribution-driven modulation relies on multimodal predictions as a reference; if the multimodal fusion itself is biased, the reference signal becomes unreliable.
- Behavior under extreme missing rates (e.g., a modality missing >90% of the time) is not thoroughly analyzed.

## Related Work & Insights
- OGM-GE (Peng et al.) is a pioneering work on gradient modulation for modality imbalance.
- MMIN and Mi-CGA are representative methods for missing modality learning, but neither addresses IMR.
- BALM's gradient rebalancing strategy is potentially generalizable to broader multi-task / multi-branch learning settings.
- The problem upgrade from SMR to IMR provides meaningful guidance for the research community.

## Rating
- Novelty: ⭐⭐⭐⭐ The IMR problem is clearly defined, and the dual-module design is logically coherent, though individual technical components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, diverse backbones, and various IMR configurations provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous; the formal definition of IMR is a valuable contribution.
- Value: ⭐⭐⭐⭐ The model-agnostic plug-and-play framework has direct practical significance for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Calibrated Multimodal Representation Learning with Missing Modalities](../../ICML2026/multimodal_vlm/calibrated_multimodal_representation_learning_with_missing_modalities.md)
- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](../../NeurIPS2025/multimodal_vlm/midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)
- [\[ICML 2026\] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](../../ICML2026/multimodal_vlm/injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)

</div>

<!-- RELATED:END -->
