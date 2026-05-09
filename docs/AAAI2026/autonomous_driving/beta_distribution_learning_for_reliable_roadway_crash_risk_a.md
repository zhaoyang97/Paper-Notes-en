---
title: >-
  [Paper Note] Beta Distribution Learning for Reliable Roadway Crash Risk Assessment
description: >-
  [AAAI 2026][Autonomous Driving][Beta Distribution] A geospatial deep learning framework based on Beta distribution learning is proposed, which leverages multi-scale satellite imagery to predict the full probability distribution of fatal crash risk (rather than point estimates), achieving 17–23% improvement in Recall while naturally expressing uncertainty through distribution shape.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Beta Distribution
  - Crash Risk Assessment
  - Satellite Imagery
  - Uncertainty Quantification
  - Calibration
date: 2026-05-08
content_hash: f750cb02b42fa378
---

# Beta Distribution Learning for Reliable Roadway Crash Risk Assessment

**Conference**: AAAI 2026
**arXiv**: [2511.04886](https://arxiv.org/abs/2511.04886)
**Code**: [https://www.gb-liang.com/projects/betarisk](https://www.gb-liang.com/projects/betarisk)
**Area**: Traffic Safety / Autonomous Driving / Uncertainty Estimation
**Keywords**: Beta Distribution, Crash Risk Assessment, Satellite Imagery, Uncertainty Quantification, Calibration

## TL;DR

A geospatial deep learning framework based on Beta distribution learning is proposed, which leverages multi-scale satellite imagery to predict the full probability distribution of fatal crash risk (rather than point estimates), achieving 17–23% improvement in Recall while naturally expressing uncertainty through distribution shape.

## Background & Motivation

**Background**: Road traffic crashes cause over 1.3 million deaths annually worldwide, with economic losses reaching up to 3% of GDP. Traditional traffic safety research typically analyzes driving behavior, road infrastructure, traffic patterns, and weather in isolation, overlooking the complex spatial interactions among multiple factors.

**Limitations of Prior Work**: Existing DNN-based risk estimators produce point estimates without conveying model uncertainty; modern DNNs are generally miscalibrated, with predicted confidence mismatched to actual accuracy. Crash data is extremely sparse (the annual crash rate for 25m² road segments in the U.S. is approximately 0.1%), making traditional estimation methods highly unreliable.

**Key Challenge**: Safety-critical applications require models that are both (a) high-recall—dangerous areas must not be missed—and (b) well-calibrated—predicted confidence must faithfully reflect correctness probability. Point estimates cannot distinguish between "certain low risk" and "uncertain moderate risk."

**Goal**: Starting from satellite imagery, learn a crash fatality risk assessment model that is accurate, well-calibrated, and capable of outputting a complete probability distribution.

**Key Insight**: Risk estimation is framed as a Beta probability distribution learning problem, exploiting the natural $[0,1]$ support and flexible shape parameters of the Beta distribution to represent both risk and uncertainty.

**Core Idea**: By predicting the parameters $(\alpha, \beta)$ of a Beta distribution rather than a single risk value, geometric information from data augmentation is converted into structured probabilistic supervision signals, enabling uncertainty-aware assessment of crash risk.

## Method

### Overall Architecture

Three satellite image crops at different resolutions (1.19, 0.60, 0.30 m/pixel) → shared ResNet-50 backbone for feature extraction → channel-wise concatenation → two parallel prediction heads:

| Component | Output | Function | Used at Inference |
|-----------|--------|----------|-------------------|
| Beta distribution head | $(\alpha, \beta)$, two positive scalars | Defines $\text{Beta}(\alpha,\beta)$; mean $R=\alpha/(\alpha+\beta)$ is the risk score | ✓ |
| Auxiliary classification head | Single logit | Binary classification (crash / no crash); assists backbone in learning discriminative features | ✗ |

### Key Designs

**Beta Probabilistic Modeling**

- **Function**: The model outputs the two shape parameters $(\alpha, \beta)$ of a Beta distribution rather than a single risk scalar.
- **Mechanism**: A sharp Beta distribution (large $\alpha+\beta$) indicates high confidence; a broad distribution (small $\alpha+\beta$) indicates high uncertainty. The same mean of 0.5 can correspond to two semantically distinct cases: "confidently moderate risk" ($\alpha=10, \beta=10$) vs. "highly uncertain" ($\alpha=2, \beta=2$).
- **Design Motivation**: Safety-critical applications require the model to express not only its predictions but also its degree of confidence. The Beta distribution is naturally defined on $[0,1]$, perfectly matching the value range of risk probabilities.

**Programmatic Target Distribution Generation**

- **Function**: Dynamically generates target Beta distributions as supervision signals based on the geometric properties of random crop augmentation.
- **Mechanism**: For positive samples, an influence score is computed as $0.7 \times (1 - \text{normalized distance}) + 0.3 \times \text{relative crop size}$, which modulates the target distribution's mean and concentration. Crops closer to the crash center and of larger size yield higher-mean, more concentrated target distributions.
- **Design Motivation**: Crash risk decays continuously in space—when a crop deviates from the crash center, visual evidence weakens and the target distribution should be flatter with a lower mean. This approach elevates data augmentation from a simple regularization technique to a rich source of structured supervision signals.

**Multi-Scale Input Design**

- **Function**: Three satellite image crops at different resolutions are fed into the same backbone.
- **Mechanism**: High-resolution inputs capture local road details (lane markings, intersection geometry); low-resolution inputs capture macro-scale environmental context (urban density, surrounding infrastructure).
- **Design Motivation**: Crash risk is jointly determined by local road characteristics and broader environmental factors.

### Loss & Training

A composite loss function $\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{BCE} + \lambda_2 \cdot \mathcal{L}_{W_2^2}$, where $\lambda_1=5, \lambda_2=1$.

- $\mathcal{L}_{W_2^2}$ is a mean–variance surrogate for the Wasserstein-2 distance: $(\mu_p - \mu_t)^2 + (\sigma_p - \sigma_t)^2$, jointly optimizing the risk score (mean) and confidence (standard deviation).
- Compared to KL divergence, the $W_2$ surrogate provides more stable gradients when the predicted and target distributions have limited overlap.
- The larger weight $\lambda_1=5$ prioritizes classification capability and recall.

Training details: 75 epochs, AdamW + CosineAnnealingWarmRestarts, distribution head lr = 0.02, backbone lr = 1e-4, batch size 48 (multi-scale), NVIDIA A100.

## Key Experimental Results

### Main Results

Evaluated on the MSCM dataset (four major cities in Texas; 80,276 geographic locations; 240,828 multi-scale satellite images):

| Method | F1 | Precision | Recall | AUC | ECE↓ | Brier↓ |
|--------|----|-----------|--------|-----|------|--------|
| ImageNet | 0.4753 | 0.4968 | 0.4555 | 0.7980 | 0.1281 | 0.1600 |
| MSCM-SS | 0.4966 | 0.4981 | 0.4950 | 0.8165 | 0.1006 | 0.1458 |
| MSCM-MS | 0.5409 | **0.6731** | 0.4521 | 0.8572 | 0.1067 | 0.1296 |
| **Prob-MS (Ours)** | **0.5762** | 0.6296 | **0.5311** | **0.8663** | **0.0881** | **0.1211** |

Prob-MS improves Recall by 17.5% over MSCM-MS on the most critical metric, while achieving the lowest ECE (calibration error).

### Ablation Study

Deep ensemble comparison—single model vs. three-model ensemble:

| Method | F1 | Recall | ECE↓ | Brier↓ | Variance↓ | Disagr. Rate↓ |
|--------|----|--------|------|--------|-----------|---------------|
| Ensemble MSCM-MS (3 models) | 0.5966 | 0.5165 | 0.0787 | 0.1112 | 0.0925 | 16.93% |
| Ensemble Prob-MS (3 models) | **0.5976** | **0.5361** | **0.0605** | **0.1075** | **0.0822** | **15.14%** |
| Single Prob-MS | 0.5762 | 0.5311 | 0.0881 | 0.1211 | — | — |

The single-model Prob-MS already approaches the performance of ensemble MSCM-MS at three times the computational cost, and surpasses it by over 3% in Recall.

### Key Findings

- Baseline models produce severely polarized predictions (concentrated near 0 and 1), whereas Prob-MS utilizes the full probability spectrum to express varying degrees of confidence.
- Erroneous predictions (FP/FN) are consistently associated with higher uncertainty—indicating that the model can "know when it is uncertain."
- San Antonio River Walk case study: Prob-MS correctly identifies multiple fatal crash locations missed by MSCM-MS and produces spatially more coherent risk maps.

## Highlights & Insights

- **Data Augmentation → Probabilistic Supervision**: Converting geometric properties of random crops into structured Beta distribution targets is a generalizable idea transferable to other tasks requiring spatially decaying supervision.
- **Trustworthy Failure Modes**: Even when predictions are incorrect, high uncertainty provides valuable safety signals for downstream decision-making.
- **Relies Solely on Public Satellite Imagery**: No traffic sensors, road-side cameras, or other infrastructure required, enabling global scalability.
- **$W_2$ Surrogate Loss**: More stable than KL divergence and directly optimizes mean and standard deviation jointly, with approximation errors on the order of $10^{-3}$ to $10^{-2}$.

## Limitations & Future Work

- Only static geographic risk is estimated; real-time traffic flow, weather, time-of-day, and other dynamic factors are not considered.
- Geographic coverage is limited to Texas; differences in climate, road design, and driving culture may affect generalization.
- The centrality weight 0.7 and size weight 0.3 are manually set and could be replaced by a learnable adaptive mechanism.
- The approach is fundamentally correlational rather than causal—the model learns associations between visual features and crashes, which do not imply causality.

## Related Work & Insights

- **vs. MSCM-MS**: Transitioning from deterministic classification to probabilistic distribution learning yields +17% Recall and substantially improved calibration.
- **vs. Deep Ensemble**: A single model achieves ensemble-level performance at one-third the computational cost.
- **vs. Monte Carlo Simulation Methods**: No complex parameter tuning or high computational overhead required; enables near-real-time inference.
- **Insights**: Beta distribution learning can be extended to other safety-critical uncertainty estimation tasks such as medical imaging and disaster risk assessment; programmatic label generation is a generalizable training strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of Beta distribution learning and programmatic label generation is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers quantitative evaluation, qualitative analysis, case studies, and ensemble comparisons, though limited to a single region.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and excellent visualizations.
- Value: ⭐⭐⭐⭐ — The uncertainty-aware prediction paradigm offers strong transferable insights.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving](../../ICCV2025/autonomous_driving/od-rase_ontology-driven_risk_assessment_and_safety_enhancement_for_autonomous_dr.md)
- [\[CVPR 2026\] LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization](../../CVPR2026/autonomous_driving/leader_lidar_relocalization.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](../../CVPR2026/autonomous_driving/neural_distribution_prior_for_lidar_ood_detection.md)
- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)

<!-- RELATED:END -->
