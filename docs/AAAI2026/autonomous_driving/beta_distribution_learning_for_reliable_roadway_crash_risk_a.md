---
title: >-
  [Paper Note] Beta Distribution Learning for Reliable Roadway Crash Risk Assessment
description: >-
  [AAAI 2026][Autonomous Driving][Beta Distribution] A geospatial deep learning framework based on Beta distribution learning is proposed to predict the full probability distribution (rather than point estimates) of fatal roadway crash risks from multi-scale satellite imagery, achieving a 17-23% improvement in Recall and naturally expressing uncertainty through the shape of the distribution.
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Beta Distribution"
  - "Roadway Crash Risk Assessment"
  - "Satellite Imagery"
  - "Uncertainty Quantification"
  - "Calibration"
date: 2026-05-08
content_hash: 7c35f7090217294a
---

# Beta Distribution Learning for Reliable Roadway Crash Risk Assessment

**Conference**: AAAI 2026  
**arXiv**: [2511.04886](https://arxiv.org/abs/2511.04886)  
**Code**: [https://www.gb-liang.com/projects/betarisk](https://www.gb-liang.com/projects/betarisk)  
**Area**: Traffic Safety / Autonomous Driving / Uncertainty Estimation  
**Keywords**: Beta Distribution, Roadway Crash Risk Assessment, Satellite Imagery, Uncertainty Quantification, Calibration

## TL;DR

A geospatial deep learning framework based on Beta distribution learning is proposed to predict the full probability distribution (rather than point estimates) of fatal roadway crash risks from multi-scale satellite imagery, achieving a 17-23% improvement in Recall and naturally expressing uncertainty through the shape of the distribution.

## Background & Motivation

**Background**: Road traffic crashes cause over 1.3 million deaths globally each year, causing economic losses of up to 3% of GDP. Traditional traffic safety research often analyzes driving behavior, road infrastructure, traffic patterns, and weather independently, neglecting the complexity of multi-factor spatial interactions.

**Limitations of Prior Work**: Existing DNN risk estimators provide point estimates and fail to convey the model's level of uncertainty; modern DNNs are widely miscalibrated, with predictive confidence mismatched with true accuracy. Crash data is extremely sparse (the annual crash rate for a 25m² road segment in the US is only about 0.1%), making traditional estimation methods highly unreliable.

**Key Challenge**: Safety-critical applications require models to be both (a) high-recall (not missing hazardous areas) and (b) well-calibrated (predictive confidence must realistically reflect correct probability). However, point estimates cannot distinguish between "certain low risk" and "uncertain medium risk".

**Goal**: Starting from satellite imagery, to learn a road fatal crash risk assessment model that is both accurate and well-calibrated, and capable of outputting a full probability distribution.

**Key Insight**: To model risk estimation as a Beta probability distribution learning problem, leveraging the natural [0,1] support and flexible shape parameters of the Beta distribution to represent risk and uncertainty.

**Core Idea**: By predicting the (α, β) parameters of the Beta distribution instead of a single risk value, the geometric information from data augmentation is converted into structured probabilistic supervision signals, enabling uncertainty-aware assessment of crash risks.

## Method

### Overall Architecture

Inputs: satellite image patches of three resolutions (1.19, 0.60, 0.30 m/pixel) $\to$ extract features using a shared ResNet-50 backbone $\to$ concatenate along the channel dimension $\to$ two parallel prediction heads:

| Component | Output | Function | Used during inference? |
|------|------|------|--------------|
| Beta Distribution Head | Two positive scalars (α, β) | Defines the Beta(α,β) distribution; the mean R=α/(α+β) serves as the risk score | ✓ |
| Auxiliary Classification Head | Single logit | Binary classification (crash / no crash) to help the backbone learn discriminative features | ✗ |

### Key Designs

**Beta Probability Modeling**

- **Function**: The model outputs two shape parameters (α, β) of the Beta distribution, instead of a single risk scalar.
- **Mechanism**: A sharp Beta distribution (large α+β) indicates high confidence, while a wide distribution (small α+β) represents high uncertainty. The same mean of 0.5 can correspond to two completely different semantics: "highly confident medium risk" (α=10, β=10) vs. "extremely uncertain" (α=2, β=2).
- **Design Motivation**: Safety-critical scenarios require models to not only provide predictions but also express their level of certainty. The Beta distribution is naturally defined on [0,1], perfectly matching the range of risk probabilities.

**Procedural Target Distribution Generation**

- **Function**: Dynamically generates target Beta distributions as supervision signals based on the geometric properties of random crop augmentations.
- **Mechanism**: For positive samples, an influence score is calculated as 0.7×(1-normalized distance) + 0.3×relative crop size, which is used to modulate the target mean and concentration parameters. The closer to the crash center and the larger the crop, the higher and more concentrated the target distribution mean is.
- **Design Motivation**: Crash risk decays continuously in space—as a crop deviates from the crash center, the visual evidence weakens, so the target distribution should be flatter with a lower mean. This method elevates data augmentation from a simple regularization technique to a rich source of structured supervision signals.

**Multi-Scale Input Design**

- **Function**: Three satellite image inputs of different resolutions are fed into the same backbone.
- **Mechanism**: High-resolution imagery captures local road details (lane markings, intersection geometry), while low-resolution imagery captures macro-environmental context (urban density, surrounding facilities).
- **Design Motivation**: Crash risk is jointly determined by local road characteristics and macro-environmental contexts.

### Loss & Training

Compound loss function: $\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{BCE} + \lambda_2 \cdot \mathcal{L}_{W_2^2}$, where $\lambda_1=5, \lambda_2=1$.

- $\mathcal{L}_{W_2^2}$ is the mean-variance surrogate form of the Wasserstein-2 distance: $(\mu_p - \mu_t)^2 + (\sigma_p - \sigma_t)^2$, which directly optimizes the risk score (mean) and confidence (standard deviation) simultaneously.
- Compared to KL divergence, the $W_2^2$ surrogate provides more stable gradients when the predicted and target distributions have little overlap.
- A larger $\lambda_1$ (=5) is utilized to prioritize classification capability and recall.

Training details: 75 epochs, AdamW + CosineAnnealingWarmRestarts, distribution head learning rate = 0.02, backbone learning rate = 1e-4, batch size 48 (multi-scale), NVIDIA A100.

## Key Experimental Results

### Main Results

Using the MSCM dataset (four major cities in Texas, containing 80,276 geographic locations and 240,828 multi-scale satellite images):

| Method | F1 | Precision | Recall | AUC | ECE↓ | Brier↓ |
|------|-----|-----------|--------|------|------|--------|
| ImageNet | 0.4753 | 0.4968 | 0.4555 | 0.7980 | 0.1281 | 0.1600 |
| MSCM-SS | 0.4966 | 0.4981 | 0.4950 | 0.8165 | 0.1006 | 0.1458 |
| MSCM-MS | 0.5409 | **0.6731** | 0.4521 | 0.8572 | 0.1067 | 0.1296 |
| **Prob-MS (Ours)** | **0.5762** | 0.6296 | **0.5311** | **0.8663** | **0.0881** | **0.1211** |

Prob-MS improves on the most critical Recall metric by 17.5% compared to MSCM-MS, while achieving the lowest ECE (Expected Calibration Error).

### Ablation Study

Deep ensemble comparison—single model vs. three-model ensemble:

| Method | F1 | Recall | ECE↓ | Brier↓ | Variance↓ | Disagr. Rate↓ |
|------|-----|--------|------|--------|-----------|---------------|
| Ensemble MSCM-MS (3 models) | 0.5966 | 0.5165 | 0.0787 | 0.1112 | 0.0925 | 16.93% |
| Ensemble Prob-MS (3 models) | **0.5976** | **0.5361** | **0.0605** | **0.1075** | **0.0822** | **15.14%** |
| Single Model Prob-MS | 0.5762 | 0.5311 | 0.0881 | 0.1211 | — | — |

Single-model Prob-MS is close in performance to the ensemble MSCM-MS (which has $3\times$ the computational cost), and outperforms it by over 3% in Recall.

### Key Findings

- Baseline models produce highly polarized predictions (concentrated near 0 and 1), whereas Prob-MS utilizes the full probability spectrum to represent different levels of certainty.
- Incorrect predictions (FP/FN) of the model are accompanied by higher uncertainty—indicating that the model is capable of "knowing what it does not know."
- San Antonio River Walk case study: Prob-MS successfully identifies multiple fatal crash sites missed by MSCM-MS and generates spatially more coherent risk maps.

## Highlights & Insights

- **Data Augmentation to Probabilistic Supervision**: Converting geometric attributes of random cropping into structured Beta distribution targets. This is a highly generalizable concept that can be adapted to other tasks requiring spatial-decay supervision.
- **Trustworthy Failure Modes**: Even when predictions are incorrect, high uncertainty provides valuable safety signals for downstream decision-making.
- **Reliance Solely on Public Satellite Imagery**: No infrastructure like traffic sensors or road cameras is required, offering global scalability.
- **$W_2^2$ Surrogate Loss**: More stable than KL divergence and directly optimizes mean and standard deviation, keeping the error only in the range of $10^{-3}$ to $10^{-2}$.

## Limitations & Future Work

- Current models estimate only static geographic risk, omitting dynamic factors such as real-time traffic volume, weather, and time of day.
- The geographical scope is limited to Texas; differences in climate, road design, and driving culture could impact generalizability.
- The centrality weight of 0.7 and size weight of 0.3 are manually configured; they could be dynamically adjusted via learnable mechanisms.
- The methodology essentially reflects correlation analysis rather than causal inference—the model learns associations between visual features and crashes, which does not imply causation.

## Related Work & Insights

- **vs MSCM-MS**: Shifted from deterministic classification to probability distribution learning, yielding a +17% Recall improvement and substantially improved calibration.
- **vs Deep Ensemble**: A single model achieves performance comparable to ensembles, with only 1/3 of the computational cost.
- **vs Monte Carlo Simulation Methods**: Eliminates the need for complex hyperparameter tuning and high computing costs, enabling near-real-time inference.
- **Insights**: Beta distribution learning can be extended to other safety-critical uncertainty estimation tasks like medical imaging and natural disaster risk assessment; procedural label generation is a highly generalizable training strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of Beta distribution modeling and procedural label generation is simple yet highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Highly comprehensive coverage featuring quantitative, qualitative, case studies, and ensemble comparisons, though evaluated primarily on a single region.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and excellent visualizations.
- Value: ⭐⭐⭐⭐ — The idea of uncertainty-aware prediction is highly inspiring and worth borrowing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving](../../ICCV2025/autonomous_driving/od-rase_ontology-driven_risk_assessment_and_safety_enhancement_for_autonomous_dr.md)
- [\[CVPR 2026\] LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization](../../CVPR2026/autonomous_driving/leader_lidar_relocalization.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](../../CVPR2026/autonomous_driving/neural_distribution_prior_for_lidar_ood_detection.md)
- [\[CVPR 2026\] Reliable Policy Transfer for Safety-Aware End-to-End Driving with Deep Reinforcement Learning](../../CVPR2026/autonomous_driving/reliable_policy_transfer_for_safety-aware_end-to-end_driving_with_deep_reinforce.md)
- [\[CVPR 2026\] Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation](../../CVPR2026/autonomous_driving/learning_to_identify_out-of-distribution_objects_for_3d_lidar_anomaly_segmentati.md)

</div>

<!-- RELATED:END -->
