---
title: >-
  [Paper Note] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather
description: >-
  [ECCV 2024][Autonomous Driving][LiDAR Semantic Segmentation] Identifying two core interference patterns of adverse weather on LiDAR (geometric perturbation and point loss) through a data-centric analysis, this paper proposes two targeted data augmentation methods: Selective Jittering and Learnable Point Drop, achieving SOTA by improving the baseline by 8.1 mIoU on the SemanticKITTI→SemanticSTF benchmark.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "LiDAR Semantic Segmentation"
  - "Adverse Weather Robustness"
  - "Data Augmentation"
  - "Deep Q-Learning"
  - "Point Clouds"
date: 2026-05-08
content_hash: 073b1333cbeaed47
---

# Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather

**Conference**: ECCV 2024  
**arXiv**: [2407.02286](https://arxiv.org/abs/2407.02286)  
**Code**: [Yes](https://github.com/engineerJPark/LiDARWeather)  
**Area**: Autonomous Driving  
**Keywords**: LiDAR Semantic Segmentation, Adverse Weather Robustness, Data Augmentation, Deep Q-Learning, Point Clouds

## TL;DR

Identifying two core interference patterns of adverse weather on LiDAR (geometric perturbation and point loss) through a data-centric analysis, this paper proposes two targeted data augmentation methods: Selective Jittering and Learnable Point Drop, achieving SOTA by improving the baseline by 8.1 mIoU on the SemanticKITTI→SemanticSTF benchmark.

## Background & Motivation

LiDAR semantic segmentation is a fundamental task for 3D scene understanding in autonomous driving. However, the performance of existing models degrades severely under adverse weather (snow, fog, rain, wet roads), which is unacceptable for safety-critical applications.

Existing robustness methods are categorized into two types:

| Method Category | Representative Works | Limitations |
|----------|----------|--------|
| Task-agnostic methods | PointDR, teacher-student frameworks | Do not specifically target the characteristics of LiDAR weather corruption |
| Simulation methods | LiDAR snow/fog simulation | Only consider a single weather condition; precise simulation is difficult; only used for detection tasks |

**Key Insight**: The interference patterns of different weather conditions on LiDAR data are actually similar! Rain, snow, and fog all lead to similar point loss and geometric perturbation patterns. Rather than creating precise simulations for each individual weather, it is more effective to analyze common interference patterns from a data-centric perspective and design a unified augmentation strategy.

## Method

### Overall Architecture

The method consists of three steps:
1. **Analysis Phase**: Identify 4 types of interference patterns of adverse weather on LiDAR data.
2. **Toy Experiments**: Verify which interference patterns are the primary causes of performance degradation.
3. **Augmentation Design**: Design Selective Jittering (SJ) and Learnable Point Drop (LPD) targeting the primary causes.

The Toy experiment results of the 4 types of interference patterns:

| Interference Type | Mechanism | Soft mIoU | Hard mIoU | Performance Impact |
|----------|------|-----------|-----------|----------|
| D1: Point Loss | Energy absorption causes points to disappear | 61.8 | 27.7 | **Severe** |
| D2: Occlusion | Water droplets/snowflakes generate near-range echoes | 57.2 | 24.3 | **Severe** |
| D3: Geometric Perturbation | Refraction causes coordinate shift | 53.2 | 35.0 | **Severe** |
| D4: Intensity Distortion | Energy attenuation causes intensity decrease | 62.3 | 56.8 | Slight |
| Clean | No interference | 63.9 | 63.9 | - |

**Key Findings**: All D1, D2, and D3 lead to performance dropping to less than half of the baseline under the Hard setting, and the unique incorrect prediction patterns of D2 (occlusion) highly overlap with D1 (point loss) $\rightarrow$ they can be unified and treated as point loss. D4 (intensity distortion) has limited impact on performance as it does not change the local geometric structure.

### Key Designs

**1. Selective Jittering (SJ)**

An augmentation targeting geometric perturbations, consisting of three sub-methods:

- **Depth-Selective Jittering (DSJ)**: Adds Gaussian noise to the XYZ coordinates and intensity of points within a random depth range.
- **Angle-Selective Jittering (ASJ)**: Adds Gaussian noise to points within a random angular range.
- **Range Jittering (RJ)**: Jitters all points only along the range direction.

Core Idea: In adverse weather, not all beams are affected; instead, beams within specific depth/angle ranges are affected by the refraction of transparent particles such as water droplets and snowflakes. Therefore, selectively jittering a subset of beams is closer to realistic scenarios.

**2. Learnable Point Drop (LPD)**

An augmentation targeting point loss, which uses a Deep Q-Network (DQN) to learn the most destructive point loss patterns:

Training Flow:
1. Apply SJ augmentation to the input, then calculate loss L_aug and entropy H_aug.
2. The LPD module takes L_aug + H_aug as the current state.
3. DQN predicts the indices of points to be dropped.
4. Recalculate loss L_LPD and entropy H_LPD on the dropped data.
5. Reward = (L_LPD + H_LPD) - (L_aug + H_aug), i.e., finding the point drop pattern that confuses the model the most.

Difference from random dropping: Random dropping reduces points uniformly across all depths, failing to simulate the depth-dependent characteristics of weather like fog; LPD can discover more targeted vulnerable point patterns through learning.

### Loss & Training

- The baseline model is trained using the original cross-entropy loss for semantic segmentation.
- Gaussian noise parameters for SJ: mean 0, standard deviation 0.01.
- The gradient norm in LPD is limited to 100 to stabilize DQN training.
- Trained on 4 A6000 GPUs for 15 epochs with a batch size of 2.
- Training time is 3-5 hours.
- Learning rate is 0.24, and weight decay is 0.0001.

## Key Experimental Results

### Main Results

SemanticKITTI $\rightarrow$ SemanticSTF benchmark:

| Method | Dense Fog | Light Fog | Rain | Snow | mIoU |
|------|-----------|-----------|------|------|------|
| Oracle (Trained on target domain) | 51.9 | 54.6 | 57.9 | 53.7 | 54.7 |
| Baseline (MinkowskiNet) | 30.7 | 30.1 | 29.7 | 25.3 | 31.4 |
| LaserMix | 23.2 | 15.5 | 9.3 | 7.8 | 14.7 |
| PolarMix | 21.3 | 14.9 | 16.5 | 9.3 | 15.3 |
| PointDR | 37.3 | 33.5 | 35.5 | 26.9 | 33.9 |
| **Baseline+SJ+LPD** | **36.0** | **37.5** | **37.6** | **33.1** | **39.5** |
| **Gain** | +5.3 | +7.4 | +7.9 | +7.8 | **+8.1** |

### Ablation Study

| Method | Clean | D-fog | L-fog | Rain | Snow | mIoU |
|------|-------|-------|-------|------|------|------|
| Baseline | 63.9 | 30.7 | 30.1 | 29.7 | 25.3 | 31.4 |
| +ASJ | 62.1 (-1.8) | 33.3 | 35.4 | 37.8 | 31.6 | 36.8 (+5.4) |
| +DSJ | 63.0 (-0.9) | 34.8 | 36.4 | 39.0 | 29.9 | 37.6 (+6.2) |
| +RJ | 61.2 (-2.7) | 33.4 | 37.0 | 35.7 | 33.5 | 38.7 (+7.3) |
| +LPD | 62.8 (-1.1) | 36.0 | 37.5 | 37.6 | 33.1 | **39.5 (+8.1)** |

### Key Findings

- **Huge Performance Gain**: The improvement of +8.1 mIoU is more than 3 times that of the previous SOTA, PointDR (+2.5 mIoU).
- **Consistent Performance Gain Across Weather**: Significant improvements (+5.3 ~ +7.9 mIoU) under all 4 weather conditions.
- **Best Recovery in Rain**: SJ augmentation is most effective for rainy conditions (+7.9), as the geometric perturbation caused by raindrops is selective.
- **Particularity of Snow**: Range Jittering works best for snowy conditions (+8.2), as snowy weather requires perturbing more beams.
- **Minor Cost on Clean Data**: A massive improvement in robustness is achieved at a small cost of only -1.1 mIoU on clean data.
- **Generalization Across Architectures**: Effective on CENet (+7.8), SPVCNN (+10.3), and MinkowskiNet (+8.1).
- **Generalization Across Datasets**: Also achieves a +5.6 mIoU improvement on SemanticKITTI-C.
- General-purpose mix augmentations like LaserMix/PolarMix degrade performance in this scenario instead.

## Highlights & Insights

1. **Data-centric analysis methodology**: Analyze interference patterns first, verify the main causes next, and design targeted augmentations last—this is a reusable research paradigm.
2. **Unification of interference patterns**: Unifying seemingly different weather effects into two common patterns avoids the need for precise simulation of each individual weather condition.
3. **DQN for data augmentation**: Innovatively introduces reinforcement learning to learn the most "effective" data augmentation strategy.
4. **Simple and effective**: Plug-and-play without modifying the training pipeline of the baseline model (only limiting the gradient norm).
5. **Reasonable trade-off**: Only losing 1.1 mIoU on clean data in exchange for a massive 8.1 mIoU improvement in adverse weather.

## Limitations & Future Work

- LPD searches for drop patterns uniformly across all points, failing to target point loss in specific regions like wet ground.
- The mean and standard deviation of Gaussian noise are fixed empirical values, without adaptive adjustments.
- The Toy experiments use simple random/uniform interference, which still has a gap compared to the spatial distribution of real adverse weather.
- The advantages on the SynLiDAR→SemanticSTF benchmark are not as pronounced as on SemanticKITTI→SemanticSTF.
- Future work can attempt to combine SJ and LPD with simulation methods to leverage their complementary strengths.
- DQN introduces additional training overhead; more efficient search methods for augmentation strategies can be explored.

## Related Work & Insights

- **MinkowskiNet [Choy et al., CVPR 2019]**: A voxel-based method using sparse convolution, identified by multiple studies as a standard and robust baseline.
- **PointDR [Kong et al.]**: Uses a teacher-student framework and feature prototypes for robust segmentation, serving as the Prev. SOTA.
- **LaserMix / PolarMix**: LiDAR data augmentation methods, but not designed for weather robustness, degrading performance in this setting instead.
- **LiDAR Snow/Fog Simulation [Hahner et al.]**: Physical equation-based simulation for specific weather, but only used for detection.
- **DQN [Mnih et al.]**: Deep Q-Learning used for policy search, which is innovatively introduced to data augmentation design in this work.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4.5 |
| Practicality | 4.5 |
| Writing Quality | 4 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation](../../ICLR2026/autonomous_driving/adaptive_augmentation-aware_latent_learning_for_robust_lidar_semantic_segmentati.md)
- [\[ECCV 2024\] Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains](rethinking_lidar_domain_generalization_single_source_as_multiple_density_domains.md)
- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)
- [\[ECCV 2024\] Reliability in Semantic Segmentation: Can We Use Synthetic Data?](reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)
- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)

</div>

<!-- RELATED:END -->
