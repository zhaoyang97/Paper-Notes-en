---
title: >-
  [Paper Note] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation
description: >-
  [CVPR 2026][Autonomous Driving][Multi-modal fusion] To address the modality imbalance problem in multi-modal 3D human pose estimation (3D HPE), this paper proposes a Shapley-value-based modality contribution evaluation algorithm and an Adaptive Weight Constraint (AWC) regularization method based on the Fisher information matrix. The approach achieves balanced optimization across modalities without introducing additional parameters, and comprehensively outperforms existing balancing methods on the MM-Fi dataset.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Multi-modal fusion
  - modality imbalance
  - Shapley value
  - Fisher information matrix
  - 3D HPE
date: 2026-05-08
content_hash: 823923486382af99
---

# Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation

**Conference**: CVPR 2026
**arXiv**: [2501.05264](https://arxiv.org/abs/2501.05264)
**Code**: [GitHub](https://github.com/MICLAB-BUPT/AWC)
**Area**: Autonomous Driving
**Keywords**: Multi-modal fusion, modality imbalance, Shapley value, Fisher information matrix, 3D HPE

## TL;DR

To address the modality imbalance problem in multi-modal 3D human pose estimation (3D HPE), this paper proposes a Shapley-value-based modality contribution evaluation algorithm and an Adaptive Weight Constraint (AWC) regularization method based on the Fisher information matrix. The approach achieves balanced optimization across modalities without introducing additional parameters, and comprehensively outperforms existing balancing methods on the MM-Fi dataset.

## Background & Motivation

**Background**: 3D HPE is an important topic in computer vision. Traditional methods rely primarily on RGB images but are limited in occlusion and privacy-sensitive scenarios. Non-intrusive sensors such as LiDAR, millimeter-wave radar (mmWave), and WiFi have recently attracted increasing attention, and multi-modal fusion has emerged as an effective strategy for improving the robustness of 3D HPE.

**Limitations of Prior Work**: When multiple modalities are trained jointly, a "modality imbalance" phenomenon arises—dominant modalities rich in information (e.g., RGB, LiDAR) drive the optimization direction and suppress the learning of weaker modalities (e.g., mmWave, WiFi). Experiments show that fusing all four modalities (MPJPE 53.87 mm) actually underperforms fusing only RGB+LiDAR (52.93 mm), demonstrating the existence of modality competition.

**Key Challenge**: Existing modality balancing methods (e.g., OGM-GE, AGM) are primarily designed for classification tasks and rely on cross-entropy loss or auxiliary unimodal branches, making them unsuitable for regression tasks. They either overlook the intrinsic differences in information capacity across modalities or introduce additional learnable parameters that increase model complexity.

**Goal**: The paper aims to quantify the contribution of each modality and dynamically regulate the learning speed of each modality—without increasing model complexity—so that dominant modalities do not excessively suppress the optimization of weaker ones.

**Key Insight**: Shapley values from cooperative game theory are introduced for modality contribution evaluation in regression tasks, and the parameter importance information provided by the Fisher information matrix (FIM) is used to adaptively constrain parameter updates in each modality encoder.

**Core Idea**: Shapley values are used to detect modality imbalance, and FIM-weighted weight constraint losses are applied to slow down the learning of dominant modalities in the early training phase, thereby achieving balanced multi-modal learning without introducing additional parameters.

## Method

### Overall Architecture

The system consists of four modality-specific encoders—VideoPose3D for RGB, Point Transformer for LiDAR and mmWave, and MetaFi++ for WiFi—that extract features subsequently fused by a multi-modal fusion module (supporting concatenation, MLP, and attention strategies), with a regression head predicting 3D joint coordinates. During training, a Shapley module evaluates each modality's contribution, and the AWC loss constrains parameter updates within a learning window.

### Key Designs

1. **Shapley Value-Based Modality Contribution Evaluation**:

    - **Function**: Quantifies the marginal contribution of each modality in multi-modal collaboration.
    - **Mechanism**: For each modality $m$, the algorithm iterates over all subsets $S$ not containing $m$ and computes the change in utility upon adding $m$. A key innovation is replacing MSE with the Pearson correlation coefficient $\rho(y_i, \hat{y}_i)$ as the utility function $s(\cdot, \cdot)$, since predictions of weak modalities in regression tasks tend toward constant values (near-zero standard deviation), causing MSE to erroneously overestimate their contribution.
    - **Design Motivation**: In classification tasks, the near-uniform output of weak modalities has little effect on Softmax. In regression tasks, however, the constant output of weak modalities is misinterpreted by MSE as "stable and reliable." The Pearson correlation coefficient focuses solely on the trend of correlation between predictions and ground truth, unaffected by output magnitude.

2. **Adaptive Weight Constraint (AWC) Regularization**:

    - **Function**: Applies differentiated parameter constraints to each modality encoder based on the FIM, slowing the learning speed of dominant modalities.
    - **Mechanism**: Modalities are partitioned into a dominant group $\mathcal{M}_\mathcal{S}$ and a weak group $\mathcal{M}_\mathcal{I}$ via K-Means clustering, with distinct regularization coefficients $\alpha_\mathcal{S}$ and $\alpha_\mathcal{I}$ applied respectively. The AWC loss is defined as $\mathcal{L}_{AWC} = \sum_m [\alpha_\mathcal{S} \cdot \mathbf{1}_{m \in \mathcal{M}_\mathcal{S}} + \alpha_\mathcal{I} \cdot \mathbf{1}_{m \in \mathcal{M}_\mathcal{I}}] \cdot \sum_i \frac{[\mathcal{I}]_{ii}(\theta_{t,i}^m - \theta_{0,i}^{m,*})^2}{2}$, where the diagonal elements of the FIM $[\mathcal{I}]_{ii}$ measure the empirical importance of each parameter.
    - **Design Motivation**: Dominant modalities produce larger gradients in early training, resulting in higher FIM values. The FIM-weighted penalty term automatically imposes stronger constraints on strong modalities and weaker constraints on weak modalities, without manual specification.

3. **Learning Window Mechanism**:

    - **Function**: Applies AWC constraints only during the first $K$ training epochs.
    - **Mechanism**: Based on the "critical learning period" theory, most task-relevant information is acquired in the early stage of training. Constraining the rapid learning of dominant modalities during the first $K$ epochs provides weaker modalities with an opportunity to learn useful representations.
    - **Design Motivation**: Excessively prolonged regularization limits the model's final expressive capacity. A moderate window (empirically optimal at $K=20$) balances both training balance and final performance.

### Loss & Training

Within the learning window, the total loss is $\mathcal{L}_{total} = \mathcal{L}_{MPJPE} + \mathcal{L}_{AWC}$; outside the window, only $\mathcal{L}_{MPJPE}$ is used. Training employs the Adam optimizer with an initial learning rate of 1e-3, decayed by a factor of 10 every 30 epochs, for a total of 50 epochs with batch size 192. The FIM is recomputed at the beginning of each epoch.

## Key Experimental Results

### Main Results

| Method | Protocol 1 MPJPE↓ | Protocol 1 PA-MPJPE↓ | Protocol 3 MPJPE↓ | Protocol 3 PA-MPJPE↓ |
|------|-------|----------|-------|----------|
| MM-Fi | 72.90 | 47.70 | 89.80 | 63.20 |
| Concatenation | 53.87 | 35.09 | 48.17 | 32.18 |
| +G-Blending | 58.40 | 37.20 | 53.13 | 33.28 |
| +OGM-GE | 55.51 | 35.92 | 51.68 | 32.84 |
| +AGM | 55.80 | 38.10 | 53.88 | 36.30 |
| **+Ours** | **51.16** | **34.46** | **47.55** | **31.79** |

The proposed method achieves state-of-the-art results across three protocols and three fusion strategies, reducing MPJPE by approximately 2–3 mm on average.

### Ablation Study

| Config ($\alpha_\mathcal{S}$, $\alpha_\mathcal{I}$) | MPJPE | PA-MPJPE | Note |
|------|---------|------|------|
| Baseline (0, 0) | 53.87 | 35.09 | No regularization |
| (0, 10k) | 52.92 | 34.94 | Constrain weak modality only |
| (10k, 0) | 52.09 | 34.81 | Constrain strong modality only |
| **(20k, 10k)** | **51.16** | **34.46** | **Optimal configuration** |
| (30k, 20k) | 51.34 | 34.56 | Over-regularization |

Unimodal performance validates modality disparity: RGB (MPJPE 63.61) > LiDAR (66.95) >> mmWave (102.89) >> WiFi (166.92).

### Key Findings

- Four-modality fusion (53.87 mm) underperforms RGB+LiDAR-only fusion (52.93 mm), directly demonstrating the existence of modality competition.
- Shapley contribution scores show that RGB and LiDAR contribute far more than mmWave and WiFi, with the contributions of weak modalities continuously declining during training.
- Constraining both modality groups simultaneously outperforms constraining only one group (weak modalities also require constraints to suppress noise overfitting).
- The learning window $K=20$ is optimal; performance degrades with either shorter or longer windows.
- Computational overhead is negligible: Shapley evaluation adds only 0.4%–0.9% of training time under concatenation/MLP fusion.

## Highlights & Insights

- Shapley values are successfully applied for the first time to modality contribution evaluation in regression tasks; replacing MSE with the Pearson correlation coefficient resolves the key issue of biased contribution measurement for weak modalities.
- AWC introduces zero additional parameters, achieving modality balance purely through regularization—an elegant and compact design.
- The FIM-weighting mechanism is inherently adaptive: strong modalities produce large gradients → high FIM → strong constraints; weak modalities produce small gradients → low FIM → weak constraints.
- The learning window design is well-conceived, intervening only during the critical learning period and avoiding the long-term regularization that would otherwise limit model capacity.

## Limitations & Future Work

- Validation is conducted on only one dataset (MM-Fi); generalizability to broader scenarios remains to be verified.
- Among the four modalities, mmWave and WiFi possess inherently low information capacity (WiFi unimodal MPJPE reaches 167 mm), raising questions about the practical value of including them.
- K-Means clustering may not be sufficiently robust with only four modalities; performance with a larger number of modalities has not been explored.
- The learning window length $K$ requires manual tuning, and no adaptive mechanism is provided.

## Related Work & Insights

- **vs. OGM-GE**: OGM-GE modulates gradients to slow dominant modality learning, but is less effective on regression tasks (MPJPE 55.51 vs. 51.16), as it only adjusts gradient magnitude without constraining parameter update directions.
- **vs. MMPareto**: MMPareto optimizes multi-modal gradients based on the Pareto frontier but requires additional unimodal heads, introducing more parameters; the proposed method is more compact with no extra parameters.
- **vs. G-Blending**: G-Blending underperforms even the baseline on this dataset (58.40 vs. 53.87), indicating that methods designed for classification cannot be directly transferred to regression tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of Shapley values and Pearson correlation for regression-based modality contribution evaluation is novel, though the overall framework is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐ Only one dataset is used; while three protocols, three fusion strategies, and ablation studies offer reasonable coverage, external validation is insufficient.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; the analysis of weak modality prediction collapse (Figure 3) and experimental evidence of modality competition are convincing.
- **Value**: ⭐⭐⭐ The method has some generality, but the application scenario is limited, and multi-modal HPE itself is not yet a mainstream research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[CVPR 2026\] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [\[CVPR 2026\] Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)
- [\[CVPR 2026\] InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)

</div>

<!-- RELATED:END -->
