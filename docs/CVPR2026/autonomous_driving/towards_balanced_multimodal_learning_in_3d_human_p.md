---
title: >-
  [Paper Note] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation
description: >-
  [CVPR 2026][Autonomous Driving][3D pose estimation] This paper proposes a modality contribution assessment algorithm based on Shapley values and Pearson correlation coefficients, along with a Fisher Information Matrix (FIM)-guided Adaptive Weight Constraint (AWC) regularization method. The approach addresses modality imbalance in end-to-end fusion of four modalities (RGB/LiDAR/mmWave/WiFi), achieving a 2.71 mm reduction in MPJPE on the MM-Fi dataset without introducing additional learnable parameters.
tags:
  - CVPR 2026
  - Autonomous Driving
  - 3D pose estimation
  - modality imbalance
  - Shapley value
  - Fisher Information Matrix
  - multi-modal fusion
date: 2026-05-08
content_hash: 877ae71fa0f60bfb
---

# Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation

**Conference**: CVPR 2026
**arXiv**: [2501.05264](https://arxiv.org/abs/2501.05264)
**Code**: [GitHub](https://github.com/MICLAB-BUPT/AWC)
**Area**: Autonomous Driving
**Keywords**: 3D pose estimation, modality imbalance, Shapley value, Fisher Information Matrix, multi-modal fusion

## TL;DR

This paper proposes a modality contribution assessment algorithm based on Shapley values and Pearson correlation coefficients, along with a Fisher Information Matrix (FIM)-guided Adaptive Weight Constraint (AWC) regularization method. The approach addresses modality imbalance in end-to-end fusion of four modalities (RGB/LiDAR/mmWave/WiFi), achieving a 2.71 mm reduction in MPJPE on the MM-Fi dataset without introducing additional learnable parameters.

## Background & Motivation

**Background**: 3D human pose estimation (3D HPE) has been dominated by RGB-based methods, which are limited by occlusion and privacy concerns. Non-invasive sensors (LiDAR/mmWave/WiFi) can provide complementary information, yet end-to-end multi-modal training suffers from modality imbalance.

**Limitations of Prior Work**: Strong modalities (RGB/LiDAR) dominate gradient updates in the early stages of training, suppressing the optimization of weaker modalities (mmWave/WiFi) and causing their predictions to degenerate toward near-constant outputs (standard deviation approaching zero). More critically, naively adding more modalities can degrade performance — four-modality fusion (53.87 mm MPJPE) underperforms RGB+LiDAR (52.93 mm), directly confirming modality competition. Existing balancing methods (G-Blending/OGM-GE/AGM) have two key limitations: (1) they are designed for classification and rely on cross-entropy loss, making them unsuitable for regression tasks; (2) they frequently introduce auxiliary unimodal heads, increasing model complexity.

**Key Challenge**: How can balanced optimization be achieved for multi-modal regression tasks without introducing additional parameters? Two sub-problems must be resolved: accurately assessing each modality's contribution in a regression setting (where cross-entropy-based schemes are inapplicable), and adaptively regulating per-modality learning rates to achieve balance.

**Key Insight**: Weak modalities exhibit a distinctive behavior in regression — their predictions are nearly constant (extremely low standard deviation). Evaluating such predictions with MSE/MAE would incorrectly classify them as "reliable" due to small distances. The Pearson correlation coefficient measures linear correlation between predictions and ground truth rather than their distance, and is insensitive to prediction magnitude, making it a more appropriate contribution metric.

**Core Idea**: Replace MSE with the Pearson correlation coefficient as the profit function in the Shapley framework for regression tasks to detect modality imbalance, then apply FIM-weighted parameter-shift regularization to differentially constrain the learning rates of each modality.

## Method

### Overall Architecture

Four modalities (RGB/LiDAR/mmWave/WiFi) are encoded by dedicated encoders (VideoPose3D / Point Transformer / MetaFi++). Their features are merged by a fusion module (concatenation/MLP/attention) and fed into a pose regression head to produce 3D joint coordinates. Two core components are introduced: a Shapley module that assesses modality contributions, followed by AWC regularization that constrains parameter updates within a learning window.

### Key Designs

1. **Shapley Value + Pearson Correlation for Modality Contribution Assessment**

   - **Function**: Accurately quantifies each modality's contribution to the fused model in a regression setting.
   - **Mechanism**: Shapley values are computed by enumerating all modality subsets to determine marginal contributions: $\phi^m(\mathcal{M}) = \sum_{S \subseteq \mathcal{M} \setminus \{m\}} \frac{|S|!(|\mathcal{M}|-|S|-1)!}{|\mathcal{M}|!} V(S,m)$. The key innovation lies in the profit function $s(\cdot,\cdot)$: instead of cross-entropy (used in classification), this paper adopts the Pearson correlation coefficient $s(y, \hat{y}) = \sum_{i=1}^{j \times 3} \rho(y_i, \hat{y}_i)$, computing linear correlation for each joint coordinate along the batch dimension.
   - **Design Motivation**: Weak modalities (mmWave/WiFi) produce near-constant predictions in regression (standard deviation approaching zero), which MSE would erroneously interpret as "reliable" due to small distances. Pearson correlation is unaffected by prediction magnitude and correctly identifies uninformative constant predictions. Experiments confirm that RGB and LiDAR consistently receive high contribution scores, while mmWave/WiFi scores are low and decrease over training.

2. **AWC (Adaptive Weight Constraint) Regularization**

   - **Function**: Differentially constrains the parameter update rate of each modality encoder according to modality contribution.
   - **Mechanism**: K-Means clustering first partitions the four modalities by Shapley score into a superior group $\mathcal{M}_\mathcal{S}$ and an inferior group $\mathcal{M}_\mathcal{I}$. A parameter-shift regularization term is applied to each modality encoder: $\mathcal{L}_{\text{AWC}} = \sum_m [\alpha_\mathcal{S} \cdot \mathbf{1}_{\{m \in \mathcal{M}_\mathcal{S}\}} + \alpha_\mathcal{I} \cdot \mathbf{1}_{\{m \in \mathcal{M}_\mathcal{I}\}}] \cdot \sum_i \frac{[\mathcal{I}_\mathcal{D}]_{ii} (\theta_{t,i}^m - \theta_{0,i}^{m,*})^2}{2}$. The diagonal FIM approximation $[\mathcal{I}]_{ii}$ measures parameter importance: strong modalities accumulate large early-stage gradients, yielding high FIM values and stronger regularization (slowing their learning); weak modalities have low FIM values and receive weaker regularization (permitting continued learning). Setting $\alpha_\mathcal{S} > \alpha_\mathcal{I}$ ensures stronger constraints on dominant modalities. No additional learnable parameters are introduced.
   - **Design Motivation**: The FIM provides a data-driven estimate of parameter sensitivity, naturally distinguishing parameters that critically affect the loss from those that do not, constraining both the direction and magnitude of updates.

3. **Learning Window Mechanism**

   - **Function**: AWC regularization is applied only during the first $K$ epochs and disabled thereafter.
   - **Mechanism**: Prior work has established that modality-relevant critical information is acquired in early training. Experiments identify $K=20$ as optimal (out of 50 total epochs); both shorter and longer windows degrade performance.
   - **Design Motivation**: Late-stage regularization interferes with convergence. By reserving early epochs to establish balance for weak modalities, subsequent unconstrained optimization can fully exploit all modalities.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \begin{cases} \mathcal{L}_{\text{MPJPE}} + \mathcal{L}_{\text{AWC}} & \text{first } K \text{ epochs} \\ \mathcal{L}_{\text{MPJPE}} & \text{remaining epochs} \end{cases}$$

Adam optimizer, lr=1e-3, decayed by ×0.1 every 30 epochs, batch size=192, 50 total epochs, 2×RTX 3090.

## Key Experimental Results

### Main Results

| Method | Protocol 1 MPJPE↓ | PA-MPJPE↓ | Protocol 3 MPJPE↓ | PA-MPJPE↓ |
|---|---|---|---|---|
| Concatenation (baseline) | 53.87 | 35.09 | 48.17 | 32.18 |
| + G-Blending | 58.40 | 37.20 | 53.13 | 33.28 |
| + OGM-GE | 55.51 | 35.92 | 51.68 | 32.84 |
| + AGM | 55.80 | 38.10 | 53.88 | 36.30 |
| + Modality-level | 53.24 | 34.81 | 53.98 | 31.85 |
| **+ Ours (AWC)** | **51.16** | **34.46** | **47.55** | **31.79** |

### Ablation Study

| $\alpha_\mathcal{S}$ | $\alpha_\mathcal{I}$ | MPJPE | PA-MPJPE | Note |
|---|---|---|---|---|
| 0 | 0 | 53.87 | 35.09 | No regularization (baseline) |
| 0 | 10k | 52.92 | 34.94 | Constrain weak modalities only |
| 10k | 0 | 52.09 | 34.81 | Constrain strong modalities only |
| **20k** | **10k** | **51.16** | **34.46** | Optimal configuration |
| 20k | 20k | 51.69 | 34.84 | Uniform constraint underperforms |

### Key Findings

- **Constraining only strong modalities is more effective than constraining only weak modalities** (52.09 vs. 52.92 mm), confirming that suppressing the premature learning of dominant modalities is the primary balancing mechanism.
- **Jointly constraining both groups with differentiated strengths yields the best results**: uniform constraints ($\alpha_\mathcal{S}=\alpha_\mathcal{I}=20k$) are inferior to differentiated constraints (20k/10k).
- **Learning window $K=20$ is optimal**: too short ($K=10$) is insufficient to establish balance; too long ($K=25$) interferes with convergence.
- **Direct evidence of modality competition**: R+L+M+W (53.87 mm) underperforms R+L (52.93 mm), demonstrating that adding weak modalities is detrimental without proper balancing.
- **Negligible computational overhead**: Shapley computation accounts for only 0.41%–0.93% of total training time under Concat/MLP fusion.

## Highlights & Insights

- **Pearson correlation as the Shapley profit function for regression tasks**: This observation is particularly precise — when weak modalities produce near-constant predictions, MSE yields small errors and incorrectly signals utility. Pearson correlation is entirely immune to this failure mode and can be directly transferred to any multi-modal regression task.
- **FIM naturally enables differentiated regularization**: Rather than manually specifying which parameters to constrain, the FIM values — high for parameters that are frequently updated in strong modalities — identify exactly those parameters that most need to be slowed, achieving adaptive behavior. This idea is transferable to continual learning and domain adaptation.
- **The Learning Window concept**: The paper acknowledges that "balance matters only in the early phase" — all modalities should be allowed to optimize freely in later stages. This insight has reference value for other balancing methods.

## Limitations & Future Work

- **Evaluation on a single dataset (MM-Fi)**: The four-modality HPE setting is inherently niche, raising questions about generalizability.
- **Exponential complexity of Shapley computation**: With four modalities only $2^4=16$ forward passes are required, but the cost becomes infeasible as the number of modalities grows beyond six, necessitating sampling-based approximations.
- **Coarse K-Means binary grouping**: Partitioning modalities into only two groups (superior/inferior) is an oversimplification; finer-grained grouping or continuous weighting should be considered for larger modality sets.
- **Manual tuning of $\alpha_\mathcal{S}$ and $\alpha_\mathcal{I}$**: Although the method itself is adaptive, the two regularization coefficients still require hyperparameter search.

## Related Work & Insights

- **vs. G-Blending/OGM-GE/AGM**: These methods perform worse than the baseline in the paper's experiments (MPJPE increases by 2–5 mm), because they rely on cross-entropy or adjust only one of gradient direction/magnitude, rendering them unsuitable for regression tasks.
- **vs. PMR**: PMR performs balancing via prototype-based class-level representations, limiting it to classification tasks.
- **vs. MMPareto**: MMPareto optimizes multi-modal gradients via the Pareto frontier but relies on unimodal auxiliary heads, increasing parameter count.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of Pearson-Shapley and FIM-AWC is the first of its kind for multi-modal regression tasks.
- **Experimental Thoroughness**: ⭐⭐⭐ Limited to a single dataset, though ablation studies and analyses are detailed.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is thorough and motivation is clearly derived.
- **Value**: ⭐⭐⭐⭐ Provides a general framework for balanced optimization in multi-modal regression tasks.

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
