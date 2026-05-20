---
title: >-
  [Paper Note] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation
description: >-
  [CVPR2026][Autonomous Driving][3D human pose estimation] This paper proposes Adaptive Weight Constraint (AWC) regularization, combining Shapley-value-based modality contribution assessment and Fisher Information Matrix (…
tags:
  - "CVPR2026"
  - "Autonomous Driving"
  - "3D human pose estimation"
  - "multi-modal learning"
  - "modality imbalance"
  - "Shapley value"
  - "Fisher Information Matrix"
date: 2026-05-08
content_hash: 3f52050c0b0f5ffa
---

# Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation

**Conference**: CVPR2026
**arXiv**: [2501.05264](https://arxiv.org/abs/2501.05264)  
**Code**: [MICLAB-BUPT/AWC](https://github.com/MICLAB-BUPT/AWC)  
**Area**: Autonomous Driving
**Keywords**: 3D human pose estimation, multi-modal learning, modality imbalance, Shapley value, Fisher Information Matrix

## TL;DR

This paper proposes Adaptive Weight Constraint (AWC) regularization, combining Shapley-value-based modality contribution assessment and Fisher Information Matrix (FIM) weighted parameter penalties, to address modality imbalance in multi-modal (RGB/LiDAR/mmWave/WiFi) 3D human pose estimation. Balanced optimization is achieved without introducing any additional learnable parameters.

## Background & Motivation

### State of the Field

3D human pose estimation (3D HPE) is a fundamental computer vision task with broad applications in human–computer interaction, action assessment, and rehabilitation monitoring. While conventional approaches rely primarily on RGB images, their performance degrades under occlusion and privacy-sensitive scenarios. This has motivated multi-modal methods that fuse non-intrusive sensors such as LiDAR, mmWave radar, and WiFi.

### Core Motivation

Joint multi-modal training suffers from **modality imbalance**: dominant modalities (e.g., RGB, LiDAR) converge rapidly in early training and suppress the optimization of weaker modalities (mmWave, WiFi). Existing balancing methods exhibit three key deficiencies:

**Poor task adaptability**: Methods such as G-Blending and OGM-GE are designed around cross-entropy loss or explicit class membership, making them suitable only for classification tasks and not directly transferable to regression.

**Additional parameters**: Methods such as MMPareto require unimodal auxiliary heads, increasing model complexity.

**Neglect of weak-modality overfitting**: These methods regulate only the gradients of dominant modalities without accounting for the risk of weak modalities overfitting to noisy signals.

A key observation motivating this work is that in regression tasks, the prediction standard deviation of weak modalities (mmWave, WiFi) approaches zero (i.e., predictions collapse to constant values). Using MSE/MAE as the Shapley profit function in this setting produces misleading assessments — constant predictions are erroneously assigned high contribution scores.

## Method

### Overall Architecture

The framework comprises two core components:

- **Shapley Modality Contribution Assessment Module**: Quantifies per-modality contribution via Shapley values combined with Pearson correlation coefficients to detect modality imbalance.
- **Adaptive Weight Constraint (AWC) Regularization**: Applies FIM-weighted parameter deviation penalties within an early "learning window" to balance the learning speed across modalities.

The model employs modality-specific encoders (VideoPose3D for RGB, Point Transformer for LiDAR/mmWave, MetaFi++ for WiFi), followed by a multi-modal fusion module and a pose regression head predicting 3D joint coordinates.

### Key Design 1: Shapley Contribution Assessment for Regression Tasks

**Classification vs. Regression**: For feature-concatenation fusion, the final prediction can be decomposed as $\hat{y} = \hat{y}^R + \hat{y}^L + \hat{y}^M + \hat{y}^W$. In classification, weak-modality logits approximate a uniform distribution, contributing negligibly to the softmax output; cross-entropy therefore serves as a valid Shapley profit function. In regression, however, weak-modality predictions have near-zero standard deviation (nearly constant), causing MSE-based evaluation to be biased toward large-output modalities and to overestimate the reliability of weak modalities.

**Solution**: The Pearson correlation coefficient replaces MSE as the profit function:

$$s(y, \hat{y}) = \sum_{i=1}^{j \times 3} \rho(y_i, \hat{y}_i), \quad \rho(y_i, \hat{y}_i) = \frac{\text{cov}(y_i, \hat{y}_i)}{\sigma_{y_i} \cdot \sigma_{\hat{y}_i}}$$

The Pearson correlation coefficient measures the linear relationship between predictions and ground truth rather than numerical distance, making it inherently robust to constant bias and scale differences. When a weak modality produces near-constant predictions with standard deviation approaching zero, its Pearson correlation coefficient approaches zero, accurately reflecting its lack of informative content.

Missing modality features are zero-padded, and Shapley values are computed by enumerating all modality subsets to derive marginal contributions.

### Key Design 2: Adaptive Weight Constraint (AWC) Regularization

**K-Means Grouping**: The Shapley scores of the four modalities are clustered via K-Means ($K=2$). The high-score cluster constitutes the dominant modality set $\mathcal{M}_\mathcal{S}$, and the low-score cluster the inferior modality set $\mathcal{M}_\mathcal{I}$, with distinct regularization coefficients $\alpha_\mathcal{S}$ and $\alpha_\mathcal{I}$ assigned accordingly.

**AWC Loss**: The FIM diagonal is used to apply importance-weighted penalties on parameter deviations:

$$\mathcal{L}_{AWC} = \sum_{m \in \mathcal{M}} \left[\alpha_\mathcal{S} \cdot \mathbf{1}_{\{m \in \mathcal{M}_\mathcal{S}\}} + \alpha_\mathcal{I} \cdot \mathbf{1}_{\{m \in \mathcal{M}_\mathcal{I}\}}\right] \cdot \mathcal{L}_W^m$$

where $\mathcal{L}_W^m = \sum_i \frac{[\mathcal{I}_\mathcal{D}]_{ii} (\theta_{t,i}^m - \theta_{0,i}^{m,*})^2}{2}$

**Core Insight**: The FIM diagonal $[\mathcal{I}]_{ii}$ measures the empirical importance of each parameter (mean squared gradient). Dominant modalities exhibit large gradients early in training, yielding high FIM values and thus stronger penalties on parameter updates; weak modalities have small gradients, low FIM values, and lighter penalties. Combined with $\alpha_\mathcal{S} > \alpha_\mathcal{I}$, this achieves a dual effect: **suppressing premature convergence of dominant modalities** while **moderately constraining weak modalities against noise overfitting**.

### Loss & Training

- **Total loss**: $\mathcal{L}_{total} = \mathcal{L}_{MPJPE} + \mathcal{L}_{AWC}$ (AWC applied only within the learning window)
- **Learning window**: AWC regularization is applied during the first $K$ epochs; thereafter, only the task loss is used. This is motivated by the "critical learning period" theory — most task-relevant information is acquired early in training.
- **FIM update frequency**: Recomputed once at the beginning of each epoch.
- **Training setup**: Adam optimizer, lr=1e-3, batch size=192, 50 epochs, learning rate decayed by 10× at epoch 30.

## Key Experimental Results

### Main Results: Comparison with Existing Balancing Methods (MM-Fi Dataset)

| Method | Fusion | P1 MPJPE↓ | P1 PA-MPJPE↓ | P3 MPJPE↓ | P3 PA-MPJPE↓ |
|--------|--------|-----------|-------------|-----------|-------------|
| MM-Fi baseline | - | 72.90 | 47.70 | 89.80 | 63.20 |
| Concatenation | concat | 53.87 | 35.09 | 48.17 | 32.18 |
| + G-Blending | concat | 58.40 | 37.20 | 53.13 | 33.28 |
| + OGM-GE | concat | 55.51 | 35.92 | 51.68 | 32.84 |
| + AGM | concat | 55.80 | 38.10 | 53.88 | 36.30 |
| + Modality-level | concat | 53.24 | 34.81 | 53.98 | 31.85 |
| **+ Ours** | **concat** | **51.16** | **34.46** | **47.55** | **31.79** |
| Attention | attn | 53.35 | 35.20 | 49.97 | 32.33 |
| **+ Ours** | **attn** | **51.29** | **34.65** | **49.08** | **32.10** |

Key findings: (1) The proposed method reduces P1 MPJPE by 2.71 mm under concat fusion. (2) G-Blending and AGM perform worse than the baseline, demonstrating that balancing strategies designed for classification are counterproductive in regression settings. (3) The method is effective across all protocols and fusion strategies.

### Ablation Study: AWC Hyperparameter Sensitivity (Protocol 1, Concat)

| $\alpha_\mathcal{S}$ | $\alpha_\mathcal{I}$ | MPJPE↓ | PA-MPJPE↓ |
|------|------|--------|-----------|
| - (baseline) | - | 53.87 | 35.09 |
| 0 | 10k | 52.92 (-0.95) | 34.94 (-0.15) |
| 10k | 0 | 52.09 (-1.78) | 34.81 (-0.28) |
| 10k | 10k | 51.88 (-1.99) | 34.84 (-0.25) |
| **20k** | **10k** | **51.16 (-2.71)** | **34.46 (-0.63)** |
| 20k | 20k | 51.69 (-2.18) | 34.84 (-0.25) |
| 30k | 20k | 51.34 (-2.53) | 34.56 (-0.53) |

Key findings: (1) The optimal configuration is $\alpha_\mathcal{S}=20k, \alpha_\mathcal{I}=10k$, i.e., dominant modalities receive twice the regularization strength of inferior ones. (2) Constraining only the dominant modality ($\alpha_\mathcal{I}=0$) is less effective than constraining both, confirming that weak modalities also require moderate regularization against overfitting. (3) A learning window of $K=20$ (40% of total epochs) is optimal.

### Modality Fusion Analysis

| Modality Combination | MPJPE↓ | PA-MPJPE↓ |
|----------------------|--------|-----------|
| RGB only | 63.61 | 35.75 |
| LiDAR only | 66.95 | 45.70 |
| mmWave only | 102.89 | 52.21 |
| WiFi only | 166.92 | 97.39 |
| R+L | 52.93 | 34.96 |
| R+L+M+W (four modalities) | 53.87 | 35.09 |

**Key finding**: Four-modality fusion (53.87) is inferior to RGB+LiDAR two-modality fusion (52.93), providing direct empirical evidence of modality competition — weak modalities not only fail to provide performance gains but actively interfere with the learning of strong modalities.

### Computational Overhead

The overhead of Shapley contribution assessment is negligible: under Concat/MLP fusion, it accounts for only 0.41%–0.93% of total training time; under Attention fusion, approximately 3.5%–5.4%, posing no practical bottleneck.

## Highlights & Insights

1. **Key insight on Shapley values for regression**: Weak modalities in regression collapse to constant predictions (standard deviation ≈ 0), causing MSE/MAE to misestimate their contribution. The Pearson correlation coefficient is a more appropriate profit function — a finding with broad implications for all regression-based multi-modal tasks.
2. **FIM as adaptive regularization weights**: FIM naturally captures modality-wise differences in parameter importance — dominant modalities have large gradients → high FIM → heavy penalty → slower update; weak modalities have small gradients → low FIM → light penalty → protected learning — without requiring manual design of modality-specific adjustment strategies.
3. **Zero additional parameters**: Unlike methods such as MMPareto that require auxiliary unimodal heads, AWC relies entirely on statistics derived from existing parameters (mean squared gradients), making it elegant and lightweight.
4. **Direct evidence of modality competition**: Four-modality fusion yielding worse MPJPE than two-modality fusion provides compelling empirical support for the "more is not always better" phenomenon in multi-modal learning.

## Limitations & Future Work

1. **Validation on a single dataset (MM-Fi)**: Generalization to additional datasets and diverse scenarios remains unverified.
2. **Fixed set of four modalities**: Scalability to a larger number of modalities is untested; Shapley value computation grows factorially with the number of modalities, likely requiring approximation algorithms beyond 5–6 modalities.
3. **Manual tuning of the learning window $K$**: While $K=20$ is optimal for 50-epoch training, an adaptive selection mechanism for $K$ across different tasks and data scales is absent.
4. **Hard binary partitioning via K-Means**: The two-way split (dominant/inferior) is coarse; finer-grained continuous grouping may yield better performance.
5. **Limited improvement to weak modality representations**: The method mitigates suppression of weak modalities but does not enhance their feature extraction capacity at the encoder level.

## Related Work & Insights

- **Modality imbalance theory**: OGM-GE (CVPR 2022) and G-Blending (ICLR 2020) pioneered the study of multi-modal competition but are limited to classification tasks.
- **Shapley values in multi-modal learning**: SHAPE (IJCAI 2022) first introduced Shapley values for modality contribution assessment; this paper extends the framework to regression settings.
- **Fisher information and continual learning**: The design of AWC regularization draws inspiration from EWC (Elastic Weight Consolidation), which uses FIM to protect important parameters in continual learning. This paper inverts the paradigm — applying FIM to constrain the excessively fast learning of dominant modalities.
- **Broader implications**: The substitution of Pearson correlation for MSE is generalizable to other regression-based multi-modal tasks (e.g., depth estimation, optical flow); FIM-adaptive regularization can serve as a plug-and-play module.

## Rating

| Dimension | Score (1–10) | Remarks |
|-----------|-------------|---------|
| Novelty | 7 | The regression adaptation of Shapley+Pearson and FIM-adaptive regularization are original, though core components build on established theory |
| Experimental Thoroughness | 6 | Single dataset (MM-Fi), but ablations are comprehensive |
| Writing Quality | 7 | Analysis is thorough, motivation is clearly articulated, derivations are complete |
| Value | 7 | No additional parameters, plug-and-play design; broadly applicable to multi-modal regression tasks |
| **Overall** | **7** | Elegant method design, but generalization validation is insufficient |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[CVPR 2026\] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)
- [\[CVPR 2026\] InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [\[CVPR 2026\] Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)

</div>

<!-- RELATED:END -->
