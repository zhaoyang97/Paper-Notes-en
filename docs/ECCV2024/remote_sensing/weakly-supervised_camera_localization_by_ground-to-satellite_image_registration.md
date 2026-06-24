---
title: >-
  [Paper Note] Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration
description: >-
  [ECCV 2024][Remote Sensing][Weakly-Supervised Localization] Proposes the first weakly-supervised ground-to-satellite image registration localization method. By training an orientation estimator on satellite-to-satellite pairs in a self-supervised manner and training a translation estimator via contrastive learning, it achieves the best cross-area generalization performance without requiring accurate ground-truth (GT) pose labels, outperforming most fully-supervised SOTA metho…
tags:
  - "ECCV 2024"
  - "Remote Sensing"
  - "Weakly-Supervised Localization"
  - "Ground-to-Satellite Image Registration"
  - "Self-Supervised Rotation Estimation"
  - "Contrastive Learning"
  - "Cross-View Feature Matching"
date: 2026-05-08
content_hash: 7bbfab41d4d7dad0
---

# Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration

**Conference**: ECCV 2024  
**arXiv**: [2409.06471](https://arxiv.org/abs/2409.06471)  
**Code**: [https://github.com/YujiaoShi/G2SWeakly](https://github.com/YujiaoShi/G2SWeakly)  
**Area**: Remote Sensing / Cross-View Camera Localization  
**Keywords**: Weakly-Supervised Localization, Ground-to-Satellite Image Registration, Self-Supervised Rotation Estimation, Contrastive Learning, Cross-View Feature Matching

## TL;DR

Proposes the first weakly-supervised ground-to-satellite image registration localization method. By training an orientation estimator on satellite-to-satellite pairs in a self-supervised manner and training a translation estimator via contrastive learning, it achieves the best cross-area generalization performance without requiring accurate ground-truth (GT) pose labels, outperforming most fully-supervised SOTA methods.

## Background & Motivation

**Background**: Camera localization is crucial in fields such as autonomous driving, robotics, and AR/VR. Satellite images have become an inexpensive reference data source due to their wide availability and global coverage. Recently, researchers have explored refining coarse poses (from GPS or image retrieval) via ground-to-satellite image registration, making significant progress.

**Core Problem**: Existing learning-based methods (such as CVR, SliceMatch, OrienterNet, Shi et al.) all **rely on precise GT pose labels** to train networks. However:

**Obtaining precise labels is difficult and expensive**: It requires professional equipment such as Real Time Kinematics (RTK) for field measurements, which is high-cost and time-consuming.

**RTK signals are unreliable**: Affected by multipath interference, signal blockage, and atmospheric conditions, even expensive RTK GPS can generate errors.

**Limited large-scale datasets**: The cost of high-precision annotation severely limits the scale of training data.

**Goal**: Develop a weakly-supervised strategy to improve camera localization accuracy through ground-to-satellite image registration under the condition of only having noisy pose labels (such as those from consumer-grade GPS). **This paper does not assume the existence of precise GT data in any region**.

**Key Insight**: Rotation and translation estimation can be decoupled—rotation estimation can be trained using self-supervision by constructing training pairs with GT from satellite images themselves; translation estimation can be trained weakly via contrastive learning using signals from positive and negative satellite images.

## Method

### Overall Architecture

The method is split into two stages: (1) **Rotation Estimation**—estimating the relative rotation between ground and satellite images using a network regressor trained via self-supervision; (2) **Translation Estimation**—estimating translation using spatial cross-correlation (similarity matching). Feature extractors for the two stages are trained independently and combined to determine the 3-DoF pose (2-DoF position + 1-DoF orientation).

### Key Designs

#### 1. **Self-Supervised Orientation Estimator**

**Function**: Traning a network regressor to estimate the relative rotation between ground and satellite images without requiring ground image labels.

**Mechanism**: Leveraging the idea of Spatial Transformer Networks, "satellite-to-satellite" image pairs are constructed for self-supervised training. Specifically, a random rotation $\mathbf{R}^*(\theta)$ and translation $\mathbf{t}^*$ are applied to a satellite image, and a triangular mask is used to simulate the Field-of-View (FoV) of the ground camera. The transformed image is used as the query, and the original image as the reference, to train the network to predict the known relative pose.

**Training Objective**:

$$\mathcal{L}_1 = |\theta - \theta^*| + |t_x - t_x^*| + |t_y - t_y^*|$$

**Design Motivation**: Satellite images and ground pinhole camera images share similar projection geometry—both map actual straight lines to straight lines in images. Thus, the feature extractor trained on satellite image pairs can **directly generalize** to ground images. Using the ground-plane homography to project ground features to an aerial view requires no extra trainable parameters, allowing the model to be natively deployed across different domains.

**Key Findings**: Neural networks are sensitive to the rotation of input signals (allowing orientation estimation without special designs), but due to aggregation layers (like max-pooling), they are insensitive to tiny translations. Hence, regression is used for rotation, while translation leverages the equivariance of spatial correlation.

#### 2. **Weakly-Supervised Translation Estimation via Contrastive Learning**

**Function**: Training a feature extraction network under only noisy location labels, enabling it to estimate the precise camera location via similarity matching.

**Mechanism**: For each ground image, a positive satellite image (covering the camera's location) and several negative satellite images (not covering the location) are identified based on its coarse location. The similarity maps $\mathbf{S}_{\text{pos}}$ and $\mathbf{S}_{\text{neg}}$ are computed. Contrastive learning is used to maximize the maximum similarity in the positive map and minimize the maximum similarity in the negative maps:

$$\mathcal{L}_2 = \sum_l \log(1 + e^{\alpha(\max \mathbf{S}_{\text{neg}} - \max \mathbf{S}_{\text{pos}})})$$

where $\alpha=10$ controls the convergence speed.

**Optional Supplementary Loss** (when relatively accurate noisy labels are available, $\lambda=1$):

$$\mathcal{L}_3 = \sum_l |\max(\mathbf{S}_{\text{pos}}) - \max(\mathbf{S}_{\text{pos}}[u^*\pm d/\gamma, v^*\pm d/\gamma])|$$

This forces the global maximum to equal the local maximum within a region of radius $d=5$m centered at the labeled position.

The overall training objective: $\mathcal{L} = \mathcal{L}_2 + \lambda \mathcal{L}_3$, where $\lambda=0$ denotes purely weak supervision and $\lambda=1$ utilizes noisy labels.

#### 3. **Confidence-Guided Similarity Matching**

**Function**: Simultaneously predicting a confidence map when extracting ground-view features to suppress features of dynamic objects and emphasize reliable ones.

**Mechanism**: The ground branch extracts features $\mathbf{F}_g$ and a confidence map $\mathbf{C}_g$. The confidence-weighted features $\hat{\mathbf{F}_g} = \mathbf{C}_g \mathbf{F}_g$ are used as a sliding window to compute normalized spatial cross-correlation with the satellite feature map $\mathbf{F}_s$:

$$\mathbf{S}(u,v) = \frac{\sum_i \sum_j \mathbf{F}_s(u+i, v+j) \hat{\mathbf{F}_g}(i,j)}{\sqrt{\sum_i \sum_j \mathbf{F}_s^2(u+i,v+j)} \sqrt{\sum_i \sum_j \hat{\mathbf{F}_g}^2(i,j)}}$$

**Design Motivation**: The confidence map is learned implicitly through the matching training objective without explicit supervision. Visualizations show that the learned confidence map ignores dynamic objects (e.g., vehicles) and highlights reliable static features (e.g., lane lines, road edges). Confidence is only learned for the ground branch, not the satellite branch—since satellite images contain fewer and smaller dynamic objects.

### Loss & Training

- **Orientation Phase**: $\mathcal{L}_1$ (L1 loss), trained on satellite-to-satellite pairs
- **Translation Phase**: $\mathcal{L} = \mathcal{L}_2 + \lambda \mathcal{L}_3$, with parameters of the orientation estimator frozen
- **Network Architecture**: VGG16-UNet feature extraction + Swin Transformer regressor
- **Training Configuration**: Batch size of 8, RTX 3090 GPU, KITTI: 3 epochs / VIGOR: 10 epochs
- Each batch consists of 1 positive sample + (B-1) negative samples
- Feature resolution is 1/4 of the original image (to save memory consumed by spatial cross-correlation)

## Key Experimental Results

### Main Results: Pose Estimation on the KITTI Dataset

| Method | Supervision | Test-2 Lat d=1↑ | Test-2 Lat d=3↑ | Test-2 Long d=1↑ | Test-2 Long d=3↑ | Test-2 θ=1↑ | Test-2 θ=3↑ |
|------|------|-----------------|-----------------|-------------------|-------------------|------------|------------|
| DSM | Fully-Supervised | 10.77 | 31.37 | 3.87 | 11.73 | 3.53 | 14.09 |
| SliceMatch | Fully-Supervised | 32.43 | 78.98 | 8.30 | 24.48 | 46.82 | 46.82 |
| Shi et al. | Fully-Supervised | 57.72 | 86.77 | 14.15 | 34.59 | 98.98 | 100.00 |
| Xia et al. | Fully-Supervised | 44.06 | 81.72 | 23.08 | 52.85 | 57.72 | 92.34 |
| Song et al. | Fully-Supervised | 54.19 | - | 23.10 | - | 43.44 | - |
| **Ours (λ=0)** | **Weakly-Supervised** | **62.73** | **86.53** | 9.98 | 29.67 | **99.99** | **100.00** |
| **Ours (λ=1)** | **Weakly-Supervised** | **64.74** | **86.18** | 11.81 | 34.77 | **99.99** | **100.00** |

Our method achieves the **best lateral localization** and **best orientation estimation** performance in cross-area (Test-2) evaluation, with the smallest performance gap between the same-area and cross-area settings.

### Comparison on the VIGOR Dataset

| Method | Supervision | Cross-area Aligned Mean↓ | Cross-area Aligned Median↓ | Cross-area Unknown Mean↓ | Cross-area Unknown Median↓ |
|------|------|--------------------------|----------------------------|--------------------------|----------------------------|
| MCC | Fully-Supervised | 9.05 | 5.14 | 12.66 | 9.55 |
| SliceMatch | Fully-Supervised | 5.53 | 2.55 | 8.48 | 5.64 |
| Xia et al. | Fully-Supervised | 4.97 | 1.68 | 5.41 | 1.89 |
| Song et al. | Fully-Supervised | 5.01 | 2.42 | 7.67 | 3.67 |
| **Ours (λ=0)** | **Weakly-Supervised** | **5.37** | **1.93** | **5.37** | **1.93** |
| **Ours (λ=1)** | **Weakly-Supervised** | **4.70** | **1.68** | **4.52** | **1.65** |

Under the cross-area + unknown orientation condition, our method ($\lambda=1$) **outperforms all fully-supervised methods** with a 4.52m mean error and a 1.65m median error.

### Ablation Study

| Configuration | Test-2 Lat d=1↑ | Test-2 Lat d=3↑ | Test-2 Long d=3↑ | Description |
|------|-----------------|-----------------|-------------------|------|
| Satellite query+regression | 5.06 | 15.46 | 15.79 | Translation estimation of the regressor is poor (even without domain gap) |
| Ground query+regression | 5.04 | 15.46 | 15.83 | Orientation transfers well to ground images |
| Ground+correlation (w/o confidence) | 45.11 | 73.04 | 18.30 | Spatial correlation significantly improves lateral localization |
| Ground+correlation (w/ confidence) | 62.73 | 86.53 | 29.67 | Confidence map significantly improves performance (+17.6% lateral) |
| Ground+correlation+λ=1 | 64.74 | 86.18 | 34.77 | Noisy labels further improve longitudinal localization |

### Key Findings

1. **Perfect domain generalization of orientation estimation**: The orientation network trained on satellite-to-satellite pairs suffers almost no accuracy loss when directly deployed on ground images ($\theta=1^\circ$: 99.99%), validating the hypothesis that projective geometry is shareable.
2. **Regressor is unsuitable for translation estimation**: Even without domain gaps, the regressor performs poorly on translation estimation ($d=3$ is only 15%), whereas spatial correlation elevates lateral localization to 73-87%.
3. **Confidence map is crucial**: Lateral localization at $d=1$ improves from 45% to 63% (+18%), and the learned confidence automatically ignores dynamic objects like vehicles.
4. **Weak supervision -> Best generalization**: Not overfitting to GT labels yields the strongest cross-area performance, minimizing the gap between same-area and cross-area results.
5. **Simple Homography outperforms complex cross-view transformer**: Under weak supervision, parameter-free ground-plane homography projection actually outperforms the learnable Geo. Trans., which is difficult to train under weak signals.

## Highlights & Insights

1. **Elegant design of rotation-translation decoupling**: Handling the two DOFs separately by utilizing two different physical properties (sensitivity of networks to rotation vs. equivariance of spatial correlation to translation).
2. **Zero-label orientation training**: Completely bypassing the need for ground image labels through satellite-to-satellite image pairs. The discovery that projective geometry is transferable is of high value.
3. **Weak supervision outperforming full supervision**: Better cross-area generalization is obtained without relying on precise labels. This counter-intuitive result exposes the risk of fully-supervised methods overfitting to GT poses.
4. **Simpler is better under weak signals**: Complex designs (like Geo. Trans.) are effective under strong supervision but perform worse than a simple Homography projection under weak supervision.

## Limitations & Future Work

1. **Non-shareable feature extractors for panoramas**: The feature extractor trained via satellite-to-satellite pairs generalizes to pinhole cameras but not to panoramic images (due to the mapping difference: straight lines vs. curves), requiring additional treatment.
2. **Remaining gap in longitudinal localization**: Weakly-supervised localization along the driving direction (longitudinal) is still visibly inferior to some fully-supervised methods.
3. **Limited batch size**: Computing spatial correlation consumes substantial memory, restricting the batch size to 8 (and even 4 when using Geo. Trans.), which affects contrastive learning performance.
4. **Under-explored aerial-view synthesis**: The authors admit that the current Homography projection might not be optimal, and better aerial-view synthesis under weak representation remains to be developed.

## Related Work & Insights

- **Shi et al. (2023)**: A fully-supervised ground-to-satellite localization method; our orientation regressor and Homography projection inherit from its design.
- **SliceMatch / OrienterNet**: Representative fully-supervised methods that excel in same-area evaluation but lack cross-area generalization.
- **Tang et al.**: A self-supervised localization strategy for LiDAR/Radar; our work extends similar ideas to cameras.
- **Insights**: The advantages of weak supervision in pose estimation are worth extending to other visual localization tasks. "Utilizing the data's inherent structures to construct supervisory signals" is a general paradigm for resolving annotation bottlenecks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First weakly-supervised ground-to-satellite localization method; the combined design of self-supervised orientation and contrastive translation is highly ingenious.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual datasets (KITTI + VIGOR), same/cross-area evaluation, detailed ablations, and comparison with multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, in-depth experimental analysis, and rich supplementary material.
- **Value**: ⭐⭐⭐⭐⭐ — Dramatically reduces the need for high-precision data annotation for camera localization; offers highly practical cross-area generalization performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](../../ICCV2025/remote_sensing/geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)
- [\[ECCV 2024\] ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations](congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)
- [\[ECCV 2024\] Learning Representations of Satellite Images From Metadata Supervision](learning_representations_of_satellite_images_from_metadata_supervision.md)
- [\[ECCV 2024\] Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)
- [\[ICCV 2025\] AstroLoc: Robust Space to Ground Image Localizer](../../ICCV2025/remote_sensing/astroloc_robust_space_to_ground_image_localizer.md)

</div>

<!-- RELATED:END -->
