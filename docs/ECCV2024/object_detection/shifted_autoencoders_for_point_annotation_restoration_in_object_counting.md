---
title: >-
  [Paper Note] Shifted Autoencoders for Point Annotation Restoration in Object Counting
description: >-
  [ECCV 2024][Object Detection][Object Counting] Proposes **Shifted AutoEncoders (SAE)**, an MAE-inspired point annotation restoration method: by applying random shifts to point annotations and training a UNet to restore them, the model learns "general location knowledge" while ignoring individual annotation noise. The trained SAE is used to restore original annotations to make them more consistent, which consistently improves the performance of any counting model (density-map…
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Object Counting"
  - "Point Annotation Restoration"
  - "Denoising Autoencoder"
  - "Annotation Consistency"
  - "General Preprocessing"
date: 2026-05-08
content_hash: 4b144a6188e88a54
---

# Shifted Autoencoders for Point Annotation Restoration in Object Counting

**Conference**: ECCV 2024  
**arXiv**: [2312.07190](https://arxiv.org/abs/2312.07190)  
**Code**: Yes (code and corrected annotations will be released)  
**Area**: Object Counting  
**Keywords**: Object Counting, Point Annotation Restoration, Denoising Autoencoder, Annotation Consistency, General Preprocessing

## TL;DR

Proposes **Shifted AutoEncoders (SAE)**, an MAE-inspired point annotation restoration method: by applying random shifts to point annotations and training a UNet to restore them, the model learns "general location knowledge" while ignoring individual annotation noise. The trained SAE is used to restore original annotations to make them more consistent, which consistently improves the performance of any counting model (density-map or localization-based), setting new records across 9 datasets.

## Background & Motivation

Object counting tasks (crowds, vehicles, cells, etc.) commonly use 2D point annotations to label objects. Although efficient (especially in dense/overlapping scenes), this annotation approach has inherent issues:

**Annotation Inconsistency**: Point location choices of different annotators for the same class of objects exhibit subjective variations. For example, in crowd counting, point annotations on heads roughly follow a Gaussian distribution (sharing a central location but with different shift noise). This inconsistency introduces ambiguity and confusion during the training phase, harming model counting accuracy.

Existing solutions (such as ADSCNet, BL, NoiseCC, RSI, etc.) mainly enhance the model's **tolerance** to noisy annotations by modifying loss functions or network structures, but:
- They do not directly improve the quality of the annotations themselves.
- They incur extra overhead every time a new model is trained.
- They are designed solely for density map methods and have not been validated on localization-based methods.

The authors propose a more direct scheme: **restore the annotations to make them more consistent before training the counting model**. This serves as a general preprocessing step, and the restored annotations can be used for any counting method.

## Method

### Overall Architecture

SAE consists of three steps:
1. **Shift Generation**: Apply random 2D shifts to the original point annotations.
2. **SAE Training**: Train a UNet to restore the shifted points to their original locations (learning a restoration vector field).
3. **Self-Restoration**: Treat the original annotations as "shifted points" and restore them using the trained SAE.

The core intuition generalizes from MAE: MAE learns general knowledge of images while ignoring specific details through mask-reconstruction (e.g., losing text when reconstructing a bus). SAE learns general annotation positions on objects while ignoring shift noise from individual annotators through shift-restoration.

### Key Designs

1. **Shifted Point Generation Strategy**: For each annotation point $p_i = (x_i, y_i)$, a random shift vector $v_i$ is applied, consisting of an angle $a_i$ (uniformly sampled from $(0, 2\pi)$) and a magnitude $m_i$ (uniformly sampled from $(0, r_i)$). The key lies in setting $r_i$—the sampling radius should approximate the spatial scale of the object:

   Simple scheme: $r_i^s = \alpha \times d_i$ ($\alpha \le 0.5$ to prevent overlap), where $d_i$ is the distance to the nearest neighbor.
   
   Improved scheme: $r_i = \alpha \times \min(d_i, \bar{d}_i^{\mathcal{N}_3})$, which additionally considers the average distance of the 3 nearest neighbors, preventing excessively large sampling regions for sparsely distributed objects that introduce too much background. $\alpha$ is set to 0.4 across all experiments.

2. **SAE Network and Training Target**: A lightweight UNet (VGG16 backbone) is used to predict a restoration vector field $F \in \mathbb{R}^{2 \times H \times W}$ containing restoration vectors in the x and y directions. Training uses the MSE loss, calculated only at the coordinates of the shifted points:

    $$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\|\hat{v}_i - (-v_i)\|^2$$
   
   where $\hat{v}_i = (F_x(\hat{p}_i), F_y(\hat{p}_i))$ is the predicted restoration vector, and $-v_i$ is the ground-truth inverse shift vector. Intuition: When the same class of head appears multiple times with scattered annotations, the optimal restoration strategy is to move them all toward the distribution center.

3. **Point Annotation Restoration**: Treating original annotations as "shifted points", after the SAE network predicts the restoration vector field, the restored annotations are:

    $$P_r = \{(x_i + F_x(p_i), y_i + F_y(p_i))\}$$
   
   By eliminating individual shift noise, the restored annotations achieve better consistency.

### Loss & Training

- Data Augmentation: Random scaling, horizontal flipping, random cropping to 512×512.
- Shift vectors are randomly regenerated in each epoch.
- Adam optimizer, weight decay 5e-4, lr 1e-4.
- 100 epochs, batch size 8, single RTX 3090 card.
- SAE is trained independently for each dataset (no additional data is introduced).

## Key Experimental Results

### Main Results

**Crowd Counting (SH PartA/B, UCF-QNRF, JHU++)**

| Method | SHA-MAE | SHA-MSE | SHB-MAE | SHB-MSE | QNRF-MAE | JHU-MAE |
|------|---------|---------|---------|---------|----------|---------|
| P2PNet | 52.8 | 85.8 | 6.5 | 10.9 | 91.7 | 66.8 |
| P2PNet & SAE | **48.2** (↓4.6) | **76.1** (↓9.7) | 6.2 (↓0.3) | 10.0 (↓0.9) | 85.9 (↓5.8) | 62.4 (↓4.4) |
| MAN | 55.6 | 93.2 | 7.1 | 10.5 | 77.5 | 53.2 |
| MAN & SAE | **52.2** (↓3.4) | **81.9** (↓11.3) | **5.4** (↓1.7) | **7.0** (↓3.5) | 74.2 (↓3.3) | **49.7** (↓3.5) |
| STEERER | 56.5 | 89.8 | 7.1 | 10.7 | 74.1 | 55.8 |
| STEERER & SAE | 54.3 (↓2.2) | 84.5 (↓5.3) | 6.4 (↓0.7) | 10.1 (↓0.6) | **71.4** (↓2.7) | 52.9 (↓2.9) |

**Remote Sensing Object Counting (RSOC Dataset)**

| Method| Building-MAE | Small-Vehicle-MAE | Large-Vehicle-MAE | Ship-MAE |
|------|-------------|-------------------|-------------------|----------|
| P2PNet | 6.3 | 63.1 | 8.1 | 28.2 |
| P2PNet & SAE | **5.7** (↓0.6) | **53.3** (↓9.8) | **7.0** (↓1.1) | **25.0** (↓3.2) |
| ASPDNet | 7.6 | 252.6 | 19.7 | 81.8 |
| ASPDNet & SAE | 6.7 (↓0.9) | 182.0 (↓70.6) | 16.9 (↓2.8) | 75.1 (↓6.7) |

**Cell Counting (MBM, ADI, DCC)**

| Method | MBM-MAE | MBM-MSE | ADI-MAE | DCC-MAE | DCC-MSE |
|------|---------|---------|---------|---------|---------|
| SAUNet | 5.7 | 7.7 | 14.3 | 3.0 | 4.8 |
| SAUNet & SAE | **4.2** (↓1.5) | **5.8** (↓1.9) | **11.2** (↓3.1) | **2.6** (↓0.4) | **3.4** (↓1.4) |

### Ablation Study

**Effect of Hyperparameter $\alpha$ on Performance (BL Method)**

| $\alpha$ | QNRF-MAE | QNRF-MSE | RSV-MAE | RSV-MSE | MBM-MAE | MBM-MSE |
|---|----------|----------|---------|---------|---------|---------|
| Baseline | 87.4 | 149.6 | 173.0 | 477.8 | 8.4 | 10.3 |
| 0.3 | 86.5 | 148.3 | 149.7 | 389.4 | 7.3 | 9.2 |
| **0.4** | **81.6** | 146.5 | **130.6** | **341.4** | **6.3** | 8.8 |
| 0.5 | 82.4 | **146.3** | 138.2 | 364.8 | 6.9 | **8.6** |
| 0.6 | 91.0 | 158.9 | 190.6 | 516.5 | 8.3 | 10.8 |

Performance drops at $\alpha=0.6$ due to sampling region overlap. $\alpha=0.4$ is optimal in most scenarios.

### Key Findings

- SAE brings consistent improvements to **all tested counting methods**, including density-map and localization-based methods.
- It is equally effective for **noise-tolerant methods** (BL, NoiseCC, RSI)—while these methods enhance model tolerance, SAE directly improves annotation quality, making them complementary.
- It is effective across all 5 baseline methods on **11 datasets** (3 domains), setting new records on 9 datasets.
- In additional artificial noise tests, SAE exhibits better robustness than BL and NoiseCC, showing minimal impact especially at moderate noise levels ($\le 0.4 \times r$).
- On average: MAE is reduced by 3.0 and MSE by 6.4 across 4 crowd counting datasets; MAE is reduced by 9.2 and MSE by 40.6 across 4 remote sensing datasets.

## Highlights & Insights

- **Elegant Analogy**: Intuitively mapping MAE's "mask-reconstruction to learn general knowledge" to "shift-restoration to learn general locations" for point annotations is highly ingenious.
- **Solves a Meta-Problem**: It is not tailored to a specific counting model but directly improves data quality, serving as a general preprocessing step.
- **Extremely Simple**: The core is just a UNet + MSE loss, without complex module designs.
- **Cross-Domain Generalization**: Validated and effective across three entirely different domains: crowd, remote sensing, and cells.
- **Additivity**: It is complementary to existing noise-tolerant methods, enabling further performance enhancements.

## Limitations & Future Work

- Author-disclosed: SAE relies on repeatable visual patterns of objects; its effectiveness might be limited for object categories with extremely large appearance variations.
- The estimation of the sampling radius $r_i$ is based on the nearest neighbor distance, which may be inaccurate when object size varies greatly or distribution is highly non-uniform.
- The current restoration is one-off (running SAE once). Could iterative restoration yield further improvements?
- Missing annotations (missed labels) are not addressed; it only handles positional shifts.
- The choice of $\alpha=0.4$ might not be optimal for all datasets.

## Related Work & Insights

- Deep connection to the MAE/DAE family: drawing inspiration from the principle of "learning the general while ignoring the specific" in self-supervised representation learning.
- Universally extendable to other tasks using point annotations (e.g., keypoint detection, pose estimation).
- Aligned with data cleaning and annotation correction research, offering an automated path without requiring extra human labeling.
- Unified improvements on both density-map and localization-based methods demonstrate that the root of the problem lies in the data rather than the model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The analogy from MAE to SAE is original and effective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (11 datasets, 3 domains, 5 baseline methods, noise-tolerant methods, noise robustness)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation, intuitive analogy, and well-designed experiments)
- Value: ⭐⭐⭐⭐⭐ (Simple, general, effective, with extremely high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)
- [\[ECCV 2024\] TAPTR: Tracking Any Point with Transformers as Detection](taptr_tracking_any_point_with_transformers_as_detection.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/object_detection/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[ICCV 2025\] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](../../ICCV2025/object_detection/yolo-count_differentiable_object_counting_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
