---
title: >-
  [Paper Note] MinCD-PnP: Learning 2D-3D Correspondences with Approximate Blind PnP
description: >-
  [ICCV 2025][3D Vision][Image-to-point cloud registration] This paper proposes MinCD-PnP, which reduces the computationally expensive Blind PnP to a problem of minimizing the Chamfer distance between 2D-3D keypoints via a…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Image-to-point cloud registration"
  - "2D-3D correspondences"
  - "PnP"
  - "Chamfer distance"
  - "multi-task learning"
date: 2026-05-08
content_hash: ccc2303bb4263206
---

# MinCD-PnP: Learning 2D-3D Correspondences with Approximate Blind PnP

**Conference**: ICCV 2025
**arXiv**: [2507.15257](https://arxiv.org/abs/2507.15257)  
**Code**: [https://github.com/anpei96/mincd-pnp-demo](https://github.com/anpei96/mincd-pnp-demo)  
**Area**: 3D Vision
**Keywords**: Image-to-point cloud registration, 2D-3D correspondences, PnP, Chamfer distance, multi-task learning

## TL;DR

This paper proposes MinCD-PnP, which reduces the computationally expensive Blind PnP to a problem of minimizing the Chamfer distance between 2D-3D keypoints via a triple approximation strategy. A lightweight multi-task learning module, MinCD-Net, is designed and integrated into existing I2P registration frameworks, achieving significant improvements in inlier ratio and registration recall under cross-scene and cross-dataset settings.

## Background & Motivation

Image-to-point cloud (I2P) registration is a fundamental task in computer vision, aiming to establish 2D-3D correspondences between image pixels and 3D points in a point cloud, followed by pose estimation via PnP algorithms. This task is widely applied in visual localization, navigation, SLAM, and 3D reconstruction.

**Limitations of Prior Work**: Mainstream deep learning-based I2P registration methods (e.g., P2-Net, 2D3D-MATR) establish correspondences through pixel-to-point feature matching, but this purely feature-level approach ignores the projective geometric constraints of 2D-3D correspondences (i.e., a valid correspondence ⟨q,p⟩ must satisfy $q = \pi(Tp)$), making it difficult to effectively reject outliers.

**Problem with Differentiable PnP**: To introduce geometric constraints, recent methods adopt differentiable PnP (e.g., EPro-PnP) to supervise correspondence learning. However, differentiable PnP is highly sensitive to noise and outliers in predicted correspondences — when the network inevitably generates incorrect correspondences, the PnP-estimated pose becomes unreliable, in turn hindering the effectiveness of correspondence learning.

**Inspiration from Blind PnP**: Blind PnP achieves robustness to noise and outliers by jointly optimizing the transformation matrix $T$ and the correspondence matrix $C$. However, its computational complexity is prohibitive (involving a Boolean matrix search of size $M \times N$), rendering it incompatible with gradient backpropagation in deep learning frameworks.

**Core Problem**: Can the robustness of Blind PnP be retained while reducing its computational complexity to make it applicable for end-to-end correspondence learning? This paper answers affirmatively: through a triple approximation strategy, Blind PnP is reformulated as an optimizable problem of minimizing Chamfer distance.

## Method

### Overall Architecture

MinCD-PnP consists of two components: (1) a theoretical triple approximation that reformulates Blind PnP into the MinCD-PnP problem; and (2) the MinCD-Net module at the implementation level, which solves MinCD-PnP via multi-task learning. MinCD-Net can be plug-and-play integrated into existing I2P registration architectures (e.g., 2D3D-MATR).

### Key Designs

1. **Approximation I: From Inlier Maximization to Chamfer Distance Minimization**

    - Function: Eliminates the Boolean correspondence matrix $C$ in Blind PnP, reformulating the inlier count maximization as Chamfer distance minimization.
    - Mechanism: Through inequality derivation, it is shown that $\max_{\mathbf{T},\mathbf{C}} \kappa(\mathbf{T},\mathbf{C}) \leq \max_{\mathbf{T}} \kappa^{\star}(\mathbf{T})$, where $\kappa^{\star}$ is an upper bound in Chamfer distance form. Leveraging this bound, the original discrete combinatorial optimization is relaxed into continuous Chamfer distance minimization: $L_{\text{Chamfer}}(\mathbf{T}|\mathbf{S}_I, \mathbf{S}_P) = \sum_{q \in S_I} \min_{p \in S_P} \|q - \pi(Tp)\|^2 + \sum_{p \in S_P} \min_{q \in S_I} \|q - \pi(Tp)\|^2$
    - Design Motivation: The $M \times N$ Boolean matrix search is computationally prohibitive, whereas Chamfer distance requires only nearest-neighbor search per point, substantially reducing complexity while remaining differentiable.

2. **Approximation II: Keypoint Sampling to Reduce Chamfer Distance Computation**

    - Function: Samples representative keypoints from the full pixel set and point cloud, reducing the Chamfer distance matrix from $M \times N$ (~$10^{11}$) to $M_0 \times N_0$ (~$10^6$).
    - Mechanism: Keypoint sets $K_I$ and $K_P$ are sampled from $S_I$ and $S_P$ respectively, and $L_{\text{Chamfer}}(\mathbf{T}|\mathbf{K}_I, \mathbf{K}_P)$ replaces the full computation. With 2D/3D keypoint counts on the order of $10^3$, the matrix size is reduced by approximately $10^5$.
    - Design Motivation: Even after eliminating the Boolean matrix, computing Chamfer distance over all pixels and points remains infeasible; keypoint sampling preserves representativeness while drastically reducing computational cost.

3. **Approximation III: 2D Keypoint-Guided 3D Keypoint Learning**

    - Function: Simplifies joint learning of $K_I$ and $K_P$ to learning only $K_P$, with $K_I$ provided by a pretrained detector.
    - Mechanism: The Shi-Tomasi corner detector is used to extract 2D keypoints $K_I$ in advance. For each 2D keypoint $q$, the nearest 3D point is found via feature matching: $p_q^{\star} = \arg\min_{p} d(\mathbf{f}_q^{2D}, \mathbf{f}_p^{3D})$, and 3D keypoint learning is supervised by an IoU-style loss $L_{\text{key}}(q)$. A threshold $s_{th} = e^{-0.4}$ is introduced to filter low-confidence matches.
    - Design Motivation: Jointly learning 2D and 3D keypoints with sufficient inliers is difficult, whereas mature 2D keypoint detectors already provide good spatial representativeness in image space.

4. **MinCD-Net Multi-Task Learning Module**

    - Function: Unifies the three approximations of MinCD-PnP into a single end-to-end trainable lightweight module.
    - Mechanism: The final optimization objective is $\varphi^{\star} = \arg\min_\varphi \left( L_{\text{corr}} + \lambda_1 \sum_{q \in K_I} L_{\text{key}}(q) + \lambda_2 \min_T L_{\text{Chamfer}}(T|K_I, K_P) \right)$. A Point Transformer encodes 2D/3D keypoint features; global features are then passed through an MLP to predict pose $T$ (se(3)→SE(3)).
    - Design Motivation: The three loss terms work synergistically — $L_{\text{corr}}$ handles feature-space matching, $L_{\text{key}}$ ensures 3D keypoints approximate 2D keypoints, and $L_{\text{Chamfer}}$ enforces global geometric consistency.

### Loss & Training

The total loss consists of three terms:
- **$L_{\text{corr}}$**: Circle Loss for pixel-to-point feature matching, alleviating extreme inlier/outlier imbalance.
- **$L_{\text{key}}$**: Keypoint learning loss in IoU form, ensuring learned 3D keypoints approximate 2D keypoints within a projection error threshold $\tau$.
- **$L_{\text{Chamfer}}$**: Chamfer distance loss, computed after pose prediction via Point Transformer + MLP as a reprojection-based Chamfer distance.

Training strategy: The first 20 epochs train only $L_{\text{corr}}$ ($\lambda_1 = \lambda_2 = 0$); the subsequent 20 epochs incorporate $L_{\text{key}}$ and $L_{\text{Chamfer}}$ ($\lambda_1 = 0.2$, $\lambda_2 = 0.0001$). The threshold $\tau$ is adaptively determined based on camera intrinsics and the RR threshold.

## Key Experimental Results

### Main Results

Cross-scene generalization (7-Scenes dataset, trained on Office, tested on other scenes):

| Method | IR (AVG) | RR (AVG) | Gain (IR) | Gain (RR) |
|--------|----------|----------|-----------|-----------|
| P2-Net | 0.393 | 0.510 | - | - |
| MATR | 0.456 | 0.491 | - | - |
| MATR+Diff.PnP | 0.463 | 0.501 | +0.007 | +0.010 |
| MATR+BPnPNet | 0.486 | 0.578 | +0.030 | +0.087 |
| **MATR+MinCD-Net** | **0.568** | **0.647** | **+0.112** | **+0.156** |

Cross-dataset generalization (trained on Kitchen, tested on RGBD-V2):

| Method | IR (AVG) | RR@0.1 (AVG) |
|--------|----------|--------------|
| MATR | 0.291 | 0.692 |
| MATR+Diff.PnP | 0.303 | 0.708 |
| MATR+BPnPNet | 0.315 | 0.799 |
| **MATR+MinCD-Net** | **0.371** | **0.870** |

### Ablation Study

| Configuration | IR (AVG) | RR (AVG) | Note |
|---------------|----------|----------|------|
| MATR (baseline) | 0.456 | 0.491 | No PnP constraint |
| + Diff.PnP | 0.463 | 0.501 | Differentiable PnP, limited gain |
| + BPnPNet | 0.486 | 0.578 | Weighted Blind PnP, improved but gradient-limited |
| + MinCD-Net ($L_{\text{key}}$ only) | ~0.52 | ~0.58 | Keypoint learning only |
| + MinCD-Net (full) | **0.568** | **0.647** | Three losses combined, best performance |

### Key Findings

- MinCD-Net improves IR by ~+0.10 and RR by ~+0.15 over Diff.PnP in the cross-scene setting, demonstrating that the robustness advantage of Blind PnP is effectively preserved.
- In the most challenging Stairs scene (severe texture scarcity), MinCD-Net improves RR from 0.226 (Diff.PnP) to 0.571, a gain of ~150%.
- MinCD-Net also achieves top performance in cross-dataset experiments, demonstrating strong generalization.
- Using the multi-dataset pretrained model MinCD-Net†, IR reaches 0.581 and RR reaches 0.914 on RGBD-V2.

## Highlights & Insights

- The paper successfully introduces Blind PnP — a classical but computationally expensive robust estimation method — into a deep learning framework via a three-level theoretical approximation, representing an excellent fusion of theoretical rigor and engineering practicality.
- The plug-and-play design of MinCD-Net allows integration into arbitrary I2P registration architectures rather than being tied to a specific network.
- The keypoint-guided strategy cleverly leverages mature 2D detectors to circumvent the difficulty of joint learning.
- Replacing exact PnP solving with Chamfer distance confers inherent robustness to noise and outliers.

## Limitations & Future Work

- 2D keypoints rely on a pretrained detector (Shi-Tomasi), which may be suboptimal in scenes with extreme texture scarcity; learned keypoint detection could be considered as an alternative.
- The theoretical bounds of the triple approximation are upper bounds rather than tight bounds; a more rigorous optimality analysis remains to be conducted.
- Evaluation is currently limited to indoor scenes; performance on large-scale outdoor environments (e.g., LiDAR-camera registration in autonomous driving) remains to be explored.
- Encoding keypoints with Point Transformer may incur increasing computational overhead as the number of keypoints grows.

## Related Work & Insights

- **2D3D-MATR**: The strong baseline for I2P registration, upon which MinCD-Net is integrated to achieve improvements.
- **EPro-PnP**: End-to-end probabilistic PnP with some robustness to noise, but struggles with large numbers of outliers.
- **BPnPNet**: The only prior work applying Blind PnP to correspondence learning, but with limited gradient information due to RANSAC-based filtering.
- Insight: The strategy of "softening" classical robust estimation methods and embedding them into deep learning is broadly applicable to other geometric estimation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Triple approximation is novel, though it builds on existing theoretical frameworks)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets, cross-scene + cross-dataset, comparison with multiple methods)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear, though mathematical notation is occasionally heavy)
- Value: ⭐⭐⭐⭐ (Practical improvement for I2P registration; the plug-and-play module design is worth adopting)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](../../NeurIPS2025/3d_vision/concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)

</div>

<!-- RELATED:END -->
