---
title: >-
  [Paper Note] Symmetry-Robust 3D Orientation Estimation
description: >-
  [ICML2025][3D Vision][3D orientation estimation] A robust two-stage 3D orientation estimation pipeline is proposed to handle rotational symmetry. The first stage recovers the orientation within an equivalence class of the octahedral symmetry group via quotient regression, and the second stage predicts one of the 24 octahedral flips using a classifier to achieve precise recovery, achieving state-of-the-art results on ShapeNet.
tags:
  - "ICML2025"
  - "3D Vision"
  - "3D orientation estimation"
  - "rotational symmetry"
  - "quotient regression"
  - "octahedral group"
  - "conformal prediction"
date: 2026-05-08
content_hash: bc27ad6a8c0ecb45
---

# Symmetry-Robust 3D Orientation Estimation

**Conference**: ICML2025  
**arXiv**: [2410.02101](https://arxiv.org/abs/2410.02101)  
**Code**: [GitHub](https://github.com/cscarv/3d-orienter)  
**Area**: 3D Rotation Estimation / Geometric Deep Learning  
**Keywords**: 3D orientation estimation, rotational symmetry, quotient regression, octahedral group, conformal prediction

## TL;DR

A robust two-stage 3D orientation estimation pipeline is proposed to handle rotational symmetry. The first stage recovers the orientation within an equivalence class of the octahedral symmetry group via quotient regression, and the second stage predicts one of the 24 octahedral flips using a classifier to achieve precise recovery, achieving state-of-the-art results on ShapeNet.

## Background & Motivation

**Core Problem**: 3D orientation estimation aligns the lateral, upward, and forward axes of an object with the coordinate axes, serving as a critical preprocessing step in 3D deep learning. However, current methods suffer from several limitations:

**$L_2$ regression degenerates on symmetrical shapes**: For shapes with rotational symmetry (e.g., a bench with 180° symmetry about the y-axis), the optimal $L_2$ regression solution is the Euclidean mean of all symmetrical orientations (Proposition 3.1), which is not a valid orientation. Even with a single non-trivial symmetry, the solution space degenerates into a submanifold of $SO(3)$.

**Discretized classification methods fail**: Discretizing $SO(3)$ into $K$ rotations for classification (Poursaeed et al., 2020) drops the accuracy to 1.6% when $K=100$.

**Limitations of prior work**: Upright-Net only predicts the up-axis and relies on a support plane prior; reinforcement learning methods are computationally expensive to train; classic PCA-based methods are not robust to asymmetric shapes.

**Key Insight**: Decomposing orientation estimation into two solvable subproblems—continuous regression (modding out octahedral symmetry) and discrete classification (24-class flip)—fundamentally circumvents the degeneracy problem caused by symmetry.

## Method

### Problem Formulation

Orientation estimation learns a mapping $f: \mathcal{S} \to SO(3)$ that maps a shape $S$ to its orientation matrix $\Omega_S = (\omega_S^x, \omega_S^y, \omega_S^z)$, where the three columns represent the lateral, upward, and forward axes, respectively. The orientation satisfies rotational equivariance: $\Omega_{RS} = R\Omega_S$.

### Stage 1: Quotient Orienter

**Intuition**: Direct $L_2$ regression requires predictions to be close to all symmetric orientations, leading to mean collapse. Quotient regression only requires proximity to any one of them.

Choosing $\hat{\mathcal{R}} = \mathcal{O}$ (the octahedral group, 24 rotational symmetries), the quotient $L_2$ loss is optimized:

$$\min_{f_\theta} \mathbb{E}_{R \sim U(SO(3)), (S,\Omega_S) \in \mathcal{D}} \left[ \min_{Q \in \mathcal{O}} \|f_\theta(RS) - RQ\Omega_S\|_F^2 \right]$$

- **Theoretical Guarantees (Proposition 3.2)**: The solution to quotient regression is $f^*(RS) = RQ^*\Omega_S$, which represents the correct orientation modded out by a rotation $Q^*$ in $\mathcal{O}$.
- **Network Architecture**: DGCNN operates on point clouds and outputs $\mathbb{R}^{3\times 3}$, which is projected onto $SO(3)$ via the constrained orthogonal Procrustes problem.
- **Test-Time Augmentation (TTA)**: Applies $K$ random rotations $R_k$ to the input and selects the prediction with the minimal average quotient distance to other predictions.

### Stage 2: Flipper

**Task**: Classify and predict $Q^* \in \mathcal{O}$ (24-class classification) from the quotient orienter output $f^*(RS) = RQ^*$.

The training loss is the cross-entropy loss:

$$\min_{p_\phi} \mathbb{E}_{Q \sim U(\mathcal{O})} \left[ \text{CE}(p_\phi(QS), \delta_Q) \right]$$

- During training, random rotational noise in the range $[0°, 10°]$ is added on top of the octahedral flips to simulate the errors of the first stage.
- **Theoretical Guarantees (Proposition 3.3)**: After the two-stage cascade, $((Q^*)^\top F)^\top f^*(RS)^\top RS = S$, which recovers the canonical orientation (modding out the object's intrinsic symmetry $F \in \mathcal{R}_S$).

### Adaptive Prediction Sets

Conformal prediction is utilized to output adaptive prediction sets for the Flipper. Flips are added in descending order of probability until the cumulative probability mass reaches a threshold $\tau$ (learned from the calibration set with a coverage probability $\alpha=0.3$). Symmetric or ambiguous shapes produce larger sets, leaving the selection of the optimal orientation to human users.

## Key Experimental Results

### Up-axis Estimation

| Method | ShapeNet Accuracy | ModelNet40 Accuracy |
|------|:-:|:-:|
| **Ours (with TTA)** | **89.2%** | **77.7%** |
| Ours (w/o TTA) | 85.3% | 72.3% |
| Upright-Net (Pang et al., 2022) | 69.5% | 62.3% |

Compared to the prior SOTA, the error rate is reduced by **64.6%** (with a 10° threshold).

### Full-orientation Estimation

| Method | Average Chamfer Distance ↓ |
|------|:-:|
| Upright-Net-Random | 0.10801 ± 0.13824 |
| Upright-Net-Oracle | 0.05481 ± 0.13016 |
| **Ours (TTA)** | **0.00856 ± 0.03960** |
| Ours (w/o TTA) | 0.01107 ± 0.04342 |
| **Ours (with APS)** | **0.00208 ± 0.01407** |

- Achieves an **84%** reduction in Chamfer Distance compared to the Oracle baseline.
- APS further reduces this by **4x**, with the median prediction set containing only 2 candidate orientations, and 90% containing $\le$ 8 candidates.

### Key Details

- Training data: All 55 classes of ShapeNet (not a subset), with a 90-10 split.
- Point cloud sampling: 10k points per mesh, and 2k points per iteration during training.
- Learning rate of $10^{-4}$, training the Quotient Orienter for 1919 epochs and the Flipper for 3719 epochs.
- Generalization tests: ModelNet40 and Objaverse (out-of-distribution qualitative results).

## Highlights & Insights

1. **Deep Integration of Theory and Engineering**: Three propositions (Props 3.1–3.3) rigorously prove the degeneracy mechanism of $L_2$ regression under symmetry and guarantee the correctness of the two-stage pipeline.
2. **Novel Quotient Regression Idea**: Modding the loss function out by the octahedral group using $\min_{Q \in \mathcal{O}}$ simplifies the ill-posed problem into a well-posed one in an elegant manner.
3. **All-Category Training**: This is the first orientation estimation method trained and evaluated on all 55 classes of ShapeNet without needing category labels.
4. **Engineering Application of Conformal Prediction**: Conformal prediction from statistics is cleverly introduced to 3D geometry tasks, generating adaptive prediction sets for human-in-the-loop disambiguation, showing high practicality.
5. **Significant SOTA Improvements**: The up-axis accuracy improves from 69.5% to 89.2%, and the full-orientation Chamfer Distance is reduced by 84%.

## Limitations & Future Work

1. **Dependence on ShapeNet Canonical Orientation Labels**: Requires large-scale, manually labeled canonical orientation data, while ShapeNet annotation quality is inconsistent.
2. **Limitations of the Octahedral Group Selection**: For symmetries that do not belong to the subgroups of the octahedral group (e.g., a regular dodecahedron with 5-fold rotational symmetry), theoretical guarantees may not fully hold.
3. **Limited Out-of-Distribution Generalization**: The Flipper performs poorly on Objaverse, suggesting generalization requires larger training sets.
4. **Older DGCNN Backbone**: The potential of more powerful backbone networks, such as Transformers, is under-explored.
5. **Rigid Rotations Only**: Reflection (improper rotation) and deformable objects are not considered.
6. **Inference Overhead from TTA**: Test-time augmentation introduces additional computational costs, limiting real-time applications.

## Related Work & Insights

- **Upright-Net (Pang et al., 2022)**: Predicts the up-axis by segmenting bottom points and fitting planes; a previous SOTA but limited to objects with supported surfaces.
- **Learned Canonicalization (Kaba et al., 2023)**: Learns canonicalization functions to achieve equivariance. This work can be viewed as an efficient canonicalization method for 3D rotational symmetry scenarios.
- **Conformal Prediction (Romano et al., 2020)**: Represents the first application of the statistical conformal prediction framework to geometric learning.
- **Insights**: The quotient regression concept can be generalized to other geometric learning problems involving symmetry (e.g., molecular conformation prediction and crystal symmetry).

## Rating

- Novelty: ⭐⭐⭐⭐ — The two-stage decomposition of quotient regression + classification is novel, with solid theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Complete ablation studies on all ShapeNet categories + ModelNet40 + Objaverse, though limited to point cloud inputs.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear theoretical derivations, well-articulated motivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — A practical tool for 3D preprocessing, though its scope of application is limited to rigid shape orientation estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Orientation Matters: Making 3D Generative Models Orientation-Aligned](../../NeurIPS2025/3d_vision/orientation_matters_making_3d_generative_models_orientation-aligned.md)
- [\[CVPR 2025\] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](../../CVPR2025/3d_vision/symmetry_strikes_back_from_single-image_symmetry_detection_to_3d_generation.md)
- [\[ICCV 2025\] χ: Symmetry Understanding of 3D Shapes via Chirality Disentanglement](../../ICCV2025/3d_vision/kh_symmetry_understanding_of_3d_shapes_via_chirality_disentanglement.md)
- [\[CVPR 2025\] Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](../../CVPR2025/3d_vision/leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)
- [\[CVPR 2026\] KASALv2: Fully Automatic 3D Rotational Symmetry Classification and Axis Localization](../../CVPR2026/3d_vision/kasalv2_fully_automatic_3d_rotational_symmetry_classification_and_axis_localizat.md)

</div>

<!-- RELATED:END -->
