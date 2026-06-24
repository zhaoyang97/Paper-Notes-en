---
title: >-
  [Paper Note] LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation
description: >-
  [ECCV 2024][Human Understanding][Category-Level Object Pose Estimation] The LaPose framework is proposed to model object shape uncertainty using the Laplacian Mixture Model (LMM). Combined with a dual-stream architecture comprising a DINOv2 general 3D stream and a Convolutional specialized feature stream, it predicts the NOCS coordinate distribution. It also introduces a scale-invariant pose representation to resolve the inherent scale ambiguity in RGB-only scenarios…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Category-Level Object Pose Estimation"
  - "RGB-only"
  - "Laplacian Mixture Model"
  - "Scale-Invariant Representation"
  - "PnP"
date: 2026-05-08
content_hash: c947d9198ca2851e
---

# LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation

**Conference**: ECCV 2024  
**arXiv**: [2409.15727](https://arxiv.org/abs/2409.15727)  
**Code**: [github.com/lolrudy/LaPose](https://github.com/lolrudy/LaPose)  
**Area**: Human Understanding  
**Keywords**: Category-Level Object Pose Estimation, RGB-only, Laplacian Mixture Model, Scale-Invariant Representation, PnP

## TL;DR

The LaPose framework is proposed to model object shape uncertainty using the Laplacian Mixture Model (LMM). Combined with a dual-stream architecture comprising a DINOv2 general 3D stream and a Convolutional specialized feature stream, it predicts the NOCS coordinate distribution. It also introduces a scale-invariant pose representation to resolve the inherent scale ambiguity in RGB-only scenarios, achieving SOTA performance on the NOCS dataset.

## Background & Motivation

Category-level object pose estimation requires predicting the 9DoF pose (rotation, translation, and size). Although RGBD methods perform well, they are limited by the availability of depth sensors. RGB-only methods face two core challenges:

**Shape Uncertainty**: The lack of depth information makes predicting the 3D shape of an object more difficult, especially in regions with large intra-category shape variations (e.g., the length of a camera lens cannot be determined from a frontal view), which increases the uncertainty of NOCS coordinate predictions.

**Scale Ambiguity**: It is impossible to distinguish between "a large object far from the camera" and "a small object close to the camera" solely from an RGB image, making absolute scale prediction an ill-posed problem.

Limitations of existing RGB methods (such as MSOS, OLD-Net, and DMSR): They treat NOCS predictions for all pixels equally, rely on RANSAC to filter out outliers (which is slow and non-robust), and fail to adequately address scale ambiguity.

## Method

### Overall Architecture

The overall pipeline of LaPose:
1. MaskRCNN detects and crops the target object.
2. **Dual-Stream Feature Extraction**: General 3D information stream (DINOv2) + Specialized feature stream (ConvNet).
3. The two streams independently predict the mean and variance of pixel-level Laplacian distributions.
4. Merged into a **Laplacian Mixture Model (LMM)** to establish 2D-3D correspondences.
5. A PnP module solves for rotation and normalized translation.
6. An independent size head predicts the normalized object size.

### Key Designs

1. **Laplacian Mixture Model (LMM)**: The core innovation. It models the NOCS coordinates of each pixel as a probability distribution rather than a deterministic value, explicitly quantifying shape uncertainty. The Laplacian distribution is preferred over the Gaussian distribution due to its greater robustness to outliers (L1 distance vs. L2 distance). The two streams predict $\text{Laplace}(\mu_{dino}, \sigma^2_{dino})$ and $\text{Laplace}(\mu_{conv}, \sigma^2_{conv})$ respectively, and simultaneously learn the mean and variance via a Laplacian aleatoric uncertainty loss:
    $\mathcal{L}_{3D} = \frac{\lambda}{\sigma^2} \| \mathbf{M}_{vis} \cdot (\mathbf{C}^{3D}_{gt} - \mu) \|_1 + \mathbf{M}_{vis} \cdot \log(\sigma^2)$
   When the coordinate error is large, $\sigma^2$ is forced to increase to minimize the first term. When the error is small, $\sigma^2$ is encouraged to decrease to minimize the logarithmic term, achieving self-supervised uncertainty learning. The PnP module dynamically aggregates information from both streams based on this and filters out erroneous correspondences in high-variance regions.

2. **Dual-Stream Feature Architecture**:

    - **General 3D Stream (DINOv2)**: Extracts category-independent, SE(3)-equivariant local features, which naturally fit the SE(3)-invariance of NOCS coordinates. However, DINOv2 lacks category-specific knowledge.
    - **Specialized Feature Stream (ConvNet)**: Trains a convolutional network to extract category-specific features, supplementing the limitations of DINOv2.
    - Ablation shows: Using either DINOv2 alone or ConvNet alone yields poorer performance than the dual-stream combination; the complementary nature of the dual streams brings a 6.7% gain in $10°0.5d$.

3. **Scale-Agnostic Pose Representation (SAP)**: Resolves the inherent scale ambiguity of RGB-only settings. The object is normalized such that its tight bounding box diagonal length is 1:

    - Normalized size: $\mathbf{s}_{norm} = \{s_x/d, s_y/d, s_z/d\}$ where $d = \sqrt{s_x^2 + s_y^2 + s_z^2}$
    - Normalized translation: $\mathbf{t}_{norm} = \{t_x/d, t_y/d, t_z/d\}$
    - Predicts the residual of normalized size: $\mathbf{s}_{out} = \mathbf{s}_{norm} - \mathbf{s}_{avg}$ to reduce learning difficulty.
    - Absolute scale $d$ is predicted by an independent MobileNet, decoupling scale and pose, preventing scale prediction errors from propagating to the pose.
    - Correspondingly, scale-agnostic evaluation metrics such as Normalized IoU (NIoU) and $10°0.2d$ are proposed.

### Loss & Training

$$\mathcal{L} = \lambda_{pose} \mathcal{L}_{pose} + \lambda_{3D}(\mathcal{L}_{3D\text{-}dino} + \mathcal{L}_{3D\text{-}conv})$$

- **$\mathcal{L}_{pose}$**: Supervises the scale-agnostic 9DoF parameters (rotation vector, normalized translation, normalized size).
- **$\mathcal{L}_{3D}$**: Laplacian aleatoric uncertainty loss, driving both streams' NOCS predictions and uncertainty estimation.
- Hyperparameters $\{\lambda_1, \lambda_2, \lambda_{pose}, \lambda_{3D}\} = \{15, 15, 1, 0.1\}$.
- Ranger optimizer, learning rate $10^{-3}$, cosine annealing, batch size 32.
- Dynamic Zoom-In to enhance detection robustness, 100 epochs (single model) or 150 epochs (per-category individual models).

## Key Experimental Results

### Main Results

**NOCS-REAL275 (Scale-Agnostic Metrics mAP %)**:

| Method | NIoU25 | NIoU50 | NIoU75 | 10°0.2d | 10°0.5d | 10° |
|------|--------|--------|--------|---------|---------|-----|
| MSOS | 36.9 | 9.7 | 0.7 | 3.3 | 15.3 | 17.0 |
| OLD-Net | 31.5 | 6.2 | 0.1 | 2.8 | 12.2 | 14.8 |
| DMSR | 57.2 | 38.4 | 9.7 | 26.0 | 44.9 | 36.9 |
| **LaPose** | **70.7** | **47.9** | 15.8 | 37.4 | **57.4** | **60.7** |
| **LaPose (M)** | 66.4 | **48.8** | **20.5** | **39.7** | 55.4 | 60.2 |

**NOCS-REAL275 (Absolute Scale Metrics mAP %)**:

| Method | IoU25 | IoU50 | IoU75 | 10°10cm |
|------|-------|-------|-------|---------|
| DMSR | 37.4 | 16.3 | 3.2 | 25.2 |
| **LaPose (M)** | **40.2** | **18.3** | **4.1** | 27.7 |
| **LaPose** | 41.2 | 17.5 | 2.6 | **30.5** |

### Ablation Study

**NOCS-REAL275 Component Ablation**:

| Config | $\mathcal{F}_{conv}$ | $\mathcal{F}_{dino}$ | $\sigma^2$ | SAP | NIoU25 | 10°0.5d |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| (A) Conv only | ✓ | | | ✓ | 60.3 | 46.1 |
| (B) Without SAP | ✓ | | | | 37.8 | 33.2 |
| (C) DINO only | | ✓ | | ✓ | 61.7 | 38.5 |
| (D) Dual Stream, No Distribution | ✓ | ✓ | | ✓ | 64.9 | 52.8 |
| (E) Conv+Lap | ✓ | | Lap | ✓ | 65.5 | 51.2 |
| (F) Conv+Gaus | ✓ | | Gaus | ✓ | 59.1 | 44.7 |
| **(G) Full LaPose** | **✓** | **✓** | **Lap** | **✓** | **70.7** | **57.4** |

### Key Findings

- **SAP is Crucial**: Without SAP, NIoU25 plummets from 60.3% to 37.8%, establishing the severe interference of scale ambiguity during training.
- **Laplacian Outperforms Gaussian**: Using a Gaussian distribution actually degrades performance (59.1% vs 60.3%), proving that the outlier robustness of the L1 loss is key in this task.
- **Complementary Dual Streams**: Both DINOv2 and ConvNet have distinct strengths, and combining them increases NIoU25 by approximately 5%.
- **LMM Modeling**: The variance map correlates highly with NOCS error, effectively guiding the PnP module to focus on low-uncertainty regions.
- The advantage in rotation prediction is particularly outstanding (the $10°$ metric reaches 60.7% vs. DMSR's 36.9%), boosting performance by 23.8%.

## Highlights & Insights

- **Probabilistic Shape Modeling**: Upgrading deterministic NOCS predictions to probability distributions elegantly solves the problem of "which pixels are more trustworthy."
- **Exploration of DINOv2's 3D Capabilities**: Utilizing the SE(3)-equivariant features of DINOv2 to assist NOCS prediction offers a new paradigm for applying foundation models to 3D tasks.
- **Scale Decoupled Design**: SAP not only improves training efficiency but also provides more meaningful evaluation metrics (NIoU), contributing to the standardization of metrics in this field.
- Realizes an inference speed of approximately 10 FPS, demonstrating potential for real-time applications.

## Limitations & Future Work

- The absolute scale $d$ is predicted by an independent MobileNet, and fully decoupling it from pose estimation may result in the loss of useful joint information.
- The method relies on MaskRCNN detection results; a failure in detection directly causes subsequent processes to fail.
- DINOv2 backbone is frozen; fine-tuning might further improve performance but would increase training costs.
- Validation has currently only been performed on the 6 NOCS categories, without extension to more categories or open-vocabulary settings.
- The multi-model version (M) trains a separate model for each category, which limits scalability.

## Related Work & Insights

- DMSR uses the normals and relative depth of pre-trained DPT models as additional inputs, which inspired the idea of "introducing external 3D priors."
- The end-to-end probabilistic PnP framework of EPro-PnP influenced the design of the PnP module.
- The effectiveness of DINOv2 in 3D tasks (as validated in SecondPose, SD-3D, etc.) motivated this paper.
- The Laplacian aleatoric uncertainty loss originates from monocular 3D detection works such as MonoPair.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — LMM shape uncertainty modeling + dual-stream architecture + SAP compose a tightly integrated set of three innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations, finding and fixing a bug in the evaluation script, though datasets are limited to NOCS.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous logic trace from problem motivation to solution design, with clear and professional diagrams.
- **Value**: ⭐⭐⭐⭐ — SOTA in RGB-only category-level pose estimation, advancing the standardization of scale-agnostic evaluation metrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [\[ECCV 2024\] U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](../../CVPR2025/human_understanding/gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[ECCV 2024\] FoundPose: Unseen Object Pose Estimation with Foundation Features](foundpose_unseen_object_pose_estimation_with_foundation_features.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)

</div>

<!-- RELATED:END -->
