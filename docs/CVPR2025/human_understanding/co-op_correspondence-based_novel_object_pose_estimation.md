---
title: >-
  [Paper Note] Co-op: Correspondence-based Novel Object Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][6DoF Pose Estimation] This paper proposes Co-op, a correspondence-based 6DoF pose estimation framework for novel objects. In the coarse estimation stage, a hybrid representation (patch-level classification + offset regression) is used to estimate the initial pose quickly and accurately with only 42 templates. In the refinement stage, probabilistic flow regression combined with differentiable PnP is utilized for end-to-end optimization…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "6DoF Pose Estimation"
  - "Novel Object Generalization"
  - "Correspondence Matching"
  - "Hybrid Representation"
  - "Probabilistic Flow Regression"
date: 2026-05-08
content_hash: fef38044c2aed236
---

# Co-op: Correspondence-based Novel Object Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.17731](https://arxiv.org/abs/2503.17731)  
**Code**: Yes (inference, NAVER LABS)  
**Area**: Human Understanding  
**Keywords**: 6DoF Pose Estimation, Novel Object Generalization, Correspondence Matching, Hybrid Representation, Probabilistic Flow Regression

## TL;DR
This paper proposes Co-op, a correspondence-based 6DoF pose estimation framework for novel objects. In the coarse estimation stage, a hybrid representation (patch-level classification + offset regression) is used to estimate the initial pose quickly and accurately with only 42 templates. In the refinement stage, probabilistic flow regression combined with differentiable PnP is utilized for end-to-end optimization, significantly outperforming existing methods on seven core datasets of the BOP Challenge.

## Background & Motivation

**Background**: 6DoF object pose estimation is crucial in scenarios such as robotic grasping and augmented reality. Traditional methods require retraining for each new object, which limits their utility. Model-based novel object pose estimation methods leverage 3D CAD models to achieve generalization through template matching or feature matching, such as MegaPose and GenFlow.

**Limitations of Prior Work**: (1) Template matching methods (such as MegaPose) require a large number of templates for exhaustive comparison, leading to high computational overhead; (2) DINOv2-based feature matching methods (such as FoundPose, GigaPose) are essentially detect-and-describe frameworks that rely on segmentation masks as feature detectors, making them non-robust to noisy masks; (3) Flow regression in render-and-compare refinement methods is easily affected by inaccurate optical flow, and RANSAC is sensitive to outlier distributions.

**Key Challenge**: High-precision pose estimation requires dense correspondences and reliable confidence estimation. However, existing methods either suffer from low efficiency in the coarse estimation stage (requiring a large number of templates) or exhibit poor robustness in the refinement stage (due to insufficient handling of outlier optical flows).

**Goal**: To design an efficient and robust two-stage pose estimation framework based on correspondences in both stages, achieving fast and accurate coarse estimation with a small number of templates alongside precise refinement.

**Key Insight**: Reframe pose estimation as a correspondence matching problem between two images. A hybrid representation of classification (discretization) + regression (continualization) is used in the coarse estimation stage, while probabilistic flow modeling is employed in the refinement stage to learn the uncertainty of correspondences.

**Core Idea**: Utilize patch-level classification to determine the approximate corresponding region (for robustness) and offset regression to pin down the exact location within the patch (for precision). The combination of both enables highly accurate coarse estimation with fewer templates. In the refinement stage, reliable probabilistic confidence is obtained by learning the parameters of the Laplace distribution of the flow.

## Method

### Overall Architecture
Given the cropped query image and the object's CAD model, Co-op operates in two stages: (1) **Coarse Estimation**—Finds the best match from 42 pre-rendered templates, estimates semi-dense correspondences, and solves for the initial pose using EPnP+RANSAC; (2) **Refinement**—Renders the image based on the initial pose, estimates the probabilistic dense optical flow and confidence between the query and rendered images, and solves for the precise pose using a differentiable PnP solver. Both stages share a ViT encoder + Transformer decoder architecture.

### Key Designs

1. **Hybrid Representation — Coarse Estimation**:

    - Function: Achieve fast and accurate initial pose estimation with a minimal number of templates (42).
    - Mechanism: Pass both the query image and templates through a ViT encoder (downsampled by 16 times) to obtain feature maps. For each query patch, predict: (1) a classification tensor $\mathcal{C} \in \mathbb{R}^{H/16 \times W/16 \times K}$ indicating which template patch matches (where there are K=H/16×W/16+1 classes, with the last class representing no match/occlusion); (2) an offset $\mathcal{U} \in \mathbb{R}^{H/16 \times W/16 \times 2}$ to pin down the precise location within the matched template patch (range [-0.5, 0.5]). The final corresponding position is formulated as $\mathcal{M}^T_{i,j} = (\text{patch中心} + \mathcal{U}_{i,j}) \times 16$.
    - Design Motivation: Directly regressing continuous coordinates is not robust to domain shift, while pure classification accuracy is limited by the patch resolution. The hybrid representation combines the robustness of classification (learning low-level information to resist domain shift) with the precision of regression (sub-patch localization). Consequently, it achieves the same efficacy with only 42 templates as previous methods did with hundreds of templates.

2. **Probabilistic Flow Regression — Refinement**:

    - Function: Precisely correct the initial pose via render-and-compare.
    - Mechanism: Append a DPT module after the refinement model to achieve pixel-level prediction. Unlike traditional flow regression which outputs only the mean, this method models the flow as a Laplace distribution $p(Y|\mathcal{I}_Q, \mathcal{I}_R; \theta)$, simultaneously predicting the mean $\mu$ and the scale parameter $b$. The flow probability $P_R = P(\|y - \mu\|_1 < R) = 1 - \exp(-R/b)$ provides an interpretable measure of accuracy. The final flow confidence $\mathcal{W}$ is obtained by an element-wise multiplication of three terms: certainty (non-occluded probability) × sensitivity (richness of pose-related information) × flow probability.
    - Design Motivation: GenFlow only learns confidence without improving the accuracy of the flow itself, and PFA relies on RANSAC to eliminate inaccurate flows but is sensitive to the distribution of outliers. Probabilistic flow enables the model to simultaneously enhance flow accuracy and uncertainty estimation, achieving optimal 6D poses through end-to-end training.

3. **Differentiable PnP End-to-End Training**:

    - Function: Directly convert flow and confidence into pose updates to achieve end-to-end gradient propagation.
    - Mechanism: Utilize a differentiable PnP solver based on Levenberg-Marquardt. Given the rendered depth map, it converts 2D correspondences into 3D points, and solves for $\mathbf{P}_{\text{refined}}$ after weighting with the confidence $\mathcal{W}$. The entire process is differentiable, allowing 6D pose loss gradients to propagate directly back to the flow prediction and confidence networks.
    - Design Motivation: Distinct from RANSAC, end-to-end differentiable PnP enables the network to learn what combination of flow and confidence yields the best pose.

### Loss & Training
Coarse estimation stage: Categorical cross-entropy loss is used for classification, and L1 loss for offset regression. Refinement stage: Laplace negative log-likelihood is used to train the flow probability, binary cross-entropy is used for certainty (occluded vs. visible), and sensitivity is learned end-to-end through the 6D pose loss. An optional Pose Selection module is employed to further improve accuracy.

## Key Experimental Results

### Main Results

| Method | BOP Average AR | No. Templates | Speed |
|------|----------|--------|------|
| Co-op | **SOTA** | 42 | Fast |
| MegaPose | Second-best | 576 | Slow |
| GenFlow | Competitive | Many | Slow |
| GigaPose | Competitive | 42 | Fast |
| FoundPose | Competitive | Many | Medium |

### Ablation Study

| Configuration | Key Metric Changes |
|------|-------------|
| Classification only (no offset) | Accuracy limited by patch resolution (16×16) |
| Regression only (no classification) | Non-robust to domain shift, large errors increase |
| Hybrid representation | Combines the advantages of both, achieving optimal performance with 42 templates |
| Deterministic flow (no probability) | Unreliable confidence estimation, refinement performance degrades |
| Probabilistic flow + Differentiable PnP | Best, end-to-end optimization significantly improves accuracy |

### Key Findings
- Hybrid representation is a key innovation: Combining the robustness of classification with the precision of regression allows 42 templates to outperform hundreds of templates in other methods.
- Probabilistic flow is more effective than deterministic flow + RANSAC: Learning uncertainty allows the network to automatically focus on reliable regions.
- End-to-end differentiable PnP performs better than standalone RANSAC post-processing.
- Accomplishes SOTA results on all seven core datasets of the BOP Challenge, showing distinct advantages especially on occlusion and textureless objects.

## Highlights & Insights
- The design that applies correspondence matching through both stages is highly unified, allowing the model to learn low-level geometric and structural information, naturally resisting domain shift.
- The "coarse-to-fine" approach of the hybrid representation (classification anchors the region $\rightarrow$ offset precisely localizes) can be migrated to other tasks requiring robust dense matching (e.g., visual localization, pose transfer).
- The three-factor decomposition of flow confidence (certainty × sensitivity × flow probability) offers excellent interpretability, with each factor having a clear physical meaning.

## Limitations & Future Work
- Reliance on object bounding boxes provided by CNOS or SAM-6D limits the entire pipeline when detection fails.
- Using only RGB input lacks depth information, which might cause ambiguities in symmetric objects.
- In extreme occlusion scenarios (>80%), available correspondences are too sparse, which could result in a significant drop in performance.
- Adaptation/selection strategies for learning more template viewpoints could be explored.

## Related Work & Insights
- **vs MegaPose**: Requires exhaustive comparison of 576 templates, which is computationally expensive. Co-op achieves better results with only 42 templates and the hybrid representation.
- **vs GigaPose**: Similarly uses fewer templates but relies on a detect-and-describe framework based on DINOv2 features, which is sensitive to segmentation mask noise. The detector-free approach of Co-op is more robust.
- **vs GenFlow**: Uses deterministic flow + RANSAC for refinement. The probabilistic flow + differentiable PnP of Co-op provides more precise and reliable estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ The designs of hybrid representation and probabilistic flow are novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on seven core BOP datasets with solid ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described with rich illustrations.
- Value: ⭐⭐⭐⭐⭐ Direct application value for robotic grasping and augmented reality, delivering SOTA results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Structure-Aware Correspondence Learning for Relative Pose Estimation](structure-aware_correspondence_learning_for_relative_pose_estimation.md)
- [\[CVPR 2025\] Any6D: Model-free 6D Pose Estimation of Novel Objects](any6d_model-free_6d_pose_estimation_of_novel_objects.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
