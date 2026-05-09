---
title: >-
  [Paper Note] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training
description: >-
  [ICCV 2025][LLM Pretraining][scene coordinate regression] ACE-G decomposes a scene coordinate regressor into a scene-agnostic Transformer and a scene-specific map code, and achieves significant generalization gains under illumination and viewpoint variation by conducting alternating mapping/query pre-training across tens of thousands of scenes, while maintaining lightweight computational overhead.
tags:
  - ICCV 2025
  - LLM Pretraining
  - scene coordinate regression
  - visual relocalization
  - Transformer pretraining
  - generalization
  - map code
date: 2026-05-08
content_hash: 222dc75f2ec6387f
---

# ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training

**Conference**: ICCV 2025
**arXiv**: [2510.11605](https://arxiv.org/abs/2510.11605)
**Code**: [nianticspatial/ace-g](https://github.com/nianticspatial/ace-g) (available, includes pretrained models)
**Area**: Visual Localization
**Keywords**: scene coordinate regression, visual relocalization, Transformer pretraining, generalization, map code

## TL;DR
ACE-G decomposes a scene coordinate regressor into a scene-agnostic Transformer and a scene-specific map code, and achieves significant generalization gains under illumination and viewpoint variation by conducting alternating mapping/query pre-training across tens of thousands of scenes, while maintaining lightweight computational overhead.

## Background & Motivation

### State of the Field

**State of the Field**: Scene Coordinate Regression (SCR) is a learning-based visual relocalization approach whose core mechanism is to train a network that maps image patch features to corresponding 3D coordinates, followed by a PnP solver for camera pose estimation. The representative work ACE achieves high accuracy with only a few minutes of scene-specific training.

### Limitations of Prior Work

**Limitations of Prior Work**: SCR methods generalize far worse than classical feature matching approaches (e.g., hloc). When the imaging conditions of a query image (illumination, viewpoint, object placement) differ substantially from those of the training views, SCR models tend to fail.

### Root Cause

**Root Cause**: The fundamental issue lies in the training mechanism itself: prior SCR methods encode training views of a scene into the regressor's network weights, such that the regressor essentially overfits to the training views—by design rather than as a defect. As a result, the network lacks generalization to unseen query conditions.

### Root Cause

**Root Cause**: Classical feature matching methods are inherently robust to appearance changes by virtue of feature extractors pre-trained on large-scale data, whereas SCR methods lack this cross-scene pre-training capability.

### Paper Goals

**Paper Goals**: How can SCR methods acquire generalization to illumination changes, large viewpoint differences, and seasonal/weather variation, while retaining the advantages of rapid scene adaptation (a few minutes of training) and lightweight computation?

The central challenge is that conventional SCR couples "scene understanding" and "coordinate regression" within a single network, forcing the network to be trained from scratch for each scene and preventing the accumulation of cross-scene generalizable knowledge.

## Method

### Overall Architecture
ACE-G decomposes the conventional monolithic scene coordinate regressor into two components:
- **Scene-agnostic Transformer regressor**: A general-purpose coordinate regression network responsible for learning "how to infer 3D coordinates from features," which can be pre-trained across a large number of scenes.
- **Scene-specific map code**: A set of learnable encoding vectors that store 3D structural information of a specific scene, optimized via backpropagation.

The overall pipeline consists of three stages:
1. **Pre-training stage**: The Transformer is pre-trained across tens of thousands of scenes to acquire cross-scene generalization.
2. **Mapping stage**: For a new scene, the Transformer is frozen and only the map code is optimized to encode that scene.
3. **Relocalization stage**: Given a query image, the Transformer combines the map code to predict 2D–3D correspondences, from which a PnP solver estimates the camera pose.

### Key Designs

1. **Decoupling map code from the regressor**: Scene-specific information is separated from the network weights into an external map code. The map code is a set of learnable vectors that represent the 3D structure of a scene through interaction with the Transformer. This decoupling allows the Transformer to be shared and trained across many scenes, accumulating general coordinate regression capability. In the conventional ACE framework, all network weights are scene-specific; in ACE-G, only the map code is scene-specific while the Transformer weights remain fixed during deployment.

2. **Query Pre-Training**: This is the central contribution of the paper. The pre-training process simulates the actual deployment scenario, running in parallel across hundreds of scenes, with two types of iterations alternating:

   - **Mapping iteration**: Uses mapping images to jointly optimize both the map code and the Transformer weights—enabling the map code to learn scene encoding and the Transformer to learn how to read the map code.
   - **Query iteration**: Uses query images that differ from the mapping images (e.g., novel viewpoints, different illumination or object placement), but **only updates the Transformer weights** without modifying the map code—forcing the Transformer to learn to extract information from the map code to generalize to unseen query conditions.

   The key insight of this alternating training strategy is that during query iterations, the scene information is already "locked" into the map code, compelling the Transformer to leverage those fixed map codes to handle diverse query images, thereby acquiring generalization.

3. **Uncertainty-based correspondence filtering**: During relocalization, the network predicts not only 3D coordinates but also prediction uncertainty. This uncertainty is used to pre-filter 2D–3D correspondences, discarding low-confidence predictions before passing them to a robust PnP solver.

4. **DINOv2 feature backbone**: A pre-trained DINOv2 model serves as the image feature extractor, providing high-quality patch-level features as input to the Transformer.

### Loss & Training
- **Training loss**: The negative log-likelihood of reprojection error is used as the loss function. Predicted 3D coordinates are projected onto the image plane, and the error against the true 2D positions is computed.
- **Pre-training data**: Pre-training is conducted on tens of thousands of scenes using 3D supervision (ground-truth 3D coordinates provided by depth maps or SfM point clouds).
- **Mapping stage requires no 3D information**: Although pre-training uses 3D supervision, the mapping stage for a new scene does not require 3D ground truth—only pose-annotated images are needed.
- **Configuration options**: Two mapping configurations are provided: 5-minute (`ace_g_5min`) and 25-minute (`ace_g_25min`).
- **Training parameters**: Default `batch_size=40960`, `max_buffer_size=4000000`, `num_iterations=1000`.

## Key Experimental Results

| Dataset | Metric | Description |
|--------|------|------|
| Indoor-6 | Pose accuracy | Large-scale indoor scenes with significant illumination and viewpoint variation |
| 7Scenes | Median error | Classic indoor relocalization benchmark |
| 12Scenes | Median error | Indoor scenes |
| Cambridge Landmarks | Pose accuracy | Large-scale outdoor scenes with seasonal/weather/illumination variation |
| RIO-10 | Pose accuracy | Indoor scenes with object placement changes |

The paper demonstrates substantial generalization improvements of ACE-G over ACE and ACE-DINOv2 across multiple challenging relocalization benchmarks, while maintaining competitive computational overhead. Please refer to Table in the original paper for specific numerical results.

### Ablation Study
- **Contribution of query pre-training**: Query pre-training is the primary source of generalization gains. Removing the query iteration (retaining only the mapping iteration) leads to significant performance degradation.
- **Map code design**: The capacity and structure of the map code affect the quality of scene representation.
- **Pre-training scene scale**: Using more scenes for pre-training generally yields better generalization.
- **5min vs. 25min mapping**: Longer mapping time (25 minutes) produces higher-quality scene encoding.

## Highlights & Insights
- **Exceptionally clear formulation**: The paper elegantly decouples "scene memory" from "coordinate regression capability." Overfitting is reframed not as a bug but as a feature—since overfitting is unavoidable, the component that must overfit (map code) is separated from the component that must generalize (Transformer).
- **Sophisticated pre-training strategy**: The alternating mapping/query training simulates actual deployment, enabling the Transformer to learn to generalize during pre-training rather than merely learning to overfit.
- **Strong practicality**: The mapping stage requires no 3D ground truth, only pose-annotated images; code and pre-trained models are fully open-sourced; scene mapping is completed in as little as 5 minutes.
- **Transferable design philosophy**: The concept of externalizing scene-specific information into learnable encodings is broadly applicable to other vision tasks requiring scene or task adaptation.

## Limitations & Future Work
- **Pre-training code not released**: Although pre-trained models are provided, the pre-training code is not published, limiting community extension to new data or architectures.
- **Pre-training requires 3D supervision**: The pre-training stage requires large-scale scene data with 3D annotations, which entails substantial data acquisition costs.
- **Map code is static after mapping**: Once mapping is complete, the map code is fixed; if the scene undergoes physical changes (e.g., renovation, furniture rearrangement), re-mapping is required, with no incremental update capability.
- **Computational cost**: Although lighter than feature matching methods, the Transformer + map code combination is still heavier than the original ACE.
- **Large-scale outdoor scenes**: Performance in city-scale outdoor localization scenarios remains to be further validated.

## Related Work & Insights
- **vs. ACE / ACE-DINOv2**: ACE uses a lightweight MLP as a scene-specific regressor with all weights being scene-specific, precluding the use of cross-scene knowledge. ACE-G significantly improves generalization by separating the map code from the Transformer and introducing pre-training. ACE-DINOv2 adopts DINOv2 features but retains ACE's scene-specific training paradigm, yielding inferior generalization compared to ACE-G.
- **vs. feature matching methods (hloc, etc.)**: Classical feature matching methods are inherently generalizable via pre-trained feature extractors, but require explicit storage and retrieval of scene feature maps with large storage and computational overhead. ACE-G narrows the generalization gap with feature matching methods while preserving the compact scene representation of SCR approaches.
- **vs. NeRF/3DGS-based localization**: Neural rendering-based localization methods also require scene-specific training and similarly exhibit limited generalization. The decoupling strategy of ACE-G offers inspirational value for such approaches as well.

## Related Work & Insights
- **Universality of the "memory externalization" idea**: Separating task-specific knowledge from network weights into external learnable encodings bears a conceptual resemblance to parameter-efficient fine-tuning methods such as Prompt Tuning and LoRA. Similar ideas could be explored for other vision tasks requiring rapid adaptation.
- **Innovation in pre-training strategy**: The alternating mapping/query training paradigm that simulates real deployment conditions can be generalized as a universal "deployment-conditioned pre-training" framework.
- **Integration with 3D foundation models**: Replacing DINOv2 with a more powerful 3D foundation model as the feature backbone, or pre-training on larger-scale 3D data, holds the potential to further improve performance.

## Rating
- Novelty: ⭐⭐⭐⭐ — Both the architectural design of decoupling map code from the regressor and the query pre-training strategy represent important contributions to the SCR field.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on 5 datasets with ablation studies and visual analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, method motivation is excellently explained, and the project page is well crafted.
- Value: ⭐⭐⭐⭐ — Open-sourced code and pre-trained models make a concrete contribution to the visual localization community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](syncity_training-free_generation_of_3d_worlds.md)
- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[ICCV 2025\] Synchronization of Multiple Videos](synchronization_of_multiple_videos.md)

<!-- RELATED:END -->
