---
title: >-
  [Paper Note] NOVUM: Neural Object Volumes for Robust Object Classification
description: >-
  [ECCV2024][3D Vision][3D object representation] This paper proposes the NOVUM architecture, which maintains a neural volume representation composed of 3D Gaussians for each object category. By matching image features with the Gaussian features of each category, it achieves classification. NOVUM improves classification accuracy by 6-33% compared to standard architectures like ResNet/ViT/Swin under occlusion, corruption, and real-world OOD scenarios…
tags:
  - "ECCV2024"
  - "3D Vision"
  - "3D object representation"
  - "robust classification"
  - "3D Gaussians"
  - "contrastive learning"
  - "OOD generalization"
date: 2026-05-08
content_hash: ff4cad69f25a30e8
---

# NOVUM: Neural Object Volumes for Robust Object Classification

**Conference**: ECCV2024  
**arXiv**: [2305.14668](https://arxiv.org/abs/2305.14668)  
**Code**: [https://github.com/GenIntel/NOVUM](https://github.com/GenIntel/NOVUM)  
**Area**: 3D Vision  
**Keywords**: 3D object representation, robust classification, 3D Gaussians, contrastive learning, OOD generalization

## TL;DR
This paper proposes the NOVUM architecture, which maintains a neural volume representation composed of 3D Gaussians for each object category. By matching image features with the Gaussian features of each category, it achieves classification. NOVUM improves classification accuracy by 6-33% compared to standard architectures like ResNet/ViT/Swin under occlusion, corruption, and real-world OOD scenarios, while supporting 3D pose estimation and interpretable visualization.

## Background & Motivation

**Background**: Image classification models (ResNet, ViT, Swin) perform exceptionally well on IID data, but their generalization capability drops significantly when facing OOD data (occlusions, image corruptions, unseen scenarios/textures/weather).

**Limitations of Prior Work**: Standard discriminative models learn entangled 2D image-level representations that do not explicitly capture the 3D compositional structure of objects. When an object is partially occluded or undergoes appearance changes, the global representation collapses. Data augmentation can only partially alleviate this issue.

**Key Challenge**: Human vision is believed to rely on 3D compositional object representations (analysis-by-synthesis), allowing inference of the whole from few visible parts. Existing deep models lack this inductive bias.

**Goal**: Can the OOD robustness of classification classification models be enhanced by explicit 3D compositional object representations?

**Key Insight**: Inspired by cognitive science and 3D Gaussian splatting, a "neural object volume" is constructed for each category. It consists of spatially distributed 3D Gaussian kernels, with each Gaussian emitting a feature vector. Classification then becomes feature matching: finding which category's Gaussians best match the image features.

**Core Idea**: Embed 3D compositional object representations (Gaussian volumes) into neural network architectures, and leverage the independent matching of local Gaussians to achieve robust classification—even if some Gaussians fail to match (due to occlusion), the remaining Gaussians can still enable correct classification.

## Method

### Overall Architecture
NOVUM consists of two parts: (1) a shared feature extractor (ResNet50 + upsampling layers) that outputs $D=128$ dimensional feature maps; (2) a neural object volume for each category, consisting of $K \approx 1100$ 3D Gaussian kernels arranged on a cube's surface, where each Gaussian is associated with a $D$-dimensional feature vector. During inference: the backbone extracts feature maps $\to$ for each category, its Gaussian features are matched with the feature maps $\to$ the category with the highest matching score is selected as the prediction.

### Key Designs

1. **3D Gaussian Neural Object Volumes**:

    - **Function**: Defines around 1100 3D Gaussians for each category, uniformly distributed on a cube's surface, forming a category-level 3D representation.
    - **Mechanism**: The 3D positions and covariances of the Gaussians are fixed (without requiring precise geometry); only the feature vector $C_k \in \mathbb{R}^D$ of each Gaussian is learned. Approximating the object shape with a cube is sufficient to encode category-level spatial information.
    - **Design Motivation**: The volume representation based on Gaussian Splatting naturally supports differentiable rendering (for pose estimation) and efficient matching (for classification). Moreover, the compositional structure provides robustness to occlusion.

2. **Three-Level Contrastive Learning Loss**:

    - **Function**: Models the feature matching probability using a vMF distribution, and ensures feature discriminativeness via three levels of contrastive learning.
    - **Mechanism**: For each visible Gaussian $C_k$, its corresponding image feature $f_{k \to i}$ should have the highest similarity with $C_k$, while remaining far from (i) other non-neighboring Gaussians of the same category, (ii) Gaussians of other categories, and (iii) background features. The total loss is an InfoNCE-style contrastive loss.
    - **Design Motivation**: Single inter-class contrast is insufficient—different parts of the same object also need to have discriminative features, otherwise matching degrades to global features.

3. **Geometry-Free Fast Classification Inference**:

    - **Function**: Discards the 3D spatial structure during classification, performing only convolution-like matching between Gaussian features and image features.
    - **Mechanism**: For each pixel location $i$, the model computes $\max_{C_k \in \mathcal{C}_y} f_i \cdot C_k$ and the background matching score, takes the larger one, and sums them up to obtain the total score $S_y$ for category $y$. The predicted category is $\arg\max_y S_y$.
    - **Design Motivation**: Retaining the 3D structure is only necessary for pose estimation. Classification only requires local feature matching and can achieve a speed comparable to standard classifiers (~50 FPS).

### Loss & Training
- Training requires 3D pose annotations (used to establish the correspondence between Gaussians and image features).
- Gaussian features $C_k$ are updated via momentum ($\sigma=0.9$) rather than backpropagation.
- The background feature set $\mathcal{B}$ (2560 elements) is updated by resampling from the latest batch.
- 200 epochs, 4× RTX 3090, took approximately 20 hours.

## Key Experimental Results

### Main Results

**Classification Accuracy (%)**:

| Method | P3D+ (IID) | Occlusion L1 | Occlusion L2 | Occlusion L3 | OOD-CV Mean | Corruption Mean |
|------|-----------|---------|---------|---------|------------|---------|
| ResNet50 | 99.3 | 93.8 | 77.8 | 45.2 | 51.4 | 78.7 |
| ViT-b-16 | 99.3 | 94.7 | 80.3 | 49.4 | 59.0 | 87.6 |
| ConvNeXt | 99.4 | 95.3 | 81.3 | 50.9 | 56.0 | 85.6 |
| **NOVUM** | **99.5** | **97.2** | **88.3** | **59.2** | **85.2** | **91.3** |

Key Findings: Performance is almost on par under IID, but NOVUM outperforms the best baseline by **21%** on OOD-CV (85.2 vs 64.2) and by 8.3% under heavy occlusion (L3).

### Ablation Study

| Configuration | L0 | L1 | L2 | L3 |
|------|------|------|------|------|
| Single Gaussian (no compositional structure) | 93.2 | 90.3 | 80.4 | 44.0 |
| Spherical + no background features | 99.3 | 97.0 | 85.7 | 53.0 |
| Spherical + background features | 99.3 | 97.0 | 87.9 | 59.0 |
| Cube + background features (Full) | **99.5** | **97.2** | **88.3** | **59.2** |

### Key Findings
- **Compositional structure is key**: The single Gaussian baseline (no spatial composition) is 15.2% lower than the full model under L3 occlusion, proving that the independent matching of local Gaussians is the source of robustness.
- **Background features are crucial in occlusion scenarios**: Without background features, L3 performance drops from 59.2% to 53.0%, because occluded regions cannot be correctly assigned as background.
- **Pose estimation also benefits**: NOVUM achieves a pose $\pi/6$ accuracy of 88.2% on P3D+, outstanding NeMo (86.1%), with only 1/12 of NeMo's parameters.
- **High consistency between classification and pose**: Under IID, the proportion of cases where both tasks are correct simultaneously (89.8%) is only 0.3% lower than for classification alone.

## Highlights & Insights
- **3D Compositional Representation = Natural Robustness**: Rather than improving robustness by "seeing more OOD data" through data augmentation, it injects a 3D compositional inductive bias via architectural design. Independent matching per Gaussian makes partial occlusion almost harmless—something standard classifiers cannot achieve.
- **Extremely Simple and Efficient Inference for Classification**: 3D geometry is used during training to establish correspondences, but discarded completely during inference, performing only feature matching (convolutions). This achieves 50 FPS real-time inference. The "3D during training, 2D during inference" design is ingenious.
- **Out-of-the-box Interpretability**: Gaussian matching results can be directly visualized as object part correspondence maps (color-coded by spatial position), without requiring additional post-processing like CAM or GradCAM.

## Limitations & Future Work
- **Requires 3D pose annotations**: Training relies on the ground truth 3D pose of objects in each image, significantly limiting the available datasets (currently validated only on 12 categories of PASCAL3D+).
- **Fixed geometry**: All categories share the same cube shape, which is insufficiently precise for categories with large intra-class deformation (e.g., animals). Deformable Gaussian distributions could be explored.
- **Limited number of categories**: Validated only on 12 classes, and the viability of scaling to ImageNet dimensions (1000 classes $\times$ 1100 Gaussians = 1.1 million feature vectors) remains unknown.
- **No integration with pre-trained large models**: The backbone only uses ResNet50. Replacing it with self-supervised pre-trained features like DINOv2 could further enhance OOD robustness.

## Related Work & Insights
- **vs NeMo (Neural Mesh Model)**: NeMo uses mesh surface vertices for render-and-compare pose estimation but does not perform classification. NOVUM extends this to a shared backbone + inter-class contrastive design to simultaneously achieve classification and pose estimation, while reducing parameter count by 12$\times$.
- **vs Standard Classifiers + Data Augmentation**: Augmentation strategies like AugMix improve corruption robustness but have limited benefits for occlusion. NOVUM’s architecture-level robustness is consistently superior across all OOD types.
- **vs Compositional Nets**: Prior compositional models were mainly 2D. NOVUM introduces 3D compositional structures as its key innovation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce 3D Gaussian volume representation to image classification, with a unique idea and complete theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐ Validating OOD robustness on 4 datasets is substantial, but it is limited in scale (12 categories only and requires pose annotations).
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations and excellent visualizations, though notations are somewhat dense.
- Value: ⭐⭐⭐⭐ High concept-validation value—demonstrating that 3D compositional inductive bias can significantly improve robustness, though practical use is limited by pose annotation requirements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Category-Agnostic Neural Object Rigging](../../CVPR2025/3d_vision/category-agnostic_neural_object_rigging.md)
- [\[CVPR 2025\] End-to-End Implicit Neural Representations for Classification](../../CVPR2025/3d_vision/end-to-end_implicit_neural_representations_for_classification.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation](omni6d_large-vocabulary_3d_object_dataset_for_category-level_6d_object_pose_esti.md)
- [\[CVPR 2026\] NeuROK: Generative 4D Neural Object Kinematics](../../CVPR2026/3d_vision/neurok_generative_4d_neural_object_kinematics.md)

</div>

<!-- RELATED:END -->
