---
title: >-
  [Paper Note] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation
description: >-
  [CVPR 2025][3D Vision][Symmetry Detection] Reflect3D proposes a scalable zero-shot 3D reflection symmetry detector that resolves single-view ambiguity through a Transformer architecture and multi-angle aggregation from multi-view diffusion models. Integrating the detected symmetry into a single-image 3D generation pipeline significantly improves structural accuracy and texture quality.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Symmetry Detection"
  - "Single-Image 3D Generation"
  - "Zero-Shot Generalization"
  - "Multi-View Diffusion"
  - "DINOv2"
date: 2026-05-08
content_hash: 4ac86dc77df3a6f5
---

# Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.17763](https://arxiv.org/abs/2411.17763)  
**Code**: [https://ryanxli.github.io/reflect3d](https://ryanxli.github.io/reflect3d)  
**Area**: 3D Vision  
**Keywords**: Symmetry Detection, Single-Image 3D Generation, Zero-Shot Generalization, Multi-View Diffusion, DINOv2

## TL;DR

Reflect3D proposes a scalable zero-shot 3D reflection symmetry detector that resolves single-view ambiguity through a Transformer architecture and multi-angle aggregation from multi-view diffusion models. Integrating the detected symmetry into a single-image 3D generation pipeline significantly improves structural accuracy and texture quality.

## Background & Motivation

**Background**: Symmetry is a ubiquitous and fundamental property in the visual world, long used as a structural constraint for pose estimation, grasp detection, and 3D reconstruction. Existing symmetry detection methods mainly operate on 3D or depth data, and detecting 3D reflection symmetry from a single RGB image remains an under-explored challenge.

**Limitations of Prior Work**: Prior methods (e.g., NeRD, NeRD++) rely on 3D cost volume construction, require known camera intrinsics, and are trained and evaluated on in-domain categories, leading to significant performance degradation when generalizing to in-the-wild scenes. They are restricted to a few object categories and fail to achieve true zero-shot symmetry detection. In 3D generation, SDS optimization-based methods (e.g., DreamGaussian) often generate 3D objects with missing geometry and blurry textures on the back, whereas symmetric objects can leverage frontal information to infer the back.

**Key Challenge**: Single-view symmetry detection faces fundamental perspective ambiguity—occlusion, perspective distortion, and unknown depth all obscure symmetry cues. Meanwhile, detection capability and generalizability usually conflict—using more explicit 3D priors may improve in-domain accuracy but limit generalization.

**Goal**: (1) Train a zero-shot single-image symmetry detector that generalizes to arbitrary objects; (2) Integrate the detected symmetry prior into single-image 3D generation to improve generation quality.

**Key Insight**: Inspired by the success of foundation models—large-scale data + a generic Transformer architecture + frozen DINOv2 geometry-aware features. Multi-view diffusion models are utilized to generate multi-angle views to resolve single-view ambiguity.

**Core Idea**: Minimize explicit 3D priors and train a Transformer symmetry detector on large-scale diverse data to achieve generalization; generate surrounding views using a multi-view diffusion model and aggregate multi-view symmetry predictions to resolve ambiguity; inject symmetry as a prior into the SDS optimization process of DreamGaussian.

## Method

### Overall Architecture

Reflect3D consists of two main components. First is symmetry detection: given a single input RGB image, geometry-aware features are extracted via a frozen DINOv2, a Transformer decoder queries multiple symmetry hypotheses using cross-attention, and MLP heads perform binary classification and normal vector regression. Optionally, a multi-view diffusion model is utilized to generate $M=8$ surrounding views, from which symmetry is individually detected and aggregated via K-Means clustering. Second is symmetry-aware 3D generation: building upon DreamGaussian, three steps of symmetry alignment, symmetry SDS optimization, and symmetric texture refinement are introduced.

### Key Designs

1. **Feed-Forward Symmetry Detector**:

    - **Function**: Predict the 3D reflection symmetry plane from a single RGB image.
    - **Mechanism**: Discretize the potential space of symmetry plane normal vectors into $N=31$ unit vectors uniformly covering the hemisphere as symmetry hypotheses. A shallow MLP maps these hypotheses into high-dimensional query features, which perform cross-attention and self-attention with frozen DINOv2 features to yield $N$ feature vectors. For each feature, MLP heads perform binary classification (whether a symmetry plane exists within the hypothesis neighborhood) and quaternion regression (precise normal vector). Training is supervised using BCE loss for classification and MSE loss for quaternion regression.
    - **Design Motivation**: Frozen DINOv2 provides powerful geometry-aware features while maintaining generalizability; fine-tuning actually degrades performance significantly ($F@5^\circ$ drops from 0.191 to 0.038). 31 hypotheses are sufficient to cover all possible normal vector directions. The two-stage (coarse classification + fine regression) strategy balances accuracy and coverage.

2. **Multi-view Symmetry Enhancement**:

    - **Function**: Leverage synthetic multi-views to resolve single-view ambiguity.
    - **Mechanism**: Use a multi-view diffusion model to generate $M=8$ surrounding views for the input image, filtering inconsistent generations via CLIP similarity. Apply the feed-forward detector to each view, rotate all predictions back to the input view coordinate system, and aggregate them using K-Means clustering, taking the cluster centers as the final predicted symmetry normal vectors.
    - **Design Motivation**: The uncertainty of the back of the object causes inherent ambiguity in single-view regression training. Multi-views provide more complete observation angles, and clustering eliminates redundant predictions while merging predictions from different viewpoints pointing to the same symmetry plane. 8 views are sufficient; more views lead to performance saturation.

3. **Symmetry-Aware 3D Generation**:

    - **Function**: Integrate the detected symmetry prior into SDS optimization to improve 3D generation quality.
    - **Mechanism**: A three-step pipeline—(a) Symmetry Alignment: Optimize DreamGaussian without MSE loss for a few steps to obtain a coarse Gaussian representation, extract the point cloud, and align the symmetry plane to the point cloud using ICP; (b) Symmetric SDS Optimization: during camera viewpoint sampling, compute SDS loss not only for the sampled viewpoint but also for its symmetric viewpoint, and every 100 steps, reflect the Gaussians along the symmetry plane and randomly sample 50% to supplement the original set; (c) Symmetric Texture Refinement: the visible region of the input view is directly refined with MSE loss, the visible region of the symmetric view is refined with the MSE of the flipped image, and other regions use standard texture refinement loss.
    - **Design Motivation**: The back of the object generated by DreamGaussian often suffers from missing geometry and blurry textures. The symmetry prior transfers high-quality frontal information to the back. Randomly sampling 50% (instead of all) reflected Gaussians allows for natural, slight asymmetries.

### Loss & Training

- Symmetry detector: Adam optimizer, learning rate 3e-5, batch size 120, trained for 15 epochs.
- Training data: Objaverse LVIS subset + ShapeNet, totaling 84,789 objects across 1,154 categories, approximately 1.1 million images, and 152,019 annotated symmetry planes.
- Symmetry ground truth is automatically generated via optimization: uniformly sampling candidate planes $\rightarrow$ Chamfer distance validation after reflection $\rightarrow$ ICP refinement.
- Zero-shot evaluation on GSO (572 objects) and OmniObject3D (100 objects).

## Key Experimental Results

### Main Results (Symmetry Detection)

| Method | GSO $F@5^\circ \uparrow$ | GSO $F@15^\circ \uparrow$ | GSO GD $\downarrow$ | OmniObj $F@5^\circ \uparrow$ | OmniObj GD $\downarrow$ |
|------|-----------|------------|---------|---------------|-------------|
| NeRD | 0.040 | 0.398 | 36.2 | 0.055 | 41.3|
| Reflect3D-FF | 0.191 | 0.452 | 22.7 | 0.103 | 31.1 |
| **Reflect3D** | **0.390** | **0.756** | **13.3** | **0.173** | **22.8** |

### Main Results (3D Generation)

| Method | GSO CLIP-Sim $\uparrow$ | GSO CD $\downarrow$ | GSO $F@0.5 \uparrow$ | OmniObj CLIP-Sim $\uparrow$ |
|------|---------------|---------|------------|-------------------|
| DreamGaussian | 0.592 | 0.442 | 0.767 | 0.704 |
| **+ Symmetry Prior** | **0.629** | **0.414** | **0.827** | **0.734** |

### Ablation Study

| Configuration | GSO $F@5^\circ \uparrow$ | GSO GD $\downarrow$ |
|------|-----------|---------|
| Reflect3D Full | 0.390 | 13.3 |
| w/o Clustering | 0.312 | 16.0 |
| Reflect3D-FF | 0.191 | 22.7 |
| DINOv2 $\rightarrow$ ViT | 0.094 | 24.7 |
| Frozen $\rightarrow$ Fine-tuned DINOv2 | 0.038 | 34.2 |

### Key Findings

- Even without multi-views, Reflect3D-FF with a single image already achieves SOTA ($F@5^\circ$ of 0.191 vs. NeRD’s 0.040), proving the effectiveness of the large-scale data + Transformer paradigm.
- Multi-view aggregation further improves $F@5^\circ$ from 0.191 to 0.390 (approx. two-fold), reducing the mean geodesic distance by $9.4^\circ$.
- Freezing DINOv2 is crucial—fine-tuning plummets the performance from 0.191 to 0.038, as fine-tuning disrupts the pre-trained geometry-aware capabilities.
- The symmetry prior significantly improves 3D generation across both 2D (CLIP-Sim) and 3D (CD, F-score) metrics.
- Symmetry helps avoid geometric errors (such as glasses temples incorrectly connected to the frame) and completes back-view details.

## Highlights & Insights

1. The foundation model concept of "minimizing 3D priors + training on large-scale data" is proven to be more effective in symmetry detection, contrasting with traditional approaches in many 3D vision tasks that rely on explicit 3D priors.
2. The finding that fine-tuning a frozen DINOv2 actually degrades performance is highly valuable—pre-trained geometry-aware features from DINOv2 appear to be key for generalizing zero-shot symmetry detection.
3. Although views generated by multi-view diffusion are imperfect (requiring CLIP filtering), they are sufficient to significantly resolve single-view ambiguity.
4. The utility of symmetry as a 3D prior is clearly demonstrated in modern SDS generation frameworks, especially for improving the quality of the back views.

## Limitations & Future Work

- Unable to handle completely asymmetric or highly deformable objects.
- Only detects the normal vector $n_p$ with no direct prediction of the distance $d_p$, requiring other cues (e.g., 3D representations) to determine the plane position.
- For practical applications, it is necessary to first determine whether the object exhibits symmetry before deciding whether to apply the symmetry prior.
- Future work could explore the detection of partial and rotational symmetries.

## Related Work & Insights

- Compared to NeRD's 3D cost volume approach, Reflect3D is fully based on 2D features and exhibits stronger zero-shot generalization.
- The improvement on DreamGaussian by the symmetry prior indicates that geometric constraints still holds significant value in SDS optimization.
- The idea of multi-view aggregation can be generalized to other tasks that require resolving single-view ambiguity (e.g., normal estimation, depth estimation).

## Rating

- **Novelty**: 7/10 — Applying the foundation model concept to symmetry detection is novel, although the technical components (DINOv2 + Transformer + multi-view diffusion) are combinations of existing tools.
- **Experimental Thoroughness**: 8/10 — Zero-shot evaluation on two real scanned datasets + detailed ablation studies + validation in 3D generation applications.
- **Writing Quality**: 8/10 — Clear problem definition, with a solid narrative arc from symmetry detection to 3D generation.
- **Value**: 7/10 — Although symmetry detection itself has a relatively narrow scope of application, the idea of using it as a prior for 3D generation is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)
- [\[ICML 2025\] Symmetry-Robust 3D Orientation Estimation](../../ICML2025/3d_vision/symmetry-robust_3d_orientation_estimation.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[ICCV 2025\] χ: Symmetry Understanding of 3D Shapes via Chirality Disentanglement](../../ICCV2025/3d_vision/kh_symmetry_understanding_of_3d_shapes_via_chirality_disentanglement.md)
- [\[CVPR 2025\] StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)

</div>

<!-- RELATED:END -->
