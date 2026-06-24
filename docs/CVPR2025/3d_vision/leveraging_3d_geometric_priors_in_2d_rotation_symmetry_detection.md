---
title: >-
  [Paper Note] Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection
description: >-
  [CVPR 2025][3D Vision][Rotation Symmetry Detection] This paper proposes a rotation symmetry detection model leveraging 3D geometric priors. By directly predicting the rotation center and vertices in 3D space and projecting them back to 2D, combined with a seed-point and rotation-axis based vertex reconstruction module, the method achieves an F1-score of 33.2 on the DENDI dataset, outperforming the previous segmentation-based SOTA method EquiSym (22.5).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Rotation Symmetry Detection"
  - "3D Geometric Priors"
  - "Vertex Reconstruction"
  - "DETR"
  - "Set Prediction"
date: 2026-05-08
content_hash: 6e9deb71e2e7210a
---

# Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.20235](https://arxiv.org/abs/2503.20235)  
**Code**: [http://cvlab.postech.ac.kr/research/RotSymDETR](http://cvlab.postech.ac.kr/research/RotSymDETR)  
**Area**: Image Segmentation/Symmetry Detection  
**Keywords**: Rotation Symmetry Detection, 3D Geometric Priors, Vertex Reconstruction, DETR, Set Prediction

## TL;DR

This paper proposes a rotation symmetry detection model leveraging 3D geometric priors. By directly predicting the rotation center and vertices in 3D space and projecting them back to 2D, combined with a seed-point and rotation-axis based vertex reconstruction module, the method achieves an F1-score of 33.2 on the DENDI dataset, outperforming the previous segmentation-based SOTA method EquiSym (22.5).

## Background & Motivation

**Background**: Symmetry detection is an important visual cue for understanding object structures. Rotation symmetry refers to an object remaining invariant after rotating around a central axis. Traditional methods rely on handcrafted feature matching (such as SIFT and frequency analysis). Recently, CNN-based methods (such as EquiSym) detect rotation centers using segmentation heatmaps, but focus only on center detection while ignoring the prediction of supporting vertices and symmetry groups.

**Limitations of Prior Work**: Existing 2D detection methods cannot enforce the inherent geometric constraints of rotation symmetry (equal edge lengths, equal internal angles) because ground truth annotations are provided from a 3D perspective, and 2D annotations lose geometric consistency due to perspective variations. Segmentation-based methods output heatmaps, which require post-processing to analyze individual symmetries and cannot directly solve for vertex coordinates.

**Key Challenge**: There is an inherent discrepancy between the 2D annotation space and the 3D annotation semantics—annotators perceive symmetry from a 3D perspective, but 2D projection distorts the geometric properties of regular polygons. Directly predicting vertices in 2D space fails to exploit strong geometric constraints like "equal edges and equal angles."

**Goal**: (1) How to simultaneously predict the rotation center, symmetry group, and supporting vertices within a detection framework; (2) How to leverage 3D geometric priors to ensure structural consistency of predictions.

**Key Insight**: The authors observe that predicting the rotation center and vertices in 3D space allows the natural enforcement of regular polygon geometric constraints (equal edge lengths, equal internal angles, coplanarity), which, when projected back to 2D, preserves structural integrity while adapting to perspective changes.

**Core Idea**: By reconstructing all vertices from a seed point and a rotation axis in 3D space (instead of predicting them individually), 3D geometric priors are embedded into the detection pipeline, which is then projected back to 2D to complete rotation symmetry detection.

## Method

### Overall Architecture

An input image is processed through a Swin-T backbone to extract multi-scale features. The introduced camera queries (grid-like learnable parameters) interact with backbone features via Camera Cross Attention (CCA) to encode 2D image features into 3D camera coordinates. After processing by the Transformer encoder, they are fed into the detection head to predict the 3D rotation center, seed vertex, rotation axis vector, and symmetry group classification. The vertex reconstruction module rotates the seed vertex around the rotation axis according to the predicted symmetry group to generate all vertices. Finally, perspective projection maps the 3D coordinates back to the 2D image plane.

### Key Designs

1. **Camera Cross Attention (CCA)**:

    - Function: Maps 2D image features into 3D camera coordinate space.
    - Mechanism: Samples $N_{\text{ref}}$ depth values at each x-y position to generate 3D reference points, projecting these 3D points onto the 2D plane to sample backbone features via deformable attention. Mathematically, for each query position $\mathbf{p}_q$, multiple depth sampling points $z_i$ are generated along the z-axis, projected to image coordinates, and features are aggregated using deformable attention.
    - Design Motivation: Draws inspiration from the spatial cross-attention concept in BEVFormer to efficiently encode 3D spatial location information from 2D features, providing space-aware features for subsequent 3D coordinate prediction.

2. **Vertex Reconstruction**:

    - Function: Generates all symmetric vertices from a seed point and a rotation axis, enforcing geometric constraints.
    - Mechanism: Predicts a seed point $\mathbf{s}$, rotation center $\mathbf{c}$, and rotation axis $\mathbf{a}$. Rodrigues' rotation formula is used to rotate the seed point around the axis by $\theta_k = 2\pi k/N$ to generate the $k$-th vertex $\mathbf{v}_k$. For the $C_2$ group (rectangle), an additional angle offset $\beta$ is predicted to generate four vertices.
    - Design Motivation: Using parameterized generation instead of point-by-point regression naturally guarantees that all vertices are equidistant from the rotation center with equal edges and angles, eliminating geometric inconsistency.

3. **Two-step Bipartite Matching Training Strategy (RCM + RVM)**:

    - Function: Formulates set prediction as a two-step matching problem.
    - Mechanism: The first step, Rotation Center Matching (RCM), uses the Hungarian algorithm to match predicted and ground truth rotation centers and symmetry group classifications. The second step, Rotation Vertex Matching (RVM), performs bipartite matching of vertex sets within the matched center pairs. The overall loss combines classification cross-entropy and L1 coordinate regression.
    - Design Motivation: Rotational symmetric vertex sets themselves exhibit rotational equivalence (e.g., an equilateral triangle rotated by 120° is the same), requiring set matching to eliminate permutation ambiguity.

### Loss & Training

The total loss is $\mathcal{L}_{\text{total}} = \sum_i \mathcal{L}_{\text{center}} + \mathcal{L}_{\text{vertex}}$. The center loss includes classification cross-entropy and L1 regression; the vertex loss is the L1 distance after matching. Training is conducted for 200 epochs using the AdamW optimizer with a learning rate of 0.0002, with the backbone learning rate set 10 times lower. The model uses 800 object queries, and the regression cost weight is set to 10.

## Key Experimental Results

### Main Results

| Method | Prediction Type | Max F1-score |
|------|---------|-------------|
| EquiSym | Segmentation | 22.5 |
| **Ours** | **Detection** | **33.2** |

**Rotation vertex detection mAP:**

| Method | mAP |
|------|-----|
| 2D baseline | 24.7 |
| 3D baseline | 23.5 |
| **Ours (3D + vertex recon.)** | **30.6** |

### Ablation Study

| Configuration | 3D query/pred. | vertex recon. | mAP |
|------|:-:|:-:|-----|
| 2D baseline | ✗ | ✗ | 24.7 |
| 3D baseline | ✓ | ✗ | 23.5 |
| Full model | ✓ | ✓ | **30.6** |

### Key Findings

- Simply introducing 3D predictions (3D baseline) result in a slightly lower mAP (by 1.2) compared to the 2D baseline, suggesting that 3D prediction without geometric constraints introduces projection noise and alignment issues.
- Integrating vertex reconstruction increases the mAP from 23.5 to 30.6 (+7.1), proving that explicitly modeling 3D geometric constraints is key to performance improvement.
- Performance is highest on the C8 group (AP 46.7), whereas C3 and C6 groups have lower AP due to a smaller number of test samples.
- 3D visualization demonstrates that the model correctly captures the depth arrangement and spacing of rectangular structures.

## Highlights & Insights

- **The concept of "reconstructing instead of predicting vertices" is highly clever.** By utilizing parameterization (seed point + rotation axis + symmetry group) instead of point-by-point regression, geometric constraints are hard-coded into the network architecture, preventing predictions that violate physical laws. This "structured output" design can be transferred to any detection task requiring geometric constraints.
- **Scaling up from 2D detection to 3D and then projecting back.** While seemingly increasing problem complexity, it actually enables the model to leverage stronger constraints (equal edges and angles in 3D space), presenting an elegant workaround to the difficulty of enforcing constraints in 2D.
- The camera queries design draws inspiration from BEV perception, importing 3D perception technologies from autonomous driving into symmetry detection, showcasing a great example of cross-domain technology transfer.

## Limitations & Future Work

- The DENDI dataset is small (1459 training images), with highly imbalanced categories, and very few samples for groups like C3/C6 resulting in low AP.
- It assumes known or fixed camera intrinsic parameters (focal length set to 1000), which may not apply to uncalibrated scenes; the authors suggest predicting camera intrinsics as a future direction.
- Evaluation is limited to a single dataset, lacking cross-dataset generalization validation.
- Failures in estimating the rotation axis still occur under extreme perspective changes.
- Future work could extend this approach to reflection symmetry detection or integrate it with instance segmentation for a more complete understanding of symmetry.

## Related Work & Insights

- **vs EquiSym**: EquiSym is dynamic-CNN-based and outputs segmentation heatmaps to detect rotation centers, failing to predict vertices and symmetry groups, with heatmaps requiring post-processing. This work utilizes a DETR detection framework to directly predict structured outputs, enabling individual symmetry analysis.
- **vs Traditional Methods (SIFT-based)**: Traditional methods rely on handcrafted feature matching to find periodic signals, showing poor robustness in real-world scenarios. This model learns end-to-end and enhances robustness to perspective variations through 3D priors.
- **vs DETR3D/BEVFormer**: This work borrows the concepts of camera-centric queries and spatial cross-attention from 3D object detection, showing that the BEV perception technology stack has the potential to transfer to more geometry-aware tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce 3D geometric priors to 2D rotation symmetry detection; the vertex reconstruction idea is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐ Validated on only one relatively small dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear method descriptions and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Insightful for the symmetry detection field, though the application domain is somewhat specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](symmetry_strikes_back_from_single-image_symmetry_detection_to_3d_generation.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)
- [\[ICCV 2025\] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction](../../ICCV2025/3d_vision/geo4d_leveraging_video_generators_for_geometric_4d_scene_reconstruction.md)
- [\[CVPR 2025\] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)
- [\[ICCV 2025\] SAS: Segment Any 3D Scene with Integrated 2D Priors](../../ICCV2025/3d_vision/sas_segment_any_3d_scene_with_integrated_2d_priors.md)

</div>

<!-- RELATED:END -->
