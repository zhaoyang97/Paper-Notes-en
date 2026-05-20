---
title: >-
  [Paper Note] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation
description: >-
  [ICCV 2025][Human Understanding][Novel Object Pose Estimation] This paper proposes MixRI, a lightweight network with only 12 reference images and 5.3M parameters…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Novel Object Pose Estimation"
  - "Feature Matching"
  - "Multi-View Fusion"
  - "Lightweight Network"
  - "Edge AI"
date: 2026-05-08
content_hash: a881a98642d940c0
---

# MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation

**Conference**: ICCV 2025
**arXiv**: [2601.06883](https://arxiv.org/abs/2601.06883)  
**Code**: [Project Page](https://npucvr.github.io/MixRI/)  
**Area**: Human Understanding
**Keywords**: Novel Object Pose Estimation, Feature Matching, Multi-View Fusion, Lightweight Network, Edge AI

## TL;DR

This paper proposes MixRI, a lightweight network with only 12 reference images and 5.3M parameters, which establishes 2D–3D correspondences between multiple reference images and a query image via a multi-view feature fusion strategy. MixRI achieves pose estimation performance comparable to methods requiring hundreds of reference images across 7 core BOP challenge datasets.

## Background & Motivation

Six degrees-of-freedom (6DoF) object pose estimation is fundamental to embodied AI applications such as robotic manipulation and augmented reality. Traditional methods require observing target objects during training and generating large amounts of synthetic data, making deployment cumbersome. **CAD-model-based novel object pose estimation** has emerged to address this: the specific objects to be inferred at test time are unknown during training, and pose is estimated using CAD-rendered reference images at test time.

Existing methods commonly suffer from three problems that hinder deployment on practical embedded AI systems:

**Excessive number of reference images**: FoundPose requires 798, MegaPose 520, and GigaPose 162. Rendering and storing large numbers of reference images per object severely limits the number of objects that can be handled simultaneously, and more references imply greater feature pre-extraction and caching demands.

**Large model sizes**: GigaPose has 316.3M parameters and FoundPose has 302.9M, making them unfriendly to memory-constrained edge devices.

**Slow inference**: Existing methods typically adopt a **two-stage pipeline**—first retrieving the nearest viewpoint from a large reference set (template matching), then performing fine-grained matching with the selected reference. The retrieval stage itself introduces additional latency.

More critically, when the number of reference images is drastically reduced, **large viewpoint gaps may exist between the nearest reference and the query image** (wide-baseline problem), causing severe performance degradation in traditional two-stage methods. The authors observe two key points: (1) multi-view geometry provides richer information than a single image; (2) occlusion is more frequent with sparse references, making multi-view integration even more necessary. Therefore, **the viewpoint retrieval stage should be bypassed, and information from all reference images should be utilized directly**.

## Method

### Overall Architecture

MixRI is a **purely feature-matching, single-stage** method. Given a query image and a small set of reference images (12 views), the network aims to: for each point $\mathbf{p}_k$ sampled on the 3D object surface, determine its projected coordinate $\mathbf{u}_{0,k}$ in the query image and its occlusion flag $O_{0,k}$. After establishing sufficient 2D–3D correspondences, RANSAC + PnP is applied to solve for the 6DoF pose. The pipeline is: shared encoder extracts features → dual-attention feature mixer fuses multi-view information → cost volume construction → two prediction heads output coordinates and occlusion flags, respectively.

### Key Designs

1. **Cross-Reference Correspondence Establishment**: Prior to training, the known reference image poses and depths are used to back-project 2D points sampled on each reference image into 3D object coordinates, which are then projected onto other reference images, yielding 2D projection coordinates $\{\mathbf{u}_{i,k}\}$ of each 3D point across all reference images along with their occlusion flags $O_{i,k}$. Occlusion is determined similarly to Z-buffering: if the projected depth deviates from the actual depth by more than a threshold $\tau$, the point is marked as occluded.

$$\mathbf{p}_k = d_{i,k} \mathbf{T}_i^{-1} \mathbf{K}_i^{-1} \tilde{\mathbf{u}}_{i,k}$$

This provides a precise spatial alignment basis for feature fusion.

2. **Dual-Attention Based Feature Mixer**: This is the core innovation of MixRI. It consists of three modules applied in alternating iterations:

    - **SAP (Self-Attention across Points)**: Applies attention among the $N$ sampled points within the same reference image to integrate spatial location information.
    - **SAF (Self-Attention across Frames)**: Applies attention among $S$ reference frames and interacts with a learnable fusion token $\bar{\mathbf{F}}$ to aggregate cross-view information.
    - **MARQ (Mixed Attention for Reference and Query)**: Applies cross-attention between the fused reference tokens and query image features to enhance query features.

   The overall formulation is:
    $\hat{\mathbf{F}} = \text{Self}_N(g_N(\hat{\mathbf{F}}), \mathbf{O}), \quad \hat{\mathbf{F}}, \bar{\mathbf{F}} = \text{Self}_S(g_S(\hat{\mathbf{F}}), \bar{\mathbf{F}}, \mathbf{O}), \quad \bar{\mathbf{F}}, \mathbf{F}_0 = \text{Mix}(\bar{\mathbf{F}}, \mathbf{F}_0, \mathbf{O})$

   The occlusion mask $\mathbf{O}$ is used for masked attention to prevent erroneous features from occluded points from being fused. The process is iterated $n_0$ times to obtain the final fused result.

3. **Occlusion-Aware Cost Volume and Dual-Head Prediction**: The fused reference token $\bar{\mathbf{F}}$ and query image features $\tilde{\mathbf{F}}_0$ are used to construct a cost volume $\mathbf{C} \in \mathbb{R}^{H' \times W' \times N}$. After processing by a Conv3D backbone, two independent prediction heads output: (a) 2D heatmaps (converted to coordinates via spatial soft argmax); and (b) occlusion probabilities. Only points predicted as visible are used in PnP solving.

   The encoder uses a ResNet-like backbone with shared weights and incorporates a rotation-invariant feature module to make the extracted features invariant to object pose.

### Loss & Training

- **Occlusion loss**: $L_{occ} = \text{BCE}(O_{gt,k}, O_{0,k})$
- **Localization loss**: $L_{loc} = \text{Huber}(\mathbf{U}_{gt,k}, \mathbf{U}_{0,k}) \cdot \mathbf{1}\{O_{gt,k}=0\}$ (computed only for visible points)
- **Total loss**: $L = L_{occ} + 100 \cdot L_{loc}$
- Training uses only synthetic images from the GSO-Dataset; testing is performed on real-scene BOP datasets.
- CNOS is used as a general object detector (replaceable with other lightweight alternatives).

## Key Experimental Results

### Main Results

**BOP Challenge 7 Core Datasets (Average Recall ↑)**

| Method | Params | # Ref. | LM-O | IC-BIN | HB | YCB-V | MEAN | Inference Time |
|--------|--------|--------|------|--------|----|-------|------|----------------|
| MegaPose | 21.6M | 520 | 22.9 | 15.2 | 25.1 | 28.1 | 20.8 | 15.5s |
| GigaPose | 316.3M | 162 | 29.9 | 23.1 | 34.8 | 29.0 | 27.6 | 0.8s |
| FoundPose | 302.9M | 798 | 39.6 | 23.9 | 50.8 | 45.2 | 37.2 | 1.6s |
| **MixRI (12)** | **5.3M** | **12** | 27.0 | **29.7** | 44.9 | **52.8** | 31.4 | **0.5s** |
| **MixRI (24)** | **5.3M** | **24** | 30.4 | **30.8** | **50.2** | **54.6** | 34.1 | 0.7s |

MixRI achieves near-SOTA performance using 33× fewer reference images, 57× fewer parameters, and 3× faster inference speed.

### Ablation Study

| Configuration | YCB-V | LM-O | TUD-L | MEAN | Note |
|---------------|-------|------|-------|------|------|
| w/o SAP + w/o MARQ | 2.9 | 2.8 | 2.2 | 2.6 | Baseline matching |
| w/ SAP | 19.2 | 7.8 | 8.7 | 11.9 | +9.3% AR |
| w/ MARQ | 33.2 | 13.4 | 21.9 | 22.8 | +20.2% AR |
| **w/ SAP + MARQ** | **54.6** | **30.4** | **33.6** | **39.5** | **+36.9% AR** |

**Comparison Under Limited Reference Images (MixRI vs. GigaPose)**

| # Ref. | GigaPose MEAN | MixRI MEAN |
|--------|---------------|------------|
| 4 | 9.1 | 17.9 |
| 6 | 11.0 | 31.4 |
| 12 | 14.7 | 36.4 |
| 24 | 18.3 | 39.5 |

### Key Findings

- GigaPose performance degrades sharply as the number of reference images decreases (an inherent weakness of two-stage methods), whereas MixRI is naturally well-suited to the sparse-reference setting.
- MARQ (Mixed Attention for Reference and Query) contributes the most (+20.2%), demonstrating that cross-attention between query and reference images is critical for matching.
- SAP and MARQ exhibit strong synergy: individually contributing +9.3% and +20.2%, their combination reaches +36.9%.
- Only 60 correspondences suffice to achieve 45% AR on YCB-V, demonstrating the robustness of the method.
- Performance is poor on the ITODD dataset (grayscale + strong reflection + weak texture), exposing the limitation of RGB-trained models in grayscale scenarios.

## Highlights & Insights

- Successful validation of a **"less is more" philosophy**—smarter information fusion matters more than simply using more reference images.
- The paradigm shift from two-stage (retrieval + matching) to single-stage (direct all-reference fusion matching) is the key innovation.
- With only 5.3M parameters (60× smaller than GigaPose), MixRI is highly suitable for deployment on edge devices.
- Occlusion detection is a natural output of the network rather than a post-processing step, simplifying the pipeline.
- The approach unifies ideas from point tracking and feature matching—the projections of the same 3D point across multiple reference images are analogous to tracking trajectories.

## Limitations & Future Work

- Poor performance on grayscale images (ITODD); generalization across color modalities needs to be addressed.
- No dedicated design for symmetric objects, which may limit performance on highly symmetric industrial parts.
- The uniform sampling strategy for 12 reference images may be suboptimal—adaptive viewpoint selection strategies are worth exploring.
- The method has not been combined with refinement approaches (e.g., MegaPose's coarse-to-fine), which could further improve accuracy.
- Although performance is strong under heavy occlusion (IC-BIN), weak texture in ITODD remains a challenge.

## Related Work & Insights

- **GigaPose** and **FoundPose** are the most direct comparisons, representing the pinnacle of the retrieval-then-matching paradigm.
- Detector-free matching methods such as **LoFTR** inspired the end-to-end matching design.
- Occlusion modeling in point tracking methods such as **TAP-Net** inspired the multi-view correspondence design.
- Insight: The lightweight multi-view fusion strategy is generalizable to downstream tasks such as grasp pose estimation and object placement in AR.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The paradigm shift to single-stage multi-view fusion matching is a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 7 BOP challenge datasets with detailed reference-count comparisons and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, method description is rigorous, and bubble chart comparisons are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical value for edge AI deployment, with significant advantages in parameter count and inference speed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[AAAI 2026\] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](../../AAAI2026/human_understanding/coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)

</div>

<!-- RELATED:END -->
