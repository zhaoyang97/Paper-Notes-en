---
title: >-
  [Paper Note] 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds
description: >-
  [CVPR 2026][3D Vision][3D self-supervised learning] This paper proposes LAM3C, a framework that, for the first time, demonstrates that video-generated point clouds (VGPCs) reconstructed from unlabeled online videos (e.g.…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D self-supervised learning"
  - "video-generated point clouds"
  - "Sinkhorn-Knopp clustering"
  - "noise regularization"
  - "indoor scene understanding"
date: 2026-05-08
content_hash: cbb8b7e1cdb50273
---

# 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds

**Conference**: CVPR 2026
**arXiv**: [2512.23042](https://arxiv.org/abs/2512.23042)  
**Code**: [https://github.com/ryosuke-yamada/lam3c](https://github.com/ryosuke-yamada/lam3c)  
**Area**: 3D Vision / Self-Supervised Learning
**Keywords**: 3D self-supervised learning, video-generated point clouds, Sinkhorn-Knopp clustering, noise regularization, indoor scene understanding

## TL;DR
This paper proposes LAM3C, a framework that, for the first time, demonstrates that video-generated point clouds (VGPCs) reconstructed from unlabeled online videos (e.g., property walkthroughs) can replace real 3D scans for 3D self-supervised pre-training. By introducing a Laplacian smoothing loss and a noise consistency loss to stabilize representation learning on noisy point clouds, and paired with the authors' RoomTours dataset (49K scenes), LAM3C matches or surpasses methods that rely on real 3D scans on indoor semantic and instance segmentation benchmarks.

## Background & Motivation

**Background**: 2D visual foundation models (e.g., DINOv2) have achieved remarkable success by leveraging massive unlabeled image collections (1.7B+). In contrast, 3D data acquisition remains constrained by the high equipment and labor costs of 3D scanning — the largest existing indoor scene dataset contains only ~5K unique scenes.

**Limitations of Prior Work**: Even state-of-the-art 3D-SSL methods such as Sonata, which mix real and synthetic data, operate at a training scale of only ~140K samples (with merely 18K real 3D scans). This limited data scale prevents 3D-SSL from achieving success comparable to its 2D counterpart. The data bottleneck is the fundamental constraint on progress in 3D self-supervised learning.

**Key Challenge**: 3D scene data is scarce and expensive to acquire, yet 3D-SSL requires large-scale data to succeed in the same manner as 2D-SSL.

**Key Observation**: Platforms such as YouTube host an abundance of indoor walkthrough videos (real-estate listings, apartment tours). Recent feed-forward 3D reconstruction models (e.g., π³ and VGGT) can infer 3D structure directly from multi-view images at quality comparable to traditional SfM/MVS pipelines.

**Core Idea**: (1) Reconstruct a large-scale VGPC dataset — RoomTours (49K scenes) — entirely from online videos, without using any real 3D scans; (2) Design a noise-regularized clustering pre-training framework, LAM3C, that makes representation learning on imperfect/noisy point clouds feasible and stable.

## Method

### Overall Architecture
YouTube video collection (3,462 videos + RealEstate10K) → CLIP zero-shot scene classification and segmentation (indoor/outdoor → living/bedroom/bathroom) → π³ feed-forward 3D reconstruction (200–400 frames/scene) → confidence masking + outlier filtering + post-processing → RoomTours dataset (49,219 VGPC scenes) → LAM3C teacher–student pre-training (Sinkhorn-Knopp clustering + Laplacian smoothing + noise consistency) → PTv3 backbone downstream fine-tuning / linear probing.

### Key Designs

1. **RoomTours Dataset Construction**:

    - Function: Build a large-scale 3D point cloud pre-training dataset from unlabeled online videos.
    - **Video Collection**: Keyword search across multiple cities ("city, real-estate, walk-through") → manual channel selection → automatic filtering (duration, metadata to exclude CG/drone/short clips) → 3,462 videos + RealEstate10K + YouTube House Tours + HouseTours.
    - **Scene Segmentation**: CLIP frame-level zero-shot classification (indoor/outdoor) → indoor frames classified by scene type (living/bedroom/bathroom) with boundary detection → segmentation with 0.5-second temporal consistency smoothing.
    - **VGPC Generation**: π³ feed-forward reconstruction with uniform sampling → single-precision mixed forward pass → confidence masking + edge suppression + outlier removal → colored 3D point clouds.
    - Output: 49,219 VGPC scenes averaging ~5 minutes per scene. Visually close to real scans but containing noise (blurring in camera-shake regions, potential overlapping surfaces on walls/floors) and missing regions.

2. **LAM3C Pre-training Framework**:

    - **Base Clustering Loss**: Teacher–student architecture with EMA-updated teacher. Three-component combination: $\mathcal{L}_{clustering} = w_u\mathcal{L}_{unmask} + w_m\mathcal{L}_{mask} + w_r\mathcal{L}_{roll}$. The unmask term aligns student local features to teacher global features (via kNN matching); the mask term distills teacher global representations to the student's masked global view; the roll-mask term swaps global views to enforce cross-view consistency. Weights are set to 4:2:2.
    - **Laplacian Smoothing Loss (Core Regularization 1)**: A kNN graph is constructed over the VGPC; for each edge, a distance-weighted coefficient $w_{ij} = \exp(-\|p_i-p_j\|^2/\sigma^2)$ is computed (with $\sigma$ adaptively set to the median kNN distance), encouraging spatially proximate points to produce similar embeddings: $R_{Lap} = \sum_{(i,j)\in E} w_{ij}\|z_i-z_j\|^2$. Distant neighbors are truncated for robustness, and in practice an L2 penalty is replaced with a Huber penalty. This loss smooths features along local geometry — noisy points are down-weighted due to their small coefficients.
    - **Noise Consistency Loss (Core Regularization 2)**: Two augmented views $x^{(a)}, x^{(b)}$ of the same VGPC are passed through the EMA teacher and student, respectively: $R_{cons} = \frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\|g_{EMA}(x^{(a)})_j - f_\theta(x^{(b)})_i\|^2$, where $\mathcal{P}$ is the set of kNN-matched point pairs. This ensures that the same point produces consistent embeddings across different noisy views.
    - **Total Objective**: $\mathcal{L}_{total} = \mathcal{L}_{clustering} + \lambda R_{Lap} + \mu R_{cons}$
    - **Scheduling**: $\lambda$ is linearly increased from 2e-4 to 3e-3 (progressively strengthening regularization); $\mu$ is fixed at 0.05.

3. **Design Motivation**:

    - Noise and missing regions in VGPCs destabilize standard clustering-based learning (point-level embeddings fluctuate severely).
    - Laplacian smoothing stabilizes **local** features (enforcing embedding consistency among spatially adjacent points).
    - Noise consistency stabilizes **global** representations (enforcing cross-view embedding consistency).
    - The two losses are complementary: the former exploits local geometric relationships, while the latter leverages global scene consistency.
    - Neither loss relies on hand-crafted indoor priors — both rely solely on inter-point relational structure.

### Loss & Training
PTv3 (Base/Large) backbone. Pre-training runs for up to 437K steps. Multi-level clustering is performed using masked global views and unmasked local views.

## Key Experimental Results

### Main Results (Indoor Semantic Segmentation mIoU, PTv3 Base, 100 epochs)

| Method | Pre-training Data | ScanNet LP | ScanNet FT | ScanNet200 FT | S3DIS FT |
|--------|------------------|:---:|:---:|:---:|:---:|
| PTv3 (no pre-training) | - | 16.1 | 74.7 | 32.0 | 67.8 |
| MSC | Real 7K | 21.8 | 78.2 | 33.4 | 69.9 |
| Sonata (real only, 15K) | Real 15K | 69.4 | 78.5 | 35.3 | 75.2 |
| Sonata (full) | Real 18K + Synthetic 121K | **72.5** | 79.4 | 36.8 | **76.0** |
| LAM3C (16K VGPC) | **Zero real** | 58.9 | 75.6 | 32.8 | 71.9 |
| LAM3C (49K VGPC) | **Zero real** | 66.0 | 77.7 | 35.1 | 72.9 |
| **LAM3C* (49K, Large)** | **Zero real** | 69.5 | **79.5** | 35.9 | 75.5 |

LAM3C* (PTv3 Large + 437K steps), without using any real 3D scans, achieves 79.5% on ScanNet FT, **matching Sonata (18K real + 121K synthetic) at 79.4%**.

### Instance Segmentation Results
On S3DIS instance segmentation, LAM3C surpasses Sonata trained exclusively on real scans.

### Ablation Study (ScanNet LP/FT, PTv3 Base)

| Configuration | ScanNet LP | ScanNet FT | Notes |
|--------------|:---:|:---:|-------|
| Clustering loss only | Unstable | Unstable | VGPC noise causes learning collapse |
| + Laplacian smoothing | Large gain | Gain | Local feature stabilization |
| + Noise consistency | Further gain | Gain | Global representation stabilization |
| 16K VGPC | 58.9 | 75.6 | Effect of data scale |
| **49K VGPC** | **66.0** | **77.7** | 3× more data → +7 mIoU on LP |

### Key Findings
- **Zero real scans suffice to match/surpass methods using real scans**: This is the central finding — VGPCs are a viable alternative data source for 3D-SSL and, given sufficient scale and model capacity, can match the state of the art.
- **Data scale is critical**: Scaling from 16K to 49K VGPCs yields a 7 mIoU gain on linear probing — 3D-SSL follows the same "more data is better" principle observed in 2D.
- **Both regularization terms are indispensable**: Clustering loss alone is unstable on VGPCs; Laplacian smoothing and noise consistency each contribute independently and are complementary.
- Under the 10% annotation fine-tuning setting on ScanNet, LAM3C already outperforms methods trained on real scans (validated in Figure 1, left).
- LAM3C is equally competitive on instance segmentation.

## Highlights & Insights
- **Paradigm shift: "3D without 3D scans"**: This work fundamentally redefines the data acquisition pathway for 3D pre-training. YouTube videos represent a near-unlimited source of 3D data — 49K scenes is merely a starting point; scaling to 100K+ is entirely feasible.
- **General-purpose noise regularization design**: The Laplacian smoothing loss (based on local geometric structure of point clouds) and the noise consistency loss (based on global cross-view consistency) do not rely on indoor-scene-specific priors and can generalize to self-supervised learning on any imperfect point clouds.
- **Novel application of feed-forward reconstruction models**: Models such as π³/VGGT were originally developed for reconstruction per se; LAM3C is the first to treat their outputs as pre-training data for 3D-SSL, substantially broadening the application scope of reconstruction models.
- **New perspective on the 2D–3D relationship**: The 3D geometric information latent in video is sufficient to support 3D representation learning, opening new directions for joint 2D–3D pre-training.

## Limitations & Future Work
- Noise and missing regions in VGPCs still impose an upper bound on performance — higher-quality feed-forward reconstruction models (e.g., successors to VGGT) may further improve point cloud fidelity.
- RoomTours covers only indoor scenes — VGPC quality for outdoor scenes may be lower due to larger scale, more dynamic objects, and greater illumination variation.
- Video collection relies on YouTube keyword search, which may introduce distributional bias toward specific geographic regions and property types.
- Larger-scale VGPC datasets (100K+) combined with longer training schedules may unlock additional performance gains.
- Temporal information in video (e.g., inter-frame temporal consistency) can be explored as an additional pre-training signal.

## Related Work & Insights
- **vs. Sonata**: Sonata relies on real + synthetic 3D scans (18K + 121K); LAM3C requires no 3D scans whatsoever, offering superior scalability. Performance matches Sonata when scale and model capacity are sufficient.
- **vs. PointContrast/MSC**: Earlier 3D-SSL methods are constrained to much smaller data scales (1K–7K real scans).
- **vs. PPT**: PPT uses synthetic data with supervision signals; LAM3C is purely self-supervised.
- **vs. π³/VGGT reconstruction models**: These serve as 3D reconstruction tools; LAM3C is the first to use their outputs as pre-training data for 3D-SSL.
- **Inspiration**: Multimodal (2D visual + 3D geometric) joint pre-training is a natural next step — 2D features from video frames and reconstructed 3D structure are naturally complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Paradigm-shifting "3D without 3D scans" concept with a complete and innovative pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, semantic + instance segmentation, linear probing + fine-tuning, data scale ablations, and regularization ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, complete pipeline description, and intuitive visualizations.
- Value: ⭐⭐⭐⭐⭐ Removes the 3D data bottleneck and exerts a paradigm-level influence on the 3D vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [\[CVPR 2026\] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance](gaussiangrow_geometry-aware_gaussian_growing_from_3d_point_clouds_with_text_guid.md)
- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp3d_joint_open_vocabulary_semantic_segmentation.md)
- [\[CVPR 2026\] S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds](s2am3d_scale-controllable_part_segmentation_of_3d_point_cloud.md)
- [\[CVPR 2026\] PointINS: Instance-Aware Self-Supervised Learning for Point Clouds](pointins_instance-aware_self-supervised_learning_for_point_clouds.md)

</div>

<!-- RELATED:END -->
