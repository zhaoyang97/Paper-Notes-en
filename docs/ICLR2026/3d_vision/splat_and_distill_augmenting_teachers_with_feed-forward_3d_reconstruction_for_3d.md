---
title: >-
  [Paper Note] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation
description: >-
  [ICLR 2026][3D Vision][3D-Aware Distillation] Within a student-teacher distillation framework, this work augments the teacher with a pretrained feed-forward 3D reconstruction model (MVSplat) that lifts 2D features into a 3D Gaussian representation and renders them to novel viewpoints, enabling the student to learn geometrically consistent, 3D-aware 2D features. The proposed method surpasses existing approaches across downstream tasks including depth estimation, surface normal estimation, semantic segmentation, and multi-view correspondence.
tags:
  - ICLR 2026
  - 3D Vision
  - 3D-Aware Distillation
  - 3D Gaussian Splatting
  - Feed-Forward Reconstruction
  - Vision Foundation Models
  - Student-Teacher
date: 2026-05-08
content_hash: f015b8e12dd13626
---

# Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation

**Conference**: ICLR 2026
**arXiv**: [2602.06032](https://arxiv.org/abs/2602.06032)
**Code**: Available (GitHub)
**Area**: 3D Vision / Vision Foundation Models
**Keywords**: 3D-Aware Distillation, 3D Gaussian Splatting, Feed-Forward Reconstruction, Vision Foundation Models, Student-Teacher

## TL;DR

Within a student-teacher distillation framework, this work augments the teacher with a pretrained feed-forward 3D reconstruction model (MVSplat) that lifts 2D features into a 3D Gaussian representation and renders them to novel viewpoints, enabling the student to learn geometrically consistent, 3D-aware 2D features. The proposed method surpasses existing approaches across downstream tasks including depth estimation, surface normal estimation, semantic segmentation, and multi-view correspondence.

## Background & Motivation

**Background**: Vision foundation models (VFMs) such as DINOv2 are trained on large-scale 2D data via self-supervised distillation, achieving strong performance on 2D tasks such as semantic segmentation. However, these models inherently lack 3D awareness and perform poorly on tasks requiring understanding of three-dimensional geometry, such as depth estimation, surface normal prediction, and multi-view correspondence.

**Limitations of Prior Work**:
1. **FiT3D** lifts 2D features into a 3DGS representation via per-scene optimization and renders the results to generate training data for fine-tuning VFMs. However, because input features from different viewpoints are themselves inconsistent, the optimization produces a "least-squares compromise," resulting in semantic blurring and feature-averaging artifacts.
2. **MEF** enforces feature consistency through multi-view correspondences, but relies solely on feature similarity constraints at corresponding points and cannot provide complete dense geometric understanding.
3. Per-scene optimization methods are computationally expensive, require large Gaussian representations, and are difficult to scale.

**Key Challenge**: Equipping 2D features with 3D awareness requires constraining feature learning through multi-view 3D geometry; yet per-scene optimization is slow and introduces averaging artifacts due to inconsistent input features. Feed-forward reconstruction is fast but had previously been applied only to appearance reconstruction rather than semantic features.

**Goal**: Within a student-teacher distillation framework, this work employs a frozen feed-forward 3D reconstruction model (MVSplat) as an augmentation component for the teacher. Teacher-extracted 2D features are upsampled in a mask-aware manner, attached to 3D Gaussians, rendered to the target viewpoint, and then processed via semantic blending to produce high-quality supervision signals. The student learns from a single 2D image to match the teacher's 3D-aware features, thereby acquiring geometric consistency. The teacher is iteratively updated via EMA, avoiding the static feature-averaging problem.

## Method

### Overall Architecture

Splat and Distill (SnD) builds upon the DINO/DINOv2 student-teacher self-distillation paradigm. At each training iteration, two context views and one target view are sampled from a scene $\mathcal{S} = \{(\mathbf{I}_i, \mathbf{P}_i)\}_{i=1}^N$. The core pipeline is as follows:

1. **3D Reconstruction**: Two context images are fed into the frozen MVSplat model, which predicts a set of 3D Gaussian primitives $\{(\mu_j, \Sigma_j, \alpha_j)\}$ in a feed-forward manner.
2. **Feature Extraction and Lifting**: The teacher network extracts 2D features $\mathbf{F}_j^{ctx} \in \mathbb{R}^{h \times w \times C}$ from the context images; these are upsampled via mask-aware interpolation and attached to the 3D Gaussians through pixel-to-Gaussian correspondences.
3. **Rendering and Blending**: The feature-augmented 3D scene is rendered to the target viewpoint, and a supervision feature map $\mathbf{F}_{blend}^{tgt}$ is obtained via semantic blending.
4. **Distillation**: The student extracts features $\mathbf{F}_s^{tgt}$ from only the target-view 2D image and aligns them with the teacher's rendered features via a cross-entropy distillation loss.
5. **Teacher Update**: Teacher parameters are updated via EMA: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$.

### Key Design 1: Mask-Aware Feature Upsampling

The feature map resolution ($h \times w$) output by the teacher network differs from the input image resolution ($H \times W$) by a factor of $\times 14$. Naive bilinear interpolation causes feature mixing across object boundaries. This work proposes guiding interpolation with semantic segmentation masks:

$$\mathbf{F}_u^{high} = \sum_{v \in \mathcal{N}(u)} w_{uv} \cdot \mathbf{F}_v^{low}$$

where the weight $w_{uv}$ is nonzero only when $\text{mask}(v) = \text{mask}(u)$, i.e., interpolation is performed exclusively from neighboring feature points within the same semantic region. This ensures sharp upsampled features at object boundaries and prevents feature leakage across semantic regions.

### Key Design 2: Semantic Blending Regularization

3D scenes reconstructed from sparse views may exhibit geometric artifacts when rendered to novel viewpoints. This work regularizes the rendered feature map via semantic blending:

$$\mathbf{F}_{blend}(u) = \alpha \cdot \mathbf{F}_{rendered}(u) + (1-\alpha) \cdot \frac{1}{|\mathcal{M}_u|} \sum_{v \in \mathcal{M}_u} \mathbf{F}_{rendered}(v)$$

where $\alpha = 0.5$ and $\mathcal{M}_u$ is the set of all pixels sharing the same semantic mask as pixel $u$. This corrects minor geometric inconsistencies by averaging features within semantic boundaries while preserving precise details at object edges.

### Key Design 3: Feed-Forward 3D and EMA Joint Dynamic Learning

Unlike the static optimization of FiT3D, the teacher in SnD is updated in conjunction with the student via EMA. As training progresses, the 2D features produced by the teacher become increasingly consistent, continuously improving the quality of features fed into the 3D reconstruction and forming a positive feedback loop. The distillation loss is:

$$\min_{\theta_s} \mathcal{L}_{distill}(\text{head}(\mathbf{F}_s^{tgt}), \text{sg}(\text{head}(\mathbf{F}_{blend}^{tgt})))$$

where $\text{head}(\cdot)$ denotes the DINO head (a small MLP) and $\text{sg}(\cdot)$ denotes the stop-gradient operation. This design enables the framework to learn generalizable 3D consistency from a large and diverse collection of scenes.

## Key Experimental Results

### Main Results

Linear probing evaluations are conducted on ScanNet++, ScanNet, and NYUv2, with comparisons against three baselines: DINOv2, FiT3D, and MEF.

**Monocular Depth Estimation (ViT-Small)**:

| Method | ScanNet++ RMSE↓ | ScanNet RMSE↓ | NYUv2 RMSE↓ |
|------|:---:|:---:|:---:|
| DINOv2 | 0.3777 | 0.2817 | 0.5210 |
| FiT3D | 0.3506 | 0.2713 | 0.5075 |
| MEF | 0.4000 | 0.3042 | 0.5656 |
| **SnD (Ours)** | **0.3299** | **0.2555** | **0.4912** |

**Surface Normal Estimation (NYUv2 RMSE↓)**:

| Method | ViT-Small | ViT-Base |
|------|:---:|:---:|
| DINOv2 | 30.99 | 31.40 |
| FiT3D | 30.57 | 30.57 |
| MEF | 33.05 | 32.60 |
| **SnD (Ours)** | **28.93** | **29.37** |

**Semantic Segmentation (ViT-Small, mIoU↑)**:

| Method | ScanNet++ | ScanNet | NYUv2 |
|------|:---:|:---:|:---:|
| DINOv2 | 29.54 | 51.27 | 64.73 |
| FiT3D | 31.77 | 54.50 | 66.33 |
| MEF | 27.44 | 47.44 | 63.17 |
| **SnD (Ours)** | **31.78** | **56.01** | **67.50** |

**Out-of-Domain Generalization (ViT-Base)**: The method achieves a mIoU of 50.01 on ADE20K segmentation (FiT3D: 48.29) and an RMSE of 2.1741 on KITTI depth estimation (FiT3D: 2.2485), demonstrating transferability from indoor to outdoor settings.

### Ablation Study

Ablation analysis is conducted on ScanNet++ using ViT-Small:

| Configuration | Seg mIoU↑ | Depth RMSE↓ |
|----------|:---:|:---:|
| w/o Blending (A) | 30.99 | 0.3435 |
| Bilinear upsampling instead of Mask-Aware (B) | 31.46 | 0.3309 |
| Cosine loss instead of distillation loss (C) | 31.27 | 0.3310 |
| Frozen teacher (D) | 31.90 | 0.3444 |
| Context views instead of novel views (E) | 32.08 | 0.3332 |
| SAM masks instead of ground-truth masks (F) | 31.51 | 0.3328 |
| Direct feature rendering loss (G) | 31.40 | 0.3430 |
| Minimal baseline (H) | 30.66 | 0.3520 |
| **Full model** | **31.78** | **0.3299** |

The ablation results indicate: (1) semantic blending and mask-aware upsampling each contribute critical gains in depth estimation performance; (2) the EMA-updated learnable teacher (vs. frozen teacher) improves RMSE from 0.3444 to 0.3299; (3) SAM masks achieve performance close to ground-truth annotations, validating the practical applicability of the method.

## Highlights & Insights

### Strengths

1. **Clear and elegant formulation**: Embedding feed-forward 3D reconstruction into the student-teacher distillation framework avoids the computational bottleneck and feature-averaging artifacts of per-scene optimization.
2. **Comprehensive and rigorous experiments**: Four downstream tasks, multiple datasets, in-domain and out-of-domain evaluations, and thorough ablations with sufficient reproducibility details.
3. **EMA dynamic update** establishes a positive feedback loop between teacher and student, enabling continuous improvement in feature quality throughout training.
4. **Practical applicability**: SAM masks approach the quality of ground-truth annotations, requiring no correspondence labels.

### Limitations & Future Work

1. The method depends on the quality of the pretrained MVSplat model; scenes with reconstruction failures may produce harmful supervision signals.
2. Training is conducted exclusively on ScanNet++ data, and 3D reconstruction quality for large outdoor scenes may be limited.
3. Multi-view data is required during training, restricting applicability in settings where only single-image data is available.

## Rating

⭐⭐⭐⭐ — The method is novel and clearly presented, the experiments are comprehensive and rigorous, and the work represents a significant contribution to the direction of 3D-aware VFMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[ICLR 2026\] Splat Feature Solver](splat_feature_solver.md)
- [\[AAAI 2026\] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction](../../AAAI2026/3d_vision/splat-sap_feed-forward_gaussian_splatting_for_human-centered_scene_with_scale-aw.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](../../CVPR2026/3d_vision/off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
