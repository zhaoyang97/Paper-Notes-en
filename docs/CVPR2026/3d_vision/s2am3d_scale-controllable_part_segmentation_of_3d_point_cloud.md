---
title: >-
  [Paper Note] S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds
description: >-
  [CVPR2026][3D Vision][Point cloud part segmentation] This paper presents S2AM3D, a point cloud part segmentation framework that integrates 2D pretrained priors with 3D contrastive supervision. A point-consistent encoder…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "Point cloud part segmentation"
  - "multi-granularity control"
  - "contrastive learning"
  - "2D-3D joint supervision"
  - "SAM"
date: 2026-05-08
content_hash: bc1ad6f1f77e1e2b
---

# S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds

**Conference**: CVPR2026
**arXiv**: [2512.00995](https://arxiv.org/abs/2512.00995)
**Code**: [Project Page](https://sumuru789.github.io/S2AM3D-website/)
**Area**: 3D Vision
**Keywords**: Point cloud part segmentation, multi-granularity control, contrastive learning, 2D-3D joint supervision, SAM

## TL;DR

This paper presents S2AM3D, a point cloud part segmentation framework that integrates 2D pretrained priors with 3D contrastive supervision. A point-consistent encoder produces globally coherent per-point features, while a scale-aware prompt decoder enables continuously controllable segmentation granularity. The method substantially outperforms existing approaches across multiple benchmarks.

## Background & Motivation

Part-level segmentation of point clouds is a critical task bridging fine-grained geometric detail and high-level semantic understanding, with important applications in 3D content creation, robotic manipulation, and reverse engineering. Existing methods face three key challenges:

1. **Poor generalization of native 3D methods**: High-quality 3D part annotations are costly, and existing datasets are limited in scale and category diversity (e.g., ShapeNet-Part, PartNet), severely constraining generalization to open-domain shapes.
2. **Cross-view inconsistency of 2D prior-based methods**: Methods that apply 2D models such as SAM to rendered views and fuse results suffer from cross-view inconsistencies under occlusion, elongated structures, and complex topology, accumulating errors that degrade global 3D coherence.
3. **Inflexible granularity control**: Feature-clustering methods such as PartField rely on post-processing clustering, yielding discontinuous and unintuitive control; prompt-based methods such as Point-SAM lack explicit granularity control mechanisms.

## Method

### Overall Architecture

S2AM3D adopts a **decoupled training** strategy with two stages:

- **Stage 1**: Train a Point-Consistent Part Encoder that fuses 2D segmentation priors with 3D contrastive supervision to produce globally consistent per-point features.
- **Stage 2**: Freeze the encoder and train a Scale-Aware Prompt Decoder conditioned on a point prompt index $p$ and an optional scale prompt $s \in [0,1]$ for flexible part segmentation.

Given input point cloud $\mathbf{P} \in \mathbb{R}^{N \times 3}$, the encoder outputs per-point features $\mathbf{F} \in \mathbb{R}^{N \times D}$, and the decoder produces a probability mask $\hat{\mathbf{m}} \in [0,1]^N$.

### Point-Consistent Part Encoder

The core idea is to **layer 3D contrastive supervision on top of 2D distillation**, resolving cross-view inconsistencies that arise from multi-view 2D distillation alone.

**Base architecture**: A PVCNN voxel encoder extracts point features, which are converted into a tri-plane representation $\mathbf{T} \in \mathbb{R}^{3 \times D \times H \times W}$ (three orthogonal planes: $xy$, $yz$, $zx$), followed by Transformer blocks for feature aggregation. Tri-plane features are rendered from random viewpoints into 2D latents and supervised via SAM distillation.

**Tri-plane feature extraction**: Given 3D coordinates $(x,y,z)$, each point is back-projected onto the three feature planes and summed:

$$\mathbf{F} = \Big[\mathbf{T}_{xy}(x_n, y_n) + \mathbf{T}_{yz}(y_n, z_n) + \mathbf{T}_{zx}(z_n, x_n)\Big]_{n=1}^{N}$$

**3D contrastive supervision**: The key innovation is the introduction of contrastive learning using native 3D labeled data. Contrastive pairs are constructed intra-instance, with each mini-batch containing a single object to avoid cross-instance semantic mismatch. For anchor point $i$ with label $y_i$, the positive set is:

$$\hat{P}(i) = \{j \in \hat{P} \setminus \{i\} \mid y_j = y_i\}$$

Using cosine similarity with temperature $\tau$, $s_{ij} = \mathbf{f}_i^\top \mathbf{f}_j / \tau$, the contrastive loss is:

$$\mathcal{L}_{\text{contr}} = \frac{1}{|\hat{P}|} \sum_{i \in \hat{P}} -\log \frac{\sum_{j \in \hat{P}(i)} e^{s_{ij}}}{\sum_{j \in \hat{P} \setminus \{i\}} e^{s_{ij}}}$$

This objective compactly clusters features from the same part and separates features from different parts, yielding globally consistent embeddings and sharp boundaries.

### Scale-Aware Prompt Decoder

The decoder receives the encoder's output $\mathbf{F}$ and 3D coordinates $\mathbf{P}$, augmented with 3D sinusoidal positional encodings to form the base representation:

$$\mathbf{X}^{(0)} = \mathbf{F} + \mathrm{PE}(\mathbf{P})$$

#### Scale Modulator

The scale $s$ is defined as the relative size of a part (the fraction of total points belonging to that part). For continuous scale $s \in [0,1]$, **learnable sinusoidal embeddings** are constructed:

$$\mathbf{e}(s) = \big[\sin(\omega_k s + \phi_k), \ \cos(\omega_k s + \phi_k)\big]_{k=1}^{M}$$

where $\{\omega_k, \phi_k\}$ are learnable parameters and $M$ is the number of frequency pairs. Global features are then modulated channel-wise via **FiLM (Feature-wise Linear Modulation)**:

$$[\boldsymbol{\gamma}, \boldsymbol{\beta}] = \text{Linear}(\mathrm{LN}(\mathbf{e}(s)))$$

$$\mathrm{FiLM}(\mathbf{X}; s) = \mathbf{X} \odot (1 + \alpha \boldsymbol{\gamma}) + \alpha \boldsymbol{\beta}$$

where $\alpha$ is a learnable scalar gate. FiLM layers and Transformer blocks are interleaved for $L_m$ layers:

$$\mathbf{X}^{(\ell+1)} = T_\ell\big(\mathrm{FiLM}(\mathbf{X}^{(\ell)}; s)\big), \quad \ell = 0, \dots, L_m - 1$$

This yields the scale-conditioned representation $\tilde{\mathbf{F}} = \mathbf{X}^{(L_m)}$.

**Scale Dropout**: During training, $\mathbf{e}(s)$ is randomly zeroed with probability 0.1, reducing FiLM to an identity mapping and ensuring the model remains functional at inference time without a scale input.

#### Bi-directional Cross-Attention

Unidirectional cross-attention struggles to simultaneously perform context aggregation and fine-grained refinement in a single forward pass. Bi-directional cross-attention enables mutual interaction between the prompt point feature $\tilde{\mathbf{F}}_p \in \mathbb{R}^{1 \times D}$ and global features $\tilde{\mathbf{F}} \in \mathbb{R}^{N \times D}$:

$$\mathbf{q}^{(\ell+1)} = \mathbf{q}^{(\ell)} + \mathrm{CAttn}(\mathbf{q}^{(\ell)}; \mathbf{Y}^{(\ell)})$$

$$\mathbf{Y}^{(\ell+1)} = \mathrm{FFN}\Big(\mathbf{Y}^{(\ell)} + \mathrm{CAttn}(\mathbf{Y}^{(\ell)}; \mathbf{q}^{(\ell+1)})\Big)$$

After stacking $L_d$ layers, a per-point probability mask is output via MLP and Sigmoid:

$$\hat{\mathbf{m}} = \sigma(\mathrm{MLP}(\mathbf{H})) \in [0,1]^N$$

### Loss & Training

The segmentation loss uses a **dynamically weighted BCE + Dice** hybrid objective:

$$\mathcal{L}_{\text{seg}} = \lambda_{\text{bce}} \mathrm{BCE}_{\text{dyn}}(\hat{\mathbf{m}}, \mathbf{m}) + \lambda_{\text{dice}} \left(1 - \frac{2\hat{\mathbf{m}}^\top \mathbf{m}}{\|\hat{\mathbf{m}}\|_1 + \|\mathbf{m}\|_1}\right)$$

Dynamic BCE adaptively computes the weight $\beta = (1-\pi)/(\pi + \varepsilon)$ based on each sample's positive ratio $\pi$, alleviating class imbalance. The Dice term directly optimizes set-level overlap, yielding greater robustness for small parts and long-tail distributions.

### Dataset Construction

A large-scale dataset containing **100K+ point cloud instances and approximately 1.2M part annotations** is constructed from Objaverse, covering 400+ categories. The automated pipeline consists of three steps:

1. **Part annotation**: Parts are sampled and assigned labels based on surface area ratios.
2. **Quality filtering**: A binary PointNet classifier is trained to automatically filter samples with unreliable annotations.
3. **Connectivity refinement**: Spatially disconnected regions sharing the same label are split into independent labels using DBSCAN clustering.

## Key Experimental Results

### Main Results

**Interactive segmentation (point prompt → single part mask)**:

| Method | PartObjaverse-Tiny (IoU%) | PartNet-E (IoU%) | Mean |
|------|:---:|:---:|:---:|
| Point-SAM | 31.46 | 50.23 | 40.85 |
| P3-SAM | 35.05 | 39.98 | 37.52 |
| **S2AM3D** | **46.47** | **62.52** | **54.50** |
| **S2AM3D (+scale)** | **61.19** | **77.51** | **69.35** |

**Full segmentation (predicting part labels for all points)**:

| Method | PartObjaverse-Tiny (IoU%) | PartNet-E (IoU%) | Mean |
|------|:---:|:---:|:---:|
| Find3D | 20.76 | 21.69 | 21.23 |
| SAMPart3D | 48.79 | 56.17 | 52.48 |
| SAMesh | - | 26.66 | - |
| PartField | 51.54 | 59.10 | 55.32 |
| P3-SAM | 58.10 | 65.39 | 61.75 |
| **S2AM3D** | **63.29** | **77.98** | **70.64** |

### Ablation Study

| Setting | PartObjaverse-Tiny | PartNet-E | Mean |
|------|:---:|:---:|:---:|
| **+scale full model** | **61.19** | **77.51** | **69.35** |
| +scale w/o 3D supervision | 53.94 | 64.11 | 59.03 |
| +scale w/o custom dataset | 53.12 | 66.12 | 59.62 |
| **No scale full model** | **46.47** | **62.52** | **54.50** |
| No scale w/o 3D supervision | 41.14 | 55.39 | 48.27 |
| No scale w/o custom dataset | 42.12 | 58.56 | 50.34 |
| No scale w/o scale embedding | 42.31 | 58.28 | 50.30 |

### Key Findings

- **3D contrastive supervision is the largest performance contributor**: Removing it causes a 10.32% mean IoU drop under the +scale setting; feature visualizations reveal blurred boundaries and internal inconsistencies.
- **Scale prompts yield substantial gains**: Adding scale conditioning improves mean IoU from 54.50% to 69.35% (+14.85%) in the interactive segmentation setting.
- **Critical role of the custom dataset**: Replacing it with PartNet training data leads to a notable performance drop, demonstrating the complementary distributional value of the large-scale, high-quality dataset.
- **Scale embedding enhances decoding robustness**: Even without a scale input at inference time, models trained with scale embeddings outperform those without (54.50 vs. 50.30).
- **Only XYZ coordinates required**: Unlike Point-SAM, which requires color, and P3-SAM, which requires normals, S2AM3D achieves state-of-the-art results using coordinates alone.

## Highlights & Insights

- The 2D-3D joint training paradigm is elegantly designed: 2D priors provide generalization capability while 3D contrastive supervision enforces global consistency, with strong complementarity between the two.
- The scale-aware decoder achieves continuous granularity control via FiLM combined with bi-directional cross-attention, supporting smooth coarse-to-fine transitions.
- The automated data pipeline (annotation → filtering → refinement) is scalable, producing one of the largest 3D part segmentation datasets to date.
- A single unified framework handles both interactive segmentation and full segmentation tasks.

## Limitations & Future Work

- Only point prompts and scale signals are supported for interaction; future work could incorporate text instructions for more intuitive semantic control.
- The encoder relies on PartField pretrained weights for initialization, limiting the novelty of the encoder architecture itself.
- Dataset annotation depends on an automated pipeline, and the generalizability of the filtering strategy (PointNet classifier) has not been thoroughly validated.
- Inference speed and memory consumption are not discussed; the tri-plane representation at $448 \times 512 \times 512$ dimensions for 10,000 sampled points is relatively large.
- Evaluation datasets are small in scale (PartObjaverse-Tiny contains only 200 samples); larger-scale evaluation would strengthen the conclusions.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of 2D-3D joint supervision and continuous scale control is creative, though the base encoder architecture follows PartField.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two task settings, ablations, and visualizations are provided, though the evaluation datasets are limited in scale.
- Writing Quality: ⭐⭐⭐⭐ — Logical structure is clear, mathematical derivations are rigorous, and figures are of high quality.
- Value: ⭐⭐⭐⭐ — Provides a practical granularity-controllable solution for 3D part segmentation; the dataset contribution adds additional value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp3d_joint_open_vocabulary_semantic_segmentation.md)
- [\[ICLR 2026\] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](../../ICLR2026/3d_vision/partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)
- [\[CVPR 2026\] 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds](3d_sans_3d_scans_scalable_pre-training_from_video-generated_point_clouds.md)
- [\[CVPR 2026\] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance](gaussiangrow_geometry-aware_gaussian_growing_from_3d_point_clouds_with_text_guid.md)
- [\[CVPR 2026\] PointINS: Instance-Aware Self-Supervised Learning for Point Clouds](pointins_instance-aware_self-supervised_learning_for_point_clouds.md)

</div>

<!-- RELATED:END -->
