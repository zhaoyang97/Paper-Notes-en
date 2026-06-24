---
title: >-
  [Paper Note] CrossScore: Towards Multi-View Image Evaluation and Scoring
description: >-
  [ECCV 2024][3D Vision][Image Quality Assessment] A new Cross-Reference (CR) image quality assessment paradigm is proposed. By comparing a query image with multiple reference images from different perspectives, a cross-attention neural network is utilized to predict pixel-level quality scores highly correlated with SSIM, enabling the evaluation of novel view synthesis quality without ground-truth reference images.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Image Quality Assessment"
  - "Cross-Reference Evaluation"
  - "Novel View Synthesis"
  - "Cross-Attention"
  - "Self-Supervising"
date: 2026-05-08
content_hash: c201b7fd6786ff4e
---

# CrossScore: Towards Multi-View Image Evaluation and Scoring

**Conference**: ECCV 2024  
**arXiv**: [2404.14409](https://arxiv.org/abs/2404.14409)  
**Code**: Yes ([https://crossscore.active.vision](https://crossscore.active.vision))  
**Area**: 3D Vision  
**Keywords**: Image Quality Assessment, Cross-Reference Evaluation, Novel View Synthesis, Cross-Attention, Self-Supervising

## TL;DR

A new Cross-Reference (CR) image quality assessment paradigm is proposed. By comparing a query image with multiple reference images from different perspectives, a cross-attention neural network is utilized to predict pixel-level quality scores highly correlated with SSIM, enabling the evaluation of novel view synthesis quality without ground-truth reference images.

## Background & Motivation

Existing image quality assessment (IQA) paradigms include:
- **Full-Reference (FR)**: such as SSIM, LPIPS, which require pixel-aligned GT images.
- **No-Reference (NR)**: such as NIQE, BRISQUE, which evaluate based solely on the statistical features of a single image.
- **General-Reference (GR)**: such as FID, which evaluate distribution discrepancies at the dataset level.
- **Multi-Modal-Reference (MMR)**: such as CLIPScore, which evaluate image-text similarity.

**Limitations of Prior Work in Novel View Synthesis (NVS) Evaluation**:

1. Traditional FR evaluation requires extracting test images from the training trajectory, necessitating a trade-off between the number of training and evaluation images.
2. For **genuinely novel trajectories** rendering, no GT is available, making FR metrics completely unusable.
3. NR and GR metrics lack pixel-level detailed analysis capabilities, making them unsuitable for NVS.

**Core Idea**: Replace a single GT image with multiple reference images from different views to achieve pixel-realignment-free SSIM prediction—a "perspective-variant" FR evaluation.

## Method

### Overall Architecture

Given a query image $\tilde{I}_q$ and a cross-reference image set $\mathcal{I}_r = \{I_r^i | i=1...N_{ref}\}$ (different perspectives of the same scene), the goal is to find a function $g(\cdot)$ such that:

$$g(\tilde{I}_q, \mathcal{I}_r) \mapsto \mathbf{S}_{cross} \approx \mathbf{S}_{ssim}$$

That is, approximating the output of the SSIM function using multi-view reference images, without requiring aligned GT.

The network $\Phi$ consists of three components:
1. **Image Encoder** $\Phi_{enc}$: Extracts feature maps.
2. **Cross-Reference Module** $\Phi_{cross}$: Associates query and reference images.
3. **Score Regression Head** $\Phi_{dec}$: Outputs pixel-level score maps.

### Key Designs

#### 1. Image Encoder — DINOv2

- A pretrained DINOv2-small is utilized as the encoder.
- $14\times14$ patch encoding is used, outputting $384$-channel feature maps.
- A shared encoder is used for the query and all reference images.
- Patch-wise positional encodings are used, while image-level encodings are excluded (as the reference set is unordered).

#### 2. Cross-Reference Module — Transformer Decoder

The core is the **cross-attention mechanism**:

- The query image features $\mathbf{F}_q$ serve as the **query** of the cross-attention.
- The reference image feature set $\mathcal{F}_r$ serves as the **key and value**.
- A 2-layer Transformer Decoder is used, with a hidden dimension of 384.

Intuitive understanding: For each patch in the query image, the most relevant observations are located in all reference images, and this information is used to judge the rendering quality of that patch.

#### 3. Score Regression Head — MLP

- A 2-layer MLP decodes the latent score map into a pixel-level score map.
- Since DINOv2 encodes by patch, the final MLP layer expands each latent score into a $14\times14$ patch score.
- These are finally concatenated into a full-resolution CrossScore map $\mathbf{S}_{cross} \in \mathbb{R}^{H \times W}$.

#### 4. Self-Supervised Training Data Generation

**The most ingenious design** — leveraging the training process of existing NVS systems to generate training data:

- Three NVS methods are trained on the MFR dataset: Gaussian Splatting, Nerfacto, and TensoRF.
- Checkpoints are saved every 1000 steps (11 checkpoints in total), and images are rendered at each checkpoint.
- The rendered images contain **different types and degrees of artifacts**, and are compared with the GT to obtain SSIM score maps.
- The different representations of the three NVS methods (point cloud, voxel, and plane decomposition) ensure **artifact diversity**.
- The overall data generation took approximately two weeks on 4×A5000 GPUs, yielding around 1.5TB of data.

### Loss & Training

$$\mathcal{L} = |\mathbf{S}_{ssim} - \mathbf{S}_{cross}|$$

A straightforward L1 loss. The SSIM map is clipped to [0,1] to stabilize training.

**Training Settings**:
- Random cropping of a $518\times518$ region (matching the DINOv2 input).
- Random selection of $N_{ref}=5$ reference images at each iteration.
- Trained on 2×A5000 24GB GPUs for 160K steps (60 hours).
- AdamW optimizer, with a learning rate of 5e-4 and a batch size of 24 per GPU.
- Trained solely on the MFR dataset, and evaluated on MFR + Mip360 + RE10K.

## Key Experimental Results

### Correlation with SSIM (Pearson Correlation Coefficient)

| Dataset | PSNR (FR) | BRISQUE (NR) | NIQE (NR) | PIQE (NR) | **CrossScore (CR)** |
|---|---|---|---|---|---|
| RE10K | 0.92 | 0.46 | 0.32 | 0.27 | **0.99** |
| Mip360 | 0.91 | 0.19 | 0.61 | 0.69 | **0.95** |
| MFR | 0.92 | 0.23 | -0.30 | -0.11 | **0.83** |

### Evaluating Few-shot NeRF (MFR Dataset)

| NVS Method | SSIM↑ | PSNR↑ | CrossScore↑ |
|---|---|---|---|
| PixelNeRF | 0.26 | 9.17 | 0.40 |
| IBRNet | 0.44 | 18.51 | 0.71 |

CrossScore is consistent with the ranking of SSIM/PSNR and can be used for cross-method comparisons.

### Novel Trajectory Evaluation (14 scenarios in MFR)

The Pearson correlation coefficient between traditional SSIM (subsampled test views) and CrossScore (novel trajectory) reaches **0.84**, with a close Spearman rank correlation.

### Ablation Study

| Reference Set | Correlation Coefficient |
|---|---|
| **Enabled** (✓) | **0.83** |
| Disabled (✗) | Decreased to ~0.7 |

Disabling the reference set degrades the model to an NR-style evaluation, with reduced details in the score map and a tendency to assign high scores to all areas.

### Key Findings

1. The correlation between CrossScore and SSIM reaches 0.99 on RE10K, even surpassing PSNR (0.92).
2. NR metrics (BRISQUE, NIQE, PIQE) exhibit extremely low or even negative correlations across multiple datasets, rendering them unsuitable for NVS evaluation.
3. Despite being trained only on MFR (outdoor objects/buildings), it successfully generalizes to Mip360 (360° indoor/outdoor scenes) and RE10K.
4. Attention visualization demonstrates that the model learns to locate semantic regions in the reference images that correspond to the query.

## Highlights & Insights

1. **Brand-New IQA Paradigm**: Cross-Reference fills the gap between FR and NR, particularly suitable for NVS scenarios.
2. **Self-Supervised Data Engine**: Generates training data using intermediate results from the NVS training process, requiring no human annotations.
3. **Strong Generalization**: Being trained on just a single dataset yet generalizing across different domains indicates that the model has learned a general quality-to-multiview association.
4. **DINOv2 + Cross-Attention**: A simple yet effective architectural choice, proving the applicability of pretrained vision Transformers to 3D tasks.
5. **High Practical Value**: Enables the rendering quality evaluation of novel trajectories without GT, which significantly advances the methodology of NVS evaluation.

## Limitations & Future Work

1. Currently, the model only predicts the SSIM metric; it could be extended to perceptual metrics such as LPIPS.
2. Training data generation relies on specific NVS methods (GS, Nerfacto, TensoRF); incorporating more methods could increase artifact diversity.
3. The number of reference images is fixed at 5; dynamically selecting the optimal reference set might improve performance.
4. Performance may be limited in scenarios with extremely large baseline differences (where there is a significant distance between the reference and query).
5. Computational Cost: Encoding all reference images with DINOv2 incurs a non-negligible overhead.

## Related Work & Insights

- **SSIM**: A classic FR metric, which this work aims to approximate when GT is unavailable.
- **DINOv2**: A strong visual feature extractor, providing the foundation for patch-level correspondences.
- **FID/CLIPScore**: Respective evaluations of distribution and semantics, but they lack pixel-level details.
- **RR-IQA**: Reduced-reference IQA metrics also attempt to lower dependency on GT, but still require partial GT information.
- Insight: **Leveraging the NVS training process itself as a data engine** is an elegant self-supervised strategy.

## Rating

| Dimension | Score (1-10) |
|---|---|
| Novelty | 8 |
| Technical Depth | 7 |
| Experimental Thoroughness | 8 |
| Writing Quality | 9 |
| Value | 8 |
| **Overall Score** | **8.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](../../NeurIPS2025/3d_vision/nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[ECCV 2024\] Differentiable Convex Polyhedra Optimization from Multi-view Images](differentiable_convex_polyhedra_optimization_from_multi-view_images.md)

</div>

<!-- RELATED:END -->
