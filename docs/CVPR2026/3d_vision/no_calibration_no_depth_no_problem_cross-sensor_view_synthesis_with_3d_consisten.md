---
title: >-
  [Paper Note] No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency
description: >-
  [3D Vision] This paper proposes the first cross-sensor view synthesis framework that requires neither calibration nor depth. Through a match-densify-consolidate pipeline…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: ec5318c4f04e390c
---

# No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency

## Basic Information

- **Conference**: CVPR 2026
- **arXiv**: [2602.23559](https://arxiv.org/abs/2602.23559)
- **Authors**: Cho-Ying Wu, Zixun Huang, Xinyu Huang, Liu Ren (Bosch Research North America & BCAI)
- **Code**: To be confirmed (project page available)
- **Area**: 3D Vision / Cross-Sensor View Synthesis
- **Keywords**: Cross-Sensor View Synthesis, RGB-X Alignment, 3D Gaussian Splatting, Image Matching, Confidence-Aware Densification

## TL;DR

This paper proposes the first cross-sensor view synthesis framework that requires neither calibration nor depth. Through a match-densify-consolidate pipeline, sparse cross-modal keypoints are expanded into dense X-modality images (thermal/NIR/SAR) aligned with the RGB viewpoint. Synthesis quality is further improved via confidence-aware densification fusion (CADF) and self-matching filtering.

## Background & Motivation

Sensors beyond RGB — thermal, near-infrared (NIR), and synthetic aperture radar (SAR) — are critical for applications such as nighttime autonomous driving and leakage detection, yet they have received far less research attention than RGB. The core bottleneck lies in the extreme difficulty of acquiring pixel-aligned RGB-X paired data:

1. Traditional industrial pipelines require intrinsic calibration, sensor synchronization, relative pose estimation, and accurate metric depth, leading to cascading errors with no way to handle occlusion.
2. SfM methods such as COLMAP apply only to RGB and typically fail on low-texture sensors (e.g., thermal cameras).
3. Cross-modal matchers (XoFTR, MINIMA) can only estimate a homography $H \in \mathbb{R}^{3\times3}$ for warping, which assumes a planar scene structure and produces severe misalignment when foreground–background layering is present.
4. Image translation approaches (RGB-to-Thermal) suffer from inherent ambiguity — the temperature of a cup of water cannot be inferred from its appearance alone.

This paper proposes the first scalable cross-sensor view synthesis framework that requires no 3D prior of the X sensor (no depth, no calibration), relying solely on COLMAP applied to RGB at virtually zero additional cost.

## Method

### Overall Architecture

The method proceeds in three stages:

1. **Matching stage**: Cross-modal feature matching combined with region-based sampling to produce a semi-dense X-map $\mathcal{X}_m$.
2. **Densification stage**: RGB-guided densification with confidence-aware densification fusion (CADF) to produce a dense X image $\mathcal{X}_d$.
3. **Consolidation stage**: Self-matching filtering, refined densification, and RGB-X 3DGS 3D consolidation.

### RGB-X Matching

Given an RGB image $\mathcal{I}$ and an X-modality image $\mathcal{X}$, a cross-modal matcher (XoFTR) finds a match set $\{(p^{\mathcal{I}}, p^{\mathcal{X}}, c)\}$. X keypoints from $N=7$ frames (3 preceding and 3 following) are accumulated into the current RGB coordinate frame:

$$\mathcal{X}_m[p] = \frac{\sum_n \mathbf{1}[p=p_n^{\mathcal{I}}] \, \mathcal{X}[p_n^{\mathcal{X}}]}{\sum_n \mathbf{1}[p=p_n^{\mathcal{I}}]}$$

For textureless regions (sky, ground, walls), GroundedSAM is used for segmentation, after which only 5% of points are uniformly sampled from the homography-warped X image as a supplement. This avoids propagating warping errors into subsequent densification:

$$\mathcal{X}_m[p] = \mathcal{X}_W[p], \quad p \sim \mathrm{U}(\{p \mid \mathcal{M}(p)=1 \wedge \mathcal{X}_m[p]=-1\})$$

### Confidence-Aware Densification and Fusion (CADF)

The densification network $D$ adopts a recurrent unit with dynamic spatial propagation (DySPN). It takes the RGB image and sparse X-map as input and outputs a dense X image. The original DySPN iteration is:

$$L^{t+1} = (1 - C_s) \sum_r \sum_{(a,b)} w_{r,a,b} * L_{a,b}^t + C_s \mathcal{X}_m$$

where $C_s$ is a certainty map predicted by the backbone. **Key modification**: A matching confidence map $C_m$ (aggregated from match scores $c$) is incorporated into the iteration to reduce the contribution of low-confidence keypoints:

$$L^{t+1} = (1 - C_s C_m) \sum_r \sum_{(a,b)} w_{r,a,b} * L_{a,b}^t + C_s C_m \mathcal{X}_m$$

**Multi-level threshold fusion**: Different confidence thresholds $\delta$ involve a trade-off — high thresholds retain reliable points but are too sparse, while low thresholds accept more points at the cost of noise. The method employs $K=3$ threshold levels $\delta = 0.15, 0.3, 0.5$, generating densified results $\hat{\mathcal{X}}_{d,k}$ for each, which are then fused via mean pooling through a fusion module $F$ (an image enhancement network pretrained on DIV2K).

$F$ is trained with two self-supervised losses:

**Cosine similarity loss** (based on the SigLIP2 image encoder):

$$\mathcal{L}_{\text{cos}}(\mathcal{I}, \mathcal{X}_d) = 1 - \frac{f_{\text{SigLIP}}(\mathcal{I})^\top f_{\text{SigLIP}}(\mathcal{X}_d)}{\|f_{\text{SigLIP}}(\mathcal{I})\|_2 \|f_{\text{SigLIP}}(\mathcal{X}_d)\|_2}$$

### Self-Matching Filtering and 3D Consolidation

**Self-matching mechanism**: Aligned RGB-X pairs carry a prior — each patch should match to the same position in itself. Patch-level similarity matrices are computed using transformer features from the matcher:

$$A = \frac{F_{\mathcal{I}} F_{\mathcal{X}}^\top}{\tau}$$

The ideal similarity matrix is diagonal. During training of $F$, the diagonal sum is maximized while off-diagonal elements are minimized:

$$\mathcal{L}_{\text{sim}}(A) = -\frac{\operatorname{Tr}(A)}{\|A\|_F} + \lambda \frac{\|A \odot (\hat{\mathbf{1}} - I)\|_1}{\|A\|_F}$$

**Filtering**: A concentration metric $q = Q_{50}(\mathbf{A}) / Q_{99}(\mathbf{A})$ is computed; patches with low diagonal scores are filtered using the $(1-q)$ quantile as threshold. A high $q$ indicates good self-matching and fewer patches requiring filtering.

**Refined densification**: A single-level densification pass is applied on the filtered X image, using the normalized self-matching scores as $C_m$.

**RGB-X 3DGS consolidation**: 3DGS is trained on RGB-viewpoint COLMAP camera poses, with an additional X channel appended to each Gaussian. Unlike methods that use separate parameter sets, the proposed approach shares a single set of geometric parameters — as RGB images are of higher quality and can more accurately localize each 3D Gaussian in space.

### Loss & Training

- Densification network $D$: Pretrained on synthetic RGB-X paired data (MINIMA for RGB-Thermal; Deep-NIR for RGB-NIR).
- Fusion module $F$: First pretrained on DIV2K for image enhancement (denoising, deblurring, super-resolution), then fine-tuned with the cosine similarity loss and self-matching loss.
- Hyperparameters: $N=7$ frames, $K=3$ levels, $\delta=0.15/0.3/0.5$, $\lambda=0.1$, $\tau=0.1$, region sampling confidence $c=0.3$.

## Key Experimental Results

### Datasets and Modalities

| Modality | Test Dataset | Training Data |
|:--|:--|:--|
| RGB-Thermal | METU-VisTIR-Cloudy (6 sequences), RGBT-Scenes (4 scenes) | MINIMA synthetic RGB-Thermal |
| RGB-NIR | RGB-NIR-Stereo (5 sequences) | Deep-NIR synthetic data |
| RGB-SAR | DDHR-HK (3 satellite image pairs cropped into 512×512 patches) | Multiple RGB-SAR datasets |

### Main Results: METU-VisTIR-Cloudy RGB-Thermal (No GT, Mean over 6 Sequences)

| Method | Icos↑ | p30↑ | p50↑ | p70↑ | p90↑ | ITM↑ | ITcos↑ |
|:--|:--|:--|:--|:--|:--|:--|:--|
| XoFTR | 0.62 | 25.13 | 27.49 | 29.31 | 31.48 | 0.69 | 0.39 |
| LightGlue | 0.61 | 25.89 | 28.28 | 30.14 | 32.35 | 0.91 | 0.40 |
| LoFTR | 0.66 | 29.38 | 32.07 | 33.95 | 36.04 | 0.89 | 0.45 |
| MINIMA | 0.67 | 29.93 | 32.78 | 34.72 | 36.99 | 0.88 | 0.44 |
| **Ours** | **0.69** | **31.18** | **34.39** | **36.43** | **38.72** | **0.92** | **0.45** |

### Ablation Study (Mean over RGB-NIR-Stereo)

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|:--|:--|:--|:--|
| Full method | **21.152** | **0.581** | **0.344** |
| w/o 3DGS | 21.042 | 0.597 | 0.378 |
| w/o self-matching & filtering | 20.235 | 0.522 | 0.386 |
| w/o DySPN confidence | 19.621 | 0.508 | 0.396 |
| w/o multi-level thresholds | 19.215 | 0.495 | 0.420 |
| w/o region sampling | 16.454 | 0.408 | 0.467 |

### RGB-NIR Results (All Methods Use 3DGS)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|:--|:--|:--|:--|
| PixNext (generative) | 11.283 | 0.441 | 0.452 |
| XoFTR | 14.846 | 0.321 | 0.486 |
| LoFTR | 20.179 | 0.551 | 0.356 |
| MINIMA | 20.392 | 0.568 | 0.360 |
| **Ours** | **21.152** | **0.581** | **0.344** |

### RGB-SAR Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|:--|:--|:--|:--|
| MINIMA | 14.849 | 0.229 | 0.377 |
| **Ours** | **17.102** | **0.302** | **0.339** |

### Key Findings

1. **Confidence-aware fusion contributes the most**: Injecting matching confidence into DySPN yields approximately 1 dB PSNR improvement (19.215→20.235), validating the effectiveness of propagating matching uncertainty into densification.
2. **Even without 3DGS, the proposed method outperforms all baselines with 3DGS**: Ours (w/o 3DGS) achieves PSNR=21.042, still exceeding all methods with 3DGS (best: MINIMA=20.392). The core contributions lie in the sampling and CADF strategies.
3. **Region sampling is foundational**: Removing region sampling causes PSNR to drop sharply to 16.454 (a reduction of ~4.7 dB), demonstrating that sparse sampling in textureless regions is critical for global densification.
4. **Temporal consistency outperforms image generation**: StyleBooth MEt3R=0.297 vs. the proposed method's 0.171. Matching retrieves true sensor values rather than hallucinated ones.
5. **Cross-modal generalization**: The same framework achieves state-of-the-art performance across three substantially different modalities: Thermal, NIR, and SAR.

## Highlights & Insights

- **Well-motivated problem formulation**: This is the first systematic study of cross-sensor view synthesis, identifying the blind spot in prior RGB-X work that universally assumes paired data already exists.
- **Zero-cost assumption**: Only COLMAP on RGB is required; the X sensor needs no 3D prior whatsoever.
- **Confidence propagated throughout the pipeline**: Matching confidence flows through keypoint selection, DySPN iteration, multi-level fusion, and self-matching filtering, forming a complete uncertainty propagation chain.
- **Novel self-matching idea**: The matcher is repurposed as an evaluator, leveraging the prior that aligned patches should self-match, enabling quality filtering without additional models.

## Limitations & Future Work

- Only static scenes are handled; dynamic objects degrade 3D consolidation (an inherent limitation of 3DGS).
- Sensors such as thermal cameras exhibit high native noise and low resolution, limiting performance when data quality is poor.
- The method still depends on cross-modal matchers and fails in highly homogeneous regions with no valid descriptors.
- The 5% ratio for region sampling is set manually, lacking an adaptive mechanism.
- The densification network must be trained separately for each modality.

## Rating

⭐⭐⭐⭐ — The problem is clearly defined and practically valuable. The match-densify-consolidate framework is naturally structured with well-justified module contributions supported by comprehensive ablation studies. Consistent state-of-the-art performance across three distinct modalities is convincing. Primary deductions are for the static scene restriction and strong dependence on matcher quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](../../ICCV2025/3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](../../ICLR2026/3d_vision/urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)

</div>

<!-- RELATED:END -->
