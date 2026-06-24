---
title: >-
  [Paper Note] MGGTalk: Monocular and Generalizable Gaussian Talking Head Animation
description: >-
  [CVPR 2025][3D Vision][Talking head generation] The MGGTalk framework is proposed, which can generalize to unseen identities using only monocular datasets for training. The core mechanism is to leverage depth estimation and facial symmetry priors to compensate for the incomplete geometric and appearance information in monocular data, enabling high-quality 3DGS-based talking head animation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Talking head generation"
  - "3D Gaussian Splatting"
  - "Monocular training"
  - "Facial symmetry prior"
  - "Depth estimation"
  - "Generalizable"
date: 2026-05-08
content_hash: 8f9b398640489beb
---

# MGGTalk: Monocular and Generalizable Gaussian Talking Head Animation

**Conference**: CVPR 2025  
**arXiv**: [2504.00665](https://arxiv.org/abs/2504.00665)  
**Code**: [Project Homepage](https://scut-mmpr.github.io/MGGTalk-Homepage/)  
**Area**: 3D Vision  
**Keywords**: Talking head generation, 3D Gaussian Splatting, Monocular training, Facial symmetry prior, Depth estimation, Generalizable

## TL;DR

The MGGTalk framework is proposed, which can generalize to unseen identities using only monocular datasets for training. The core mechanism is to leverage depth estimation and facial symmetry priors to compensate for the incomplete geometric and appearance information in monocular data, enabling high-quality 3DGS-based talking head animation.

## Background & Motivation

One-shot talking head generation aims to synthesize realistic talking head videos from a single reference image and a driving source (audio or motion), which is widely applied in video dubbing, filmmaking, and video conferencing.

Existing methods are divided into two categories: (1) **2D generator methods** (GAN/diffusion models), which lack 3D facial modeling and are prone to facial distortion and identity inconsistency issues; (2) **3D rendering methods** (NeRF/3DGS), which offer higher modeling quality; however, NeRF is computationally expensive, and 3DGS-based methods either require multi-view datasets (which are difficult to acquire) or require personalized training for specific individuals (making them unable to generalize to new identities).

**Key Challenge**: How to build a 3DGS talking head system that can generalize to unseen identities using only monocular data (which is easy to acquire)? The main challenge lies in the fact that monocular data provides incomplete information, failing to provide the geometric and appearance information of invisible facial areas.

MGGTalk's key insight is to leverage two priors: (1) depth estimation to provide pixel-level geometric information; (2) left-right facial symmetry to supplement the geometry and texture of invisible areas.

## Method

### Overall Architecture

MGGTalk consists of three core stages: Given a segmented head image $\rightarrow$ (1) the **DSGR module** utilizes depth estimation and symmetry operations to generate visible-region point clouds and mirrored point clouds $\rightarrow$ (2) the **deformation network** adjusts point clouds based on expression features (from 3DMM of the driving image or audio encoding) $\rightarrow$ (3) the **SGP module** combines identity embeddings and the deformed point clouds to predict complete Gaussian parameters $\rightarrow$ render and synthesize the final image.

### Key Designs

**1. Depth-aware Symmetrical Geometric Reconstruction (DSGR)**

This module addresses the issue of "how to obtain complete 3D geometry from a monocular image." First, a pre-trained GeoWizard model is used to estimate the depth map and normal map from the input image, and a coarse point cloud is obtained through surface reconstruction via the BINI algorithm. Then, a 2D UNet-based geometric refinement network is introduced to learn depth offsets to correct errors in the initial depth estimation. For invisible facial areas (e.g., the occluded side during a profile view), the x-coordinates are flipped in the canonical pose space to perform a mirroring operation. Finally, a voxel filter is applied to remove the overlapping areas between the mirrored and original point clouds to avoid mutual interference.

**2. Symmetrical Gaussian Prediction (SGP)**

This module solves the problem of "how to generate reliable Gaussian parameters for invisible areas." A two-stage strategy is adopted: In the first stage, a Gaussian Decoder generates Gaussian parameters from the visible-region point cloud and identity embedding (supervised with labels for more precise learning); in the second stage, a Sym-Gaussian Decoder takes the Gaussian parameters from the first stage as additional input, combining identity embedding and the symmetric point cloud to predict the Gaussian parameters for invisible areas. This "guided-by-visible" strategy reduces the difficulty of prediction. Finally, parent-child node densification is applied to increase the density of Gaussian points and enhance rendering details.

**3. Deformation Network and Driving Mechanism**

Both video-driven and audio-driven modes are supported. For video driving, a 3DMM estimator is used to extract expression coefficients from the driving image; for audio driving, an audio-to-expression network encodes the audio into expression features. An MLP deformation network takes expression features as input and learns point cloud displacements to control facial expressions.

### Loss & Training

During training, the facial images are rendered before and after densification, respectively. A combination of three losses is utilized: L1 reconstruction loss, SSIM structural similarity loss ($\lambda_{ssim}=0.2$), and perceptual loss ($\lambda_p=0.01$). Losses are calculated for both rendering results before and after densification to ensure high-quality outputs in both stages.

## Key Experimental Results

### Main Results: Video-Driven Talking Head Generation (Table 1)

| Method | HDTF Self-PSNR↑ | HDTF Self-FID↓ | HDTF Cross-FID↓ | NeRS-Mono Self-FID↓ |
|------|:---:|:---:|:---:|:---:|
| Portrait4D-v2 | 30.12 | 36.57 | 42.82 | 54.95 |
| Real3DPortrait | 31.62 | 33.26 | 51.36 | 79.09 |
| DaGAN | 30.94 | 33.23 | 48.20 | 82.50 |
| **MGGTalk** | **32.40** | **18.95** | **27.85** | **51.35** |

MGGTalk significantly dominates in FID, achieving an HDTF self-reenactment FID of only 18.95 (compared to 33.23 for the runner-up) and a cross-reenactment FID of 27.85 (compared to 42.82 for the runner-up), demonstrating superior image quality and identity preservation.

### Ablation Study (Table 3, HDTF Self-Reenactment)

| Variant | PSNR↑ | FID↓ | AED↓ |
|------|:---:|:---:|:---:|
| w/o Geo. Refine. | 29.98 | 24.82 | 0.157 |
| w/o Sym. | 31.24 | 20.23 | 0.112 |
| w/o Gauss. Filter | 29.56 | 27.42 | 0.148 |
| w/o Sym. Gauss. Dec. | 32.12 | 19.74 | 0.116 |
| w/o Densify | 30.32 | 21.58 | 0.138 |
| **Full model** | **32.40** | **18.95** | **0.102** |

- Removing the voxel filter has the most significant impact (with FID dropping from 18.95 to 27.42), indicating that overlapping point cloud interference is severe.
- Removing geometric refinement causes the PSNR to drop by 2.4 dB, proving that depth correction is vital for quality.
- Removing densification leads to obvious Moiré patterns.

### Key Findings

- The volume of training data is less than 1/10 of that of other methods (about 400 + 300 video clips), but it remains competitive on in-the-wild data.
- The inference speed exceeds 40 FPS (on RTX 4090), supporting real-time generation.
- In the audio-driven scenario, a FID of 18.73 is achieved, which is far superior to all one-shot methods.
- In the user study, 45% of participants favored MGGTalk in terms of identity preservation.

## Highlights & Insights

1. **Symmetry prior is the key to unlocking monocular generalization**: The natural left-right symmetry of human faces is cleverly exploited, converting "invisible area completion" into "visible area mirroring + de-overlapping," thus avoiding dependency on multi-view datasets.
2. **Two-stage Gaussian prediction**: The strategy of "visible first, then invisible, and using the former to guide the latter" is natural, lowering the prediction difficulty for invisible areas.
3. **Importance of voxel filtering**: The ablation study shows that simple mirroring actions cause severe overlap; although voxel filtering is simple, it dramatically impacts final quality.

## Limitations & Future Work

- The symmetry assumption performs well under non-frontal angles, but during extreme profile views, the visible region is minimal, causing symmetric completion to yield limited information.
- It relies heavily on the accuracy of GeoWizard depth estimation; systematic errors in depth estimation will propagate to subsequent modules.
- Only the head region is modeled; the torso and background are handled by a separate inpainter, which may result in unnatural boundary artifacts.
- Non-symmetric areas such as hair and accessories are not explicitly handled; the symmetry assumption may fail in these regions.

## Related Work & Insights

- **GaussianAvatars/SplattingAvatar**: Binds 3DGS based on 3DMM meshes, requiring multi-view or video data; MGGTalk demonstrates that depth + symmetry is a more lightweight alternative.
- **GeoWizard depth estimation**: Provides pixel-level geometric priors for monocular 3D reconstruction.
- **Insight**: Symmetry is a specific prior for human faces, and generalizing to general objects requires exploring new "completion strategies"; this framework is easy to extend regarding the quality of audio-to-expression mapping.

## Rating

⭐⭐⭐⭐ — Achieving a substantial breakthrough in monocular talking head generation, the utilization of the symmetry prior is both elegant and effective, backed by comprehensive experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[ECCV 2024\] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](../../ECCV2024/3d_vision/talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)
- [\[CVPR 2026\] EmoDiffTalk: Emotion-aware Diffusion for Editable 3D Gaussian Talking Head](../../CVPR2026/3d_vision/emodifftalk_emotion-aware_diffusion_for_editable_3d_gaussian_talking_head.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2026\] EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization](../../CVPR2026/3d_vision/emotag_emotion-aware_talking_head_synthesis_on_gaussian_splatting_with_few-shot_.md)

</div>

<!-- RELATED:END -->
