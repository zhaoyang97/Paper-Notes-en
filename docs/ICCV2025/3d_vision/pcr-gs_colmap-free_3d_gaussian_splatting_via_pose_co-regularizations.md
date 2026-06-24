---
title: >-
  [Paper Note] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] PCR-GS is proposed to achieve high-quality 3D-GS reconstruction and pose estimation under complex camera trajectories without COLMAP priors, via co-regularizing camera poses through DINO feature reprojection regularization and wavelet-based frequency regularization.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "COLMAP-Free"
  - "Camera Pose Estimation"
  - "DINO Features"
  - "Wavelet Transform"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 3c421e447638a086
---

# PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations

**Conference**: ICCV 2025
**arXiv**: [2507.13891](https://arxiv.org/abs/2507.13891)  
**Authors**: Yu Wei, Jiahui Zhang, Xiaoqin Zhang, Ling Shao, Shijian Lu (NTU, ZJUT, UCAS)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, COLMAP-Free, Camera Pose Estimation, DINO Features, Wavelet Transform, Novel View Synthesis

## TL;DR

PCR-GS is proposed to achieve high-quality 3D-GS reconstruction and pose estimation under complex camera trajectories without COLMAP priors, via co-regularizing camera poses through DINO feature reprojection regularization and wavelet-based frequency regularization.

## Background & Motivation

3D Gaussian Splatting (3D-GS) achieves remarkable performance in novel view synthesis but relies heavily on accurate camera poses provided by COLMAP. COLMAP itself is computationally expensive and prone to failure in scenes with sparse textures or repetitive patterns. Existing COLMAP-free methods (e.g., CF-3DGS) jointly optimize relative poses between adjacent frames using RGB photometric loss, but suffer from the following key issues:

**Failure under complex camera trajectories**: When adjacent frames involve large rotations and translations, the overlapping regions between neighboring views are limited, making RGB-based alignment unreliable and leading to inaccurate relative pose estimation.

**Local optima traps**: Inaccurate relative poses cause the joint optimization of poses and 3D-GS to fall into local optima, producing severe artifacts and blurring.

**Difficulty in rotation matrix optimization**: Small errors in the rotation matrix of camera poses lead to spatial misalignment of geometry and texture, whereas regularization in RGB space is insensitive to such structural shifts.

These issues are prevalent in real-world scenarios where handheld capture typically involves large camera motions. PCR-GS is motivated by co-regularizing camera poses from two complementary perspectives — **semantic feature alignment** and **frequency-domain analysis** — to overcome the performance bottleneck under complex trajectories.

## Method

### Overall Architecture

PCR-GS builds upon CF-3DGS, retaining its progressive optimization strategy (frame-by-frame expansion of 3D Gaussians), while introducing two core regularization modules operating in parallel:

1. **Feature Reprojection Regularization (FRR)**: Utilizes DINO semantic features for cross-view alignment.
2. **Wavelet-based Frequency Regularization (WFR)**: Captures rotation errors in the frequency domain.

The total loss function is:

$$\mathcal{L} = \lambda_0 \mathcal{L}_{\text{rgb}} + \lambda_1 \mathcal{L}_{\text{feat}} + \lambda_2 \mathcal{L}_{\text{freq}}$$

where $\lambda_0=0.6$, $\lambda_1=0.2$, $\lambda_2=0.2$.

### Key Design 1: Feature Reprojection Regularization (FRR)

**Mechanism**: DINO features are robust to viewpoint changes — point correspondences in DINO feature space remain stable even under large rotations. Accordingly, DINO semantic features are used in place of unstable RGB information to constrain relative poses.

**Pipeline**:

- A pretrained DINO model (layer 9 features) is used to extract semantic feature maps $F_i$, $F_{i+1}$ for each frame.
- Depth maps are rendered using the trained 3D Gaussians $G_i^*$ to obtain per-pixel depth values.
- 2D pixels of frame $I_i$ are back-projected to 3D camera coordinates via depth, transformed to the coordinate system of frame $I_{i+1}$ using transformation matrix $T_i$, and projected back to 2D.
- The L2 difference between DINO features at original and reprojected locations is minimized:

$$\mathcal{L}_{\text{feat}} = \|F_i\langle P_i\rangle - F_{i+1}\langle \mathbf{K} P_i T_i \rangle\|_2$$

**Pose Initialization Strategy**:

- A foreground mask is constructed from the DINO feature map's saliency map to filter out background regions.
- Sparse correspondences between foreground keypoints are established using the Best Buddies algorithm.
- $N_s=20$ sparse correspondences are randomly selected to optimize the initial relative pose (replacing identity matrix initialization).
- This step effectively reduces the risk of subsequent optimization falling into local optima.

### Key Design 2: Wavelet-based Frequency Regularization (WFR)

**Mechanism**: Rotation errors cause spatial misalignment of edges and textures, which manifests prominently in high-frequency details. RGB-space supervision primarily captures pixel intensity changes and is insensitive to structural shifts induced by rotation. Frequency-domain regularization is better suited to capture such errors.

**Pipeline**:

- Wavelet transforms are applied to both the rendered image and the ground-truth image, decomposing them into four components: $LL$ (low-frequency), $LH$ (horizontal high-frequency), $HL$ (vertical high-frequency), and $HH$ (diagonal high-frequency).
- A weighted Euclidean distance across components is computed:

$$d = \sum_{x \in \{LL, LH, HL, HH\}} w_x \|W_x(I_t) - W_x(\hat{I}_t)\|$$

**Annealing Strategy**:

- Directly optimizing high-frequency components tends to introduce noise; thus, a progressive strategy from low to high frequency is designed:

$$\mathcal{L}_{\text{freq}} = \begin{cases} d_{LL} & 0 < n \leq 100 \\ (1-w_h)d_{LL} + w_h d_H & 100 < n \leq 200 \\ d_H & n > 200 \end{cases}$$

- Only low-frequency components are optimized in the first 100 iterations; the weight on high-frequency components linearly increases between iterations 100–200; only high-frequency components are optimized after iteration 200.
- The advantage of wavelet transforms lies in preserving spatial location information in high-frequency details, unlike global frequency transforms such as FFT.

### Loss & Training

- **$\mathcal{L}_{\text{rgb}}$**: Standard photometric loss $= (1-\lambda)\|I - \hat{I}\| + \lambda \mathcal{L}_{\text{D-SSIM}}$, with $\lambda=0.2$.
- **$\mathcal{L}_{\text{feat}}$**: DINO feature reprojection L2 loss.
- **$\mathcal{L}_{\text{freq}}$**: Wavelet frequency-domain annealing loss.

## Key Experimental Results

### Main Results: Novel View Synthesis (Tanks&Temples, mean over 8 scenes)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| NeRFmm | 14.10 | 0.36 | 0.66 |
| BARF | 14.05 | 0.42 | 0.69 |
| Nope-NeRF | 21.95 | 0.57 | 0.52 |
| CF-3DGS | 19.79 | 0.60 | 0.33 |
| **PCR-GS** | **23.68** | **0.73** | **0.23** |

PCR-GS surpasses CF-3DGS by **+3.89 dB** in PSNR and reduces LPIPS by 30%.

### Main Results: Novel View Synthesis (Free-Dataset, mean over 3 scenes)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| CF-3DGS | 15.00 | 0.37 | 0.55 |
| **PCR-GS** | **17.78** | **0.49** | **0.46** |

PSNR improves by **+2.78 dB** on Free-Dataset.

### Main Results: Pose Estimation (Tanks&Temples, mean over 8 scenes)

| Method | RPE_t↓ | RPE_r↓ | ATE↓ |
|--------|--------|--------|------|
| NeRFmm | 8.261 | 1.950 | 0.446 |
| BARF | 7.641 | 2.121 | 0.436 |
| Nope-NeRF | 3.519 | 0.751 | 0.403 |
| CF-3DGS | 0.211 | 0.520 | 0.013 |
| **PCR-GS** | **0.109** | **0.350** | **0.008** |

PCR-GS outperforms all baselines in pose accuracy, reducing RPE_t by 48% and ATE by 38% relative to CF-3DGS.

### Ablation Study (Tanks&Temples - Horse scene)

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|-------|-------|--------|
| Base (CF-3DGS) | 18.34 | 0.64 | 0.32 |
| Base+FRR | 23.16 | 0.72 | 0.17 |
| Base+WFR (w/o high-freq) | 18.40 | 0.65 | 0.32 |
| Base+WFR (w/ high-freq) | 19.31 | 0.66 | 0.28 |
| **Base+FRR+WFR (PCR-GS)** | **24.20** | **0.79** | **0.17** |

### Key Findings

1. **FRR contributes the most**: Adding FRR alone yields a +4.82 dB PSNR gain, confirming that DINO feature alignment is the core improvement.
2. **High-frequency components in WFR are critical**: WFR without high-frequency components is nearly ineffective (+0.06 dB), whereas including them yields a clear improvement (+0.97 dB), validating that high-frequency details effectively capture rotation errors.
3. **FRR and WFR are complementary**: Adding WFR on top of FRR yields a further +1.04 dB gain, demonstrating significant synergistic effects.
4. **Greater advantage under complex trajectories**: Experiments intentionally reduce the Tanks&Temples sampling rate from 20fps to 4fps to increase camera motion complexity; CF-3DGS degrades sharply while PCR-GS maintains strong performance.

## Highlights & Insights

1. **Precise problem formulation**: The paper accurately identifies the core bottleneck of COLMAP-free 3D-GS under complex camera trajectories — RGB alignment fails under large viewpoint changes — and proposes solutions from two complementary perspectives: semantic features and frequency domain.
2. **Sophisticated use of DINO features**: Rather than naively incorporating a DINO loss, a complete feature reprojection pipeline is designed (3D back-projection → transformation → 2D projection → feature sampling alignment), with DINO saliency maps additionally used for initialization, forming a coherent and complete scheme.
3. **Frequency-domain annealing strategy**: The coarse-to-fine design from low to high frequency demonstrates strong engineering insight, preventing high-frequency noise from interfering with training.
4. **Rigorous experimental design**: Rather than adopting the overly idealized 20fps smooth trajectories from CF-3DGS, the authors construct more challenging 4fps sampled data, yielding evaluations that better reflect real-world application conditions.
5. **Well-motivated choice of wavelet transform**: Compared to global frequency transforms such as FFT, wavelet transforms preserve spatial location information, making them more suitable for detecting local structural shifts caused by rotation.

## Limitations & Future Work

1. **Restricted to video sequences**: The method is based on progressive optimization between adjacent frames and is not applicable to unordered image collection reconstruction scenarios.
2. **Reliance on monocular depth estimation**: DPT is used to initialize point clouds; inaccuracies in monocular depth may degrade feature reprojection quality.
3. **DINO feature extraction overhead**: Per-frame DINO feature extraction incurs additional computational and memory costs, though the paper provides no concrete runtime comparison.
4. **Fixed annealing hyperparameters**: $n_0=100$ and $n_1=200$ are manually set hyperparameters; whether these require adjustment across different scenes is not discussed.
5. **Absence of comparisons with more recent methods**: No comparison is made against recent pose-free reconstruction methods such as InstantSplat or DUSt3R.
6. **Limited scene scale**: Experiments are conducted primarily on medium-scale indoor and outdoor scenes; generalization to large-scale scenarios (e.g., street-level or city-scale) is not validated.

## Related Work & Insights

1. **CF-3DGS [Fu et al., CVPR 2024]**: The direct baseline of this work and the first pose-free 3D-GS method, which progressively estimates relative poses. PCR-GS augments it with two regularization modules.
2. **BARF [Lin et al., ICCV 2021]**: Progressive joint pose-NeRF optimization using a coarse-to-fine positional encoding strategy. The coarse-to-fine philosophy is reflected in the annealing strategy of this paper.
3. **Nope-NeRF [Bian et al., CVPR 2023]**: Constrains relative poses using distortion-free depth priors. Achieves better NVS quality than CF-3DGS under complex trajectories but exhibits inferior pose accuracy.
4. **DINO [Caron et al., ICCV 2021]**: Self-supervised ViT features that are robust to viewpoint changes. The cross-view consistency of DINO features is central to this paper.
5. **Future directions**: Stronger foundation model features such as DINOv2 or SAM could be explored for pose regularization; the frequency-domain regularization idea could be extended to detect and disentangle object motion in dynamic scenes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both regularization modules are thoughtfully designed; DINO feature reprojection and wavelet frequency annealing are not trivial additions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 11 scenes, dual evaluation of NVS and pose, detailed ablation; however, comparisons with more recent methods and efficiency analysis are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete formulations, intuitive figures, and well-articulated motivation.
- **Value**: ⭐⭐⭐⭐ — Addresses a practical bottleneck of COLMAP-free 3D-GS under complex trajectories with clear application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](../../CVPR2025/3d_vision/selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](../../ECCV2024/3d_vision/cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
