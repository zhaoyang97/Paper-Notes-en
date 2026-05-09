---
title: >-
  [Paper Note] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction
description: >-
  [CVPR 2026][Remote Sensing][Spectral Compressed Imaging] The first work to advance spectral compressed imaging (SCI) from image-level to video-level reconstruction, introducing the first high-quality dynamic hyperspectral dataset DynaSpec (30 sequences / 300 frames), and proposing PG-SVRT with spatial-then-temporal attention plus bridge tokens that achieves 41.52 dB PSNR with optimal temporal consistency at lower FLOPs (28.18G) than several image-level SOTAs.
tags:
  - CVPR 2026
  - Remote Sensing
  - Spectral Compressed Imaging
  - Hyperspectral Video Reconstruction
  - Spatiotemporal Feature Propagation
  - Transformer
  - DynaSpec Dataset
date: 2026-05-08
content_hash: 7e25b7c34c33198a
---

# Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction

**Conference**: CVPR 2026  
**arXiv**: [2603.00611](https://arxiv.org/abs/2603.00611)  
**Code**: [DynaSpec](https://github.com/nju-cite/DynaSpec)  
**Area**: Computational Spectral Imaging  
**Keywords**: Spectral Compressed Imaging, Hyperspectral Video Reconstruction, Spatiotemporal Feature Propagation, Transformer, DynaSpec Dataset

## TL;DR

The first work to advance spectral compressed imaging (SCI) from image-level to video-level reconstruction, introducing the first high-quality dynamic hyperspectral dataset DynaSpec (30 sequences / 300 frames), and proposing PG-SVRT with spatial-then-temporal attention plus bridge tokens that achieves 41.52 dB PSNR with optimal temporal consistency at lower FLOPs (28.18G) than several image-level SOTAs.

## Background & Motivation

**State of the field**: Hyperspectral images (HSI) capture material spectral properties and are widely used in classification, detection, tracking, and autonomous driving. Spectral compressed imaging (SCI) compresses 3D data $X \in \mathbb{R}^{H \times W \times C}$ into 2D measurements $Y \in \mathbb{R}^{H \times W'}$ for snapshot acquisition via spatial-spectral encoding. Existing reconstruction methods (MST-L, DPU, RDLUF, etc.) have achieved excellent image-level performance.

**Existing limitations**: (1) **Reconstruction uncertainty**—mask encoding inevitably loses spatial-spectral information, and recovering occluded content from a single frame has inherent ambiguity; (2) **Temporal inconsistency**—frame-by-frame independent reconstruction cannot guarantee temporal continuity, manifesting as flickering intensity curves and inter-frame jitter, which fails to meet video perception requirements.

**Core tension**: Video-level reconstruction faces dual obstacles—**data scarcity** (existing datasets are all image-level; pseudo-video cropping lacks real motion freedom) and **algorithmic bottlenecks** (existing methods struggle to efficiently model high-dimensional spatiotemporal dependencies—joint attention has explosive complexity, while fully separated processing limits interaction).

**Objective**: To advance spectral reconstruction from image-level to video-level across data, model, and benchmark dimensions.

**Approach**: Fixed encoding patterns differentially capture complementary features across adjacent frames—occluded information can be recovered through propagation from neighboring frames, while naturally enhancing temporal consistency. This physical property provides a solid signal foundation for video-level reconstruction.

**Core idea**: Leveraging complementary features and temporal continuity in sequential measurements, efficient video-level hyperspectral reconstruction is achieved through spatial-then-temporal progressive attention with bridge tokens.

## Method

### Overall Architecture

PG-SVRT adopts a U-Net architecture, taking $T=3$ measurement frames as input, composed of three core modules: Mask-Guided Degradation Prior (MGDP), Cross-Domain Propagation Attention (CDPA), and Multi-Domain Feed-Forward Network (MDFFN). A shuffle operation aligns degraded features with measurements along the spectral dimension. Module counts $(N_1, N_2, N_3) = (4, 8, 8)$, with base channel number $C = N_\lambda = 30$.

### Key Designs

1. **DynaSpec Dataset**:

    - **Function**: Constructing the first high-quality dynamic hyperspectral image dataset
    - **Core idea**: A GaiaField push-broom hyperspectral camera captures frame-by-frame with controlled objects, manually designing translation/rotation/articulated motion to simulate real-world scene dynamics. Specifications: 30 scenes, 300 HSI frames, spatial resolution 1280×1280, spectral resolution 2 nm, wavelength range 400–700 nm (151 channels)
    - **Five quality principles**: (i) inter-frame motion is continuous and physically plausible; (ii) long exposure for noise reduction; (iii) spectral response calibration; (iv) illumination spectrum exclusion to approximate reflectance data; (v) invariant object intensity calibration to eliminate temperature drift
    - **Design motivation**: Existing datasets are either image-level (CAVE/KAIST) or downstream task datasets with unreliable low spectral resolution. Controlled frame-by-frame scanning ensures ground truth authenticity

2. **Cross-Domain Propagation Attention (CDPA)**:

    - **Function**: Spatial-then-temporal progressive attention for efficient spatiotemporal feature propagation
    - **Core idea**: **Spatial attention**—features are partitioned into non-overlapping windows ($H_{win}=8, W_{win}=32$), with bridge tokens $B_s \in \mathbb{R}^{Thw \times N_B \times C}$ (generated by pooling $Q_s$, $N_B=64$) serving as Q-K-V intermediaries without extra parameters: $Y_s^{out} = \text{GConv}(A(Q_s, B_s, A(B_s, K_s, V_s, \tau_1), \tau_2)) + Y_{N1}$. **Temporal attention**—after dimension rearrangement, **the spatial output is shared as value**: $Y_t^{out} = A(Q_t, K_t, Y_t, \tau_3)$, with no temporal windowing ($T$ is small and frames are strongly correlated)
    - **Complexity**: $O = 4THWC^2 + 4THWN_BC + 2T^2HWC$. Bridge tokens reduce cost when $2N_B < H_{win}W_{win}$ ($128 < 256$, satisfied)
    - **Design motivation**: Joint spatiotemporal attention is too expensive, while fully separated processing limits interaction. Shared value cross-domain propagation + bridge tokens achieve a balance between $O(N\log N)$-level efficiency and high-quality interaction

3. **Mask-Guided Degradation Prior (MGDP)**:

    - **Function**: Explicitly modeling the compression degradation process before the main architecture
    - **Core idea**: The mask $\Phi$ is compressed according to the SCI architecture (SD/DD) to obtain $\Phi_s$, cropped/replicated to $\Phi_p$, and the intensity distribution difference between $\Phi$ and $\Phi_p$ is learned (Conv$_{1\times1}$ + sigmoid) to produce weights $W_\Phi$, which are element-wise applied to measurement features and concatenated: $Y_{in} = \text{Concat}(\text{Conv}(W_m \odot F_m(Y)), Y)$
    - **Design motivation**: Degradation priors help the network understand the degree of encoding loss at each spatial-spectral location, guiding targeted reconstruction

### Loss Function / Training Strategy

Multi-stage RMSE loss; Adam optimizer ($\beta_1=0.9, \beta_2=0.999$); learning rate $3 \times 10^{-4}$ with cosine annealing to $1 \times 10^{-6}$; 80 epochs, batch size 2; RTX 3090 GPU. Unified comparison across 4 SCI systems (SD-CASSI/DD-CASSI/PMVIS/NDSSI).

## Key Experimental Results

### Main Results — Comparison with SOTAs (DD-CASSI System)

| Method | Venue | PSNR-K↑ | PSNR-D↑ | SAM-K↓ | ST-RRED-K↓ | GFLOPs |
|--------|-------|---------|---------|--------|-----------|--------|
| MST-L | CVPR'22 | 39.99 | 39.58 | 3.82 | 30.99 | 28.23 |
| PADUT | ICCV'23 | 38.61 | 40.41 | 4.72 | 47.19 | 32.78 |
| DPU | CVPR'24 | 40.02 | 41.01 | 5.22 | 25.90 | 31.04 |
| DPU* (with temporal) | CVPR'24 | 40.50 | 41.36 | 5.17 | 26.71 | 77.36 |
| **PG-SVRT** | **Ours** | **41.23** | **41.82** | **3.81** | **19.35** | **28.18** |

### Ablation Studies

| Configuration | PSNR | SSIM | SAM↓ | ST-RRED↓ | GFLOPs |
|--------------|------|------|------|----------|--------|
| Baseline (F-MSA+FFN) | 39.97 | 0.9827 | 5.53 | 43.90 | 30.11 |
| + CDPA | 41.30 (+1.33) | 0.9884 | 4.32 | 25.44 | 21.11 |
| + CDPA + MGDP | 41.41 (+0.11) | 0.9886 | 4.25 | 24.63 | 21.31 |
| + CDPA + MGDP + MDFFN | **41.52** (+0.11) | **0.9893** | **3.91** | **23.25** | 28.18 |

### Key Findings

- DD-CASSI dominates among the four SCI architectures (PSNR 41.52 vs. runner-up NDSSI 37.84), owing to its high spectral sampling efficiency and clear structural representation
- CDPA contributes the most (+1.33 dB PSNR) while actually reducing FLOPs (30.11→21.11G), as bridge tokens replace full-window attention
- Spatial-then-temporal + shared value strategy achieves the best result (41.52), outperforming parallel processing (41.35) and temporal-then-spatial (41.04)
- Despite being a video model, PG-SVRT's per-frame FLOPs (28.18G) are lower than image-level methods such as DAUHST (35.93G)

## Highlights & Insights

- **Data + model + benchmark trinity**: The DynaSpec dataset, PG-SVRT model, and four-SCI-system comparison benchmark together provide substantial momentum for the dynamic computational spectral imaging field
- **Elegant bridge token design**: Pooling queries to generate intermediary tokens enables indirect attention with zero extra parameters and reduced complexity. Computation is strictly reduced when $2N_B < H_{win}W_{win}$
- **Shared value cross-domain propagation**: The spatial attention output directly serves as the temporal attention value, elegantly enabling multi-domain feature interaction without additional projection overhead
- **Compelling DPU* comparison**: Naively concatenating temporal frames costs significantly more (77.36G) than PG-SVRT (28.18G) with inferior performance

## Limitations & Future Work

- DynaSpec contains only 30 scenes / 300 frames; limited diversity and scale may lead to overfitting to specific motion patterns
- Frame count is fixed at $T=3$; extension to longer sequences is not validated, and real dynamic scenarios may require larger temporal windows
- Training uses 256×256 crops; full-resolution (1280×1280) inference efficiency and effectiveness are not discussed
- Explicit motion modeling methods (optical flow alignment, deformable convolutions) combined with CDPA remain unexplored

## Related Work & Inspiration

- **vs. DPU (CVPR'24)**: Image-level SOTA; naively concatenating temporal frames (DPU*) causes a 2.5× FLOPs explosion but limited improvement (+0.48 dB). PG-SVRT elegantly achieves video-level reconstruction via shared value propagation at lower FLOPs
- **vs. MST-L/CST-L**: Earlier image-level methods show far worse temporal consistency (ST-RRED 30–35) compared to PG-SVRT (19.35)
- The bridge token concept is generalizable to other attention mechanism designs requiring efficient processing of high-dimensional data
- Fair comparison across SCI systems provides important reference for spectral imaging hardware selection (DD-CASSI is clearly superior)

## Rating

- Novelty: ⭐⭐⭐⭐ Video-level spectral reconstruction is a novel problem formulation; CDPA bridge tokens and shared value propagation are creative designs
- Experimental rigor: ⭐⭐⭐⭐⭐ Comparison across four SCI systems, 12 SOTAs, multi-dimensional ablations, and real prototype validation
- Writing quality: ⭐⭐⭐⭐ Clear problem motivation with a complete unified SCI mathematical framework
- Impact: ⭐⭐⭐⭐⭐ The combination of dataset + method + benchmark has far-reaching impact on the dynamic computational spectral imaging field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction](../../AAAI2026/remote_sensing/m3sr_multi-scale_multi-perceptual_mamba_for_efficient_spectral_reconstruction.md)
- [\[CVPR 2026\] Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[CVPR 2026\] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](sdfnet_structureaware_disentangled_feature_learnin.md)
- [\[CVPR 2026\] No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)

</div>

<!-- RELATED:END -->
