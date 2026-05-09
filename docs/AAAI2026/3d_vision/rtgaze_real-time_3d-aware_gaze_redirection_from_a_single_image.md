---
title: >-
  [Paper Note] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image
description: >-
  [AAAI 2026][3D Vision][Gaze redirection] This paper proposes RTGaze, a real-time 3D-aware gaze redirection method that achieves high-quality gaze redirection from a single image at 61 ms/frame via a hybrid-frequency feature encoder, a gaze injection module, and 3D facial geometry prior distillation — approximately 800× faster than the previous state-of-the-art 3D method, GazeNeRF.
tags:
  - AAAI 2026
  - 3D Vision
  - Gaze redirection
  - 3D-aware
  - NeRF
  - knowledge distillation
  - real-time inference
date: 2026-05-08
content_hash: 8caf2c88c887a5f5
---

# RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image

**Conference**: AAAI 2026
**arXiv**: [2511.11289](https://arxiv.org/abs/2511.11289)
**Code**: Unavailable
**Area**: 3D Vision
**Keywords**: Gaze redirection, 3D-aware, NeRF, knowledge distillation, real-time inference

## TL;DR

This paper proposes RTGaze, a real-time 3D-aware gaze redirection method that achieves high-quality gaze redirection from a single image at 61 ms/frame via a hybrid-frequency feature encoder, a gaze injection module, and 3D facial geometry prior distillation — approximately 800× faster than the previous state-of-the-art 3D method, GazeNeRF.

## Background & Motivation

### State of the Field

Gaze redirection refers to generating facial images with controllable gaze direction while preserving subject identity. It has broad applications in VR/AR, digital humans, and CG film production (e.g., eye-contact correction in video conferencing).

Existing methods fall into two categories:
- **2D methods** (ST-ED, FAZE, ReDirTrans): Generate target images directly via GAN/VAE/encoder-decoder architectures. These are fast but **lack 3D consistency**, perform poorly under large head pose variations, and cannot model the inherently 3D nature of gaze redirection.
- **3D methods** (EyeNeRF, GazeNeRF, HeadNeRF): Build 3D facial representations based on NeRF, naturally achieving 3D consistency. However, inference requires a **GAN inversion process** (~60 seconds), making real-time deployment infeasible.

### Limitations of Prior Work

The core tension lies in the **trade-off between 3D consistency and real-time performance**.

- GazeNeRF requires ~60 seconds per frame (encoding 60 s + rendering 0.06 s); the encoding stage optimizes learnable latent codes via inversion on the input image, forming the speed bottleneck.
- 2D methods are fast but produce artifacts under large pose angles and suffer from poor identity preservation.
- No existing method simultaneously achieves 3D consistency, image quality, and real-time performance.

### Starting Point

The core idea is to **replace GAN inversion with a feedforward network**. The model directly encodes a triplane representation from a single image and a gaze label, bypassing the costly optimization process. Facial geometry priors are distilled from a pretrained 3D portrait generator to compensate for the under-constrained nature of single-image 3D inference.

## Method

### Overall Architecture

**Input**: A single facial image $\mathbf{I}$ and a target gaze direction $\mathbf{g} \in \mathbb{R}^2$ (pitch and yaw)
**Output**: A gaze-redirected facial image $\hat{\mathbf{I}}$

**Pipeline**:
1. A hybrid image encoder extracts high-frequency and low-frequency features.
2. A gaze injection module fuses the gaze prompt into high-frequency features.
3. A triplane decoder generates the 3D facial representation.
4. Neural rendering produces the final image.

$$f = \mathcal{F}(\mathbf{I}, \mathbf{g}), \quad \mathbf{T} = \mathcal{G}(f), \quad \hat{I} = \mathcal{N}(\mathbf{T}, \mathbf{c})$$

### Key Designs

#### 1. Hybrid-Frequency Facial Feature Encoder

**Function**: Disentangles and extracts high-frequency (appearance details) and low-frequency (global geometry) features from the input image.

**Mechanism**:
- **Low-frequency encoder $\mathcal{F}_l$**: Extracts global semantic information using ImageNet-pretrained DeepLabV3, refined by a Vision Transformer encoder to produce low-frequency features $z_l$.
- **High-frequency encoder $\mathcal{F}_h$**: A CNN extracts fine-grained appearance details (texture, hair, etc.) to produce high-frequency features $z_h$.

**Design Motivation**: Gaze redirection primarily affects eye appearance (high-frequency changes), while global facial geometry (low-frequency) remains relatively stable. Disentangled encoding allows the gaze prompt to be precisely injected into the appropriate feature branch.

#### 2. Gaze Prompt Injection Module

**Function**: Injects the target gaze direction into the facial representation.

**Mechanism**:
- The gaze prompt $\mathbf{g}$ (pitch + yaw) is embedded via an MLP to match the length of the high-frequency features.
- **Cross-attention** is used for injection: high-frequency features serve as queries, while the gaze embedding serves as both keys and values.
- The injected high-frequency features are fused with low-frequency features to form the final gaze-controllable facial representation.

**Key Decision**: Ablation experiments confirm that the gaze prompt must be injected into **high-frequency features** rather than low-frequency features. Injecting into low-frequency features degrades FID to 67.298 (vs. 38.346) and raises gaze error from 9.047° to 18.973°, validating the assumption that gaze changes are primarily appearance-level phenomena.

#### 3. Face Geometric Prior Distillation

**Function**: Distills facial geometry knowledge from a pretrained 3D portrait generation model to enhance single-image 3D inference quality.

**Mechanism**:
- A pretrained 3D portrait generator (Trevithick et al. 2023) serves as the teacher.
- A frontal image of the same identity and gaze is fed to the teacher to obtain its triplane features.
- Depth maps $\mathbf{D}^t$ and $\mathbf{D}^s$ are rendered from the teacher and student models, respectively.
- An L1 depth distillation loss is applied:

$$\mathcal{L}_{\mathcal{D}} = \|\mathbf{D}^t - \mathbf{D}^s\|_1$$

**Design Motivation**:
- Single-image 3D reconstruction is an under-constrained problem with insufficient constraints for accurate 3D recovery.
- Although the teacher model is not designed for gaze redirection, its facial geometry knowledge (depth structure) is transferable.
- Only depth is distilled — not color — since the teacher's synthesized appearance may not align with the target image.

### Loss & Training

**Total loss**:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\mathcal{R}} + \beta \cdot \mathcal{L}_{\mathcal{D}} + \gamma \cdot \mathcal{L}_{\mathcal{P}}$$

where $\alpha = 1, \beta = 1, \gamma = 0.8$.

- **Mask-guided reconstruction loss** $\mathcal{L}_{\mathcal{R}}$: Separately weighted for facial and eye regions; eye region weight $\alpha_2 = 2$ (face region $\alpha_1 = 1$), computed with L1 norm.
- **Distillation loss** $\mathcal{L}_{\mathcal{D}}$: L1 loss on depth maps.
- **Perceptual loss** $\mathcal{L}_{\mathcal{P}}$: L1 distance across multiple VGG16 feature layers.

**Training Details**:
- AdamW optimizer; learning rate $10^{-5}$ for both encoder and renderer.
- Batch size 4, 50 epochs.
- ETH-XGaze training set: 10 frames/subject × 18 views/frame × 80 subjects.
- 2 × NVIDIA A100 (40 GB), ~18 hours.
- Input resolution 512×512, preprocessed following EG3D conventions.

## Key Experimental Results

### Main Results

**ETH-XGaze — Image Quality and Inference Speed**:

| Method | 3D | FID↓ | PSNR↑ | LPIPS↓ | SSIM↑ | Total Time↓ |
|--------|----|------|-------|--------|-------|------------|
| ST-ED | × | 115.020 | 17.530 | 0.300 | **0.726** | - |
| HeadNeRF | ✓ | 69.487 | 15.298 | 0.294 | 0.720 | 60.058s |
| GazeNeRF | ✓ | 81.816 | 15.453 | 0.291 | **0.733** | 60.060s |
| **RTGaze** | **✓** | **38.346** | **19.007** | **0.262** | 0.715 | **0.061s** |

- FID is substantially superior (38.3 vs. the next best 69.5); PSNR and LPIPS are also best.
- **Inference speed is 0.061 s, approximately 1000× faster than GazeNeRF** (60 s → 0.061 s).
- SSIM is slightly below GazeNeRF (0.715 vs. 0.733), but RTGaze leads comprehensively on all other metrics.

**Cross-Dataset Generalization** (gaze/head error + identity preservation):

| Dataset | Method | LPIPS↓ | ID↑ | Gaze↓ | Head↓ |
|---------|--------|--------|-----|-------|-------|
| ColumbiaGaze | GazeNeRF | 0.352 | 23.157 | 9.464 | 3.811 |
| ColumbiaGaze | **RTGaze** | **0.249** | **61.765** | **7.625** | **3.326** |
| MPIIFaceGaze | GazeNeRF | 0.272 | 30.981 | 14.933 | 7.118 |
| MPIIFaceGaze | **RTGaze** | **0.251** | **46.098** | **9.409** | **6.444** |

RTGaze outperforms GazeNeRF on all metrics across both ColumbiaGaze and MPIIFaceGaze, demonstrating strong cross-dataset generalization. The particularly large gain in identity preservation (ID: 23→62, 31→46) suggests that the inversion process itself is a source of identity loss.

### Ablation Study

**Gaze Injection Location**:

| Injection Target | FID↓ | ID↑ | Gaze↓ | Head↓ |
|-----------------|------|-----|-------|-------|
| Low-frequency features | 67.298 | 38.517 | 18.973 | 5.409 |
| **High-frequency features** | **38.346** | **60.708** | **9.047** | **3.631** |

Injecting into low-frequency features causes a dramatic performance collapse: gaze error doubles (9°→19°), FID nearly doubles, and identity preservation deteriorates sharply.

**Loss Function Ablation**:

| Configuration | FID↓ | ID↑ | Gaze↓ | Head↓ |
|--------------|------|-----|-------|-------|
| $\mathcal{L}_\mathcal{R}$ only | 101.053 | 47.251 | 9.332 | 4.208 |
| $\mathcal{L}_\mathcal{R} + \mathcal{L}_\mathcal{P}$ | 54.682 | 52.518 | 10.911 | 3.700 |
| $\mathcal{L}_\mathcal{R} + \mathcal{L}_\mathcal{P} + \mathcal{L}_\mathcal{D}$ | **38.346** | **60.708** | **9.047** | **3.631** |

The perceptual loss reduces FID from 101 to 55; the distillation loss further lowers it to 38, while identity preservation improves from 47 to 61. The contribution of 3D geometric prior distillation is substantial.

### Key Findings

1. A feedforward approach can fully replace GAN inversion — achieving ~1000× speedup while also improving image quality (FID 38 vs. 82).
2. The large gain in identity preservation indicates that the inversion process itself is likely the primary cause of identity degradation.
3. 3D geometric prior distillation is critical for quality improvement, even when the teacher model is not designed for gaze redirection.
4. Gaze changes are fundamentally high-frequency appearance-level variations; low-frequency geometric features should not be directly modified.

## Highlights & Insights

1. **Speed breakthrough**: RTGaze is the first 3D-aware method to achieve real-time performance (61 ms), making 3D gaze redirection practically deployable in video conferencing, VR, and similar applications.
2. **High/low frequency disentanglement**: The design reflects a deep understanding of the task — gaze alters eye texture (high frequency) without changing facial bone structure (low frequency) — which directly motivates the correct architecture.
3. **Elegant distillation strategy**: Distilling only depth and not color avoids the appearance inconsistency between teacher and student outputs.
4. **Cross-attention injection**: Using cross-attention rather than simple concatenation or addition for gaze prompt injection enables more flexible feature interaction.

## Limitations & Future Work

1. **Slightly lower SSIM**: On ETH-XGaze, SSIM (0.715) falls below GazeNeRF (0.733), indicating room for improvement in pixel-level alignment.
2. **Dependency on frontal images during training**: The distillation process requires frontal images of the same identity and gaze as the target, constraining the training data pipeline.
3. **Resolution limitation**: The 512×512 input resolution may be insufficient to capture very fine eye details such as iris texture.
4. **Extreme pose not evaluated**: Although 3D methods are inherently more robust, the paper does not present results under extreme head pose or large viewing angles.
5. **Code not released** (as of the time of writing), limiting reproducibility.

## Related Work & Insights

- **EG3D** (CVPR 2022): The seminal work on triplane representations; RTGaze's triplane decoder is built upon this foundation.
- **Trevithick et al. 2023**: The 3D portrait generator serving as the teacher for geometry prior distillation.
- Advances in single-image-to-3D methods (e.g., Zero-1-to-3) will continue to lower the barrier to 3D facial modeling.
- The paradigm of "feedforward networks replacing iterative optimization" is generalizable to other NeRF-based applications such as expression editing and style transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of feedforward replacement of inversion, high/low frequency disentanglement, and distillation strategy constitutes effective compositional innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluation spans three datasets with efficiency comparisons, multi-dimensional metrics, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rich quantitative results; some implementation details are slightly underspecified.
- **Value**: ⭐⭐⭐⭐⭐ — Real-time 3D gaze redirection has significant practical value; the speed breakthrough represents a milestone contribution.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](../../ICCV2025/3d_vision/gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[AAAI 2026\] PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos](pfavatar_pose-fusion_3d_personalized_avatar_reconstruction_from_real-world_outfi.md)

<!-- RELATED:END -->
