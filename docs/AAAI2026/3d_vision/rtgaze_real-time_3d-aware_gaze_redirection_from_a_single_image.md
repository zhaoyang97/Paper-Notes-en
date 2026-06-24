---
title: >-
  [Paper Note] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image
description: >-
  [AAAI 2026][3D Vision][Gaze Redirection] RTGaze is proposed, a real-time 3D-aware gaze redirection method. By utilizing a hybrid-frequency feature encoder, a gaze injection module, and 3D facial geometric prior distillation, it achieves high-quality gaze redirection from a single image at 61ms/frame, which is over 800 times faster than the previous SOTA 3D method (GazeNeRF).
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Gaze Redirection"
  - "3D-Aware"
  - "NeRF"
  - "Knowledge Distillation"
  - "Real-Time Inference"
date: 2026-05-08
content_hash: 981ca211765a05ac
---

# RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image

**Conference**: AAAI 2026  
**arXiv**: [2511.11289](https://arxiv.org/abs/2511.11289)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Gaze Redirection, 3D-Aware, NeRF, Knowledge Distillation, Real-Time Inference

## TL;DR

RTGaze is proposed, a real-time 3D-aware gaze redirection method. By utilizing a hybrid-frequency feature encoder, a gaze injection module, and 3D facial geometric prior distillation, it achieves high-quality gaze redirection from a single image at 61ms/frame, which is over 800 times faster than the previous SOTA 3D method (GazeNeRF).

## Background & Motivation

### Background

Gaze redirection refers to historical/facial image generation with controllable eye-gaze directions while preserving identity. It is widely applied in fields like VR/AR, digital humans, and CG movie production (e.g., eye contact correction in videoconferencing).

Existing methods are divided into two main categories:
- **2D Methods** (ST-ED, FAZE, ReDirTrans): Direct target image generation via GAN/VAE/encoder-decoder. They are fast but **lack 3D consistency**, performing poorly under large head poses, and fail to model the 3D nature of gaze redirection.
- **3D Methods** (EyeNeRF, GazeNeRF, HeadNeRF): Built upon NeRF to construct 3D facial representations, naturally offering 3D consistency. However, inference requires a **GAN inversion process** (~60 seconds), making real-time performance completely unfeasible.

### Limitations of Prior Work

Key Challenge: **The trade-off between 3D consistency and real-time performance**.

- GazeNeRF requires about 60 seconds/frame for inference (60s encoding + 0.06s rendering). During the encoding stage, inversion optimization on the input image is required to optimize learnable latent codes, presenting a major speed bottleneck.
- 2D methods, while fast, suffer from artifacts under large pose angles and poor identity preservation.
- No existing method simultaneously achieves 3D consistency, high image quality, and real-time inference.

### Key Insight

Core Idea: **Replacing GAN inversion with a feedforward network**. It directly encodes triplane representations from a single image and gaze labels, bypassing the time-consuming optimization process. Concurrently, facial geometric priors are distilled from a pre-trained 3D portrait generator to compensate for the source under-constraint in single-image 3D inference.

## Method

### Overall Architecture

Input: A single facial image $\mathbf{I}$ + target gaze direction $\mathbf{g} \in \mathbb{R}^2$ (pitch and yaw)  
Output: Gaze-redirected facial image $\hat{\mathbf{I}}$  

Pipeline:
1. A hybrid image encoder extracts high-frequency and low-frequency features.
2. A gaze injection module integrates the gaze prompt into the high-frequency features.
3. A triplane decoder generates 3D facial representations.
4. Neural rendering synthesizes the final image.

$$f = \mathcal{F}(\mathbf{I}, \mathbf{g}), \quad \mathbf{T} = \mathcal{G}(f), \quad \hat{I} = \mathcal{N}(\mathbf{T}, \mathbf{c})$$

### Key Designs

#### 1. Hybrid-Frequency Facial Feature Encoder

**Function**: Separates and extracts high-frequency (appearance details) and low-frequency (global geometry) features from the input image.

**Mechanism**:
- **Low-Frequency Encoder $\mathcal{F}_l$**: Extracts global semantic information using ImageNet pre-trained DeepLabV3 → Refines global features via a Vision Transformer encoder → Yields low-frequency features $z_l$.
- **High-Frequency Encoder $\mathcal{F}_h$**: Extracts fine appearance details (textures, hair, etc.) using a CNN → Yields high-frequency features $z_h$.

**Design Motivation**: Gaze redirection primarily affects eye appearance (high-frequency variations), while global facial geometry (low-frequency) remains relatively stable. Separating encoding allows the gaze prompt to be precisely injected into the appropriate feature level.

#### 2. Gaze Prompt Injection Module

**Function**: Injects the target gaze direction into the facial representation.

**Mechanism**:
- Embeds the gaze prompt $\mathbf{g}$ (pitch + yaw) through an MLP to ensure its length matches the high-frequency features.
- Uses **cross-attention** for injection: high-frequency features act as the query, and the gaze embedding serves as both key and value.
- Blends the injected high-frequency features with the low-frequency features to obtain the final gaze-controllable facial representation.

**Key Decision**: Ablation studies demonstrate that the gaze prompt must be injected into the **high-frequency features** rather than low-frequency features. Injecting into low-frequency features degrades the FID to 67.298 (vs. 38.346) and causes the gaze error to surge from 9.047° to 18.973°. This validates the hypothesis that gaze changes are primarily appearance-level variations.

#### 3. Face Geometric Prior Distillation

**Function**: Distills facial geometric knowledge from a pre-trained 3D portrait generation model to enhance the quality of single-image 3D inference.

**Mechanism**:
- Uses a pre-trained 3D portrait generator (Trevithick et al. 2023) as the teacher.
- Feeds the frontal image of the same identity with the target gaze into the teacher to acquire the teacher's triplane features.
- Renders depth maps $\mathbf{D}^t$ and $\mathbf{D}^s$ from the teacher and student models, respectively.
- Applies an L1 depth distillation loss:

$$\mathcal{L}_{\mathcal{D}} = \|\mathbf{D}^t - \mathbf{D}^s\|_1$$

**Design Motivation**:
- Single-image to 3D recovery is fundamentally an under-constrained problem, lacking sufficient constraints for accurate 3D reconstruction.
- Although the teacher model is not designed for gaze redirection, its facial geometric knowledge (depth structure) is transferable.
- Only depth, rather than color, is distilled because the appearance of the teacher's synthesized image may not perfectly match the target image.

### Loss & Training

**Total Loss**:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\mathcal{R}} + \beta \cdot \mathcal{L}_{\mathcal{D}} + \gamma \cdot \mathcal{L}_{\mathcal{P}}$$

where $\alpha = 1, \beta = 1, \gamma = 0.8$.

- **Mask-Guided Reconstruction Loss** $\mathcal{L}_{\mathcal{R}}$: Divided into facial and eye regions, with the eye region weighted $\alpha_2 = 2$ (facial as $\alpha_1 = 1$), utilizing the L1 norm.
- **Distillation Loss** $\mathcal{L}_{\mathcal{D}}$: L1 loss on depth maps.
- **Perceptual Loss** $\mathcal{L}_{\mathcal{P}}$: L1 distance of multi-layer VGG16 features.

**Training Details**:
- AdamW optimizer, with a learning rate of $10^{-5}$ for both encoding and rendering parts.
- Batch size 4, 50 epochs.
- ETH-XGaze training set: 10 frames/subject × 18 views/frame × 80 subjects.
- 2 × NVIDIA A100 (40GB), taking approximately 18 hours.
- Input resolution $512\times512$, processed in the style of EG3D.

## Key Experimental Results

### Main Results

**ETH-XGaze Dataset—Image Quality and Inference Speed**:

| Method | 3D | FID↓ | PSNR↑ | LPIPS↓ | SSIM↑ | Total Time↓ |
|------|-----|------|-------|--------|-------|--------|
| ST-ED | × | 115.020 | 17.530 | 0.300 | **0.726** | - |
| HeadNeRF | ✓ | 69.487 | 15.298 | 0.294 | 0.720 | 60.058s |
| GazeNeRF | ✓ | 81.816 | 15.453 | 0.291 | **0.733** | 60.060s |
| **RTGaze** | **✓** | **38.346** | **19.007** | **0.262** | 0.715 | **0.061s** |

- FID leads by a large margin (38.3 vs. the second-best 69.5), and both PSNR and LPIPS are optimal.
- **Inference speed is 0.061s, which is approximately 1000 times faster than GazeNeRF** (60s → 0.061s).
- SSIM is slightly lower than GazeNeRF (0.715 vs. 0.733), but all other metrics perform significantly better.

**Cross-Dataset Generalization Evaluation** (Gaze/Head Error + Identity Preservation):

| Dataset | Method | LPIPS↓ | ID↑ | Gaze↓ | Head↓ |
|--------|------|--------|-----|-------|-------|
| ColumbiaGaze | GazeNeRF | 0.352 | 23.157 | 9.464 | 3.811 |
| ColumbiaGaze | **RTGaze** | **0.249** | **61.765** | **7.625** | **3.326** |
| MPIIFaceGaze | GazeNeRF | 0.272 | 30.981 | 14.933 | 7.118 |
| MPIIFaceGaze | **RTGaze** | **0.251** | **46.098** | **9.409** | **6.444** |

All metrics lead comprehensively across ColumbiaGaze and MPIIFaceGaze, indicating robust cross-dataset generalization ability. Notably, the identity preservation rate (ID) is significantly improved (from 23→62, and 31→46), showing that the feedforward approach retains identity better than inversion.

### Ablation Study

**Gaze Injection Position Ablation**:

| Injection Target | FID↓ | ID↑ | Gaze↓ | Head↓ |
|---------|------|-----|-------|-------|
| Low-Frequency Features | 67.298 | 38.517 | 18.973 | 5.409 |
| **High-Frequency Features** | **38.346** | **60.708** | **9.047** | **3.631** |

Injecting into low-frequency features results in a severe performance drop: gaze error doubles (9→19°), FID nearly doubles, and identity preservation drops sharply.

**Loss Function Ablation**:

| Configuration | FID↓ | ID↑ | Gaze↓ | Head↓ |
|------|------|-----|-------|-------|
| Only $\mathcal{L}_\mathcal{R}$ | 101.053 | 47.251 | 9.332 | 4.208 |
| $\mathcal{L}_\mathcal{R} + \mathcal{L}_\mathcal{P}$ | 54.682 | 52.518 | 10.911 | 3.700 |
| $\mathcal{L}_\mathcal{R} + \mathcal{L}_\mathcal{P} + \mathcal{L}_\mathcal{D}$ | **38.346** | **60.708** | **9.047** | **3.631** |

The perceptual loss reduces FID from 101 to 55, and the distillation loss further lowers it to 38, while improving identity preservation from 47 to 61. The contribution of 3D prior distillation is highly significant.

### Key Findings

1. It is feasible for a feedforward scheme to fully replace GAN inversion—not only is it ~1000x faster, but the image quality is also improved (FID 38 vs. 82).
2. The huge improvement in identity preservation suggests that the inversion process itself might be the root cause of identity detail loss.
3. 3D geometric prior distillation is crucial to boosting quality—even though the teacher model is not specifically designed for gaze redirection.
4. Gaze changes are inherently high-frequency appearance variations, and low-frequency geometric features should not be directly modified.

## Highlights & Insights

1. **Breakthrough Speed**: This is the first time a 3D-aware method has realized real-time performance (61ms), paving the way for practical deployment of 3D gaze redirection (e.g., in videoconferencing, VR, etc.).
2. **High/Low-Frequency Decoupling**: A profound understanding of the gaze redirection task—where gaze changes eye texture (high-frequency) without altering the facial skeleton (low-frequency)—guided the correct architectural design.
3. **Elegant Distillation Strategy**: Only depth is distilled instead of color, avoiding the issue of appearance discrepancies between the teacher and student models.
4. **Cross-Attention Injection**: Cross-attention is employed to inject gaze prompts rather than simple concatenation or addition, facilitating more flexible feature interaction.

## Limitations & Future Work

1. **Slightly Lower SSIM**: The SSIM on ETH-XGaze (0.715) is lower than GazeNeRF (0.733), indicating room for improvement in pixel-level alignment.
2. **Frontal Image Dependency**: The distillation process relies on frontal images of the target identity with matching gaze, limiting how the training dataset can be utilized.
3. **Resolution Limit**: The 512×512 input resolution may be insufficient to capture extremely fine eye details (e.g., iris texture).
4. **Untreated Extreme Poses**: Although 3D methods are inherently better, the paper does not show the performance under extreme profiles or extreme pose angles.
5. **Code Not Open-Sourced** (as of currently), which limits reproducibility.

## Related Work & Insights

- **EG3D** (CVPR 2022): Pioneering work in triplane representation, upon which RTGaze's triplane decoder is based.
- **Trevithick et al. 2023**: 3D portrait generator, providing geometric prior distillation as a teacher model.
- The development of single-image-to-3D methods like **Zero-1-to-3** will continue to lower the barrier for 3D facial modeling.
- Similar 'feedforward replacing optimization' concepts can be extended to other NeRF-based applications (e.g., expression editing, style transfer, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ — Feedforward replacing inversion + high/low-frequency decoupling + distillation strategy. The combined innovation is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, efficiency comparisons, multi-dimensional metric evaluations, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, extensive quantitative results, but some implementation details are slightly lacking.
- Value: ⭐⭐⭐⭐⭐ — Real-time 3D gaze redirection holds substantial practical application value, and the speed breakthrough marks a significant milestone.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](../../ICCV2025/3d_vision/gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[AAAI 2026\] PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos](pfavatar_pose-fusion_3d_personalized_avatar_reconstruction_from_real-world_outfi.md)

</div>

<!-- RELATED:END -->
