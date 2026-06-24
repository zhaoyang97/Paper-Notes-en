---
title: >-
  [Paper Note] IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation
description: >-
  [ECCV 2024][3D Vision][Video Generation] The IDOL framework is proposed to achieve joint video-depth generation for human-centric tasks by unifying a dual-modal U-Net and motion consistency loss, significantly outperforming existing methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Video Generation"
  - "Depth Estimation"
  - "Dual-Modal Diffusion"
  - "Human Animation"
  - "Video-Depth Alignment"
date: 2026-05-08
content_hash: cd6e890f0bf7b1c5
---

# IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation

**Conference**: ECCV 2024  
**arXiv**: [2407.10937](https://arxiv.org/abs/2407.10937)  
**Code**: [https://yhzhai.github.io/idol/](https://yhzhai.github.io/idol/) (project page available)  
**Area**: 3D Vision  
**Keywords**: Video Generation, Depth Estimation, Dual-Modal Diffusion, Human Animation, Video-Depth Alignment

## TL;DR

The IDOL framework is proposed to achieve joint video-depth generation for human-centric tasks by unifying a dual-modal U-Net and motion consistency loss, significantly outperforming existing methods.

## Background & Motivation

**Background**: Significant progress has been made in diffusion-model-based human video generation, but the generated videos lack depth information, which limits downstream applications requiring spatial awareness, such as AR/VR.

**Limitations of Prior Work**: Discriminative monocular depth estimation methods (e.g., MiDaS, HDNet) generalize poorly on synthetic images, producing incomplete or over-simplified depth maps; multi-view methods struggle to control human appearance and motion.

**Key Challenge**: Videos (3-channel RGB sequences) and depth (scalar depth map sequences) are fundamentally different modalities; existing pre-trained diffusion models are designed only for single-modality image generation. Meanwhile, maintaining precise video-depth spatial alignment in the latent space is a difficult problem.

**Goal**: How to design a unified framework to simultaneously generate high-quality human videos and their corresponding depth maps, while ensuring their spatiotemporal alignment.

**Key Insight**: Render depth maps as RGB heatmaps, redefining depth generation as a "stylized video generation" problem, which allows direct utilization of pre-trained image generation models; cross-modality information interaction is achieved through parameter sharing and consistency loss.

**Core Idea**: A unified dual-modal U-Net shares parameters for joint video-depth denoising, supplemented by motion consistency loss and cross-attention consistency loss to facilitate precise video-depth spatial alignment.

## Method

### Overall Architecture

IDOL is built on a 3D U-Net and trained in two stages: (1) Human Attribute Outpainting Pre-training (HAOP) to learn human appearance, and (2) joint video-depth denoising training. The inputs are a human foreground image $f$, a background image $b$, and a pose sequence $p=\{p_1,...,p_L\}$; the outputs are a video $v$ and its corresponding depth map sequence $d$. Foreground appearance is encoded by CLIP and fed into cross-attention, while poses are controlled via ControlNet. Background latents are directly added to the input noise.

### Key Designs

1. **Unified Dual-Modal U-Net**:
The architecture and parameters of a single U-Net are shared for video and depth denoising. A learnable modality embedding (one-hot modality label $y_v$ or $y_d$) is added to the timestep embedding to control the output modality. The joint denoising objective is:

$$\mathcal{L}_{\text{denoise}} = \mathbb{E}\left[\|\epsilon_v - \epsilon_\theta(z_{v,t}, t, f, b, p; y_v)\|_2^2 + \|\epsilon_d - \epsilon_\theta(z_{d,t}, t, f, b, p; y_d)\|_2^2\right]$$

Design Motivation: Sharing parameters not only saves parameter capacity (only 1.39B vs 2×1.39B) but also implicitly learns structural information from depth to enhance video quality. A cross-modal attention layer is added at the end of each U-Net block, which concatenates video and depth features to perform spatial self-attention, enabling explicit information interaction.

2. **Motion Consistency Loss**:
It is observed that although the intermediate features of video and depth have similar spatial layouts, their temporal motion patterns may not be synchronized, leading to misaligned outputs. Cost volumes between adjacent frame features are calculated and normalized into a motion field:

$$u_{v,l,i,j,h,k} = \frac{\exp(c_{v,l,i,j,h,k}/\tau)}{\sum_{h'}\sum_{k'}\exp(c_{v,l,i,j,h',k'}/\tau)}$$

Then, the MSE between video and depth motion fields is minimized: $\mathcal{L}_{\text{mo}} = \frac{1}{LHWHW}\sum\|u_v - u_d\|_2^2$.

Design Motivation: Self-attention features in intermediate layers of the U-Net contain semantic information. Forcing the motion of the two modal features to synchronize promotes video-depth alignment.

3. **Cross-Attention Map Consistency Loss**:
Cross-attention maps influence the spatial layout of the generated image. A MSE loss is applied to enforce consistency between the cross-attention maps of the video flow and the depth flow:

$$\mathcal{L}_{\text{xattn}} = \|M_v - M_d\|_2^2$$

Design Motivation: Drawing on the existing finding that cross-attention maps have a decisive impact on image layout, this is extended to cross-modal consistency scenarios.

4. **Human Attribute Outpainting Pre-training (HAOP)**:
Improves upon DisCo's HAP pre-training: (a) expands the background mask via dilation to force the model to fill in the background surrounding the foreground; (b) randomly crops and scales the foreground image to prompt the model to extrapolate partial human attributes. This solves the prominent background masking artifacts when the target pose deviates from the original position.

### Loss & Training

Total training loss:

$$\mathcal{L} = \mathcal{L}_{\text{denoise}} + \sum_{n=1}^{N}(w_{\text{mo}}\mathcal{L}_{\text{mo},n} + w_{\text{xattn}}\mathcal{L}_{\text{xattn},n})$$

Consistency losses are only applied to the U-Net upsampling blocks, as the ControlNet feature fusion occurs at these locations.

## Key Experimental Results

### Main Results

| Method | TikTok FID-FVD↓ | TikTok FVD↓ | TikTok Depth L2↓ | NTU120 FID-FVD↓ | NTU120 FVD↓ | NTU120 Depth L2↓ |
|------|---------|---------|---------|---------|---------|---------|
| FOMM | 38.36 | 404.31 | - | 40.34 | 1439.50 | - |
| DisCo | 20.75 | 257.90 | 0.0975† | 26.21 | 458.92 | 0.0371† |
| LDM3D | 45.30 | 553.03 | 0.0637 | 71.11 | 587.84 | 0.0650 |
| MM-Diff | 48.92 | 771.32 | 0.0367 | 58.44 | 504.05 | 0.0404 |
| **IDOL** | **17.86** | **223.69** | **0.0336** | **20.23** | **314.82** | **0.0317** |

† indicates depth estimated using HDNet on synthetic images.

### Ablation Study

| Setup | Parameters | FID-FVD↓ | FVD↓ | Depth L2↓ | FID↓ |
|------|--------|---------|------|-----------|------|
| Separate U-Net | 2×1.39B | 24.28 | 282.50 | 0.0822 | 41.72 |
| Shared U-Net | 1.39B | 22.10 | 272.37 | 0.0369 | 39.43 |
| + Cross-Modal Attention | 1.41B | 19.28 | 260.65 | 0.0360 | 39.01 |
| +$\mathcal{L}_{\text{xattn}}$ | - | 19.99 | 244.58 | 0.0351 | 37.89 |
| +$\mathcal{L}_{\text{mo}}$+$\mathcal{L}_{\text{xattn}}$ | - | **17.86** | **223.69** | **0.0336** | **36.04** |

### Key Findings

- Sharing U-Net parameters halves the parameter count but achieves better performance, indicating that implicit cross-modal structural learning is beneficial.
- Motion consistency loss simultaneously improves all three metrics: video, depth, and image quality.
- IDOL has the lowest FLOPs (39.35T), the fewest parameters (1.41B), and the fastest inference (12.23s).
- Even with OpenPose lacking hand keypoints, IDOL can still generate reasonable hands.

## Highlights & Insights

- **Depth as RGB heatmap**: Cleverly redefines depth generation as stylized video generation, avoiding modifications to the output layer of the pre-trained model.
- **Motion field consistency**: Grounded in the semantic properties of U-Net intermediate features, aligning at the motion level rather than the pixel level is a novel idea.
- **Parameter efficiency**: The shared U-Net design saves parameters while improving performance, offering a win-win solution.
- The HAOP pre-training strategy is simple yet effective, resolving background masking artifacts during pose shifting.

## Limitations & Future Work

- Dual-modal high-resolution processing incurs high computational overhead, making it unsuitable for real-time applications.
- The dependency on high-quality depth training data limits scene generalization capability.
- Unsupervised depth or data augmentation strategies can be explored to alleviate data quality constraints.
- There is a risk of deepfake ethical concerns; measures such as invisible watermarking should be considered.

## Related Work & Insights

- Compared to LDM3D (which modifies the autoencoder to output RGB + depth) and MM-Diffusion (which couples the U-Net to simultaneously denoise video and audio latents), IDOL's parameter sharing + modality label scheme is more elegant and efficient.
- It can be extended to other joint multimodal generation tasks (e.g., video + normal map, video + segmentation map).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified dual-modal U-Net and motion consistency loss are novel and effective designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual datasets, multiple depth types, extensive ablation studies, and computational complexity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear layout and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Holds promising application prospects in fields like VR/AR and video games.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](../../ICCV2025/3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[ECCV 2024\] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image](compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)

</div>

<!-- RELATED:END -->
