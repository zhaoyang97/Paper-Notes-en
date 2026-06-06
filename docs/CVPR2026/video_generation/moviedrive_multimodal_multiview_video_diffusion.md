---
title: >-
  [Paper Note] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer
description: >-
  [CVPR 2026][Video Generation][Multi-modal multi-view video generation] The first method to simultaneously generate RGB + depth + semantic tri-modal multi-view driving scene videos within a unified DiT framework. Through…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Multi-modal multi-view video generation"
  - "Diffusion Transformer"
  - "urban scene synthesis"
  - "conditional control"
  - "CogVideoX"
date: 2026-05-08
content_hash: 334ff9e67f397a1f
---

# MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer

**Conference**: CVPR 2026
**arXiv**: [2508.14327](https://arxiv.org/abs/2508.14327)  
**Code**: Unavailable  
**Area**: Autonomous Driving / Video Generation
**Keywords**: Multi-modal multi-view video generation, Diffusion Transformer, urban scene synthesis, conditional control, CogVideoX

## TL;DR

The first method to simultaneously generate RGB + depth + semantic tri-modal multi-view driving scene videos within a unified DiT framework. Through a decomposed design of modal-shared layers (temporal + multi-view spatiotemporal attention) and modal-specific layers (cross-modal interaction + projection heads), a unified layout encoder, and diverse conditioning, the method achieves FVD 46.8 on nuScenes (22% improvement over CogVideoX+SyntheOcc), depth AbsRel 0.110, and semantic mIoU 37.5, outperforming pipelines based on separate model generation and estimation.

## Background & Motivation

**Background**: Autonomous driving scene video generation has advanced rapidly. Methods such as MagicDrive, DriveDreamer, and MaskGWM leverage diffusion models to achieve promising multi-view RGB video generation. However, these methods focus exclusively on the RGB modality.

**Limitations of Prior Work**: Autonomous driving requires multi-modal data (RGB + depth + semantics) for comprehensive scene understanding. Although multiple independent models can generate different modalities separately (e.g., generating RGB first and then estimating depth with Depth-Anything-V2), this increases deployment complexity, fails to exploit complementary inter-modal information, and results in poor cross-modal consistency.

**Key Challenge**: How can multi-modal multi-view driving videos be generated simultaneously within a unified framework? The key challenges are: (1) different modalities exhibit large content variation yet share underlying scene structure, requiring a distinction between shared and modality-specific knowledge; (2) both multi-view spatiotemporal consistency and cross-modal consistency must be ensured simultaneously; (3) complex driving scenes require fine-grained conditional control.

**Goal**: To build a unified multi-modal multi-view video DiT model that simultaneously generates 6-view, 49-frame videos across three modalities while guaranteeing spatiotemporal and cross-modal consistency.

**Key Insight**: Based on the observation that CogVideoX's shared 3D VAE can process videos of different modalities, the authors hypothesize that different modalities share a common latent space and require only a small number of modality-specific parameters to differentiate them. This motivates the modal-shared + modal-specific decomposition design.

**Core Idea**: Modal-shared layers in the unified DiT learn common spatiotemporal structure; modal-specific layers capture modality differences; diverse condition encodings control scene generation.

## Method

### Overall Architecture

Built upon CogVideoX (v1.1-2B). Three types of conditions (text, contextual reference frames, and layout) are processed through unified encoders to extract embeddings, which are concatenated with noisy latents and fed into a DiT composed of modal-shared and modal-specific layers. A shared 3D VAE encodes and decodes all modalities. DDPM noise scheduling is used during training, and DDIM with classifier-free guidance is applied at inference. The default configuration is 6 cameras × 49 frames × 512×256 resolution.

### Key Designs

1. **Diverse Condition Encoding**

    - **Function**: Encodes text, layout constraints, and reference frames into unified conditional embeddings to control scene generation.
    - **Mechanism**: (a) *Text conditioning* — camera intrinsics and extrinsics are encoded via Fourier encoding + MLP encoder $E^\text{cam}$; video descriptions are encoded by a frozen T5 encoder $E^\text{text}$; the concatenated embeddings are injected via cross-attention in the DiT. (b) *Layout conditioning* — 3D bounding box projection maps $c^b$, road structure maps $c^r$, and 3D occupancy sparse semantic maps $c^o$ are fused through a unified layout encoder (independent causal ResNets per condition + a shared causal ResNet): $f^\text{layout} = E_s^l(E_b^l(c^b) \otimes E_r^l(c^r) \otimes E_o^l(c^o))$. (c) *Contextual reference* — the first frame is encoded by the 3D VAE for future prediction.
    - **Design Motivation**: The unified layout encoder achieves implicit alignment of condition embedding spaces, proving more effective than multiple independent encoders.

2. **Modal-Shared Components (Temporal + Multi-View Spatiotemporal Blocks)**

    - **Function**: Learn temporal consistency and multi-view spatial structure shared across all modalities.
    - **Mechanism**: (a) *Temporal attention layer* $D^\text{tem}$ — CogVideoX's 3D full attention learns inter-frame consistency; text is injected via cross-attention; operating on dimension $\mathcal{R}^{V \times (NKW) \times C}$. (b) *Multi-view spatiotemporal block* $D^\text{st}$ — inserted every $\alpha_1$ layers; contains 3D spatial attention ($\mathcal{R}^{K \times (VHW) \times C}$ for cross-view structure), Hash grid 3D spatial embeddings, and full spatiotemporal attention ($\mathcal{R}^{(VKHW) \times C}$ for global context).
    - **Design Motivation**: Temporal attention alone cannot guarantee multi-view consistency (FVD degrades from 46.8 to 153.7 without spatiotemporal blocks); the multi-view spatiotemporal block explicitly models cross-view spatial relationships.

3. **Modal-Specific Components (Cross-Modal Interaction + Projection Heads)**

    - **Function**: Learn modality-specific content on top of shared representations while maintaining cross-modal alignment.
    - **Mechanism**: Cross-modal interaction layers are inserted every $\alpha_2$ layers, comprising self-attention, cross-modal cross-attention (query = current modality latent; key/value = concatenated latents of other modalities), and FFN. Modality-specific projection heads (linear layer + adaptive normalization) independently predict noise for each modality: $h'_m = D_m^\text{cm}(h, h_m^\text{modal}, t)$.
    - **Design Motivation**: Cross-modal cross-attention enables different modalities to exchange complementary information; unified generation yields higher quality than independent generation with external models.

### Loss & Training

- $\mathcal{L} = \sum_m \lambda_m \mathbb{E}_{x_{0,m}, t_m, \epsilon_m, C} \|\epsilon_m - \epsilon_{\theta,m}(x_{t,m}, t_m, C)\|^2$, with per-modality weighting.
- AdamW, lr=2e-4; 3D VAE and T5 are frozen; conditioning dropout is applied for improved generalization.
- Depth ground truth is generated by Depth-Anything-V2; semantic ground truth is generated by Mask2Former (not real annotations).

## Key Experimental Results

### Main Results — nuScenes

| Method | Conference | FVD↓ | mAP↑ | mIoU↑ | AbsRel↓ | Sem mIoU↑ |
|--------|------------|------|------|-------|---------|-----------|
| MagicDrive | ICLR24 | 236.2 | 9.7 | 15.6 | 0.255 | 23.5 |
| MagicDrive-V2 | ICCV25 | 112.7 | 11.5 | 17.4 | 0.280 | 22.4 |
| DriveDreamer-2 | AAAI25 | 55.7 | — | — | — | — |
| CogVideoX+SyntheOcc | — | 60.4 | 15.9 | 28.2 | 0.124 | 32.4 |
| **MoVieDrive** | — | **46.8** | **22.7** | **35.8** | **0.110** | **37.5** |

On Waymo: MoVieDrive FVD 61.6 vs. CogVideoX+SyntheOcc 82.3 (25% improvement).

### Ablation Study — Multi-Modal Generation

| Configuration | FVD↓ | AbsRel↓ | Sem mIoU↑ | Note |
|---------------|------|---------|-----------|------|
| RGB only + external model estimation | 42.0 | 0.121 | 36.4 | Best RGB quality but poor multi-modal quality |
| RGB+Depth unified + external semantics | 43.4 | 0.111 | 36.0 | Depth quality improves |
| **RGB+Depth+Semantics fully unified** | **46.8** | **0.110** | **37.5** | Best overall multi-modal quality |

### Ablation Study — DiT Components

| Configuration | FVD↓ | Note |
|---------------|------|------|
| L1 (temporal layers only) | 153.7 | No multi-view consistency |
| L1 + L3 (temporal + modal-specific) | 78.8 | No cross-view spatial learning |
| **L1 + L2 + L3 (full model)** | **46.8** | All components present |
| CogVideoX + cross-view attention | 118.4 | Simple modification is insufficient |

### Key Findings

- The multi-view spatiotemporal block is critical: removing it causes FVD to degrade from 46.8 to 153.7 (3.3× worse).
- Unified multi-modal generation achieves better depth (AbsRel 0.110) and semantics (mIoU 37.5) than RGB + external model estimation (0.121 / 36.4), at the cost of a marginal increase in RGB FVD (42.0 → 46.8), indicating slight inter-modal interference.
- The unified layout encoder outperforms independent encoders, attributed to implicit alignment of condition embedding spaces.
- Simply adding cross-view attention to CogVideoX still yields FVD 118.4, far inferior to MoVieDrive's 46.8.

## Highlights & Insights

- **First unified multi-modal multi-view generation framework** — filling a gap in autonomous driving scene generation. The modal-shared + modal-specific decomposition exploits the shared latent space hypothesis of the 3D VAE and is parameter-efficient.
- The cross-modal interaction layers enable different modalities to exchange complementary information — unified generation not only reduces the number of models required but also genuinely improves depth and semantic quality compared to independent generation.
- The unified layout encoder design outperforms multiple independent encoders through implicit embedding space alignment when fusing diverse layout conditions, and is generalizable to other multi-condition controlled generation tasks.
- Good scalability: supports long video generation (without reference frames) and scene editing across different weather/time conditions via text.

## Limitations & Future Work

- Long video generation in distant regions still exhibits noise; temporal consistency degrades in longer sequences.
- Depth and semantic ground truth are derived from pretrained model estimates (Depth-Anything-V2 / Mask2Former) rather than real annotations, imposing a ceiling on training signal quality.
- Multi-modal generation slightly increases RGB FVD (42.0 → 46.8), calling for improved inter-modal disentanglement strategies.
- The framework has not been extended to 3D modalities such as LiDAR point clouds.
- Integration with closed-loop simulators has not been explored, and downstream task benefits remain unquantified.
- Training cost is high (6 views × 49 frames × multiple modalities), demanding substantial computational resources.

## Related Work & Insights

- **vs. MagicDrive / MagicDrive-V2**: These methods generate RGB only and require additional models for depth and semantics. MoVieDrive substantially outperforms them on FVD (46.8 vs. 112.7 / 236.2) and controllability (mAP 22.7 vs. 11.5 / 9.7) across the board.
- **vs. UniScene (CVPR25)**: Uses separate models for RGB and LiDAR generation, remaining a non-unified approach. MoVieDrive achieves true single-model multi-modal generation.
- **vs. CogVideoX+SyntheOcc**: The most direct competitor. MoVieDrive consistently leads on all metrics (FVD 46.8 vs. 60.4), demonstrating the necessity of dedicated multi-modal multi-view architectural design.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: First unified multi-modal multi-view driving video generation; the decomposition design is principled.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Evaluated on nuScenes and Waymo with comprehensive ablations and detailed supplementary material.
- **Writing Quality** ⭐⭐⭐⭐: Clear structure, detailed methodology, and rich figures and tables.
- **Value** ⭐⭐⭐⭐: Provides a more complete solution for autonomous driving scene generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[CVPR 2026\] Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)
- [\[CVPR 2026\] DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior](dreamshot_storyboard_synthesis.md)

</div>

<!-- RELATED:END -->
