---
title: >-
  [Paper Note] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control
description: >-
  [ICCV 2025][Video Generation] MagicDrive-V2 proposes a multi-view driving video generation framework based on DiT + 3D VAE. Through a spatial-temporal condition encoding module and a progressive training strategy, it achieves high-resolution long video generation at 848×1600×6 views and 241 frames, significantly surpassing existing methods in both resolution and frame count.
tags:
  - ICCV 2025
  - Video Generation
  - DiT
  - 3D VAE
  - Multi-view
  - Controllable Generation
date: 2026-05-08
content_hash: bf6a8c2269d07ea1
---

# MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control

**Conference**: ICCV 2025  
**arXiv**: [2411.13807](https://arxiv.org/abs/2411.13807)  
**Code**: [https://flymin.github.io/magicdrive-v2/](https://flymin.github.io/magicdrive-v2/) (Project Page)  
**Area**: Video Generation  
**Keywords**: Video Generation, DiT, 3D VAE, Multi-view, Controllable Generation

## TL;DR

MagicDrive-V2 proposes a multi-view driving video generation framework based on DiT + 3D VAE. Through a spatial-temporal condition encoding module and a progressive training strategy, it achieves high-resolution long video generation at 848×1600×6 views and 241 frames, significantly surpassing existing methods in both resolution and frame count.

## Background & Motivation

**Background**: Controllable video generation for autonomous driving is a key research direction, requiring high resolution (for detail recognition) and long video sequences (for evaluating algorithmic interactions). Existing methods are primarily based on UNet + 2D VAE architectures, such as MagicDrive and Drive-WM.

**Limitations of Prior Work**: Constrained by the scalability of UNet and the compression capacity of 2D VAE, existing methods are severely limited in resolution and frame count. For example, MagicDrive supports only 224×400×6 views at 60 frames, and Delphi supports only 512×512×6 views at 10 frames.

**Key Challenge**: DiT + 3D VAE has become the standard paradigm for video generation, as 3D VAE reduces computational overhead by an order of magnitude through spatial-temporal compression. However, 3D VAE disrupts the frame-level alignment between geometric control signals and video frames — 2D VAE preserves the temporal axis, allowing image-level control methods to be directly extended to video; whereas 3D VAE produces $T/f$ spatial-temporal latents (where $f$ is the temporal compression factor), breaking the dimensional alignment between control signals and latents.

**Goal**: (1) How to achieve per-frame geometric control within the DiT + 3D VAE framework? (2) How to support multi-view consistency? (3) How to train efficiently to support high resolution and long video?

**Key Insight**: The authors observe that naively applying global reduction along the temporal dimension causes ghosting artifacts, and therefore design a spatial-temporal encoding module aligned with the downsampling rate of the 3D VAE.

**Core Idea**: Spatial-temporal condition encoding is used to re-align geometric control signals with the spatial-temporal latents of the 3D VAE. MVDiT is introduced to enable multi-view generation, and progressive training with mixed resolutions and durations enables extrapolation capability.

## Method

### Overall Architecture

MagicDrive-V2 is built upon the STDiT-3 architecture and adopts a dual-branch design (similar to ControlNet). Inputs include text descriptions $\mathbf{L}$, road maps $\mathbf{M}_t$, 3D bounding boxes $\mathbf{B}_t$, camera poses $\mathbf{C}$, and ego-vehicle trajectories $\mathbf{Tr}_t^0$. The 3D VAE from CogVideoX is used for spatial-temporal compression (256× compression ratio), and the DiT performs denoising generation in the latent space. Training is based on Flow Matching with v-prediction loss.

### Key Designs

1. **MVDiT Multi-view DiT Block**:

    - Function: Integrates cross-view attention layers into STDiT-3 blocks to enable multi-view consistent generation.
    - Mechanism: A cross-view attention layer is added to each STDiT-3 block, allowing features from different camera views to interact. Text, bounding box, camera, and trajectory signals are injected via cross-attention, while map signals are injected via an additive branch.
    - Design Motivation: Autonomous driving requires simultaneous generation across 6 camera views with inter-view consistency; independent generation per view leads to inconsistency.

2. **Spatial-Temporal Condition Encoder**:

    - Function: Aligns per-frame geometric control signals to the spatial-temporal latent dimensions of the 3D VAE.
    - Mechanism: For map $\mathbf{M}_t$, the ControlNet design is extended with temporal downsampling modules from the 3D VAE (with newly introduced trainable parameters) to align control features with base block features. For 3D bounding boxes $\mathbf{B}_t$, a downsampling module equipped with a temporal transformer and RoPE is introduced to capture temporal correlations, producing spatial-temporal embeddings aligned with the video latents. The downsampling ratio is consistent with the 3D VAE: $8n$ or $8n+1$ inputs → $2n$ or $2n+1$ outputs.
    - Design Motivation: Experiments show that naive global temporal reduction causes ghosting artifacts, hypothesized to result from repeat operations. Downsampling aligned with the VAE preserves the uniqueness of temporal information and avoids such artifacts.

3. **Enhanced Text Control**:

    - Function: Generates richer scene text descriptions via an MLLM.
    - Mechanism: Existing datasets (e.g., nuScenes) contain only simple weather/time-of-day descriptions. A multimodal large language model is used to generate richer contextual descriptions (road type, background elements, etc.) from middle frames of the video. The MLLM is prompted to describe only static scenes to avoid conflict with geometric control signals.
    - Design Motivation: Enriched text control enables more diverse generation scenarios.

### Loss & Training

A three-stage progressive training strategy is adopted: (1) low-resolution images → (2) high-resolution short videos → (3) high-resolution long videos. In the third stage, videos of mixed resolutions and durations are used (up to 241 frames at 224×400 and up to 848×1600 at 33 frames), enabling the model to acquire extrapolation capability. The loss function is the standard Conditional Flow Matching (CFM) loss: $\mathcal{L}_{CFM} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \|v_\Theta(\mathbf{z}_t, t) - (\mathbf{z}_1 - \epsilon)\|_2^2$.

## Key Experimental Results

### Main Results

| Method | FVD↓ | mAP↑ | mIoU↑ |
|--------|------|------|-------|
| MagicDrive (16f) | 218.12 | 11.86 | 18.34 |
| MagicDrive (60f) | 217.94 | 11.49 | 18.27 |
| MagicDrive3D | 210.40 | 12.05 | 18.27 |
| **MagicDrive-V2** | **94.84** | **18.17** | **20.40** |

FVD is reduced by over 55%, mAP improves by over 50%, while resolution is 3.3× higher and frame count is 4× greater than prior methods.

### Ablation Study — Training Data Configuration

| Training Data | FVD↓ | mAP↑ | mIoU↑ |
|---------------|------|------|-------|
| 17×224×400 | 97.21 | 10.17 | 12.42 |
| (1-65)×224×400 | 100.73 | 10.51 | 12.74 |
| 17×(224×440–424×800) | 96.34 | 14.91 | 17.53 |
| 1-65×(mixed resolution) | 99.66 | 15.44 | 18.26 |

### Key Findings

- **Spatial-temporal encoding is highly effective**: The 4× downsampling approach (proposed method) converges fastest in overfitting experiments and achieves the lowest validation loss; naive reduction baselines produce ghosting and artifacts.
- **High resolution is easier to adapt to than long video**: The model adapts to higher resolution faster than to longer video durations.
- **Extrapolation capability**: Although training uses at most 33 frames at 848×1600, the model can extrapolate to generate 241 frames at 848×1600 (8× extrapolation) while maintaining stable FVD.
- **Cross-dataset generalization**: Fine-tuning on Waymo with only 1 day of training (1k+ steps) enables 3-view video generation.

## Highlights & Insights

- **Spatial-temporal condition encoding** is the core contribution: by aligning with the 3D VAE downsampling rate, it resolves the incompatibility between 3D VAE and per-frame geometric control. This approach is transferable to any controllable video generation task using a 3D VAE.
- **Mixed resolution/duration training enables extrapolation**: Training on mixed resolutions and frame counts allows the model to generalize across dimensions, producing videos beyond the training configuration.
- **Progressive training accelerates convergence**: The image → short video → long video progression exploits the tendency of models to first learn content quality and then controllability.

## Limitations & Future Work

- Validation is limited to nuScenes and Waymo, lacking coverage of more diverse driving scenarios (e.g., severe weather, nighttime).
- The model trains the DiT from scratch without leveraging pretrained text-to-video models, incurring high training costs.
- Video quality degrades significantly with rollout-based long video generation; currently only single-pass inference is supported.
- The downstream task effectiveness of generated videos (e.g., improving perception model performance) has not been thoroughly validated.

## Related Work & Insights

- **vs. MagicDrive**: The predecessor uses UNet + 2D VAE; this work upgrades to DiT + 3D VAE, achieving substantial gains in resolution and frame count while preserving the core control condition design (BEV map, 3D box, trajectory).
- **vs. Vista/GAIA-1**: These methods support only front-view single-camera generation with limited controllability; MagicDrive-V2 supports 6 views and multiple geometric control signals.
- **vs. DiVE/Delphi**: Both are multi-view methods but operate at far lower resolution and frame count compared to this work.

## Rating

- Novelty: ⭐⭐⭐⭐ The spatial-temporal condition encoding design addresses a practical problem from an engineering perspective, though the overall framework is a composition of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablation studies are comprehensive, covering VAE comparisons, encoding strategy comparisons, training strategy comparisons, and extrapolation validation.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, figures and tables are informative, and problem motivation is well articulated.
- Value: ⭐⭐⭐⭐⭐ Significantly advances the resolution and frame count limits of autonomous driving video generation, with important practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation](../../NeurIPS2025/video_generation/rlgf_reinforcement_learning_with_geometric_feedback_for_autonomous_driving_video.md)
- [\[ICCV 2025\] Long Context Tuning for Video Generation](long_context_tuning_for_video_generation.md)
- [\[NeurIPS 2025\] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](../../NeurIPS2025/video_generation/foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)

<!-- RELATED:END -->
