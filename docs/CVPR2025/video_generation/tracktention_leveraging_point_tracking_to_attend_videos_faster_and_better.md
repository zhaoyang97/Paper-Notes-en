---
title: >-
  [Paper Note] Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better
description: >-
  [CVPR 2025][Video Generation][Point Tracking] Tracktention proposes a novel point-tracking-based attention layer. By injecting pre-extracted point trajectory information into Vision Transformers, it achieves motion-aware temporal feature aggregation. This layer upgrades image-only models to SOTA video models, significantly improving temporal consistency in video depth estimation and video colorization tasks.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Point Tracking"
  - "Temporal Consistency"
  - "Video Depth Estimation"
  - "Attention Mechanism"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 1617352cbbcc20a6
---

# Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better

**Conference**: CVPR 2025  
**arXiv**: [2503.19904](https://arxiv.org/abs/2503.19904)  
**Code**: [https://zlai0.github.io/TrackTention](https://zlai0.github.io/TrackTention)  
**Area**: Video Generation  
**Keywords**: Point Tracking, Temporal Consistency, Video Depth Estimation, Attention Mechanism, Plug-and-Play

## TL;DR

Tracktention proposes a novel point-tracking-based attention layer. By injecting pre-extracted point trajectory information into Vision Transformers, it achieves motion-aware temporal feature aggregation. This layer upgrades image-only models to SOTA video models, significantly improving temporal consistency in video depth estimation and video colorization tasks.

## Background & Motivation

**Background**: Video analysis tasks (such as video segmentation, depth estimation, and colorization) require temporal consistency in outputs compared to image tasks. Existing temporal modeling methods mainly fall into two categories: 3D convolutions and space-time attention. Meanwhile, the field of point tracking has made significant progress in recent years, with trackers like PIPs, TAPIR, and CoTracker reliably tracking a large number of points across long videos.

**Limitations of Prior Work**: 3D convolutions assume local spat-temporal correlation and fail to handle large-displacement motions. Space-time attention typically requires downsampling spatial resolution or restricting the temporal window to control computational costs, making it difficult to represent motion precisely. Optical flow methods underperform under occlusions and large displacements. Essentially, these methods all attempt to establish spatio-temporal correspondences "implicitly".

**Key Challenge**: Precise motion modeling requires fine-grained spat-temporal correspondences, but existing methods are either computationally prohibitive (full space-time attention) or lack sufficient modeling capacity (3D convolutions, divided attention).

**Goal**: To design an efficient temporal modeling component that explicitly utilizes motion information, capable of upgrading image models to video models in a plug-and-play manner.

**Key Insight**: Modern point trackers are already highly capable "motion experts" that provide precise cross-frame correspondences. Rather than forcing networks to learn these correspondences implicitly, it is more effective to directly leverage tracker outputs as a bridge.

**Core Idea**: Utilizing pre-extracted point trajectories as "intermediaries"—first sampling information from image features along the trajectory paths, propagating it temporally, and then splatting it back into the image features to achieve explicit motion-aware temporal alignment.

## Method

### Overall Architecture

The Tracktention layer consists of three sub-modules: (1) Attentional Sampling, which pools information from video feature maps to track tokens along point trajectory positions; (2) Track Transformer, which updates track tokens along the temporal dimension to achieve temporal information propagation; and (3) Attentional Splatting, which writes the updated track tokens back to the video feature maps. The entire process is embedded into existing networks via residual connections: $F' = F + \text{Tracktention}(F)$.

### Key Designs

1. **Attentional Sampling**:

    - **Function**: Pools information from video feature maps to track tokens along point trajectory locations
    - **Mechanism**: Performs cross-attention by treating the positional embeddings of track tokens as queries, and feature map tokens as keys/values. Crucially, a Gaussian positional bias $B_{tij} = -\frac{\|P_{ti} - \text{pos}(j)\|^2}{2\sigma^2}$ ($\sigma=1/2$) is introduced to encourage attention to concentrate around the trajectory points. Meanwhile, RoPE coding is used to capture relative spatial relationships, and QK-normalization is used to stabilize training.
    - **Design Motivation**: It is more flexible than simple bilinear sampling, allowing the model to learn better sampling strategies. The Gaussian bias ensures that attention is spatially constrained by the trajectory location.

2. **Track Transformer**:

    - **Function**: Propagates and smooths track token information along the temporal dimension
    - **Mechanism**: Reshapes the track tokens from $T \times M \times D_f$ to $M \times T \times D_f$, and performs self-attention along the temporal dimension for each trajectory independently. It uses a 2-layer transformer encoder with sinusoidal positional encodings.
    - **Design Motivation**: Information is not exchanged across different trajectories because spatial information exchange is already handled by the ViT itself. Experiments verify that cross-trajectory attention performs slightly worse and runs slower.

3. **Attentional Splatting**:

    - **Function**: Writes the updated track tokens' information back to the video feature maps
    - **Mechanism**: Designed symmetrically to Attentional Sampling—with feature map grid coordinates as queries and track tokens as keys/values, using the transposed bias matrix $B_t$, and finally producing the result via the output projection $W_{\text{out}}$. $W_{\text{out}}$ is initialized to zero to ensure the network output remains unchanged in the early training phases.
    - **Design Motivation**: The symmetric sampling-splatting design ensures consistency in information processing. Zero initialization ensures that inserting the Tracktention layer does not disrupt pretrained weights.

### Loss & Training

CoTracker3 is used as the point-tracking pre-processor to uniformly and randomly sample 576 points in the space-time volume and track them bidirectionally. Tracktention layers are inserted in a residual manner after all (or a subset of) the transformer blocks of the ViT, with their output projections initialized to zero to preserve pretraining effects. The model is then fine-tuned on the training set of downstream video tasks.

## Key Experimental Results

### Main Results

| Method | Type | Params | Sintel AbsRel↓ | Scannet AbsRel↓ | KITTI AbsRel↓ | Bonn AbsRel↓ | Avg AbsRel↓ |
|------|------|--------|----------|----------|----------|----------|----------|
| DepthCrafter | Video | 1521M | 0.343 | 0.125 | 0.110 | 0.075 | 0.163 |
| DepthAnything | Image | 343M | 0.325 | 0.130 | 0.142 | 0.078 | 0.169 |
| **Tracktention** | **Video** | **140M** | **0.295** | **0.087** | **0.104** | **0.066** | **0.138** |

### Ablation Study

Key ablation comparisons (design choices extracted from the paper):

| Configuration | Description of Effect |
|------|---------|
| Cross-trajectory attention | Slightly worse and slower; spatial information exchange is already completed by ViT |
| Without Gaussian positional bias | Decentralized attention, degenerating into global attention |
| Replacing Attentional Sampling with bilinear sampling | Performance drop; fixed sampling is inferior to learnable sampling |
| Random vs. grid initialization for trajectories | Random is better; grid initialization leads to non-uniform coverage under motion |

### Key Findings

- **Smallest parameter count with optimal performance**: Only 140M parameters (based on DepthAnything-Base 97M), surpassing the 1521M DepthCrafter.
- Upgrading the image-only model (DepthAnything) to a video model reduces the AbsRel from 0.169 to 0.138, yielding an 18.3% improvement.
- Outperforms both specifically designed video depth models and large-parameter image models across all datasets.
- Tracktention is also successfully applied to video colorization tasks, outperforming models native to video design.

## Highlights & Insights

- **Point tracking as a general infrastructure for video understanding**—Instead of being restricted to specific tasks, it is injected as a general component into any image model. This "expert module mixture" paradigm is highly elegant.
- **Zero-initialized residual insertion strategy** allows Tracktention to be embedded into pretrained models losslessly, gradually activating temporal capabilities via fine-tuning. While this trick is also utilized in works like ControlNet, it represents a novel application in video temporal modeling.
- **Better results with fewer parameters** can be achieved because explicitly leveraging the tracker's outputs spares the network from the overhead of learning correspondences from scratch.

## Limitations & Future Work

- Dependency on the quality of external point trackers—If the tracker fails in certain scenarios (e.g., highly repetitive textures, severe occlusions), the effectiveness of Tracktention will be constrained.
- Pre-extracting point trajectories increases preprocessing overhead, which may pose a bottleneck in real-time applications.
- Currently only validated on depth estimation and colorization; it can be further extended to more tasks such as video segmentation and video generation.
- Future work can explore end-to-end training of trackers and Tracktention layers.

## Related Work & Insights

- **vs. DepthCrafter**: DepthCrafter generates video depth based on video diffusion models with 1521M parameters, while Tracktention has only 140M parameters but outperforms it, as explicit motion modeling is more efficient than implicit learning.
- **vs. 3D Convolutions / Space-Time Attention**: These methods either suffer from limited receptive fields (3D convolutions) or are computationally expensive yet lack precision (divided attention). Tracktention explicitly aligns features at the resolution of tracking points (which is much higher than the feature map resolution).
- **vs. Optical Flow Methods**: Optical flow only models relationships between adjacent frames and fails during occlusions or large displacements, whereas point tracking handles long-range dependencies and occlusions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Elevating point tracking into a general foundational component for video understanding is a highly innovative perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two tasks, depth estimation and colorization, with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, precise methodological description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ The plug-and-play design gives it extremely high practical value and broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Exploring Temporally-Aware Features for Point Tracking](exploring_temporally-aware_features_for_point_tracking.md)
- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[CVPR 2025\] ReCapture: Generative Video Camera Controls for User-Provided Videos Using Masked Video Fine-Tuning](recapture_generative_video_camera_controls_for_user-provided_videos_using_masked.md)
- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](../../NeurIPS2025/video_generation/vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)

</div>

<!-- RELATED:END -->
