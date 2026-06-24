---
title: >-
  [Paper Note] Exploring Temporally-Aware Features for Point Tracking
description: >-
  [CVPR 2025][Video Generation][Point tracking] This work proposes Chrono, a temporally-aware feature backbone designed for point tracking. By inserting temporal adapters (2D convolutional downsampling + 1D local temporal attention + 2D convolutional upsampling) between the Transformer blocks of DINOv2, Chrono achieves state-of-the-art performance in a refiner-free setting using only simple feature matching (soft-argmax).
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Point tracking"
  - "temporally-aware features"
  - "DINOv2"
  - "feature backbone"
  - "temporal adapter"
date: 2026-05-08
content_hash: 386ca6f81a342b91
---

# Exploring Temporally-Aware Features for Point Tracking

**Conference**: CVPR 2025  
**arXiv**: [2501.12218](https://arxiv.org/abs/2501.12218)  
**Code**: [https://cvlab-kaist.github.io/Chrono/](https://cvlab-kaist.github.io/Chrono/)  
**Area**: Video Generation  
**Keywords**: Point tracking, temporally-aware features, DINOv2, feature backbone, temporal adapter

## TL;DR

This work proposes Chrono, a temporally-aware feature backbone designed for point tracking. By inserting temporal adapters (2D convolutional downsampling + 1D local temporal attention + 2D convolutional upsampling) between the Transformer blocks of DINOv2, Chrono achieves state-of-the-art performance in a refiner-free setting using only simple feature matching (soft-argmax).

## Background & Motivation

Current point tracking methods generally adopt a two-stage pipeline: first, a simple backbone (such as ResNet) is used to extract features for coarse estimation, and then iterative refiners are employed to inject temporal information and correct errors. This paradigm has two core issues:

1. **Outdated backbones**: The point tracking field still relies on shallow ResNet/TSM-ResNet trained from scratch, whereas tasks like segmentation and detection have long benefited from large-scale pretrained backbones. DINOv2 provides powerful spatial feature representation but lacks temporal awareness.

2. **Heavy refiner overhead**: Refiners (such as those in TAPIR and LocoTrack) must perform iterative temporal processing individually for each query point, which is computationally expensive and inefficient. If the backbone itself is capable of capturing temporal information, the burden on the refiner can be significantly alleviated.

Key Insight: Quality point tracking features must simultaneously possess both **spatial discriminability** (strong feature representation $\rightarrow$ DINOv2) and **temporal consistency** (cross-frame motion understanding $\rightarrow$ temporal adapter), rather than delegating all temporal reasoning to the downstream refiner.

## Method

### Overall Architecture

Chrono is built upon DINOv2 (ViT-S/14 or ViT-B/14). The weights of DINOv2 are frozen, and trainable temporal adapters are inserted between each Transformer block. After feature extraction, point prediction is completed using a simple cosine similarity matching followed by soft-argmax, without requiring any learnable refining layers.

### Key Designs

1. **Temporal Adapter**:
    - **Function**: Injects temporal awareness without undermining the pretrained knowledge of DINOv2.
    - **Mechanism**: Adopts a bottleneck structure—first using 2D convolution (stride $s=4$) for spatial downsampling to reduce computational load, followed by 1D local window attention (window size $N=13$, corresponding to $\pm6$ frames) to aggregate temporal information, and finally using 2D convolution for upsampling to restore resolution, with a residual connection to preserve original features.
    - **Design Motivation**: TSM-ResNet only considers adjacent frames (window size = 2), leading to insufficient temporal context. Chrono utilizes a $6\times$ longer temporal window to capture complex motion dynamics. Spatial downsampling not only reduces computational cost but also expands the spatial receptive field.

2. **Correlation-based Point Prediction**:
    - **Function**: Predicts point trajectories without relying on learnable modules.
    - **Mechanism**: For a query point $q=(x_q, y_q, t_q)$, bilinear interpolation is used to extract the query feature $\mathbf{f}_q$. Its cosine similarity with all positions in each frame is computed to generate a correlation map $\mathcal{C}_t$. Finally, a soft-argmax (temperature $\tau=20$, local mask $M=5$ pixels) is applied to obtain sub-pixel level position estimates.
    - **Design Motivation**: If the features are sufficiently strong (temporal smoothness + fine-grained spatial metrics), simple non-parametric matching can provide high-quality initial trajectories, bypassing the query-dependent computations of refiners.

3. **Full-layer Temporal Adapter Deployment**:
    - **Function**: Inserts adapters across all 12 Transformer blocks of DINOv2.
    - **Mechanism**: Shallow blocks capture local detailed motion while deep blocks capture global motion patterns. Full-layer deployment achieves multi-scale motion modeling.
    - **Design Motivation**: Ablation studies show that placing adapters only in the first half, second half, or alternate blocks yields suboptimal performance compared to full-layer deployment ($\delta_{avg}^x$ improves from 61.7–65.9 to 68.0).

### Loss & Training

- **Loss function**: Huber Loss (robust to outliers), with loss ignored for occluded points: $\mathcal{L}_t = (1-v_t) \cdot \mathcal{L}_{\text{Huber}}(\hat{p}_t, p_t)$
- **Training data**: Kubric Panning-MOVi-E synthetic dataset
- **Optimizer**: AdamW, learning rate $10^{-4}$, weight decay $10^{-4}$
- **Training setup**: 4 $\times$ A100 GPUs, 100K iterations, batch size of 1/GPU, sampling 256 query points per batch

## Key Experimental Results

### Main Results (TAP-Vid-DAVIS Strided Mode $\delta_{avg}^x$)

| Backbone | DAVIS | Kinetics | RGB-Stacking | Requires Refiner |
|---------|-------|----------|-------------|------------|
| Chrono (ViT-B/14) | **70.1** | **68.5** | **86.0** | No |
| Chrono (ViT-S/14) | 68.0 | 66.8 | 84.3 | No |
| DINOv2 (ViT-B/14) | 54.4 | 46.6 | 46.6 | No |
| TSM-ResNet-18 | 49.2 | 54.5 | 67.9 | No |
| ResNet-18 | 53.3 | 56.3 | 73.9 | No |

### Comparison with full pipelines with refiners

| Method | RGB-Stacking | DAVIS | Throughput (pts/s) | Refiner Parameters |
|------|-------------|-------|---------------|-----------|
| Chrono (ViT-B/14) | **86.0** | 70.1 | **26,140** | 0M |
| TAPIR | 74.6 | **73.6** | 2,097 | 25.9M |
| Chrono + LocoTrack | 91.0 (AJ:83.2) | 80.2 (AJ:68.2) | - | - |

### Ablation Study

| Configuration | DAVIS $\delta_{avg}^x$ | Description |
|------|----------------------|------|
| 1D Attention (Ours) | **68.0** | Adaptive inter-frame correlation modeling |
| 3D Convolution | 66.4 | Fixed weights, poor flexibility |
| 1D Convolution | 65.9 | Fixed weights |
| All Blocks (12 adapters) | **68.0** | Multi-scale motion modeling |
| Later Blocks (6 adapters) | 65.8 | Lacks local details |
| Early Blocks (6 adapters) | 61.7 | Lacks global patterns |

### Key Findings

1. Chrono improves upon DINOv2 by **+15.7%p** on DAVIS (ViT-B) and outperforms TSM-ResNet-18 by **+20.9%p**, HTML tags demonstrating the immense value of temporal adapters.
2. Without a refiner, Chrono even **outperforms TAPIR (with refiner)** by 11.4%p on RGB-Stacking while achieving $12.5\times$ the throughput of TAPIR.
3. When combined with the LocoTrack refiner, performance increases even further, surpassing all state-of-the-art trackers.

## Highlights & Insights

- **Simple yet Profound Design Philosophy**: Instead of spending significant effort designing complex refiner architectures, it is better to directly endow the features with temporal awareness. A high-quality backbone paired with a simple soft-argmax can match or even exceed complex multi-stage systems.
- **Compelling PCA Visualizations**: Chrono's features are temporally smoother and exhibit finer-grained differentiation within the same object, whereas DINOv2 features jitter over time and show uniform, undifferentiated representations within an object.
- **Elegant Balance of Efficiency and Accuracy**: The temporal adapter adds only 16.2M to 26.0M parameters (relative to DINOv2 itself), with an inference time approximately $3\times$ that of DINOv2, but yielding massive accuracy improvements.

## Limitations & Future Work

- Training relies solely on synthetic data (Kubric), which may lead to domain gaps when transitioning to real-world data.
- The fixed window size of $N=13$ may not be optimal for all motion velocity scales.
- Current efforts focus primarily on positional accuracy, leaving occlusion prediction less explored.
- The computational overhead of the temporal adapter may become a bottleneck when handling extremely long videos.

## Related Work & Insights

- **Difference from DINO-Tracker**: DINO-Tracker requires 1 hour of test-time optimization for each video, whereas Chrono performs direct inference after a single training phase.
- **Inspiration for Adapter Design**: The bottleneck structure and residual connections are borrowed from ResNet, and the local window attention is inspired by Longformer.
- **Broader Takeaways**: Extending pretrained vision backbones for temporal tasks represents a general paradigm. It is applicable not only to point tracking but can also be generalized to tasks such as video segmentation and optical flow.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Embedding temporal awareness directly into pretrained feature backbones is simple yet highly effective, and the viewpoint is inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 3 datasets and 2 modes, robust comparison with refiners, multi-dimensional ablation, and detailed visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, with a highly logical flow running from the motivation to the method and experiments.
- **Value**: ⭐⭐⭐⭐ Successfully demonstrates that "good features > complex post-processing," providing practical and meaningful insights for the point tracking community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better](tracktention_leveraging_point_tracking_to_attend_videos_faster_and_better.md)
- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[CVPR 2025\] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion](posetraj_pose-aware_trajectory_control_in_video_diffusion.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)

</div>

<!-- RELATED:END -->
