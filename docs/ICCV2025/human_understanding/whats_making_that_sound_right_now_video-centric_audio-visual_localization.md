---
title: >-
  [Paper Note] What's Making That Sound Right Now? Video-centric Audio-Visual Localization
description: >-
  [Human Understanding] This paper proposes AVATAR, a video-level audio-visual localization benchmark, and TAVLO, a temporally-aware model that addresses the neglect of temporal dynamics in conventional AVL methods through high-resolution temporal modeling.
tags:
  - Human Understanding
date: 2026-05-08
content_hash: 550e1edfcc5884e4
---

# What's Making That Sound Right Now? Video-centric Audio-Visual Localization

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2507.04667](https://arxiv.org/abs/2507.04667)
- **Code**: [Project Page](https://hahyeon610.github.io/Video-centric_Audio_Visual_Localization/)
- **Area**: Human Understanding
- **Keywords**: Audio-visual localization, temporal modeling, video-level annotation, multi-scenario evaluation, self-supervised learning

## TL;DR

This paper proposes AVATAR, a video-level audio-visual localization benchmark, and TAVLO, a temporally-aware model that addresses the neglect of temporal dynamics in conventional AVL methods through high-resolution temporal modeling.

## Background & Motivation

Audio-visual localization (AVL) aims to identify sound-producing objects within visual scenes. Existing work suffers from two critical limitations:

**Limitations of image-level association**: Existing benchmarks (e.g., Flickr-SoundNet, VGGSS) adopt image-level annotation — annotators watch a complete video but label the sounding object in only a single frame, treating that frame as representative of the entire video. Consequently, existing methods process only single-frame inputs and entirely disregard temporal dynamics. In realistic scenarios, tracking moving sound sources and handling dynamic changes requires spatiotemporal modeling.

**Oversimplified assumptions**: Existing benchmarks assume that sounding objects are always visible and typically involve only a single sound source. Real-world scenarios, however, frequently include multiple simultaneous sound sources and off-screen sounding objects. Some studies partially address this through mismatched audio-image negative pairs or synthesized mixed audio, but coverage remains insufficient.

## Method

### Overall Architecture

The paper presents two core contributions: (1) the AVATAR benchmark dataset; and (2) TAVLO, a temporally-aware AVL model.

### AVATAR Benchmark

AVATAR is constructed from VGGSound via a semi-automatic annotation pipeline, comprising 5,000 videos, 24,266 frames, and 80 categories. A key innovation is the introduction of four evaluation scenarios:

- **Single-sound**: Only one instance in the frame produces sound; evaluates one-to-one audio-visual correspondence.
- **Mixed-sound**: Multiple simultaneous sound sources; requires distinguishing and associating sounds with the correct visual sources.
- **Multi-entity**: Multiple visually similar objects where only one produces sound; requires spatiotemporal reasoning.
- **Off-screen**: The sounding object is outside the frame; evaluates the model's ability to avoid false positives.

The semi-automatic annotation pipeline consists of three stages:
1. Candidate video selection: approximately 70k raw videos from VGGSound are filtered by resolution, frame rate, and duration, yielding 39k videos.
2. Automatic clip and frame sampling: RMS energy is used to detect audio-active regions; the sharpest frames are selected via Laplacian filtering.
3. Model-driven annotation: YoloV8 detection + CAV-MAE audio classification + SAM instance segmentation + human verification.

### TAVLO Model

TAVLO explicitly incorporates temporal information for spatiotemporal AVL. The core architecture includes:

**Modality-specific feature encoding**:

A visual encoder $f_v$ (ResNet-18) extracts frame-level features $\mathbf{V} = f_v(V) \in \mathbb{R}^{T \times H \times W \times D_f}$. An audio encoder $f_a$ employs rectangular 2D CNN kernels with sizes $K_w = \lfloor W_a / T \rfloor, K_h = H_a$, ensuring each audio segment is aligned with the corresponding visual frame, producing $\mathbf{A} = f_a(A) \in \mathbb{R}^{T \times D_f}$.

**Positional encoding**: Spatial positional encoding $\text{Pos}_s \in \mathbb{R}^{T \times H \times W \times D_s}$ and temporal positional encoding $\text{Pos}_t \in \mathbb{R}^{T \times D_t}$ are defined as:

$$\tilde{\mathbf{V}} = [\mathbf{V} + \text{Pos}_s; \text{Pos}_t] \in \mathbb{R}^{T \times H \times W \times D}$$
$$\tilde{\mathbf{A}} = [\mathbf{A}; \text{Pos}_t] \in \mathbb{R}^{T \times D}$$

Spatial encodings are added element-wise to visual features; temporal encodings are concatenated across both modalities, yielding final dimension $D = D_f + D_t$.

**AST attention module**: Audio-visual features are concatenated as $\mathbf{Z}^0 = [\tilde{\mathbf{A}}; \tilde{\mathbf{V}}] \in \mathbb{R}^{T \times (1+H \cdot W) \times D}$. A factorized attention strategy is adopted to avoid the quadratic complexity of applying self-attention directly over flattened video features:

- **Spatial attention**: Multi-head self-attention over the $1 + H \cdot W$ dimension, capturing intra-frame cross-modal audio-visual interactions.
- **Temporal attention**: Multi-head self-attention over the $T$ dimension, modeling cross-frame temporal dependencies.

### Loss & Training

A cross-modal multiple-instance contrastive learning loss adapted from EZ-VSL is employed with two modifications: (1) a temporal component is introduced to define visual bags at the frame level; (2) mean similarity instead of maximum similarity is used for negative sample bags, reducing the dominance of noisy instances:

$$\mathcal{L}_{a \rightarrow v} = -\mathbb{E}_{t,i}\left[\log \frac{\exp(\text{p}_i^t)}{\exp(\text{p}_i^t) + \sum_{j \neq i}^B \exp(\text{n}_{ij}^t)}\right]$$

The final loss is bidirectional: $\mathcal{L} = \mathcal{L}_{a \rightarrow v} + \mathcal{L}_{v \rightarrow a}$.

## Key Experimental Results

### Main Results

| Method | Single-sound CIoU(%) | Single-sound AUC(%) | Mixed-sound CIoU(%) | Multi-entity CIoU(%) | Off-screen TN(%) |
|------|---------------------|--------------------|--------------------|---------------------|-----------------|
| SLAVC(144k) | 9.07 | 10.60 | 6.31 | 6.41 | 96.46 |
| EZ-VSL(10k) | 9.66 | 11.07 | 8.16 | 6.87 | 96.91 |
| EZ-VSL(144k) | 10.92 | 12.22 | 6.97 | 5.80 | 96.47 |
| SSL-TIE(144k) | 13.10 | 14.23 | 5.19 | 5.50 | 90.82 |
| **TAVLO(10k)** | **13.42** | **14.08** | **14.13** | **12.08** | 91.18 |

### Robustness under Cross-event Scenarios

| Method | Total CIoU(%) | Cross-event CIoU(%) | Δ |
|------|--------------|--------------------|----|
| EZ-VSL(full) | 10.50 | 5.26 | -5.24 |
| SSL-TIE(144k) | 10.39 | 5.03 | -5.36 |
| **TAVLO(10k)** | **13.37** | **13.04** | **-0.33** |

### Key Findings

1. TAVLO trained on only 10k samples surpasses baselines trained on 144k samples.
2. The performance advantage is most pronounced in Mixed-sound and Multi-entity scenarios, with CIoU improvements of approximately 6–7 percentage points.
3. In cross-event scenarios, baseline CIoU drops by 3–5 percentage points, whereas TAVLO drops by only 0.33, demonstrating that temporal modeling is critical for tracking dynamic sound sources.
4. Qualitative analysis shows that TAVLO correctly distinguishes the actively sounding drum in multi-drum scenes and handles off-screen-to-on-screen speaker transitions.

## Highlights & Insights

1. **Precise problem formulation**: The first systematic extension of AVL to the video level, defining four comprehensive evaluation scenarios.
2. **Elegant factorized attention design**: The AST module processes video features in linear time via spatial-temporal factorized attention, avoiding quadratic complexity.
3. **High data efficiency**: Surpassing 144k-trained baselines with only 10k training samples highlights the inductive bias advantage conferred by temporal modeling.
4. **Practical semi-automatic annotation pipeline**: Combining model-assisted labeling with human verification balances annotation quality and efficiency.

## Limitations & Future Work

1. The approach assumes at least partial audio-visual alignment; independent localization of off-screen sounds remains an open problem.
2. The benchmark does not provide scenario-specific optimization strategies for each evaluation setting.
3. Threshold selection in off-screen evaluation has a considerable impact on results.

## Related Work & Insights

- **AVL methods**: Self-supervised approaches including EZ-VSL, SSL-TIE, and SLAVC; semi-supervised method DMT.
- **AVL benchmarks**: Flickr-SoundNet, VGGSS, AVSBench, all lacking video-level temporal annotations.
- **Video understanding**: Spatiotemporal attention factorization strategies (TimeSformer, ViViT).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The video-level AVL benchmark and temporally-aware model represent pioneering contributions to the field.
- **Technical Depth**: ⭐⭐⭐⭐ — Temporal encoding and AST attention design are well-motivated and technically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four-scenario comprehensive evaluation and cross-event analysis are convincing.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/human_understanding/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)

<!-- RELATED:END -->
