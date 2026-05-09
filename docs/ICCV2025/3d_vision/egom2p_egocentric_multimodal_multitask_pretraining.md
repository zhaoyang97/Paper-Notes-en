---
title: >-
  [Paper Note] EgoM2P: Egocentric Multimodal Multitask Pretraining
description: >-
  [ICCV 2025][3D Vision][Egocentric Vision] EgoM2P is the first large-scale multimodal multitask model for egocentric 4D understanding. It unifies four modalities — RGB video, depth, gaze, and camera trajectory — within a temporally-aware masked modeling framework, matching or surpassing task-specific models on multiple downstream tasks while being an order of magnitude faster.
tags:
  - ICCV 2025
  - 3D Vision
  - Egocentric Vision
  - Multimodal Pretraining
  - Multitask Learning
  - Masked Modeling
  - Gaze Prediction
  - Camera Tracking
  - Depth Estimation
  - Video Generation
date: 2026-05-08
content_hash: 1aa041e09cc0c97b
---

# EgoM2P: Egocentric Multimodal Multitask Pretraining

**Conference**: ICCV 2025
**arXiv**: [2506.07886](https://arxiv.org/abs/2506.07886)
**Code**: [Project Page](https://egom2p.github.io/)
**Area**: 3D Vision
**Keywords**: Egocentric Vision, Multimodal Pretraining, Multitask Learning, Masked Modeling, Gaze Prediction, Camera Tracking, Depth Estimation, Video Generation

## TL;DR

EgoM2P is the first large-scale multimodal multitask model for egocentric 4D understanding. It unifies four modalities — RGB video, depth, gaze, and camera trajectory — within a temporally-aware masked modeling framework, matching or surpassing task-specific models on multiple downstream tasks while being an order of magnitude faster.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Multimodal signals in egocentric vision (RGB, depth, camera pose, gaze, etc.) are critical for augmented reality, robotics, and human-computer interaction. However, building large-scale egocentric multimodal multitask models faces unique challenges:

**Data Heterogeneity**: Modality coverage varies greatly across devices and scenarios — some datasets include gaze but no depth, others include depth but no gaze — resulting in severe missing-modality issues.

**Infeasibility of Pseudo-Labels**: Generating pseudo-labels for modalities such as gaze and head-mounted camera trajectories is often impractical, making standard supervised learning difficult to scale.

**Spatiotemporal Complexity**: Dynamic camera motion and complex spatiotemporal structures in egocentric video pose additional challenges for existing multimodal foundation models.

**Limitations of Prior Work**: Third-person video models are ill-suited for the egocentric perspective; existing egocentric foundation models focus solely on video question answering, neglecting body-related modalities and 3D/4D reconstruction capabilities.

**Limitations of Image-Level Models**: Multimodal foundation models such as 4M operate on single images and lack temporal consistency guarantees.

### Paper Goals

**Goal**: ### Overall Architecture

EgoM2P is built on the T5-Base architecture with an encoder-decoder design.

## Method

### Overall Architecture

EgoM2P is built on the T5-Base architecture with an encoder-decoder design. The pipeline consists of three stages: data curation → modality tokenization → masked pretraining.

### Key Design 1: Data Curation Pipeline

Eight heterogeneous datasets are integrated (EgoExo4D, HoloAssist, HOT3D, ARCTIC, TACO, H2O, EgoGen, etc.):

- **Segmentation**: Clips are uniformly cut into $T$-frame segments and encoded as high-quality mp4 files.
- **Annotation**: RollingDepth is used to generate depth pseudo-labels; EgoGen provides ~30 hours of synthetic data; missing gaze annotations are not pseudo-labeled.
- **Normalization**: Unified to 30 FPS; depth is represented as inverse depth and normalized per sequence; gaze is projected to 2D; camera trajectories are standardized to OpenCV camera-to-world convention with the first frame as reference.

### Key Design 2: Temporal Tokenizers

- **Video Modalities** (RGB/Depth): Cosmos Tokenizer with $4\times$ temporal and $8\times$ spatial compression.
- **Gaze/Camera Trajectory**: Transformer-based VQ-VAE
    - N-dimensional convolution with $2\times$ temporal downsampling + N-dimensional positional encoding + 12-layer Transformer
    - Cosine-similarity vector quantization with modality-specific codebooks
    - Training loss: reconstruction + codebook + commitment losses

### Key Design 3: Multimodal Masked Pretraining

Four key adaptations are made relative to the image-level masked modeling of 4M:

1. **Token Count Expansion**: From 256 visible tokens in 4M to 2048, accommodating 5000+ tokens per video.
2. **Dataset Balancing**: ~4 billion vs. 13 million gaze tokens → sampling proportional to dataset size combined with uniform modality sampling.
3. **Missing Modality Handling**: Placeholder tokens with masking, rather than pseudo-labels.
4. **Temporal Embeddings**: 1D sinusoidal-cosine encoding for gaze/camera, 3D sinusoidal-cosine encoding for video, plus learnable modality-type embeddings.

### Inference

After pretraining, arbitrary-to-arbitrary modality prediction is achieved by sampling different token combinations, supporting parallel inference.

## Key Experimental Results

### Supported Downstream Tasks

1. **Gaze Estimation**: Predicting 2D gaze points.
2. **Egocentric Camera Tracking**: 6DoF trajectory estimation.
3. **Monocular Depth Estimation**: Estimating depth from RGB video.
4. **Conditional Video Generation**: Generating video conditioned on multimodal inputs.

### Main Results

- **Matches or surpasses** task-specific models on all four tasks.
- Inference speed is **an order of magnitude faster**.
- A single model covers both understanding and generation.

### Training Scale

### Main Results

| Item | Data |
|------|------|
| Number of Datasets | 8 (real + synthetic) |
| Total Training Tokens | ~4 billion |
| Gaze Tokens | ~13 million |
| Video Tokens / Sample | >5000 |
| Visible Token Count | 2048 |

### Key Findings

1. EgoGen synthetic data yields notable improvements for depth estimation.
2. Sampling proportional to dataset size combined with uniform modality sampling is the most stable strategy.
3. Missing modalities can be effectively predicted without pseudo-labels.
4. Temporal tokenization design significantly outperforms image-level processing.

## Highlights & Insights

1. **Elegant Handling of Missing Modalities**: Masked placeholder tokens naturally accommodate missing modalities, offering a pragmatic solution to the heterogeneity of egocentric data.
2. **Unified Tokenizer Architecture**: Adding new modalities requires only training a new tokenizer and codebook, conferring strong extensibility.
3. **Unified Understanding and Generation**: A single model supports both perception and generation tasks.
4. **Speed Advantage**: Parallel inference is an order of magnitude faster than running multiple specialized models, which is critical for real-time AR/VR applications.
5. **Data Engineering**: The complete pipeline for integrating eight heterogeneous datasets constitutes a practical engineering contribution.

## Limitations & Future Work

1. Gaze data is far less abundant than other modalities (13 million vs. 4 billion tokens), potentially limiting gaze prediction accuracy.
2. Only four modalities are covered; audio, touch, and hand pose are not included.
3. Video resolution is constrained by the compression capacity of the Cosmos Tokenizer.
4. Detailed per-task numerical comparisons against specialized models are provided in the supplementary material.
5. The datasets are biased toward indoor and manipulation scenarios; generalization to outdoor settings has not been thoroughly validated.

## Related Work & Insights

- **Image-Level Multimodal Foundation Models**: 4M (masked modeling) → EgoM2P extends this paradigm to temporal video.
- **Video Foundation Models**: VideoMAE, Cosmos → primarily third-person perspective.
- **Egocentric Foundation Models**: EgoVLP/v2 (video-language QA) → lack body-related modalities and 3D capabilities.
- **Multimodal Binding**: ImageBind → aligns embedding spaces only.

## Rating

- **Novelty**: 8/10 — First multimodal multitask model targeting egocentric 4D understanding.
- **Technical Depth**: 7/10 — Extension of 4M is well-motivated but not revolutionary.
- **Experimental Thoroughness**: 6/10 — Detailed numbers are partially deferred to supplementary material.
- **Writing Quality**: 9/10 — Single-model multitask design with fast inference has direct value for AR/VR.
- **Value**: 7.5/10

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views](../../NeurIPS2025/3d_vision/look_and_tell_a_dataset_for_multimodal_grounding_across_egocentric_and_exocentri.md)
- [\[ICCV 2025\] TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation](tokenunify_scaling_up_autoregressive_pretraining_for_neuron_segmentation.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[ICCV 2025\] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)
- [\[NeurIPS 2025\] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span](../../NeurIPS2025/3d_vision/gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)

<!-- RELATED:END -->
