---
title: >-
  [Paper Note] Elysium: Exploring Object-level Perception in Videos via MLLM
description: >-
  [ECCV 2024][Multimodal VLM][Multimodal Large Language Models] Elysium is proposed as an end-to-end trainable Multimodal Large Language Model (MLLM). By constructing a million-scale video object perception dataset (ElysiumTrack-1M) and designing a visual token compression network (T-Selector), it extends the object-level perception capability of MLLMs from static images to the video domain, supporting three major tasks: Single Object Tracking (SOT)…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Object Tracking"
  - "Video Object Perception"
  - "Token Compression"
  - "Large-scale Dataset"
date: 2026-05-08
content_hash: 43b85e67db14837f
---

# Elysium: Exploring Object-level Perception in Videos via MLLM

**Conference**: ECCV 2024  
**arXiv**: [2403.16558](https://arxiv.org/abs/2403.16558)  
**Code**: [Yes (GitHub)](https://github.com/Hon-Wong/Elysium)  
**Area**: Video Understanding  
**Keywords**: Multimodal Large Language Models, Object Tracking, Video Object Perception, Token Compression, Large-scale Dataset

## TL;DR

Elysium is proposed as an end-to-end trainable Multimodal Large Language Model (MLLM). By constructing a million-scale video object perception dataset (ElysiumTrack-1M) and designing a visual token compression network (T-Selector), it extends the object-level perception capability of MLLMs from static images to the video domain, supporting three major tasks: Single Object Tracking (SOT), Referring Single Object Tracking (RSOT), and Video Referring Expression Generation (Video-REG).

## Background & Motivation

Existing MLLMs (such as Shikra, MiniGPT-v2, etc.) have demonstrated excellent object-level perception capabilities (e.g., image grounding, object detection) at the image level, but research on **object-level tasks in the video domain** (such as object tracking) remains insufficient. The authors categorize video tasks into three levels based on granularity:

1. **Video-level tasks** (VideoQA, video captioning): Focus on global information, allowing feature extraction through temporal fusion operations.
2. **Frame-level tasks** (video grounding, dense video captioning): Require frame-by-frame differentiation and analysis.
3. **Object-level tasks** (SOT, MOT, VOS): Require localizing targets in each frame and maintaining temporal consistency across frames.

Applying MLLMs to video object-level tasks faces two core challenges:

- **Data scarcity**: Existing tracking datasets are limited in scale (e.g., LaSOT contains only 1.4K trajectories), which is far from sufficient to support large-scale MLLM pre-training.
- **Computational bottleneck**: Processing a large number of video frames imposes a massive visual token burden on the LLM's context window, severely limiting the number of frames that can be processed.

Existing video MLLM works (such as Video-LLaMA, VideoChat) primarily focus on video-level understanding, either losing frame-level details by compressing information through temporal fusion, or relying on external expert models for object perception (such as PG-Video-LLaVA), lacking a unified end-to-end framework. The goal of Elysium is to handle object-level tasks in videos using a pure MLLM architecture without relying on any external models.

## Method

### Overall Architecture

Elysium adopts a classic MLLM architecture: **Visual Encoder (CLIP-ViT-L) + Token Compression Module (T-Selector) + Large Language Model (Vicuna)**. For each frame $\mathbf{X}_v^i$ in a video, features $\mathbf{F}_v^i \in \mathbb{R}^{N \times C}$ are first extracted by the visual encoder, and then compressed by the T-Selector into $\mathbf{T}_v^i \in \mathbb{R}^{\alpha N \times D}$, where $\alpha \in (0,1]$ represents the compression ratio and $D$ represents the hidden dimension of the LLM.

### Key Designs

#### 1. **ElysiumTrack-1M Dataset Construction**: Addressing the lack of large-scale video object perception training data

Core Idea: Starting from the WebVid-10M video dataset, a million-scale "noun phrase-trajectory" pair dataset is generated via an automated pipeline.

The construction pipeline consists of two steps:
- **Step 1 (Noun phrase-bounding box generation)**: Parse video captions into noun phrases using spaCy, filter out virtual and plural words, and generate bounding boxes on the first, middle, and last frames using Grounding DINO, retaining results with confidence > 0.6.
- **Step 2 (Trajectory expansion)**: Generate trajectories from the first-frame bounding boxes using MixFormer, retain trajectories with confidence > 0.8, filter out drifting trajectories with Kalman filtering, and calculate the IoU of the middle and last frames (discarding those with IoU < 0.3).

Ultimately, **1.27 million** noun phrase-trajectory pairs are generated, which is 41 times larger than TrackingNet (30.6K trajectories). The entire process takes only 6 days on 24 A100 GPUs.

Two new tasks are also defined:
- **RSOT (Referring Single Object Tracking)**: Localizing and tracking targets in videos solely based on language descriptions, without requiring position priors.
- **Video-REG (Video Referring Expression Generation)**: Generating target descriptions given frame coordinates, requiring temporal perception capabilities.

#### 2. **T-Selector Token Compression Network**: Balancing Performance and Computational Efficiency

Core Idea: Based on the assumption that "videos contain redundant information", it selects the most important visual tokens through a gating mechanism instead of performing fusion across spatial dimensions.

Design Motivation: Experiments show that spatial fusion operations (such as cross-attention, concatenation) lead to a drastic degradation in performance. T-Selector avoids the information loss caused by spatial fusion by scoring each token and keeping the Top-K.

$$\mathbf{G}_v = \text{KeepTopK}(\text{Softmax}(\text{MLP}(\mathbf{F}_v)), k, \mathbf{F}_v)$$
$$\mathbf{T}_v = \text{MLP}(\mathbf{G}_v)$$

Where $k = \alpha N$, the MLP gating layer calculates the importance score of each token. After Softmax normalization, the Top-K tokens are selected, and finally another MLP maps the dimension to the LLM hidden dimension. The default setting is $\alpha N = 108$ (compressing the original 576 tokens to 108, achieving a compression ratio of approximately 5.3x).

#### 3. **Input-Output Format Design**: Efficiently Utilizing the Token Budget

- Add timestamps to the visual tokens of each frame to allow the model to distinguish consecutive frames.
- Coordinates are represented as integers in the range [0, 100). For example, "[23,45,46,72]" requires only 13 LLaMA tokens, which saves approximately half of the budget compared to Shikra's floating-point format (28 tokens).
- Design task-specific prompt templates to enhance the model's robustness to diverse question formats.

### Loss & Training

A two-stage progressive training strategy is adopted:

**Stage 1: Large-Scale Image Pre-training**
- First, freeze ViT and LLM using LLaVA-558K, and train only the T-Selector (lr=2e-3, 8 GPUs).
- Then, unfreeze all parameters for end-to-end training on mixed image data for 30K steps (lr=5e-5, 32 GPUs).

**Stage 2: High-Quality Instruction Tuning**
- Fine-tune on a mixture of high-quality image data and video data (VideoChat + ElysiumTrack-1M) for 22K steps.
- In the first 20K steps, randomly sample 2-8 frames per video with intervals of 1-60 frames (simulating different frame rates and motion speeds).
- In the last 2K steps, extend the sequence length to 32 frames per video with a batch size of 1.

During inference: VideoQA uniformly samples 16 frames; SOT/RSOT segments long videos into 8-frame clips, where adjacent clips overlap by 1 frame to initialize tracking.

## Key Experimental Results

### Main Results

**Image Grounding (RefCOCO Series)**

| Model | Tokens | RefCOCO val | RefCOCO test-A | RefCOCO+ val | RefCOCOg val |
|------|---------|-------------|----------------|--------------|--------------|
| Shikra (7B) | 256 | 87.01 | 90.61 | 81.60 | 82.27 |
| MiniGPT-v2 (7B) | 256 | 88.69 | 91.65 | 79.97 | 84.44 |
| Ferret (7B) | 608 | 87.49 | 91.35 | 80.78 | 83.93 |
| **Elysium (7B)** | **108** | **89.07** | **92.12** | **82.86** | 82.92 |

Elysium surpasses baseline methods using 256-608 tokens with only 108 visual tokens.

**Zero-Shot SOT (LaSOT Dataset)**

| Model | Zero-Shot | AUC | P | P_Norm |
|------|-----------|-----|---|--------|
| SiamRPN++ | No | 49.6 | 56.9 | 49.1 |
| DiMP | No | 56.9 | 65.0 | 56.7 |
| SiamGAT | No | 53.9 | 63.3 | 53.0 |
| **Elysium** | **Yes** | **56.1** | 61.0 | 50.1 |

Under the zero-shot setting, Elysium achieves an AUC of 56.1, which is close to the specialized tracker (DiMP 56.9).

### Ablation Study

**T-Selector Compression Ratio Ablation (RefCOCO Series Avg Precision)**

| Compression Module | Tokens $\alpha N$ | Avg Precision | Description |
|---------|---------|---------|------|
| No Compression | 256 | 76.81 | ViT@224 + MLP |
| No Compression | 576 | 81.45 | ViT@336 + MLP |
| Concat | 144 | 54.65 | Spatial fusion causes severe degradation |
| Cross-Attention (C.A.) | 144 | 49.23 | Worse spatial fusion |
| T-Selector | 1 | 43.70 | Extreme compression |
| T-Selector | 36 | 74.48 | Significant performance loss |
| T-Selector | 108 | 78.54 | **Best performance-efficiency trade-off** |
| T-Selector | 256 | 80.09 | Close to uncompressed performance |

### Key Findings

1. **Spatial fusion operations (concatenation/cross-attention) are not suitable for video object perception**: Given the same token budget, the gating selection strategy of T-Selector significantly outperforms fusion-based methods (78.54 vs. 54.65/49.23).
2. **A critical threshold exists for token compression**: Performance drops significantly when tokens are reduced from 108 to 72; 108 constitutes the optimal balance between performance and efficiency.
3. **Zero-shot SOT is feasible**: Trained on large-scale video-object data, the MLLM can closely match dedicated trackers without downstream fine-tuning.
4. **MLLM capability for small objects is limited**: Performance is suboptimal on datasets containing small targets, such as UAV123, which is constrained by the resolution of the visual encoder.

## Highlights & Insights

- **Data-driven engineering insights**: By building an automated data pipeline (combining Grounding DINO, MixFormer, and various filtering strategies), millions of high-quality object tracking annotations were generated from generic videos. This paradigm of "generating data with models, then training models with generated data" is highly valuable.
- **Token selection outperforms token fusion**: Compared to fusion methods like concatenation and cross-attention, selecting tokens based on importance preserves spatial details more effectively.
- **Token efficiency in coordinate formatting**: Representing coordinates as integers instead of floats halves the token cost, providing significant benefits in long video scenarios.

## Limitations & Future Work

- Small object tracking remains suboptimal, necessitating the introduction of high-resolution visual encoders.
- Only SOT-related tasks were explored; more complex object-level tasks such as MOT, VOS, and RVOS were not investigated.
- Dataset construction depends on the quality of Grounding DINO and MixFormer, potentially inheriting biases from these foundational models.
- The token selection of T-Selector is static, lacking adaptive token allocation policies across different frames/tasks.

## Related Work & Insights

- Compared to schemes like PG-Video-LLaVA that rely on external expert models, Elysium achieves end-to-end training with a much simpler architecture.
- The RSOT task integrates language referencing with object tracking, presenting a new paradigm for vision-language interaction.
- The construction methodology of ElysiumTrack-1M can be generalized to build datasets for other video-understanding tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematically implements video object-level perception with a pure MLLM for the first time, and proposes two new tasks: RSOT and Video-REG.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple tasks including image grounding, VideoQA, SOT, RSOT, and Video-REG with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear with well-defined problems and rich figures/tables.
- **Value**: ⭐⭐⭐⭐ — The datasets and methods lay a foundation for research on video MLLM object-level perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](../../CVPR2025/multimodal_vlm/videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[ECCV 2024\] Zero-shot Object Counting with Good Exemplars (VA-Count)](zero-shot_object_counting_with_good_exemplars.md)
- [\[ECCV 2024\] MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)
- [\[CVPR 2026\] From Failure to Feedback: Group Revision Unlocks Hard Cases in Object-Level Grounding](../../CVPR2026/multimodal_vlm/from_failure_to_feedback_group_revision_unlocks_hard_cases_in_object-level_groun.md)
- [\[ICLR 2026\] Catching the Details: Self-Distilled RoI Predictors for Fine-Grained MLLM Perception](../../ICLR2026/multimodal_vlm/catching_the_details_self-distilled_roi_predictors_for_fine-grained_mllm_percept.md)

</div>

<!-- RELATED:END -->
