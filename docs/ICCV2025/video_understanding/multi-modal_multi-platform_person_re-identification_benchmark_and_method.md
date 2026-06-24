---
title: >-
  [Paper Note] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method
description: >-
  [ICCV 2025][Video Understanding][Person Re-Identification] This paper presents MP-ReID, the first multi-modal multi-platform person re-identification benchmark encompassing three modalities (RGB, infrared, thermal) and two platforms (ground and UAV), along with a unified prompt learning framework, Uni-Prompt ReID, which leverages modality-aware, platform-aware, and visual-enhanced prompts to substantially improve ReID performance under complex real-world conditions.
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Person Re-Identification"
  - "Multi-modal"
  - "Multi-platform"
  - "Prompt Learning"
  - "CLIP"
date: 2026-05-08
content_hash: 2fb3b04c8c14f7ea
---

# Multi-modal Multi-platform Person Re-Identification: Benchmark and Method

**Conference**: ICCV 2025
**arXiv**: [2503.17096](https://arxiv.org/abs/2503.17096)  
**Code**: [GitHub](https://mp-reid.github.io/)  
**Area**: Video Understanding
**Keywords**: Person Re-Identification, Multi-modal, Multi-platform, Prompt Learning, CLIP

## TL;DR

This paper presents MP-ReID, the first multi-modal multi-platform person re-identification benchmark encompassing three modalities (RGB, infrared, thermal) and two platforms (ground and UAV), along with a unified prompt learning framework, Uni-Prompt ReID, which leverages modality-aware, platform-aware, and visual-enhanced prompts to substantially improve ReID performance under complex real-world conditions.

## Background & Motivation

Conventional ReID research has been largely confined to **single-modality (RGB) + fixed-camera** settings, which are inadequate for the increasingly heterogeneous sensor deployments found in real urban environments. Consider a 24/7 urban pedestrian surveillance system comprising:

- **Ground RGB cameras**: daytime scenarios
- **Infrared/thermal sensors**: nighttime or adverse lighting
- **Unmanned Aerial Vehicles (UAVs)**: dynamic tracking with flexible viewpoints

Such a multi-modal + multi-platform configuration introduces three compounding challenges: (1) modality gap (appearance discrepancy among RGB, infrared, and thermal imaging), (2) platform gap (viewpoint and resolution differences between ground-level and aerial capture), and (3) the extreme difficulty when both gaps are present simultaneously.

**Limitations of existing datasets**:
- Cross-modal datasets (SYSU-MM01, LLCM) cover only RGB + infrared and use only ground cameras.
- UAV datasets (AG-ReID) cover only the RGB modality.
- **No existing dataset simultaneously addresses multiple modalities and multiple platforms.**

This critical gap motivates the construction of MP-ReID and the design of a corresponding unified learning framework.

## Method

### Overall Architecture

Uni-Prompt ReID is built upon the CLIP vision-language model and is fine-tuned through carefully designed multi-part textual prompts. The framework consists of three categories of learnable prompts and a visual-enhanced network that injects image features into the text prompt space.

### Key Designs

1. **MP-ReID Dataset Construction**

   The dataset spans 3 modalities × 2 platforms:
    - Ground RGB (6 Hikvision 1920×1080 full-color cameras)
    - Ground infrared (6 cameras in infrared night-vision mode)
    - UAV RGB (DJI Mavic 3T, 3840×2160)
    - UAV thermal (DJI Mavic 3T thermal camera, 640×512)

   Dataset scale: 1,930 identities, 136,156 annotated bounding boxes, 14 cameras, and over 13 hours of total video footage. UAV data was collected at three altitudes (5 m / 7 m / 10 m) with pitch angles ranging from 30° to 80°. All data underwent facial mosaicking and original footage deletion to protect privacy.

2. **Uni-Prompt Multi-Part Textual Prompts**

   The textual prompt is formed by concatenating three parts:

   $t_i(a) = X_1(a) \cdots X_M(a) \; P_1(a) \cdots P_R(a) \; M_1(a) \cdots M_B(a), \text{person}_i$

    - **Specific ReID Prompt** ($X$): encodes individual-specific information (identity level)
    - **Modality-Aware Prompt** ($M$): captures modality-specific details (RGB vs. infrared vs. thermal)
    - **Platform-Aware Prompt** ($P$): incorporates platform-specific context (ground vs. aerial)

3. **Visual-Enhanced Network**

   A lightweight neural network $g_\theta(\cdot)$ maps the image feature $a$ to context vectors:

   $\sigma = (\sigma_X, \sigma_P, \sigma_M) = g_\theta(a)$

   These are added to the corresponding prompts: $S_m(a) = [S]_m + \sigma_S$

   Intuition: visual features of infrared images inherently contain modality cues, which can guide the modality-aware prompts to specialize toward the infrared domain.

### Loss & Training

**Two-stage training**:

- Stage 1: Freeze modality and platform prompts; learn the Specific ReID Prompt using CLIP-ReID's $\mathcal{L}_{i2t} + \mathcal{L}_{t2i}$.
- Stage 2: Freeze the ReID Prompt; learn the remaining prompts using modality-level and platform-level contrastive losses:

$$\mathcal{L}_{\text{Uni-Prompt}} = \mathcal{L}_{mi2t} + \mathcal{L}_{mt2i} + \mathcal{L}_{pi2t} + \mathcal{L}_{pt2i}$$

Each term is a contrastive loss (InfoNCE form) aligning features to modality and platform labels respectively. Data augmentation includes random erasing ($p=0.5$), random horizontal flipping, and random cropping.

## Key Experimental Results

### Main Results

**Average results across three MP-ReID benchmark settings**

| Method | Cross-Platform Rank-1 | Cross-Modal Rank-1 | Cross-Modal+Platform Rank-1 | Avg. Rank-1 | Avg. mAP |
|---|---|---|---|---|---|
| CAJ | 40.36 | 45.34 | 10.62 | 32.11 | 21.51 |
| CAJ+ | 47.60 | 58.16 | 21.51 | 42.42 | 30.61 |
| AGW | 53.68 | 51.88 | 19.21 | 41.59 | 30.56 |
| DEEN | 60.05 | 69.59 | 27.59 | 52.41 | 39.33 |
| OTLA-ReID | 73.24 | 68.12 | 29.31 | 56.89 | 43.03 |
| **Uni-Prompt** | **78.77** | **72.26** | **43.16** | **64.73** | **58.45** |

Average Rank-1 improves by **+7.87%** and mAP by **+15.42%**. The gain is most pronounced in the hardest cross-modal + cross-platform setting (+13.85% Rank-1).

### Ablation Study

| Configuration | Cross-Platform R1 | Cross-Modal R1 | Cross-Modal+Platform R1 | Avg. R1 | Avg. mAP |
|---|---|---|---|---|---|
| Base (ReID Prompt) | 77.01 | 61.11 | 28.40 | 55.51 | 47.98 |
| +Modality-Aware | 77.18 | 67.34 | 31.57 | 58.70 | 51.67 |
| +Platform-Aware | 78.62 | 70.31 | 40.66 | 63.20 | 57.48 |
| +Visual-Enhanced (Full) | **78.77** | **72.26** | **43.16** | **64.73** | **58.45** |

### Key Findings

- **Cross-modal + cross-platform is the most challenging setting**: existing methods degrade drastically (CAJ achieves only 10.62% Rank-1), while Uni-Prompt reaches 43.16%, demonstrating the necessity of dedicated design for jointly handling both gaps.
- **Modality-aware prompts** are most effective in the cross-modal setting (+6.23% Rank-1) with minimal impact on the cross-platform setting.
- **Platform-aware prompts** contribute most substantially in the cross-modal + cross-platform setting (+9.09% Rank-1), constituting the key component for the hardest scenario.
- **The visual-enhanced network** yields marginal but consistent gains across all settings, with a 2.50% improvement in the cross-modal + cross-platform setting, confirming that visual cues provide auxiliary guidance for prompt learning.
- Existing baseline methods perform acceptably only in single-gap settings where ground RGB data is available, and degrade significantly once UAV capture and multi-modal inputs are jointly introduced.

## Highlights & Insights

- **The first multi-modal multi-platform ReID benchmark** fills a critical gap—1,930 identities, 14 cameras, 3 modalities, and 2 platforms—combining both scale and diversity.
- **The unified prompt learning framework** elegantly decomposes modality and platform information into separate learnable prompts, avoiding complex feature fusion networks.
- The two-stage training strategy (learning identity prompts first, then modality/platform prompts) resembles curriculum learning, ensuring the model first establishes an identity concept before learning cross-domain alignment.
- Privacy protection measures are comprehensive: facial mosaicking, deletion of raw footage, ethics committee approval, and public notification.

## Limitations & Future Work

- Dataset scale is constrained by the high cost of multi-modal multi-platform collection (1,930 identities vs. 4,101 in MSMT17).
- Evaluation is conducted on a single dataset, leaving transferability to other benchmarks unverified.
- Wearable-device platforms and event camera modalities are not addressed; the authors encourage future extensions in these directions.
- The visual-enhanced network design is relatively simple (lightweight linear mapping); more sophisticated adapters may yield further improvements.
- The low resolution of the UAV thermal camera (640×512) degrades YOLOX tracking performance, necessitating substantial manual annotation effort.

## Related Work & Insights

- CLIP-ReID and DAPrompt serve as the foundational prompt learning baselines; CoCoOp inspires the design of visually-conditioned prompts.
- Cross-modal datasets such as SYSU-MM01 and LLCM are limited to RGB + infrared with ground-only cameras; AG-ReID covers aerial imagery but is restricted to the RGB modality.
- The multi-platform design of MP-ReID has direct application value for smart city and public safety scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multi-modal multi-platform dataset design constitutes the primary contribution; the method represents an incremental extension of existing prompt learning frameworks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 12 experimental settings + detailed ablations + 10-run averaging; the evaluation protocol is rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ The dataset description is thorough and the methodological exposition is clear.
- **Value**: ⭐⭐⭐⭐⭐ The dataset and benchmark make a significant contribution to the person re-identification community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](../../NeurIPS2025/video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_multimodal_llms_token_merging_pruning.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)

</div>

<!-- RELATED:END -->
