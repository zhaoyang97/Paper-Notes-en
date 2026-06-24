---
title: >-
  [Paper Note] Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation
description: >-
  [CVPR 2025][Audio & Speech][audio-visual understanding] This paper proposes Crab, a unified audio-visual scene understanding model. By constructing the AV-UIE dataset (200K samples) with explicit reasoning processes, it clarifies the collaborative relationships across tasks. Combined with interaction-aware LoRA (multi-head LoRA) designed to learn different audio-visual interaction patterns, Crab outperforms specialized models across multiple tasks.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "audio-visual understanding"
  - "unified model"
  - "interaction-aware LoRA"
  - "instruction tuning"
  - "multi-task learning"
date: 2026-05-08
content_hash: 3c17f6b6ca41256f
---

# Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation

**Conference**: CVPR 2025  
**arXiv**: [2503.13068](https://arxiv.org/abs/2503.13068)  
**Code**: [GeWu-Lab/Crab](https://github.com/GeWu-Lab/Crab)  
**Institution**: Renmin University of China / Tsinghua University / Tencent PCG
**Area**: Audio-visual understanding / Multimodal learning  
**Keywords**: audio-visual understanding, unified model, interaction-aware LoRA, instruction tuning, multi-task learning

## TL;DR
This paper proposes Crab, a unified audio-visual scene understanding model. By constructing the AV-UIE dataset (200K samples) with explicit reasoning processes, it clarifies the collaborative relationships across tasks. Combined with interaction-aware LoRA (multi-head LoRA) designed to learn different audio-visual interaction patterns, Crab outperforms specialized models across multiple tasks.

## Background & Motivation

**Background**: Audio-visual scene understanding comprises various tasks: temporal localization (AVE, AVVP), spatio-temporal reasoning (AVQA), spatial localization (ARIG), and pixel-level understanding (AVS, Ref-AVS). Although humans possess a unified ability to understand multiple tasks, most existing works design specialized models for individual tasks.

**Limitations of Prior Work**:
   - **Simple Joint Training**: Interference arises between multiple tasks due to the heterogeneous nature of audio-visual data and the complex relationships among tasks.
   - **Existing Unified Models** (e.g., VideoLLaMA, GroundingGPT): These models lack explicit cooperation mechanisms between tasks, leading to limited performance.
   - Existing datasets only provide simple labels (word-level), failing to capture the reasoning and collaborative relationships among tasks.

**Key Challenge**: How to handle temporal, spatial, and pixel-level multi-granularity tasks simultaneously within a single model while avoiding task interference.

**Key Insight**: Establishing explicit task cooperation across both data and model levels.

**Core Idea**: Explicit reasoning dataset (AV-UIE) + interaction-aware LoRA (multi-head) = Unified Audio-Visual Understanding.

## Method

### Overall Architecture
- **Visual Encoder**: CLIP-ViT-L/14, extracting patch-level features.
- **Audio Encoder**: BEATs, extracting acoustic features.
- **Segmentation Decoder**: SAM decoder.
- **Language Model**: LLaMA-2-7b-Chat.
- **Multimodal Bridge**: Audio Q-Former + Visual Q-Former (32 query tokens each).

### Key Designs

1. **AV-UIE Dataset (Audio-Visual Unified Instruction-tuning with Explicit reasoning)**

    - **Function**: Constructing a unified instruction-tuning dataset of 200K samples that includes explicit reasoning processes.
    - **Mechanism**: Expanding the simple labels of existing datasets into an instruction format that incorporates reasoning chains.
    - **Task Coverage**: Temporal localization, spatio-temporal reasoning, spatial localization, pixel-level segmentation, and referring segmentation.
    - **Effect**: Clarifying the cooperative relationships between tasks (e.g., "temporal localization aids spatial localization").

2. **Interaction-aware LoRA**

    - **Function**: Inserting multi-head LoRA into all linear layers of the LLM to learn different audio-visual interaction patterns.
    - **Structure**: Shared matrix $\mathbf{A}$ + $n=3$ LoRA heads (independent $\mathbf{B}$ matrices).
    - The three heads focus respectively on: temporal interaction / spatial interaction / pixel-level interaction.
    - rank = 8.
    - **Design Motivation**: Different tasks require focusing on different interaction dimensions of audio-visual data.
    - **Output**: The weighted sum of the three heads serves as the final adaptation.

3. **Mask Decoder Design**

    - Two groups of `<MASK>` tokens correspond to visual features at two scales (14th layer and second-to-last layer).
    - 3 tokens per group.
    - Supports audio-visual semantic segmentation (AVSS) and referring audio-visual segmentation (Ref-AVS).

### Loss & Training
- **Phase 1: Pre-training Alignment**
    - Visual branch: Video-LLaVA data.
    - Audio branch: AudioCaps data.
    - Segmentation branch: LVIS data.
    - Global batch size 256, 3 epochs.
- **Phase 2: Instruction Tuning**
    - AV-UIE dataset, mixing all tasks.
    - Trainable components: Three multimodal branches + interaction-aware LoRA (encoders are frozen).
    - Global batch size 512, 5 epochs.

Loss function: $\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice} + \lambda_{ce}\mathcal{L}_{ce}$

## Key Experimental Results

### Comprehensive Comparison with Specialized Models

| Task | Metric | Prev. SOTA | Crab |
|------|------|----------|------|
| AVE Temporal Localization | Acc | MM-Pyramid 77.80 | **80.15** |
| AVQA Spatio-temporal Reasoning | Avg | TSPM 76.79 | **78.94** |
| ARIG Spatial Localization | cIoU | FNAC 27.15 | **41.78** |
| ARIG Spatial Localization | AUC | FNAC 0.31 | **0.42** |
| AVS-MS3 Pixel Segmentation | mIoU | AVSegFormer 58.40 | **58.21** |

### AVQA Subcategory Comparison

| Method | Audio | Visual | Audio-Visual | Avg |
|------|-------|--------|-------------|-----|
| LAVISH | 75.97 | 80.22 | 71.26 | 74.46 |
| TSPM | 76.91 | 83.61 | 73.51 | 76.79 |
| **Crab** | 76.58 | **90.73** | **74.13** | **78.94** |

The performance improvement on the Visual subcategory is significant (90.73 vs. 83.61), which likely benefits from enhanced visual understanding supported by explicit reasoning.

### Key Findings
- Each LoRA head automatically learns distinct audio-visual understanding capabilities (validated through visualization).
- Although the temporal localization task has the smallest proportion in AV-UIE, it still achieves significant improvement due to cross-task collaboration.
- Comparable performance is achieved against VALOR (78.94 vs. 78.90), which was trained on the million-scale VALOR-1M dataset, using significantly less data.

## Highlights & Insights
- The **multi-head LoRA** design is simple yet effective: the shared A matrix reduces the parameter footprint, while the multi-head B matrices capture different interaction patterns.
- **Explicit reasoning data** is far more effective than simple labels, enabling the model to understand "why different tasks need to cooperate."
- Outperforms specialized methods by a large margin on spatial localization (ARIG) (+14.63 cIoU), demonstrating the cross-task transfer advantage of a unified model.
- The unified model paradigm is more elegant and efficient than assembling a pipeline of specialized models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)
- [\[ICLR 2026\] UALM: Unified Audio Language Model for Understanding, Generation and Reasoning](../../ICLR2026/audio_speech/ualm_unified_audio_language_model_for_understanding_generation_and_reasoning.md)
- [\[CVPR 2025\] Towards Open-Vocabulary Audio-Visual Event Localization](towards_open-vocabulary_audio-visual_event_localization.md)
- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[NeurIPS 2025\] DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis](../../NeurIPS2025/audio_speech/deepasa_an_object-oriented_multi-purpose_network_for_auditory_scene_analysis.md)

</div>

<!-- RELATED:END -->
