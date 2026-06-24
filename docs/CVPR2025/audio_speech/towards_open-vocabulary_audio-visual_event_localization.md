---
title: >-
  [Paper Note] Towards Open-Vocabulary Audio-Visual Event Localization
description: >-
  [CVPR 2025][Audio & Speech][Open-vocabulary] This work formally defines the Open-Vocabulary Audio-Visual Event Localization (OV-AVEL) task, constructs the OV-AVEBench benchmark containing 24,800 videos across 67 event categories, and proposes two baselines (training-free and fine-tuning) based on ImageBind. Fine-tuning with just a single-layer temporal Transformer achieves an average performance of 57.8%.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Open-vocabulary"
  - "Audio-visual event localization"
  - "ImageBind"
  - "Temporal modeling"
  - "Zero-shot generalization"
date: 2026-05-08
content_hash: 52c1526d0111e02e
---

# Towards Open-Vocabulary Audio-Visual Event Localization

**Conference**: CVPR 2025  
**arXiv**: [2411.11278](https://arxiv.org/abs/2411.11278)  
**Code**: [https://github.com/jasongief/OV-AVEL](https://github.com/jasongief/OV-AVEL)  
**Area**: Audio & Speech / Multimodal  
**Keywords**: Open-vocabulary, Audio-visual event localization, ImageBind, Temporal modeling, Zero-shot generalization

## TL;DR

This work formally defines the Open-Vocabulary Audio-Visual Event Localization (OV-AVEL) task, constructs the OV-AVEBench benchmark containing 24,800 videos across 67 event categories, and proposes two baselines (training-free and fine-tuning) based on ImageBind. Fine-tuning with just a single-layer temporal Transformer achieves an average performance of 57.8%.

## Background & Motivation

**Background**: Audio-Visual Event Localization (AVEL) aims to identify and classify temporally localized events in videos that are audio-visually consistent. Existing methods (e.g., CMRA, PSP, MM-Pyramid) train and test on a closed-set of categories, assuming only seen event categories will be encountered during testing.

**Limitations of Prior Work**: It is impractical to enumerate all event categories in real-world scenarios, as new auditory and visual events emerge constantly. Existing closed-set methods suffer from catastrophic performance drops when facing unseen categories (e.g., MM-Pyramid drops from 68.6% to 29.9%). Furthermore, existing AVE datasets consist of only 4,143 videos and 28 categories, which is too limited in scale.

**Key Challenge**: The model must acquire zero-shot generalization capabilities for unseen event categories, but specialized training for new tasks can disrupt the general representation capacity of pre-trained models.

**Goal**: Formally define the OV-AVEL task (training only on seen categories, while testing on both seen and unseen categories), construct a large-scale benchmark, and establish simple yet effective baselines.

**Key Insight**: Leverage the unified cross-modal representation space of ImageBind—where audio, visual, and text features are aligned in a shared space, making it naturally suited for open-vocabulary scenarios.

**Core Idea**: ImageBind unified representation + geometric mean fusion + single-layer temporal Transformer = a simple and elegant open-vocabulary audio-visual event localization baseline.

## Method

### Overall Architecture

A video is divided into $T=10$ segments of 1 second each. ImageBind is used to extract the audio feature $\bm{a}_t$ and visual feature $\bm{v}_t$ (dimension $d=1024$) for each segment, as well as the text feature $\bm{e}_c$ for all candidate categories. The audio-text and visual-text similarity matrices are computed separately, and then fused using geometric mean to predict the event category for each segment.

### Key Designs

1. **Training-free Baseline (Zero-Shot)**:

    - **Function**: Performs open-vocabulary event localization without any training.
    - **Mechanism**: Computes the cosine similarity of the audio and visual features of each segment with the text features of all categories. For each segment, the top predicted categories from the audio and visual modalities are obtained. If they match, the segment is classified as that event; otherwise, it is classified as background (the "other" category).
    - **Design Motivation**: Leverages the cross-modal alignment capability of pre-trained ImageBind, where audio-visual consistency checking naturally filters out unimodal noise.

2. **Fine-tuning Baseline (Lightweight Temporal Enhancement)**:

    - **Function**: Enhances temporal modeling with a few learnable parameters to significantly improve performance.
    - **Mechanism**: Inserts $L=1$ layer of learnable Transformer blocks after ImageBind's audio and visual encoders to capture temporal dependencies, training only on seen categories. Audio-visual fusion is performed using the geometric mean $S_{ave}' = \sqrt{S_{ae}' \odot S_{ve}'}$ (square root of the Hadamard product), and the predicted category corresponds to the highest probability.
    - **Design Motivation**: Compared to the arithmetic mean, the geometric mean is more robust to unimodal false alarms—if one modality yields a low score, the geometric mean is significantly dragged down (57.8% vs 39.0%).

3. **Critical Role of the "Other" Category**:

    - **Function**: Handles background segments that do not belong to any candidate category.
    - **Mechanism**: Appends the text feature of an "other" category to the end of the candidate text list, allowing the model to classify background segments as "other" instead of misclassifying them into specific event categories.
    - **Design Motivation**: Without the "other" category, performance drops drastically from 57.8% to 47.0% (-10.8%), as background segments are forced into irrelevant event categories.

### Loss & Training

Single cross-entropy loss: $\mathcal{L} = \text{CE}(S_{ave}', Y')$. Optimized using Adam with $lr=5\times 10^{-5}$ and $batch\ size=32$, training for only 5 epochs. Learnable parameters are only 8.4M (1-layer Transformer), with the remaining ImageBind parameters frozen.

## Key Experimental Results

### Main Results

Average performance on OV-AVEBench (mean of Acc/Seg-F1/Event-F1):

| Method | Seen Avg | Unseen Avg | Total Avg |
|------|----------|------------|-----------|
| Video-LLaMA2 | 40.9 | 38.6 | 39.3 |
| CLIP&CLAP | 41.6 | 41.8 | 41.7 |
| Training-free (Ours) | 45.5 | 47.0 | 46.6 |
| **Fine-tuning (Ours)** | **62.9** | **55.8** | **57.8** |

Performance collapse of closed-set methods from seen to unseen categories when transferred to OV-AVEL:

| Closed-set Method | Seen Avg | Unseen Avg | Drop |
|---------|----------|------------|------|
| MM-Pyramid | 68.6 | 29.9 | -38.7 |
| AVE method | 65.4 | 34.0 | -31.4 |
| CMRA | 59.4 | 31.1 | -28.3 |

### Ablation Study

| Configuration | Total Avg | Description |
|------|-----------|------|
| Full model ($L=1$ Transformer) | **57.8** | Best |
| Without "other" category | 47.0 | Drops by 10.8 points |
| Arithmetic mean fusion | 39.0 | Drops by 18.8 points |
| Linear layer instead of Transformer | 39.0 (unseen only 28.3) | Temporal modeling is crucial for generalization |
| $L=2$ Transformer layers | 56.9 | Overfitting |
| Cross-modal temporal interaction only | 46.5 | Harms performance instead |

### Key Findings
- **Geometric mean fusion is crucial**: It outperforms arithmetic mean by 10.8 points (57.8% vs. 39.0%) because it naturally suppresses unimodal false alarms (a low score in one modality pulls down the product).
- **Temporal Transformer is key to generalization**: The Linear layer performs better on seen categories (65.8 vs 62.9) but collapses on unseen categories (28.3 vs 55.8), indicating that temporal modeling helps extract more generic event patterns.
- **Cross-modal interaction is counterproductive**: ImageBind already provides sufficient cross-modal semantic alignment. Additional cross-modal attention layers may destroy the pre-trained representation.
- **High data efficiency**: Achieving 57.7% using only 25% of the training data (~3,300 videos), which is close to the 57.8% achieved with 100% of the data.

## Highlights & Insights

- **Task definition is the major contribution**: The formal definition of OV-AVEL as a new task and the construction of a large-scale benchmark hold more long-term value than the method itself. The dataset with 24,800 videos and 67 event categories significantly outperforms existing AVE datasets.
- **Simplicity is effective**: The combination of three minimalist designs—a 1-layer Transformer, geometric mean fusion, and the "other" category—establishes a strong baseline. This sets a clear starting line for future works.
- **Intuition behind geometric mean fusion**: Audio-visual consistency is naturally suited for multiplication—the fusion score is high only when both modalities are "confident" about the presence of an event. This insight is transferable to other multimodal fusion scenarios.

## Limitations & Future Work

- **Significant Seen-Unseen performance gap**: After fine-tuning, the seen performance is 62.9% vs. 55.8% for unseen, suggesting that there is still room for improving generalization.
- **Low Event-level F1**: The Event-F1 on unseen data is only 47.5%, indicating insufficient temporal boundary localization precision.
- **Fixed temporal segmentation**: The design of dividing a 10-second video into 10 equal segments is too rigid to handle variable-length events and cross-segment events.
- **Relatively simple method**: Marked by clear baseline characteristics, necessitating more complex temporal modeling and cross-modal interaction designs in future work.
- **Reliance on ImageBind**: The overall approach heavily relies on the representation quality of ImageBind, and the performance with other foundation models remains unexplored.

## Related Work & Insights

- **vs. Closed-set AVEL Methods (CMRA/PSP)**: Closed-set methods experience a steep performance drop of 28–38 points on unseen categories, whereas the fine-tuning baseline for OV-AVEL drops by only 7 points, demonstrating the necessity of the open-vocabulary framework.
- **vs. Video-LLaMA2**: Applying large video-language models directly to OV-AVEL yields average results (39.3%), likely because the temporal event localization capability of LLMs is inferior to specialized approaches.
- **vs. AudioCLIP/CLAP**: Audio-text-only alignment methods show average performance in open-vocabulary scenarios due to the lack of complementary information from the visual modality.

## Rating
- Novelty: ⭐⭐⭐⭐ The definition of a new task and the construction of a large-scale benchmark are the core contributions; the method itself is of baseline level.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparison with various baselines, adaptation of closed-set methods, and comprehensive ablation studies (fusion strategies, number of layers, data volume, and modal interaction).
- Writing Quality: ⭐⭐⭐⭐ Clear task definition and comprehensive evaluation protocols.
- Value: ⭐⭐⭐⭐ Paves a new open-vocabulary direction for the audio-visual event localization community, offering lasting value in both the benchmark and baselines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)
- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](../../ECCV2024/audio_speech/label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[CVPR 2025\] Improving Sound Source Localization with Joint Slot Attention on Image and Audio](improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)
- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[CVPR 2025\] Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation](crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)

</div>

<!-- RELATED:END -->
