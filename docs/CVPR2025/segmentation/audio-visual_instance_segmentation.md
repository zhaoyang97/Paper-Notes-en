---
title: >-
  [Paper Note] "Audio-Visual Instance Segmentation"
description: >-
  [CVPR2025][Segmentation][Paper Note] Academic paper note for "Audio-Visual Instance Segmentation".
tags:
  - CVPR2025
  - Segmentation
date: 2023-10-27
content_hash: bd2c9f4f09139cd7
---
# Audio-Visual Instance Segmentation

## Background & Motivation

Humans naturally integrate auditory and visual information in daily life to locate and identify sound sources in their environment. For instance, when hearing a dog barking, we automatically focus our attention on the dog in the visual field and precisely perceive its contour. This capability of **joint audio-visual perception** holds significant application value in fields such as robot navigation, video editing, and autonomous driving.

Existing audio-visual learning tasks primarily include:

| Task | Output | Granularity | Temporal Modeling |
|------|------|------|---------|
| Sound Source Localization | Heatmap | Region-level | Single-frame |
| Audio-Visual Segmentation | Semantic mask | Pixel-level | Single-frame/Short-sequence |
| Audio-Visual Source Separation | Separated audio | Source-level | Multi-frame |
| **AVIS (Ours)** | **Instance mask + ID** | **Instance-level** | **Full-video** |

Limitations of existing methods:

**Sound Source Localization** only provides coarse, region-level localization and fails to generate precise pixel-level segmentations.

**Audio-Visual Segmentation** performs semantic-level segmentation but does not distinguish among different individuals of the same category.

**No existing method** simultaneously performs instance-level segmentation and cross-frame tracking at the video level.

This work introduces a new task, **Audio-Visual Instance Segmentation (AVIS)**: given a video and its corresponding audio, the goal is to perform instance-level segmentation of all sounding objects and track them across frames. In addition, the AVISeg benchmark dataset and the AVISM baseline method are established.

## Method

### AVISeg Benchmark Dataset

AVISeg is the first large-scale annotated dataset for audio-visual instance segmentation:

| Attribute | Value |
|------|------|
| Number of videos | 926 |
| Annotated masks | 90,000+ |
| Object categories | 26 |
| Average video duration | 8.3 seconds |
| Resolution | 720p/1080p |
| Annotation frequency | Annotated every 5 frames |
| Data sources | YouTube, ActivityNet, VGGSound |

#### Data Collection and Annotation Pipeline

1. **Video Filtering**: Filter videos containing distinct sounding sources from large-scale video databases.
2. **Audio Event Annotation**: Annotate the temporal boundaries and categories of audio events in each video.
3. **Instance Segmentation Annotation**: Delineate instance-level segmentation masks for each sounding object.
4. **Cross-Frame Association**: Assign consistent instance IDs to masks of the same object across different frames.
5. **Quality Control**: Implement multiple rounds of verification to ensure annotation quality.

### AVISM Baseline Method

AVISM (Audio-Visual Instance Segmentation and tracking Model) consists of two core modules:

#### Frame-Level Sound Localizer (FLSL)

FLSL is responsible for locating sounding objects in each frame:

- **Audio Feature Extraction**: Extract audio embeddings $\mathbf{a}_t \in \mathbb{R}^{d}$ using a pre-trained AudioSet model.
- **Visual Feature Extraction**: Extract visual feature maps $\mathbf{V}_t \in \mathbb{R}^{d \times H \times W}$ using ResNet-50 / Swin-T.
- **Cross-Modal Attention**:

$$\mathbf{A}_{t} = \text{softmax}\left(\frac{\mathbf{a}_t \mathbf{V}_t^T}{\sqrt{d}}\right) \mathbf{V}_t$$

- **Instance Prediction**: Predict instance masks and categories using a Mask2Former-style decoder based on attention-enhanced features.

#### Video-Level Sounding Tracker (VLST)

VLST is responsible for cross-frame instance association and tracking:

- **Instance Embedding**: Generate an appearance embedding $\mathbf{e}_i^t$ for each detected instance.
- **Audio Consistency Constraint**: Ensure that the correlation between the same instance and its corresponding audio remains consistent across different frames.
- **Matching Strategy**: Perform bipartite graph matching by combining appearance similarity and audio correlation:

$$\text{cost}(i, j) = \alpha \cdot \text{IoU}(m_i^t, m_j^{t+1}) + \beta \cdot \cos(\mathbf{e}_i^t, \mathbf{e}_j^{t+1}) + \gamma \cdot \text{audio\_sim}(i, j)$$

### Evaluation Metrics

The AVIS task adopts evaluation metrics adapted from VIS (Video Instance Segmentation) with additional audio-related constraints:

| Metric | Meaning | Description |
|------|------|------|
| FSLA (Frame-level Sound Localization Accuracy) | Frame-level Sound Localization Accuracy | Measures whether the sounding objects in each frame are correctly segmented |
| HOTA (Higher Order Tracking Accuracy) | Higher Order Tracking Accuracy | Comprehensively measures detection and association quality |
| mAP | Mean Average Precision | Standard instance segmentation metric |
| IDsw | ID Switches | Measures tracking consistency |

## Experimental Results

### Main Results

| Method | FSLA↑ | HOTA↑ | mAP↑ | IDsw↓ |
|------|-------|-------|------|-------|
| Mask2Former (Vision-only) | 31.25 | 48.92 | 27.8 | 145 |
| AVS + SORT | 35.67 | 52.34 | 31.2 | 128 |
| TAPIS | 38.91 | 56.18 | 34.5 | 107 |
| **AVISM (Ours)** | **42.78** | **61.73** | **38.9** | **82** |

AVISM significantly outperforms combinations of existing methods across all metrics, validating the superiority of the architecture specifically designed for the AVIS task.

### Ablation Study

| Configuration | FSLA↑ | HOTA↑ |
|------|-------|-------|
| Full AVISM | 42.78 | 61.73 |
| w/o Audio input | 32.14 | 49.85 |
| w/o Cross-modal attention | 37.92 | 55.41 |
| w/o Audio consistency constraint | 40.15 | 58.62 |
| w/o VLST (No tracking) | 42.78 | 52.37 |

Audio cue is a core contributing factor (FSLA drops by 10.64% upon its removal), and the VLST tracking module significantly contributes to HOTA.

### Category Analysis

| Category | FSLA↑ | Difficulty |
|------|-------|------|
| Musical instruments (piano, guitar) | 52.3 | Easy |
| Animals (dog, cat, bird) | 43.7 | Medium |
| Vehicles (car, motorcycle) | 38.9 | Medium |
| Human activities (speaking, clapping) | 31.2 | Hard |

The musical instruments category is the easiest to localize due to stationary sound source positions and highly distinctive visual features. Human activities present the greatest difficulty because there may be multiple sound sources (people) in the video with complex motion patterns.

## Summary & Future Work

This work proposes AVIS, a new audio-visual understanding task, establishes the AVISeg benchmark (926 videos, over 90K masks, 26 categories), and designs the AVISM baseline method (FLSL + VLST). AVISM achieves 42.78% FSLA and 61.73% HOTA, providing a solid foundation for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[CVPR 2025\] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](../../ICCV2025/segmentation/implicit_counterfactual_learning_for_audio-visual_segmentation.md)

</div>

<!-- RELATED:END -->
