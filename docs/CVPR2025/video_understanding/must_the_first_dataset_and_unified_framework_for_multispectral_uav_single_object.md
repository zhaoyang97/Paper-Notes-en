---
title: >-
  [Paper Note] MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking
description: >-
  [CVPR 2025][Video Understanding][Multispectral Tracking] This work proposes the first large-scale multispectral UAV single object tracking dataset, MUST (250 sequences, 43K frames, 8 spectral bands), and designs a unified framework named UNTrack to fuse spectral, spatial, and temporal features, achieving efficient and robust tracking via an asymmetric Transformer and a spectral prompt encoder.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Multispectral Tracking"
  - "UAV Object Tracking"
  - "Asymmetric Transformer"
  - "Spectral Prompt"
  - "Background Elimination"
date: 2026-05-08
content_hash: d658610ae17a12d5
---

# MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking

**Conference**: CVPR 2025  
**arXiv**: [2503.17699](https://arxiv.org/abs/2503.17699)  
**Code**: [https://github.com/q2479036243/MUST-Multispectral-UAV-Single-Object-Tracking](https://github.com/q2479036243/MUST-Multispectral-UAV-Single-Object-Tracking)  
**Area**: Video Understanding  
**Keywords**: Multispectral Tracking, UAV Object Tracking, Asymmetric Transformer, Spectral Prompt, Background Elimination

## TL;DR

This work proposes the first large-scale multispectral UAV single object tracking dataset, MUST (250 sequences, 43K frames, 8 spectral bands), and designs a unified framework named UNTrack to fuse spectral, spatial, and temporal features, achieving efficient and robust tracking via an asymmetric Transformer and a spectral prompt encoder.

## Background & Motivation

UAV tracking faces unique challenges:
1. **Small Object Issue**: UAVs fly at altitudes of 20-250m, where the target scale accounts for only $3 \times 10^{-5}$ of the image.
2. **Insufficiency of RGB Features**: In scenarios with cluttered backgrounds, similar colors, and occlusions, spatial features (color, texture) cannot effectively distinguish the target from the background.
3. **Drastic Appearance Variations**: Large changes in UAV viewpoints can cause the target's appearance to differ significantly from the initial template.

**Advantages of Multispectral Images (MSI)**: Spectral information reflects the intrinsic reflective characteristics of objects. Even if spatial features are similar, the spectral curves of different materials vary significantly, which can be utilized for effective differentiation. Moreover, spectral features remain relatively stable during tracking.

**Limitations of Prior Work**:
- Lack of large-scale, highly challenging multispectral UAV tracking datasets.
- Existing MSI tracking datasets (e.g., HOT, with only 50 sequences) are small-scale and focus on ground-based scenarios.
- Existing UAV trackers are mostly based on Siamese architectures and only process RGB data.
- Prior methods simply adapt RGB trackers to MSI without fully exploiting spectral features.

## Method

### Overall Architecture

UNTrack unitarily processes three types of inputs: spectrum prompts, the initial template, and sequential search frames. The core comprises three components: (1) a Unified Asymmetric Transformer to extract features and model relationships; (2) a Spectrum Prompt Encoder to generate and update spectral feature prompts; and (3) a dual-branch prediction head to output target locations.

### Key Designs

1. **Unified Asymmetric Transformer**: It concatenates the tokens of the spectral prompt P, template T, and search frame S into $F = [P; T; S]$ for global attention interaction. However, the core innovation lies in the **Asymmetric Attention**—analysis reveals that among the complete $9$ blocks of the attention map, only the self-attentions of P/T/S (Blocks 1, 5, 9) and the cross-attentions of S to P+T (Blocks 7, 8) are useful for tracking, whereas the cross-attention between P and T is not only useless but also introduces noise. Therefore, it is pruned to: P self-attention + T self-attention + S attention to all, significantly reducing computation (FLOPs reduced by 19%) while improving AUC by 4.1%.

2. **Spectral Background Eliminate**: Utilizing the attention map $A_S$ of the search tokens, the confidence of each region belonging to the target is quantified as $B = \|\text{AvgPool}(A_S)\|$, retaining only the top-$\rho$ regions to progressively eliminate background areas. $\rho$ is dynamically adjusted during training using a cosine annealing strategy. **Effect**: This further reduces computational overhead (106.9G vs. 137.5G FLOPs), boosting inference speed from 24.3 to 38.0 FPS.

3. **Spectrum Prompt Encoder**: It concatenates the prompt token $\bar{P}$ output by the asymmetric Transformer and the globally pooled template $\text{GPool}(\bar{T})$. Through two FC layers (a squeeze-and-excitation operation, similar to SE attention) and an MLP, it encodes a new spectral prompt $\hat{P}$ to record the material spectral properties of the target. **Mechanism**: The spectral prompt is continuously updated across frames, providing a stable spectral reference that is untainted by external interference for subsequent tracking.

### Loss & Training

- **Classification**: Focal Loss $\mathcal{L}_{cls}$
- **Regression**: $\ell_1$ Loss + GIoU Loss: $\mathcal{L} = \mathcal{L}_{cls} + 5 \mathcal{L}_1 + 2 \mathcal{L}_{GIoU}$
- AdamW optimizer, initial learning rate of $10^{-4}$, decayed by 10 times after 30 epochs, with 50 epochs in total.
- Search frame size $384 \times 384$, template size $192 \times 192$, batch size 24.
- **Parameter Reconstruction Strategy**: Extends ImageNet RGB pre-trained parameters to 8-channel MSI via interpolation (improving average AUC by 10%+).
- The number of continuous search frames is set to 2 (achieving the optimal speed-accuracy trade-off).

## Key Experimental Results

### Main Results — MUST Dataset

| Method | Type | AUC ↑ | SR₀.₅ ↑ | Pre ↑ | PreN ↑ |
|------|------|-------|---------|-------|--------|
| SiamRPN | Siamese | 38.9 | 49.5 | 56.7 | 50.1 |
| OSTrack₃₈₄ | One-stream | 44.5 | 56.7 | 63.9 | 56.8 |
| ODTrack | One-stream | 46.3 | 58.4 | 67.6 | 60.4 |
| OSTrack*₃₈₄ | + Parameter Reconstruction | 55.1 | 69.6 | 73.3 | 68.8 |
| **UNTrack*** | **+ Parameter Reconstruction** | **59.7** | **75.8** | **79.2** | **74.8** |

UNTrack* outperforms OSTrack*₃₈₄ by **4.6% AUC** and **5.9% Pre**.

### Cross-Dataset Validation

| Method | HOT AUC ↑ | HOT Pre ↑ | FPS ↑ | GOT10K AO ↑ |
|------|-----------|-----------|-------|-------------|
| SiamHYPER | 67.8 | 94.5 | 27.7 | - |
| HANet | 68.8 | 94.8 | 21.2 | - |
| TMTNet | 69.9 | 92.8 | 12.6 | - |
| **UNTrack** | **70.4** | **93.7** | **37.0** | **77.3** |

Achieves SOTA accuracy and the fastest speed (37 FPS) on the HOT dataset. Performance is on par with SOTA on GOT10K.

### Ablation Study

| Configuration | AUC ↑ | FLOPs | FPS ↑ | Description |
|------|-------|-------|-------|------|
| No Prompt (Baseline) | 53.8 | 140.9G | 23.9 | T+S self-attention only |
| + Prompt Full Attention | 55.4 | 169.8G | 21.4 | Introduces spectral prompt but with high computation |
| + Asymmetric Attention | 59.5 | 137.5G | 24.3 | Substantial improvement after pruning |
| + Background Elimination | **59.7** | **106.9G** | **38.0** | Maintains accuracy, increases efficiency by 58% |

| Spectral Prompt Source | AUC ↑ | Params | Description |
|-------------|-------|--------|------|
| No Prompt | 53.9 | 93.1M | Baseline |
| Random Initialization (No Update) | 53.7 | 101.3M | Introduces irrelevant information |
| Asymmetric Transformer Output | 55.0 | 101.3M | Historical updates are effective |
| **Spectral Prompt Encoder** | **59.7** | 112.6M | Fully utilizes spectral features |

### Key Findings

- The parameter reconstruction strategy (RGB $\to$ MSI interpolation) improves trackers by 10%+ AUC on average at an extremely low cost.
- Asymmetric attention pruning reduces FLOPs by 19% while improving AUC by 4.10%, demonstrating that the pruned attention blocks indeed contain noise.
- Continuous search of 2 frames yields a 5.1% AUC gain over a single frame, but using 3 or more frames brings no further improvement while doubling the computation.
- Spectral information offers the most pronounced advantages in challenges like motion blur (MB), camera motion (CM), and similar color (SC).
- UNTrack can relocate the target using historical prompts after out-of-view (OV) events, outperforming other trackers.

## Highlights & Insights

- **The First Multispectral UAV Tracking Dataset**: 250 sequences, 43K frames, 8 bands (390-950nm coverage including visible and near-infrared), and 12 challenging attributes, filling a critical gap in the field.
- **Elegant Conception of Asymmetric Attention**: Analyzing the $3 \times 3$ grid of the attention map unveils blocks irrelevant to tracking; pruning them simultaneously improves speed and accuracy.
- **Continuous Update Mechanism for Spectral Prompts**: Aligns with the memory bank concept but is much lighter, requiring only 1 token to encode the target's spectral characteristics.
- **Simple yet Practical Parameter Reconstruction Strategy**: Extends RGB pre-training to MSI solely through interpolation.

## Limitations & Future Work

- The frame rate of the multispectral camera is limited to 5 FPS, which restricts application in high-speed motion scenarios.
- The availability of only 8 spectral bands is limiting; more bands could yield richer representations.
- The dataset contains only 250 sequences, leaving a huge scale gap compared to standard RGB tracking datasets.
- There are no pre-trained MSI parameters, necessitating reliance on the RGB $\to$ MSI interpolation strategy.
- Downstream search and expansion tasks, such as multi-object tracking or spectral anomaly detection, are not explored.

## Related Work & Insights

- OSTrack / ODTrack: Representative one-stream trackers, serving as the primary baselines.
- HOT Dataset (50 sequences): The only existing MSI tracking dataset, which is excessively small.
- UAV123 / VisDrone: Regular RGB UAV tracking datasets that lack a spectral dimension.
- The proposed asymmetric attention concept can be extended to other multi-modal fusion tasks (e.g., RGB-T tracking).
- The squeeze-and-excitation design of the spectral prompt encoder draws inspiration from SENet, but is newly applied along the spectral dimension.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneer dataset + specialized framework, opening up a new direction for multispectral UAV tracking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive dataset analysis, main experiments, cross-dataset verification, multiple ablation studies, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Detailed descriptions of both the dataset and methodology, with illustrations aiding the attention analysis.
- **Value**: ⭐⭐⭐⭐⭐ The dataset and framework provide foundational contributions to the UAV tracking community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](../../CVPR2026/video_understanding/uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2025\] Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking](similarity-guided_layer-adaptive_vision_transformer_for_uav_tracking.md)
- [\[CVPR 2025\] OmniTrack: Omnidirectional Multi-Object Tracking](omnidirectional_multi-object_tracking.md)
- [\[CVPR 2026\] TGTrack: Temporal Generative Learning for Unified Single Object Tracking](../../CVPR2026/video_understanding/tgtrack_temporal_generative_learning_for_unified_single_object_tracking.md)
- [\[CVPR 2025\] Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)

</div>

<!-- RELATED:END -->
