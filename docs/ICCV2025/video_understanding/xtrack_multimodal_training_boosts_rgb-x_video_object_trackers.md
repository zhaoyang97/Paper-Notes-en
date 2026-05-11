---
title: >-
  [Paper Note] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers
description: >-
  [ICCV 2025][Video Understanding][Multimodal Tracking] This paper proposes XTrack, which employs a Mixture of Modal Experts (MeME) framework and a soft-routing classifier to enable cross-modal knowledge sharing across RGB…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Multimodal Tracking"
  - "Mixture of Experts"
  - "Cross-modal Knowledge Transfer"
  - "Video Object Tracking"
  - "RGB-X"
date: 2026-05-08
content_hash: cfddb8064c82aa34
---

# XTrack: Multimodal Training Boosts RGB-X Video Object Trackers

**Conference**: ICCV 2025
**arXiv**: [2405.17773](https://arxiv.org/abs/2405.17773)
**Code**: [Available](https://arxiv.org/abs/2405.17773)
**Area**: Video Understanding
**Keywords**: Multimodal Tracking, Mixture of Experts, Cross-modal Knowledge Transfer, Video Object Tracking, RGB-X

## TL;DR

This paper proposes XTrack, which employs a Mixture of Modal Experts (MeME) framework and a soft-routing classifier to enable cross-modal knowledge sharing across RGB-D/T/E modalities, allowing inference with a single modality to benefit from multimodal training knowledge, achieving an average precision gain of 3%.

## Background & Motivation

Multimodal perception (depth, thermal infrared, and event cameras) can compensate for the limitations of RGB-only tracking in extreme scenarios, but faces critical constraints:

**Data Scarcity**: No comprehensive dataset containing all modalities simultaneously exists; typically only paired RGB-X data is available.

**Rigid Branch Design**: Existing unified models (e.g., ViPT, UnTrack) activate predetermined branches at inference time based on the input modality, with no cross-modal interaction.

**Cross-modal Knowledge Waste**: Strict modality isolation prevents cross-modal knowledge transfer — for instance, in RGB-Depth sequences under fast-motion, low-light conditions, the model should learn knowledge that transcends single-modality boundaries.

Core Insight: **Similar samples across different modalities share more transferable knowledge.** When a "weak" classifier fails to accurately distinguish which modality a sample belongs to, it indicates that the sample resides at an optimal position for cross-modal knowledge sharing (i.e., minimal domain gap). This "confusion" is not a defect but rather a signal for knowledge transfer.

## Method

### Overall Architecture

XTrack inserts MeME modules after each attention block and FFN in a frozen RGB base tracker (OSTrack/SeqTrack). MeME bidirectionally processes RGB and X modality tokens to enhance feature modeling:
$$T_{rgb}^{attn} = T_{rgb}^l + Attn(T_{rgb}^l) + MeME(T_{rgb}^l, T_x^l)$$
$$T_{rgb}^{l+1} = T_{rgb}^{attn} + FFN(T_{rgb}^{attn}) + MeME(T_{rgb}^{attn}, T_x^{attn})$$

### Key Designs

1. **Soft Router with Classification Loss**: The routing function $y = \sum_{i \in top\_k} p_i(T_x) \epsilon_i(T_x)$ incorporates not only the conventional expert load-balancing loss $\mathcal{L}_{balance} = \mathcal{L}_{Imp} + \mathcal{L}_{Load}$, but also a modal classification loss $\mathcal{L}_{cls}$. The classification loss preserves a degree of modality specialization within each expert, while the soft (non-rigid) classification allows cross-modal samples to access experts from other modalities — the optimal balance is achieved at a classification probability of approximately 80%.

2. **Modality-Specific Experts + Shared Experts**:

    - **Modality-Specific Experts**: Each modality is assigned $k$ experts (experiments show $k=2$ is optimal), performing feature decomposition and reconstruction in a low-dimensional space $k \ll c$.
    - **Edge-Gated Shared Experts**: Shared experts incorporate an EdgeMix gating module initialized with a Laplacian filter, using high-frequency edge information as a natural cross-modal common prior: $Out = (\sigma(EdgeMix(XW_1)) \cdot XW_2)W_3 + m_{s_k}$

3. **Modal Prompting**: The low-dimensional modal matrix $M_k$ output by MeME serves as a gating signal to modulate RGB tokens: $Out = ((X_i W_5 \cdot \sigma(X_m W_6))W_7 + I_k)W_8$, rendering RGB features modality-aware.

### Loss & Training

- Tracking loss: Inherits the IoU + L1 losses from the RGB base tracker.
- MoE loss: $\mathcal{L}_{moe} = \mathcal{L}_{cls} + \lambda \cdot \mathcal{L}_{balance}$
- Only MeME parameters are trained; the base tracker is frozen.
- Training settings: batch size 32, learning rate 4e-4, 90 epochs with a 10× decay after epoch 78.
- Training data: DepthTrack (RGB-D) + LasHeR (RGB-T) + VisEvent (RGB-E), with only one RGB-X pair available at a time.

## Key Experimental Results

### Main Results

RGB-Depth Tracking:

| Method | DepthTrack F-score | VOT-RGBD22 EAO | VOT-RGBD22 Acc. | VOT-RGBD22 Rob. |
|--------|--------------------|-----------------|-----------------|-----------------|
| ViPT | 59.4 | 72.1 | 81.5 | 87.1 |
| UnTrack | 61.0 | 72.1 | 82.0 | 86.9 |
| SDSTrack | 61.4 | 72.8 | 81.2 | 88.3 |
| **XTrack-B** | **61.5** | **74.0** | **82.1** | **88.8** |
| **XTrack-L** | **64.8** | **74.0** | **82.8** | **88.9** |

RGB-Thermal Tracking (LasHeR/RGBT234):

| Method | LasHeR Pr | LasHeR Sr | RGBT234 MPR | RGBT234 MSR |
|--------|----------|----------|-------------|-------------|
| ViPT | 65.1 | 52.5 | 83.5 | 61.7 |
| SDSTrack | 66.5 | 53.1 | 84.8 | 62.5 |
| OneTracker | 67.2 | 53.8 | 85.7 | 64.2 |
| **XTrack-B** | **69.1** | **55.7** | **87.4** | **64.9** |
| **XTrack-L** | **73.1** | **58.7** | **87.8** | **65.4** |

RGB-Event Tracking (VisEvent):

| Method | Pr | Sr |
|--------|-----|-----|
| ViPT | 75.8 | 59.2 |
| OneTracker | 76.7 | 60.8 |
| SDSTrack | 76.7 | 59.7 |
| **XTrack-B** | **77.5** | **60.9** |
| **XTrack-L** | **80.5** | **63.3** |

### Ablation Study

Component Analysis:

| Shared Expert | Modal Expert | DepthTrack F-score | LasHeR Pr | VisEvent Pr |
|--------------|-------------|-------------------|----------|-------------|
| ✓ | - | 59.1 | 67.8 | 76.5 |
| - | ✓ | 57.6 | 68.0 | 76.2 |
| **✓** | **✓** | **61.5** | **69.1** | **77.5** |

Gains from Multimodal Joint Training:

| Training Modalities | VisEvent Pr | DepthTrack F-score | LasHeR Pr |
|--------------------|------------|-------------------|----------|
| Baseline (w/o MeME) | 69.5 | 52.9 | 51.5 |
| Event only | 76.3 | 45.5 | 58.1 |
| Event + Depth | 76.6 | 60.8 | 58.1 |
| **E + D + T** | **77.5** | **61.5** | **69.1** |

### Key Findings

- A domain gap exists in VOT-RGBD22, yet XTrack achieves larger margins over the prior SOTA, indicating that multimodal training enhances domain generalization.
- Training with Event data alone yields reasonable zero-shot generalization to the Thermal modality (58.1 Pr), as event cameras and thermal cameras both handle illumination variation.
- A soft classification probability of 80% yields the best performance; both rigid separation and random assignment perform worse.
- Two experts per modality is optimal; one is insufficiently expressive, while three introduces internal conflicts.

## Highlights & Insights

- The core idea that "confusion is opportunity" is novel: classifier failure is not a flaw but a signal for cross-modal knowledge sharing.
- This work is the first to systematically achieve cross-modal knowledge transfer in RGB-X video object tracking.
- The Laplacian prior in EdgeMix provides a well-motivated inductive bias for shared experts.
- The experimental design is rigorous, with modalities added incrementally to clearly demonstrate each modality's contribution.

## Limitations & Future Work

- Training still relies on paired RGB-X data and cannot exploit unpaired single-modality data.
- Low-dimensional projection reduces computational cost but may result in some information loss.
- Validation on additional sensor modalities (e.g., LiDAR, SAR) has not been performed.
- At inference, modality-specific experts must still be selected, so truly modality-agnostic inference has not been achieved.

## Related Work & Insights

- The soft-routing MoE design strategy (balancing classification loss and load-balancing loss) has general applicability to other multimodal tasks.
- The theoretical framing of "modality confusion" as a knowledge transfer signal can be extended to other domains of multimodal fusion.
- Laplacian-initialized edge-sharing priors represent an initialization strategy worth exploring in other vision tasks.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](an_empirical_study_of_autoregressive_pre-training_from_videos.md)
- [\[ICCV 2025\] What You Have is What You Track: Adaptive and Robust Multimodal Tracking](what_you_have_is_what_you_track_adaptive_and_robust_multimodal_tracking.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](../../CVPR2026/video_understanding/out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](../../NeurIPS2025/video_understanding/unleashing_hour-scale_video_training_for_long_video-language_understanding.md)

</div>

<!-- RELATED:END -->
