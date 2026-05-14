---
title: >-
  [Paper Note] UETrack: A Unified and Efficient Framework for Single Object Tracking
description: >-
  [CVPR 2026][Video Understanding][single object tracking] This paper proposes UETrack, a unified and efficient single object tracking framework capable of handling five modalities simultaneously: RGB, Depth, Thermal…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "single object tracking"
  - "multi-modal tracking"
  - "mixture of experts"
  - "knowledge distillation"
  - "efficient inference"
date: 2026-05-08
content_hash: 0a093d8ea4f84d92
---

# UETrack: A Unified and Efficient Framework for Single Object Tracking

**Conference**: CVPR 2026
**arXiv**: [2603.01412](https://arxiv.org/abs/2603.01412)
**Code**: [https://github.com/kangben258/UETrack](https://github.com/kangben258/UETrack)
**Area**: Video Understanding
**Keywords**: single object tracking, multi-modal tracking, mixture of experts, knowledge distillation, efficient inference

## TL;DR

This paper proposes UETrack, a unified and efficient single object tracking framework capable of handling five modalities simultaneously: RGB, Depth, Thermal, Event, and Language. UETrack addresses a critical gap in efficient multi-modal tracking — existing efficient trackers are limited to RGB, while multi-modal trackers are too slow for practical deployment due to complex designs. The core contributions include: (1) Token-Pooling-based Mixture-of-Experts (TP-MoE), which replaces conventional gating mechanisms with similarity-based soft assignment to enable efficient expert collaboration and specialization; and (2) Target-aware Adaptive Distillation (TAD), which adaptively determines whether each sample is suitable for distillation, filtering out unreliable teacher signals. Evaluated across 12 benchmarks on 3 hardware platforms, UETrack achieves an optimal speed-accuracy trade-off — UETrack-B attains 69.2% AUC on LaSOT at 163/56/60 FPS on GPU/CPU/AGX respectively, with only 13M parameters.

## Background & Motivation

1. **Efficient trackers are limited to RGB**: Existing efficient tracking methods (HiT, MixFormerV2-S, etc.) are almost exclusively designed for RGB scenarios, where single-modal information is insufficient under challenging conditions such as occlusion and illumination changes.
2. **Multi-modal trackers incur high computational cost**: Multi-modal methods such as SDSTrack, OneTracker, and ViPT rely on complex modality fusion modules and large backbone models, making inference speed unsuitable for real-time deployment.
3. **Difficulty in modeling modality heterogeneity**: Efficient models with limited parameters struggle to capture complementary information and shared representations across modalities, necessitating novel architectural designs to enhance modeling capacity.
4. **Latency introduced by conventional MoE gating**: MoE-based tracking methods (MoETrack, eMoE-Tracker) employ discrete gating routing, introducing additional overhead from token sorting and inter-expert communication.
5. **Negative transfer in knowledge distillation**: On challenging samples with occlusion, distractor objects, or deformation, teacher model predictions may be unreliable, and direct distillation propagates erroneous information to the student model.
6. **Strict speed requirements for practical deployment**: Edge devices such as the Jetson AGX Xavier have limited resources, requiring tracking solutions that operate in real time across diverse hardware platforms.

## Method

### Overall Architecture

UETrack is built upon Fast-iTPN-T as a lightweight backbone, adopting the unified token encoding strategy from SUTrack to process multi-modal inputs. For Depth/Thermal/Event modalities, RGB and auxiliary modal images are concatenated along the channel dimension to form a 6-channel composite image; for the Language modality, a frozen CLIP text encoder is used for token extraction. All tokens are concatenated into a unified sequence and fed into Transformer blocks.

During training, the framework comprises a teacher model (SUTrack-B, frozen), a student model, and an Adaptive Net. At inference, only the student model is used.

### Token-Pooling-based MoE (TP-MoE)

TP-MoE replaces discrete gating with similarity-based soft assignment. The core procedure is as follows:

1. **Local aggregation**: Input tokens $\mathbf{T}_{in} \in \mathbb{R}^{L_1 \times D}$ are partitioned into $L_1/E$ subspaces (where $E$ is the number of experts), and average pooling is applied within each subspace.
2. **Expert embedding**: Aggregated tokens are projected via linear transformation and reshaped into compact expert tokens $\mathbf{T}_e \in \mathbb{R}^{L_2 \times D}$.
3. **Soft routing**: A similarity matrix $\mathbf{S} \in \mathbb{R}^{L_1 \times L_2}$ is computed between input tokens and expert tokens, and softmax produces continuous routing weights.
4. **Expert processing**: Input tokens are aggregated according to routing weights and grouped; each expert processes its respective inputs independently.
5. **Output aggregation**: Expert outputs are combined back into the original token space via weighted summation using the similarity matrix.

This attention-style soft assignment eliminates token sorting and cross-expert communication, supports fully parallel computation, and ensures stable gradient propagation through differentiable matrix operations, making it well-suited for real-time tracking.

### Target-aware Adaptive Distillation (TAD)

The core component of TAD is the Adaptive Net, which takes search region features from both student and teacher as input:
- Features from each are globally average-pooled and concatenated into a fused vector.
- An MLP reduces the dimensionality to a 2D vector, and Gumbel-Softmax produces a binary decision $\alpha \in \{0, 1\}$.
- When $\alpha=1$, distillation is performed (KL divergence + MSE loss); when $\alpha=0$, distillation is skipped.

The Adaptive Net is trained using a surrogate prediction strategy: either the teacher's or the student's prediction is selected as a proxy target, and loss is computed against the ground truth, enabling automatic assessment of distillation reliability.

### Training Objective

The student loss comprises classification (Focal), regression (GIoU + L1), task loss, and adaptive distillation loss. The Adaptive Net and the student model are updated separately to avoid gradient conflicts.

## Key Experimental Results

### Table 1: Comparison with State-of-the-Art Real-Time Methods on RGB Benchmarks

| Method | LaSOT AUC | TrackingNet AUC | GOT-10k AO | GPU FPS | CPU FPS | AGX FPS | Params (M) |
|------|:---------:|:---------------:|:----------:|:-------:|:-------:|:-------:|:---------:|
| **UETrack-B** | **69.2** | **82.7** | **72.6** | 163 | 56 | 60 | 13 |
| **UETrack-S** | 66.9 | 81.4 | 71.1 | 183 | 68 | 67 | 9 |
| **UETrack-T** | 63.4 | 78.9 | 65.3 | **221** | **83** | **77** | 6 |
| AsymTrack-B | 64.7 | 80.0 | 67.7 | 197 | 38 | 64 | – |
| HiT-Base | 64.6 | 80.0 | 64.0 | 175 | 33 | 61 | – |
| MixFormerV2-S | 60.6 | 75.8 | 61.9 | 299 | 47 | 70 | – |
| OSTrack-256* | 69.1 | 83.1 | 71.0 | 105 | 11 | 19 | – |

*OSTrack is a non-real-time method included for reference. UETrack-B outperforms AsymTrack-B by 4.5%/2.7%/4.9% on LaSOT/TrackingNet/GOT-10k, respectively.

### Table 2: Multi-Modal Tracking Performance Comparison

| Modality | Benchmark | Metric | UETrack-B | SUTrack-T | ViPT(-Tiny) | SDSTrack | AGX FPS Comparison |
|------|------|------|:---------:|:---------:|:-----------:|:--------:|:------------:|
| RGB-D | VOT-RGBD22 | EAO | 68.3 | 68.1 | 68.5 | 72.8 | 60 vs 34/20/7 |
| RGB-D | DepthTrack | F-score | 60.6 | 61.7 | 53.9 | 61.9 | 60 vs 34/20/7 |
| RGB-T | LasHeR | AUC | **55.5** | 53.9 | 47.5 | 53.1 | 60 vs 34/20/7 |
| RGB-T | RGBT234 | MSR | **64.2** | 63.8 | 58.8 | 62.5 | 60 vs 34/20/7 |

UETrack-B surpasses most non-real-time methods on the thermal infrared modality while maintaining real-time speed (60 FPS on AGX). On RGB-Depth, a performance gap relative to non-real-time state-of-the-art methods remains, though UETrack-B runs 4–8× faster.

## Highlights & Insights

- **First efficient multi-modal tracking framework**: Bridges the gap between efficient (real-time and lightweight) tracking and multi-modal tracking, handling five modalities in a unified manner without modality-specific modules.
- **Elegant TP-MoE design**: Replaces discrete gating with continuous similarity-based soft assignment, eliminating token sorting overhead and supporting fully parallel computation, thereby enhancing multi-modal modeling capacity while controlling latency.
- **Practical and effective TAD strategy**: Employs Gumbel-Softmax to realize sample-level adaptive distillation, automatically filtering unreliable teacher signals on challenging samples and addressing the negative transfer problem in knowledge distillation.
- **Thorough cross-platform deployment validation**: Speed is benchmarked on three platforms — GPU (2080Ti), CPU (i9-14900KF), and edge device (Jetson AGX Xavier) — demonstrating practical deployment value.

## Limitations & Future Work

- **Remaining performance gap in multi-modal settings**: On RGB-Depth (VOT-RGBD22) and RGB-Thermal benchmarks, UETrack-B has yet to surpass certain non-real-time methods (e.g., SeqTrackv2, BAT), leaving room for improvement in accuracy-speed trade-offs for high-precision scenarios.
- **Language modality relies on frozen CLIP**: The language encoder uses a frozen CLIP model without end-to-end optimization, potentially limiting language-guided tracking performance.
- **Fixed number of experts in TP-MoE**: The number of experts is set to 2/4/8 without exploration of dynamic expert allocation or modality-adaptive expert selection.
- **Single teacher model for distillation**: Only SUTrack-B is used as the teacher; the effects of different teacher models or teacher ensembles on distillation performance are not investigated.
- **Lack of evaluation on long videos and online update scenarios**: Experiments focus on standard benchmarks without assessing robustness in extremely long videos or dynamic scenes with drastic appearance changes.

## Related Work & Insights

- **Efficient trackers**: HiT (Kang et al.), MixFormerV2 (Cui et al.), and AsymTrack (Li et al.) pursue speed-accuracy trade-offs in RGB settings — UETrack extends this to multi-modal tracking and achieves comprehensive improvements in both accuracy and speed.
- **Multi-modal tracking**: SUTrack (Chen et al.) adopts unified token processing for multi-modal inputs but lacks sufficient speed; ViPT (Zhu et al.), SDSTrack (Hou et al.), and OneTracker (Lin et al.) add modality adaptation modules at the cost of high computational overhead — UETrack reconciles unified modeling with efficient inference.
- **Knowledge distillation**: Conventional methods (Hinton et al., Romero et al.) apply distillation uniformly across all samples — TAD performs sample-level selective distillation, better suited to the large variation in sample difficulty encountered in tracking.
- **Mixture of Experts**: MoETrack, SPMTrack, and related methods employ MoE in tracking but incur gating latency — the gate-free soft assignment design of TP-MoE is more appropriate for real-time visual tracking.

## Rating

| Dimension | Score (1–10) |
|------|:-----------:|
| Novelty | 7 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Value | 9 |
| **Overall** | **7.6** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[CVPR 2026\] Temporally Consistent Long-Term Memory for 3D Single Object Tracking](chronotrack_temporally_consistent_long_term_memory_for_3d_single_object_tracking.md)
- [\[CVPR 2026\] SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)

</div>

<!-- RELATED:END -->
