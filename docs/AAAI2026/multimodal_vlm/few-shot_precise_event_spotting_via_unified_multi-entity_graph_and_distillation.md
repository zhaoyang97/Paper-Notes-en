---
title: >-
  [Paper Note] Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation
description: >-
  [AAAI 2026][Multimodal VLM][Precise Event Spotting] This paper proposes UMEG-Net for few-shot Precise Event Spotting (PES). The method constructs a unified multi-entity graph integrating human skeletal keypoints, sports object keypoints, and environmental landmarks, combined with efficient spatiotemporal graph convolution and a parameter-free multi-scale temporal shift module. A multimodal knowledge distillation scheme transfers graph features to an RGB student network. The approach significantly outperforms existing methods across five sports datasets under extremely limited annotation budgets.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Precise Event Spotting
  - Few-Shot Learning
  - Unified Multi-Entity Graph
  - Knowledge Distillation
  - Sports Video Analysis
date: 2026-05-08
content_hash: faf16d38ee45d27a
---

# Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation

**Conference**: AAAI 2026
**arXiv**: [2511.14186](https://arxiv.org/abs/2511.14186)
**Code**: [github.com/LZYAndy/UMEG-Net](https://github.com/LZYAndy/UMEG-Net)
**Area**: Multimodal VLM
**Keywords**: Precise Event Spotting, Few-Shot Learning, Unified Multi-Entity Graph, Knowledge Distillation, Sports Video Analysis

## TL;DR

This paper proposes UMEG-Net for few-shot Precise Event Spotting (PES). The method constructs a unified multi-entity graph integrating human skeletal keypoints, sports object keypoints, and environmental landmarks, combined with efficient spatiotemporal graph convolution and a parameter-free multi-scale temporal shift module. A multimodal knowledge distillation scheme transfers graph features to an RGB student network. The approach significantly outperforms existing methods across five sports datasets under extremely limited annotation budgets.

## Background & Motivation

### Precise Event Spotting (PES)

Precise Event Spotting aims to identify fine-grained events and their exact timestamps from untrimmed long videos, with extremely tight tolerance windows (1–2 frames). Typical scenarios include racket-impact moments in racket sports and takeoff/landing in gymnastics.

Three major challenges in PES:

**Rapid successive events**: Multiple events occur at very short intervals in sports videos.

**Motion blur**: High-speed motion degrades visual features.

**Subtle visual differences**: Low visual discriminability across different event types.

### Limitations of Prior Work

**End-to-end RGB methods** (E2E-Spot, T-DEED, F3ED): Rely on large-scale frame-level annotated datasets; performance degrades sharply under few-shot conditions.

**Skeleton-based methods** (STGCN++, BlockGCN, etc.): Use only human pose, ignoring critical information such as the ball and court.

**Conventional few-shot methods**: Designed for coarse-grained action recognition; cannot satisfy the frame-level precision required by PES.

### Root Cause

Event occurrence in sports involves **interactions among multiple entities** — a player hitting a ball requires joint modeling of the human body, ball, and court. Constructing a **unified multi-entity graph** to represent these interactions is key to improving few-shot PES performance. Additionally, the unreliability of keypoint detection necessitates **multimodal distillation** to enhance robustness.

## Method

### Overall Architecture

1. Keypoint extraction: HRNet (human pose) + YOLOv8 (ball/player detection) + dedicated methods (court corner points)
2. Unified multi-entity graph construction: All keypoints organized into a unified graph structure
3. Stacked UMEG Blocks: Spatial GCN + multi-scale temporal shifting
4. Event spotting and classification: Linear layer outputs frame-level event probabilities
5. Multimodal distillation: UMEG-Net teacher → RGB student network

### Key Designs

#### 1. **Unified Multi-Entity Graph Construction**

Node set $\mathcal{V}_t = \{V_p^t, V_b^t, V_c^t\}$ (player joints + ball + court corner points); edge set comprises four types:
$$\mathcal{E}_t = \mathcal{E}_t^{intra} \cup \mathcal{E}_t^{p-b} \cup \mathcal{E}_t^{p-c} \cup \mathcal{E}_t^{c-c}$$

- Intra-skeleton connections (standard human joint topology)
- Person–ball connections (racket sports: wrist → ball; soccer: ankle/shoulder → ball)
- Person–court connections (foot joints → court corner points)
- Court corner interconnections (forming a rectangle)

**Design Motivation**: Different sports employ different person–ball connection schemes, making full use of sport-specific domain knowledge. Information ignored by conventional skeleton graphs is critical for event determination.

#### 2. **UMEG Block: Spatial GCN + Multi-Scale Temporal Shifting**

**Spatial GCN**: Graph convolution is performed over the entire multi-entity graph (rather than independently per person), jointly modeling person–person and person–entity interactions:
$$\mathcal{H}^{(\ell+1)} = \text{ReLU}(A^{(\ell)} \mathcal{H}^{(\ell)} W^{(\ell)})$$

**Multi-scale temporal shift module (parameter-free)**: Replaces temporal convolution with parameter-free temporal shifts, substantially reducing trainable parameters:

1. Features are split along the channel dimension into three parts (static, forward, backward, ratio $\alpha = 1/8$)
2. Bidirectional shifts are applied for $\Delta \in \{1, 2, 4\}$
3. Each shifted stream is updated via the spatial GCN and then fused across scales

**Design Motivation**: $\Delta \in \{1, 2, 4\}$ simultaneously captures short-, medium-, and long-range temporal dependencies. Temporal convolution is prone to overfitting under few-shot conditions; temporal shifting is a zero-parameter-cost alternative.

#### 3. **Multimodal Knowledge Distillation**

- Teacher (frozen): Trained UMEG-Net
- Student: VideoMAEv2 feature extractor + BiGRU
- Distillation loss (computed on unlabeled data): $\mathcal{L}_{feat} = \frac{1}{T}\sum_t \|\mathbf{F}_{tch}^{(t)} - \mathbf{F}_{stu}^{(t)}\|_2^2$
- At inference, only the student is used; no keypoint detection is required

**Design Motivation**: Distillation over large amounts of unlabeled video enables the RGB student to acquire visual representations complementary to the graph model, while eliminating dependence on pose estimation.

### Loss & Training

- Joint training of event-type classification and event localization
- Foreground class loss weight increased by 5× (event frames account for <3%; severe class imbalance)
- AdamW optimizer with cosine annealing
- UMEG-Net: 50 epochs, lr = 0.001; distillation: 50 epochs, lr = 0.0001

## Key Experimental Results

### Main Results

**F1/Edit score comparison under the 100-clip few-shot setting**:

| Method | F3Set F1 | ShuttleSet F1 | FineGym F1 | FigureSkating F1 | SoccerNet F1 |
|--------|----------|---------------|------------|------------------|--------------|
| E2E-Spot_800MF | 13.3 | 54.6 | 53.1 | 42.7 | 43.1 |
| F3ED | 15.3 | 55.1 | 52.1 | 34.4 | 34.5 |
| BlockGCN | 18.3 | 59.4 | 49.1 | 48.2 | 43.3 |
| **UMEG-Net** | **31.7** | **64.0** | **54.4** | **49.6** | **44.8** |
| **UMEG-Net_distill** | **40.7** | **69.0** | **61.2** | **56.2** | **50.8** |

UMEG-Net achieves consistent improvements over all baselines on all five datasets. UMEG-Net_distill yields average gains of +5.8% F1 and +6.7% Edit.

### Ablation Study

**Effect of graph entity composition**:

| Graph Configuration | F3Set F1 | F3Set Edit | ShuttleSet F1 |
|---------------------|----------|------------|---------------|
| pose×N | 23.9 | 47.4 | 61.5 |
| pose×N + court | 26.1 | 46.7 | 61.5 |
| pose×N + ball | 30.2 | 48.1 | 62.5 |
| **pose×N + ball + court** | **31.7** | **49.2** | **64.0** |

**Temporal module configuration**:

| $\Delta$ Configuration | FineGym F1 | FigureSkating F1 |
|------------------------|------------|------------------|
| {1} | 50.3 | 36.8 |
| {1, 2} | 49.8 | 45.3 |
| **{1, 2, 4}** | **54.4** | **49.6** |

**Full-supervision comparison**: UMEG-Net remains competitive in the fully supervised setting, surpassing E2E-Spot on 3 out of 5 datasets.

### Key Findings

1. Ball information contributes most (F1 +6.3), followed by court information (+2.2).
2. UMEG-Net has only 2.2M parameters — the fewest among all methods while achieving the best performance.
3. Distillation substantially outperforms self-supervised contrastive pretraining (F3Set F1: 40.7 vs. 29.1).
4. The k-clip setting is more practically reasonable than conventional k-shot.

## Highlights & Insights

1. **Precise problem formulation**: Few-shot PES is a real and important problem, as frame-level annotation is extremely costly.
2. **k-clip is more reasonable than k-shot**: Sports events are very short, densely occurring, and multi-class; k-clip better reflects practical annotation scenarios.
3. **Parameter-free temporal module**: Zero-parameter-cost replacement for multi-scale temporal convolution, reducing overfitting in few-shot settings.
4. **Distillation leverages unlabeled data**: An elegant use of in-domain data.

## Limitations & Future Work

1. Dependence on keypoint detection quality (though the distilled version does not require it).
2. Person–ball connection designs require manual specification, lacking generalizability.
3. Relatively smaller gains in multi-person scenarios (SoccerNet).
4. UMEG-Net_distill employs VideoMAEv2 (67.8M parameters), far larger than the teacher (2.2M).

## Related Work & Insights

- **E2E-Spot / T-DEED / F3ED**: Representative RGB end-to-end PES methods.
- **BlockGCN / STGCN++**: Skeleton-based action recognition methods.
- **TSM**: Source of inspiration for the temporal shift module.
- **Hong et al.**: Pioneer work on pose-to-RGB distillation in figure skating.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified multi-entity graph and parameter-free temporal shifting represent solid contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five datasets, comprehensive ablations, and multiple k-clip settings.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ — The few-shot setting directly addresses practical annotation cost challenges.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](../../CVPR2026/multimodal_vlm/noiseaware_fewshot_learning_through_bidirectional.md)
- [\[AAAI 2026\] Towards Long-window Anchoring in Vision-Language Model Distillation](towards_long-window_anchoring_in_vision-language_model_distillation.md)
- [\[AAAI 2026\] vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs](vmfcoop_towards_equilibrium_on_a_unified_hyperspherical_manifold_for_prompting_b.md)
- [\[AAAI 2026\] RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)

</div>

<!-- RELATED:END -->
