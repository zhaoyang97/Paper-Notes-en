---
title: >-
  [Paper Note] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection
description: >-
  [CVPR 2026][Multimodal VLM][Video Anomaly Detection] This paper proposes LAVIDA, an end-to-end zero-shot video anomaly detection framework that transforms semantic segmentation datasets into pseudo-anomaly training data via an Anomaly Exposure Sampler. Combined with MLLM-based deep anomaly semantic feature extraction and reverse-attention token compression for spatiotemporal sparsity, LAVIDA achieves frame-level and pixel-level SOTA without any real VAD data.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Video Anomaly Detection
  - Zero-Shot
  - MLLM
  - Pseudo Anomaly
  - Token Compression
date: 2026-05-08
content_hash: ba626da2a9eb27f1
---

# No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection

**Conference**: CVPR 2026
**arXiv**: [2602.19248](https://arxiv.org/abs/2602.19248)
**Code**: [https://github.com/VitaminCreed/LAVIDA](https://github.com/VitaminCreed/LAVIDA)
**Area**: Multimodal VLM
**Keywords**: Video Anomaly Detection, Zero-Shot, MLLM, Pseudo Anomaly, Token Compression

## TL;DR
This paper proposes LAVIDA, an end-to-end zero-shot video anomaly detection framework that transforms semantic segmentation datasets into pseudo-anomaly training data via an Anomaly Exposure Sampler. Combined with MLLM-based deep anomaly semantic feature extraction and reverse-attention token compression for spatiotemporal sparsity, LAVIDA achieves frame-level and pixel-level SOTA without any real VAD data.

## Background & Motivation

**Background**: Video Anomaly Detection (VAD) faces two major challenges: data scarcity and scene variability. Traditional approaches fall into unsupervised (learning normal patterns) and weakly supervised (video-level annotations) categories, both constrained by training scenes and anomaly types. Recent open-set/open-vocabulary VAD attempts to handle unseen anomaly types, while training-free MLLM-based methods directly score with LLMs, but rely on per-frame/per-clip text outputs, incurring high inference costs and lacking localization capability.

**Limitations of Prior Work**: (1) Existing VAD datasets are limited in scene diversity and anomaly types, resulting in poor generalization; (2) Anomaly semantics are context-dependent (the same behavior may be normal or anomalous in different scenes), and current methods lack deep semantic understanding; (3) Anomalies are extremely sparse in space and time, and large numbers of background tokens increase computational cost and interfere with detection.

**Key Challenge**: Real anomaly data is scarce, yet models require exposure to sufficiently diverse anomalies to generalize.

**Goal**: Train entirely without VAD data, achieving zero-shot frame-level and pixel-level detection across scenes and anomaly types.

**Key Insight**: Semantic segmentation datasets contain rich scene semantics and pixel-level annotations; segmentation targets can be repurposed as "anomalies" for training.

**Core Idea**: Transform segmentation datasets into pseudo-anomaly training sets via an Anomaly Exposure Sampler, leverage MLLM for anomaly semantic understanding, and apply reverse-attention compression to suppress background tokens.

## Method

### Overall Architecture
LAVIDA consists of five components: (1) Anomaly Exposure Sampler — reconstructs segmentation datasets as pseudo-anomaly data; (2) Feature Encoding Module — extracts visual and textual features; (3) Semantic Feature Extraction — encodes anomaly semantics via MLLM; (4) Multi-scale Semantic Projector — fuses semantic features with learnable queries; (5) Multi-level Mask Decoder — outputs frame-level and pixel-level anomaly scores.

### Key Designs

1. **Anomaly Exposure Sampler**:

    - Function: Transforms semantic segmentation datasets into VAD training data.
    - Mechanism: Two-step transformation — (1) randomly sample $K_E - 1$ irrelevant categories from other samples to form an anomaly category set $c_i$; (2) with probability $p$, label a sample as anomalous (containing the true category plus irrelevant categories, frame label positive) or normal (containing only irrelevant categories, frame label negative).
    - Trains the model to distinguish true anomaly categories from irrelevant ones, simulating real-world scenarios where most categories are absent.
    - Design Motivation: Exploits the rich semantics and pixel-level annotations of segmentation datasets, completely bypassing the need for real VAD data.

2. **Reverse-Attention Token Compression**:

    - Function: Compresses background tokens and retains anomaly-relevant features, reducing MLLM computational cost.
    - Mechanism: Identifies high-density background token set $Z^b$ via KNN density estimation, then applies reverse attention to each background token — $Z_i' = \text{Softmax}(-\frac{Z_i^b Z_{\mathcal{N}_i}^T}{\sqrt{D_z}}) \cdot Z_{\mathcal{N}_i}$
    - The negated attention weights cause features least similar to the background to receive the highest weights.
    - Compresses $L_z$ tokens into $L_r$ tokens, preserving anomaly candidate information.
    - Design Motivation: While anomalies are difficult to identify directly (sparse distribution), background tokens — being numerous and mutually similar — are easy to identify and can be aggregated inversely.

3. **MLLM Anomaly Semantic Extraction**:

    - Function: Leverages MLLM's open-world understanding to extract deep anomaly semantics.
    - Mechanism: A special token `<SEG>` is added to the MLLM vocabulary; a prompt "Find the anomaly in this video. Anomaly types may contain {c_i}" is constructed; the last-layer embedding $f_{sem}$ of `<SEG>` is extracted as the anomaly semantic feature.
    - Design Motivation: MLLMs possess context-aware semantic understanding, enabling dynamic adjustment of detection targets based on scene and anomaly type descriptions.

### Loss & Training
- Trained exclusively on the Anomaly Exposure dataset; no VAD data is used.
- Simultaneous frame-level and pixel-level supervision.
- $K_E$ is set to a random value to adapt the MLLM to an arbitrary number of anomaly types.

## Key Experimental Results

### Frame-Level Zero-Shot Performance

| Dataset | Metric | Best Unsupervised | Best Weakly Supervised | LAVIDA (Zero-Shot) |
|--------|------|-----------|-----------|---------------|
| UBnormal | AUC | 72.8 (MULDE) | — | **76.45** |
| ShanghaiTech | AUC | 81.3 (MULDE) | — | **85.28** |
| UCF-Crime | AUC | 78.5 (MULDE) | 90.33 (PI-VAD) | 82.18 |
| XD-Violence | AP | — | 88.96 (Holmes) | **90.62** |

### Pixel-Level Zero-Shot Performance

| Dataset | Metric | LAVIDA |
|--------|------|--------|
| UCSD Ped2 | pixel-AUC | **87.68** |

### Key Findings
- Zero-shot performance surpasses all unsupervised methods, requiring no scene-specific training.
- On XD-Violence, LAVIDA even outperforms weakly supervised methods (90.62 vs. 88.96 AP).
- Compared to training-free MLLM methods (LAVAD: 80.82), LAVIDA achieves a 1.36% improvement on UCF-Crime.

### Ablation Study
- Removing the Anomaly Exposure Sampler causes significant performance degradation.
- Removing token compression increases computational cost and reduces accuracy due to background noise interference.
- Removing MLLM semantic extraction substantially diminishes cross-scene generalization.

## Highlights & Insights
- Requires no real VAD data for training, constituting a genuinely zero-shot framework.
- The Anomaly Exposure Sampler is elegantly designed — repurposing "segmentation targets" as "anomalies" to reuse rich segmentation datasets.
- Reverse-attention token compression simultaneously reduces computational cost and background interference.
- Supports both frame-level and pixel-level detection, offering high practical deployment value.
- Zero-shot performance exceeds supervised methods across multiple benchmarks.

## Limitations & Future Work
- The distributional gap between pseudo-anomalies and real anomalies may affect detection accuracy in specific scenes.
- MLLM inference still carries non-trivial computational overhead, limiting real-time applicability.
- The current approach uses only segmentation targets as pseudo-anomalies; synthetic anomalies (e.g., video perturbations) could be further incorporated.
- The reverse-attention compression idea could be extended to other video understanding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[AAAI 2026\] HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](../../AAAI2026/multimodal_vlm/headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)
- [\[ICLR 2026\] Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](../../ICLR2026/multimodal_vlm/zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)
- [\[ICLR 2026\] Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection](../../ICLR2026/multimodal_vlm/steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)

</div>

<!-- RELATED:END -->
