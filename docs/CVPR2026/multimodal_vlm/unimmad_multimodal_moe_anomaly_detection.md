---
title: >-
  [Paper Note] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression
description: >-
  [CVPR 2026][Multimodal VLM][anomaly detection] This paper proposes UniMMAD, the first unified framework for multi-modal (RGB/Depth/IR, etc.) and multi-class anomaly detection. It follows a General-to-Specific paradigm: a general multi-modal encoder compresses features, and a Cross Mixture-of-Experts (C-MoE) decompresses them into domain-specific features. The method achieves state-of-the-art results on 5 datasets spanning industrial, medical, and synthetic scenarios at 59 FPS inference speed.
tags:
  - CVPR 2026
  - Multimodal VLM
  - anomaly detection
  - multi-modal
  - mixture of experts
  - feature decompression
  - unified framework
date: 2026-05-08
content_hash: 5af248bc683b7b78
---

# UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression

**Conference**: CVPR 2026
**arXiv**: [2509.25934](https://arxiv.org/abs/2509.25934)
**Code**: [GitHub](https://github.com/yuanzhao-CVLAB/UniMMAD)
**Area**: Multi-modal VLM / Anomaly Detection
**Keywords**: anomaly detection, multi-modal, mixture of experts, feature decompression, unified framework

## TL;DR
This paper proposes UniMMAD, the first unified framework for multi-modal (RGB/Depth/IR, etc.) and multi-class anomaly detection. It follows a General-to-Specific paradigm: a general multi-modal encoder compresses features, and a Cross Mixture-of-Experts (C-MoE) decompresses them into domain-specific features. The method achieves state-of-the-art results on 5 datasets spanning industrial, medical, and synthetic scenarios at 59 FPS inference speed.

## Background & Motivation
**Background**: Existing anomaly detection methods treat modalities and categories as independent factors, training dedicated models for each combination, leading to fragmented solutions and high memory overhead.

**Limitations of Prior Work**: (a) Multi-class reconstruction-based methods use shared decoders, causing distortion of normality boundaries and inter-domain interference under large domain gaps; (b) Each modality–category pair requires a separate model, which is not scalable; (c) Existing methods struggle to handle cross-domain scenarios spanning industrial, medical, and synthetic settings simultaneously.

**Key Challenge**: A unified model must handle multi-modal inputs (RGB, 3D, infrared, etc.) and up to 66 categories, yet features across domains vary drastically, and naive parameter sharing leads to inter-domain interference.

**Goal**: Design a parameter-efficient unified framework that simultaneously handles multi-modal, multi-class, and multi-domain anomaly detection.

**Key Insight**: A General-to-Specific paradigm — first compress multi-modal features with a general encoder (suppressing anomalies), then decompress them into domain-specific features with sparse MoE (restoring normality).

**Core Idea**: General encoder compression combined with Cross MoE sparse-routing decompression, where different domains activate different experts, enabling domain isolation within a single unified model.

## Method

### Overall Architecture
Two stages: (1) **General multi-modal encoder**: an input embedding layer unifies channel dimensions; residual blocks with a Feature Compression Module (FCM) progressively compress features and suppress potential anomalies. (2) **Cross MoE decoder**: a conditional router selects top-K experts based on domain-specific statistics; a MoE-in-MoE structure achieves ~75% parameter reduction.

### Key Designs

1. **Feature Compression Module (FCM)**:

    - **Function**: Suppress potential anomalies and facilitate cross-modal interaction.
    - **Mechanism**: Inner multi-scale bottleneck (parallel $1\times1$, $3\times3$, $5\times5$ convolutions) combined with an outer bottleneck, forming a two-level compression structure.
    - **Design Motivation**: Anomalies at different scales require receptive fields of different sizes for effective detection.

2. **Cross Mixture-of-Experts (C-MoE)**:

    - **Function**: Decompress general features into domain-specific features.
    - **Mechanism**: A conditional router projects general features as key/value and prior features as query; global average pooling generates domain statistics; sparse top-K gating selects experts. Annealing load-balancing loss: $\mathcal{L}_{\text{MoE}} = (1 - e/E)^2 \cdot \text{CV}(G)$.
    - **Design Motivation**: Different domains activate different experts to avoid inter-domain interference; the annealing mechanism gradually relaxes the load-balancing constraint as training progresses.

3. **MoE-in-MoE Nested Structure**:

    - **Function**: Embed dense base experts inside each leader expert of the sparse MoE.
    - **Mechanism**: The convolutional kernels of each MoE-Leader are formed by a weighted combination of shared base experts, reducing parameters by ~75%.
    - **Design Motivation**: Reduce parameter count while preserving sparse activation characteristics.

4. **Inference Acceleration: Grouped Dynamic Convolution**:

    - **Function**: Pre-compute and cache weighted kernels; convert multi-expert parallel computation into grouped convolution.
    - **Effect**: 59 FPS, 45× faster than CFM and 150×+ faster than M3DM.

### Loss & Training
- Decompression consistency loss $\mathcal{L}_{\text{DeC}}$: negative cosine similarity with focal-loss modulation ($\gamma = 2$).
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{DeC}} + \mathcal{L}_{\text{MoE}}$.
- Weighted sampling: training probability inversely proportional to per-class sample count.
- 300 epochs, batch size 10; WideResNet50 used as prior feature generator.

## Key Experimental Results

### Main Results: Image-Level AUC (across 5 multi-scene datasets)

| Dataset | UniMMAD | Best Baseline | Baseline |
|---|---|---|---|
| MVTec-3D | 92.53 | 92.44 | CFM |
| Eyecandies | 85.57 | 81.85 | CFM |
| MulSen-AD | 85.46 | 78.87 | MulSen-TripleAD |
| BraTs (Medical) | 95.84 | 96.06 | INP-Former |
| UniMed (Medical) | 96.34 | 95.98 | MambaAD |

### Ablation Study: Component Contributions (Mean across datasets)

| Configuration | AUC_I | AUC_P | MF1_P |
|---|---|---|---|
| Baseline (Reverse TS) | 75.62 | 86.62 | 28.46 |
| + FCM | 77.37 | 86.65 | 28.93 |
| + General-to-Specific | 84.31 | 96.11 | 37.13 |
| + C-MoE (Full) | 91.15 | 96.68 | 42.91 |

The General-to-Specific paradigm contributes the largest gain (+8.9%), followed by C-MoE (+6.8%).

### Efficiency Comparison

| Metric | UniMMAD | CFM | M3DM |
|---|---|---|---|
| FPS | 59.09 | 13.18 | 0.39 |
| FLOPs (GFlops) | 110.91 | 431.14 | — |

### Key Findings
- The General-to-Specific paradigm is the single largest contributor (+8.9% AUC_I), validating the compress-then-decompress design.
- Removing cross-condition routing drops AUC_I from 91.1 to 85.1 (−6.7%), demonstrating the critical role of domain-conditioned routing.
- Unified training vs. task-specific training differs by only 0.1–0.3%, confirming effective task isolation.
- Supports continual learning: performance degradation on old tasks remains below 8% when new tasks are added.

## Highlights & Insights
- **The General-to-Specific paradigm is an elegant design**: compression suppresses anomalies, decompression restores normality, and MoE routing achieves domain isolation.
- **MoE-in-MoE nested structure**: achieves 75% parameter reduction without sacrificing performance, far more efficient than simply increasing the number of experts.
- **Practical inference acceleration**: pre-computation combined with grouped convolution brings MoE inference close to single-model speed (59 FPS).
- **Cross-domain unification**: a single model handles industrial, medical, and synthetic scenarios simultaneously, demonstrating the generality of the paradigm.

## Limitations & Future Work
- Pixel-level detection (MF1_P) does not surpass baselines on certain datasets (e.g., MVTec-3D: 44.16 vs. 44.72).
- The method requires a pre-trained WideResNet50 as a prior feature generator, introducing an additional dependency.
- Image-level AUC on the BraTs medical dataset is slightly below INP-Former (95.84 vs. 96.06).

## Related Work & Insights
- **vs. M3DM**: M3DM relies on memory-bank retrieval and is extremely slow at inference (0.39 FPS); UniMMAD is 150× faster.
- **vs. CFM**: CFM uses fixed fusion, resulting in lower AUC and slower inference; UniMMAD's C-MoE dynamic fusion is more effective.
- **vs. INP-Former**: UniMMAD remains competitive even in purely RGB single-modal, many-class settings (MVTec-AD + VisA, 66 classes).

## Rating
- **Novelty**: ⭐⭐⭐⭐ General-to-Specific + C-MoE constitutes a systematic innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five multi-modal datasets + two single-modal datasets, with comprehensive ablation and efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear and well-organized.
- **Value**: ⭐⭐⭐⭐ A unified anomaly detection framework with practical industrial application value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

<!-- RELATED:END -->
