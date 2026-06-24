---
title: >-
  [Paper Note] KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception
description: >-
  [CVPR 2025][Interpretability][Video Quality Assessment] Inspired by the human visual system, KVQ explicitly decouples global video quality into two factors: visual saliency and local texture. It extracts cross-region saliency via Fusion-Window Attention and enhances texture perception in independent regions using a Local Perception Constraint, significantly outperforming SOTA methods on five VQA benchmarks.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Video Quality Assessment"
  - "Visual Saliency"
  - "Local Texture Perception"
  - "Fusion-Window Attention"
  - "Human Visual System"
date: 2026-05-08
content_hash: aa4577d323f2bc75
---

# KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception

**Conference**: CVPR 2025  
**arXiv**: [2503.10259](https://arxiv.org/abs/2503.10259)  
**Code**: [https://github.com/qyp2000/KVQ](https://github.com/qyp2000/KVQ)  
**Area**: Interpretability  
**Keywords**: Video Quality Assessment, Visual Saliency, Local Texture Perception, Fusion-Window Attention, Human Visual System

## TL;DR

Inspired by the human visual system, KVQ explicitly decouples global video quality into two factors: visual saliency and local texture. It extracts cross-region saliency via Fusion-Window Attention and enhances texture perception in independent regions using a Local Perception Constraint, significantly outperforming SOTA methods on five VQA benchmarks.

## Background & Motivation

**Background**: Video Quality Assessment (VQA) aims to predict the perceived quality of videos, which is a key techonology for optimizing user experience on short video platforms. Current no-reference VQA methods mostly predict global quality scores directly, while a few methods (such as Fast-VQA) attempt to perceive regional quality by sampling local patches.

**Limitations of Prior Work**: (1) Quality differences across various spatiotemporal regions in videos are significant (e.g., motion blur, compression distortion, and differing texture complexity), yet existing methods lack reliable modeling of region-level quality; (2) due to the extremely high cost of annotating regional quality (approximately $\mathcal{O}(N^3)$ times that of global annotation), region-level annotated data is lacking as a constraint; (3) the attention mechanisms of existing methods (such as window attention) are limited to local neighborhoods, preventing effective extraction of global visual saliency.

**Key Challenge**: Global quality is jointly determined by local texture and visual saliency, but these two represent concepts at different levels—saliency involves high-level semantics and inter-region correlations, whereas texture only concerns low-level features within a region. Existing methods conflate the two, leading to: (a) local quality predictions being affected by saliency, and (b) saliency extraction being restricted to local windows.

**Goal**: (1) How to reliably predict region-level quality without local annotations; (2) how to effectively extract global visual saliency.

**Key Insight**: Starting from the Human Visual System (HVS), two assumptions are proposed—Assumption 1: Global quality = Saliency-weighted local texture; Assumption 2: Local texture is solely determined by features within the region, independent of other regions. Saliency extraction and local perception constraints are designed accordingly.

**Core Idea**: Explicitly decompose quality into saliency $\times$ texture, using cross-window attention to extract saliency and a consistency constraint to ensure texture is independent of neighboring regions.

## Method

### Overall Architecture

Using Video Swin-T as the backbone to extract spatiotemporal features, a dual-branch head at the top predicts the saliency map $\mathcal{S} \in \mathbb{R}^{T \times H \times W}$ and the local texture map $\mathcal{Q} \in \mathbb{R}^{T \times H \times W}$, respectively. Global quality is calculated as $q = \frac{1}{THW} \sum_{i,j,k} \mathcal{S}_{i,j,k} \cdot \mathcal{Q}_{i,j,k}$. Fusion-Window Attention is embedded into the backbone to replace standard window attention, and a Local Perception Constraint is incorporated during training to constrain the texture branch.

### Key Designs

1. **Fusion-Window Attention (FWA)**:

    - Function: Achieves cross-region global attention allocation to effectively extract visual saliency
    - Mechanism: Three steps—(a) **Correlated-Window Selection (CWS)**: Calculate the global patch correlation map $\mathbf{I}_p = Softmax(Flatten(Q) \cdot Flatten(K)^T)$, average-pool it to the window level $\mathbf{I}_w$, and select the indices of the top-k most correlated windows $\mathbf{Idx}$; (b) **Intra-Window Attention (IWA)**: Standard intra-window self-attention is performed to preserve neighborhood information; (c) **Cross-Window Attention (CWA)**: The Query of each window performs cross-attention with the Key/Value of its top-k correlated windows. Ultimately, $FWA = IWA + CWA$
    - Design Motivation: Standard window attention (e.g., Swin Transformer) only performs attention within local windows, failing to model the global attention allocation behavior of human vision. FWA establishes global long-range connections through adaptive selection of representative windows to better capture saliency.

2. **Multi-scale Ensemble Saliency Map**:

    - Function: Fuses multi-scale correlation maps to generate the final saliency map
    - Mechanism: The patch correlation map $\mathbf{I}_p$ in FWA inherently reflects attention allocation. By transposing and summing it, the attention of each patch $\mathbf{I}_p'$ is obtained, which is reshaped and pooled to a unified resolution to derive the saliency estimation $\mathbf{I}_p^{(l)}$ of each layer. Weighted fusion is performed with the output of the saliency branch $\tilde{S}$: $\mathcal{S} = Softmax(w^0 \tilde{S} + \sum_{l} w^l \mathbf{I}_p^{(l)})$
    - Design Motivation: Referring to the process of visual information flowing through cortical hierarchies in HVS, multi-scale fusion can capture saliency information ranging from fine-grained to coarse-grained.

3. **Local Perception Constraint (LPC)**:

    - Function: Ensures that the texture map only reflects intra-region features, free from the influence of other regions
    - Mechanism: Input the complete video into the model to obtain the texture map $\mathcal{Q}$. Simultaneously, partition the video into independent patches, input them separately to the same model, and reassemble them to obtain $\hat{\mathcal{Q}}$. Consistently constrain both: $\mathcal{L}_{lpc} = 1 - \frac{\sum \mathcal{Q}_{i,j,k} \cdot \hat{\mathcal{Q}}_{i,j,k}}{||\mathcal{Q}|| \cdot ||\hat{\mathcal{Q}}||}$ (Cosine Similarity Loss).
    - Design Motivation: If texture prediction depends on neighboring contexts, predictions would differ when the video is sliced into independent patches. Restricting this consistency forces the model's texture branch to focus strictly on low-level features within the region (e.g., distortion, sharpness, texture patterns) without mixing in semantic context information.

### Loss & Training

The overall loss is $\mathcal{L} = \mathcal{L}_{plcc} + \lambda_r \mathcal{L}_{rank} + \lambda_p \mathcal{L}_{lpc}$. Here, $\mathcal{L}_{plcc}$ is the PLCC loss (for quality score regression), $\mathcal{L}_{rank}$ is the ranking loss (for learning relative quality relations), and $\mathcal{L}_{lpc}$ is the local perception constraint. Video Swin-T Tiny (pretrained on Kinetics-400) is used as the backbone, with a window size of [8,7,7]. 32 frames are sampled from each video and resized to 448×448.

## Key Experimental Results

### Main Results

| Method | LSVQtest SRCC↑ | LSVQ1080p SRCC↑ | KoNViD-1k SRCC↑ | LIVE-VQC SRCC↑ |
|------|----------------|-----------------|-----------------|----------------|
| Fast-VQA | 0.876 | 0.779 | 0.859 | 0.823 |
| Faster-VQA | 0.873 | 0.772 | 0.863 | 0.813 |
| **KVQ** | **0.896** | **0.814** | **0.890** | **0.820** |
| Gain vs Fast-VQA | +2.3% | +4.5% | +3.6% | -0.4% |

KVQ's advantages are more pronounced in transfer learning scenarios: achieving SRCC of 0.909 on KoNViD-1k (vs. Fast-VQA's 0.891) and SRCC of 0.903 on YouTube-UGC (vs. 0.855, a +5.6% gain).

### Ablation Study

| Component | Role | Impact |
|------|------|------|
| FWA (Fusion-Window Attention) | Global saliency extraction | Cross-window attention significantly improves long-range dependency modeling |
| LPC (Local Perception Constraint) | Local texture constraint | Ensures that the texture map is not contaminated by saliency |
| Multi-scale Ensemble | Multi-scale saliency fusion | Hierarchical information complementarity enhances saliency estimation |
| CWS (Correlated-Window Selection) | Adaptive window selection | Avoids the $O(N^2)$ computational complexity of global attention |

### Key Findings

- KVQ achieves the largest improvement on LSVQ1080p (high-resolution videos) with a +4.5% SRCC gain, showing that the cross-region attention of FWA is particularly effective in high-resolution scenarios.
- The transfer learning performance shows massive gains (YouTube-UGC +5.6%), indicating that saliency-texture decoupling enhances model generalization.
- The newly established LPVQ dataset (consisting of 50 images, 14 experts, and 34,300 annotations) verifies that KVQ indeed perceives local quality differences.
- The computational cost (353 GFlops) is two orders of magnitude lower than Li et al. (112,537 GFlops) and is comparable to Fast-VQA (279 GFlops).

## Highlights & Insights

- **Explicit decoupling inspired by HVS is elegant**: The formulation of global quality = saliency $\times$ texture is clear and highly interpretable. Saliency tells "where to look", while texture indicates "how bad/good the quality of what is seen is". This decoupling enables the model to independently output both its saliency map and quality map, enhancing overall interpretability.
- **LPC serves as an unsupervised constraint without the need for manual labels**: It cleverly utilizes the prior that "local features should not be affected by contextual information". This allows the texture branch to be trained without any regional quality annotations, overcoming the core challenge of high labeling costs.
- **The design motif of FWA can be easily transferred to other video understanding tasks**: Relevance-based cross-window attention is not restricted to quality assessment. Any Video Transformer that requires long-range dependencies but is limited by localized window attention can benefit from this approach.

## Limitations & Future Work

- The LPVQ dataset only contains 50 images as static "video" annotations, which is relatively small in scale; extending it to genuine spatiotemporal region annotations for videos would yield more persuasive evaluations.
- The top-k window selection in FWA introduces extra global correlation computation, whose computational efficiency on extremely high-resolution videos warrants further attention.
- Assumption 1 (global quality = weighted sum) is a linear formulation, whereas human perception of quality might actually be non-linear (e.g., dominated by the "worst-region-dominant" effect).
- It was validated only under a specific window size of [8,7,7]; adaptive window design may be required for videos with varying resolutions and frame rates.

## Related Work & Insights

- **vs. Fast-VQA**: Fast-VQA uniformly samples local patches but lacks local quality constraints, blending local texture and saliency. In contrast, KVQ's explicit decoupling results in an overall improvement of 2-5%.
- **vs. SGDNet/TranSLA**: Although these methods introduce saliency prediction as a subtask, their training processes are complex and rely on pseudo-labels from SOTA saliency models. KVQ computes saliency directly from the attention mechanism without requiring extra labels.
- **vs. PVQ**: PVQ attempts to leverage crowdsourcing to annotate local patch quality, incurring high costs with texture and saliency still blended. KVQ achieves a more reliable decoupling in an unsupervised manner via LPC.

## Rating

- Novelty: ⭐⭐⭐⭐ The HVS-inspired saliency-texture decoupling approach is exceptionally clear, and the FWA and LPC designs are clever, though individual components are not entirely brand new when examined in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 5 benchmarks across 3 evaluation configurations (intra, cross, and transfer), with a newly established LPVQ dataset.
- Writing Quality: ⭐⭐⭐⭐ The hypothesis-driven methodology is logically clear, with standardized mathematical formulations.
- Value: ⭐⭐⭐⭐ Provides novel insights into local perception modeling within the VQA field, and the LPVQ dataset carries independent value as well.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](../../AAAI2026/interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[ICML 2026\] IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring](../../ICML2026/interpretability/iqa-spider_unifying_multi-granularity_image_quality_assessment_with_reasoning_gr.md)
- [\[CVPR 2025\] Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)
- [\[ICML 2025\] Towards Flexible Perception with Visual Memory](../../ICML2025/interpretability/towards_flexible_perception_with_visual_memory.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)

</div>

<!-- RELATED:END -->
