---
title: >-
  [Paper Note] Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness
description: >-
  [ACL 2026][Image Restoration][Knowledge Boundary Awareness] GeoDe trains linear probes in LLM latent space to construct truth hyperplanes, using sample-to-hyperplane geometric distance as confidence signals to filter high-quality abstention fine-tuning data, effectively eliminating "gray zone" noise near decision boundaries and significantly improving model truthfulness and reliability.
tags:
  - ACL 2026
  - Image Restoration
  - Knowledge Boundary Awareness
  - Abstention Fine-Tuning
  - Latent Space Probe
  - Geometric Denoising
  - Hallucination Mitigation
content_hash: 10b4b3a5eccdc5b6
---

# Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness

**Conference**: ACL 2026
**arXiv**: [2604.14324](https://arxiv.org/abs/2604.14324)
**Code**: [GitHub](https://github.com/Notbesidemoon/GeoDe)
**Area**: Image Restoration
**Keywords**: Knowledge Boundary Awareness, Abstention Fine-Tuning, Latent Space Probe, Geometric Denoising, Hallucination Mitigation

## TL;DR
GeoDe trains linear probes in LLM latent space to construct truth hyperplanes, using sample-to-hyperplane geometric distance as confidence signals to filter high-quality abstention fine-tuning data, effectively eliminating "gray zone" noise near decision boundaries and significantly improving model truthfulness and reliability.

## Background & Motivation

**State of the Field**: LLMs frequently produce hallucinations. Abstention fine-tuning trains models to answer correctly on known questions and respond "I don't know" on unknown ones.

**Root Cause**: Near the decision boundary in latent space exists a "gray zone" where known and unknown sample representations highly overlap, with ambiguous internal beliefs. Forcing label assignment in this region causes the model to learn contradictory decision rules.

**Core Idea**: Use geometric distance instead of response-accuracy-based hard partitioning. Distance thresholds filter out gray zone samples, training only on high-confidence subsets.

## Method

### Key Designs

1. **Linear Probe and Truth Hyperplane**: Extracts LLM hidden states $\mathbf{x} = f_{LLM}(q)$ and trains logistic regression probe $f_{probe}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$. The probe weights define the truth hyperplane.

2. **Geometric Distance-Based Data Filtering**: Computes distance $d(\mathbf{x}) = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|_2}$. Retains only samples with $|d(\mathbf{x})| > \theta$ (default X=25% quantile).

3. **Two Hidden State Extraction Methods (TBG and SLT)**: TBG directly inputs questions and extracts the last token's hidden state; SLT concatenates question and answer then extracts the second-to-last token's hidden state for richer context.

## Key Experimental Results

### Main Results

| Method | TriviaQA F1_rel | NQ F1_rel | SciQ F1_rel | SimpleQA F1_rel |
|--------|----------------|-----------|-------------|-----------------|
| R-Tuning | 74.4 | 58.7 | 63.5 | 25.4 |
| **GeoDe TBG** | **77.1** | **64.0** | **68.3** | **30.7** |
| **GeoDe SLT** | **77.0** | **64.9** | **68.9** | **34.9** |

### Key Findings
- GeoDe outperforms baselines on all four datasets, especially in OOD scenarios
- Simultaneously improves F1_ans (helpfulness) and F1_abs (truthfulness), breaking the trade-off

## Highlights & Insights
- The "gray zone" concept is intuitive and powerful, providing a geometric explanation for abstention fine-tuning challenges
- Extremely concise method — just a logistic regression probe plus a distance threshold yields significant improvement

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](../../ICCV2025/image_restoration/generic_event_boundary_detection_via_denoising_diffusion.md)
- [\[ICCV 2025\] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention](../../ICCV2025/image_restoration/consistent_time-of-flight_depth_denoising_via_graph-informed_geometric_attention.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](../../NeurIPS2025/image_restoration/latent_harmony_synergistic_unified_uhd_image_restoration_with_pre-trained_diffus.md)
- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](../../CVPR2026/image_restoration/shadow_removal_cascaded_refinement.md)

<!-- RELATED:END -->
