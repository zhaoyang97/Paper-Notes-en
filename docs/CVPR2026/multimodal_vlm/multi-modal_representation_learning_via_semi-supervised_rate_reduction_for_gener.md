---
title: >-
  [Paper Note] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery
description: >-
  [CVPR2026][Multimodal VLM][Generalized Category Discovery] This paper proposes SSR²-GCD, a framework that learns structured representations with uniformly compressed intra-modal distributions via a Semi-Supervised Rate R…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Generalized Category Discovery"
  - "Multi-Modal Representation Learning"
  - "Semi-Supervised Rate Reduction"
  - "Intra-Modal Alignment"
  - "CLIP"
date: 2026-05-08
content_hash: f81ceb990db05a2c
---

# Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery

**Conference**: CVPR2026
**arXiv**: [2602.19910](https://arxiv.org/abs/2602.19910)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Generalized Category Discovery, Multi-Modal Representation Learning, Semi-Supervised Rate Reduction, Intra-Modal Alignment, CLIP

## TL;DR

This paper proposes SSR²-GCD, a framework that learns structured representations with uniformly compressed intra-modal distributions via a Semi-Supervised Rate Reduction (SSR²) loss, and introduces a Retrieval-based Text Aggregation (RTA) strategy to enhance cross-modal knowledge transfer. The method surpasses existing multi-modal GCD approaches on 8 benchmarks.

## Background & Motivation

1. **Practical demand for Generalized Category Discovery (GCD)**: Real-world data contains both known and novel categories. GCD leverages knowledge from known categories to discover novel ones, serving as a natural extension of open-set recognition.
2. **Rise of multi-modal methods**: Recent methods such as CLIP-GCD, TextGCD, and GET incorporate textual information into visual GCD tasks, improving performance through cross-modal alignment.
3. **Limitations of inter-modal alignment**: Existing multi-modal GCD methods focus primarily on inter-modal alignment while neglecting structural issues in the intra-modal representation distribution.
4. **Imbalanced compression problem**: The conventional contrastive loss $\mathcal{L}_{\text{con}}$ comprises an unsupervised term (pulling all augmented pairs together) and a supervised term (pulling together only labeled known-category samples), causing over-compression of known categories and under-compression of novel ones, resulting in blurred cluster boundaries.
5. **CLIP's limitations with long text**: CLIP encodes prompts exceeding 20 tokens poorly, making the conventional concatenation-based prompt construction suboptimal.
6. **Potential harm of inter-modal alignment**: Naively adding inter-modal alignment loss on top of intra-modal losses may in fact degrade intra-modal representation learning.

## Method

### Overall Architecture: SSR²-GCD

The framework consists of three modules: (a) Retrieval-based Text Aggregation (RTA) for generating text representations; (b) the Semi-Supervised Rate Reduction (SSR²) module for representation learning; and (c) a dual-branch classifier for learning pseudo-labels from each modality.

### Retrieval-based Text Aggregation (RTA)

- Following TextGCD, the method maintains a label dictionary and an attribute dictionary, retrieving the top-$c$ most similar label and attribute candidates for each query image.
- **Key improvement**: Rather than concatenating candidates into a long string for CLIP, each candidate is encoded independently and then aggregated with learned weights:

$$\boldsymbol{z}^{\text{T}} = \sum_{i=1}^{c} \sigma_i \mathcal{F}^{\text{T}}(\mathcal{T}(a_i)) + \sum_{i=1}^{c} \sigma_i \mathcal{F}^{\text{T}}(\mathcal{T}(b_i))$$

- Weight assignment: the most similar candidate receives weight $1-\alpha$; the remaining candidates each receive $\frac{\alpha}{c-1}$ ($\alpha=0.5, c=4$), effectively integrating richer candidate information.

### Semi-Supervised Rate Reduction Loss (SSR²)

The core loss is grounded in the Maximal Coding Rate Reduction principle:

$$\mathcal{L}_{\text{SSR}^2} = -R(\mathbf{Z}) + R_c^{\text{s}}(\mathbf{Z}_{\text{s}}, \mathbf{Y}^*) + R_c^{\text{u}}(\mathbf{Z}_{\text{u}}, \mathbf{Y})$$

- **$R(\mathbf{Z})$**: Global coding rate; maximized to spread all representations across the full feature space.
- **$R_c^{\text{s}}$**: Class-conditional coding rate for labeled samples; compresses each known class using ground-truth labels $\mathbf{Y}^*$.
- **$R_c^{\text{u}}$**: Class-conditional coding rate for unlabeled samples; compresses novel categories using classifier-predicted pseudo-labels $\mathbf{Y}$.
- Applied separately to the image and text encoders: $\mathcal{L}_{\text{SSR}^2}^{\text{I}}$ and $\mathcal{L}_{\text{SSR}^2}^{\text{T}}$.
- **Effect**: Global expansion combined with uniform within-class compression yields balanced low-dimensional subspace representations for both known and novel categories.

### Dual-Branch Clustering and Training Strategy

- **Warm-up phase**: $\mathcal{L}_{\text{warm}} = \mathcal{L}_{\text{SSR}^2}^{\text{I}} + \mathcal{L}_{\text{SSR}^2}^{\text{T}} + \mathcal{L}_{\text{cls}}^{\text{I}} + \mathcal{L}_{\text{cls}}^{\text{T}}$
- **Alignment phase**: A co-teaching loss $\mathcal{L}_{\text{co-teach}}$ is introduced, enabling mutual supervision using high-confidence samples.
- **Final prediction**: $\arg\max(\boldsymbol{y}_i^{\text{I}} + \boldsymbol{y}_i^{\text{T}})$

## Key Experimental Results

### Main Results (8 Datasets, All ACC %)

| Dataset | TextGCD | GET | **SSR²-GCD** | Gain |
|---|---|---|---|---|
| ImageNet-100 | 88.0 | 91.7 | **92.1** | +0.4 |
| ImageNet-1k | 64.8 | 62.4 | **66.7** | +1.9 |
| CIFAR-10 | 98.2 | 97.2 | **98.5** | +0.3 |
| CIFAR-100 | 85.7 | 82.1 | **86.4** | +0.7 |
| CUB-200 | 76.6 | 77.0 | **78.3** | +1.3 |
| Stanford Cars | 86.1 | 78.5 | **89.2** | +3.1 |
| Oxford Pets | 93.7 | 91.1 | **95.7** | +2.0 |
| Flowers102 | 87.2 | 85.5 | **93.5** | +6.3 |

Improvements are especially pronounced on Stanford Cars and Flowers102 (+3.1% and +6.3%, respectively).

### Comparison of Representation Learning Objectives (All ACC %)

| Loss Configuration | CIFAR-10 | Stanford Cars | Flowers102 |
|---|---|---|---|
| $\mathcal{L}_{\text{CLIP}}$ (inter-modal only) | 98.3 | 87.0 | 89.7 |
| $\mathcal{L}_{\text{con}}$ (intra-modal only) | 98.4 | 87.9 | 91.8 |
| $\mathcal{L}_{\text{SSR}^2}$ (intra-modal only) | **98.5** | **89.2** | **93.5** |
| $\mathcal{L}_{\text{CLIP}} + \mathcal{L}_{\text{SSR}^2}$ | 98.3 | 88.1 | 92.9 |

Key finding: adding inter-modal alignment loss on top of SSR² consistently degrades performance.

### Ablation Study (Stanford Cars / Flowers102, All ACC %)

| Dual | RTA | SSR² | Stanford Cars | Flowers102 |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 75.2 | 78.3 |
| ✓ | ✗ | ✗ | 81.7 | 83.9 |
| ✓ | ✓ | ✗ | 86.0 | 87.4 |
| ✓ | ✗ | ✓ | 85.5 | 89.1 |
| ✓ | ✓ | ✓ | **89.2** | **93.5** |

Each component contributes independently, and their combination yields optimal performance.

## Highlights & Insights

- **Novel theoretical perspective**: This is the first work to apply the Maximal Coding Rate Reduction principle to multi-modal GCD, replacing conventional contrastive learning with an information-theoretic framework that provides balanced compression guarantees.
- **Counterintuitive yet compelling finding**: Inter-modal alignment can be harmful in multi-modal GCD; intra-modal alignment alone appears sufficient to implicitly achieve cross-modal alignment.
- **Thorough empirical analysis**: The core claims are validated through multiple lenses, including similarity distribution plots, effective rank curves, $R_e$ consistency metrics, and t-SNE visualizations.
- **Elegant RTA design**: By performing weighted aggregation in the embedding space rather than concatenating long prompts, the approach circumvents CLIP's long-text limitations while incorporating richer candidate information.

## Limitations & Future Work

- Computational and memory costs scale linearly with the number of candidates $c$, as each requires a separate pass through the CLIP text encoder.
- Image and text modalities are treated symmetrically, with no adaptive mechanism for modality importance weighting.
- The number of categories $K$ must be known or estimated in advance; robustness to incorrect estimates of the number of novel categories is not discussed.
- Experiments are conducted solely on the CLIP-B/16 backbone; performance on larger models (ViT-L/H) remains unexplored.
- The unlabeled term of the SSR² loss relies on pseudo-label quality, and noisy pseudo-labels in early training may impair convergence.

## Related Work & Insights

| Method | Text Generation | Representation Learning | Clustering Strategy | Characteristics |
|---|---|---|---|---|
| TextGCD | Concatenate top-3 labels + top-2 attributes | $\mathcal{L}_{\text{CLIP}}$ (inter-modal) | Dual-branch + co-teaching | First multi-modal GCD; neglects intra-modal alignment |
| GET | Text inversion network generates prompts | $\mathcal{L}_{\text{CLIP}}+\mathcal{L}_{\text{con}}$ | Single-branch MLP | Uses both inter- and intra-modal losses but combines them naively |
| CLIP-GCD | Knowledge base retrieves similar texts | $\mathcal{L}_{\text{CLIP}}$ | SimGCD clustering | Relies solely on inter-modal alignment |
| **SSR²-GCD** | RTA: weighted aggregation of multiple candidates | $\mathcal{L}_{\text{SSR}^2}$ (intra-modal only) | Dual-branch + co-teaching | First to address imbalanced compression; eliminates inter-modal alignment |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing coding rate reduction into multi-modal GCD offers a distinctive perspective; the finding that inter-modal alignment may be harmful is thought-provoking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 8 datasets, comparison of 6 representation learning configurations, and multi-dimensional analysis (rank, consistency, distribution, visualization).
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rigorous mathematical derivations, though notation is occasionally dense.
- **Value**: ⭐⭐⭐⭐ — Provides a new direction for representation learning in multi-modal GCD, with substantial improvements on fine-grained datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](../../ICLR2026/multimodal_vlm/spectralgcd_spectral_concept_selection_and_cross-modal_representation_learning_f.md)
- [\[CVPR 2026\] IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment](isoclip_decomposing_clip_projectors_for_efficient_intramodal_alignment.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)

</div>

<!-- RELATED:END -->
