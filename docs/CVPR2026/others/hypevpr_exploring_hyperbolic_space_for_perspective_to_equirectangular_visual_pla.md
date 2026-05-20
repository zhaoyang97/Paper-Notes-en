---
title: >-
  [Paper Note] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition
description: >-
  [CVPR 2026][Visual Place Recognition] This paper proposes HypeVPR, a visual place recognition framework based on hierarchical embedding in hyperbolic space…
tags:
  - "CVPR 2026"
  - "Visual Place Recognition"
  - "Hyperbolic Space"
  - "Panoramic Images"
  - "Hierarchical Embedding"
  - "Perspective-to-Equirectangular Matching"
date: 2026-05-08
content_hash: 302de82b35a53203
---

# HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition

**Conference**: CVPR 2026
**arXiv**: [2506.04764](https://arxiv.org/abs/2506.04764)  
**Code**: [https://suhan-woo.github.io/HypeVPR/](https://suhan-woo.github.io/HypeVPR/) (Project Page)  
**Area**: Other
**Keywords**: Visual Place Recognition, Hyperbolic Space, Panoramic Images, Hierarchical Embedding, Perspective-to-Equirectangular Matching

## TL;DR
This paper proposes HypeVPR, a visual place recognition framework based on hierarchical embedding in hyperbolic space, specifically designed to address cross-field-of-view matching between perspective (query) and equirectangular panoramic (database) images. By constructing multi-level descriptors from local to global within the Poincaré ball, HypeVPR achieves a flexible balance among accuracy, efficiency, and storage, achieving retrieval speeds several times faster than sliding-window baselines at comparable accuracy.

## Background & Motivation
Visual Place Recognition (VPR) localizes a query by retrieving the most similar image from a database, and is a core capability for autonomous navigation and mobile robotics. Traditional P2P (perspective-to-perspective) methods require storing images from multiple orientations per location to cover all possible query viewpoints, resulting in substantial storage and retrieval overhead.

The **Perspective-to-Equirectangular (P2E) framework** offers a more practical alternative: the database represents each location with a single panoramic image covering all directions, while queries remain ordinary perspective images. However, P2E faces a fundamental challenge: panoramic images contain 360° information while queries cover only a limited field of view — how can one extract a descriptor from a panorama that simultaneously encodes global context and precisely matches a local viewpoint?

Existing methods (PanoVPR, Orhan et al.) either apply sliding-window cropping over panoramas with pairwise comparison ($O(n)$ comparisons, extremely slow), or fail to capture the structured internal relationships within panoramas.

The paper's core insight is that visual scenes possess a natural **hierarchical structure** — a panorama contains multiple wide-FoV sub-views, each of which contains narrower local viewpoints. **Hyperbolic space** is naturally suited for modeling hierarchical relationships (exponentially growing distances allow low-distortion embedding of tree structures), making it more appropriate than Euclidean space for organizing multi-level descriptors of panoramic images.

## Method

### Overall Architecture
HypeVPR comprises two network branches: (1) a query network $\mathcal{F}_q$ that generates a single hyperbolic descriptor $\mathbf{h}_q$ from a perspective image; and (2) a database network $\mathcal{F}_d$ that decomposes a panoramic image into multi-level windows and generates a set of multi-level hyperbolic descriptors $\mathbf{H}_d$ from local to global via a Hierarchical Aggregation Module (HAM). At retrieval time, top-level descriptors are used for fast coarse filtering, followed by re-ranking with lower-level descriptors.

### Key Designs

1. **Hierarchical Modeling of Panoramic Images**:

    - Function: Decompose a panoramic image into $L$ levels of windows with decreasing horizontal field of view.
    - Mechanism: The top level $\ell=1$ is the full panorama; each subsequent level halves the horizontal FoV: $I_d^{(\ell)} \in \mathbb{R}^{H \times \frac{W'}{2^{\ell-1}} \times C}$. The bottom-level windows match the resolution of the query image, enabling shared backbone weights. This constitutes a hierarchical tree from global to local.
    - Design Motivation: Only a small portion of a panorama matches the query; the remainder is redundant. Hierarchical modeling allows the system to first determine the approximate direction globally, then refine to the matching region.

2. **Hierarchical Aggregation Module (HAM)**:

    - Function: Progressively aggregate Euclidean features from bottom-level windows into hierarchical descriptors in hyperbolic space.
    - Mechanism: Each window's Euclidean descriptor $\mathbf{d}_d^{(\ell,j)}$ is obtained via GeM pooling followed by a linear layer, then mapped onto the Poincaré ball via the exponential map: $\mathbf{h}_d^{(\ell,j)} = \exp_0^c(\mathbf{d}_d^{(\ell,j)})$. Adjacent windows' hyperbolic descriptors are aggregated via the **Einstein midpoint**: $\mathcal{A}_{hyp}(h_1,...,h_n) = \frac{\sum_j \gamma_j h_j}{\sum_j \gamma_j}$, where $\gamma_j = \frac{1}{\sqrt{1-c\|h_j\|^2}}$ is the Lorentz factor.
    - Design Motivation: In hyperbolic space, points farther from the origin represent finer-grained concepts, while points closer to the origin represent more abstract global semantics. The Einstein midpoint respects the distance structure of hyperbolic geometry through norm-aware weighting, preventing the loss of hierarchical information during aggregation.

3. **Adjustable Hierarchical Retrieval**:

    - Function: Flexibly adjust the accuracy-efficiency trade-off without retraining.
    - Mechanism: Top-level descriptors $\mathbf{h}_d^{(1,1)}$ are first used to retrieve Top-$K'$ candidates; sub-descriptors from selected levels $\mathbb{L}$ then re-score candidates: $d_\ell = \min_k d_c(\mathbf{h}_q, \mathbf{h}_d^{(\ell,k)})$. After Z-score normalization, scores are combined as $s = \sum_{\ell \in \{1\} \cup \mathbb{L}} w_\ell \hat{s}_\ell$.
    - Design Motivation: Different deployment scenarios impose different requirements on speed and accuracy. The hierarchical structure naturally supports a cascade strategy of "coarse matching for speed, fine matching for precision," allowing users to control the trade-off by selecting which levels to activate.

### Loss & Training
Three loss functions are employed: (1) **Hierarchical triplet loss** $\mathcal{L}_{hier}$ — descriptors with overlapping FoV at adjacent levels form positive pairs, while non-overlapping descriptors at the same level form negative pairs, using hyperbolic distance; (2) a standard triplet loss for query-database matching; (3) end-to-end training under the triplet loss framework.

## Key Experimental Results

### Main Results

| Method | Backbone | Pitts250K R@1 | Pitts250K R@5 | YQ360 R@1 | YQ360 R@5 | Time/Query (ms) |
|------|----------|------|------|------|------|------|
| PanoVPR×16 (ConvNeXt-S) | ConvNeXt-S | 40.3 | 63.0 | 46.0 | 83.2 | 48.6 |
| HypeVPR-L (ConvNeXt-S) | ConvNeXt-S | **43.4** | **64.3** | **52.4** | **85.2** | **14.0** |
| Orhan et al.* | ResNet-101 | 47.0 | 66.4 | 47.6 | 79.2 | 1555.2 |
| HypeVPR-O* | ResNet-50 | **66.5** | **82.1** | **53.6** | **81.2** | **4.0** |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| HypeVPR-B vs PanoVPR×8 (Swin-T) | R@1: 29.4 vs 22.0 (Pitts) | Single descriptor outperforms 8-window sliding baseline |
| HypeVPR-L vs PanoVPR×16 (Swin-T) | R@1: 32.5 vs 33.6 (Pitts) | Comparable accuracy, 3.5× faster |
| HypeVPR-B speed vs PanoVPR×8 | 3.6ms vs 17.0ms | 4.7× speedup |
| HypeVPR-L speed vs PanoVPR×16 | 14.0ms vs 48.6ms | 3.5× speedup |
| HypeVPR-O* vs Orhan et al.* | Time: 4.0ms vs 1555.2ms | 388× speedup, R@1 +19.5 |

### Key Findings
- Under additional large-scale training data (HypeVPR-O*), Pitts250K R@1 reaches 66.5%, far surpassing Orhan et al.'s 47.0%, with 388× speedup.
- Hierarchical retrieval enables HypeVPR to outperform PanoVPR even when using fewer sub-descriptors (-B mode vs. more sliding windows).
- Hyperbolic space embedding preserves local-to-global hierarchical relationships within panoramas better than Euclidean space.
- The accuracy-efficiency trade-off can be flexibly controlled at inference time by selecting active levels, without retraining.

## Highlights & Insights
- Introducing hyperbolic space into VPR is a highly natural and compelling innovation: the panorama → sub-view → local region nesting relationship is inherently tree-structured.
- Einstein midpoint aggregation ensures that hierarchical aggregation respects hyperbolic geometry, avoiding geometric distortion introduced by naive averaging.
- Adjustable hierarchical retrieval is a practically valuable feature: on edge devices, modest accuracy sacrifices can yield substantial speedups.

## Limitations & Future Work
- The number of bottom-level windows grows exponentially with the number of levels ($2^{L-1}$), resulting in non-trivial memory and computational overhead for large $L$.
- Experiments are conducted on only two datasets (Pitts250K-P2E and YQ360), leaving generalizability insufficiently validated.
- The per-level weights $w_\ell$ in hierarchical retrieval appear to be set manually; learned or adaptive weighting could be explored.
- The shared backbone applies identical feature extraction to both perspective queries and panoramic windows, which may not be globally optimal.

## Related Work & Insights
- The key distinction from PanoVPR is that PanoVPR performs brute-force sliding-window search at the P2P level, whereas HypeVPR reduces matching complexity from $O(n)$ to $O(\log n)$ via hierarchical embedding.
- Hyperbolic embedding has been applied in NLP (Poincaré embeddings) and image retrieval; this paper represents its first application to P2E VPR.
- The hierarchical triplet loss design is generalizable to other visual matching problems with natural hierarchical structure.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of hyperbolic space, hierarchical panorama modeling, and adjustable retrieval is highly novel and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons across multiple frameworks are thorough, though the number of datasets is limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical foundations are solid, figures are clear, and motivation is articulated convincingly.
- Value: ⭐⭐⭐⭐ A practical solution for P2E VPR; the speed advantage is of considerable value for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ICCV 2025\] A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](../../ICCV2025/others/a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)
- [\[CVPR 2026\] UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](../../ICLR2026/others/out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)
- [\[ICLR 2026\] HEEGNet: Hyperbolic Embeddings for EEG](../../ICLR2026/others/heegnet_hyperbolic_embeddings_for_eeg.md)

</div>

<!-- RELATED:END -->
