---
title: >-
  [Paper Note] CAE: Hierarchical Semantic Alignment for Image Clustering
description: >-
  [AAAI 2026][image clustering] By combining two complementary semantic sources — noun-level (WordNet) and caption-level (Flickr image captions) — and constructing a semantic space via optimal transport alignment followed…
tags:
  - "AAAI 2026"
  - "image clustering"
  - "semantic alignment"
  - "optimal transport"
  - "caption-level semantics"
  - "training-free"
date: 2026-05-08
content_hash: 25f201f23b7082c4
---

# CAE: Hierarchical Semantic Alignment for Image Clustering

**Conference**: AAAI 2026
**arXiv**: [2512.00904](https://arxiv.org/abs/2512.00904)  
**Code**: None  
**Area**: Other
**Keywords**: image clustering, semantic alignment, optimal transport, caption-level semantics, training-free

## TL;DR
By combining two complementary semantic sources — noun-level (WordNet) and caption-level (Flickr image captions) — and constructing a semantic space via optimal transport alignment followed by adaptive fusion, this work achieves training-free image clustering with a 4.2% accuracy improvement on ImageNet-1K.

## Background & Motivation

**Background**: Image clustering has evolved from contrastive learning and self-supervised methods toward leveraging external textual semantics. SIC employs WordNet nouns, while TAC combines noun embeddings with image embeddings.

**Limitations of Prior Work**: Relying solely on nouns introduces two problems — polysemy (e.g., "crane" can refer to a bird or a machine) and insufficient granularity (e.g., "spaniel" cannot distinguish between different dog breeds).

**Key Challenge**: Nouns provide high-level category information but lack attribute-level detail; a single semantic type is insufficient for disambiguation.

**Goal**: Construct a more precise external semantic space to guide image clustering.

**Key Insight**: Nouns (categories) and captions (attributes) are complementary — nouns encode "what it is," while captions encode "what it looks like."

**Core Idea**: Combine WordNet nouns and Flickr captions, construct a dual semantic space via OT alignment, and adaptively fuse the two sources.

## Method

### Overall Architecture

The inputs are CLIP-encoded image embeddings, WordNet noun embeddings, and Flickr caption embeddings. The pipeline consists of two stages: (1) **Semantic space construction** — filtering relevant subsets of nouns/captions and computing semantic correspondences for each image via OT; (2) **Adaptive fusion** — a prototype-guided weighting mechanism fuses tri-modal features, followed by k-means clustering.

### Key Designs

1. **Semantic Space Construction**

    - **Function**: Filter a semantically relevant subset from the large noun/caption corpus.
    - **Mechanism**: Apply k-means on image embeddings to obtain $n = N/300$ cluster centers; for each center, select the top-$K$ most similar nouns/captions and take the union.
    - **Design Motivation**: The space must be neither too large (introducing noise) nor too small (losing information).

2. **Optimal Transport for Semantic Correspondence**

    - **Function**: Compute noun correspondences $\mathbf{x}_i^u$ and caption correspondences $\mathbf{x}_i^v$ for each image.
    - **Mechanism**: Image-noun alignment is formulated as an OT problem; the Sinkhorn-Knopp algorithm solves for the transport plan $\mathbf{T}^u$, and the correspondence is computed as $\mathbf{x}_i^u = \sum_j t_{i,j}^u s_{i,j}^u \mathbf{u}_j$.
    - **Design Motivation**: Theorem 1 proves that the column constraints of OT ensure balanced noun utilization, and the resulting semantic error is provably no greater than that of softmax.

3. **Adaptive Semantic Fusion**

    - **Function**: Dynamically adjust tri-modal fusion weights at the sample level.
    - **Mechanism**: A semantic prototype is computed as $\mathbf{x}_i^p = \frac{1}{3}(\mathbf{x}_i + \mathbf{x}_i^u + \mathbf{x}_i^v)$; the cosine similarity between each modality and the prototype is scaled by a softmax temperature to derive fusion weights.
    - **Design Motivation**: Different images rely on nouns and captions to varying degrees, necessitating sample-level adaptation.

### Loss & Training

The method is entirely training-free. OT is solved via Sinkhorn iteration; k-means is applied after fusion. Temperature $\gamma = 0.01$.

## Key Experimental Results

### Main Results

| Dataset | Metric | CAE | Prev. SOTA | Gain |
|--------|------|-----|----------|------|
| ImageNet-1K | ACC | **76.5%** | 72.3% (SIC) | +4.2% |
| ImageNet-1K | ARI | **56.8%** | 53.9% | +2.9% |
| DTD | ACC | **52.3%** | 47.7% | +4.6% |
| UCF-101 | ACC | **71.6%** | 69.3% | +2.3% |

### Ablation Study

| Configuration | ImageNet-1K ACC | Note |
|------|----------------|------|
| Full CAE | **76.5%** | Complete model |
| w/o Captions | 72.8% | −3.7% without captions |
| w/o Nouns | 71.2% | −5.3% without nouns |
| Softmax instead of OT | 74.1% | OT outperforms softmax by 2.4% |
| Simple concatenation | 74.8% | Adaptive fusion outperforms concatenation by 1.7% |

### Key Findings
- Nouns and captions are complementary: removing either leads to a substantial performance drop, with captions contributing slightly more.
- OT outperforms softmax by 2.4%: column constraints ensure balanced utilization of nouns.
- Gains are more pronounced on challenging datasets (DTD texture: +4.6%; UCF-101 action: +2.3%).

## Highlights & Insights
- **Dual semantic complementarity**: Combining noun-level categories and caption-level attributes reduces the average cosine similarity among ImageNet-1K image embeddings from 0.73 to 0.35.
- **Theoretical advantage of OT**: Theorem 1 rigorously proves OT superiority over softmax; column constraints prevent winner-takes-all collapse.
- **Fully training-free**: Relies only on CLIP and external databases; runs on a single RTX 3090.

## Limitations & Future Work
- Performance depends on the quality of CLIP embeddings and may degrade in domains where CLIP underperforms.
- The caption source is fixed (Flickr), providing insufficient coverage for specialized domains.
- The final clustering step uses k-means, which may be inadequate for non-convex cluster structures.
- The number of clusters $K$ must be specified in advance.

## Related Work & Insights
- **vs. SIC**: SIC relies solely on WordNet nouns and is susceptible to polysemy; CAE introduces captions to resolve ambiguity, yielding +4.2%.
- **vs. TAC**: TAC combines nouns via cross-modal distillation but operates at a single semantic level; CAE introduces a caption-level hierarchy.
- **vs. VIC**: VIC generates captions using MLLMs but requires known class names; CAE is fully unsupervised.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of dual semantics and OT alignment is novel, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 8 datasets with comprehensive ablations and theoretical proofs.
- **Writing Quality**: ⭐⭐⭐⭐ The polysemy motivation figure is intuitive and effective.
- **Value**: ⭐⭐⭐⭐ A significant +4.2% on ImageNet-1K; strong practical utility as a training-free method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/others/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[AAAI 2026\] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)

</div>

<!-- RELATED:END -->
