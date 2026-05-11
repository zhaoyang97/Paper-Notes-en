---
title: >-
  [Paper Note] HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation
description: >-
  [ICLR 2026][Image Generation][Visual Geolocation] HierLoc reformulates visual geolocation as an image-to-entity alignment problem in hyperbolic space…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Visual Geolocation"
  - "Hyperbolic Embeddings"
  - "Hierarchical Entities"
  - "Contrastive Learning"
  - "Retrieval"
date: 2026-05-08
content_hash: 6d42cdd54033e50b
---

# HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation

**Conference**: ICLR 2026
**arXiv**: [2601.23064](https://arxiv.org/abs/2601.23064)
**Code**: None
**Area**: Diffusion Models
**Keywords**: Visual Geolocation, Hyperbolic Embeddings, Hierarchical Entities, Contrastive Learning, Retrieval

## TL;DR
HierLoc reformulates visual geolocation as an image-to-entity alignment problem in hyperbolic space, replacing 5M+ image embeddings with ~240K geographic entity embeddings. It achieves a 19.5% reduction in mean geodesic error and a 43% improvement in sub-region accuracy on OSV5M.

## Background & Motivation
Visual geolocation—inferring the capture location from image content—is a global, cross-scale challenge. Existing approaches fall into three categories: retrieval-based (requiring indexing of millions of image embeddings), classification-based (grid-cell classification that ignores geographic continuity), and generative-based (diffusion models that struggle at fine-grained scales). The root cause lies in the inherent hierarchical structure of geography (country → region → sub-region → city): the number of entities grows exponentially from country to city level, yet Euclidean distance grows only linearly, causing deep-level entities to become crowded and lose discriminability. Hyperbolic space naturally provides exponential volume growth, perfectly matching this hierarchical branching structure. HierLoc's key starting point is reframing geolocation from "image-to-image retrieval" to "image-to-entity alignment."

## Method

### Overall Architecture
Geographic entities and images are embedded in the Lorentz hyperbolic space. Images are encoded by a frozen visual encoder (DINOv3) and projected onto the hyperbolic manifold; entities are represented by fusing image, text, and coordinate modalities. Cross-modal attention aligns images with four-level hierarchical entities, and a pretrained Geo-Weighted Hyperbolic InfoNCE (GWH-InfoNCE) loss supervises the alignment. At inference, beam search retrieves predictions by traversing the entity hierarchy.

### Key Designs
1. **Hierarchical Entity Construction and Embedding**:

    - Function: Compresses training metadata into ~240K hierarchical entities (233 countries, 4,946 regions, 29,214 sub-regions, 209,894 cities).
    - Mechanism: Each entity is associated with three modalities—mean image embedding $\text{Img}_i$ (averaged DINOv3 features of training images), text embedding $\text{Text}_i$ (CLIP-encoded entity name), and coordinate embedding $\text{Coords}_i$ (SphereM+ encoded). Anchor embeddings $A_i$ are randomly initialized in the tangent space at the origin and mapped to the hyperboloid; the final embedding is $H_i = \exp_O(\log_0(A_i) + \alpha_{\text{node}} \Delta_i)$.
    - Design Motivation: Although simple, mean embeddings yield stable and discriminative prototypes at the entity level.

2. **Cross-Modal Attention**:

    - Function: Performs multi-head attention in the tangent space with image features as queries and entity features as keys/values.
    - Mechanism: Eight-head attention is applied independently at each hierarchical level; context vectors from all four levels are concatenated, fused via MLP, and added back to the original image features. Only the image stream is updated; entity embeddings remain fixed—preventing overfitting to training data.
    - Design Motivation: This asymmetric update strategy ensures the generalizability of entity embeddings.

3. **GWH-InfoNCE Loss**:

    - Function: Incorporates geographic structure into hyperbolic contrastive learning.
    - Mechanism: Negative samples are weighted by great-circle distances $g_{\ell,k}$ computed via the haversine formula: $w_{\ell,k} = 1 + \lambda \exp(-g_{\ell,k}/\sigma)$. The loss is $\mathcal{L}_\ell = -\log \frac{\exp(-d_\ell^+/\tau)}{\exp(-d_\ell^+/\tau) + \sum_k w_{\ell,k} \exp(-d_{\ell,k}^-/\tau)}$. The total loss aggregates across levels: $\mathcal{L} = \sum_{\ell} \beta_\ell \mathcal{L}_\ell$.
    - Design Motivation: Geographically proximate negatives are harder to distinguish and should receive higher weights to enhance fine-grained discrimination.

### Loss & Training
- AdamW is used for Euclidean parameters; RiemannianAdam is used for manifold parameters.
- Batch size 16, learning rate $2\times10^{-4}$, trained for 5 epochs on 6× L40S GPUs (~60 hours).
- Inference uses beam search (beam width 10) to progressively refine predictions across the entity hierarchy.

## Key Experimental Results

### Main Results (OSV5M Benchmark)

| Method | GeoScore↑ | Distance (km)↓ | Country% | Region% | Sub-region% | City% |
|--------|-----------|----------------|----------|---------|-------------|-------|
| SC Retrieval | 3597 | 1386 | 73.4 | 45.8 | 28.4 | 19.9 |
| LocDiff | - | - | 77.0 | 46.3 | - | 11.0 |
| **HierLoc (DINOV3)** | **3963** | **861** | **82.9** | **55.0** | **40.7** | **23.3** |

### Ablation Study (Component Contributions)

| Configuration | GeoScore | Notes |
|---------------|----------|-------|
| Euclidean space | Baseline | Deep-level entity crowding |
| + Hyperbolic space | Improved | Exponential volume growth |
| + GWH-InfoNCE | Best | Geography-aware negative weighting |
| Laplace vs. Gaussian decay | Laplace superior | Choice of decay kernel matters |

### Key Findings
- Country accuracy +8.8%, region +20.1%, sub-region +43.2%, city +16.8%.
- Mean geodesic error reduced by 19.5% (1,386 km → 861 km vs. SC Retrieval).
- Search space substantially reduced by compressing ~9.6M image records to 240K entities.
- DINOv3 encoder outperforms ViT-L/14.

## Highlights & Insights
- The "image-to-entity alignment" paradigm reduces retrieval complexity from $O(N)$ to sub-linear via hierarchical traversal.
- The design intuition behind geographic distance-weighted negatives in GWH-InfoNCE is particularly elegant—geographically close samples are the truly hard negatives.
- The asymmetric cross-modal attention (updating only image features while keeping entity embeddings fixed) effectively prevents overfitting.

## Limitations & Future Work
- Using mean image embeddings for city-level entities may discard visual diversity information.
- The beam search width is fixed at 10; an adaptive strategy may yield better results.
- The method requires a pre-constructed hierarchy, which may be limited for regions lacking administrative division data.

## Related Work & Insights
- **vs. PIGEON**: Relies on large-scale classification with semantic fusion, but collapses to a single-level output, discarding hierarchical signals.
- **vs. GeoCLIP**: Directly uses coordinates as prediction targets without exploiting hierarchical structure.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of hyperbolic embeddings to global hierarchical visual geolocation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on OSV5M with validation on multiple external benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Method description is detailed with clear mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ Geometry-aware hierarchical embeddings offer broader inspiration for other hierarchical structure tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [\[AAAI 2026\] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval](../../AAAI2026/image_generation/hyperbolic_hierarchical_alignment_reasoning_network_for_text-3d_retrieval.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](../../ICCV2025/image_generation/hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [\[ICLR 2026\] Compositional amortized inference for large-scale hierarchical Bayesian models](compositional_amortized_inference_for_large-scale_hierarchical_bayesian_models.md)

</div>

<!-- RELATED:END -->
