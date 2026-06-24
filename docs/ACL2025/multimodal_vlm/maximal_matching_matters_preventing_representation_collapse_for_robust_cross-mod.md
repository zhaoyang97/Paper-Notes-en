---
title: >-
  [Paper Note] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval
description: >-
  [ACL 2025][Multimodal VLM][Cross-Modal Retrieval] Proposes the MaxMatch method, which addresses the issues of sparse supervision and set collapse in set embedding methods through a maximal pair assignment similarity based on the Hungarian algorithm and two new loss functions, achieving SOTA performance on MS-COCO and Flickr30k.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Cross-Modal Retrieval"
  - "Set Embeddings"
  - "Hungarian Algorithm"
  - "Representation Collapse"
  - "Image-Text Matching"
date: 2026-05-08
content_hash: 38ab2217579375fd
---

# Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval

**Conference**: ACL 2025  
**arXiv**: [2506.21538](https://arxiv.org/abs/2506.21538)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Cross-Modal Retrieval, Set Embeddings, Hungarian Algorithm, Representation Collapse, Image-Text Matching

## TL;DR

Proposes the MaxMatch method, which addresses the issues of sparse supervision and set collapse in set embedding methods through a maximal pair assignment similarity based on the Hungarian algorithm and two new loss functions, achieving SOTA performance on MS-COCO and Flickr30k.

## Background & Motivation

The core challenge of cross-modal image-text retrieval lies in the "one-to-many" relationship: an image can be described by multiple texts, and a single text can correspond to various visual scenes. This makes single-vector embeddings difficult to fully capture cross-modal semantics.

Existing methods mainly fall into two categories:

**Single-vector embedding methods** (e.g., VSE++, VSE∞): Represent each sample with a single vector, which offers retrieval efficiency benefits but fails to capture diverse semantic relationships.

**Cross-attention methods** (e.g., SCAN, SGRAF): Compute the similarity of image-text pairs through joint attention. While they perform better, they incur extremely high inference costs as they require joint processing for each query-candidate pair.

Set embedding methods (e.g., PVSE, SetDiv) serve as a trade-off—representing each sample with a set of multiple vectors, which maintains the efficiency of independent encoders while capturing richer semantics. However, the authors identify two key limitations in these methods:

- **Sparse Supervision**: PVSE utilizes Multiple Instance Learning (MIL) similarity, only taking the maximum similarity between sets, which results in most embeddings receiving no gradient updates.
- **Set Collapse**: SetDiv's Smooth-Chamfer similarity allows all embeddings to receive gradients, but the smoothing effect of the log function causes all embeddings to converge, losing diversity.

The authors clearly demonstrate these two issues through t-SNE visualizations and similarity heatmaps: MIL produces isolated high-similarity points (only one embedding is utilized), while Smooth-Chamfer yields a uniform high-similarity matrix (all embeddings collapse into a single point).

## Method

### Overall Architecture

MaxMatch adopts a dual-encoder architecture (visual encoder + textual encoder), where each encoder is followed by a set prediction module mapping the input to a set of $K$ embedding vectors. The core innovations lie in the similarity calculation and the design of the loss functions.

### Key Designs

1. **Set Prediction Module**: Adopts the Slot Attention design from SetDiv. $K$ learnables queries (slots) iteratively aggregate information from local features through $L$ layers of cross-attention, which are finally added to global features to form the embedding set $S$. The semantics of each slot can be interpreted as an "offset" from the global feature, representing different semantic aspects of the input.

2. **Maximal Pair Assignment Similarity**: This is the core contribution of this work. Given an image embedding set $V_i$ and a text embedding set $T_j$, a $K \times K$ cosine similarity matrix $S(V_i, T_j)$ is first computed. Then, the Hungarian algorithm is used to solve the optimal one-to-one matching:

    - Find the optimal permutation $\pi^* = \operatorname{argmax}_\pi \operatorname{tr}(S(V_i, \pi(T_j)))$
    - Generate a binary mask $M_{ij}$ to mark the matched pairs
    - The final similarity is formulated as $\Sigma(M_{ij} \odot S_{ij})$, which is summed after exponential scaling.

   The key advantage of this design is that every embedding participates in the matching (avoiding sparse supervision), but is only paired with its best-matched counterpart (preventing collapse from averaging).

3. **Global Discriminative Loss**: Pushes each embedding in the set away from its corresponding global feature vector (i.e., the global representation prior to slot attention), ensuring that residual embeddings do not degenerate into duplicates of the global feature. It uses exponential activation, cosine similarity, and a margin $\delta_2$ to control the degree of push.

4. **Intra-Set Divergence Loss**: Penalizes high similarity between any two embeddings within the same set, forcing each embedding to capture distinct semantics. It iterates over all non-identical pairs $(j, k)$ and imposes a penalty when $\cos(v_{i,j}, v_{i,k}) > \operatorname{margin} \delta_3$.

### Loss & Training

The total loss function is the sum of six terms:

$$\mathcal{L} = \mathcal{L}_{TRI} + λ_{GD}\mathcal{L}_{GD} + λ_{ISD}\mathcal{L}_{ISD} + λ_{MMD}\mathcal{L}_{MMD} + λ_{Div}\mathcal{L}_{Div} + λ_{CON}\mathcal{L}_{CON}$$

where:
- **Triplet Loss (TRI)**: Hinge triplet loss with hardest negative mining.
- **GD / ISD**: The two new losses proposed in this paper.
- **MMD**: Maximum Mean Discrepancy, aligning embedding distributions of both modalities.
- **Div**: Diversity regularization (adopting the design of PVSE).
- **CON**: Contrastive loss, assisting in stabilizing early training.

Inference Phase: Instead of using Hungarian matching (which is computationally expensive), top-$k$ most similar embedding pairs are selected to compute similarity, significantly accelerating the process.

## Key Experimental Results

### Main Results (Flickr30k)

| Method | Encoder | I→T R@1 | T→I R@1 | RSUM |
|------|--------|---------|---------|------|
| SetDiv | ResNet+GRU | 61.8 | 46.1 | 442.6 |
| **MaxMatch** | ResNet+GRU | **68.6** | **51.5** | **469.9** |
| SetDiv | FRCNN+BERT | 81.3 | 62.4 | 514.8 |
| **MaxMatch†** | FRCNN+BERT | **86.2** | **64.8** | **527.1** |
| CORA†‡ | FRCNN+BERT | 83.4 | 64.1 | 523.3 |

### Main Results (COCO 5K)

| Method | Encoder | I→T R@1 | T→I R@1 | RSUM |
|------|--------|---------|---------|------|
| SetDiv | ResNet+GRU | 47.2 | 33.8 | 377.7 |
| **MaxMatch** | ResNet+GRU | **51.84** | **36.35** | **398.95** |
| SetDiv | FRCNN+BERT | 62.3 | 42.8 | 439.6 |
| **MaxMatch†** | FRCNN+BERT | **65.4** | **45.1** | **452.6** |

### Ablation Study

| Similarity Function | Div | MMD | GD | ISD | RSUM (COCO) |
|-----------|-----|-----|----|----|-------------|
| MIL | ✓ | ✓ | | | 438.98 |
| Smooth-Chamfer | ✓ | ✓ | | | 439.57 |
| Smooth-Chamfer | ✓ | ✓ | ✓ | ✓ | 444.10 |
| MaxPair | ✓ | ✓ | | | 441.23 |
| MaxPair | | | ✓ | ✓ | 444.49 |
| **MaxPair** | **✓** | **✓** | **✓** | **✓** | **446.53** |

Set collapse analysis (circular variance, higher is better): MIL = -7.35, Smooth-Chamfer = -2.13, MaxMatch = **-1.68**

### Key Findings

1. Under the ResNet+GRU configuration, MaxMatch improves the RSUM of SetDiv by up to +27.3 (Flickr30k) and +21.25 (COCO 5K).
2. Under the FRCNN+BERT configuration, MaxMatch outperforms the CORA method which utilizes external knowledge graphs.
3. The circular variance of MaxMatch's embedding set is significantly higher than other methods, demonstrating that set collapse is effectively mitigated.
4. Using the GD+ISD losses alone (without Div/MMD) already achieves an RSUM of 444.49, proving the effectiveness of the proposed losses.

## Highlights & Insights

1. **Accurate Problem Analysis**: Explains and clearly demonstrates the sparse supervision of MIL and the set collapse of Smooth-Chamfer through t-SNE visualizations and similarity heatmaps, making the problem definition highly convincing.
2. **Ingenious Application of the Hungarian Algorithm**: Naturally formulates the optimal matching problem as an assignment problem, which guarantees that all embeddings are utilized while maintaining one-to-one matching constraints.
3. **Training-Inference Decoupled Design**: Employs the Hungarian algorithm during training to ensure precise matching, and uses top-$k$ approximation during inference for acceleration, boosting practicality.
4. **Well-Structured Loss Function Design**: GD is responsible for pushing embeddings away from the global center (preventing degeneration into single vectors), while ISD pushes embeddings within the same set away from each other (preventing set collapse), with clearly defined roles.

## Limitations & Future Work

1. **Dataset Limitations**: Only evaluated on MS-COCO and Flickr30k. The descriptions in these datasets predominantly focus on concrete objects and actions, lacking evaluations of abstract concepts or emotional content.
2. **Modality Constraints**: Currently supports only image-text bi-modality. Extending to multi-modal scenarios such as audio and video requires redesigning the matching mechanism.
3. **Computational Overhead of the Hungarian Algorithm**: Although top-$k$ replaces it during inference, Hungarian matching must still be performed for all image-text pairs within each batch during training.
4. **Fixed Set Size $K$**: All samples use a fixed number of embeddings, which may introduce redundancy for samples with simple semantics.

## Related Work & Insights

- **PVSE / SetDiv**: Direct targets of improvement in this work; MaxMatch addresses sparse supervision and set collapse issues on top of them.
- **CORA**: Uses external scene graphs to enhance textual representation, proving the importance of rich semantic representation.
- **CLIP Series**: Successful large-scale pre-training methods in cross-modal retrieval, but require joint processing.
- The **application of slot attention in set learning** is also noteworthy; more NLP/CV tasks may benefit from similar set prediction paradigms.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The application of the Hungarian algorithm in set embedding similarity is novel and natural, and the design of the two loss functions is reasonable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage on various encoder combinations, ablation studies, visualization analyses, and quantitative analysis of set collapse.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, rich visualizations, and complete methodological derivations.
- **Value**: ⭐⭐⭐⭐ — Provides a practical improvement scheme for set embedding methods, though application scenarios are limited by traditional visual encoders.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)
- [\[ACL 2025\] MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors](multimm_cultural_metaphor.md)
- [\[CVPR 2025\] NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](../../CVPR2025/multimodal_vlm/neighborretr_balancing_hub_centrality_in_cross-modal_retrieval.md)
- [\[CVPR 2026\] ReMatch: Boosting Representation through Matching for Multimodal Retrieval](../../CVPR2026/multimodal_vlm/rematch_boosting_representation_through_matching_for_multimodal_retrieval.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](../../NeurIPS2025/multimodal_vlm/on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)

</div>

<!-- RELATED:END -->
