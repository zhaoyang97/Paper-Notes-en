---
title: >-
  [Paper Note] GeoMM: On Geodesic Perspective for Multi-Modal Learning
description: >-
  [CVPR 2025][Multimodal VLM][Geodesic Distance] This work introduces geodesic distance into multimodal contrastive learning for the first time. By constructing a hierarchical graph structure, it efficiently calculates the manifold distance between samples to replace the traditional cosine distance. This enables more accurate mining of positive and negative sample relationships, improving performance in downstream tasks such as image-text retrieval and VQA.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Geodesic Distance"
  - "Contrastive Learning"
  - "Hierarchical Graph Structure"
  - "Multimodal Pre-training"
  - "Manifold Learning"
date: 2026-05-08
content_hash: bc662eac3adbafbe
---

# GeoMM: On Geodesic Perspective for Multi-Modal Learning

**Conference**: CVPR 2025  
**arXiv**: [2505.11216](https://arxiv.org/abs/2505.11216)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Geodesic Distance, Contrastive Learning, Hierarchical Graph Structure, Multimodal Pre-training, Manifold Learning

## TL;DR

This work introduces geodesic distance into multimodal contrastive learning for the first time. By constructing a hierarchical graph structure, it efficiently calculates the manifold distance between samples to replace the traditional cosine distance. This enables more accurate mining of positive and negative sample relationships, improving performance in downstream tasks such as image-text retrieval and VQA.

## Background & Motivation

**Background**:  
The current mainstream paradigm in multimodal learning (e.g., CLIP, ALBEF, TCL) is to encode images and texts into a unified representation space, pulling matching samples closer and pushing mismatching samples further apart via contrastive loss (such as InfoNCE). These methods rely on distance calculations between samples to mine positive and negative pairs.

**Limitations of Prior Work**:  
1. Existing methods assume that samples are distributed in a spherical space and use cosine distance as the metric, ignoring the possibility that the data is distributed in more complex non-Euclidean geometric spaces.
2. Some sentences with word-level similarities may have very close cosine distances but completely different semantics (as shown in Fig. 1), which traditional distance metrics fail to distinguish effectively.
3. Traditional distance computation is "one-to-one", where the distance between two points depends only on the two points themselves, without considering the global topological structure.

**Key Challenge**:  
The distribution of samples in the multimodal feature space is complex, and the simple cosine distance cannot accurately depict the true manifold distance between samples, leading to imprecise mining of positive and negative samples in contrastive learning.

**Goal**:  
How to introduce a distance metric that better reflects the data manifold space in multimodal contrastive learning, so as to more accurately depict similarity relationships between samples.

**Key Insight**:  
Drawing on the concept of geodesic distance from differential geometry—geodesic distance considers the shortest path distance along the data manifold rather than straight-line distance, which can better reflect the true relationships between samples on complex manifolds.

**Core Idea**:  
Replace the traditional cosine distance in contrastive learning with geodesic distance, and efficiently compute the geodesic distance in large-scale sample pools through a hierarchical graph structure.

## Method

### Overall Architecture

Based on the dual-stream architecture of ALBEF, a momentum feature queue is maintained as a sample pool. On this basis, a hierarchical graph structure is constructed to compute geodesic distances between samples, replacing the original cosine similarity for contrastive learning. The overall training includes three pre-training tasks: ITC (Image-Text Contrastive), MLM (Masked Language Modeling), and ITM (Image-Text Matching).

### Key Designs

1. **Topology-Based Geodesic Distance Computation**:
    - **Function**: Replace the traditional cosine distance with the shortest path distance on the manifold.
    - **Mechanism**: Build a nearest-neighbor graph for the points in the sample pool, where each point connects to its $n$ nearest neighbors with edge weights representing local simple distances; then compute the geodesic distance between any two points via the Floyd shortest path algorithm.
    - **Design Motivation**: The local space satisfies the "simple manifold assumption" (geodesic distance $\approx$ simple distance), so a local simple metric combined with global shortest paths can be used to approximate the true geodesic distance.

2. **Hierarchical Graph**:
    - **Function**: Resolve the computational complexity issues of directly calculating geodesic distances in large-scale sample pools.
    - **Mechanism**: Perform multi-layer clustering on samples using K-Means, constructing graphs and calculating Floyd shortest paths only among cluster centers at each layer. The distance between any two points is obtained by backtracking from the bottom layer and accumulating the distances to the cluster centers and the geodesic distances between cluster centers at each layer.
    - **Design Motivation**: Directly building a graph and running Floyd's algorithm on 65536 samples yields excessively high computational complexity ($O(N^3)$). After stratification, running Floyd only on 256 cluster centers significantly reduces the computation.

3. **Incremental Updates and Dynamic Graph Maintenance**:
    - **Function**: Efficiently update features of the new batch into the graph structure at each training step.
    - **Mechanism**: Maintain an index queue and a distance queue for the bottom-layer cluster centers, directly attaching new samples to their nearest bottom-layer cluster centers, and reconstruct the entire hierarchical graph structure every $T_0$ steps.
    - **Design Motivation**: Avoid the immense overhead of reconstructing the graph structure at every step, while preventing the graph structure from becoming obsolete due to lack of updates over a long period.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_{itc} + \mathcal{L}_{mlm} + \mathcal{L}_{itm}$
- In ITC loss, cosine similarity is replaced by geodesic distance.
- The cumulative angle of the geodesic path is truncated (truncation threshold $4\pi$) and normalized to $[0, \pi]$, and its cosine value is taken as the final similarity.
- Training setup: 8$\times$V100 GPUs, AdamW optimizer, 30 pre-training epochs, queue size of 65536.

## Key Experimental Results

### Main Results

**Zero-shot Image-Text Retrieval (MSCOCO / Flickr30K)**

| Method | Data Scale | COCO TR R@1 | COCO IR R@1 | Flickr TR R@1 | Flickr IR R@1 |
|------|--------|-------------|-------------|---------------|---------------|
| ALBEF | 4M | 68.7 | 50.1 | 90.5 | 76.8 |
| Geo-ALBEF | 4M | 72.0 | 53.6 | 93.2 | 79.9 |
| TCL | 4M | 71.4 | 53.5 | 93.0 | 79.6 |
| Geo-TCL | 4M | 73.9 | 54.6 | 94.0 | 80.6 |
| MAFA | 4M | 72.6 | 53.9 | 93.5 | 80.1 |
| Geo-MAFA | 4M | **74.7** | **55.4** | **94.6** | **81.1** |

**Fine-tuned Image-Text Retrieval (MSCOCO / Flickr30K)**

| Method | COCO TR R@1 | COCO IR R@1 | Flickr TR R@1 | Flickr IR R@1 |
|------|-------------|-------------|---------------|---------------|
| MAFA | 78.0 | 61.2 | 96.1 | 84.9 |
| Geo-MAFA | **79.3** | **62.5** | **96.9** | **85.6** |

**Downstream Vision-Language Tasks**

| Method | VQA test-dev | NLVR2 dev | SNLI-VE val |
|------|-------------|-----------|-------------|
| MAFA | 75.55 | 82.52 | 80.79 |
| Geo-MAFA | **76.04** | **83.12** | **81.42** |

### Ablation Study

| Ablation Item | COCO TR R@1 | COCO IR R@1 |
|--------|-------------|-------------|
| 1-layer Hierarchical Graph | 75.1 | 58.5 |
| 2-layer Hierarchical Graph | **76.2** | **59.2** |
| 3-layer Hierarchical Graph | 75.9 | 59.0 |

### Key Findings

1. As a plug-and-play module, geodesic distance consistently improves various baseline models such as ALBEF/TCL/MAFA, yielding an average improvement of 2-3 percentage points in zero-shot retrieval R@1.
2. The method can also be extended to other contrastive learning frameworks like CLIP and FLIP (Geo-CLIPFT zero-shot TR R@1: 59.6 vs 58.5) as well as self-supervised learning methods (Geo-MOCOv2, Geo-SwAV).
3. The additional computational overhead is small: CUDA memory increases by only about 2.5%, and training time increases by only about 5%.
4. The 2-layer hierarchical graph structure achieves the best results, with diminishing returns from additional layers.

## Highlights & Insights

1. **Novel Perspective**: For the first time, this work revisits the distance metric problem in multimodal contrastive learning from the perspective of differential geometry and geodesics, revealing the limitations of traditional cosine distance on complex manifolds.
2. **High Generalizability**: The geodesic distance module can be integrated in a plug-and-play manner into multiple contrastive learning frameworks (ALBEF, TCL, MAFA, CLIP, FLIP, MOCOv2, SwAV).
3. **Comprehensive Theoretical Analysis**: Theoretical analysis on the number and size of connected components in the hierarchical graph is provided, validating the rationale of the method.
4. **Engineering Feasibility**: Through hierarchical graph structures and incremental updates, the complexity of geodesic distance computation is kept within an acceptable range.

## Limitations & Future Work

1. The hierarchical graph structure requires periodic reconstruction (every 100 steps), which may not be timely during early training stages when features change rapidly.
2. The effectiveness of the geodesic distance depends on the size and quality of the sample pool, and can be less stable in small-batch scenarios.
3. It has only been validated on a 4M data scale; the effectiveness and efficiency at larger scales (such as the LAION-400M level) remain to be investigated.
4. The $O(N^3)$ complexity of Floyd's algorithm is still a bottleneck. Although $N$ is reduced via clustering (256 centers), more efficient shortest path approximation algorithms could be explored in the future.

## Related Work & Insights

- **ISOMAP**: Precedent in using geodesic distance in classical dimensionality reduction algorithms, which serves as the direct inspiration for this work.
- **ALBEF/TCL/MAFA**: Baseline methods for this work, all of which use momentum feature queues for contrastive learning.
- **GraphWalk**: Proposes a differentiable geodesic distance estimator but features lower accuracy.
- **Insight**: The choice of distance metric is crucial for contrastive learning. Future work can explore more geometry-aware distance metrics.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [\[CVPR 2025\] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](../../NeurIPS2025/multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig.md)

</div>

<!-- RELATED:END -->
