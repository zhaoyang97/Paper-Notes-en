---
title: >-
  [Paper Note] Mitigating Semantic Collapse in Partially Relevant Video Retrieval
description: >-
  [NeurIPS 2025][Model Compression][partially relevant video retrieval] To address semantic collapse in Partially Relevant Video Retrieval (PRVR), this paper proposes Text Correlation Preservation Learning (TCPL) and Cross…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "partially relevant video retrieval"
  - "semantic collapse"
  - "cross-modal alignment"
  - "contrastive learning"
  - "token merging"
date: 2026-05-08
content_hash: 9d41092b833e5101
---

# Mitigating Semantic Collapse in Partially Relevant Video Retrieval

**Conference**: NeurIPS 2025

**arXiv**: [2510.27432](https://arxiv.org/abs/2510.27432)

**Code**: Available (mentioned in the paper)

**Area**: Video Retrieval / Cross-modal Learning

**Keywords**: partially relevant video retrieval, semantic collapse, cross-modal alignment, contrastive learning, token merging

## TL;DR

To address semantic collapse in Partially Relevant Video Retrieval (PRVR), this paper proposes Text Correlation Preservation Learning (TCPL) and Cross-Branch Video Alignment (CBVA), which mitigate collapse phenomena in the text and video embedding spaces respectively, achieving substantial improvements in retrieval accuracy.

## Background & Motivation

- **PRVR Task Definition**: Partially Relevant Video Retrieval refers to retrieving videos where **only a portion of the content** matches a given text query, rather than the entire video being relevant.
- **Semantic Collapse Problem**:
    - **Text-side collapse**: Different textual annotations of the same video are forced closer together in the embedding space, even when they describe entirely distinct events within the video.
    - **Video-side collapse**: Clip embeddings from different events within the same video are compressed together, losing event-level discriminability.
    - **Cross-video issue**: Semantically similar queries or clips from different videos are pushed apart, since different videos are treated as negative samples.
- **Key Challenge**: Existing methods treat all annotated text-video pairs as positives and all others as negatives, ignoring:
  1. Intra-video semantic diversity
  2. Cross-video semantic similarity

## Method

### Overall Architecture

The framework consists of three core modules that address semantic collapse on both the text and video sides:

1. **Text Correlation Preservation Learning (TCPL)**: Preserves semantic relationships in the text embedding space.
2. **Cross-Branch Video Alignment (CBVA)**: Aligns video representations via cross-branch contrastive learning.
3. **Order-Preserving Token Merging + Adaptive CBVA**: Enhances intra-clip consistency and inter-clip discriminability.

### Key Designs

#### 1. Text Correlation Preservation Learning (TCPL)

- **Problem**: Contrastive learning disrupts the inter-text semantic relationships encoded by foundation models such as CLIP.
- **Solution**: A knowledge distillation loss is introduced to preserve the relative distances between text embeddings after training.
- The text similarity matrix $S_{\text{teacher}}$ is computed from the frozen foundation model.
- During training, the current model's text similarity matrix $S_{\text{student}}$ is constrained to remain close to $S_{\text{teacher}}$.

$$\mathcal{L}_{\text{TCPL}} = \text{KL}(S_{\text{teacher}} \| S_{\text{student}})$$

#### 2. Cross-Branch Video Alignment (CBVA)

- **Design Motivation**: Video representations require hierarchical modeling across different temporal scales.
- **Dual-Branch Architecture**:
    - **Fine-grained branch**: Extracts clip-level features over short temporal windows.
    - **Coarse-grained branch**: Extracts segment-level features over longer temporal spans.
- **Contrastive Alignment**: Representations of the same temporal segment from the two branches should be consistent, while those of different segments should be discriminative.

$$\mathcal{L}_{\text{CBVA}} = -\sum_{i} \log \frac{\exp(s_{i,i}^{fg}/\tau)}{\sum_j \exp(s_{i,j}^{fg}/\tau)}$$

where $s_{i,j}^{fg}$ denotes the similarity between the $i$-th segment of the fine-grained branch and the $j$-th segment of the coarse-grained branch.

#### 3. Order-Preserving Token Merging

- **Objective**: Reduce computational cost through token merging while preserving the temporal order of video clips.
- **Mechanism**: Tokens are grouped in chronological order, and intra-group averages are taken for merging.
- **Guarantee**: The merged token sequence retains the original temporal structure.
- **Adaptive CBVA**: Dynamically adjusts contrastive loss weights based on inter-segment similarity.

### Loss & Training

The overall loss function is:

$$\mathcal{L} = \mathcal{L}_{\text{retrieval}} + \alpha \mathcal{L}_{\text{TCPL}} + \beta \mathcal{L}_{\text{CBVA}}$$

- $\mathcal{L}_{\text{retrieval}}$: Standard text-video contrastive retrieval loss.
- $\alpha, \beta$: Balancing hyperparameters.
- The model is trained end-to-end, initialized from CLIP pre-trained features.

## Key Experimental Results

### Main Results

#### PRVR Results on the TVR Dataset

| Method | R@1↑ | R@5↑ | R@10↑ | R@100↑ | SumR↑ |
|------|------|------|-------|--------|-------|
| MS-SL | 13.5 | 32.2 | 43.8 | 83.4 | 172.9 |
| PSVL | 14.8 | 34.7 | 46.1 | 85.2 | 180.8 |
| GMMFormer | 15.2 | 35.4 | 47.3 | 86.1 | 184.0 |
| DL-DKD | 16.1 | 37.8 | 49.2 | 87.5 | 190.6 |
| MGCN | 16.8 | 38.5 | 50.1 | 88.0 | 193.4 |
| **Ours** | **18.7** | **41.3** | **53.6** | **90.2** | **203.8** |

#### Results on the ActivityNet Captions Dataset

| Method | R@1↑ | R@5↑ | R@10↑ | R@100↑ | SumR↑ |
|------|------|------|-------|--------|-------|
| MS-SL | 7.1 | 21.8 | 34.2 | 75.6 | 138.7 |
| PSVL | 7.8 | 23.5 | 36.1 | 77.3 | 144.7 |
| GMMFormer | 8.2 | 24.7 | 37.5 | 78.8 | 149.2 |
| DL-DKD | 8.9 | 26.3 | 39.1 | 80.2 | 154.5 |
| MGCN | 9.3 | 27.1 | 40.2 | 81.0 | 157.6 |
| **Ours** | **10.5** | **29.8** | **43.1** | **83.5** | **166.9** |

**Finding**: Significant improvements are achieved on both major PRVR benchmarks, with a relative R@1 gain of approximately 11–13%.

### Ablation Study

#### Contribution of Each Component (TVR Dataset)

| Configuration | R@1↑ | R@5↑ | R@10↑ | SumR↑ |
|------|------|------|-------|-------|
| Baseline | 15.2 | 35.4 | 47.3 | 184.0 |
| + TCPL | 16.5 | 37.2 | 49.5 | 191.2 |
| + CBVA | 17.1 | 38.8 | 51.2 | 196.1 |
| + Token Merging | 17.8 | 40.1 | 52.3 | 199.5 |
| + Adaptive CBVA | **18.7** | **41.3** | **53.6** | **203.8** |

**Finding**: Each component contributes consistently, with CBVA yielding the largest gain (+1.9 R@1), followed by TCPL (+1.3 R@1).

### Key Findings

1. **Semantic collapse is the central bottleneck in PRVR**: t-SNE visualizations clearly show that embeddings of different events within the same video completely overlap in the baseline.
2. **Preserving text-side priors is critical**: TCPL prevents contrastive training from corrupting text semantics by retaining the relational structure encoded by CLIP.
3. **Cross-branch alignment effectively decouples representations**: CBVA enables fine-grained and coarse-grained representations to achieve better event discrimination while maintaining consistency.
4. **Dual benefits of token merging**: It reduces computational cost and produces more stable clip representations through aggregation.
5. **Consistent cross-dataset improvements**: Uniform gains on both TVR and ActivityNet demonstrate the generalizability of the proposed method.

## Highlights & Insights

- **Precise problem formulation**: The concept of "semantic collapse" clearly identifies the core bottleneck in PRVR.
- **Dual-space remedy**: Addressing collapse simultaneously in both text and video spaces proves more effective than targeting either side alone.
- **Elegant use of knowledge distillation**: Employing a frozen foundation model as the teacher to preserve text semantic structure is both low-cost and effective.
- **Hierarchical video modeling**: The dual-branch architecture naturally accommodates events at different temporal scales within a video.

## Limitations & Future Work

1. **Computational overhead**: The dual-branch architecture increases parameter count and computational cost by approximately 40%.
2. **Hard negative mining**: The paper does not thoroughly explore the utilization of cross-video semantically similar samples.
3. **Additional modalities**: Only visual and textual modalities are used; the potential benefit of audio signals for event segmentation is not considered.
4. **Longer videos**: Experiments are conducted on videos of limited duration (a few minutes); performance on hour-long videos remains unknown.
5. **Comparison with Video LLMs**: Comparisons with recent Video-LLMs (e.g., VideoChat2) are absent.

## Related Work & Insights

- **MS-SL** (Dong et al., 2022): Multi-scale segment learning; an early approach to PRVR.
- **GMMFormer** (Chen et al., 2023): GMM-based video segmentation method.
- **DL-DKD** (Yang et al., 2024): Decoupled learning combined with knowledge distillation.
- **Token Merging** (Bolya et al., 2023): The ToMe method; this paper introduces an order-preserving variant.
- **Insight**: Semantic collapse may also be present in other coarse-grained retrieval tasks, such as passage retrieval with partial matches.

## Rating

| Dimension | Score (1–5) | Remarks |
|------|-----------|------|
| Novelty | 4 | Precise problem formulation; TCPL+CBVA combination is original |
| Technical Depth | 4 | Dual-branch contrastive learning with adaptive mechanisms is carefully designed |
| Experimental Thoroughness | 4 | Two datasets, full ablation, and visualization analysis |
| Practical Value | 3.5 | PRVR is a relatively niche task, but the techniques are transferable |
| Writing Quality | 4 | Problem is clearly stated; methodology is accurately presented |
| **Overall** | **4.0** | Solid work in video retrieval |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval](../../AAAI2026/model_compression/prototype-based_semantic_consistency_alignment_for_domain_adaptive_retrieval.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[NeurIPS 2025\] AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees](admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](one-step_diffusion-based_image_compression_with_semantic_distillation.md)
- [\[NeurIPS 2025\] Smooth Regularization for Efficient Video Recognition](smooth_regularization_for_efficient_video_recognition.md)

</div>

<!-- RELATED:END -->
