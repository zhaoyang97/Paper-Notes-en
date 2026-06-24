---
title: >-
  [Paper Note] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer
description: >-
  [AAAI 2026][Face Clustering] This paper proposes a prediction-driven Top-K Jaccard similarity coefficient to improve neighbor purity, combined with a Sparse Differential Transformer (SDT) to eliminate noisy attention, achieving state-of-the-art performance on large-scale face clustering datasets such as MS-Celeb-1M.
tags:
  - "AAAI 2026"
  - "Face Clustering"
  - "Jaccard Similarity"
  - "Sparse Differential Transformer"
  - "Noise Edges"
  - "Adaptive Neighbor Discovery"
date: 2026-05-08
content_hash: 7a2ae39fe8a18dc9
---

# Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer

**Conference**: AAAI 2026
**arXiv**: [2512.22612](https://arxiv.org/abs/2512.22612)  
**Code**: None  
**Area**: Other
**Keywords**: Face Clustering, Jaccard Similarity, Sparse Differential Transformer, Noise Edges, Adaptive Neighbor Discovery

## TL;DR
This paper proposes a prediction-driven Top-K Jaccard similarity coefficient to improve neighbor purity, combined with a Sparse Differential Transformer (SDT) to eliminate noisy attention, achieving state-of-the-art performance on large-scale face clustering datasets such as MS-Celeb-1M.

## Background & Motivation

**Background**: GCN-based face clustering methods learn features via graph message passing, but constructing face graphs using kNN cosine distance introduces numerous noisy edges (connecting nodes of different identities), causing feature contamination during message propagation. Ada-NETS and FC-ESER replace cosine distance with Jaccard similarity coefficients, but the inclusion of excessive irrelevant nodes leads to insufficient discriminability of Jaccard scores.

**Limitations of Prior Work**:
   - The Jaccard coefficients computed by FC-ESER between different faces are too similar — a slightly lower threshold merges different identities, while a slightly higher threshold fragments the same identity.
   - Ada-NETS produces inaccurate predictions of the optimal neighbor count $k$, deviating from the optimal value.
   - Vanilla Transformer assigns attention to all feature relationships (including irrelevant and noisy ones) during relation prediction, leading to erroneous clustering.

**Key Challenge**: How to precisely determine the effective neighbor range for each node while reliably judging node relationships near the Top-K boundary?

**Goal**: Improve the reliability of Jaccard similarity computation + handle uncertainty near the Top-K boundary + eliminate noisy attention in the Transformer.

**Key Insight**: (1) Use a Transformer to predict the optimal neighbor count Top-K for each node, computing Jaccard similarity only within Top-K neighbors; (2) use SDT to handle uncertainty near the Top-K boundary.

**Core Idea**: Prediction-driven Top-K Jaccard for neighbor purification + sparse differential attention to eliminate noisy relation judgment.

## Method

### Overall Architecture
Construct face graph → Transformer predicts Top-K neighbor boundary → Refine face graph using Top-K → SDT judges node relationships near Top-K → Map Equation clustering.

### Key Designs

1. **Prediction-Driven Top-K Jaccard Similarity**:

    - **Function**: Dynamically predicts the optimal neighbor count for each node to improve Jaccard computation quality.
    - **Mechanism**: Replaces Ada-NETS's LSTM with a Transformer to predict Top-K; Jaccard similarity is computed using only neighbors within Top-K.
    - **Distance Transform Improvement**: $p_{ij} = \frac{1}{1 + e^{\delta d_{ij} + \epsilon}}$ (sigmoid form, $\delta=7.5, \epsilon=-5$), amplifying differences among small distances.
    - **Design Motivation**: The exponential distance transform in FC-ESER compresses similarity differences, causing Jaccard coefficients between different identities to be overly close.

2. **Sparse Differential Transformer (SDT)**:

    - **Function**: Handles uncertain relationships near the Top-K boundary.
    - **Mechanism**: Applies differential attention from the Differential Transformer for noise cancellation + Top-K sparse masking to suppress irrelevant nodes.
    - **Differential Attention**: Computes the difference between two independent softmax attention maps to eliminate noisy attention.
    - **Sparse Mask**: Focuses only on relevant nodes within Top-K and suppresses irrelevant nodes beyond Top-K.
    - An MoE-SDT variant is also proposed to further enhance model capacity.
    - **Design Motivation**: Vanilla Transformer assigns attention to all feature relationships, including irrelevant or noisy ones, leading to misclassification.

### Loss & Training
Binary cross-entropy loss. The Transformer is first trained to predict Top-K; SDT then refines pairwise relationships; finally, Map Equation is applied for clustering.

## Key Experimental Results

### Main Results (MS-Celeb-1M, 5 Scales)

| Method | 584K $F_P$/$F_B$ | 5.21M $F_P$/$F_B$ |
|--------|---------|---------|
| K-Means | 79.21/81.23 | 66.47/69.42 |
| GCN(V+E) | 87.93/86.09 | 79.30/79.25 |
| Ada-NETS | ~89/~87 | ~81/~80 |
| **Ours** | **SOTA** | **SOTA** |

### Ablation Study
- Top-K Jaccard vs. standard Jaccard: Top-K yields significantly improved clustering accuracy.
- SDT vs. Vanilla Transformer: SDT outperforms across all scales.
- Distance transform (sigmoid vs. exponential): sigmoid better distinguishes similar/dissimilar samples.
- MoE-SDT further improves performance at the cost of additional computation.

### Key Findings
- The accuracy of Top-K prediction directly affects clustering quality — a Top-K that is too large introduces noise, while one that is too small loses information.
- The sparse mask in SDT leverages Top-K prior information, making it more effective than generic denoising.
- The advantage is more pronounced at the largest scale (5.21M images), as the noise problem intensifies with scale.

## Highlights & Insights
- The **two-stage design of "predicting neighbor count + refining relationships"** is practically effective — it addresses problems at different granularities in a layered manner.
- The **combination of differential attention and sparse masking** cleverly exploits prior information specific to the clustering task.
- The **sigmoid distance transform** is simple yet effective, amplifying discriminability.

## Limitations & Future Work
- The accuracy of Top-K prediction itself remains limited.
- SDT increases model complexity.
- Validation is limited to face clustering; generalization to generic graph clustering remains to be tested.
- The computational overhead of MoE-SDT warrants consideration.

## Related Work & Insights
- **vs. Ada-NETS**: Ada-NETS's $k_{off}$ prediction is inaccurate; this paper replaces LSTM with a Transformer for more reliable prediction.
- **vs. FC-ESER**: FC-ESER's Jaccard discriminability is insufficient; the proposed Top-K + sigmoid distance transform significantly improves this.
- **vs. Differential Transformer**: DiffTransformer was originally proposed for NLP; this paper extends it to graph clustering with the addition of a sparse mask.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of prediction-driven Top-K and SDT denoising is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments across 5 scales with comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ A practical solution for large-scale face clustering.

## Supplementary Analysis
- The proposed method represents a meaningful technical advancement within its specific sub-domain.
- The core innovation lies in encoding domain-specific structural prior knowledge into the model design, rather than relying entirely on data-driven end-to-end learning.
- Compared with other top-venue contemporaneous works, this paper demonstrates a high level of research sophistication in problem formulation and the systematic design of its methodology.
- For practical deployment, additional engineering considerations — including computational efficiency, real-time requirements, data privacy, and system scalability — must be addressed.
- The core ideas of the method have transferable potential; similar design paradigms may prove effective on related but distinct tasks and data modalities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](../../NeurIPS2025/others/coresets_for_clustering_under_stochastic_noise.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States](enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)
- [\[AAAI 2026\] Predict and Resist: Long-Term Accident Anticipation under Sensor Noise](predict_and_resist_long-term_accident_anticipation_under_sensor_noise.md)

</div>

<!-- RELATED:END -->
