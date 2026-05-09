---
title: >-
  [Paper Note] CarGait: Cross-Attention based Re-ranking for Gait Recognition
description: >-
  [ICCV 2025][Human Understanding][Gait Recognition] This paper proposes CarGait, a cross-attention based re-ranking method for gait recognition. By performing strip-wise cross-attention between probe and candidate sequences, CarGait learns fine-grained gait correspondences and maps global features from pretrained single-stage models into a new discriminative embedding space. The method consistently improves Rank-1/5 accuracy across seven gait models on three major benchmarks: Gait3D, GREW, and OU-MVLP.
tags:
  - ICCV 2025
  - Human Understanding
  - Gait Recognition
  - Re-ranking
  - Cross-Attention
  - Metric Learning
  - Fine-grained Matching
date: 2026-05-08
content_hash: 0c72fbcb1901c7b3
---

# CarGait: Cross-Attention based Re-ranking for Gait Recognition

**Conference**: ICCV 2025
**arXiv**: [2503.03501](https://arxiv.org/abs/2503.03501)
**Code**: N/A
**Area**: Human Understanding / Gait Recognition
**Keywords**: Gait Recognition, Re-ranking, Cross-Attention, Metric Learning, Fine-grained Matching

## TL;DR

This paper proposes CarGait, a cross-attention based re-ranking method for gait recognition. By performing strip-wise cross-attention between probe and candidate sequences, CarGait learns fine-grained gait correspondences and maps global features from pretrained single-stage models into a new discriminative embedding space. The method consistently improves Rank-1/5 accuracy across seven gait models on three major benchmarks: Gait3D, GREW, and OU-MVLP.

## Background & Motivation

Gait recognition is typically formulated as a retrieval task: given a probe sequence, the system ranks gallery entries by feature distance to identify the same subject. Performance is measured by Rank-K accuracy, with Rank-1 being the most critical for practical applications such as surveillance.

Existing models follow a **single-stage** paradigm: encoding gait sequences into global feature vectors and directly performing nearest-neighbor retrieval. While these methods achieve reasonable Rank-5 performance, their Rank-1 accuracy tends to be substantially lower — primarily because the top-K results contain many hard negatives (samples with similar gait patterns but different identities) that global features fail to distinguish.

For example, GaitPart achieves only 28.2% Rank-1 yet 47.6% Rank-5 on Gait3D, a gap of nearly 20 percentage points. This indicates that **the correct identity is likely present within the top-5 results, but fails to be ranked first** — a gap that re-ranking is well-positioned to close.

However, re-ranking remains largely unexplored in gait recognition. While methods such as k-reciprocal encoding exist in image retrieval and person re-identification, they operate on the relative structure of global features without accounting for the spatiotemporal stripe characteristics inherent to gait data.

## Method

### Overall Architecture

CarGait is a two-stage approach: (1) a pretrained gait model produces a global ranking to obtain the top-K list; (2) cross-attention is applied between the feature maps of the probe and each candidate in the top-K list to learn a new embedding space for re-ranking.

### Key Designs

1. **Strip-wise Multi-head Cross-Attention**:
   - The pretrained model outputs feature maps $F_p, F_c \in \mathbb{R}^{s \times d}$, where $s$ is the number of body strips and $d$ is the feature dimension.
   - Multi-head cross-attention is applied with probe $F_p$ as Query and candidate $F_c$ as Key/Value, yielding $E_p$.
   - The operation is performed in reverse: $F_c$ as Query and $F_p$ as Key/Value, yielding $E_c$.
   - A residual connection preserves information from the pretrained model: $E_p = E_p + F_p$.
   - **Design Motivation**: Single-stage models only compute distances between corresponding strips (part-to-part alignment), whereas cross-attention enables **arbitrary inter-strip interactions** — the head strip can attend to the leg strip of the counterpart, capturing global gait dynamics.

2. **New Metric Space and Distance Computation**:
   - The post-attention representations $E_p, E_c$ define a new embedding space.
   - The new distance $d_{p,c}^r = \mathcal{Z}(E_p, E_c)$ is computed as the mean Euclidean distance across all strip features.
   - Re-ranking reorders the top-K list in ascending order of this new distance.
   - **Design Motivation**: Pairwise fine-grained comparison built upon global features provides stronger discriminability for hard negatives.

3. **Training Data Construction and Loss Functions**:
   - Training set construction: for each probe in the training set, the top-$v$ ($v=30$) candidates are retrieved using the pretrained model, containing both positive (same identity) and negative samples.
   - Ranking loss (modified BPR loss):
     $$\mathcal{L}_i^* = -\log[\sigma(d_{p_i,neg_i}^r - d_{p_i,pos_i}^r)]$$
     Triplets that are already correctly ranked are down-weighted by $\beta=0.1$, focusing training on hard cases.
   - Classification loss: an MLP classifier is applied on $E_p$ and $E_c$ with standard cross-entropy, serving as a regularizer.
   - Total loss: $\mathcal{L} = \mathcal{L}_{ranking} + \alpha \mathcal{L}_{CE}$, with $\alpha=0.01$.
   - **Design Motivation**: The ranking loss directly optimizes the ranking objective, while the classification loss preserves identity-discriminative information.

### Inference Strategy

- The pretrained model retrieves an initial top-$K=10$ list via global ranking.
- Cross-attention is applied between the probe and each candidate to compute the new distance.
- The top-10 list is reordered according to the new distance.
- Inference speed: approximately 6.5 ms per probe, substantially faster than traditional methods such as k-reciprocal encoding.

## Key Experimental Results

### Main Results (Seven Models × Three Datasets)

| Method | Gait3D R1 | Gait3D R5 | GREW R1 | GREW R5 | OU-MVLP R1 |
|--------|-----------|-----------|---------|---------|------------|
| GaitPart (baseline) | 28.2 | 47.6 | 47.6 | 60.7 | 88.5 |
| GaitPart + **CarGait** | **29.5** | **48.5** | **52.5** | **67.5** | **89.1** |
| GaitBase (baseline) | 64.6 | 81.5 | 60.1 | 75.5 | 90.8 |
| GaitBase + **CarGait** | **66.1** | **82.8** | **67.2** | **78.5** | **91.1** |
| SG++ (baseline) | 77.6 | 89.4 | 85.8 | 92.6 | - |
| SG++ + **CarGait** | **78.1** | **90.4** | **88.2** | **94.6** | - |
| DGV2-P3D (baseline) | 74.4 | 88.0 | 77.7 | 87.9 | 91.9 |
| DGV2-P3D + **CarGait** | **75.1** | 87.5 | **79.2** | **88.7** | **92.0** |

### Comparison with Other Re-ranking Methods (Gait3D)

| Method | GaitPart R1 | GaitSet R1 | GaitBase R1 | GaitBase mAP |
|--------|-------------|------------|-------------|--------------|
| KR (k-reciprocal) | 26.5 | 34.8 | 60.0 | 57.78 |
| LBR | 23.3 | 33.0 | 63.8 | 51.43 |
| GCR | 26.0 | 35.7 | 63.1 | 53.12 |
| **CarGait** | **29.5** | **41.5** | **66.1** | **57.66** |

### Key Findings

- CarGait consistently improves both Rank-1 and Rank-5 accuracy across all models and datasets, demonstrating strong generalizability.
- Improvements are more pronounced on the challenging in-the-wild datasets Gait3D and GREW, while gains on the controlled indoor dataset OU-MVLP are modest, likely due to performance saturation.
- GaitBase on GREW achieves the largest absolute gain, with Rank-1 increasing from 60.1% to 67.2% (+7.1%).
- Traditional re-ranking methods (KR, LBR) can degrade performance in galleries with sparse positive samples, a problem that CarGait does not exhibit.
- Visualization of cross-attention maps reveals strengthened off-diagonal strip correlations, confirming that the model learns meaningful cross-part relationships.
- At 6.5 ms per probe, CarGait is substantially faster than methods such as KR that require computing a full gallery similarity matrix.

## Highlights & Insights

- **Precise problem identification**: The large gap between Rank-1 and Rank-5 precisely characterizes the potential value of re-ranking, providing strong motivation for the proposed approach.
- **Plug-and-play design**: The pretrained model is frozen; only the lightweight cross-attention module is trained, making CarGait compatible with any existing gait model.
- The bidirectional cross-attention design is critical — modifying both the probe and candidate representations ensures that the recomputed distance more accurately reflects the matching relationship.
- The $\beta=0.1$ down-weighting strategy is well-motivated: triplets already correctly ranked require less optimization pressure, allowing gradients to concentrate on hard cases.

## Limitations & Future Work

- Re-ranking is restricted to the top-10 results; subjects ranked beyond position 10 in the initial list cannot be recovered. Larger values of $K$ or cascaded re-ranking could address this limitation.
- A separate re-ranker must be trained for each pretrained model, increasing deployment complexity.
- Validation is limited to appearance-based models; the applicability to model-based methods, which define stripes differently, remains to be verified.
- Cross-dataset generalization (e.g., whether a re-ranker trained on Gait3D transfers to GREW) is not evaluated.

## Related Work & Insights

- The paper provides thorough comparisons against re-ranking methods from person re-identification, including KR, LBR, and GCR, and CarGait consistently outperforms all baselines in the gait recognition setting.
- Cross-attention has been widely adopted in multimodal fusion (e.g., CoCa, Flamingo); this work innovatively applies it to pairwise fine-grained comparison within the same modality.
- Insight: strip-wise cross-attention re-ranking may also benefit other part-based retrieval tasks, such as vehicle re-identification and fine-grained image retrieval.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first deep learning-based re-ranking method for gait recognition; using cross-attention for pairwise fine-grained matching is a well-motivated and elegant design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Seven models, three major datasets, and comparisons against three re-ranking baselines constitute an exceptionally comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — The problem formulation is clear, and the qualitative visualizations are informative.
- **Value**: ⭐⭐⭐⭐ — A practical plug-and-play solution with consistent Rank-1 improvements that carry clear industrial relevance.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)

<!-- RELATED:END -->
