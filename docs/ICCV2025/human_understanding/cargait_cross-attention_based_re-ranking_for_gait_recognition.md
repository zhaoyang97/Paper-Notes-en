---
title: >-
  [Paper Note] CarGait: Cross-Attention based Re-ranking for Gait Recognition
description: >-
  [ICCV 2025][Human Understanding][gait recognition] This paper proposes CarGait, a cross-attention-based re-ranking method for gait recognition. By performing strip-wise cross-attention between the probe and candidate sequences, CarGait learns fine-grained gait correspondences and maps global features from a frozen single-stage model into a new discriminative embedding space. Consistent Rank-1/5 accuracy improvements are achieved across seven gait models on three major benchmarks: Gait3D, GREW, and OU-MVLP.
tags:
  - ICCV 2025
  - Human Understanding
  - gait recognition
  - re-ranking
  - cross-attention
  - metric learning
  - fine-grained matching
date: 2026-05-08
content_hash: ca12ace775ecf057
---

# CarGait: Cross-Attention based Re-ranking for Gait Recognition

**Conference**: ICCV 2025
**arXiv**: [2503.03501](https://arxiv.org/abs/2503.03501)
**Code**: N/A
**Area**: Human Understanding / Gait Recognition
**Keywords**: gait recognition, re-ranking, cross-attention, metric learning, fine-grained matching

## TL;DR

This paper proposes CarGait, a cross-attention-based re-ranking method for gait recognition. By performing strip-wise cross-attention between the probe and candidate sequences, CarGait learns fine-grained gait correspondences and maps global features from a frozen single-stage model into a new discriminative embedding space. Consistent Rank-1/5 accuracy improvements are achieved across seven gait models on three major benchmarks: Gait3D, GREW, and OU-MVLP.

## Background & Motivation

Gait recognition is typically formulated as a retrieval task: given a probe sequence, candidates in the gallery are ranked by feature distance to identify the same subject. Performance is measured by Rank-K accuracy, with Rank-1 being the most critical metric for real-world applications such as surveillance.

Existing models are **single-stage** approaches that encode gait sequences into global feature vectors and directly perform nearest-neighbor ranking. While these methods achieve reasonable Rank-5 performance, their Rank-1 accuracy is often substantially lower — because the top-K list contains numerous hard negatives (samples with similar gait patterns but different identities) that global features lack the discriminative power to distinguish.

For instance, GaitPart achieves only 28.2% Rank-1 but 47.6% Rank-5 on Gait3D — a gap of nearly 20%. This indicates that **the correct identity is likely present in the top-5 but fails to be ranked first**, a gap that re-ranking is well-positioned to close.

Nevertheless, re-ranking remains largely unexplored in gait recognition. Although re-ranking methods exist in image retrieval and person re-identification (e.g., k-reciprocal encoding), they operate on the relative structure of global features without accounting for the spatiotemporal stripe characteristics of gait data.

## Method

### Overall Architecture

CarGait is a two-stage framework: (1) a pre-trained gait model performs global retrieval to produce a top-K list; (2) cross-attention is applied between the feature maps of the probe and each candidate in the top-K list to learn a new embedding space for re-ranking.

### Key Designs

1. **Strip-wise Multi-Head Cross-Attention**:

    - The pre-trained model outputs feature maps $F_p, F_c \in \mathbb{R}^{s \times d}$, where $s$ denotes the number of body strips and $d$ is the feature dimension.
    - Multi-head cross-attention is performed with probe $F_p$ as Query and candidate $F_c$ as Key/Value to produce $E_p$.
    - The operation is applied in reverse: $F_c$ as Query and $F_p$ as Key/Value to produce $E_c$.
    - A residual connection preserves information from the pre-trained model: $E_p = E_p + F_p$.
    - **Design Motivation**: Single-stage models only compute distances between corresponding strips (part-to-part), whereas cross-attention enables **arbitrary inter-strip interactions** — e.g., the head strip can attend to the leg strip of the counterpart — capturing global gait dynamics.

2. **New Metric Space and Distance Computation**:

    - The cross-attended representations $E_p, E_c$ constitute a new embedding space.
    - The new distance is defined as $d_{p,c}^r = \mathcal{Z}(E_p, E_c)$: the mean Euclidean distance across all strip features.
    - Re-ranking reorders the top-K list in ascending order of the new distance.
    - **Design Motivation**: Pairwise fine-grained comparison built upon global features enables better separation of hard negatives.

3. **Training Data Construction and Loss Function**:

    - Training set construction: for each probe in the training set, top-$v$ ($v=30$) candidates are retrieved using the pre-trained model, containing both positives (same identity) and negatives.
    - Ranking loss (modified BPR loss):
    $\mathcal{L}_i^* = -\log[\sigma(d_{p_i,neg_i}^r - d_{p_i,pos_i}^r)]$
      Correctly ordered triplets are down-weighted by $\beta=0.1$, focusing optimization on hard cases.
    - Classification loss: an MLP classifier is applied on $E_p, E_c$ with standard cross-entropy, serving as a regularizer.
    - Total loss: $\mathcal{L} = \mathcal{L}_{ranking} + \alpha \mathcal{L}_{CE}$, where $\alpha=0.01$.
    - **Design Motivation**: The ranking loss directly optimizes the ranking objective, while the classification loss preserves identity-discriminative information.

### Inference Strategy

- The pre-trained model performs global retrieval to produce a top-$K=10$ list.
- Cross-attention is applied between the probe and each candidate, and new distances are computed.
- The top-10 list is reordered by the new distances.
- Inference speed: approximately 6.5 ms/probe, significantly faster than traditional methods such as k-reciprocal encoding.

## Key Experimental Results

### Main Results (Seven Models × Three Datasets)

| Method | Gait3D R1 | Gait3D R5 | GREW R1 | GREW R5 | OU-MVLP R1 |
|------|-----------|-----------|---------|---------|------------|
| GaitPart (baseline) | 28.2 | 47.6 | 47.6 | 60.7 | 88.5 |
| GaitPart + **CarGait** | **29.5** | **48.5** | **52.5** | **67.5** | **89.1** |
| GaitBase (baseline) | 64.6 | 81.5 | 60.1 | 75.5 | 90.8 |
| GaitBase + **CarGait** | **66.1** | **82.8** | **67.2** | **78.5** | **91.1** |
| SG++ (baseline) | 77.6 | 89.4 | 85.8 | 92.6 | - |
| SG++ + **CarGait** | **78.1** | **90.4** | **88.2** | **94.6** | - |
| DGV2-P3D (baseline) | 74.4 | 88.0 | 77.7 | 87.9 | 91.9 |
| DGV2-P3D + **CarGait** | **75.1** | 87.5 | **79.2** | **88.7** | **92.0** |

### Comparison with Other Re-ranking Methods (Gait3D Dataset)

| Method | GaitPart R1 | GaitSet R1 | GaitBase R1 | GaitBase mAP |
|------|-------------|------------|-------------|--------------|
| KR (k-reciprocal) | 26.5 | 34.8 | 60.0 | 57.78 |
| LBR | 23.3 | 33.0 | 63.8 | 51.43 |
| GCR | 26.0 | 35.7 | 63.1 | 53.12 |
| **CarGait** | **29.5** | **41.5** | **66.1** | **57.66** |

### Key Findings

- CarGait consistently improves both Rank-1 and Rank-5 across all models and datasets, demonstrating its generalizability.
- Improvements are more pronounced on Gait3D and GREW (in-the-wild, high difficulty), while gains on OU-MVLP (controlled indoor setting, near-saturated performance) are modest.
- GaitBase on GREW achieves the largest gain: Rank-1 improves from 60.1% to 67.2% (+7.1%).
- Conventional re-ranking methods (KR, LBR) may degrade performance in galleries with sparse positives; CarGait does not exhibit this issue.
- Visualization of the learned cross-attention patterns reveals strengthened off-diagonal strip correlations, confirming that meaningful cross-part relationships are captured.
- At 6.5 ms/probe, CarGait is substantially faster than KR and other methods that require computing a full gallery similarity matrix.

## Highlights & Insights

- **Precise problem identification**: The large gap between Rank-1 and Rank-5 precisely quantifies the headroom available for re-ranking.
- **Plug-and-play design**: The pre-trained model is frozen; only a lightweight cross-attention module is trained, making CarGait compatible with any existing gait model.
- The bidirectional cross-attention design is critical — modifying both the probe and candidate representations yields a more accurate distance that better reflects the pairwise matching relationship.
- The $\beta=0.1$ down-weighting strategy is elegant: correctly ordered triplets require no further optimization, concentrating gradients on hard cases.

## Limitations & Future Work

- Re-ranking is limited to the top-10 candidates; positives ranked beyond position 10 cannot be recovered — larger $K$ values or cascaded re-ranking could be explored.
- A separate re-ranker must be trained for each pre-trained backbone, increasing deployment complexity.
- Validation is limited to appearance-based models; compatibility with model-based methods, which define strips differently, remains to be verified.
- Cross-dataset generalization is not evaluated (e.g., whether a re-ranker trained on Gait3D transfers to GREW).

## Related Work & Insights

- Comprehensive comparisons with re-ranking methods from person re-identification (KR, LBR, GCR) are provided, with CarGait outperforming all baselines in the gait recognition setting.
- Cross-attention is widely used in multimodal fusion (e.g., CoCa, Flamingo); this work innovatively applies it to pairwise fine-grained comparison within a single modality.
- The proposed strip-wise cross-attention re-ranking paradigm may generalize to other part-based retrieval tasks, such as vehicle re-identification and fine-grained image retrieval.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First deep learning-based re-ranking method for gait recognition; the use of cross-attention for pairwise fine-grained matching is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Seven models, three datasets, and comparisons against three re-ranking baselines constitute a highly comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and visualization analyses are informative.
- **Value**: ⭐⭐⭐⭐ — A practical plug-and-play solution with consistent Rank-1 improvements and clear industrial applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)

<!-- RELATED:END -->
