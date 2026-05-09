---
title: >-
  [Paper Note] CarGait: Cross-Attention based Re-ranking for Gait Recognition
description: >-
  [ICCV 2025][Human Understanding][gait recognition] This paper proposes CarGait, a cross-attention-based re-ranking method for gait recognition. Given the top-K retrieval results of any single-stage gait model, CarGait learns fine-grained pair-wise interactions between the probe and each candidate via cross-attention over gait strips, generates new conditioned representations, and recomputes distances for re-ranking. CarGait consistently improves Rank-1/5 accuracy across three datasets (Gait3D, GREW, OU-MVLP) and seven baseline models, with an inference speed of 6.5 ms/probe that substantially outperforms existing re-ranking methods.
tags:
  - ICCV 2025
  - Human Understanding
  - gait recognition
  - re-ranking
  - cross-attention
  - gait strip
  - fine-grained matching
  - metric learning
date: 2026-05-08
content_hash: a0a48b57fcae825a
---

# CarGait: Cross-Attention based Re-ranking for Gait Recognition

**Conference**: ICCV 2025
**arXiv**: [2503.03501](https://arxiv.org/abs/2503.03501)
**Code**: Unknown
**Area**: Computer Vision / Gait Recognition / Re-ranking
**Keywords**: gait recognition, re-ranking, cross-attention, gait strip, fine-grained matching, metric learning

## TL;DR

This paper proposes CarGait, a cross-attention-based re-ranking method for gait recognition. Given the top-K retrieval results of any single-stage gait model, CarGait learns fine-grained pair-wise interactions between the probe and each candidate via cross-attention over gait strips, generates new conditioned representations, and recomputes distances for re-ranking. CarGait consistently improves Rank-1/5 accuracy across three datasets (Gait3D, GREW, OU-MVLP) and seven baseline models, with an inference speed of 6.5 ms/probe that substantially outperforms existing re-ranking methods.

## Background & Motivation

Gait recognition identifies individuals through their walking patterns and finds applications in surveillance, forensics, and healthcare. Mainstream methods are single-stage: a gait sequence is encoded into a single global feature, and nearest-neighbor retrieval is performed over the gallery. The core limitations of this paradigm are:

- **Large Rank-1 to Rank-5 gap**: For example, GaitPart achieves only 28.2% Rank-1 but 47.6% Rank-5 on Gait3D, indicating that the correct identity is often within top-K but not at top-1.
- **Hard negatives**: Global features struggle to discriminate between individuals with highly similar gait patterns.
- **Limitations of single representations**: A single global vector lacks the capacity for fine-grained discrimination of local body motion patterns.

Re-ranking has been widely adopted in image retrieval and person Re-ID, but remains largely unexplored in gait recognition. Existing re-ranking methods (k-reciprocal, LBR, GCR) operate as post-processing on global feature similarity matrices and are not tailored to the spatiotemporal characteristics of gait.

## Core Problem

How to design a general re-ranking module that improves top-rank accuracy in gait recognition through fine-grained pair-wise interactions between the probe and candidates?

## Method

### Overall Architecture (Two-Stage)

1. **Initial Retrieval**: A pretrained single-stage gait model $M$ extracts global features from the probe and retrieves the top-K candidates from the gallery.
2. **CarGait Re-ranking**: Cross-attention interactions are performed between the feature maps of the probe and each top-K candidate, and new distances are computed for re-ranking.

### Gait Strip Representation

The pretrained model $M$ extracts a feature map $F \in \mathbb{R}^{s \times d}$ from a gait sequence $G$, where $s$ is the number of horizontal body strips and $d$ is the feature dimension. Strips are spatiotemporal aggregation units loosely associated with spatial body regions (head, torso, legs, etc.) and carry fine-grained gait information.

### Cross-Attention Module

Given probe features $F_p$ and candidate features $F_c$:

- **Forward cross-attention**: $F_p$ as Query, $F_c$ as Key/Value → produces $E_p \in \mathbb{R}^{s \times d}$
- **Backward cross-attention**: $F_c$ as Query, $F_p$ as Key/Value → produces $E_c \in \mathbb{R}^{s \times d}$
- **Residual connections**: Skip connections from $F_p \to E_p$ and $F_c \to E_c$ preserve the pretrained feature space information.
- Each strip in $E_p$ is modulated by attention relationships with all strips of the candidate, enabling cross-strip interaction.

Module design: single block, 8-head attention, hidden dimension 256.

### New Metric Space and Re-ranking

The re-ranking distance is $d^r_{p,c} = \mathcal{Z}(E_p, E_c)$, defined as the mean pairwise Euclidean distance between corresponding strips after cross-attention. At inference, the top-K candidates are re-sorted in ascending order of this new distance.

### Loss & Training

The model is jointly optimized with two losses:

$$\mathcal{L} = \mathcal{L}_{ranking} + \alpha \mathcal{L}_{CE}$$

- **Ranking loss**: Triplet-based, penalizing cases where a negative sample is closer to the probe than a positive. A damping parameter $\beta = 0.1$ down-weights triplets that are already correctly ordered.
- **Classification loss**: $E_p$ and $E_c$ are passed through an MLP classifier to produce identity logits; standard cross-entropy serves as regularization to maintain identity discriminability ($\alpha = 0.01$).

### Training Data Construction

The pretrained model $M$ is frozen. For each sample in the training set, the top-$v = 30$ nearest neighbors are retrieved to construct the training set $\mathcal{D}$, containing both positive (same identity) and negative (different identity) pairs. A validation set is constructed analogously; the best checkpoint is selected every 10k iterations based on validation ranking loss.

## Key Experimental Results

### Main Results (Rank-1 Improvement)

| Model | Gait3D | GREW | OU-MVLP |
|-------|--------|------|---------|
| GaitSet | 36.7→41.5 (+4.8) | 48.4→52.0 | 87.1→87.5 |
| GaitBase | 64.6→66.1 (+1.5) | 60.1→67.2 (+7.1) | 90.8→91.1 |
| SwinGait-3D | 75.0→76.3 | 79.3→* | — |
| SG++ | 77.6→78.1 | 85.8→88.2 (+2.4) | — |

### Comparison with Other Re-ranking Methods (Gait3D / SG++)

| Method | Rank-1 | Rank-5 | mAP |
|--------|--------|--------|-----|
| Initial | 77.6 | 89.4 | 70.30 |
| k-reciprocal | 69.7 | 85.6 | 70.30 |
| LBR | 61.7 | 90.2 | 58.99 |
| GCR | 76.1 | 89.6 | 68.72 |
| **CarGait** | **78.1** | **90.4** | **70.86** |

On OU-MVLP (gallery contains only one positive sample per identity), k-reciprocal and LBR actually degrade Rank-1 (e.g., GaitPart: 88.5→68.4/80.6), whereas CarGait still yields improvements (88.5→89.1).

### Inference Speed

| Method | Inference Time (ms/probe) |
|--------|--------------------------|
| k-reciprocal | 214 |
| GCR | 1866 |
| LBR | 19.81 |
| **CarGait** | **6.52** |

### Ablation Study (SwinGait-3D on Gait3D)

- Removing classification loss ($\alpha = 0$): R1 drops to 76.0 vs. 76.3, confirming the regularization benefit of the CE loss.
- Removing loss damping ($\beta = 1$): R1 drops to 75.5, validating the effectiveness of down-weighting already-correct triplets.
- Binary classification baseline (no cross-attention, simple concatenation + MLP): R1 drops to 71.6, directly demonstrating the central role of cross-attention.
- K=5 yields slightly higher R1 (76.5) but unchanged R5 (86.7); K=10 offers a better balance.

## Highlights & Insights

- **Strip-wise cross-attention is the key**: Unlike global feature post-processing, modeling probe–candidate interactions at the body-part level and learning cross-strip correlations (visualized in Figure 4) is the source of performance gains.
- **Dual objectives of metric learning and classification**: The ranking loss focuses on ordering correctness while the CE loss preserves identity discriminability; the two objectives are complementary.
- **Loss damping technique**: Using $\beta < 1$ to down-weight already-correct triplets directs optimization toward failure cases.
- **Strong plug-and-play generality**: Consistent improvements are achieved across 7 architectures (CNN/Transformer/multimodal) on 3 datasets.
- **Extremely fast inference**: At 6.5 ms/probe, CarGait is far faster than k-reciprocal (214 ms) and GCR (1866 ms), making it suitable for practical deployment.

## Limitations & Future Work

- A separate re-ranker must be trained for each pretrained model $M$, incurring additional offline training overhead.
- Improvement margins are limited on saturated datasets such as OU-MVLP (Rank-1 > 90%).
- Only gait recognition is validated; the cross-strip cross-attention paradigm may generalize to retrieval re-ranking in person Re-ID, vehicle Re-ID, and related tasks.
- The fixed settings of K=10 and v=30 are not tuned per model or dataset, leaving room for further optimization.
- Code is not publicly released.

## Rating

- **Novelty**: ⭐⭐⭐ Cross-attention combined with re-ranking is an effective integration of mature techniques; the strip-wise interaction design is noteworthy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 3 datasets × 7 models + 3 competing re-ranking methods + comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with informative visualizations (attention matrices in Figure 4, spider charts in Figure 5).
- **Value**: ⭐⭐ Specific to the gait recognition sub-field; however, the strip-wise cross-attention paradigm for pair-wise re-ranking is transferable to other retrieval tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)

</div>

<!-- RELATED:END -->
