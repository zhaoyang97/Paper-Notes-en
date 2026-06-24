---
title: >-
  [Paper Note] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment
description: >-
  [ECCV 2024][Segmentation][Temporal action segmentation] This work systematically addresses the long-tail problem in temporal action segmentation for the first time, proposing the Group-wise Temporal Logit Adjustment (G-TLA) framework. By leveraging activity labels for group-wise classification combined with temporal action priors for logit adjustment, it substantially improves the performance of tail classes without sacrificing head classes.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Temporal action segmentation"
  - "long-tail distribution"
  - "logit adjustment"
  - "procedural video"
  - "over-segmentation"
date: 2026-05-08
content_hash: e6006df56ef1c45d
---

# Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment

**Conference**: ECCV 2024  
**arXiv**: [2408.09919](https://arxiv.org/abs/2408.09919)  
**Code**: [pangzhan27/GTLA](https://github.com/pangzhan27/GTLA)  
**Area**: Temporal Action Segmentation  
**Keywords**: Temporal action segmentation, long-tail distribution, logit adjustment, procedural video, over-segmentation

## TL;DR

This work systematically addresses the long-tail problem in temporal action segmentation for the first time, proposing the Group-wise Temporal Logit Adjustment (G-TLA) framework. By leveraging activity labels for group-wise classification combined with temporal action priors for logit adjustment, it substantially improves the performance of tail classes without sacrificing head classes.

## Background & Motivation

Temporal action segmentation (TAS) classifies video frames of procedural activities into different action categories. This task suffers from a severe long-tail distribution problem originating from dual sources:

**Segment-level imbalance**: Certain actions are optional (e.g., adding sugar is optional when making tea), leading to vast discrepancies in occurrence frequency (the imbalance ratio of the Breakfast dataset is up to 639:1).

**Frame-level imbalance**: Different actions vary significantly in duration; for instance, "pouring water" occupies far more frames than "adding a teabag."

Existing state-of-the-art (SOTA) methods (e.g., ASFormer, DiffAct) completely ignore the long-tail problem, resulting in zero accuracy on several tail classes. Directly applying long-tail methods from image classification to temporal action segmentation faces unique challenges:
- There are **temporal dependencies** between actions, violating the class-independence assumption of traditional methods.
- Simple Logit Adjustment (LA) introduces activity-irrelevant false positives (e.g., predicting "stirring coffee" while making tea) and chronologically inconsistent false positives (e.g., "adding a teabag" appearing after "stirring tea").
- Frame-level and segment-level metrics must be balanced simultaneously, whereas simple methods often compromise one for the other, leading to over-segmentation.

## Method

### Overall Architecture

The G-TLA framework consists of two core components, both applied to the classification layer of a base segmentation model (e.g., MSTCN, ASFormer):
1. **Group-wise Classification**: Groups sequences according to activity labels, using an independent classifier for each group.
2. **Temporal Logit Adjustment**: Restricts the temporal scope of logit adjustment within each group using action sequential priors.

### Key Designs

1. **Group-wise Classification Strategy**:

    - Video sequences are partitioned into mutually exclusive groups $\mathbf{G}$ based on activity labels, e.g., "making tea" as $G_1$ and "making coffee" as $G_2$.
    - An auxiliary class "others" is introduced for each group to represent actions not belonging to that group.
    - Shared actions (e.g., "adding sugar" appearing in both making tea and making coffee) are treated as distinct classes in different groups.
    - The final feature layer $z_t$ is simultaneously fed into $n$ group classifiers: $s_{c,t}^{(i)}(X) = \sum_j z_t[j] \cdot W_{j,c}^{(i)} + b^{(i)}$.
    - The loss function is split into two terms: action classification for the target group and "others" classification for non-target groups.
    - Design Motivation: To eliminate interference from activity-incompatible categories (classes with $p(c|a)=0$ will not be falsely promoted by LA), reducing confusion among semantically similar actions.

2. **Temporal Logit Adjustment**:

    - Standard LA applies a uniform adjustment to all frames: $s_{c,t}^{(k)}(X) + \tau \log p(c|G_k)$.
    - G-TLA introduces a temporal factor $\mathcal{T}_{c,t}^{(k)}(X)$: $s_{c,t}^{(k)}(X) + \tau \mathcal{T}_{c,t}^{(k)}(X) \log p(c|G_k)$.
    - **Temporal Boundary**: For each action $c$, its predecessor set $S_{bf}[c]$ and successor set $S_{af}[c]$ are computed to determine the permissible temporal window $[t_1(c,X), t_2(c,X)]$.
    - Normal adjustment is applied inside the window ($\mathcal{T}=1$), whereas consistent adjustment is maintained outside the window to prevent violating temporal priors.
    - The adjustment factor outside the window is: $\mathcal{T}_{c,t}^{(k)} = \frac{\log p(y_t|G_k)}{\log p(c|G_k)}$, which ensures that the decision boundary between the ground-truth label and candidate classes is not compromised.
    - Design Motivation: To prevent LA from introducing false positives at chronologically illogical positions (e.g., stirring tea should occur after pouring water, not before).

3. **Inference Strategy**:

    - When activity labels are unknown, the group with the lowest predicted probability for "others" is selected as the predicted activity: $\hat{k} = \arg\min_i \frac{1}{T}\sum_t \hat{p}(o_t^{(i)})$.
    - Temporal logit segment adjustment is not utilized during inference; prediction probabilities are directly processed via argmax.
    - In the absence of activity labels, KL-divergence-based clustering can be employed as an alternative.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_{GTLA} + \lambda \mathcal{L}_{sm}$
- Group-wise loss: $\mathcal{L}_{GTLA} = \alpha_k \frac{1}{T}\sum_t -\log \tilde{p}(y_t^{(k)}) + \eta \sum_{i \neq k}^n \frac{1}{T}\sum_t -\log \hat{p}(o_t^{(i)})$
- $\eta$ controls the balance between target and non-target group losses, while $\alpha_k$ balances the sample counts across groups.
- $\tau$ controls the trade-off between balanced error and skewed error.
- The smoothing loss $\mathcal{L}_{sm}$ encourages smooth transitions between frame predictions, with a threshold of $\delta=4$.
- Pre-extracted I3D features are used, adhering to the training protocols of the original backbones.

## Key Experimental Results

### Main Results

YouTube Instructional Videos (ASFormer backbone, reported incrementally):

| Method | Type | Frame Acc Hmean | Seg F1@25 Hmean | Global Acc |
|------|------|----------------|-----------------|---------|
| ASFormer Baseline | — | 26.0 | 28.4 | 69.8 |
| + CB (reweighting) | reweight | +2.8 | +0.8 | -0.2 |
| + LA (logit adj.) | logit adj. | +5.3 | +2.1 | -1.9 |
| + BAGS (ensemble) | ensemble | +3.3 | +1.6 | -0.5 |
| **+ G-TLA (Ours)** | **logit adj.** | **+7.5** | **+4.6** | **+0.1** |

Breakfast (MSTCN backbone):

| Method | Frame Acc Hmean | Seg F1@25 Hmean | Global Acc |
|------|----------------|-----------------|---------|
| MSTCN Baseline | 47.7 | 44.8 | 67.7 |
| + LA | +2.1 | +0.9 | -0.1 |
| + Seesaw | +2.4 | +0.5 | +0.9 |
| **+ G-TLA (Ours)** | **+5.0** | **+6.7** | **+2.6** |

On the MSTCN backbone, G-TLA improves the Seg F1 on the Breakfast dataset by 6.7 points and the Global Acc by 2.6 points, while most competing methods sacrifice global metrics.

### Ablation Study

Stepwise addition of components on Breakfast (MSTCN backbone):

| GP | LA | TF | Frame Hmean | Seg F1 Hmean | Description |
|----|----|----|------------|-------------|------|
| ✗ | ✗ | ✗ | 47.7 | 44.8 | Baseline |
| ✗ | ✓ | ✗ | 49.8 | 45.7 | Naive LA, head performance drops |
| ✓ | ✗ | ✗ | 50.9 | 51.3 | Group-wise classification significantly reduces over-segmentation (+5.5% F1) |
| ✓ | ✓ | ✗ | 51.3 | 51.2 | Intra-group LA |
| ✓ | ✓ | ✓ | **52.7** | **51.5** | Temporal factor further reduces false positives |

Hyperparameter sensitivity: Optimal performance is achieved at $\eta=0.5$ and $\tau=0.5$, with relatively stable performance within the range of 0.1 to 0.7.

### Key Findings

- Group-wise classification is the most critical component: GP alone yields a +5.5 F1 improvement on MSTCN.
- Traditional long-tail methods (e.g., CB, Focal, LA) perform poorly in temporal segmentation due to neglecting inter-action dependencies.
- While many methods privilege frame accuracy but sacrifice segment-level F1 (the over-segmentation issue), G-TLA simultaneously improves both metrics.
- SOTA methods like ASFormer and DiffAct suffer from zero accuracy on nearly 10% of the classes, which is effectively alleviated by G-TLA.
- G-TLA also yields stable improvements on balanced datasets (e.g., 50Salads, GTEA), demonstrating wide applicability.

## Highlights & Insights

- **The contribution of problem formulation outweighs that of the methodology**: This work is the first to systematically reveal the long-tail problem in temporal action segmentation and proposes corresponding evaluation metrics (per-class harmonic mean).
- Group-wise classification cleverly bypasses the numerical issues (the $\log 0$ problem) caused by a conditional probability of $p(c|a)=0$.
- The design of the temporal factor is elegant: it performs standard adjustment inside the temporal window, and maintains consistency outside the window via ratio matching, preventing both over-suppression and false positives.
- Plug-and-play: G-TLA only modifies the classification layer and can be directly integrated into various backbones such as MSTCN, ASFormer, and DiffAct.

## Limitations & Future Work

- It requires activity labels (or clustering results) to determine grouping, which increases the demand for prior information.
- The temporal constraints $S_{bf}[c]$ and $S_{af}[c]$ are statistically derived from the training data, which might fail for novel chronological sequences unseen in the training set.
- The performance gain on tail classes in large-scale datasets like Assembly101 remains limited (increasing Frame Acc from 4.7 to 9.2, which is still low in absolute terms).
- Incorporating group information into representation learning has not been explored (currently, it is only applied to the classification layer).

## Related Work & Insights

- Relationship with Logit Adjustment [Menon et al.]: G-TLA is a non-trivial extension of logit adjustment to temporal structured prediction tasks, with the core innovation lying in the conditional priors and the temporal factor.
- Relationship with BAGS (ensemble method): Group-wise classification shares a similar divide-and-conquer philosophy, but G-TLA is more systematic and incorporates temporal constraints.
- Insight: Structural priors in procedural activity videos (activity-action hierarchy, temporal order) are heavily undervalued sources of information.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Address the long-tail problem in temporal action segmentation for the first time; the joint design of group-wise classification and temporal logit adjustment is logical and novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Tested on 5 datasets with 3 backbones and compared against 7 long-tail baselines, complete with detailed ablation and hyperparameter analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation is clear, and the running example of "making tea" is highly intuitive throughout the paper.
- **Value**: ⭐⭐⭐⭐ — Fills an important research gap; the plug-and-play method is highly practical, and the new evaluation metric possesses good potential for broader adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](../../ICCV2025/segmentation/skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ECCV 2024\] VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation via Disentanglement and Spatio-Temporal Side Network](vp-sam_taming_segment_anything_model_for_video_polyp_segmentation_via_disentangl.md)
- [\[ECCV 2024\] Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills](segmentation-guided_layer-wise_image_vectorization_with_gradient_fills.md)
- [\[ECCV 2024\] DreamLIP: Language-Image Pre-training with Long Captions](dreamlip_language-image_pre-training_with_long_captions.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](../../ICCV2025/segmentation/temporal_rate_reduction_clustering_for_human_motion_segmentation.md)

</div>

<!-- RELATED:END -->
