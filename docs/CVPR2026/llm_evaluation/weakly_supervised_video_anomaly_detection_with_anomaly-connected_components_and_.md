---
title: >-
  [Paper Note] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning
description: >-
  [CVPR 2026][LLM Evaluation][Weakly supervised video anomaly detection] This paper proposes the LAS-VAD framework, which introduces an Anomaly-Connected Components (ACC) mechanism to partition video frames into semantically consistent groups for pseudo-label generation to compensate for the absence of frame-level annotations, and an Intention-Aware Mechanism (IAM) that leverages position-velocity-acceleration features to distinguish normal from anomalous behaviors with similar appearances but different intentions. The method achieves 89.96% AP (I3D) on XD-Violence.
tags:
  - CVPR 2026
  - LLM Evaluation
  - Weakly supervised video anomaly detection
  - connected components
  - intention reasoning
  - CLIP
  - multiple instance learning
date: 2026-05-08
content_hash: 913fd486211912da
---

# Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning

**Conference**: CVPR 2026
**arXiv**: [2603.00550](https://arxiv.org/abs/2603.00550)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: Weakly supervised video anomaly detection, connected components, intention reasoning, CLIP, multiple instance learning

## TL;DR

This paper proposes the LAS-VAD framework, which introduces an Anomaly-Connected Components (ACC) mechanism to partition video frames into semantically consistent groups for pseudo-label generation to compensate for the absence of frame-level annotations, and an Intention-Aware Mechanism (IAM) that leverages position-velocity-acceleration features to distinguish normal from anomalous behaviors with similar appearances but different intentions. The method achieves 89.96% AP (I3D) on XD-Violence.

## Background & Motivation

**Background**: Weakly supervised video anomaly detection (WS-VAD) uses only video-level labels and identifies anomalous temporal intervals via multiple instance learning (MIL). Mainstream approaches adopt a pipeline of pretrained feature extraction followed by a classifier.

**Limitations of Prior Work**:
   - **Insufficient semantic information**: The absence of frame-level annotations makes it difficult for models to learn semantic representations of anomalies; learning can only proceed indirectly through MIL's top-K strategy.
   - **Ambiguous behavioral discrimination**: Normal and anomalous behaviors can appear highly similar (e.g., "picking up an object" vs. "stealing an object"), making appearance-only features insufficient for discrimination.

**Key Challenge**: Missing frame-level annotations ↔ requirement for frame-level semantic understanding; similar appearance ↔ different intentions.

**Key Insight**:
   - Semantic problem: Construct a connected-component graph based on inter-frame similarity so that frames in the same group share semantic labels → pseudo-labels.
   - Intention problem: Anomalous behaviors often exhibit abnormal velocity/acceleration (stealing is faster than normally picking up an object); kinematic features are used to reason about intention.

**Core Idea**: Learning anomaly semantics = spatial semantic grouping (ACC) + motion intention reasoning (IAM) + anomaly attribute augmentation.

## Method

### Overall Architecture

The core pipeline of LAS-VAD:
1. Extract frame-level features $X_\text{video} \in \mathbb{R}^{T \times D}$ using a CLIP visual encoder.
2. Model temporal dependencies with a local Transformer and GCN to obtain enhanced features $X_f$.
3. Extract anomaly category features $X_\text{lang}$ via a text encoder; generate anomaly attribute description features $X_\text{aux}$ with an LLM.
4. The ACC module generates frame-level pseudo-labels to guide learning.
5. The IAM module distinguishes intentions via kinematic features.
6. Multi-branch prediction fusion: $p^f = \frac{1}{3}(q^m + q^a + q^l)$.

### Key Designs

1. **Anomaly-Connected Components (ACC)**:

    - **Function**: Partition video frames into semantically consistent, non-overlapping groups to generate frame-level pseudo-labels.
    - **Mechanism**:
        - Compute inter-frame visual similarity $\mathcal{A}_v = \frac{X_f \cdot X_f^T}{\|X_f\| \cdot \|X_f\|}$.
        - Correct bias using cross-modal semantic similarity: $\hat{\mathcal{A}}_w[i,j] = \mathcal{A}_v[i,j] \cdot (1 + \eta \cdot \max_c \min(q^l[i,c], q^l[j,c]))$.
        - Binarize $\mathcal{A} = (\hat{\mathcal{A}} > \tau)$ to construct an adjacency matrix.
        - Apply DFS to identify connected components $B_1, B_2, ..., B_r$; frames within the same component share a semantic label.
    - **Design Motivation**: Circumvents the absence of frame-level annotations — it is unnecessary to know the label of each individual frame; it suffices to identify which frames belong to the same semantic group.

2. **Intention-Aware Mechanism (IAM)**:

    - **Function**: Infer behavioral intention from kinematic features to distinguish behaviors with similar appearances but different intentions.
    - **Mechanism**:
        - Extract positional features $X_p$ from $X_f$; compute velocity $X_v$ and acceleration $X_a$ via differencing.
        - Gating: $X_v = \text{Sigmoid}(\text{Conv}(X_v^\text{diff})) \times X_v^\text{diff}$.
        - Concatenate to obtain intention features $X_\text{int} \in \mathbb{R}^{T \times D}$.
        - Establish **intention prototypes** $Z \in \mathbb{R}^{(C+1) \times D}$ with momentum updates.
        - **Cross-intention contrastive learning**: mine the least similar positive samples within the same class and the most similar negative samples from different classes, constrained by infoNCE:
       $$\mathcal{L}_\text{cst} = -\frac{1}{T}\sum_{t=1}^T \log \frac{\exp(X_\text{int}^t \cdot S_\text{pos}^t)}{\sum_{i=1}^M \exp(X_\text{int}^t \cdot S_\text{neg}^t)}$$
    - **Design Motivation**: The difference between theft and normal retrieval lies in "grasping speed" — velocity/acceleration features naturally encode this intentional distinction.

3. **Anomaly Attribute Augmentation**:

    - **Function**: Leverage LLM-generated attribute descriptions for each anomaly category (e.g., "explosion → flames, dense smoke") as auxiliary features to support detection.
    - **Mechanism**: $X_\text{text} = [X_\text{lang}; X_\text{aux}]$; compute cross-modal cosine similarity with video features to obtain $q^l$.
    - **Design Motivation**: Anomalous events are accompanied by characteristic attributes; textual descriptions provide additional semantic guidance.

### Loss & Training

$$\mathcal{L}_\text{all} = \mathcal{L}_\text{ags} + \mathcal{L}_\text{fg} + \mathcal{L}_\text{aux} + \lambda \mathcal{L}_\text{reg}$$

- $\mathcal{L}_\text{ags}$: Binary cross-entropy (coarse-grained anomaly/normal).
- $\mathcal{L}_\text{fg}$: Multi-class cross-entropy (fine-grained anomaly categories).
- $\mathcal{L}_\text{aux}$: L1 loss on ACC pseudo-labels.
- $\mathcal{L}_\text{reg}$: Consistency regularization between coarse- and fine-grained predictions.

## Key Experimental Results

### Main Results

| Dataset | Feature | Metric | LAS-VAD | Prev. SOTA | Gain |
|--------|------|------|---------|---------|------|
| XD-Violence | I3D | AP(%) | **89.96** | LEC-VAD 88.47 | +1.49 |
| XD-Violence | CLIP | AP(%) | **87.92** | LEC-VAD 86.56 | +1.36 |
| UCF-Crime | I3D | AUC(%) | **91.05** | π-VAD 90.33 | +0.72 |
| UCF-Crime | CLIP | AUC(%) | **90.86** | LEC-VAD 89.97 | +0.89 |

Fine-grained mAP (XD-Violence, avg IoU 0.1–0.5):

| Method | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | AVG |
|------|-----|-----|-----|-----|-----|-----|
| LEC-VAD | 19.65 | 17.17 | 14.37 | 9.45 | 7.18 | 13.56 |
| **LAS-VAD** | **22.07** | **19.96** | **16.18** | **11.24** | **8.64** | **15.62** |

### Ablation Study

| ATT | ACC | IAM | mAP AVG | Note |
|-----|-----|-----|---------|------|
| ✗ | ✗ | ✗ | 24.24 | Baseline |
| ✓ | ✗ | ✗ | 26.50 | Attribute augmentation is effective |
| ✓ | ✓ | ✗ | 29.78 | ACC contributes most (+3.28) |
| ✓ | ✓ | ✓ | **29.98** | IAM yields further improvement |

### Key Findings
- ACC (connected components) is the most impactful module; pseudo-labels provide critical frame-level supervision.
- IAM's intention reasoning is notably effective in scenarios with similar appearances, although the overall gain is relatively modest (+0.20).
- LLM-generated anomaly attribute descriptions provide meaningful semantic supplementation (+2.26).
- The method achieves state-of-the-art results across two datasets and three feature extractors (C3D/I3D/CLIP).

## Highlights & Insights
- **Connected components for frame grouping**: The paper elegantly applies the graph-theoretic concept of connected components to semantic grouping of video frames — a concise and effective idea. The key lies in the cross-modal semantic correction step: pure visual similarity is biased, and cross-modal correction yields more accurate groupings.
- **Kinematic encoding of intention**: Designing features from the physics-inspired notion of position-velocity-acceleration is intuitively sound — theft actions are genuinely faster than normal retrieval. The gating mechanism for noise filtering is also a well-motivated design choice.
- **LLM attribute descriptions as textual priors**: Using GPT-4 to generate anomaly attribute descriptions is simple yet effective and requires no manual prompt engineering.

## Limitations & Future Work
- The threshold $\tau$ in ACC requires manual tuning (set to 0.9) and may be sensitive to different video types.
- The position/velocity/acceleration feature extraction in IAM is relatively simple (fully connected layers + differencing), which may be insufficient for modeling complex motion patterns.
- The method relies on GPT-4 for attribute description generation, introducing an external model dependency.
- The momentum update mechanism for intention prototypes may be unstable during early training.
- Validation on larger-scale datasets (e.g., anomaly subsets of Kinetics-700) has not been conducted.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of ACC and IAM is original; connected-component-based frame grouping is a notable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across two datasets and multiple features, with complete ablation studies.
- Writing Quality: ⭐⭐⭐ Motivation descriptions are somewhat verbose, and the notation is dense.
- Value: ⭐⭐⭐⭐ Represents solid progress in the weakly supervised VAD domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](../../AAAI2026/llm_evaluation/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[NeurIPS 2025\] Normal-Abnormal Guided Generalist Anomaly Detection](../../NeurIPS2025/llm_evaluation/normal-abnormal_guided_generalist_anomaly_detection.md)
- [\[CVPR 2026\] Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)

</div>

<!-- RELATED:END -->
