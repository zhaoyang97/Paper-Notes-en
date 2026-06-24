---
title: >-
  [Paper Note] AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting
description: >-
  [ACL 2025][Self-Supervised Learning][Keyword Spotting] AnalyticKWS is proposed, an exemplar-free incremental learning method for keyword spotting. By freezing the feature extractor and analytically updating the classifier via recursive least squares, it outperforms all rehearsal-based methods on the GSC and SC-100 datasets with extremely low training time and memory overhead.
tags:
  - "ACL 2025"
  - "Self-Supervised Learning"
  - "Keyword Spotting"
  - "Class Incremental Learning"
  - "Exemplar-free"
  - "Analytic Learning"
  - "Edge Devices"
date: 2026-05-08
content_hash: df41affd41fe6b9d
---

# AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting

**Conference**: ACL 2025  
**arXiv**: [2505.11817](https://arxiv.org/abs/2505.11817)  
**Code**: None  
**Area**: Others  
**Keywords**: Keyword Spotting, Class Incremental Learning, Exemplar-free, Analytic Learning, Edge Devices

## TL;DR
AnalyticKWS is proposed, an exemplar-free incremental learning method for keyword spotting. By freezing the feature extractor and analytically updating the classifier via recursive least squares, it outperforms all rehearsal-based methods on the GSC and SC-100 datasets with extremely low training time and memory overhead.

## Background & Motivation
**Background**: Keyword Spotting (KWS) is a core component of voice interaction on edge devices, requiring lightweight models (such as TC-ResNet) for rapid response. As user needs evolve, models must continuously learn new keywords.

**Limitations of Prior Work**: Direct fine-tuning on new keywords leads to catastrophic forgetting. Most existing continual learning methods rely on storing old data for rehearsal, which poses two issues: (a) privacy risks, as storing user voice data may violate regulations like GDPR, and (b) memory and computational overhead, making them unsuitable for resource-constrained edge devices.

**Key Challenge**: Enabling lightweight KWS models to continuously learn new keywords without storing any historical data, while avoiding forgetting previous ones.

**Goal**: To design an exemplar-free, privacy-friendly, and computationally efficient incremental learning method for KWS.

**Key Insight**: Leveraging the recursive least squares formulation in Analytic Learning to transition classifier updates from iterative gradient descent to closed-form analytical solutions, mathematically guaranteeing consistency between old and new knowledge.

**Core Idea**: To freeze the CNN feature extractor and analytically update the linear classifier using recursive least squares, achieving theoretical equivalence to joint training without storing any historical data.

## Method

### Overall Architecture
A three-stage pipeline:
- **Stage 1 - Feature Extractor Pre-training**: Train the full model (CNN + classifier) on the initial task using standard gradient descent, then freeze the CNN feature extractor.
- **Stage 2 - Feature Recalibration**: Use Acoustic Feature Expansion (AFE) to increase feature dimensions, retrain the classifier using analytical solutions (least squares) instead of gradient descent, and save the Acoustic Feature Autocorrelation Matrix (AFAM).
- **Stage 3 - Incremental Keyword Adaptation**: For each new task, only one epoch of forward propagation is required to recursively update the AFAM and classifier weights.

### Key Designs

1. **Acoustic Feature Expansion (AFE)**:

    - Function: Inserts a randomly initialized and fixed linear layer after the frozen CNN feature extractor to map features to a higher-dimensional space.
    - Mechanism: $\mathbf{S}_0' = \text{AFE}(\mathbf{S}_0)$ expanding dimensions (e.g., to 128 or 256), while the random weights $\theta_{afe}$ remain fixed and untrained.
    - Design Motivation: Lightweight KWS models have low feature dimensions, making direct analytical solutions prone to underfitting. Expanding features to higher dimensions preserves more subtle discriminative details, similar to random projection or reservoir computing concepts.

2. **Analytic Learning Classifier**:

    - Function: Replaces gradient descent with the closed-form solution of Ridge regression to train the linear classifier.
    - Mechanism: $\hat{\theta}_{cls}^{(0)} = (\mathbf{S}_0'^T \mathbf{S}_0' + \gamma I)^{-1} \mathbf{S}_0'^T y_0$ to directly calculate optimal weights.
    - Design Motivation: The closed-form solution requires no iterative epochs and completes in a single step; more importantly, it can be mathematically extended to new tasks recursively.

3. **Recursive Incremental Update (Core Innovation)**:

    - Function: When a new task $\tau_t$ arrives, recursively update the classifier weights and AFAM using only the current task data.
    - Mechanism: Maintain the acoustic feature autocorrelation matrix $\mathbb{A}_t$, updating it recursively via the Woodbury matrix identity: $\mathbb{A}_t = \mathbb{A}_{t-1} - \Delta$. The new classifier weights are obtained from preceding weights + AFAM + closed-form updates of the new data.
    - Design Motivation: The recursive formulation is mathematically equivalent to the analytical solution of joint training on all historical and new task data, theoretically guaranteeing **zero-forgetting** without storing historical data. This serves as the core theoretical foundation of the method.

### Loss & Training
- Train the initial task for 50 epochs using SGD to obtain a robust feature extractor.
- Each incremental task requires only 1 epoch of forward propagation and analytical updates.
- Backpropagation and optimizers are bypassed, yielding extremely low computational cost.

## Key Experimental Results

### Main Results
Comparison across GSC-v1, GSC-v2, and SC-100 datasets (* indicates the use of an exemplar buffer of size 500):

| Method | GSC-v1 T=11 ACC | GSC-v2 T=11 ACC | SC-100 T=11 ACC | Exemplar Buffer? |
|------|-----------------|-----------------|-----------------|-------------|
| Finetune | 17.99 | 16.82 | 15.07 | No |
| EWC | 71.65 | 68.20 | 43.90 | No |
| iCaRL* | 81.14 | 79.16 | 69.30 | 500 |
| Rwalk* | 85.38 | 87.27 | 76.93 | 500 |
| DE-KWS* | 85.59 | 85.34 | 67.71 | 500 |
| **AnalyticKWS-256** | **85.83** | **89.53** | **87.99** | **No** |

### Ablation Study
Ablation: Feature Expansion Dimensions

| Config | GSC-v2 T=11 ACC | SC-100 T=11 ACC |
|------|-----------------|-----------------|
| AnalyticKWS-128 | 88.87 | 85.77 |
| AnalyticKWS-256 | 89.53 | 87.99 |

### Key Findings
- AnalyticKWS consistently outperforms all rehearsal methods with a buffer size of 500, despite not using any exemplar buffer.
- The performance gain is most prominent on SC-100 (100 keywords): 87.99 vs 76.93 (Rwalk), yielding an 11% improvement.
- The backward transfer (BWT) metric is close to 0 (virtually zero forgetting), significantly outperforming all compared methods.
- The performance gap widens as the number of tasks increases (with an even larger gap at T=51).
- The training time is only 1/10 of Finetune, much lower than EWC, Rwalk, etc.

## Highlights & Insights
- **Mathematically Guaranteed Zero-Forgetting**: Rather than relying on empirical heuristic tricks, the approach uses recursive least squares to mathematically guarantee similarity to joint training, presenting an elegant formulation.
- **Extreme Efficiency**: Each new task demands only 1 epoch of forward propagation and matrix calculations without backpropagation, suited for real-time updates on edge devices.
- **Privacy-Friendly**: Storing no historical user audio data ensures compliance with privacy regulations like GDPR inherently.
- **Ingenious Feature Expansion**: Leveraging a fixed random projection to project low-dimensional features to higher dimensions is simple yet highly effective, retaining richer information for analytic solutions.

## Limitations & Future Work
- Dependency on a robust initial-stage feature extractor: If the initial feature extractor is suboptimal, it caps the performance ceiling of the analytic classifier.
- Only validated on TC-ResNet-8: Larger or more modern models (e.g., Conformer) remain untested.
- Selection of AFE resolution/dimensions lacks theoretical guidance, with evaluations limited to 128 and 256 dimensions.
- Unaddressed domain shift scenarios: If acoustic properties of new keywords deviate considerably from initial training data, the frozen feature extractor might struggle.
- Future work could integrate adapters or LoRA to allow minor parameters updates to the feature extractor while preserving overall efficiency.

## Related Work & Insights
- **vs iCaRL/Rwalk**: These methods mandate storing 500 exemplars for rehearsal. AnalyticKWS completely avoids this requirement while delivering superior performance, demonstrating that for linear classification layers, analytic solutions are more efficient than gradient descent with replay.
- **vs EWC**: EWC penalizes parameter changes using the Fisher information matrix but performs mediocrely on KWS (only 43.9% on SC-100). AnalyticKWS fundamentally circumvents forgetting.
- **vs ACIL**: ACIL represents the initial application of analytic learning in image classification; AnalyticKWS extends this to the audio domain and incorporates acoustic feature expansion.
- The core concept of this approach (frozen feature extractor + analytic classifier updating) can be transferred to other sequential classification tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces analytic learning to incremental KWS, with a simple and effective AFE design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on three datasets, various task settings, comparisons against multiple baselines, and complete efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations with systematic and complete methodology descriptions.
- Value: ⭐⭐⭐⭐ Highly practical for deploying KWS on edge devices, with the mathematical guarantee of performance serving as a key highlight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](../../CVPR2026/self_supervised/exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[ICLR 2026\] Two-Way is Better Than One: Bidirectional Alignment with Cycle Consistency for Exemplar-Free Class-Incremental Learning](../../ICLR2026/self_supervised/two-way_is_better_than_one_bidirectional_alignment_with_cycle_consistency_for_ex.md)
- [\[ECCV 2024\] Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](../../ECCV2024/self_supervised/exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)
- [\[CVPR 2025\] Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping](../../CVPR2025/self_supervised/order-robust_class_incremental_learning_graph-driven_dynamic_similarity_grouping.md)
- [\[CVPR 2026\] Exemplar-Free Continual Learning for State Space Models](../../CVPR2026/self_supervised/exemplar-free_continual_learning_for_state_space_models.md)

</div>

<!-- RELATED:END -->
