---
title: >-
  [Paper Note] DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets
description: >-
  [NeurIPS 2025][bias amplification] This paper proposes Directional Predictability Amplification (DPA), a predictability-based metric for measuring bias amplification. It is the only one-stop metric that simultaneously satisfies directionality, applicability to both balanced and imbalanced datasets, and correct identification of positive and negative bias amplification, by measuring the relative change between model bias and dataset bias.
tags:
  - NeurIPS 2025
  - bias amplification
  - fairness metric
  - predictability
  - directional bias
  - classification fairness
date: 2026-05-08
content_hash: 88581f61af1fe33b
---

# DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets

**Conference**: NeurIPS 2025
**arXiv**: [2412.11060](https://arxiv.org/abs/2412.11060)
**Code**: Available
**Area**: Other
**Keywords**: bias amplification, fairness metric, predictability, directional bias, classification fairness

## TL;DR
This paper proposes Directional Predictability Amplification (DPA), a predictability-based metric for measuring bias amplification. It is the only one-stop metric that simultaneously satisfies directionality, applicability to both balanced and imbalanced datasets, and correct identification of positive and negative bias amplification, by measuring the relative change between model bias and dataset bias.

## Background & Motivation

**State of the Field**: ML models not only learn biases present in training data but also amplify them. For example, 67% of cooking images in the ImSitu dataset feature women, yet model predictions may push this proportion to 90%. Existing metrics fall into two categories: co-occurrence-based methods (BA, Multi, BA_MALS) and predictability-based methods (LA).

**Limitations of Prior Work**: Co-occurrence-based methods (BA, BA_MALS) fail on balanced datasets; Multi supports balanced datasets but cannot detect negative bias amplification; LA handles balanced datasets but lacks directionality. No single metric simultaneously satisfies all three key properties.

**Root Cause**: Co-occurrence-based methods naturally support directionality but are constrained by annotation information, while predictability-based methods can capture hidden biases but lack directionality — the challenge lies in combining the advantages of both.

**Paper Goals**: Design a one-stop metric that simultaneously satisfies: (1) directionality; (2) applicability to both balanced and imbalanced datasets; (3) correct identification of positive and negative bias amplification.

**Starting Point**: Extend the predictability framework of LA into a bidirectional one, and quantify amplification via relative change rather than absolute change.

**Core Idea**: Measure bias amplification using the normalized relative change of directional predictability — one metric addressing three problems.

## Method

### Overall Architecture
Given a classification dataset (images $I$, task labels $T$, protected attributes $A$) and the predictions of model $M$, DPA measures bias amplification in two directions: $A \to T$ and $T \to A$. Predictability in each direction is quantified by training an attacker function.

### Key Designs

1. **Directional Predictability Measure**:

    - Function: Separately quantify dataset bias and model bias in the $A \to T$ and $T \to A$ directions.
    - Core definitions (for the $A \to T$ direction):
        - Dataset bias: $\Psi_A^D = Q(f_A^T(A), T)$ — quality of an attacker predicting true task label $T$ from true attribute $A$
        - Model bias: $\Psi_A^M = Q(f_A^{\hat{T}}(A), \hat{T})$ — quality of an attacker predicting the model's task prediction from true attribute $A$
    - The $T \to A$ direction is analogous, with the roles of $A$ and $T$ exchanged.

2. **Normalized Relative Change (DPA Formula)**:

    - Function: Quantify bias amplification as a bounded value in $[-1, 1]$.
    - Core formula: $DPA_{A \to T} = \frac{\Psi_A^M - \Psi_A^D}{\Psi_A^M + \Psi_A^D}$, $DPA_{T \to A} = \frac{\Psi_T^M - \Psi_T^D}{\Psi_T^M + \Psi_T^D}$
    - Design Motivation: Normalization ensures DPA lies within $[-1, 1]$; relative change is more principled than absolute change — an increase of 0.05 on an unbiased dataset is far more severe than the same increase on a highly biased one.

3. **Accuracy Alignment**:

    - Function: Eliminate the confounding effect of model prediction errors on bias measurement.
    - Mechanism: If the model's task prediction accuracy is 70%, 30% of the true labels $T$ are randomly flipped to match that accuracy level.
    - Design Motivation: Prevents model prediction errors from being mistakenly attributed to bias amplification.

### Loss & Training
The attacker function can be any ML model (SVM, decision tree, MLP), evaluated using a quality function $Q$ such as accuracy or F1 score. The paper uses a two-hidden-layer MLP as the attacker.

## Key Experimental Results

### Main Results: COCO Dataset $T \to A$ Bias Amplification (ViT_b_16, Progressive Person Occlusion)

| Metric | Original | Partial Occlusion | Full Occlusion (Segmentation) | Full Occlusion (Bounding Box) |
|--------|----------|-------------------|-------------------------------|-------------------------------|
| Attribution Score | 0.3827 | 0.4327 | 0.4461 | 0.5247 |
| **DPA (ours)** | **-0.0152** | **0.1365** | **0.6015** | **0.8085** |
| BA_-> | -0.0227 | 0.0097 | 0.0188 | 0.0601 |
| Multi_-> | 0.1506 | 0.1179 | 0.3606 | 0.5607 |
| LA | 0.0368 | 0.0715 | 0.0942 | 0.0265 |
| BA_MALS | 0.0001 | 0.0004 | -0.0016 | 0.015 |

### Metric Capability Comparison

| Method | Balanced Datasets | Directionality | Negative Bias Amplification |
|--------|-------------------|----------------|-----------------------------|
| BA_MALS | No | No | Yes |
| BA_-> | No | Yes | Yes |
| Multi_-> | Yes | Yes | No |
| LA | Yes | No | Yes |
| **DPA (ours)** | **Yes** | **Yes** | **Yes** |

### ImSitu $A \to T$ Model Ranking Consistency

| Model | Ground-Truth Rank (Sen) | DPA Rank | LA Rank | BA_-> Rank |
|-------|-------------------------|----------|---------|------------|
| MaxViT | 1 | 1 | 4 | 3 |
| ViT_b_32 | 2 | 2 | 6 | 5 |
| VGG16 | 9 | 9 | 7 | 9 |

### Key Findings
- In the COCO experiment, as persons are progressively occluded, models increasingly rely on background objects to predict gender; only DPA and BA_-> correctly reflect this trend.
- On the balanced COCO dataset, BA_-> and BA_MALS consistently report zero bias amplification (incorrectly), whereas DPA correctly captures the model's exploitation of unannotated objects.
- On the ImSitu dataset, DPA's model rankings are in perfect agreement with the ground-truth bias rankings, while all other metrics exhibit multiple ranking errors.
- LA decreases as occlusion intensifies (0.0942 → 0.0265), since absolute-change-based measurement cannot handle this scenario.

## Highlights & Insights
- **One-stop solution to three longstanding problems**: Directionality + balanced dataset support + negative bias amplification detection — no prior single metric achieved all three simultaneously.
- **Insight on relative vs. absolute change**: Adding a small amount of bias to an already highly biased dataset is fundamentally different in severity from introducing the same bias to an unbiased dataset.
- **Robustness to the attacker function**: By measuring relative change, systematic biases introduced by the attacker are cancelled out; in contrast, LA's absolute-change measurement is highly sensitive to attacker hyperparameters.

## Limitations & Future Work
- The choice of quality function $Q$ requires attention to directionality (where 0 indicates worst performance); loss functions such as cross-entropy must be inverted.
- Accuracy alignment is implemented via random label flipping, which introduces stochasticity and necessitates confidence intervals.
- The metric is designed for classification tasks only; extensions to regression and generative tasks require additional work.
- Training the attacker function still demands computational resources, particularly in large-scale, multi-attribute settings.

## Related Work & Insights
- **vs. BA_-> (Wang & Russakovsky 2021)**: The first directional metric, but applicable only to imbalanced datasets; DPA extends this to the balanced setting.
- **vs. LA (Wang et al. 2019)**: The first predictability-based method, but lacks directionality and measures absolute change; DPA introduces directionality and normalization.
- **vs. Multi_-> (Zhao et al. 2023)**: Supports multi-attribute and balanced datasets but cannot detect negative bias amplification; DPA naturally supports this via sign changes.

## Rating
- Novelty: ⭐⭐⭐⭐ — Clear problem formulation, concise yet complete solution, and an elegant normalized relative change design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets (COMPAS, COCO, ImSitu), nine models, and carefully designed controlled experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, rigorous metric design logic, and well-crafted experimental setup.
- Value: ⭐⭐⭐⭐ — Directly beneficial to ML fairness practice, eliminating the need for cross-validation across multiple metrics.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [\[AAAI 2026\] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets](../../AAAI2026/others/intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)

<!-- RELATED:END -->
