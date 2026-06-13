---
title: >-
  [Paper Note] Enhancing Out-of-Distribution Detection with Extended Logit Normalization
description: >-
  [CVPR 2026][AI Safety][OOD Detection] This paper identifies two forms of feature collapse induced by LogitNorm during training—dimensional collapse and origin collapse—and proposes a hyperparameter-free Extended Logit No…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "OOD Detection"
  - "Logit Normalization"
  - "Feature Collapse"
  - "Decision Boundary"
  - "Model Calibration"
date: 2026-05-08
content_hash: 24af1141cae15262
---

# Enhancing Out-of-Distribution Detection with Extended Logit Normalization

**Conference**: CVPR 2026
**arXiv**: [2504.11434](https://arxiv.org/abs/2504.11434)  
**Code**: [https://github.com/limchaos/ElogitNorm](https://github.com/limchaos/ElogitNorm)  
**Area**: LLM Evaluation
**Keywords**: OOD Detection, Logit Normalization, Feature Collapse, Decision Boundary, Model Calibration

## TL;DR
This paper identifies two forms of feature collapse induced by LogitNorm during training—dimensional collapse and origin collapse—and proposes a hyperparameter-free Extended Logit Normalization (ELogitNorm) that replaces the distance-to-origin scaling factor with the distance from features to the decision boundary. ELogitNorm significantly improves both post-hoc OOD detection performance and confidence calibration without sacrificing classification accuracy.

## Background & Motivation
Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning models. Existing approaches either design post-hoc scoring functions (MSP, KNN, SCALE, etc.) or modify training objectives to improve OOD discriminability. LogitNorm, which normalizes the logit vector to mitigate overconfidence, is a representative training-time method.

However, LogitNorm suffers from three key limitations: (1) it induces **feature collapse**—feature variance concentrates along a few directions and OOD samples cluster near the origin; (2) it trades classification accuracy for OOD performance; and (3) it is only effective for a limited set of scoring functions, and actually degrades performance when combined with certain post-hoc methods.

The core insight of this paper is that the LogitNorm normalization factor $\tau\|\mathbf{f}\|$ is essentially equivalent to scaling by the distance-to-origin $\|\mathbf{z}\|$ (since $\|\mathbf{f}\| \approx \bar{\sigma}\|\mathbf{z}\| + \eta$), which encourages features to collapse toward the origin. A more principled alternative is to use the **distance from features to the decision boundary** $\mathcal{D}(\mathbf{z})$ as the scaling factor—samples close to the boundary have higher uncertainty, while those far from the boundary are more reliably classified.

## Method

### Overall Architecture
ELogitNorm serves as a drop-in replacement for the standard cross-entropy loss. The model architecture remains unchanged (ResNet-18/50); only the loss function is substituted from $\mathcal{L}_{CE}$ to $\mathcal{L}_{ELogitNorm}$. After training, any post-hoc OOD scoring method can be applied seamlessly.

### Key Designs

1. **Feature Collapse Diagnosis**:

    - Function: Reveal two collapse phenomena induced by LogitNorm.
    - Mechanism: (a) **Dimensional collapse**—the singular value spectrum of the feature covariance matrix trained with LogitNorm contains many near-zero singular values, indicating a significant reduction in effective feature dimensionality; (b) **Origin collapse**—OOD samples cluster near the origin in feature space, a tendency further exacerbated by LogitNorm's normalization.
    - Design Motivation: Proposition 1 proves that $\|\mathbf{f}\|$ is approximately proportional to $\|\mathbf{z}\|$ (i.e., $\sigma_{min}\|\mathbf{z}\| - \|\mathbf{b}\| \leq \|\mathbf{f}\| \leq \sigma_{max}\|\mathbf{z}\| + \|\mathbf{b}\|$), showing that LogitNorm implicitly imposes constraints based on distance to the origin.

2. **Decision Boundary Distance Scaling (Core of ELogitNorm)**:

    - Function: Replace the logit norm with the average distance from features to all competing class decision boundaries as the scaling factor.
    - Mechanism: Let $f_{max}$ denote the predicted class index. The scaling factor is defined as $\mathcal{D}(\mathbf{z}) = \frac{1}{c-1}\sum_{i \neq f_{max}} \frac{|(\mathbf{w}_{f_{max}} - \mathbf{w}_i)^T\mathbf{z} + (b_{f_{max}} - b_i)|}{\|\mathbf{w}_{f_{max}} - \mathbf{w}_i\|_2}$, and the training loss is $\mathcal{L}_{ELogitNorm} = -\log \frac{e^{f_y/\mathcal{D}(\mathbf{z})}}{\sum_i e^{f_i/\mathcal{D}(\mathbf{z})}}$.
    - Design Motivation: Samples near the decision boundary receive larger scaling, producing stronger gradient signals that force the network to push ambiguous samples away from the boundary.

3. **Minimum Scaling Factor Space Analysis (Proposition 2)**:

    - Function: Prove that the minimum scaling factor space of ELogitNorm has substantially higher dimensionality than that of LogitNorm.
    - Mechanism: The minimum scaling factor of LogitNorm corresponds to the origin (a zero-dimensional point), whereas that of ELogitNorm corresponds to the intersection of all decision boundaries—an affine subspace of dimension $m-c+1$ (e.g., 503-dimensional vs. 0-dimensional for ResNet-18 on CIFAR-10).
    - Design Motivation: A higher-dimensional minimum scaling space provides greater degrees of freedom during optimization, preventing the representation from collapsing to a single point.

### Loss & Training
The sole training objective is $\mathcal{L}_{ELogitNorm}$, with **no additional hyperparameters** (unlike LogitNorm, which requires tuning the temperature $\tau$). Training settings are identical to standard cross-entropy: ResNet-18 on CIFAR for 100 epochs, SGD, lr=0.1, momentum=0.9, weight decay $5 \times 10^{-4}$.

## Key Experimental Results

### Main Results

| ID Dataset | Scoring Method | Metric | Cross-Entropy | LogitNorm | ELogitNorm | Gain |
|-----------|---------|------|---------------|-----------|------------|------|
| CIFAR-10 | SCALE | far-OOD AUROC | 86.46 | — | **96.94** | +10.48 |
| CIFAR-10 | SCALE | far-OOD FPR95 | 67.49 | — | **13.18** | -54.31 |
| CIFAR-10 | MSP | far-OOD AUROC | 90.73 | 96.74 | **96.68** | +5.95 |
| ImageNet-1K | MSP | far-OOD AUROC | 85.23 | 91.54 | **93.19** | +7.96 |
| ImageNet-1K | MSP | far-OOD FPR95 | 51.45 | 31.32 | **27.74** | -23.71 |
| ImageNet-200 | KNN | far-OOD AUROC | 93.16 | — | **96.08** | +2.92 |

### Ablation Study

| Configuration | ECE (%) ↓ | Notes |
|------|---------|------|
| Cross-Entropy + original logit | 3.3 | Baseline calibration |
| LogitNorm + $\mathbf{f}/(\tau\|\mathbf{f}\|)$ | 4.1 | Best LogitNorm configuration |
| ELogitNorm + $\mathbf{f}/\mathcal{D}(\mathbf{z})$ | **1.8** | Best calibration, lowest ECE |
| LogitNorm classification accuracy (CIFAR-10) | 94.83 | Below Cross-Entropy (95.10) |
| ELogitNorm classification accuracy (CIFAR-10) | **95.11** | On par with or better than Cross-Entropy |
| ELogitNorm classification accuracy (ImageNet-200) | **87.12** | Surpasses Cross-Entropy (86.58) |

### Key Findings
- ELogitNorm yields the most substantial gains in the far-OOD setting; the FPR95 of the SCALE method drops from 67.49% to 13.18%.
- Unlike LogitNorm, ELogitNorm is compatible with all post-hoc methods (LogitNorm + ReAct leads to severe degradation).
- Singular value spectrum analysis confirms that ELogitNorm produces more uniformly distributed features, effectively avoiding dimensional collapse.
- The hyperparameter-free design simplifies deployment, eliminating the need to reserve a validation set for temperature tuning.

## Highlights & Insights
- The feature collapse diagnostic perspective is highly original: linking the LogitNorm normalization factor to the distance-to-origin in feature space reveals an implicit collapse mechanism.
- Proposition 2 provides an elegant geometric justification for why distance to the decision boundary is a superior scaling factor compared to distance to the origin.
- The hyperparameter-free design is a significant practical advantage: LogitNorm requires tuning $\tau$, whereas ELogitNorm is fully adaptive.

## Limitations & Future Work
- Improvements on near-OOD benchmarks are relatively modest, a challenge the authors acknowledge as common to all training-time methods.
- Computing the decision boundary distance involves $c-1$ hyperplanes; when the number of classes is large (e.g., 1,000 for ImageNet-1K), this may incur additional computational overhead, despite the authors claiming an efficient implementation.
- The method has not been validated on Transformer-based architectures such as ViT.

## Related Work & Insights
- Compared to methods designed to work with KNN scoring (e.g., CIDER, NPOS), ELogitNorm achieves superior results with a simpler approach (far-OOD AUROC on ImageNet-200: 96.08 vs. 94.83/90.66).
- The decision-boundary-aware scaling idea generalizes naturally to other settings, including uncertainty estimation and domain adaptation.
- The unified adaptive temperature scaling perspective ($s = \tau\|\mathbf{f}\|$ vs. $s = \mathcal{D}(\mathbf{z})$) provides a principled framework for designing improved calibration losses.

## Rating
- Novelty: ⭐⭐⭐⭐ — The motivation from feature collapse diagnosis and decision boundary distance scaling is well-grounded, though the core technical modification is relatively minor.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — OpenOOD framework, 4 in-distribution datasets, 6 post-hoc methods, 3 repeated runs, comprehensive evaluation of calibration and accuracy.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical analysis is rigorous and figures are clear, though some equations are repeated and slightly verbose.
- Value: ⭐⭐⭐⭐ — Offers practical value to the OOD detection community; the hyperparameter-free design lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[CVPR 2026\] LogitDynamics: Reliable ViT Error Detection from Layerwise Logit Trajectories](logitdynamics_vit_error_detection.md)
- [\[ICLR 2026\] AP-OOD: Attention Pooling for Out-of-Distribution Detection](../../ICLR2026/ai_safety/ap-ood_attention_pooling_for_out-of-distribution_detection.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/ai_safety/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[NeurIPS 2025\] Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis](../../NeurIPS2025/ai_safety/double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)

</div>

<!-- RELATED:END -->
