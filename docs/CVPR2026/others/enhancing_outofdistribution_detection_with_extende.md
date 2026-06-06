---
title: >-
  [Paper Note] ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization
description: >-
  [CVPR 2026][out-of-distribution detection] This paper diagnoses two feature collapse problems in LogitNorm (dimensional collapse and origin collapse)…
tags:
  - "CVPR 2026"
  - "out-of-distribution detection"
  - "logit normalization"
  - "feature collapse"
  - "decision boundary"
  - "calibration"
date: 2026-05-08
content_hash: 413b527e50dd9eb4
---

# ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization

**Conference**: CVPR 2026
**arXiv**: [2504.11434](https://arxiv.org/abs/2504.11434)  
**Code**: [GitHub](https://github.com/limchaos/ElogitNorm)  
**Area**: Other
**Keywords**: out-of-distribution detection, logit normalization, feature collapse, decision boundary, calibration

## TL;DR
This paper diagnoses two feature collapse problems in LogitNorm (dimensional collapse and origin collapse), and proposes ELogitNorm — replacing the feature norm with the average distance to decision boundaries as an adaptive temperature scaling factor. The method requires no hyperparameters, is compatible with all post-hoc OOD detection methods, achieves a 10.48% far-OOD AUROC improvement on CIFAR-10 (with SCALE), reduces FPR95 from 51.45% to 27.74% on ImageNet-1K, and simultaneously improves classification accuracy and ECE calibration.

## Background & Motivation
Among training-time OOD detection methods, LogitNorm improves post-hoc detection performance by modifying the loss function — specifically by dividing logits by their norm to alleviate overconfidence. However, the authors identify two critical issues: (1) **Dimensional collapse**: the singular value spectrum of features contains many near-zero values, indicating compression into a small number of dominant directions; (2) **Origin collapse**: since $\|f\| \propto \|z\|$, LogitNorm implicitly regularizes features by their distance to the origin, pulling both OOD and ID samples toward the origin. These issues limit compatibility with various post-hoc methods and degrade classification accuracy.

## Core Problem
How to design a hyperparameter-free training-time method that improves OOD detection without sacrificing ID classification accuracy, without restricting the choice of post-hoc methods, and while improving confidence calibration?

## Method

### Overall Architecture
The scaling factor in LogitNorm, $s = \tau\|f\|$ (distance to origin), is replaced by $s = D(z)$ (average distance to decision boundaries), thereby extending distance-awareness from a single origin point to all inter-class decision hyperplanes.

### Key Designs
1. **Distance from features to decision boundaries**: For the predicted class $f_{\max}$, the method computes the average point-to-plane distance to the decision boundaries of all other classes:
$$D(z) = \frac{1}{c-1} \sum \frac{|(w_{f_{\max}} - w_i)^T z + (b_{f_{\max}} - b_i)|}{\|w_{f_{\max}} - w_i\|_2}$$
This is a geometrically precise measure of distance to decision boundaries.

2. **ELogitNorm loss**: $L = -\log\!\left(\frac{\exp(f_y / D(z))}{\sum \exp(f_i / D(z))}\right)$. This directly replaces the CE loss during training with no additional hyperparameters (LogitNorm requires tuning $\tau$).

3. **Collapse prevention mechanism (Proposition 2)**: The minimum scaling factor space of LogitNorm is the origin (0-dimensional), whereas that of ELogitNorm is the intersection of all decision boundaries ($m - c + 1$ dimensional, e.g., 503-dimensional for ResNet-18 on CIFAR-10). Optimization is therefore no longer attracted to a single point but is distributed over a high-dimensional affine subspace.

### Loss & Training
ResNet-18 on CIFAR-10/100: 100 epochs, SGD momentum=0.9, lr=0.1, weight decay 5e-4, batch size 128. ImageNet-1K ResNet-50: fine-tuned for 30 epochs with lr=0.001. No additional hyperparameters.

## Key Experimental Results

### CIFAR-10 Far-OOD (ResNet-18, AUROC improvement with various post-hoc methods)

| Post-hoc Method | CE → +ELogitNorm (AUROC↑) |
|----------------|--------------------------|
| MSP | 90.73 → 96.68 (+5.95) |
| GEN | 91.19 → 97.30 (+6.11) |
| ReAct | 92.56 → 97.63 (+5.07) |
| SCALE | 86.99 → 97.47 (+10.48) |
| KNN | 93.86 → 97.75 (+3.89) |

### ImageNet-1K (ResNet-50, MSP)

| Method | Near AUROC | Far AUROC | Far FPR95↓ |
|--------|-----------|-----------|-----------|
| CE | 76.02 | 85.23 | 51.45 |
| LogitNorm | 74.62 | 91.54 | 31.32 |
| **ELogitNorm** | **76.88** | **92.81** | **27.74** |

### Classification Accuracy (Table 5, 200 epochs)

| Dataset | CE | LogitNorm | ELogitNorm |
|---------|----|-----------|-----------|
| CIFAR-10 | 95.10 | 94.83 | **95.11** |
| CIFAR-100 | **77.47** | 76.06 | 77.37 |
| ImageNet-200 | 86.58 | 86.41 | **87.12** |

### Calibration (ECE, CIFAR-10 ResNet-18)

| Method | $f$ raw | $f/\tau\|f\|$ | $f/D(z)$ |
|--------|--------|--------------|---------|
| CE | 3.3 | 4.8 | **2.3** |
| LogitNorm | 58.7 | 4.1 | 52.3 |
| **ELogitNorm** | 26.7 | 4.7 | **1.8** |

### Ablation & Analysis Highlights
- **LogitNorm degrades with ReAct**: On CIFAR-100, LogitNorm+ReAct underperforms CE+ReAct (Fig. 3), whereas ELogitNorm consistently improves all post-hoc methods.
- **Singular value spectrum**: LogitNorm exhibits many near-zero singular values (collapse), while ELogitNorm yields a more uniformly distributed spectrum.
- **$D(z)$ vs. $\|z\|$**: The two quantities are no longer linearly correlated (Fig. 2d vs. 2c), confirming that ELogitNorm introduces additional decision boundary information.
- **Limited near-OOD improvement**: All training-time methods show limited gains on near-OOD benchmarks, which is a common challenge in the field.

## Highlights & Insights
- **"Distance to what?" is the core question**: LogitNorm measures distance to the origin; ELogitNorm measures distance to decision boundaries — the latter is physically more meaningful (farther from the boundary = more certain).
- **Hyperparameter-free design**: LogitNorm requires tuning $\tau$; ELogitNorm requires no additional hyperparameters, as $D(z)$ adapts naturally to the data.
- **Geometric insight of Proposition 2**: The minimum scaling factor space expands from 0-dimensional (the origin) to $m - c + 1$ dimensional, fundamentally altering the optimization landscape and preventing feature collapse to a single point.
- **Orthogonality of training-time and post-hoc methods**: As a training-time method, ELogitNorm consistently benefits all post-hoc methods (MSP/GEN/ReAct/SCALE/KNN) — this composability is a key practical advantage for deployment.
- **Diagnosing feature collapse**: Singular value spectrum analysis combined with 2D feature visualization is an effective tool for assessing representation quality.

## Limitations & Future Work
- Near-OOD improvement remains limited (on the IDK dataset), a challenge shared by all training-time methods.
- Validation is limited to ResNet-18/50; modern architectures such as ViT have not been tested.
- Computing $D(z)$ involves all $c$ decision boundaries; while efficiently implemented, it scales in principle with $c$ (e.g., $c = 1000$ for ImageNet).
- Outlier synthesis methods (VOS/NPOS/Dream) represent a complementary direction; combinations with ELogitNorm remain unexplored.

## Related Work & Insights
- **vs. LogitNorm**: Both are training-time logit scaling methods, but ELogitNorm replaces the norm with decision boundary distance, resolving feature collapse, eliminating hyperparameters, and improving compatibility with more post-hoc methods.
- **vs. CIDER/NPOS**: These follow a deep metric learning + outlier synthesis paradigm (two-stage); ELogitNorm is end-to-end (one-stage) and requires no external data generation.
- **vs. SCALE**: SCALE performs poorly on CIFAR-10 (Fig. 1), whereas ELogitNorm consistently improves all settings.
- **vs. fDBD**: Both leverage decision boundary distances, but fDBD applies them to the scoring function (inference-time), while ELogitNorm applies them to the training loss (training-time).

## Relevance to My Research
- The adaptive temperature scaling framework (Eq. 9) can be generalized to other settings requiring confidence calibration (e.g., VLMs).
- The concept of feature-to-decision-boundary distance may be useful in multimodal learning (e.g., identifying boundary regions between modalities).
- The paradigm of "improving representation quality at training time → benefiting multiple post-hoc methods at inference time" is worth further attention.

## Rating
- Novelty: ⭐⭐⭐⭐ — The diagnosis of feature collapse is valuable; replacing the norm with decision boundary distance is a natural and effective idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — OpenOOD benchmark across 4 datasets, 5+ post-hoc methods, comparisons with training-time methods, calibration analysis, singular value spectrum analysis, and classification accuracy verification.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations (Prop. 1/2) are clear; the motivation figure (Fig. 2) is convincing.
- Value: ⭐⭐⭐ — OOD detection is not a core research direction, but the adaptive temperature scaling and feature collapse diagnosis ideas are useful references.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Stronger Normalization-Free Transformers](stronger_normalization-free_transformers.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](../../AAAI2026/others/certified_branch-and-bound_maxsat_solving_extended_version.md)
- [\[CVPR 2026\] Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[CVPR 2026\] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)

</div>

<!-- RELATED:END -->
