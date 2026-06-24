---
title: >-
  [Paper Note] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach
description: >-
  [CVPR 2025][Active Learning] The EAOA framework is proposed, which utilizes free energy-based epistemic uncertainty (EU) and aleatoric uncertainty (AU) metrics, combined with an adaptive coarse-to-fine query strategy, to effectively select samples that are both known-class and highly informative in active open-set annotation scenarios.
tags:
  - "CVPR 2025"
  - "Active Learning"
  - "Open-Set Annotation"
  - "Epistemic Uncertainty"
  - "Aleatoric Uncertainty"
  - "Energy-Based Models"
date: 2026-05-08
content_hash: a606413a40a90c13
---

# Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach

**Conference**: CVPR 2025  
**arXiv**: [2502.19691](https://arxiv.org/abs/2502.19691)  
**Code**: [GitHub](https://github.com/)  
**Area**: Other  
**Keywords**: Active Learning, Open-Set Annotation, Epistemic Uncertainty, Aleatoric Uncertainty, Energy-Based Models

## TL;DR

The EAOA framework is proposed, which utilizes free energy-based epistemic uncertainty (EU) and aleatoric uncertainty (AU) metrics, combined with an adaptive coarse-to-fine query strategy, to effectively select samples that are both known-class and highly informative in active open-set annotation scenarios.

## Background & Motivation

Active Learning (AL) reduces annotation costs by iteratively selecting the most informative samples for labeling, but faces severe challenges in open-set scenarios where the unlabeled pool contains unknown-class samples.

Existing methods suffer from two fundamental problems:

1. **Focusing only on low epistemic uncertainty (EU)**: Methods like LfOSA and EOAL prioritize samples that are likely to belong to known classes. Although this achieves high query precision, the selected samples provide limited information, leading to suboptimal model performance.
2. **Focusing only on high aleatoric uncertainty (AU)**: Methods like BUAL prioritize samples with highly uncertain predictions. However, measuring AU for unknown-class samples is inherently meaningless (as AU is only valid within a closed-set context).

The authors empirically verify that using either EU or AU in isolation yields poor performance, but effectively combining them significantly boosts results. The core idea is to first filter a candidate set that is highly likely to belong to known classes using EU (ensuring the closed-set property) and then select the most informative samples from this candidate set using AU.

This insight is simple yet profound: the semantic premise of AU is that the sample belongs to the known-class distribution; only after filtering with EU does AU make sense.

## Method

### Overall Architecture

EAOA maintains two networks: a $(C+1)$-class detector (for EU assessment) and a $C$-class target classifier (for AU assessment), employing an adaptive coarse-to-fine two-stage query strategy.

### Key Designs

**Design 1: Energy-based Epistemic Uncertainty (Energy-based EU)**

- **Function**: To reliably evaluate whether a sample belongs to a known class from both learning and data-driven perspectives.
- **Mechanism**: EU is defined as the difference between the free energy score of known classes and that of unknown classes: $EU(x) = E_{kno}(x) - E_{unk}(x)$. The learning perspective is directly predicted by the detector. The data-driven perspective constructs a probability distribution through K-nearest neighbors (KNN) arrow voting. Both are converted to probabilities using GMMs and then fused by element-wise multiplication.
- **Design Motivation**: A single free energy is insufficient to measure EU because it does not utilize the information of annotated unknown-class samples. By employing a $(C+1)$-class detector and dual-perspective fusion, reliable EU estimation is achieved in data-scarce AL scenarios.

$$EU(x) = -\log\sum_{c=1}^{C} e^{f_c(x)} + \log(1 + e^{f_{C+1}(x)})$$

**Design 2: Energy-based Aleatoric Uncertainty (Energy-based AU)**

- **Function**: To evaluate the degree of confusion among known classes for a sample (boundary samples are more informative).
- **Mechanism**: AU is defined as the difference between the free energy score of all classes and that of the minor classes (excluding the most likely class), reflecting how close a sample is to the decision boundary.
- **Design Motivation**: Traditional entropy metrics are invalid in open-set scenarios because unknown-class samples can also yield high entropy. By using the free energy difference, AU is utilized effectively only within the closed-set candidates.

$$AU(x) = -\log\sum_{c=1}^{C} e^{f_c(x)} + \log\left[\sum_{c=1}^{C} e^{f_c(x)} - e^{f_{y_{max}}(x)}\right]$$

**Design 3: Target-Driven Adaptive Sampling Strategy**

- **Function**: Automatically adjusting the candidate set size to balance EU filtering and AU selection.
- **Mechanism**: In each round, $kb$ samples with low EU are first selected to form a candidate set, from which $b$ samples with high AU are then chosen as the query set. The value of $k$ is adaptively adjusted by comparing the actual query precision $rP$ with the target precision $tP$.
- **Design Motivation**: If $k$ is too small, AU cannot take effect; if $k$ is too large, the closed-set assumption is violated. The adaptive strategy avoids manual parameter tuning and enhances generalization across datasets.

### Loss & Training

The detector is trained using both cross-entropy loss and margin-based energy loss:

$$\mathcal{L}_{detector} = \mathcal{L}_{ce} + \lambda_e \mathcal{L}_{energy}$$

The energy loss maximizes $E_{kno}$ for known-class samples and minimizes $E_{kno}$ for unknown-class samples, with the separation controlled by margin parameters $m_{kno}$ and $m_{unk}$.

## Key Experimental Results

### Main Results: CIFAR-100 (mismatch ratio=40%)

| Method | Final Test Accuracy | Average Query Precision |
|------|-------------|-------------|
| Random | ~68% | ~60% |
| CCAL | ~70% | ~72% |
| LfOSA | ~72% | ~82% |
| EOAL | ~73% | ~85% |
| BUAL | ~72% | ~70% |
| **EAOA (Ours)** | **~76%** | **~83%** |

### Cross-Dataset Consistency

| Dataset | Mismatch Ratio | EAOA Advantage |
|--------|---------------|---------|
| CIFAR-10 | 20%/30%/40% | Optimal in all rounds |
| CIFAR-100 | 20%/30%/40% | Optimal in all rounds |
| Tiny-ImageNet | 10%/15%/20% | Optimal in all rounds |

### Ablation Study

| Component | Test Accuracy | Average Query Precision |
|------|---------|-------------|
| Free Energy (EU) only | ~73% | ~84% |
| + Data-driven EU | ~74% | ~85% |
| + Energy Loss | ~74.5% | ~86% |
| + AU (Full EAOA) | **~76%** | **~83%** |

### Key Findings

1. Selecting based on low EU (Certainty) alone achieves performance close to Random, proving the necessity of high AU under the closed-set assumption.
2. Fusing EU and AU significantly improves performance, validating the core assumption of "ensuring closed-set first, then assessing informativeness."
3. EAOA has the shortest training time among all AOSA methods while maintaining the highest accuracy.
4. The parameters show strong robustness, with very small performance fluctuations when the target precision $tP$ varies in the range of 0.4-0.8.

## Highlights & Insights

1. **Re-understanding open-set AL from the perspective of uncertainty decomposition**: This work clearly points out the different roles of EU and AU in open-set scenarios, and explains why they need to be cascaded rather than simply blended.
2. **Unification of the energy framework**: Both EU and AU are derived from free energy theory, providing an elegant and consistent theoretical foundation.
3. **Dual-perspective EU estimation**: Fusing learning and data-driven perspectives is particularly suitable for data-scarce scenarios in AL.

## Limitations & Future Work

1. The experiments are only validated on image classification tasks, without expansion to more complex tasks such as detection or segmentation.
2. It assumes that unknown-class data can be unified into a single category, which may perform poorly when the intra-class variance of the unknown classes is extremely large.
3. GMM fitting is a heuristic choice; whether there is a better probability transformation method remains to be explored.
4. Extending the framework to more practical streaming settings can be considered.

## Related Work & Insights

- **LfOSA**: First introduced the $(C+1)$-class detector framework, upon which this paper improves the EU metric.
- **EOAL**: Proposed using entropy in the detector to distinguish between known/unknown classes, but ignored the role of AU.
- **Energy-Based Models (EBMs)**: The idea of using free energy as an OOD detection metric originates from Liu et al. (NeurIPS 2020).
- **Insight**: In other scenarios involving data selection (such as curriculum learning or data cleaning), the decomposition strategy of EU/AU might be equally valuable.

## Rating

⭐⭐⭐⭐ — Clear theoretical motivation, rigorous methodological derivation, and comprehensive experiments. Unifying the analysis of EU and AU into a free energy framework provides both theoretical depth and practical efficiency. A limitation is the relatively simple experimental tasks (only classification), lacking validation on complex tasks such as object detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](../../ICML2025/others/rethinking_aleatoric_and_epistemic_uncertainty.md)
- [\[ECCV 2024\] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](../../ECCV2024/others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)
- [\[CVPR 2025\] Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](instance-wise_supervision-level_optimization_in_active_learning.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
