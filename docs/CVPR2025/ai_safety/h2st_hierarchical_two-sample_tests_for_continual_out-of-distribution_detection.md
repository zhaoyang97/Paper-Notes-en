---
title: >-
  [Paper Note] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection
description: >-
  [CVPR 2025][AI Safety][OOD detection] Proposes the H2ST method, which utilizes a hierarchical two-sample test framework to achieve OOD detection in continual learning. Each task corresponds to a feature-level source-target binary classifier layer that automatically determines ID/OOD through Clopper-Pearson confidence interval hypothesis testing (requiring no manual threshold), while providing task ID prediction capabilities. It outperforms MSP, Energy…
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "OOD detection"
  - "Continual learning"
  - "Two-sample test"
  - "Hierarchical architecture"
  - "Threshold-free"
date: 2026-05-08
content_hash: 025615cd475b668d
---

# H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.14832](https://arxiv.org/abs/2503.14832)  
**Code**: [https://github.com/YuhangLiuu/H2ST](https://github.com/YuhangLiuu/H2ST)  
**Area**: Others  
**Keywords**: OOD detection, Continual learning, Two-sample test, Hierarchical architecture, Threshold-free

## TL;DR
Proposes the H2ST method, which utilizes a hierarchical two-sample test framework to achieve OOD detection in continual learning. Each task corresponds to a feature-level source-target binary classifier layer that automatically determines ID/OOD through Clopper-Pearson confidence interval hypothesis testing (requiring no manual threshold), while providing task ID prediction capabilities. It outperforms MSP, Energy, and ODIN across 7 benchmarks and improves computational efficiency by $(T+1)/2$ times.

## Background & Motivation

### Background

**Background**: Task-continual learning (TIL) assumes data is within a closed-set distribution, but models in open-world environments must identify OOD samples. Existing OOD detection methods (such as MSP, Energy, and ODIN) rely on threshold selection, heavily depend on model performance, and cannot provide task-level IDs.

**Limitations of Prior Work**: (1) Threshold sensitivity: different deployment environments require different thresholds, making manual tuning infeasible; (2) Reliance on model outputs: softmax/energy scores are heavily influenced by model overfitting; (3) Binary OOD decision only: unable to identify which specific task the OOD sample belongs to.

**Key Challenge**: As tasks continuously increase in continual learning, the complexity of OOD detection and the demand for task ID identification grow synchronously, yet existing methods lack the design for an incremental OOD detection mechanism.

**Goal**: Design a threshold-free, incrementally extensible statistical detection framework that can perform both OOD detection and task ID prediction.

**Key Insight**: Replace score-based threshold determination with two-sample tests (comparing whether the feature distribution of test samples matches that of training data) and use a hierarchical architecture to achieve incremental scalability and early-exit mechanisms.

**Core Idea**: Use hierarchically cascaded two-sample test layers for feature-level OOD detection. Each layer corresponds to one task, automatically determining ID/OOD via Clopper-Pearson confidence intervals (without thresholds), where the early-exit mechanism simultaneously outputs the task ID.

## Method

### Overall Architecture
$T$ tasks correspond to $T$ cascaded two-sample test layers. A test sample starts at the first layer; if accepted as ID, the model outputs the prediction for Task 1 and exits. Otherwise, it is passed to the second layer to continue the test, and so forth, until a layer accepts it or all layers reject it (classified as OOD). Each layer consists of a source-target binary classifier and a Clopper-Pearson hypothesis test.

### Key Designs

1. **Feature-level Two-Sample Test**:

    - **Function**: Match and determine distribution consistency within the feature space rather than the output space.
    - **Mechanism**: Train a binary classifier $h(\psi(x))$ for each task to distinguish "training data of this task" from "other data". Estimate online testing performance using a sliding window accuracy $\hat{\mu}_{w,\tau,j}$, and determine whether it is significantly higher than 50% (random guess) via Clopper-Pearson confidence intervals. If the lower bound of the confidence interval is $> 0.5$, the sample is accepted as ID for this task.
    - **Design Motivation**: The feature level is more robust than the output level—softmax outputs are often unreliable due to overfitting or poor calibration, whereas intermediate features $\psi(x)$ are more stable.

2. **Hierarchical Early-Exit Architecture**:

    - **Function**: Simultaneously achieve OOD detection and task ID prediction.
    - **Mechanism**: The $T$ test layers are ordered sequentially by tasks. Samples are tested layer-by-layer starting from the first layer. Once a layer is determined as ID, the prediction for that task is outputted and the process exits. If all layers reject the sample, it is classified as OOD. The expected number of tests is $(T+1)/2$ (vs. $T$ times in standard C2ST).
    - **Design Motivation**: Traditional OOD methods require running detection for all $T$ tasks ($O(T)$). The hierarchical architecture half-splits the average complexity by utilizing sequential early exit.

3. **Clopper-Pearson Calibrated Statistical Test**:

    - **Function**: Threshold-free automatic determination of ID/OOD.
    - **Mechanism**: Use Clopper-Pearson exact binomial confidence intervals to replace fixed thresholds. Given the detection accuracy $\hat{\mu}$ over $w$ samples in a window, calculate the lower bound $\underline{\mu}$ at confidence level $1-\alpha$. If $\underline{\mu} > 0.5$, it is determined as ID (the detector accuracy is significantly higher than random guessing).
    - **Design Motivation**: Fixed thresholds need to be readjusted across different datasets/models; interval-based testing automatically adapts and provides statistical guarantees.

### Loss & Training
Each test layer trains a simple binary classifier (distinguishing source/target features) using standard cross-entropy. It is compatible with TIL methods like Experience Replay. The window size $w$ and significance level $\alpha$ serve as the main hyperparameters.

## Key Experimental Results

### Main Results (7 benchmarks, F1 and Task Accuracy)
- Evaluated on MNIST, SVHN, CIFAR-10/100, Mini-ImageNet, CoRe50, and Stream-51.
- H2ST outperforms MSP, Energy, ODIN, and MaxLogit in terms of F1 and Task Accuracy.
- Computational efficiency: H2ST has an expected $(T+1)/2$ tests, compared to $T$ tests for C2ST.

### Ablation Study

| Config | Description |
|------|------|
| Hierarchical vs. Non-hierarchical | Hierarchical structure maintains detection performance while reducing computational overhead |
| Feature-level vs. Output-level | Feature-level is more robust and does not rely on softmax calibration |
| Threshold-free (CP interval) vs. Fixed threshold | CP interval eliminates the need for parameter tuning |
| Impact of window size $w$ | Too small causes high variance, too large results in slow response |

### Key Findings
- The threshold-free design demonstrates stable performance across multiple datasets, eliminating the hassle of tuning thresholds for each individual scenario.
- Feature-level detection is robust against model overfitting, showing clearer advantages when general model performance is poor.
- Hierarchical early-exit reduces average detection complexity by half while providing task-ID capability that traditional methods lack.

## Highlights & Insights
- **Threshold-free OOD detection**: Replaces manual thresholds with statistical hypothesis testing, which is crucial for safety-critical applications (e.g., medical, autonomous driving).
- **Dual capabilities of detection + identification**: Not only decides OOD but also identifies which task the ID sample belongs to—a two-in-one capability.
- **Incremental scalability**: When a new task arrives, it only requires adding a new test layer without affecting the existing ones.

## Limitations & Future Work
- Hierarchical ordering may affect performance—if a task's OOD samples are similar to another task's ID samples, an improper layer order can lead to misclassifications.
- The sliding window mechanism requires streaming arrival of test samples, making it unsuitable for single-image OOD detection.
- The selection of $\alpha$ and $w$ still requires some domain knowledge.

## Related Work & Insights
- **vs. MSP/Energy/ODIN**: These methods rely on thresholds and model outputs; H2ST conducts statistical tests in the feature space, which is more robust and threshold-free.
- **vs. C2ST**: Standard two-sample testing requires running detection for all $T$ tasks; H2ST's hierarchical design cuts complexity in half.
- **vs. OSDN**: OSDN models logits with a Weibull distribution; H2ST is based on non-parametric hypothesis testing, involving weaker and more general assumptions.

## Rating
- Novelty: ⭐⭐⭐⭐ Combinatorial innovation of hierarchical two-sample testing and threshold-free design
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 datasets, multiple baselines, and statistical guarantees
- Writing Quality: ⭐⭐⭐⭐ Clear description of the framework and a solid theoretical foundation for the statistical methods
- Value: ⭐⭐⭐⭐ Highly significant for safe deployment in continual learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[CVPR 2025\] Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)

</div>

<!-- RELATED:END -->
