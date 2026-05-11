---
title: >-
  [Paper Note] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms
description: >-
  [ICLR 2026][LLM Evaluation][positive-unlabeled learning] This paper proposes the first unified benchmark for PU learning and systematically addresses two critical issues: (1) enabling model selection without negative sam…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "positive-unlabeled learning"
  - "benchmark"
  - "model selection"
  - "label shift"
  - "fair evaluation"
date: 2026-05-08
content_hash: 9cf9b9c2929d26c4
---

# Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms

**Conference**: ICLR 2026
**arXiv**: [2509.24228](https://arxiv.org/abs/2509.24228)
**Code**: Available (benchmark)
**Area**: Weakly Supervised Learning
**Keywords**: positive-unlabeled learning, benchmark, model selection, label shift, fair evaluation

## TL;DR
This paper proposes the first unified benchmark for PU learning and systematically addresses two critical issues: (1) enabling model selection without negative samples via proxy accuracy and proxy AUC; (2) identifying and resolving intra-dataset label shift in the one-sample setting through a simple calibration strategy that merges positive samples into the unlabeled set, enabling fair comparison of two-sample algorithms under one-sample evaluation.

## Background & Motivation

**Background**: PU learning (binary classification with only positive and unlabeled samples) has seen a proliferation of algorithms in recent years, yet experimental setups remain highly inconsistent, making it difficult to determine which algorithm is superior.

**Limitations of Prior Work**: (a) Many algorithms rely on validation sets containing negative samples for model selection, which contradicts the fundamental PU learning assumption of no access to negatives; (b) PU learning encompasses one-sample (OS) and two-sample (TS) settings, and existing evaluations favor the OS setting while overlooking a critical distinction — in the OS setting, the class prior and marginal distribution of unlabeled data differ.

**Key Challenge**: TS algorithms assume unlabeled data follows the marginal distribution $p(x)$, but in the OS setting the density of unlabeled data is $\bar{p}(x) = \bar{\pi}p(x|y=+1) + (1-\bar{\pi})p(x|y=-1)$, where $\bar{\pi} \neq \pi$. This breaks the risk consistency of TS algorithms.

**Goal**: To provide an accessible, realistic, and fair evaluation framework for PU learning.

**Key Insight**: (a) Deriving surrogate metrics computable solely from positive and unlabeled data via information theory; (b) Identifying the intra-dataset label shift problem and providing theoretical guarantees.

**Core Idea**: Merging positive samples into the unlabeled set restores unbiasedness of the marginal distribution, eliminating the fairness gap between OS and TS settings with a single line of code.

## Method

### Overall Architecture
The evaluation framework operates on three levels: (1) a unified data generation, training, and evaluation pipeline; (2) realistic model selection criteria requiring no negative samples; (3) fair cross-setting comparison by resolving intra-dataset label shift.

### Key Designs

1. **Proxy Accuracy (PA) and Proxy AUC (PAUC)**:

   - **Function**: Enable model selection using only positive and unlabeled validation data.
   - **Mechanism**: PA exploits the relationship $\mathbb{E}[\text{PA}(f)] = \text{ACC}(f) - 1 + 2\pi$ (requiring class prior $\pi$); PAUC treats unlabeled data as "noisy negatives" to compute AUC (not requiring $\pi$). Propositions 1–2 establish the rank-preserving property of both surrogate metrics.
   - **Design Motivation**: Existing work universally adopts oracle accuracy computed on negative-containing validation sets, which violates the PU learning premise. Surrogate metrics render evaluation genuinely unsupervised.

2. **Identification and Calibration of Intra-dataset Label Shift (ILS)**:

   - **Function**: Detect class prior shift in unlabeled data under the OS setting and propose a simple calibration method.
   - **Mechanism**: In the OS setting, after positive samples are drawn out, the positive class proportion in the unlabeled set drops from $\pi$ to $\bar{\pi} = (1-c)\pi/(1-c\pi)$. The calibration is straightforward — merge positive samples into the unlabeled set when computing the unlabeled loss: $\mathcal{D}_k^U \cup \mathcal{D}_k^P$. Theorem 1 proves the calibrated risk estimator is unbiased; Theorem 2 provides convergence guarantees.
   - **Design Motivation**: This is the first identification of the ILS problem in the literature. A one-line code change restores the performance of TS algorithms under OS evaluation.

3. **Unified Benchmark Framework**:

   - **Function**: Standardize data generation, training, and evaluation protocols.
   - **Mechanism**: Covers three major algorithm families — cost-sensitive, sample-selection, and biased PU learning — evaluated on standard datasets including CIFAR-10/100 and Fashion-MNIST.
   - **Design Motivation**: Eliminate the confounding effects of implementation details (data augmentation, warm-up, etc.) on comparisons.

### Loss & Training
Logistic loss is uniformly adopted across all algorithms. The calibration method only modifies the computation of the unlabeled loss (by incorporating positive samples) without altering the core algorithmic logic.

## Key Experimental Results

### Main Results

**CIFAR-10 PU version (varying positive sample sizes):**

| TS Algorithm | OS Setting (uncalibrated) | OS Setting (calibrated) | TS Setting |
|---|---|---|---|
| uPU | Significant degradation | ≈ TS performance | Baseline |
| nnPU | Degraded | Recovered | Baseline |
| Dist-PU | Degraded | Recovered | Baseline |
| VPU | Degraded | Recovered | Baseline |

### Ablation Study

**Comparison of model selection criteria:**

| Selection Criterion | Requires Negatives | ACC Correlation | AUC Correlation |
|---|---|---|---|
| Oracle Accuracy (OA) | ✓ | Best | Best |
| Proxy Accuracy (PA) | ✗ | Close to OA | Moderate |
| Proxy AUC (PAUC) | ✗ | Moderate | Close to OA |

### Key Findings
- No single algorithm dominates across all datasets and metrics; several early, simple methods remain highly competitive.
- TS algorithms suffer significant performance degradation in the OS setting without calibration, and recover after calibration — validating the practical impact of ILS.
- PA is better suited for accuracy-oriented model selection, while PAUC is preferable for AUC-oriented selection — different evaluation metrics call for different selection criteria.
- The calibration method is consistently effective across diverse settings and algorithms with negligible computational overhead.

## Highlights & Insights
- **Impact of a one-line fix**: The calibration method essentially amounts to including positive samples in the unlabeled loss computation. Such a minimal modification carries theoretical guarantees and yields substantial empirical gains — illustrating that understanding the problem is more important than designing elaborate methods.
- **First identification of ILS**: All prior work evaluates TS algorithms in the OS setting without calibration, propagating an unfair comparison for years. This finding has the potential to redefine evaluation standards in the field.
- **Practical value of surrogate metrics**: The paper demonstrates that effective model selection is achievable without negative samples, making PU learning evaluation genuinely self-consistent.

## Limitations & Future Work
- Proxy accuracy requires knowledge or estimation of the class prior $\pi$; inaccurate estimation may degrade model selection quality.
- The benchmark currently covers only image classification datasets; extension to modalities such as text and tabular data remains future work.
- The calibration method assumes a constant labeling probability $c$ (SCAR assumption) and does not cover non-uniform labeling scenarios.
- In deep learning settings, hyperparameter search may render the theoretical guarantees of surrogate metrics insufficiently tight under finite samples.

## Related Work & Insights
- **vs. existing PU learning papers**: Nearly all prior work uses negative-containing validation sets for early stopping and model selection; this paper identifies this practice as unrealistic and provides viable alternatives.
- **vs. weakly supervised learning benchmarks**: Analogous to CleanLab's contribution to noisy-label learning, this work provides a long-overdue standardized benchmark for PU learning.
- **vs. du Plessis et al. 2015**: The risk consistency of uPU is broken in the OS setting; the proposed calibration restores it — an important complement to the classical method.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The identification of ILS and the elegant calibration method, together with theoretically grounded surrogate metrics, constitute a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six algorithm families × multiple datasets × OS/TS settings × pre- and post-calibration comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is precise, theoretical derivations are rigorous, and the benchmark contribution is substantive.
- **Value**: ⭐⭐⭐⭐⭐ As the first PU learning benchmark, this work has the potential to transform evaluation practices across the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](../../ACL2026/llm_evaluation/subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/llm_evaluation/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [\[ICLR 2026\] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/llm_evaluation/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)

</div>

<!-- RELATED:END -->
