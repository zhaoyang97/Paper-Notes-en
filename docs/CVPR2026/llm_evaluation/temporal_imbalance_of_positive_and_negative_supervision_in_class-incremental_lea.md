---
title: >-
  [Paper Note] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning
description: >-
  [CVPR 2026][LLM Evaluation][Class-Incremental Learning] This paper identifies temporal imbalance as a previously overlooked source of bias in class-incremental learning (CIL) and proposes the Temporal-Adjusted Loss (TAL)…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Class-Incremental Learning"
  - "catastrophic forgetting"
  - "Temporal Imbalance"
  - "Loss Reweighting"
  - "continual learning"
date: 2026-05-08
content_hash: b58ba16a0fa982cc
---

# Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning

**Conference**: CVPR 2026
**arXiv**: [2603.02280](https://arxiv.org/abs/2603.02280)
**Code**: To be confirmed
**Area**: LLM Evaluation
**Keywords**: Class-Incremental Learning, catastrophic forgetting, Temporal Imbalance, Loss Reweighting, continual learning

## TL;DR

This paper identifies temporal imbalance as a previously overlooked source of bias in class-incremental learning (CIL) and proposes the Temporal-Adjusted Loss (TAL), which dynamically downweights negative supervision for old classes via a temporally decaying memory kernel. TAL integrates in a plug-and-play manner and significantly alleviates catastrophic forgetting.

## Background & Motivation

**Core challenge of CIL**: Models must learn new classes sequentially without access to data from old classes, leading to catastrophic forgetting and severe prediction bias toward new classes.

**Background**: Mainstream methods (Balanced Fine-tuning, Prototype-based Classifiers, Output Layer Calibration) attribute prediction bias to class imbalance between old and new classes, applying corrections only at the classification head.

**Limitations of Prior Work**: Even when old classes have equal numbers of samples, classes whose positive samples appeared earlier accumulate more negative supervision during later training stages, yielding an asymmetry of high precision but low recall.

**Key Challenge**: The temporal ordering of training data in CIL introduces systematic bias that affects the entire model—including the backbone feature space—not merely the classification head.

**Key illustration (Figure 1)**: In Task 2, classes A and B have identical sample counts, but class A's positive samples are concentrated in Task 0 while class B's are in Task 1. Class A suffers more severe forgetting, demonstrating that class balance alone cannot explain all prediction bias.

**Goal**: Although temporal decay is widely used in time-series forecasting, reinforcement learning, and online learning, no prior CIL work explicitly models the temporal imbalance of positive and negative supervision at the loss level.

## Method

### Overall Architecture

The core mechanism of TAL: for each class $k$, a temporal positive supervision intensity $Q_k[N]$ is maintained and updated via an exponentially decaying memory kernel that tracks the recent positive supervision status of that class. The cross-entropy loss is then dynamically reweighted according to $Q_k[N]$ to suppress negative supervision for old classes, preventing excessive inhibition.

### Key Designs

**1. Temporal Positive Supervision Intensity $Q_k[N]$**

- For each class $k$, a supervision polarity sequence $a_k[n] \in \{+1, -1\}$ is defined (positive sample: $+1$, negative sample: $-1$).
- An exponentially decaying memory kernel $f[n] = \lambda^{n+1}$ ($0 < \lambda < 1$) is introduced, where $\lambda$ is the memory parameter.
- $Q_k[N] = \sum_{n=0}^{N-1} f[N-1-n] \cdot a_k[n]$, i.e., the discrete convolution of the decay kernel with the supervision sequence.
- Recursive form: $Q_k[N+1] = \lambda(Q_k[N] + a_k[N])$, with both time and space complexity of $\mathcal{O}(1)$.
- Upper bound: $Q_{\max} = \lambda / (1-\lambda)$.

**2. Temporal Imbalance Theorem (Theorem 1)**

Given two classes with identical total sample counts, the class whose positive samples appear earlier will have a smaller $Q$ value at the end of training, meaning it is subject to greater negative supervision pressure, manifesting as high precision and low recall.

**3. Temporal-Adjusted Loss**

$$\ell_{\text{TAL}}(y, z, Q[N]) = -\log \frac{e^{z_y}}{e^{z_y} + \alpha \sum_{k \neq y} w(Q_k[N]) \cdot e^{z_k}}$$

where the weight function $w(Q_k[N]) = (Q_k[N] / Q_{\max})^r$:

- Small $Q_k$ (old class lacking recent positive supervision) → small $w$ → negative supervision is suppressed.
- $Q_k$ near $Q_{\max}$ (new class with sufficient positive supervision) → $w$ near 1 → full negative supervision retained.
- Exponent $r > 0$ controls the sharpness of the weight function.

**4. Frequency Alignment Parameter $\alpha$**

- Ensures TAL reduces to standard cross-entropy under temporally uniform and class-balanced conditions.
- Uniquely determined by the number of classes $C$ and exponent $r$, requiring no additional tuning.

### Loss Function Properties

- Only two hyperparameters: $\lambda$ (decay rate) and $r$ (weight sharpness).
- Consistency of $Q$ updates: negative supervision updates are also scaled by $w(Q_k)$.
- Plug-and-play: no architectural modifications required; TAL can directly replace CE loss in existing CIL frameworks.

## Key Experimental Results

### Main Results: Consistent Improvement Across Datasets and Baselines

| Method | CIFAR-100 10-task $A_{\text{Mean}}$ | CIFAR-100 10-task $A_{\text{Last}}$ | ImageNet-100 10-task $A_{\text{Mean}}$ | ImageNet-100 10-task $A_{\text{Last}}$ |
|------|------|------|------|------|
| iCaRL | 58.76 | 45.39 | 43.71 | 24.38 |
| iCaRL + TAL | **60.82** | **47.36** | **52.19** | **32.78** |
| DER | 63.53 | 50.75 | 52.25 | 40.28 |
| DER + TAL | **66.33** | **53.82** | **54.57** | **42.62** |
| TagFex | 65.97 | 55.99 | 54.73 | 41.70 |
| TagFex + TAL | **68.68** | **57.91** | **57.05** | **43.01** |

Under 10-task and 20-task settings on CIFAR-100, ImageNet-100, and Food101, TAL yields consistent and substantial improvements over all five baselines (iCaRL, FOSTER, MEMO, DER, TagFex).

### Ablation Study

| $r$ \ $\lambda$ | 0.99 $A_{\text{Mean}}$ | 0.995 $A_{\text{Mean}}$ | 0.999 $A_{\text{Mean}}$ |
|------|------|------|------|
| 0.5 | 62.12 | 61.27 | 61.72 |
| 1.0 | 62.60 | **63.36** | 62.46 |
| 2.0 | 62.51 | 60.24 | 62.85 |
| CE baseline | 59.96 | - | - |

- The optimal hyperparameter combination is $\lambda = 0.995, r = 1.0$.
- TAL consistently outperforms CE across a wide range of hyperparameter settings, demonstrating robustness to hyperparameter selection.
- Excessively large $r$ (e.g., 5.0) over-suppresses new classes and degrades performance.

### Key Findings

1. **Precision–recall asymmetry is pervasive**: Early-arriving classes consistently exhibit high precision and low recall across iCaRL, DER, MEMO, and TagFex (Figure 2c).
2. **TAL affects the feature space**: UMAP visualizations show that TAL alleviates the encroachment of new class features on old class regions, indicating that its effects extend beyond the classification head (Figure 5).
3. **TAL provides non-uniform protection**: Different old classes receive different degrees of protection; more recently learned old classes may even be mildly suppressed (Figure 7).
4. **Negligible computational overhead**: The additional cost is approximately 0.8%, enabled by the $\mathcal{O}(1)$ recursive update of $Q$.

## Highlights & Insights

- **Novel perspective**: This is the first work to formally analyze prediction bias in CIL through the lens of temporal imbalance, distinguishing it from the conventional class imbalance explanation.
- **Solid theoretical foundation**: A temporal supervision model is established, with proofs of the temporal imbalance theorem and the reduction of TAL to standard CE under balanced conditions.
- **Plug-and-play design**: Only the loss function is modified; the architecture remains unchanged, allowing seamless integration into any CIL framework.
- **Global effect**: TAL not only corrects the classification head but also improves inter-class distributions in the backbone feature space.
- **Extensive validation**: Evaluated across 5 baselines × 3 datasets × 2 task settings, with consistent effectiveness throughout.

## Limitations & Future Work

- The decay kernel is fixed as an exponential form; in practice, $\lambda$ may need to vary across task stages.
- Exponential decay trends toward zero over time, potentially failing to fully capture the lasting influence of previously learned representations.
- Non-parametric or adaptive temporal modeling forms are not explored.
- Experiments are conducted only on medium-scale datasets (100 classes); large-scale scenarios remain to be validated.

## Related Work & Insights

- **CIL prediction bias correction**: Balanced Fine-tuning [49], iCaRL prototype classifier [29], Weight Alignment [50]—all limited to classification head corrections.
- **Temporal modeling**: Exponential smoothing (Holt-Winters), Eligibility Traces / TD($\lambda$) in reinforcement learning [37], ADWIN in online learning [4]—all share the intuition that recent data should carry greater weight, but none are applied to CIL loss design.
- **Dynamic architecture methods**: DER [50], MEMO [55], TagFex [52]—expand network capacity to accommodate new tasks; TAL serves as an orthogonal complement to these approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ — The temporal imbalance perspective is original and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across multiple baselines and datasets with well-designed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Logical flow from problem formulation to method derivation, with intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Strong practical utility due to plug-and-play nature; the novel perspective can inspire future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[ICLR 2026\] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](../../ICLR2026/llm_evaluation/accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[ICLR 2026\] In-Context Learning of Temporal Point Processes with Foundation Inference Models](../../ICLR2026/llm_evaluation/in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](../../NeurIPS2025/llm_evaluation/climb_class-imbalanced_learning_benchmark_on_tabular_data.md)

</div>

<!-- RELATED:END -->
