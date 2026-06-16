---
title: >-
  [Paper Note] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning
description: >-
  [CVPR 2026][Self-Supervised Learning][Class-Incremental Learning] This work identifies Temporal Imbalance as a neglected source of bias in Class-Incremental Learning (CIL) and proposes the Temporal-Adjusted Loss (TAL). By utilizing a time-decay memory kernel to dynamically reduce the weight of negative supervision for old classes, TAL significantly alleviates catastrophic forgetting
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Class-Incremental Learning
  - catastrophic forgetting
  - Temporal Imbalance
  - Loss Reweighting
  - continual learning
date: 2026-05-08
content_hash: 08870d8fa8df3203
---
# Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning

**Conference**: CVPR2026  
**arXiv**: [2603.02280](https://arxiv.org/abs/2603.02280)  
**Code**: TBD  
**Area**: Self-supervised  
**Keywords**: Class-Incremental Learning, catastrophic forgetting, Temporal Imbalance, Loss Reweighting, continual learning

## TL;DR

This work identifies Temporal Imbalance as a neglected source of bias in Class-Incremental Learning (CIL) and proposes the Temporal-Adjusted Loss (TAL). By utilizing a time-decay memory kernel to dynamically reduce the weight of negative supervision for old classes, TAL significantly alleviates catastrophic forgetting in a plug-and-play manner.

## Background & Motivation

**Key Challenge of Class-Incremental Learning (CIL)**: Models must learn new classes sequentially without access to old class data, leading to catastrophic forgetting where model predictions are heavily biased toward new classes.

**Limitations of Prior Work**: Mainstream methods (Balanced Fine-tuning, Prototype-based Classifier, Output Layer Calibration) attribute prediction bias solely to the class imbalance between old and new classes, applying corrections only at the classification head level.

**Neglect of Temporal Imbalance**: Even if the number of samples across old classes is identical, the different arrival times of positive samples cause earlier classes to accumulate more negative supervision during later training stages. This results in an asymmetric phenomenon of high precision but low recall.

**Key Challenge (Temporal Bias)**: The chronological order of training data in CIL introduces systematic bias that is not limited to the classification head but affects the entire model, including the backbone feature space.

**Key Insight from Figure 1**: In Task 2, Class A and Class B have the same number of samples, but Class A's positive samples were concentrated in Task 0, while Class B's were in Task 1. Class A suffers more severe forgetting, proving that class balance alone cannot explain all biases.

**Lack of Loss-level Temporal Modeling**: Although temporal decay is widely used in time-series forecasting, reinforcement learning, and online learning, no prior work in CIL has explicitly modeled the temporal imbalance of positive and negative supervision at the loss function level.

## Method

### Overall Architecture

This work addresses a neglected bias in CIL: old classes are forgotten not just because samples become scarce, but because they "appear early," being repeatedly suppressed as negative samples in every subsequent task, eventually falling into a state of high precision and low recall. The logic of TAL centers on this: first, a scalar $Q_k[N]$ tracks the "recent degree of being a positive sample" online for each class using an exponential decay memory kernel. This scalar is then integrated into the denominator of the Cross-Entropy loss to dynamically down-weight negative supervision based on its magnitude. Negative supervision for old classes is relaxed, while it remains standard for new classes. This process modifies only the loss function without changing the model architecture.

### Key Designs

**1. Temporal Positive Supervision Intensity $Q_k[N]$: Tracking the "Recent Positive" Degree via Exponential Decay**

To quantify the phenomenon where "earlier classes accumulate more negative supervision," the authors maintain a scalar state $Q_k[N]$ for each class $k$. The supervision polarity at each step is denoted as a sequence $a_k[n] \in \{+1, -1\}$ (+1 if the sample belongs to class $k$, otherwise -1). An exponential decay kernel $f[n] = \lambda^{n+1}$ ($0 < \lambda < 1$, where $\lambda$ is the memory parameter) is used for a weighted convolution of history: $Q_k[N] = \sum_{n=0}^{N-1} f[N-1-n] \, a_k[n]$. More recent supervision carries higher weight. It features a recursive form $Q_k[N+1] = \lambda(Q_k[N] + a_k[N])$, requiring only $\mathcal{O}(1)$ time and space for online updates, with an upper bound $Q_{\max} = \lambda / (1-\lambda)$. For an old class that hasn't seen positive samples for a long time, $Q$ decays toward 0; for a new class currently being learned, $Q$ stays near $Q_{\max}$. This scalar compresses "temporal freshness" into a usable signal.

**2. Temporal Imbalance Theorem: Earlier Classes Inevitably Receive More Negative Supervision**

With $Q$ as a metric, the authors prove that for two classes with the same total sample count, the one whose positive samples appear earlier will have a smaller $Q$ value at the end of training. A small $Q$ implies the class has recently been suppressed almost exclusively as a negative sample, leading to "high precision, low recall"—the model hesitates to predict the class, but is usually correct when it does, while missing many instances. This explains the anomaly in Figure 1, where Class A and B have the same sample size, but A suffers more forgetting because its positive samples appeared earlier in Task 0.

**3. Temporal-Adjusted Loss: Dynamic Down-weighting of Negative Supervision**

Since the problem stems from excessive suppression of old classes by negative supervision, TAL applies a weight to each negative class in the Cross-Entropy denominator that varies with $Q$:

$$\ell_{\text{TAL}}(y, z, Q[N]) = -\log \frac{e^{z_y}}{e^{z_y} + \alpha \sum_{k \neq y} w(Q_k[N]) \cdot e^{z_k}}$$

The weighting function is $w(Q_k[N]) = (Q_k[N] / Q_{\max})^r$. When $Q_k$ is small (old class lacking recent positive supervision), $w \to 0$, nearly erasing the negative supervision for that class in the denominator. When $Q_k$ is near $Q_{\max}$ (new class with sufficient positive supervision), $w \to 1$, retaining full negative supervision. The exponent $r > 0$ controls the steepness of the curve. Unlike methods restricted to head calibration, TAL modifies the gradient source during training, transmitting protection to the backbone feature space.

**4. Frequency Alignment Parameter $\alpha$: Ensuring Consistency with Standard CE**

Down-weighting negative supervision changes the overall scale of the loss. $\alpha$ is an alignment term designed to eliminate this scale drift so that TAL matches CE in an ideal, temporally uniform, and class-balanced scenario (prior $p=1/C$). The authors require the weight coefficient for each negative class to be 1 in such scenarios: $\alpha \cdot w(Q^{*}) = 1$, where $Q^{*}$ is the steady-state $Q$ under balanced conditions. Letting $x^{*} = Q^{*} / Q_{\max}$, it is the unique solution to $(1 - \frac{1}{C})(x^{*})^r + x^{*} - \frac{1}{C} = 0$ on $(0, 1)$, resulting in $\alpha = 1/(x^{*})^r$. Thus, $\alpha$ is determined solely by the number of classes $C$ and the exponent $r$, introducing no additional tunable hyperparameters.

### Loss & Training

TAL introduces only two primary hyperparameters: the decay rate $\lambda$ and the weighting steepness $r$. To maintain consistency, negative supervision used to update $Q$ is also multiplied by $w(Q_k)$. The method does not change the model architecture and is equivalent to replacing the CE loss in existing CIL frameworks, allowing it to be integrated into baselines like iCaRL, DER, or TagFex.

## Main Results

**Consistent Improvements across Datasets and Baselines**

| Method | CIFAR-100 10-task $A_{\text{Mean}}$ | CIFAR-100 10-task $A_{\text{Last}}$ | ImageNet-100 10-task $A_{\text{Mean}}$ | ImageNet-100 10-task $A_{\text{Last}}$ |
|------|------|------|------|------|
| iCaRL | 58.76 | 45.39 | 43.71 | 24.38 |
| iCaRL + TAL | **60.82** | **47.36** | **52.19** | **32.78** |
| DER | 63.53 | 50.75 | 52.25 | 40.28 |
| DER + TAL | **66.33** | **53.82** | **54.57** | **42.62** |
| TagFex | 65.97 | 55.99 | 54.73 | 41.70 |
| TagFex + TAL | **68.68** | **57.91** | **57.05** | **43.01** |

Under 10-task and 20-task settings on CIFAR-100, ImageNet-100, and Food101, TAL brings consistent and significant improvements to all five baselines (iCaRL, FOSTER, MEMO, DER, TagFex).

## Ablation Study

| $r$ \ $\lambda$ | 0.99 $A_{\text{Mean}}$ | 0.995 $A_{\text{Mean}}$ | 0.999 $A_{\text{Mean}}$ |
|------|------|------|------|
| 0.5 | 62.12 | 61.27 | 61.72 |
| 1.0 | 62.60 | **63.36** | 62.46 |
| 2.0 | 62.51 | 60.24 | 62.85 |
| CE baseline | 59.96 | - | - |

- The optimal hyperparameter combination is $\lambda = 0.995, r = 1.0$.
- TAL outperforms CE across a wide range of hyperparameters, demonstrating robustness.
- Excessively large $r$ (e.g., 5.0) overly inhibits information from new classes, leading to performance degradation.

## Key Findings

1.  **Universal Precision-Recall Asymmetry**: In methods like iCaRL, DER, MEMO, and TagFex, earlier classes consistently show a high precision and low recall pattern.
2.  **TAL Affects Feature Space**: UMAP visualizations reveal that TAL alleviates the encroachment of new class features onto old class spaces, indicating its effect is not limited to the classification head.
3.  **Non-uniform Protection**: TAL provides varying levels of protection to different old classes; relatively "newer" old classes might even be slightly suppressed compared to very "old" ones.
4.  **Minimal Computational Overhead**: The additional overhead is only approximately 0.8%, thanks to the $\mathcal{O}(1)$ recursive updates of $Q$.

## Highlights & Insights

-   **Novel Perspective**: Formally analyzes prediction bias in CIL from a temporal imbalance perspective for the first time, distinguishing it from traditional class imbalance explanations.
-   **Theoretical Soundness**: Establishes a temporal supervision model and proves the Temporal Imbalance Theorem and the property that TAL reduces to CE under balanced conditions.
-   **Plug-and-play**: Modifies only the loss function without architectural changes, enabling seamless integration into any CIL method.
-   **Global Effect**: Not only corrects the classification head but also improves the inter-class distribution in the backbone feature space.
-   **Extensive Verification**: Consistent effectiveness across 5 baselines, 3 datasets, and 2 task settings.

## Limitations & Future Work

-   The decay kernel is fixed as an exponential form; in practice, $\lambda$ might need to vary with task stages.
-   Exponential decay approaches zero over time, potentially failing to fully capture the sustained impact of learned representations.
-   Non-parametric or adaptive forms of temporal modeling have not been explored.
-   Verification has been conducted on medium-scale datasets (100 classes); large-scale scenarios remains to be tested.

## Related Work & Insights

-   **CIL Prediction Bias Correction**: Balanced Fine-tuning [49], iCaRL prototype classifier [29], Weight Alignment [50]—all are limited to classification head corrections.
-   **Temporal Modeling**: Exponential smoothing (Holt-Winters), Eligibility Traces / TD(λ) in reinforcement learning [37], ADWIN in online learning [4]—these share the intuition that "recent data should weigh more," but have not been used for CIL loss design.
-   **Dynamic Architecture Methods**: DER [50], MEMO [55], TagFex [52]—these expand network capacity for new tasks; TAL can serve as an orthogonal supplement.

## Rating

-   Novelty: ⭐⭐⭐⭐ — The temporal imbalance perspective is novel and theoretically supported.
-   Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive verification across multiple baselines and datasets.
-   Writing Quality: ⭐⭐⭐⭐ — Clear logic from problem definition to methodological derivation.
-   Value: ⭐⭐⭐⭐ — High practical utility due to its plug-and-play nature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[ECCV 2024\] STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](../../ECCV2024/self_supervised/stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)
- [\[CVPR 2026\] Beyond Myopic Alignment: Lookahead Optimization for Online Class-Incremental Learning](beyond_myopic_alignment_lookahead_optimization_for_online_class-incremental_lear.md)
- [\[CVPR 2026\] Geometry-driven OOD Detectors Are Class-Incremental Learners](geometry-driven_ood_detectors_are_class-incremental_learners.md)
- [\[CVPR 2026\] Semantic-Guided Global-Local Collaborative Prompt Learning for Few-Shot Class Incremental Learning](semantic-guided_global-local_collaborative_prompt_learning_for_few-shot_class_in.md)

</div>

<!-- RELATED:END -->
