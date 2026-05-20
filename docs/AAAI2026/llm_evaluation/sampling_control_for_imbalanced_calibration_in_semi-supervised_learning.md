---
title: >-
  [Paper Note] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning
description: >-
  [AAAI 2026][LLM Evaluation][semi-supervised learning] This paper proposes SC-SSL, a framework that introduces an **expansion classifier** for decoupled sampling control to mitigate feature-level imbalance…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "semi-supervised learning"
  - "class imbalance"
  - "sampling control"
  - "pseudo-labels"
  - "calibration"
date: 2026-05-08
content_hash: 97b60ec2414450fd
---

# Sampling Control for Imbalanced Calibration in Semi-Supervised Learning

**Conference**: AAAI 2026
**arXiv**: [2511.18773](https://arxiv.org/abs/2511.18773)  
**Code**: [https://github.com/Sheldon04/SC-SSL](https://github.com/Sheldon04/SC-SSL)  
**Area**: Semi-Supervised Learning / Class Imbalance
**Keywords**: semi-supervised learning, class imbalance, sampling control, pseudo-labels, calibration

## TL;DR

This paper proposes SC-SSL, a framework that introduces an **expansion classifier** for decoupled sampling control to mitigate feature-level imbalance, and leverages the **bias term** of a linear layer as an optimized bias vector to directly calibrate logits at inference time, achieving state-of-the-art performance across multiple data distribution settings.

## Background & Motivation

1. **Background**: Semi-supervised learning (SSL) exploits unlabeled data via pseudo-labels and consistency regularization, but real-world data often exhibits long-tailed distributions, causing pseudo-labels to be biased toward head classes — a problem known as class-imbalanced SSL (CISSL).
2. **Limitations of Prior Work**: Existing methods (ACR, CPE, SimPro) estimate the class distribution of unlabeled data to adjust logits, but suffer from two issues: (a) coarse-grained treatment that conflates data imbalance with bias introduced by varying per-class learning difficulty; and (b) conservative strategies that leave large amounts of unlabeled data underutilized, retaining only a small number of high-quality pseudo-labels to avoid confirmation bias.
3. **Key Challenge**: Data imbalance and optimization imbalance are two independent sources of bias, yet existing methods couple them together. In dual-classifier settings, neither the output classifier nor the original classifier can effectively adjust the sampling probability of non-head classes.
4. **Goal**: Address model bias at a finer granularity — separately at the feature level and the logit level.
5. **Key Insight**: Based on the expansion-separation assumption in self-training, even noisy pseudo-labels can propagate effective supervisory signals through consistency regularization, provided that non-head classes are sampled with sufficient frequency.
6. **Core Idea**: Introduce a third expansion classifier dedicated to increasing the sampling probability of non-head classes to balance feature learning, and at inference time use the difference in bias terms to isolate and correct optimization bias.

## Method

### Overall Architecture

SC-SSL augments FixMatch with two additional classifiers: an **output classifier** $F_b$ (trained with balanced sampling, used for inference) and an **expansion classifier** $F_e$ (trained with oversampling of non-head classes to balance feature learning). All three classifiers share a backbone network $B$. During training, theoretical analysis identifies the key variables governing sampling control ($\gamma_u$, $\Delta p$, $\rho$), which are used to dynamically adjust sampling probabilities under different distribution settings. At inference, the bias term pattern of the linear layer is analyzed to define an optimization bias vector $\mathbf{b}_{opt}$ that directly calibrates logits.

### Key Designs

1. **Expansion Classifier and Sampling Control (Training Phase)**

    - **Function**: Increases the sampling probability of non-head classes through a dedicated classifier, balancing gradient contributions in feature space.
    - **Mechanism**: Based on a simplified binary classification analysis (Theorem 0.1), the sampling probability of pseudo-labels is primarily governed by three factors: the data imbalance factor $\gamma_u$, the logit adjustment $\Delta p$, and the confidence threshold $\rho$. Neither the output classifier nor the original classifier can substantially adjust $\Delta p$ or $\rho$ — the former must preserve classification accuracy, while the latter would violate the separation assumption. An expansion classifier $F_e$ is therefore introduced with $\tau_e = 4$ (larger than $\tau_b = 2$ for the output classifier), initialized with lower thresholds $\rho_e^0(\text{non-head})$ based on the expansion factor $c$, and adapted during training via $\rho^t(k) = \rho^{t-1}(k) - \alpha \cdot \mathbb{I}(\mathbf{b}_{opt}(k) > \nu)$.
    - **Design Motivation**: The expansion assumption guarantees that label information can be propagated even from a small number of confident predictions, provided non-head classes are sampled sufficiently. Existing dual-classifier settings cannot effectively increase this sampling; a dedicated third classifier is needed.

2. **Bias Term Analysis and Inference Calibration**

    - **Function**: Directly corrects optimization-induced bias at inference time.
    - **Mechanism**: Empirical observation reveals that the linear layer bias term $\mathbf{b}$ encodes two types of bias: distributional bias and optimization bias. Under random sampling, head-class bias terms are large (data bias); under expansion sampling, tail-class bias terms are large (overcorrection). Since the output classifier is trained with balanced sampling, its bias term excludes data bias and retains only optimization bias. The optimization bias vector $\mathbf{b}_{opt}$ is thus defined accordingly, and inference is performed as $\tilde{F}(B(x)) = F_b(B(x)) - \mathbf{b}_{opt} = W_b(B(x))$, removing the bias term to yield unbiased predictions.
    - **Design Motivation**: The weight matrix $\mathbf{W}$ interacts with feature vectors and is difficult to disentangle analytically, whereas the bias term serves as a clean proxy that directly isolates optimization bias.

3. **Distribution Estimation Prior**

    - **Function**: Approximates the unlabeled data distribution using $\mathbf{b}_{opt}$ before full training, to initialize the pseudo-label sampling strategy.
    - **Mechanism**: After a few rounds of estimation training, calibrated outputs are used to infer the per-class sample counts $N^e$ on unlabeled data. A KL divergence match over predefined distributions is then performed: $o^* = \arg\min_o D_{KL}(N^e, Q^{(o)})$, which determines the expansion factor $c$ and initial thresholds.
    - **Design Motivation**: No assumption about the unlabeled data distribution is required, but leveraging an estimated prior leads to better initialization of the sampling strategy.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{basic} + \mathcal{L}_{sup}^b(\tau_b, F_b) + \mathcal{L}_{con}^b(\rho_b, F_b) + \mathcal{L}_{sup}^e(\tau_e, F_e) + \mathcal{L}_{con}^e(\rho_e, F_e)$, where $\mathcal{L}_{basic}$ is the base SSL loss, $\tau_b = 2$, and $\tau_e = 4$. The expansion factor $c$ is set to 4–6 based on the estimated distribution; $\rho_{max} = 0.95$, step size $\alpha = 0.005$, and threshold $\nu = 1.0$.

## Key Experimental Results

### Main Results

Test accuracy on CIFAR10-LT and CIFAR100-LT:

| Method | CIFAR10 Consist (100-100) | CIFAR10 Inverse (100-100) | CIFAR100 Consist (15-15) | CIFAR100 Inverse (15-15) |
|--------|--------------------------|--------------------------|-------------------------|-------------------------|
| FixMatch+LA | 81.49 | 80.68 | 58.56 | 58.21 |
| w/ ACR | 84.10 | 89.46 | 60.34 | 61.79 |
| w/ CPE | 84.46 | 87.10 | 59.83 | 60.83 |
| **w/ SC-SSL** | **86.53** | **89.97** | **60.65** | **62.99** |

Results on ImageNet-127:

| Method | 32×32 | 64×64 |
|--------|-------|-------|
| SimPro | 59.4 | 67.2 |
| **SC-SSL** | **62.3** | **69.4** |

### Ablation Study

| Configuration | CIFAR10 Consist | CIFAR10 Inverse | Note |
|---------------|----------------|----------------|------|
| 2-class split | 83.89 | 86.02 | Optimal head/non-head partition |
| 3-class split | 83.54 | 85.98 | Finer granularity yields no clear gain |
| 4-class split | 83.50 | 86.15 | Binary split is sufficient |

### Key Findings

- SC-SSL outperforms or matches state-of-the-art across all distribution settings (consistent, inverse, uniform, Gaussian, and unknown).
- Bias-term calibration yields substantial inference accuracy gains — the bias term patterns under different sampling strategies are clearly distinguishable.
- Improvements on STL10-LT under the unknown distribution setting are particularly notable (79.26% vs. 76.94%), demonstrating robustness to distributional uncertainty.
- A simple binary (head/non-head) partition is sufficient; finer-grained splits bring no additional benefit.

## Highlights & Insights

- **Bias term as a proxy for optimization bias**: The key observation is elegant — by contrasting bias term patterns across different sampling strategies, data bias and optimization bias are cleanly separated. This idea can be transferred to any classification task involving imbalanced training.
- **Theory-driven sampling control**: A simplified binary classification model is used to clearly identify three key control variables, providing a theoretical foundation for the sampling strategy design.
- **Practical application of the expansion-separation assumption**: The expansion factor from self-training theory is directly linked to threshold initialization, allowing theoretical analysis to concretely guide hyperparameter selection.

## Limitations & Future Work

- The setting of expansion factor $c$ depends on predefined anchor distributions, imposing constraints on distribution assumptions.
- Validation is limited to classification tasks; applicability to detection, segmentation, and other tasks remains unexplored.
- The three-classifier design increases training computational cost and introduces additional hyperparameters.
- Bias-term calibration assumes that feature-level imbalance has been sufficiently mitigated; performance may degrade when this condition is not adequately satisfied.

## Related Work & Insights

- **vs. ACR**: ACR adjusts consistency regularization via predefined distribution anchors; SC-SSL actively controls sampling probability through the expansion classifier, addressing feature bias at its source.
- **vs. SimPro**: SimPro models arbitrary distributions with a probabilistic framework but still relies on logit adjustment; SC-SSL calibrates directly via the bias term, offering a simpler and more effective approach.
- **vs. ABC**: ABC introduces an additional classifier to prevent bias but does not control sampling probability; the expansion classifier in SC-SSL has an explicit sampling control objective.

## Rating

- Novelty: ⭐⭐⭐⭐ The bias term analysis and three-classifier decoupled design are genuinely novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across multiple datasets and distribution settings
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and motivation is well-argued
- Value: ⭐⭐⭐⭐ Makes a practical contribution to the CISSL field, though the application scope is relatively narrow

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/llm_evaluation/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](../../NeurIPS2025/llm_evaluation/keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[AAAI 2026\] Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version](streaming_generated_gaussian_process_experts_for_online_learning_and_control_ext.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/llm_evaluation/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)

</div>

<!-- RELATED:END -->
