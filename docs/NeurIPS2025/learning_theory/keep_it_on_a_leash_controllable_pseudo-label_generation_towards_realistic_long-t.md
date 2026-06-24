---
title: >-
  [Paper Note] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
description: >-
  [NeurIPS 2025][Semi-Supervised Learning][Long-tailed distribution] This paper proposes Controllable Pseudo-label Generation (CPG), a framework that progressively incorporates reliable pseudo-labels into the labeled set via a controllable self-reinforcing optimization cycle. By training a Bayes-optimal classifier on a distribution of known composition, CPG achieves accuracy gains of up to 15.97% in the Realistic LTSSL setting where the unlabeled data distribution is entirely u…
tags:
  - "NeurIPS 2025"
  - "Semi-Supervised Learning"
  - "Long-Tailed Learning"
  - "Long-tailed distribution"
  - "pseudo-labels"
  - "distribution mismatch"
  - "Logit Adjustment"
date: 2026-05-08
content_hash: d4b415c1d25f365a
---

# Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.03993](https://arxiv.org/abs/2510.03993)  
**Code**: [https://github.com/yaxinhou/CPG](https://github.com/yaxinhou/CPG)  
**Area**: Semi-Supervised Learning / Long-Tailed Learning
**Keywords**: Long-tailed distribution, semi-supervised learning, pseudo-labels, distribution mismatch, Logit Adjustment

## TL;DR

This paper proposes Controllable Pseudo-label Generation (CPG), a framework that progressively incorporates reliable pseudo-labels into the labeled set via a controllable self-reinforcing optimization cycle. By training a Bayes-optimal classifier on a distribution of known composition, CPG achieves accuracy gains of up to 15.97% in the Realistic LTSSL setting where the unlabeled data distribution is entirely unknown.

## Background & Motivation

Long-tailed semi-supervised learning (LTSSL) represents an important practical scenario in which labeled data follows a long-tailed distribution (with abundant samples for common classes and scarce samples for rare ones), while large quantities of unlabeled data are leveraged to improve model performance. Existing methods share a critical assumption flaw:

**Conventional SSL methods** (FixMatch, FreeMatch, SoftMatch) assume that labeled and unlabeled data are both balanced and identically distributed, making them unsuitable for long-tailed settings.

**LTSSL methods** (ABC, CoSSL, BaCon) relax the balance assumption but still assume that labeled and unlabeled distributions are broadly consistent.

**ReaLTSSL methods** (ACR, CPE, SimPro) attempt to handle distribution mismatch; however, ACR and CPE assume the unlabeled data follows a predefined distribution (long-tailed, uniform, or inverse long-tailed), while SimPro, though prior-free, produces inaccurate estimates when the distributional gap is large.

**Core challenge**: In practice, the distribution of unlabeled data is completely unknown and can be arbitrary. For instance, vehicle type distributions in urban traffic surveillance vary drastically across regions and time periods. Existing methods rely on high-confidence pseudo-labels to estimate the distribution and then guide pseudo-label generation; when the distribution is unknown and arbitrary, these high-confidence pseudo-labels may themselves contain substantial errors, inducing confirmation bias.

## Method

### Overall Architecture

The CPG framework comprises three core components:
1. **Controllable Self-Reinforcing Optimization Cycle (CSOC)** — the central mechanism
2. **Class-Aware Adaptive Augmentation (CAA)** — enhancing minority-class representations
3. **Auxiliary Branch** — maximizing data utilization

### Key Designs

#### 1. Dynamic Controllable Filtering

Unlike conventional methods that generate pseudo-labels solely from weakly augmented views, CPG jointly exploits predictions from both weakly and strongly augmented views to identify reliable pseudo-labels. For each unlabeled sample $x_u$:

- Predictions and confidence scores are obtained from the weakly augmented view $\Omega_w(x_u)$ and the strongly augmented view $\Omega_s(x_u)$, respectively.
- The binary mask for reliable pseudo-labels is defined as: $\mathbb{I} = \mathbb{I}(\tilde{q}_w > \tau) \cdot \mathbb{I}(\tilde{q}_s > \tau) \cdot \mathbb{I}(\hat{q}_w = \hat{q}_s)$
- Three conditions must hold simultaneously: both views exceed the confidence threshold $\tau$ and predict the same class.
- A **voting strategy** is additionally introduced to ensure consistent pseudo-label assignment for the same sample across different training steps.

**Design Motivation**: Conventional methods propagate weakly augmented predictions to the strongly augmented view, which readily propagates erroneous predictions under distribution mismatch. CPG selectively incorporates only samples that are simultaneously high-confidence and consistent across both views, sacrificing coverage for reliability.

#### 2. Iterative Labeled Dataset Construction and Bayes-Optimal Classifier

This constitutes the core self-reinforcing cycle of CPG:

- The framework maintains per-class frequencies for the labeled set $n = \{n_1, \dots, n_C\}$ and the pseudo-labeled set $m = \{m_1, \dots, m_C\}$.
- The updated class distribution of the labeled set is $\pi_c = \phi_c / \sum_{c'} \phi_{c'}$, where $\phi_c = n_c + m_c$.
- A Bayes-optimal classifier is trained on the known distribution $\pi$ using the Logit Adjustment loss.
- The improved classifier identifies more reliable pseudo-labels in the next iteration, forming a positive feedback loop.

**Key Insight**: Rather than estimating the distribution of unlabeled data, CPG trains exclusively on distributions of known composition (labeled set + reliable pseudo-labels), rendering the model entirely insensitive to the unlabeled data distribution.

#### 3. Class-Aware Adaptive Augmentation (CAA)

To address insufficient minority-class representation:

- Class compactness is defined as $\alpha(c) = \frac{1}{\phi_c} \sum_{i} \frac{\langle h_i, \mu(c) \rangle}{\|h_i\| \cdot \|\mu(c)\|}$.
- Minority classes typically exhibit low intra-class diversity (high compactness), warranting a smaller augmentation radius $r = 1/\alpha$.
- Representation synthesis: $h'_i = h_i + \frac{h_i}{\|h_i\|} \cdot r(c) \cdot \delta_i$, where $\delta_i \sim \mathcal{N}(0, I)$.
- Ten augmented representations are synthesized for each minority-class sample.

#### 4. Auxiliary Branch

Since reliable pseudo-labels are scarce in early training, an auxiliary branch is introduced to utilize all labeled and unlabeled samples:

- A consistency regularization paradigm similar to FixMatch is adopted.
- The auxiliary loss $\ell_{aux}$ enforces agreement between weakly and strongly augmented view predictions.

### Loss & Training

The overall loss is: $\ell_{overall} = \ell_{la} + \omega \cdot \ell_{aux}$

where $\ell_{la}$ denotes the Logit-Adjusted cross-entropy loss and $\omega$ is a binary indicator ($\omega = 1$ for the auxiliary branch, $\omega = 0$ for the main branch).

Training proceeds in two stages:
- **Initial stage** (first 30 epochs): training on the labeled dataset only.
- **Iterative stage**: reliable pseudo-labels are progressively incorporated and the model is optimized on the updated labeled set.

### Theoretical Analysis

Theorem 1 provides an upper bound on the generalization error:

$$R_T \leq R_0 - \sum_{t=1}^T \lambda_t + U\sum_{t=1}^T \epsilon_t + 4\sqrt{2}\rho\sum_{t=1}^T\sum_{y=1}^C \mathcal{R}_{O_t}(\mathcal{H}_y) + 2U\sum_{t=1}^T \sqrt{\frac{\log(2/\upsilon)}{2O_t}}$$

The core implication is that as training progresses, the number of training samples $O_t$ grows while the pseudo-label error rate $\epsilon_t$ decreases or remains stable; combined with a Bayes-optimal classifier that maximizes the risk reduction $\lambda_t$, the generalization error is effectively minimized.

## Key Experimental Results

### Main Results

CPG is compared against SSL methods (FixMatch, FreeMatch, SoftMatch) and ReaLTSSL methods (ACR, SimPro, CDMAD) across four datasets:

| Dataset | Setting | Ours (CPG) | SimPro | FreeMatch | Gain |
|--------|------|---------|--------|-----------|------|
| CIFAR-10-LT (γ=100, Arbitrary) | Distribution mismatch | **82.10** | 65.81 | 65.41 | +15.97 pp |
| CIFAR-10-LT (γ=100, Inverse) | Inverse long-tail | **82.37** | 63.70 | 68.91 | +13.46 pp |
| CIFAR-10-LT (γ=100, Consistent) | Consistent distribution | **76.93** | 64.13 | 70.08 | +6.85 pp |
| CIFAR-100-LT (γ=10, Arbitrary) | Distribution mismatch | **51.48** | 44.26 | 45.97 | +5.51 pp |
| Food-101-LT (γ=10, Inverse) | Inverse long-tail | **25.52** | 17.31 | 21.89 | +3.63 pp |
| SVHN-LT (γ=100, Arbitrary) | Distribution mismatch | **93.99** | 90.10 | 85.78 | +3.89 pp |

**Average gains**: CIFAR-10-LT +11.14 pp, CIFAR-100-LT +3.09 pp, SVHN-LT +4.06 pp, Food-101-LT +2.57 pp.

### Ablation Study

| Configuration (w/ AB, CSOC, CAA) | CIFAR-10-LT Arbitrary | CIFAR-100-LT Arbitrary | Notes |
|-------------------------|----------------------|----------------------|------|
| No components | 65.18 | 46.32 | Baseline |
| +AB (Auxiliary Branch) | 65.98 (+0.80) | 48.37 (+2.05) | Improved data utilization |
| +AB +CAA | 68.97 (+3.79) | 48.90 (+2.58) | Minority augmentation effective |
| +AB +CSOC | 80.39 (+15.21) | 49.25 (+2.93) | Largest contribution from core component |
| +AB +CSOC +CAA (full) | **82.33 (+17.15)** | **51.85 (+5.53)** | Components are complementary |

CSOC alone contributes an average of 6.97 pp, CAA alone contributes 2.35 pp, and their combination yields 10.65 pp, demonstrating a super-additive effect.

### Key Findings

1. **Greatest advantage under arbitrary distribution**: CPG's margin is most pronounced in distribution-mismatch settings (up to +15.97 pp on CIFAR-10-LT).
2. **Substantially higher pseudo-label quality**: CPG's pseudo-label error rate is significantly lower than that of FreeMatch and SimPro, particularly for minority classes.
3. **Strong robustness**: As the imbalance of unlabeled data increases, baseline performance degrades sharply, while CPG remains largely stable.
4. **Generalization to arbitrary labeled distributions**: Under the extreme setting where labeled data also follows an arbitrary distribution (Table 4), CPG consistently outperforms competitors.
5. **Statistical significance**: All comparisons pass pairwise t-tests at the 0.05 significance level; CPG achieves superior performance across all datasets.

## Highlights & Insights

- **Core innovation**: Rather than estimating the unlabeled data distribution, CPG circumvents the distribution estimation problem by progressively expanding a labeled set of known composition — a philosophy of bypassing the obstacle rather than overcoming it.
- **Elegant self-reinforcing cycle**: A better classifier yields more reliable pseudo-labels, which expand and balance the training set, which in turn produces an even better classifier — a virtuous cycle.
- **Solid theoretical grounding**: Theorem 1 formally establishes the effectiveness of the cyclic mechanism from a generalization error perspective, going beyond heuristic motivation.
- **Strong practical applicability**: No prior knowledge of the unlabeled data distribution is required, making CPG genuinely suitable for real-world scenarios.

## Limitations & Future Work

1. **Limited absolute gains on CIFAR-100**: The inherent ceiling of 100-class fine-grained classification (SL baseline at 64.62%) constrains absolute improvements.
2. **Fixed warm-up period of 30 epochs**: The choice of this hyperparameter lacks an adaptive mechanism.
3. **Sensitivity of the pseudo-label threshold $\tau$**: The paper does not provide a detailed analysis of how threshold selection affects performance.
4. **Computational overhead**: Maintaining two branches (main and auxiliary) and computing per-class compactness along with synthesized representations in CAA incur additional cost.
5. **Stability of the voting strategy**: Under extreme distributional shift, early-stage votes may be dominated by erroneous pseudo-labels.

## Related Work & Insights

- **Comparison with SimPro**: SimPro employs EM to estimate the distribution and adjust pseudo-label probabilities, fundamentally still relying on distribution estimation. CPG completely bypasses this step.
- **Clever use of Logit Adjustment**: LA originally requires a known class prior; CPG naturally provides this prior by constructing an updated labeled set with a known distribution.
- **Broader applicability**: The principle of "training on known distributions" is transferable to other distribution-mismatch settings, such as domain adaptation and non-IID data in federated learning.

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea of bypassing distribution estimation is original, and the self-reinforcing cycle is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, multiple distribution settings, ablation studies, and statistical significance tests are all provided.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, methodology is systematic, and figures are informative.
- Value: ⭐⭐⭐⭐ — Represents a significant advance in the ReaLTSSL field with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](../../ICLR2026/learning_theory/conformal_prediction_for_long-tailed_classification.md)
- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](../../ICML2026/learning_theory/semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[ICLR 2026\] Learning from Label Proportions via Proportional Value Classification](../../ICLR2026/learning_theory/learning_from_label_proportions_via_proportional_value_classification.md)
- [\[NeurIPS 2025\] Computable Universal Online Learning](computable_universal_online_learning.md)

</div>

<!-- RELATED:END -->
