---
title: >-
  [Paper Note] Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training
description: >-
  [ICCV 2025][adversarial training] This paper reveals a counterintuitive phenomenon in adversarial training — the model's perceptual change on failure cases is actually smaller than on success cases (i.e.…
tags:
  - "ICCV 2025"
  - "adversarial training"
  - "accuracy-robustness trade-off"
  - "perceptual consistency"
  - "decision boundary"
  - "robust perception"
date: 2026-05-08
content_hash: 99e20e40341a5b72
---

# Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training

**Conference**: ICCV 2025
**arXiv**: [2508.02186](https://arxiv.org/abs/2508.02186)  
**Code**: [https://github.com/FlaAI/RPAT](https://github.com/FlaAI/RPAT)  
**Area**: Other
**Keywords**: adversarial training, accuracy-robustness trade-off, perceptual consistency, decision boundary, robust perception

## TL;DR

This paper reveals a counterintuitive phenomenon in adversarial training — the model's perceptual change on failure cases is actually smaller than on success cases (i.e., failure cases are "over-learned") — and proposes Robust Perception Adversarial Training (RPAT), which encourages perceptions to change smoothly with perturbations to alleviate the accuracy-robustness trade-off.

## Background & Motivation

Adversarial training (AT) is the most effective method for training robust DNNs, yet it suffers from an inherent **accuracy-robustness trade-off**: improving adversarial robustness often comes at the cost of reduced clean accuracy.

**Established consensus**: An overly complex decision boundary induced by AT is considered the root cause of this trade-off, and prior work commonly attributes this to **insufficient learning of hard adversarial examples**, motivating strategies such as data reweighting (GAIRAT/MAIL), adaptive perturbation radii (MMA), and non-one-hot supervision (TE/SOVR).

**New finding**: Through proof-of-concept experiments, this paper is the first to reveal a counterintuitive fact — in PGD-AT-trained robust models, the perceptual change (logits MSE between benign and adversarial samples) of defense failure cases is actually smaller than that of success cases. This indicates that failure cases are already **over-learned** under the current AT objective, rather than under-learned. The true issue lies in the improper placement of the decision boundary — the excessive pursuit of perceptual invariance causes the model to treat perturbations as noise, ignoring perturbation information that could be used to more appropriately position the decision boundary.

## Method

### Overall Architecture

A Robust Perception regularization term is added on top of the conventional AT objective, encouraging the model's perception to change smoothly and linearly with input perturbations rather than remaining invariant. The overall RPAT objective combines the cross-entropy loss on adversarial examples with a perceptual smoothness regularizer.

### Key Designs

1. **Robust Perception objective**: For a benign sample $\mathbf{x}$ and its adversarial counterpart $\mathbf{x}'$, for any interpolation $\alpha \in [0,1]$, the following is required:
$$\|h_{\bm{\theta}}(\mathbf{x}+\alpha \cdot \Delta) - h_{\bm{\theta}}(\mathbf{x})\| = \alpha \cdot \|h_{\bm{\theta}}(\mathbf{x}') - h_{\bm{\theta}}(\mathbf{x})\|$$
That is, the model's perception should change linearly in proportion to the perturbation magnitude.

2. **Theoretical support** (two theorems):

    - *Theorem 1* (Local Linearity): The Robust Perception constraint drives the quadratic form of the Hessian matrix toward zero, $\Delta^\top H_{h_\theta}(\mathbf{x}) \Delta \to 0$, suppressing higher-order nonlinear effects so that perception changes primarily along the linear term of the perturbation.
    - *Theorem 2* (Lipschitz Regularization): Guarantees that the variation of the Jacobian along the perturbation direction is bounded, and the increment of the global Lipschitz constant is constrained to an infinitesimal $\gamma$, thereby smoothing the decision boundary.

3. **RPAT loss function**: Using logits as the model perception representation, an MSE regularization is applied among the benign sample $\mathbf{x}$, interpolated sample $\tilde{\mathbf{x}} = \mathbf{x} + \alpha \cdot \Delta$, and adversarial sample $\hat{\mathbf{x}}'$:
$$\mathcal{L}^{\text{RPAT}} = \frac{1}{n}\sum_{i=1}^n \left(\mathcal{L}^{\text{CE}}(\mathbf{p}(\hat{\mathbf{x}}_i', \bm{\theta}), y_i) + \lambda \cdot \mathcal{L}^{\text{MSE}}\left(\frac{\mathbf{z}(\tilde{\mathbf{x}}_i) - \mathbf{z}(\mathbf{x}_i)}{\alpha} \bigg\| \frac{\mathbf{z}(\hat{\mathbf{x}}_i') - \mathbf{z}(\tilde{\mathbf{x}}_i)}{1-\alpha}\right)\right)$$

### Loss & Training

- RPAT can be flexibly added on top of any AT baseline (PGD-AT, TRADES, MART, Consistency-AT).
- RPAT++ integrates RPAT into the current state-of-the-art method ReBAT.
- The regularization weight $\lambda$ and interpolation coefficient $\alpha$ are two hyperparameters.
- Success and failure cases are not explicitly distinguished — success cases should also satisfy the Robust Perception criterion.

## Key Experimental Results

### Main Results

Gains of RPAT over 4 AT baselines (ResNet-18, $\ell_\infty$):

| Dataset | Method | Clean | PGD-20 | AA | Mean | NRR |
|---------|--------|-------|--------|-----|------|-----|
| CIFAR-10 | PGD-AT | 82.92 | 50.61 | 46.74 | 64.83 | 59.78 |
| CIFAR-10 | +RPAT | **83.20** | **51.29** | **48.00** | **65.60** | **60.88** |
| CIFAR-10 | Consistency-AT | 83.42 | 51.96 | 47.72 | 65.57 | 60.71 |
| CIFAR-10 | +RPAT | **84.12** | **52.33** | **48.98** | **66.55** | **61.91** |
| CIFAR-100 | PGD-AT | 56.56 | 28.80 | 25.02 | 40.79 | 34.69 |
| CIFAR-100 | +RPAT | **58.22** | **29.16** | **24.88** | **41.55** | **34.86** |

RPAT++ vs. 12 state-of-the-art methods (PreActResNet-18):

| Method | Clean | AA | Mean | NRR |
|--------|-------|----|------|-----|
| ReBAT (NeurIPS'23) | 82.09 | 50.72 | 66.41 | 62.70 |
| **RPAT++ (Ours)** | **82.63** | **51.00** | **66.82** | **63.07** |
| CIFAR-100 ReBAT | 56.13 | 27.60 | 41.87 | 37.00 |
| CIFAR-100 **RPAT++** | **56.84** | **27.68** | **42.26** | **37.23** |

### Ablation Study

Ablation over different regularization metrics ($\ell_\infty$, from Section 5.3):

| Metric | Applicability |
|--------|--------------|
| MSE | Base version; stable and effective |
| KL | Also shown to be effective |
| Cosine | Viable alternative |

Ablation experiments on the interpolation coefficient $\alpha$ and weight $\lambda$ confirm the robustness of the method to hyperparameter choices.

### Key Findings

- RPAT yields consistent improvements across **all** 4 baselines × 3 datasets × 2 norm settings.
- Both clean accuracy and adversarial robustness are improved simultaneously, rather than exhibiting a trade-off.
- RPAT++ surpasses 12 current state-of-the-art methods (including AWP, KD+SWA, GAIRAT, TE, etc.) on CIFAR-10 and CIFAR-100 under the $\ell_\infty$ setting.
- Effectiveness on Tiny-ImageNet demonstrates scalability to larger datasets.

## Highlights & Insights

- **Core contribution is a conceptual update**: The paper overturns the widely held belief that "failure cases are under-learned," proposing instead the new perspective that "over-learning leads to improper decision boundary placement."
- **Elegantly designed proof-of-concept experiments**: By comparing perceptual changes of success and failure cases under three settings — clean training, random perturbation, and PGD-AT — the counterintuitive phenomenon is clearly demonstrated.
- **Theoretically grounded**: The validity of Robust Perception is justified from two perspectives: local linearity and Lipschitz regularization.
- **Plug-and-play**: The RPAT regularization term can be directly integrated into any existing AT method.

## Limitations & Future Work

- The selection of hyperparameters $\alpha$ and $\lambda$ still requires dataset-specific tuning.
- Experiments are conducted primarily on relatively small-scale datasets (CIFAR-10/100, Tiny-ImageNet), with no validation at the ImageNet scale.
- The method focuses solely on classification tasks; applicability to downstream tasks such as detection and segmentation remains unverified.
- The underlying causes of the "over-learning" phenomenon warrant further analysis.
- Only a single interpolation point $\alpha$ is used; multi-point sampling may yield more accurate regularization but at greater computational cost.

## Related Work & Insights

- TRADES was among the first to consider balancing clean accuracy and robustness.
- AWP smooths the loss landscape through weight perturbation; ReBAT formulates AT as a dynamic game.
- GAIRAT/MAIL focus on reweighting hard examples.
- The perspective shift introduced in this paper — from under-learning of hard samples to improper decision boundary placement — may inspire new research directions in the AT community.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to reveal the "over-learning" phenomenon of failure cases in AT, offering a perspective entirely distinct from the mainstream.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets, 3 architectures, 4 baselines, 12 state-of-the-art methods, two attack norm settings.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative is exceptionally fluent, with a seamless logical chain from discovery → analysis → solution.
- Value: ⭐⭐⭐⭐ Provides important insights for the AT community, with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations](on_the_complexity-faithfulness_trade-off_of_gradient-based_explanations.md)
- [\[AAAI 2026\] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation](../../AAAI2026/others/forest_vs_tree_the_n_k_trade-off_in_reproducible_ml_evaluation.md)
- [\[CVPR 2026\] IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](../../CVPR2026/others/irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases](fixtalk_taming_identity_leakage_for_high-quality_talking_head_generation_in_extr.md)

</div>

<!-- RELATED:END -->
