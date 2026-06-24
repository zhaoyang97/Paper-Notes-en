---
title: >-
  [Paper Note] Quadratic Upper Bound for Boosting Robustness
description: >-
  [ICML2025][AI Safety][Adversarial Training] By leveraging the convexity of the cross-entropy loss with respect to logits, a quadratic upper bound (QUB) for the adversarial training loss is derived. This serves as a plug-and-play loss function replacement for existing fast adversarial training methods, significantly boosting robustness.
tags:
  - "ICML2025"
  - "AI Safety"
  - "Adversarial Training"
  - "Fast Adversarial Training"
  - "Quadratic Upper Bound"
  - "Loss Smoothing"
  - "Robustness"
date: 2026-05-08
content_hash: 3f83cf9ccfdc957a
---

# Quadratic Upper Bound for Boosting Robustness

**Conference**: ICML2025  
**arXiv**: [2601.13645](https://arxiv.org/abs/2601.13645)  
**Code**: To be confirmed  
**Area**: Adversarial Robustness / Fast Adversarial Training  
**Keywords**: Adversarial Training, Fast Adversarial Training, Quadratic Upper Bound, Loss Smoothing, Robustness

## TL;DR

By leveraging the convexity of the cross-entropy loss with respect to logits, a quadratic upper bound (QUB) for the adversarial training loss is derived. This serves as a plug-and-play loss function replacement for existing fast adversarial training methods, significantly boosting robustness.

## Background & Motivation

Adversarial Training (AT) is the mainstream method for defending against adversarial attacks, with its core being the min-max optimization problem:

$$\min_\theta \max_{\|\delta\|_p \le \epsilon} \mathcal{L}(f_\theta(x+\delta), y)$$

Multi-step attacks (such as PGD) generate strong adversarial examples but incur high computational costs. **Fast Adversarial Training (FAT)** employs single-step attacks (such as FGSM) to reduce training time. However, due to insufficient exploration of the adversarial space, it often suffers from **catastrophic overfitting**—where the model becomes overly robust to the attacks used during training but loses defense capability against unseen attacks.

**Core Motivation**: Most existing FAT methods focus on designing better perturbation generation strategies or regularization terms. This work takes a different approach: starting from the loss function itself, it derives an upper bound of the AT loss to replace the original loss function, boosting robustness without the need to enhance the inner maximization.

## Method

### Quadratic Upper Bound Derivation

The cross-entropy loss is a convex function with respect to the logit vector $f(x)$. Utilizing this property, a Taylor expansion of the AT loss combined with the upper bound of the Hessian yields **Lemma 1**:

$$\mathcal{L}(f(x+\delta)) \le \mathcal{L}(f(x)) + (f(x+\delta)-f(x))^T \nabla_f \mathcal{L}(f(x)) + \frac{\|\boldsymbol{H}\|_2}{2}\|f(x+\delta)-f(x)\|_2^2$$

where $\|\boldsymbol{H}\|_2$ represents the spectral norm of the loss Hessian. **Lemma 2** further proves that $\|\boldsymbol{H}\|_2 \le \frac{1}{2}$. Substituting this gives the **QUB loss**:

$$\mathcal{L}_{\text{QUB}} = \underbrace{\mathcal{L}(f(x))}_{\text{Clean Sample Loss}} + \underbrace{(f(x+\delta)-f(x))^T \nabla_f \mathcal{L}(f(x))}_{\text{Loss Increment from Perturbation}} + \underbrace{\frac{1}{4}\|f(x+\delta)-f(x)\|_2^2}_{\text{Quadratic Penalty on Logit Change}}$$

### Intuitive Interpretation of the Three Terms

| Term | Function | Connection |
|---|---|---|
| First term $\mathcal{L}(f(x))$ | Drives the model to improve standard accuracy (SA) | Focuses on clean samples |
| Second term | Steers perturbation directions away from gradient directions, smoothing the loss landscape | Similar to input gradient regularization |
| Third term $\|f(x+\delta)-f(x)\|_2^2$ | Restricts the influence of perturbations on logits | Similar to the idea of TRADES |

**Computational Advantage**: The gradient $\nabla_f \mathcal{L}$ has a closed-form solution of $\hat{y} - y$ (softmax minus one-hot), requiring no extra backpropagation. All terms operate in the $\mathbb{R}^C$ space ($C$ is the number of classes), which is significantly smaller than the input space $\mathbb{R}^{c \times H \times W}$ or parameter space.

### Training Strategy

- **QUB-static**: Replaces the AT loss with the QUB loss throughout the entire training process.
- **QUB-decreasing**: Uses the QUB loss in the early stages and linearly transitions to the AT loss as training progresses.

$$\mathcal{L}_{\text{total}} = (1-\lambda_t) \cdot \mathcal{L}_{\text{QUB}} + \lambda_t \cdot \mathcal{L}_{\text{AT}}, \quad \lambda_t = t/T$$

**Design Motivation**: The upper bound property of QUB quickly boosts robustness in the early stages, but strong gradients in the later stages can lead to over-regularization, sacrificing standard accuracy. The progressive transition balances both robustness and generalization.

## Key Experimental Results

### CIFAR-10 + ResNet18 Robust Accuracy (%)

| Method | Step | SA | PGD-50/10 | AA | Time (h) |
|---|---|---|---|---|---|
| FGSM-CKPT | 1 | 90.02 | 37.42 | 37.22 | 1.05 |
| + QUB-static | 1 | 87.63 | 42.54 | **41.53** | 1.35 |
| + QUB-decreasing | 1 | 88.56 | 40.70 | 39.85 | 1.35 |
| FGSM-GA | 1 | 82.93 | 47.74 | 45.75 | 3.02 |
| + QUB-static | 1 | 79.75 | **50.82** | **47.33** | 3.27 |
| N-FGSM | 1 | 81.21 | 47.36 | 45.17 | 0.58 |
| + QUB-static | 1 | 80.76 | **49.60** | **47.00** | 0.70 |
| FGSM-PGI(MEP) | 1 | 81.48 | 51.75 | 48.41 | 0.89 |
| + QUB-decreasing | 1 | 81.56 | 52.24 | **48.58** | 1.19 |
| PGD-AT | 10 | 81.53 | 51.82 | 48.33 | 2.34 |
| + QUB-static | 10 | 80.24 | **53.39** | **49.91** | 2.64 |
| TRADES | 10 | 82.11 | 52.77 | 50.16 | 3.50 |

**Key Findings**:

- Except for FGSM-RS, the robust accuracy of all baseline methods improves after applying +QUB.
- FGSM-CKPT +QUB-static shows the most significant improvement, with AutoAttack (AA) accuracy increasing from 37.22% to **41.53%** (+4.31%).
- PGD-AT +QUB-static achieves 49.91% AA, which is close to TRADES (50.16%) but with shorter training time.
- QUB-static provides better robustness but at the cost of standard accuracy, whereas QUB-decreasing is more balanced.
- Training time increases by about 20-30%, which is far less than the cost of upgrading to multi-step attacks.

### Loss Landscape Visualization

The loss landscape of the model trained with QUB is significantly flatter, indicating that the model maintains stable predictions over a larger region around the input. This validates the mechanism of QUB enhancing robustness by smoothing the loss landscape.

## Highlights & Insights

1. **Theoretical Elegance**: The derivation based on the convexity of cross-entropy is clean and concise, with each of the three terms in QUB having a clear physical meaning.
2. **Plug-and-Play**: It only requires replacing the loss function. It is orthogonally compatible with existing FAT methods and has a low implementation barrier.
3. **Computational Efficiency**: $\nabla_f \mathcal{L}$ has a closed-form solution, and all intermediate variables are in the class dimension $\mathbb{R}^C$, resulting in small memory and computational overheads.
4. **Strong Universality**: It is effective in 8 out of 9 FAT baselines (except FGSM-RS), and multi-step PGD-AT also benefits.
5. **Progressive Strategy**: QUB-decreasing balances robustness and generalization through a simple linear schedule without requiring additional hyperparameter tuning.

## Limitations & Future Work

1. **FGSM-RS Failure**: When the adversarial examples of the baseline method themselves are of low quality, QUB smooths over non-informative regions instead, amplifying the deficiencies.
2. **SA Drop**: QUB-static sacrifices 2-5% of standard accuracy for robustness in most methods, and the mitigation from QUB-decreasing is limited.
3. **Incomplete Large-Scale Validation**: Experiments are only conducted on CIFAR-10/100 and Tiny ImageNet, without validation on large-scale datasets like ImageNet.
4. **Limited to $\ell_\infty$ Attacks**: Performance under other attack norms such as $\ell_2$ and $\ell_1$ has not been explored.
5. **Tightness of the Upper Bound**: Taking the global upper bound of $1/2$ for the spectral norm of the Hessian might not be tight enough; adaptive estimation could be superior.
6. **Relationship with TRADES**: The third term shares a similar idea with the KL-divergence regularization in TRADES. The effect of directly combining with TRADES remains unexplored.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of deriving an upper bound from the convexity of the loss function is novel, complementing existing methods at the level of optimization objectives.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 9 baselines, 3 datasets, various attacks, and models, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivation with an intuitive explanation of the three terms.
- Value: ⭐⭐⭐⭐ — The plug-and-play loss replacement has strong engineering practicality, though validation on large-scale scenarios is a bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](../../NeurIPS2025/ai_safety/boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[CVPR 2026\] A Provable Energy-Guided Test-Time Defense Boosting Adversarial Robustness of Large Vision-Language Models](../../CVPR2026/ai_safety/a_provable_energy-guided_test-time_defense_boosting_adversarial_robustness_of_la.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](../../NeurIPS2025/ai_safety/bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)

</div>

<!-- RELATED:END -->
