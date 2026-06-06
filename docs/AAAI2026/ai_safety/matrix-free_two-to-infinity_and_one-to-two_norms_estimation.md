---
title: >-
  [Paper Note] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation
description: >-
  [AAAI 2026][AI Safety][matrix norm estimation] This paper proposes TwINEst and TwINEst++, two randomized algorithms based on the Hutchinson diagonal estimator…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "matrix norm estimation"
  - "two-to-infinity norm"
  - "Hutchinson estimator"
  - "Jacobian regularization"
  - "adversarial robustness"
date: 2026-05-08
content_hash: 0f0fa5c1fd77ba2c
---

# Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation

**Conference**: AAAI 2026
**arXiv**: [2508.04444](https://arxiv.org/abs/2508.04444)  
**Code**: [github](https://github.com/fallnlove/TwoToInfinity)  
**Area**: AI Safety / Randomized Linear Algebra
**Keywords**: matrix norm estimation, two-to-infinity norm, Hutchinson estimator, Jacobian regularization, adversarial robustness

## TL;DR

This paper proposes TwINEst and TwINEst++, two randomized algorithms based on the Hutchinson diagonal estimator, for efficiently estimating $\|A\|_{2\to\infty}$ and $\|A\|_{1\to 2}$ norms in a matrix-free setting. The algorithms come with provable oracle complexity guarantees and demonstrate significant advantages in Jacobian regularization for adversarial robustness in image classification and defense against adversarial attacks in recommender systems.

## Background & Motivation

In modern machine learning, many important matrices—such as the Jacobians of deep neural networks—are too large to construct explicitly, yet support efficient matrix-vector products via automatic differentiation. This motivates the need to estimate matrix properties in a matrix-free setting. The $\|A\|_{2\to\infty}$ norm equals the maximum $\ell_2$ row norm of the matrix and provides finer row-wise control compared to the spectral norm and Frobenius norm, making it particularly suitable for "tall-and-thin" matrices (e.g., Jacobians of image classifiers, where the input dimension far exceeds the number of output classes).

Existing methods based on adaptive power iteration lack theoretical convergence guarantees; the paper constructs a counterexample demonstrating that such methods can diverge with probability at least 29.5% even on simple diagonal matrices. The core challenge is: how to stochastically estimate $\|A\|_{2\to\infty}$ using only matrix-vector products while providing provable oracle complexity guarantees.

## Method

### Overall Architecture

The key observation is that $\|A\|_{2\to\infty}^2 = \max_{i \in [d]} \text{diag}(AA^\top)_i$, i.e., $\|A\|_{2\to\infty}$ is equivalent to the square root of the maximum diagonal entry of $AA^\top$. The Hutchinson diagonal estimator is therefore used to estimate the diagonal of $AA^\top$, after which the row index of the maximum entry is identified and the norm of the corresponding row is computed exactly.

### Key Designs

**1. TwINEst Algorithm**

Algorithm steps: (a) Sample $m$ Rademacher random vectors $X^1, \dots, X^m \in \{-1,1\}^d$; (b) for each $X^i$, compute $t_i = X^i \odot AA^\top X^i$; (c) average to obtain the diagonal estimate $D = \frac{1}{m}\sum_i t_i$; (d) identify the maximum index $j = \arg\max_i D_i$; (e) compute exactly $L = \|A^\top e_j\|_2$.

The key insight lies in step (e): rather than directly taking $\sqrt{\max_i D_i}$ (which has high variance), the estimated diagonal is used only to identify a candidate row index, after which the norm of that row is computed exactly, eliminating one layer of randomness. The oracle complexity

$$m > \frac{8\log(2d/\delta)}{\Delta^2} \|AA^\top - \text{diag}(AA^\top)\|_{2\to\infty}^2$$

guarantees that the exact value $\|A\|_{2\to\infty}$ is returned with probability $1-\delta$, where $\Delta$ is the squared gap between the largest and second-largest row norms.

**2. TwINEst++ Algorithm (Variance-Reduced Variant)**

Inspired by Hutch++, this variant decomposes $AA^\top$ into a low-rank approximation and a residual:

$$AA^\top = \underbrace{AA^\top P}_{\text{low-rank part}} + \underbrace{AA^\top(I-P)}_{\text{residual}}$$

where $P = QQ^\top$ is an orthogonal projector obtained via QR decomposition of $AA^\top S$ ($S$ being a Rademacher matrix). The diagonal of the low-rank part is computed exactly, while the residual diagonal is estimated via Hutchinson. The oracle complexity improves to:

$$m > c \cdot \left(\frac{\sqrt{\log(2/\delta)}}{\Delta} \|A\|_F^2 + \log(1/\delta)\right)$$

representing an improvement from $O(1/\Delta^2)$ to $O(1/\Delta)$, with particularly pronounced benefits for low-rank matrices.

**3. Counterexample for Divergence of Adaptive Power Iteration**

The paper constructs a simple counterexample: $A = \text{diag}(2, 1)$. After transforming the initial random vector by $A$, there is a $\geq 29.5\%$ probability that the absolute value of the second component exceeds that of the first, causing the $\text{dual}_\infty$ operation to select the wrong row. All subsequent iterations then lock onto this incorrect row, and the algorithm ultimately outputs 1 instead of the correct answer 2.

### Loss & Training

In deep learning applications, $\|A\|_{2\to\infty}$ is used as a Jacobian regularization term:

$$\mathcal{L}(x,y) = \mathcal{L}_{\text{CE}}(f(x), y) + \lambda \cdot \|J_f(x)\|_{2\to\infty}^2$$

Since computing the regularization term incurs a cost equivalent to one backward pass, the regularization term is updated every $k$ iterations in practice. In recommender systems, the weight decay term in UltraGCN is replaced with $\|\hat{R}\|_{2\to\infty}^2$, approximated using TwINEst.

## Key Experimental Results

### Main Results

Jacobian regularization for image classification (WideResNet-16-10, averaged over 3 runs):

| Regularization | CIFAR-100 Acc ↑ | FGSM ↑ | PGD ↑ | S.Rank ↓ | TinyImageNet Acc ↑ | FGSM ↑ | PGD ↑ | S.Rank ↓ |
|---|---|---|---|---|---|---|---|---|
| None | 75.5±0.2 | 24.4±0.6 | 11.7±0.3 | 32.0±1.1 | 57.8±1.3 | 30.4±0.3 | 20.2±0.1 | 30.9±4.3 |
| Frobenius | 75.7±0.5 | 23.5±0.2 | 13.3±0.2 | 31.6±0.2 | 58.6±0.3 | 31.1±0.2 | 20.7±0.5 | 27.8±0.9 |
| Spectral | 75.7±0.3 | 23.3±0.7 | 11.3±0.4 | 32.0±1.0 | 57.4±0.8 | 30.0±1.1 | 20.0±0.5 | 28.2±0.3 |
| Infinity | 75.8±0.4 | 23.7±0.7 | 11.1±0.2 | 30.7±1.2 | 57.1±0.7 | 29.6±1.4 | 19.7±1.0 | 28.8±0.9 |
| **Two-to-Infinity** | **77.3±0.1** | **26.9±0.5** | **14.5±0.5** | **18.3±0.8** | **59.6±0.9** | **31.0±1.0** | **23.4±0.7** | **24.9±0.3** |

Adversarial robustness in recommender systems (UltraGCN, NDCG@10):

| Attack Strength | Dataset | Weight Decay | Factor $\|·\|_{2\to\infty}$ | Score $\|\hat{R}\|_{2\to\infty}$ (ours) |
|---|---|---|---|---|
| No attack | MovieLens-1M | ~0.35 | ~0.35 | ~0.35 |
| Moderate attack | MovieLens-1M | ~0.28 | ~0.30 | ~0.32 |
| Strong attack | MovieLens-1M | ~0.22 | ~0.25 | ~0.28 |

### Ablation Study

Comparison of $\|A\|_{2\to\infty}$ estimation accuracy on synthetic matrices (5000×5000 Gaussian matrix, averaged over 500 trials):

| Method | $\Delta=10^{-2}$ (relative error at 400 matvecs) | $\Delta=10^{-1}$ (relative error at 400 matvecs) |
|---|---|---|
| Adaptive power iteration | ~0.15 (non-convergent) | ~0.15 (non-convergent) |
| Rademacher averaging | ~0.02 | ~0.005 |
| TwINEst | ~0.005 | ~0.0 (converged) |
| TwINEst++ | ~0.002 | ~0.0 (fastest convergence) |

On WideResNet Jacobian matrices ($3072 \times 100$): TwINEst++ achieves near-zero error at approximately 100 matrix-vector products, consistent with the low-rank structure of the Jacobian. The adaptive power iteration fails to converge even after 500 products.

### Key Findings

- On CIFAR-100, $\|·\|_{2\to\infty}$ regularization improves test accuracy by 1.8% (77.3 vs. 75.5), PGD adversarial accuracy by 2.8%, and reduces stable rank by 42.8% (18.3 vs. 32.0).
- Other norm regularizers (Frobenius, spectral, $\infty$) yield limited or negligible improvements, highlighting the advantage of $\|·\|_{2\to\infty}$ for row-wise control of the Jacobian.
- $\|\hat{R}\|_{2\to\infty}^2$ regularization improves robustness under moderate and strong attacks across all three recommender system datasets.
- TwINEst++ substantially outperforms TwINEst on low-rank matrices, consistent with theoretical predictions.

## Highlights & Insights

- The counterexample demonstrating divergence of adaptive power iteration serves as a compelling motivation, exposing a fundamental flaw in existing methods.
- The "coarse estimate then exact computation" strategy in TwINEst—using Hutchinson to identify a candidate row index and then computing the row norm exactly—is remarkably concise and efficient.
- The complexity improvement from $O(1/\Delta^2)$ to $O(1/\Delta)$ is particularly significant for hard instances with small $\Delta$.
- The unexpectedly strong performance of $\|·\|_{2\to\infty}$ norm regularization in image classification opens a new avenue for Jacobian regularization.

## Limitations & Future Work

- Oracle complexity lower bounds have not been established, leaving the optimality of TwINEst/TwINEst++ an open question.
- Deep learning experiments are conducted only on WideResNet-16-10; evaluation on broader architectures (ResNet, ViT, etc.) is lacking.
- The adversarial attack setting in recommender system experiments is relatively simple (single-step FGSM-style attacks); effectiveness under stronger attacks remains to be verified.
- Computing the regularization term still incurs non-trivial overhead (equivalent to several backward passes); while periodic updates every $k$ steps mitigate this, they introduce an additional hyperparameter.

## Related Work & Insights

- **vs. Adaptive Power Iteration (Higham/Roth)**: Power iteration lacks theoretical guarantees and can diverge; TwINEst provides oracle complexity guarantees and converges stably in experiments.
- **vs. Frobenius/Spectral Norm Regularization**: These classical regularizers do not differentiate across Jacobian rows; $\|·\|_{2\to\infty}$, by focusing on the maximum row norm, provides finer-grained control and is especially effective for tall-and-thin matrices.

## Rating

- Novelty: ⭐⭐⭐⭐ First matrix-free $\|·\|_{2\to\infty}$ estimation algorithm with theoretical guarantees; the application to DNN regularization is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three evaluation scenarios (synthetic experiments, DNN classification, recommender systems) with strong mutual validation between theory and experiments.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and clear; the counterexample is elegantly constructed; experimental presentation is thorough.
- Value: ⭐⭐⭐⭐ The algorithm is concise and practical, theoretically grounded, and has clear deployment value in DNN training and recommender systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] EFX and PO Allocation Exists for Two Types of Goods](efx_and_po_allocation_exists_for_two_types_of_goods.md)
- [\[ICML 2026\] Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks](../../ICML2026/ai_safety/unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)
- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)
- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](../../CVPR2026/ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)
- [\[NeurIPS 2025\] PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture](../../NeurIPS2025/ai_safety/pubsub-vfl_towards_efficient_two-party_split_learning_in_heterogeneous_environme.md)

</div>

<!-- RELATED:END -->
