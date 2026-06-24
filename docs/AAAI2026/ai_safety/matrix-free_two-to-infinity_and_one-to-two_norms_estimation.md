---
title: >-
  [Paper Note] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation
description: >-
  [AAAI 2026][AI Safety][matrix norm estimation] The authors propose TwINEst and TwINEst++, two randomized algorithms based on the Hutchinson diagonal estimator, to efficiently estimate $\|A\|_{2\to\infty}$ and $\|A\|_{1\to 2}$ norms in a matrix-free setting. They provide theoretical oracle complexity guarantees and demonstrate significant advantages in Jacobian regularization of DNNs (for adversarial robustness in image classification) and defense against adversarial attacks i…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "matrix norm estimation"
  - "two-to-infinity norm"
  - "Hutchinson estimator"
  - "Jacobian regularization"
  - "adversarial robustness"
date: 2026-05-08
content_hash: a80347d5702df125
---

# Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation

**Conference**: AAAI 2026  
**arXiv**: [2508.04444](https://arxiv.org/abs/2508.04444)  
**Code**: [github](https://github.com/fallnlove/TwoToInfinity)  
**Area**: AI Safety / Randomized Linear Algebra  
**Keywords**: matrix norm estimation, two-to-infinity norm, Hutchinson estimator, Jacobian regularization, adversarial robustness

## TL;DR

The authors propose TwINEst and TwINEst++, two randomized algorithms based on the Hutchinson diagonal estimator, to efficiently estimate $\|A\|_{2\to\infty}$ and $\|A\|_{1\to 2}$ norms in a matrix-free setting. They provide theoretical oracle complexity guarantees and demonstrate significant advantages in Jacobian regularization of DNNs (for adversarial robustness in image classification) and defense against adversarial attacks in recommender systems.

## Background & Motivation

In modern machine learning, many critical matrices (such as the Jacobian of deep neural networks) are too large to be explicitly constructed, but they support efficient matrix-vector products through automatic differentiation. This gives rise to the demand for matrix property estimation in a matrix-free setting. The $\|A\|_{2\to\infty}$ norm equals the maximum of the $\ell_2$ norms of the matrix rows. Compared to the spectral norm and Frobenius norm, it provides finer row-wise control, which is particularly suitable for "tall-and-skinny" matrices (such as the Jacobian of image classifiers, where the input dimension is much larger than the number of output classes).

Existing methods (such as the adaptive power method) lack theoretical convergence guarantees; the paper demonstrates through a counterexample that they can diverge with a probability of 29.5% even on a simple diagonal matrix. The core challenge is: how to stochastically estimate $\|A\|_{2\to\infty}$ using only matrix-vector products while providing provable oracle complexity guarantees.

## Method

### Overall Architecture

Core observation: $\|A\|_{2\to\infty}^2 = \max_{i \in [d]} \text{diag}(AA^\top)_i$, meaning that $\|A\|_{2\to\infty}$ is equivalent to the square root of the maximum diagonal element of $AA^\top$. Therefore, the Hutchinson diagonal estimator can be used to estimate the diagonal of $AA^\top$, and the norm of the row corresponding to the maximum index can be calculated exactly once it is identified.

### Key Designs

**1. TwINEst Algorithm**

Algorithm steps: (a) sample $m$ Rademacher random vectors $X^1, \dots, X^m \in \{-1,1\}^d$; (b) compute $t_i = X^i \odot AA^\top X^i$ for each $X^i$; (c) average them to obtain the diagonal estimate $D = \frac{1}{m}\sum_i t_i$; (d) find the maximum index $j = \arg\max_i D_i$; (e) exactly compute $L = \|A^\top e_j\|_2$.

The key trick lies in step (e): instead of directly taking $\sqrt{\max_i D_i}$ (which has high variance), the estimated values are used to find a candidate row index, after which the norm of that row is calculated exactly, eliminating a layer of randomness. The oracle complexity is:

$$m > \frac{8\log(2d/\delta)}{\Delta^2} \|AA^\top - \text{diag}(AA^\top)\|_{2\to\infty}^2$$

This guarantees that the exact value $\|A\|_{2\to\infty}$ is returned with probability $1-\delta$, where $\Delta$ is the difference between the square of the largest row norm and the square of the second largest row norm.

**2. TwINEst++ Algorithm (Variance-Reduced Version)**

Drawing inspiration from Hutch++, the algorithm decomposes $AA^\top$ into a low-rank approximation and a residual:

$$AA^\top = \underbrace{AA^\top P}_{\text{low-rank part}} + \underbrace{AA^\top(I-P)}_{\text{residual}}$$

where $P = QQ^\top$ is an orthogonal projection obtained via the QR decomposition of $AA^\top S$ ($S$ is a Rademacher matrix). The diagonal of the low-rank part is computed exactly, and the residual part is estimated using the Hutchinson estimator. The oracle complexity is improved to:

$$m > c \cdot \left(\frac{\sqrt{\log(2/\delta)}}{\Delta} \|A\|_F^2 + \log(1/\delta)\right)$$

This improves the complexity from $O(1/\Delta^2)$ to $O(1/\Delta)$, which is particularly significant on low-rank matrices.

**3. Counterexample of Adaptive Power Method Divergence**

The paper constructs a simple counterexample: $A = \text{diag}(2, 1)$. After the initial random vector is transformed by $A$, there is a $\geq 29.5\%$ probability that the absolute value of the second component is larger than the first, leading the dual$_\infty$ operation to select the wrong row. Consequently, all subsequent iterations lock onto the incorrect row, ultimately outputting 1 instead of the correct answer 2.

### Loss & Training

In deep learning applications, $\|A\|_{2\to\infty}$ is used as a regularization term for the Jacobian:

$$\mathcal{L}(x,y) = \mathcal{L}_{\text{CE}}(f(x), y) + \lambda \cdot \|J_f(x)\|_{2\to\infty}^2$$

Since computing the regularization term at every step costs as much as a backpropagation pass, a strategy of updating the regularization term every $k$ iterations is adopted in practice. In recommender systems, the weight decay term of UltraGCN is replaced by $\|\hat{R}\|_{2\to\infty}^2$, which is approximated using TwINEst.

## Key Experimental Results

### Main Results

Jacobian regularization in image classification (WideResNet-16-10, average of 3 trials):

| Regularization Method | CIFAR-100 Acc ↑ | FGSM ↑ | PGD ↑ | S.Rank ↓ | TinyImageNet Acc ↑ | FGSM ↑ | PGD ↑ | S.Rank ↓ |
|-----------|----------------|--------|-------|----------|-------------------|--------|-------|----------|
| No Regularization | 75.5±0.2 | 24.4±0.6 | 11.7±0.3 | 32.0±1.1 | 57.8±1.3 | 30.4±0.3 | 20.2±0.1 | 30.9±4.3 |
| Frobenius | 75.7±0.5 | 23.5±0.2 | 13.3±0.2 | 31.6±0.2 | 58.6±0.3 | 31.1±0.2 | 20.7±0.5 | 27.8±0.9 |
| Spectral | 75.7±0.3 | 23.3±0.7 | 11.3±0.4 | 32.0±1.0 | 57.4±0.8 | 30.0±1.1 | 20.0±0.5 | 28.2±0.3 |
| Infinity | 75.8±0.4 | 23.7±0.7 | 11.1±0.2 | 30.7±1.2 | 57.1±0.7 | 29.6±1.4 | 19.7±1.0 | 28.8±0.9 |
| **Two-to-Infinity** | **77.3±0.1** | **26.9±0.5** | **14.5±0.5** | **18.3±0.8** | **59.6±0.9** | **31.0±1.0** | **23.4±0.7** | **24.9±0.3** |

Adversarial robustness in recommender systems (UltraGCN, NDCG@10):

| Attack Intensity | Dataset | Weight Decay | Factor $\|·\|_{2\to\infty}$ | Score $\|\hat{R}\|_{2\to\infty}$ (Ours) |
|---------|-------|-------------|----------------------------|-----------------------------------------|
| No Attack | MovieLens-1M | ~0.35 | ~0.35 | ~0.35 |
| Medium Attack | MovieLens-1M | ~0.28 | ~0.30 | ~0.32 |
| Strong Attack | MovieLens-1M | ~0.22 | ~0.25 | ~0.28 |

### Ablation Study

Comparison of $\|A\|_{2\to\infty}$ estimation accuracy on synthetic matrices (5000×5000 Gaussian matrix, average of 500 trials):

| Method | $\Delta=10^{-2}$ (Relative Error for 400 Multiplications) | $\Delta=10^{-1}$ (Relative Error for 400 Multiplications) |
|-----|--------------------------------------|--------------------------------------|
| Adaptive Power Method | ~0.15 (Diverges) | ~0.15 (Diverges) |
| Rademacher Average | ~0.02 | ~0.005 |
| TwINEst | ~0.005 | ~0.0 (Converged) |
| TwINEst++ | ~0.002 | ~0.0 (Fastest Convergence) |

Convergence on WideResNet Jacobian matrix ($3072 \times 100$): TwINEst++ achieves extremely low error at around 100 matrix-vector products, consistent with the low-rank property of the Jacobian. The adaptive power method still fails to converge after 500 products.

### Key Findings

- On CIFAR-100, $\|·\|_{2\to\infty}$ regularization improves the test accuracy by 1.8% (77.3 vs 75.5), increases PGD adversarial accuracy by 2.8%, and reduces the stable rank by 42.8% (18.3 vs 32.0).
- Regularization with other norms (Frobenius, spectral, $\infty$) yields limited or even negligible improvements, highlighting the advantage of $\|·\|_{2\to\infty}$ for fine-grained row-wise control of the Jacobian.
- In recommender systems, $\|\hat{R}\|_{2\to\infty}^2$ regularization improves robustness under medium to strong attacks across all three datasets.
- TwINEst++ significantly outperforms TwINEst on low-rank matrices, aligning with theoretical expectations.

## Highlights & Insights

- Proving the divergence of the adaptive power method with a counterexample serves as a strong motivation, exposing a fundamental limitation of existing methods.
- The "coarse-estimation-then-exact-computation" strategy of TwINEst (using Hutchinson to identify candidate rows, then exactly computing the row norm) is exceptionally simple and efficient.
- Improving the complexity from $O(1/\Delta^2)$ to $O(1/\Delta)$ is highly significant for hard instances with small $\Delta$.
- The performance of $\|·\|_{2\to\infty}$ norm regularization on image classification is surprisingly strong, providing a novel alternative for Jacobian regularization.

## Limitations & Future Work

- The lower bound for oracle complexity has not yet been established, and the optimality of TwINEst/TwINEst++ remains unknown.
- Deep learning experiments are only validated on WideResNet-16-10, lacking evaluation on a wider range of architectures (such as ResNet, ViT, etc.).
- The adversarial attack configurations in the recommender system experiments are relatively simple (single-step FGSM-like attacks), and the effectiveness under stronger attacks requires further verification.
- Calculating the regularization term still incurs some computational overhead (equivalent to several backpropagation passes), which, although mitigateable by updating every $k$ steps, introduces a new hyperparameter.

## Related Work & Insights

- **vs Adaptive Power Method (Higham/Roth)**: The power method lacks theoretical guarantees and can diverge, whereas TwINEst provides an oracle complexity guarantee and converges stably in experiments.
- **vs Frobenius/Spectral Norm Regularization**: These classic regularization methods do not distinguish between the individual rows of the Jacobian, whereas $\|·\|_{2\to\infty}$ focuses on the maximum row norm, providing more refined control that is especially effective on tall-and-skinny matrices.

## Rating

- Novelty: ⭐⭐⭐⭐ This is the first matrix-free estimation algorithm for $\|·\|_{2\to\infty}$ with theoretical guarantees, and its application to DNN regularization is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering three major scenarios—synthetic experiments, DNN classification, and recommender systems—the theory and empirical results validate each other.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivations are rigorous and clear, the counterexample is elegantly constructed, and the experimental evaluation is comprehensive.
- Value: ⭐⭐⭐⭐ The algorithm is simple and practical with a solid theoretical foundation, demonstrating valuable real-world potential in DNN training and recommender systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] EFX and PO Allocation Exists for Two Types of Goods](efx_and_po_allocation_exists_for_two_types_of_goods.md)
- [\[ICLR 2026\] Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression](../../ICLR2026/ai_safety/differentially_private_two-stage_gradient_descent_for_instrumental_variable_regr.md)
- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)
- [\[CVPR 2026\] DeepfakeImpact: A Two-Stage Benchmark with Real-World Impact in Deepfake Detection](../../CVPR2026/ai_safety/deepfakeimpact_a_two-stage_benchmark_with_real-world_impact_in_deepfake_detectio.md)
- [\[ICML 2026\] Two Blind Spots in Machine Unlearning: Over-Unlearning and Prototype Re-learning Attacks](../../ICML2026/ai_safety/unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)

</div>

<!-- RELATED:END -->
