---
title: >-
  [Paper Note] Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach
description: >-
  [ICLR 2026 Oral][LLM Safety][machine unlearning] This paper proposes $(\phi,\varepsilon)$-Gaussian certifiability — a high-dimensional machine unlearning privacy framework grounded in hypothesis testing trade-off functions. It rigorously proves that, in the high-dimensional proportional regime ($p \sim n$), a single Newton step combined with calibrated Gaussian noise simultaneously satisfies privacy (GPAR) and accuracy (GED→0) requirements. The work refutes the conclusion of Zou et al. (2025) that "at least two Newton steps are necessary," and theoretically identifies the fundamental incompatibility between the classical $\varepsilon$-certifiability and noise-addition mechanisms.
tags:
  - ICLR 2026 Oral
  - LLM Safety
  - machine unlearning
  - Gaussian certifiability
  - hypothesis testing
  - high-dimensional statistics
  - Newton method
  - privacy
date: 2026-05-08
content_hash: 15c6acd75931bf07
---

# Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach

**Conference**: ICLR 2026 Oral
**arXiv**: [2510.13094](https://arxiv.org/abs/2510.13094)
**Code**: [Anonymous Repository](https://anonymous.4open.science/r/unlearning-E14D)
**Area**: AI Safety / Machine Unlearning / High-Dimensional Statistics
**Keywords**: machine unlearning, Gaussian certifiability, hypothesis testing, high-dimensional statistics, Newton method, privacy

## TL;DR

This paper proposes $(\phi,\varepsilon)$-Gaussian certifiability — a high-dimensional machine unlearning privacy framework grounded in hypothesis testing trade-off functions. It rigorously proves that, in the high-dimensional proportional regime ($p \sim n$), a single Newton step combined with calibrated Gaussian noise simultaneously satisfies privacy (GPAR) and accuracy (GED→0) requirements. The work refutes the conclusion of Zou et al. (2025) that "at least two Newton steps are necessary," and theoretically identifies the fundamental incompatibility between the classical $\varepsilon$-certifiability and noise-addition mechanisms.

## Background & Motivation

**Legal Drivers for Data Unlearning**: Regulations such as GDPR and CCPA mandate a "right to be forgotten," requiring models to efficiently erase the statistical influence of specific user data; full retraining is prohibitively expensive.

**Mainstream Newton-Based Unlearning**: Guo et al. (2020) and Sekhari et al. (2021) established that in the low-dimensional regime ($p \ll n$), a single Newton step with noise injection suffices for privacy and accuracy guarantees. However, their proofs rely on $\Omega(1)$ strong convexity and $O(1)$ smoothness of the per-example loss.

**Breakdown of Low-Dimensional Assumptions in High Dimensions**: Taking Ridge regression as a canonical example, when $p \sim n$, requiring $\|x_i\|_2 \sim 1$ (to ensure $O(1)$ smoothness) drives the minimum eigenvalue of the per-example loss Hessian down to $2\lambda/n$, entirely destroying $\Omega(1)$ strong convexity and rendering existing frameworks invalid.

**High-Dimensional Attempt by Zou et al. (2025)**: This work relaxes some optimization assumptions but adopts $(\phi,\varepsilon)$-PAR certifiability, concluding that even deleting a single data point requires at least two Newton iterations to simultaneously guarantee privacy and accuracy.

**Fundamental Deficiency of the Old Definition**: The $\varepsilon$-certifiability criterion is incompatible with noise-addition strategies — in high dimensions, it requires injecting disproportionately large noise to satisfy the privacy condition, thereby destroying model accuracy.

**Core Insight**: In high dimensions, broad families of isotropic log-concave noise mechanisms converge in behavior to the Gaussian mechanism (Dong et al., 2021). The Gaussian trade-off curve is therefore the canonical choice for high-dimensional privacy proofs, and redefining certifiability accordingly resolves the aforementioned contradiction.

## Method

### Overall Architecture

Given training data $\mathcal{D}_n$, a trained model $\hat{\beta} = A(\mathcal{D}_n)$, and a deletion subset $\mathcal{D}_\mathcal{M}$, the unlearning algorithm proceeds in two steps:

1. **Approximation Step (Newton Step)**: Starting from $\hat{\beta}$, perform a single Newton update on the objective $L_{\setminus\mathcal{M}}$ with $\mathcal{M}$ removed:

$$\hat{\beta}^{(1)}_{\setminus\mathcal{M}} = \hat{\beta} - G(L_{\setminus\mathcal{M}})^{-1}(\hat{\beta}) \nabla L_{\setminus\mathcal{M}}(\hat{\beta})$$

2. **Randomization Step (Noise Injection)**: Add calibrated Gaussian noise:

$$\tilde{\beta}_{\setminus\mathcal{M}} = \hat{\beta}^{(1)}_{\setminus\mathcal{M}} + b, \quad b \sim \mathcal{N}(0, \frac{r^2}{\varepsilon^2} I_p)$$

### Key Design 1: $(\phi,\varepsilon)$-Gaussian Certifiability (GPAR)

The core idea formalizes the privacy guarantee of unlearning as a hypothesis testing problem: an adversary observes the post-unlearning model output and attempts to distinguish between "retrained on full data with noise added" and "unlearned from the original model with noise added." The trade-off function is defined as:

$$T(P,Q)(\alpha) = \inf_{\phi}\{\beta_\phi : \alpha_\phi \leq \alpha\}$$

The Gaussian trade-off curve is adopted as the reference:

$$f_{G,\varepsilon}(\alpha) = \Phi(\Phi^{-1}(1-\alpha) - \varepsilon)$$

If the unlearning algorithm satisfies $T(\mathcal{P}_{re}, \mathcal{P}_{un})(\alpha) \geq f_{G,\varepsilon}(\alpha)$ with probability $\geq 1-\phi$, it is said to satisfy $(\phi,\varepsilon)$-GPAR. Advantages of GPAR:

- **Dimension Independence**: For any $p$-dimensional Gaussian vector, the trade-off depends only on the $\ell_2$ norm of the mean difference divided by the standard deviation, independent of dimension.
- **Canonicity**: Broad isotropic log-concave noise mechanisms converge to Gaussian behavior as $p \to \infty$ (CLT for DP).
- **Tightness**: In the sense of the Blackwell ordering, the Gaussian trade-off curve is the tightest characterization of the Gaussian mechanism.
- **All prior certifiability notions ($\varepsilon$-$\delta$, Rényi, etc.) are suboptimal under the Gaussian mechanism.**

### Key Design 2: Generalized Error Divergence (GED)

GED quantifies the generalization gap between the unlearned model and the ideally retrained model on new data:

$$\text{GED}_\ell(A, \bar{A}; \mathcal{M}, \mathcal{D}_n) = \mathbb{E}[|\ell(y_0 | x_0^\top A(\mathcal{D}_{\setminus\mathcal{M}})) - \ell(y_0 | x_0^\top \tilde{\beta}_{\setminus\mathcal{M}})| \mid \mathcal{D}_n]$$

Compared to the excess risk based on the true population risk minimizer used by Sekhari et al. (2021), GED exhibits more stable behavior in the high-dimensional proportional regime.

### Key Design 3: Relaxed Optimization Assumptions

Rather than requiring the per-example loss $f(\beta, z_i)$ to satisfy $\Omega(1)$ strong convexity and $O(1)$ smoothness, the paper requires:

- The loss $\ell$ is convex; the regularizer $r$ is strongly convex ($\nu$-strongly convex, $\nu = \Theta(1)$).
- $\ell$ and $r$ are three-times differentiable with polynomial growth.
- Features $x_i$ are sub-Gaussian; responses $y_i$ have sub-polynomial logarithmic moments.
- This covers a broad model class including Ridge regression, Logistic regression, and Poisson regression.

### Core Theoretical Results

**Theorem 2 (Privacy)**: Under the above assumptions, with noise variance set to $r^2/\varepsilon^2$ where $r = C_1(n)\sqrt{C_2(n)m^3/(2\lambda\nu n)}$, the single-step Newton unlearning satisfies $(\phi_n, \varepsilon)$-GPAR with $\phi_n \to 0$.

**Theorem 3 (Accuracy)**: Under the same setting, GED satisfies:

$$\text{GED}(\tilde{\beta}_{\setminus\mathcal{M}}, \hat{\beta}_{\setminus\mathcal{M}}) = O_p\left(\frac{m^2 \cdot \text{polylog}(n)}{\sqrt{n}}\right)$$

When $m = o(n^{1/4-\alpha})$, GED → 0, meaning $m$ data points can be simultaneously deleted while maintaining accuracy.

## Key Experimental Results

### Main Results: GED vs. Dimension (Logistic Regression, $n=p$, $\varepsilon=0.75$, $\lambda=0.5$)

| Deletion size $m$ | Noise type | log(GED) vs log(p) slope | GED behavior |
|:-----------------:|:----------:|:------------------------:|:------------:|
| 1 | Laplace (Zou) | 0.03 | Does not decay |
| 1 | Gaussian (Ours) | -0.47 | Decays $\sim p^{-0.5}$ |
| 5 | Laplace (Zou) | -0.03 | Does not decay |
| 5 | Gaussian (Ours) | -0.54 | Decays $\sim p^{-0.5}$ |
| 10 | Laplace (Zou) | -0.01 | Does not decay |
| 10 | Gaussian (Ours) | -0.51 | Decays $\sim p^{-0.5}$ |

**Core Finding**: Laplace noise (required by the Zou et al. framework) causes the GED of a one-step Newton unlearning to fail to decay, necessitating a second step; whereas Gaussian noise (GPAR framework) yields a GED that stably decays at rate $p^{-0.5}$, making one step sufficient.

### Ablation Study: GED vs. $\varepsilon$ and $m$ ($n=p=1255$)

| Experimental dimension | Observation |
|:----------------------:|:-----------:|
| Increasing $\varepsilon$ | Gaussian GED decreases monotonically toward retraining; Laplace GED remains substantially higher than Gaussian throughout |
| Increasing $m$ ($5 \to 50$) | GED increases for both; Laplace consistently and significantly exceeds Gaussian |
| $m$ vs. GED slope (Gaussian) | $\sim 1.4$ (empirically better than the theoretical bound of $m^{1.5}$) |
| $m$ vs. GED slope (Laplace) | $\sim 0.24$ (slower growth but higher absolute values due to excessive noise) |

### Key Findings

1. **Root Cause of One-Step vs. Two-Step Divergence Confirmed**: Theoretical predictions align precisely with experiments — the discrepancy originates from the choice of certifiability definition, not the algorithm itself.
2. **Dimensional Advantage of Gaussian Noise**: As $p$ increases, accuracy under the Gaussian scheme improves continuously, while the Laplace scheme stagnates.
3. **Feasibility of Multi-Point Deletion**: When $m = o(n^{1/4})$, simultaneous deletion of multiple users' data still preserves accuracy.

## Highlights & Insights

- **Conceptual Breakthrough**: The problem does not lie in an insufficient number of Newton steps, but rather in the inappropriate choice of certifiability concept. The classical $\varepsilon$-certifiability artificially forces disproportionate noise injection, degrading accuracy and producing the spurious "two-step necessity" conclusion.
- **Theoretical Unification**: Through the hypothesis testing trade-off function framework, Gaussian Differential Privacy (Dong et al., 2022) from the differential privacy literature is imported into machine unlearning, establishing an elegant bridge between the two fields.
- **Practicality**: A single Newton step requires only one Hessian-inverse-vector product, saving approximately half the computation compared to two-step methods.
- **Blessing of Dimensionality**: High dimensionality is not merely a challenge — CLT effects make Gaussian certifiability a natural and optimal choice, with GPAR becoming more accurate as dimension grows.

## Limitations & Future Work

1. **Convexity/Strong Convexity Assumptions**: The theory applies only to convex loss with strongly convex regularization (RERM); non-convex optimization in deep learning is not covered.
2. **GLM Data Assumptions**: Sub-Gaussian features and GLM-linked responses are required; real-world high-dimensional data may not satisfy these conditions.
3. **Hessian Computation Cost**: Although only a single Newton step is needed, computing and storing the Hessian inverse remains expensive for large-scale models.
4. **Limited Experimental Scale**: Simulation validation is conducted on Logistic regression with $p \leq 5000$; no evaluation on realistic deep models or LLMs is provided.
5. **Loose Multi-Point Deletion Bound**: The empirically observed GED growth slope with respect to $m$ ($\sim 1.4$) is better than the theoretical bound ($m^{1.5}$), suggesting the bound may be further tightened.

## Related Work & Insights

- **Low-Dimensional Machine Unlearning**: Guo et al. (2020) and Sekhari et al. (2021) prove that a single Newton step with noise suffices when $p \ll n$, but their assumptions break down when $p \sim n$.
- **High-Dimensional Unlearning**: Zou et al. (2025) first study the $p \sim n$ proportional regime but use $\varepsilon$-certifiability with Laplace noise, concluding that at least two steps are necessary.
- **Gaussian DP**: Dong et al. (2022) introduce Gaussian Differential Privacy ($f$-DP); this paper imports that framework into machine unlearning.
- **Exact Unlearning**: Bourtoule et al. (2021) and Cao & Yang (2015) pursue exact retraining equivalence at high computational cost.
- **Gradient-Descent-Based Unlearning**: Neel et al. (2021) and Allouah et al. (2025) analyze approximate unlearning via GD/SGD.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The new certifiability framework fundamentally reshapes the theoretical landscape of high-dimensional unlearning.
- Experimental Thoroughness: ⭐⭐⭐ Simulation results robustly support the theory, but large-scale real-world experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear; motivation is thoroughly articulated; comparative analysis is precise.
- Value: ⭐⭐⭐⭐⭐ The work delivers a paradigm-level contribution to machine unlearning theory; ICLR Oral recognition is well deserved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[CVPR 2026\] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](../../CVPR2026/llm_safety/pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](unlearning_evaluation_through_subset_statistical_independence.md)

</div>

<!-- RELATED:END -->
