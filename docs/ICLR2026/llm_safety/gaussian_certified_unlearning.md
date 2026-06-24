---
title: >-
  [Paper Note] Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach
description: >-
  [ICLR 2026 Oral][LLM Safety][machine unlearning] Ours proposes $(\phi,\varepsilon)$-Gaussian certifiability—a high-dimensional machine unlearning privacy framework based on hypothesis testing trade-off functions. It rigorously proves that in the high-dimensional proportional regime ($p \sim n$), a single-step Newton update combined with calibrated Gaussian noise can simultaneously satisfy privacy (GPAR) and accuracy (GED $\to 0$) requirements. This result refutes the conclusi…
tags:
  - "ICLR 2026 Oral"
  - "LLM Safety"
  - "machine unlearning"
  - "Gaussian certifiability"
  - "hypothesis testing"
  - "high-dimensional statistics"
  - "Newton method"
  - "privacy"
date: 2026-05-08
content_hash: edf58303f8a5319c
---

# Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach

**Conference**: ICLR 2026 Oral  
**arXiv**: [2510.13094](https://arxiv.org/abs/2510.13094)  
**Code**: [Anonymous Repository](https://anonymous.4open.science/r/unlearning-E14D)  
**Area**: AI Safety / Machine Unlearning / High-dimensional Statistics  
**Keywords**: machine unlearning, Gaussian certifiability, hypothesis testing, high-dimensional statistics, Newton method, privacy

## TL;DR

Ours proposes $(\phi,\varepsilon)$-Gaussian certifiability—a high-dimensional machine unlearning privacy framework based on hypothesis testing trade-off functions. It rigorously proves that in the high-dimensional proportional regime ($p \sim n$), a single-step Newton update combined with calibrated Gaussian noise can simultaneously satisfy privacy (GPAR) and accuracy (GED $\to 0$) requirements. This result refutes the conclusion of Zou et al. (2025) that "at least two Newton steps are required" and theoretically reveals the fundamental reason why the old $\varepsilon$-certifiability is incompatible with noise injection mechanisms.

## Background & Motivation

**Legal Drivers for Data Unlearning**: Regulations such as GDPR and CCPA mandate the "right to be forgotten." Models must efficiently remove the statistical impact of specific user data, as full retraining is computationally prohibitive.

**Mainstream Newton Unlearning Methods**: Guo et al. (2020) and Sekhari et al. (2021) demonstrated that in low dimensions ($p \ll n$), single-step Newton updates plus noise ensure both privacy and accuracy. However, their proofs rely on $\Omega(1)$ strong convexity and $O(1)$ smoothness assumptions for per-example loss.

**Collapse of Low-Dimensional Assumptions in High Dimensions**: Taking simple Ridge regression as an example, when $p \sim n$, requiring $\|x_i\|_2 \sim 1$ (to ensure $O(1)$ smoothness) causes the minimum eigenvalue of the per-example loss to drop to $2\lambda/n$, completely destroying $\Omega(1)$ strong convexity and rendering existing frameworks invalid.

**High-Dimensional Attempts by Zou et al. (2025)**: While relaxing some optimization assumptions, they adopted $(\phi,\varepsilon)$-PAR certifiability. Their conclusion suggested that even for deleting a single data point, at least two Newton iterations are required to guarantee both privacy and accuracy.

**Intrinsic Flaw of the Old Definition**: $\varepsilon$-certifiability is incompatible with noise injection strategies. In high dimensions, it necessitates injecting disproportionately large noise to satisfy privacy conditions, which collapses model accuracy.

**Key Insight**: In high dimensions, a broad class of isotropic log-concave noise mechanisms converges in behavior to the Gaussian mechanism (Dong et al., 2021). Therefore, the Gaussian trade-off curve is the canonical choice for high-dimensional privacy proofs. Redefining certifiability based on this solves the aforementioned contradiction.

## Method

### Overall Architecture

The paper addresses machine unlearning in the high-dimensional proportional regime ($p \sim n$, where parameters and samples are of the same order). Given training data $\mathcal{D}_n$, a trained model $\hat{\beta} = A(\mathcal{D}_n)$, and a subset to be deleted $\mathcal{D}_\mathcal{M}$, the unlearning algorithm is minimalist, consisting of two steps: first, using $\hat{\beta}$ as the initial value, perform a **single-step** Newton update on the target $L_{\setminus\mathcal{M}}$ (excluding $\mathcal{M}$) to obtain the approximate solution $\hat{\beta}^{(1)}_{\setminus\mathcal{M}} = \hat{\beta} - G(L_{\setminus\mathcal{M}})^{-1}(\hat{\beta})\, \nabla L_{\setminus\mathcal{M}}(\hat{\beta})$; second, add calibrated Gaussian noise $\tilde{\beta}_{\setminus\mathcal{M}} = \hat{\beta}^{(1)}_{\setminus\mathcal{M}} + b$, where $b \sim \mathcal{N}(0, \tfrac{r^2}{\varepsilon^2} I_p)$.

The true contribution lies not in the two-step algorithm itself, but in changing the "yardstick" for measuring privacy—certifiability defined by the Gaussian trade-off function. The four designs follow a "metrics then proof" structure: the first two redefine privacy (GPAR) and accuracy (GED) criteria, the third relaxes assumptions to enable proofs in high dimensions, and the fourth proves that single-step Newton + Gaussian noise is sufficient under these new criteria.

### Key Designs

**1. $(\phi,\varepsilon)$-Gaussian certifiability (GPAR): Reformulating privacy as a hypothesis testing problem**

The old $\varepsilon$-certifiability required the "unlearned model" and "retrained model" to be parameter-wise close. In high dimensions, this forces the injection of disproportionate noise. Ours formalizes privacy as a hypothesis test: an adversary observes the unlearned output and attempts to distinguish between "retraining from scratch plus noise ($\mathcal{P}_{re}$)" and "unlearning from the original model plus noise ($\mathcal{P}_{un}$)". The adversary's optimal distinguishing ability is characterized by the trade-off function $T(P,Q)(\alpha) = \inf_{\phi}\{\beta_\phi : \alpha_\phi \leq \alpha\}$. Ours selects the Gaussian curve $f_{G,\varepsilon}(\alpha) = \Phi(\Phi^{-1}(1-\alpha) - \varepsilon)$ as the benchmark. If $T(\mathcal{P}_{re}, \mathcal{P}_{un})(\alpha) \geq f_{G,\varepsilon}(\alpha)$ holds with probability $\geq 1-\phi$, the algorithm satisfies $(\phi,\varepsilon)$-GPAR.

**2. Generalized Error Divergence (GED): Stable accuracy measurement in high dimensions**

Sekhari et al. (2021)'s "excess risk" relative to the "true population minimizer" is unstable when $p \sim n$ because the empirical solution and the true optimal solution have non-negligible bias. Ours uses GED to directly compare the prediction loss difference between the unlearned model and the ideal retrained model on a **new sample**:

$$\text{GED}_\ell(A, \bar{A}; \mathcal{M}, \mathcal{D}_n) = \mathbb{E}\big[\,|\ell(y_0 \mid x_0^\top A(\mathcal{D}_{\setminus\mathcal{M}})) - \ell(y_0 \mid x_0^\top \tilde{\beta}_{\setminus\mathcal{M}})| \mid \mathcal{D}_n\,\big]$$

This avoids dependence on the true optimal solution and remains stable in high-dimensional proportional regimes.

**3. Relaxed Optimization Assumptions: Covering high-dimensional losses**

Unlike low-dimensional proofs requiring per-example strong convexity, Ours shifts assumptions to the "global target": the loss $\ell$ is convex, the regularizer $r$ is $\nu$-strongly convex with $\nu = \Theta(1)$, and features $x_i$ are sub-Gaussian. By not requiring per-example strong convexity, these relaxed conditions cover Generalized Linear Models (GLMs) like Ridge, Logistic, and Poisson regression in high dimensions.

**4. Privacy-Accuracy Guarantees for Single-Step Unlearning**

Experimental noise variance is precisely calibrated as $r^2/\varepsilon^2$ where $r$ depends on the number of deleted points $m$. Theorem 2 proves this single-step Newton unlearning satisfies $(\phi_n, \varepsilon)$-GPAR. Theorem 3 proves $\text{GED}(\tilde{\beta}_{\setminus\mathcal{M}}, \hat{\beta}_{\setminus\mathcal{M}}) = O_p\big(\tfrac{m^2 \cdot \text{polylog}(n)}{\sqrt{n}}\big)$. Combined, they show that as long as Laplace noise is replaced by Gaussian noise, **one step** of Newton is sufficient.

## Key Experimental Results

### Main Results: GED vs. Dimension (Logistic Regression, $n=p$, $\varepsilon=0.75$, $\lambda=0.5$)

| Deletion count $m$ | Noise Type | log(GED) vs log(p) Slope | GED Behavior |
|:-----------:|:--------:|:----------------------:|:--------:|
| 1 | Laplace (Zou) | 0.03 | No decay |
| 1 | Gaussian (Ours) | -0.47 | Decays as $\sim p^{-0.5}$ |
| 5 | Laplace (Zou) | -0.03 | No decay |
| 5 | Gaussian (Ours) | -0.54 | Decays as $\sim p^{-0.5}$ |
| 10 | Laplace (Zou) | -0.01 | No decay |
| 10 | Gaussian (Ours) | -0.51 | Decays as $\sim p^{-0.5}$ |

**Core Conclusion**: Laplace noise (required by Zou et al.) results in non-decaying GED for single-step Newton, necessitating a second step. Gaussian noise (GPAR framework) allows GED to decay at $p^{-0.5}$ with just one step.

### Ablation Study: GED vs. $\varepsilon$ and $m$ ($n=p=1255$)

| Observation | Result |
|:--------:|:--------|
| $\varepsilon$ Increase | Gaussian GED monotonically decreases; Laplace remains significantly higher. |
| $m$ Increase ($5 \to 50$) | Both GEDs increase, but Laplace remains consistently higher than Gaussian. |
| $m$ vs GED Slope (Gaussian) | $\sim 1.4$ (better than the theoretical $m^{1.5}$ bound). |
| $m$ vs GED Slope (Laplace) | $\sim 0.24$ (slow growth but higher absolute value due to excessive noise). |

### Key Findings

1. **Root cause of 1-step vs. 2-step divergence**: Confirmed that the discrepancy stems from the certifiability definition (Gaussian vs. $\varepsilon$) rather than the unlearning algorithm.
2. **Dimensional advantage of Gaussian noise**: As $p$ increases, the accuracy of the Gaussian scheme improves, whereas the Laplace scheme stagnates.
3. **Feasibility of multi-point deletion**: When $m = o(n^{1/4})$, accuracy is preserved even when deleting multiple users at once.

## Highlights & Insights

- **Conceptual Breakthrough**: The issue is not an insufficient number of Newton steps, but the improper choice of certifiability definition. The old $\varepsilon$-certifiability requires disproportionate noise, artificially degrading accuracy.
- **Theoretical Unification**: Leveraging the hypothesis testing trade-off framework, Ours introduces Gaussian DP concepts into machine unlearning, building an elegant bridge between the two fields.
- **Practicality**: The computational cost of single-step Newton is approximately half that of the two-step solution.
- **Benefit of Dimensionality**: High dimensionality is not just a challenge—the CLT effect makes Gaussian certifiability a natural and optimal choice.

## Limitations & Future Work

1. **Convexity Assumptions**: Theory applies only to RERM with convex loss and strongly convex regularization; non-convex deep learning is not covered.
2. **GLM Data Assumptions**: Requires sub-Gaussian features and GLM relationships, which real-world high-dimensional data may violate.
3. **Hessian Computational Cost**: While only one step is needed, computing/storing the Hessian inverse is still expensive for very large models.
4. **Limited Experimental Scale**: Validated on Logistic regression up to $p=5000$; not tested on large-scale deep models or LLMs.
5. **Loose Multi-point Bound**: Experiments show a GED growth slope of $\sim 1.4$ regarding $m$, which is better than the theoretical $m^{1.5}$ bound, suggesting potential for tighter theory.

## Related Work & Insights

- **Low-dimensional Unlearning**: Guo et al. (2020), Sekhari et al. (2021) prove single-step is enough when $p \ll n$.
- **High-dimensional Unlearning**: Zou et al. (2025) first studied $p \sim n$ but concluded two steps are needed due to $\varepsilon$-certifiability.
- **Gaussian DP**: Dong et al. (2022) proposed Gaussian DP ($f$-DP), which Ours adapts for unlearning.
- **Exact Unlearning**: Bourtoule et al. (2021) focus on exact retraining equivalence at high cost.
- **GD-based Unlearning**: Neel et al. (2021) analyze approximate unlearning via GD/SGD iterations.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ New certifiability framework fundamentally shifts the high-dimensional unlearning landscape.
- Experimental Thoroughness: ⭐⭐⭐ Simulation confirms theory well, though real-world scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation and clear motivation.
- Value: ⭐⭐⭐⭐⭐ Paradigm-shifting contribution to unlearning theory, fully deserving of an ICLR Oral.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Adversarial Attacks on High-dimensional Offline Bandits](efficient_adversarial_attacks_on_high-dimensional_offline_bandits.md)
- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)
- [\[NeurIPS 2025\] ModHiFi: Identifying High Fidelity Predictive Components for Model Modification](../../NeurIPS2025/llm_safety/modhifi_identifying_high_fidelity_predictive_components_for_model_modification.md)
- [\[ICLR 2026\] PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach](propensitybench_evaluating_latent_safety_risks_in_large_language_models_via_an_a.md)
- [\[AAAI 2026\] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](../../AAAI2026/llm_safety/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)

</div>

<!-- RELATED:END -->
