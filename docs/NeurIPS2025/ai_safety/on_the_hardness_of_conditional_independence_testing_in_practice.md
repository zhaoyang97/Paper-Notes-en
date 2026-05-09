---
title: >-
  [Paper Note] On the Hardness of Conditional Independence Testing In Practice
description: >-
  [NeurIPS 2025][AI Safety][conditional independence testing] This paper systematically analyzes the root causes of failure in kernel-based conditional independence (CI) testing in practice: estimation error in conditional mean embeddings is identified as the central driver of Type-I error inflation, while the inherent tension between the choice of conditioning kernel $k_C$—which is critical for test power—and its tendency to exacerbate false positives is formally characterized.
tags:
  - NeurIPS 2025
  - AI Safety
  - conditional independence testing
  - kernel methods
  - KCI
  - GCM
  - Type-I error
  - conditional mean embedding
  - causal discovery
date: 2026-05-08
content_hash: cd1eda913f5a09a0
---

# On the Hardness of Conditional Independence Testing In Practice

**Conference**: NeurIPS 2025
**arXiv**: [2512.14000](https://arxiv.org/abs/2512.14000)
**Authors**: Zheng He (UBC), Roman Pogodin (McGill/Mila), Yazhe Li (Microsoft AI), Namrata Deka (CMU), Arthur Gretton (UCL Gatsby), Danica J. Sutherland (UBC/Amii)
**Code**: Not released
**Area**: AI Safety
**Keywords**: conditional independence testing, kernel methods, KCI, GCM, Type-I error, conditional mean embedding, causal discovery

## TL;DR

This paper systematically analyzes the root causes of failure in kernel-based conditional independence (CI) testing in practice: estimation error in conditional mean embeddings is identified as the central driver of Type-I error inflation, while the inherent tension between the choice of conditioning kernel $k_C$—which is critical for test power—and its tendency to exacerbate false positives is formally characterized.

## Background & Motivation

### State of the Field
Conditional independence (CI) testing is a fundamental task in machine learning and statistics, with broad applications in causal discovery (e.g., the PC algorithm), fairness evaluation of predictors (equalized odds), and out-of-distribution robustness checking. When the conditioning variable $C$ is discrete, the problem reduces to unconditional independence testing; however, when $C$ is continuous, smoothness assumptions on the conditional distribution become necessary, since only a single pair $(A, B)$ is observed per value of $C$.

### Limitations of Prior Work
- Shah & Peters (2020) established an impossibility theorem showing that no CI test with finite-sample valid level across all Lebesgue-continuous null distributions can achieve power beyond the significance level $\alpha$. However, this result relies on adversarial constructions involving "hidden dependence" (e.g., extracting the 30th decimal digit of $C$), and does not explain the widespread failure of CI tests in practice.
- Kernel-based conditional independence tests (KCI) have a reputation for poor Type-I error control, yet the underlying mechanism has not been well understood.
- Existing approaches such as SplitKCI attempt to mitigate Type-I error but fall far short of resolving the issue.
- Prior work implicitly assumes that the kernel used for regression is simultaneously suitable for measuring dependence, neglecting the importance of selecting the conditioning kernel $k_C$.

### Root Cause
Rather than remaining at the theoretical level of impossibility theorems, this paper investigates the concrete mechanisms by which KCI and GCM-type tests fail in practice: (1) how estimation error in conditional mean embeddings causes Type-I error inflation; and (2) how the choice of the conditioning kernel $k_C$ creates an irreconcilable tension between test power and Type-I error control.

## Method

### A Unified Framework for KCI and GCM
The paper first reformulates conditional independence via a new theorem (Theorem 2.2): $A \perp\!\!\!\perp B \mid C$ if and only if, for all $L^2$ functions $f, g, w$,
$$\mathbb{E}_C\left[w(C) \cdot \mathbb{E}_{AB|C}\left[(f(A) - \mathbb{E}[f(A)|C])(g(B) - \mathbb{E}[g(B)|C]) \mid C\right]\right] = 0$$

Under this framework, the KCI statistic is defined as the squared Hilbert–Schmidt norm of the KCI operator:
$$\text{KCI} = \|\mathfrak{C}_{\text{KCI}}\|_{\text{HS}}^2 = \mathbb{E}_{C,C'}\left[k_C(C,C') \langle \mathfrak{C}_{AB|C}(C), \mathfrak{C}_{AB|C}(C') \rangle_{\text{HS}}\right]$$

**Key finding: GCM is almost a special case of KCI.** When scalar linear kernels $\phi_A(a)=a$ and $\phi_B(b)=b$ are used for $A$ and $B$, and $k_C(c,c')=w(c)w(c')$, KCI reduces to the population version of the (weighted) GCM. The standard GCM corresponds to $w(c)=1$ (i.e., $\ell_C=\infty$). This connection is analogous to the relationship between classifier two-sample tests and MMD tests.

### The Core Theoretical Difficulty: Estimating Conditional Mean Embeddings
Proposition 4.1 establishes that **if the true conditional mean embeddings $\mu_{A|C}$ and $\mu_{B|C}$ are known, a finite-sample valid and consistent test can be constructed**, thereby circumventing the Shah & Peters impossibility theorem. Specifically, via Hoeffding's inequality, rejecting the null hypothesis when $\text{KCI}_n > 32\kappa_A\kappa_B\kappa_C\sqrt{\frac{1}{n-1}\log\frac{1}{\alpha}}$ yields a test with level at most $\alpha$.

This demonstrates that the theoretical difficulty of CI testing **stems entirely** from the estimation of conditional mean embeddings, rather than from the test statistic itself.

### The Critical Role of the Conditioning Kernel $k_C$
A synthetic experiment (Problem 7) analyzes the setting where the conditional covariance $\gamma(C)=\sin(\beta C)$. GCM (with $\ell_C=\infty$) completely fails to detect dependence by averaging $\gamma$ globally (since $\mathbb{E}_C[\gamma(C)]=0$). An analytic derivation yields the KCI value:
$$\text{KCI} = \frac{1}{2}\tau^4 e^{-\beta^2}\sqrt{\frac{\ell_C^2}{\ell_C^2+2}}\left(e^{2\beta^2/(\ell_C^2+2)} - 1\right)$$
An optimal $\ell_C^*$ balances two competing effects: too small an $\ell_C$ causes the kernel weight term to vanish, while too large an $\ell_C$ destroys the ability to localize the covariance.

Drawing on kernel selection strategies from unconditional testing, the paper proposes maximizing the signal-to-noise ratio $\widehat{\text{SNR}} = \widehat{\text{KCI}} / \hat{\sigma}_{\mathfrak{H}_1}$ for selecting $k_C$, and establishes its consistency (Theorem 5.2).

### Effect of Regression Error on Type-I Error
Let the estimation error be $\Delta_{A|C} = \hat{\mu}_{A|C} - \mu_{A|C}$. Under the null hypothesis:
$$\mathbb{E}[\widehat{\text{KCI}}_n] = \mathbb{E}\left[k_C(C,C')\langle\Delta_{A|C}(C), \Delta_{A|C}(C')\rangle \langle\Delta_{B|C}(C), \Delta_{B|C}(C')\rangle\right]$$
This quantity is generally nonzero, inducing a positive bias. More critically, $\nu_1 > 0$ causes the variance decay to degrade from $\Theta(1/n)$ to $\Theta(1/\sqrt{n})$.

**Theorem 6.2** provides a formal upper bound on Type-I error inflation due to regression error: the probability that the test statistic exceeds the nominal threshold $q/n$ is bounded in terms of $n\widehat{\text{KCI}}$ and $n^2\text{Var}(\widehat{\text{KCI}}_n)$. Maintaining correct asymptotic calibration requires regression errors satisfying $\widehat{\text{KCI}} = o(1/n)$ and $\nu_1 = o(1/n)$.

**Theorem 6.3** further analyzes the approximation error of the wild bootstrap, showing that the Kolmogorov distance between the bootstrap statistic $Y$ and the normal approximation $nZ_n$ is controlled by the standardized mean shift $b_{\widehat{\text{KCI}}}$ and the variance mismatch $\kappa_{\text{var}}$.

## Key Experimental Results

### Experiment 1: Effect of $k_C$ Selection on Type-I/II Error in Synthetic Data

Using the synthetic data from Problem (7) (with $f_A=\cos$, $f_B=\exp$, $\tau=0.1$), the paper analyzes the effect of the kernel length scale $\ell_C^2$ on test behavior.

| Training size $m$ | Range of $\ell_C^2$ | Type-I error | Type-II error | Observation |
|:---:|:---:|:---:|:---:|:---|
| 200 | Small $\ell_C^2$ | Substantially inflated (>0.05) | High | False positives uncontrolled under poor regression quality |
| 200 | Moderate $\ell_C^2$ | ~0.05 | Lowest | Theoretically optimal range |
| 1000 | All $\ell_C^2$ | ≤0.05 (stable) | Varies with $\ell_C^2$ | Type-I controlled under good regression quality |

The theoretical SNR curve closely matches the empirical power curve, validating SNR-based selection of $\ell_C^2$. However, power maximization tends to select regions with inflated Type-I error, exposing the inherent power–validity tension.

### Experiment 2: Two Scenarios with Multivariate Conditioning Variables

| Scenario | Description | Type-I error | Type-II error |
|:---|:---|:---:|:---:|
| Scenario 1: Shared coordinates | Regression and dependence use the same coordinates of $C$ | 0.21 | 0.0 |
| Scenario 2: Independent coordinates | Regression and dependence use different coordinates of $C$ | 0.10 | 0.08 |

In Scenario 1, regression error leaks correlated noise into the test statistic through the shared dimensions, causing Type-I error (0.21) to far exceed the nominal level (0.05). The independent dimensions in Scenario 2 reduce this leakage but sacrifice some power (Type-II increases from 0 to 0.08).

## Highlights & Insights

- **Unified perspective**: The paper provides the first rigorous proof that GCM (including weighted GCM) is almost a special case of KCI (linear kernel + specific $k_C$), establishing a deep connection between the two major classes of CI testing methods.
- **Precise diagnosis**: Proposition 4.1 demonstrates that the difficulty of CI testing stems entirely from estimating conditional mean embeddings rather than from the design of the test statistic itself—a more actionable insight than the Shah & Peters impossibility theorem.
- **Analytic and empirical integration**: Closed-form expressions for KCI are derived on synthetic problems, precisely characterizing how $\ell_C$ governs test behavior, with strong agreement between theory and experiment.
- **Exposing a fundamental tension**: A well-chosen $k_C$ is critical for power, yet power maximization systematically selects regimes where regression error causes Type-I inflation—a structural dilemma inherent to CI testing.

## Limitations & Future Work

- **No remedy proposed**: The work is primarily diagnostic; it identifies the problem but does not offer a practical method for mitigating Type-I error inflation.
- **Unresolved trade-off in kernel selection**: The proposed SNR-maximizing kernel selection strategy improves power but may exacerbate false positives; how to balance these objectives in practice remains an open question.
- **Focus on linear kernels**: The theoretical analysis of Type-I error is primarily conducted for linear $k_A$ and $k_B$, with insufficient coverage of more complex nonlinear kernel settings.
- **Predominantly synthetic experiments**: Although real-data experiments are mentioned (Appendix H.3), the main analysis relies on synthetic settings.
- **Fixed regressor assumption**: The theoretical analysis assumes fixed regression parameters and does not fully account for the randomness of the regressor introduced by train-test splitting.

## Related Work & Insights

- **Shah & Peters (2020)**: Established the CI testing impossibility theorem; the present paper precisely locates this impossibility in conditional mean embedding estimation and analyzes its concrete impact within KCI.
- **Zhang et al. (2012)**: Proposed KCI; the present paper reformulates the framework and identifies the implicit assumption that $k_C$ requires no dedicated selection.
- **Scheidl et al. (2023, SplitKCI)**: Mitigates Type-I error via sample splitting; the present paper demonstrates that splitting is insufficient, as regression error can still be amplified through $k_C$ selection.
- **Lundborg et al. (2022, Weighted GCM)**: Extends GCM via weight functions; the present paper shows this is essentially equivalent to a restricted choice of $k_C$.
- **Gretton et al. (2012), Sutherland et al. (2017)**: Kernel selection strategies in unconditional testing; the present paper extends analogous methods to CI testing but notes that Type-I control guarantees cannot be inherited due to the absence of permutation testing.

## Rating

- Novelty: ⭐⭐⭐⭐ — The unified perspective connecting KCI and GCM is original, and the precise identification of conditional mean embedding estimation as the core difficulty offers deep insight.
- Experimental Thoroughness: ⭐⭐⭐ — Synthetic experiments are cleverly designed with high theory–experiment agreement, but real-data validation is insufficient.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-structured, theoretically rigorous, and logically progressive from motivation to analysis.
- Value: ⭐⭐⭐⭐ — Provides the clearest diagnosis to date of the core difficulties in CI testing, with significant practical implications for causal discovery and fairness testing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack](taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[NeurIPS 2025\] Causally Reliable Concept Bottleneck Models](causally_reliable_concept_bottleneck_models.md)
- [\[NeurIPS 2025\] A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers](a_set_of_generalized_components_to_achieve_effective_poison-only_clean-label_bac.md)

</div>

<!-- RELATED:END -->
