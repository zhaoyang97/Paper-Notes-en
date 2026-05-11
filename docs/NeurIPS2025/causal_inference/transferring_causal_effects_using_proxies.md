---
title: >-
  [Paper Note] Transferring Causal Effects using Proxies
description: >-
  [NEURIPS2025][Causal Inference][proximal causal inference] This paper proposes a multi-domain causal effect transfer method based on proxy variables. Given that only proxy variable $W$ is observed in the target domain…
tags:
  - "NEURIPS2025"
  - "Causal Inference"
  - "proximal causal inference"
  - "domain adaptation"
  - "unobserved confounders"
  - "proxy variables"
  - "interventional distribution"
date: 2026-05-08
content_hash: 86cf72b9c48d5a0e
---

# Transferring Causal Effects using Proxies

**Conference**: NEURIPS2025
**arXiv**: [2510.25924](https://arxiv.org/abs/2510.25924)
**Code**: [manueligal/proxy-intervention](https://github.com/manueligal/proxy-intervention)
**Area**: Causal Inference
**Keywords**: proximal causal inference, domain adaptation, unobserved confounders, proxy variables, interventional distribution

## TL;DR

This paper proposes a multi-domain causal effect transfer method based on proxy variables. Given that only proxy variable $W$ is observed in the target domain, the method leverages multi-source domain data to identify and estimate the interventional distribution under unobserved confounders in the target domain, and provides two consistent estimators with asymptotic confidence intervals.

## Background & Motivation

**Core Problem**: Estimating the causal effect of treatment variable $X$ on outcome $Y$ is a central goal in scientific research, yet the presence of unobserved confounders $U$ makes causal inference from observational data extremely challenging.

**Limitations of Prior Work**: Randomized controlled trials (RCTs) are the gold standard for causal inference but are often infeasible due to ethical or practical constraints. Existing proxy variable methods (proximal causal inference) typically assume that causal effects are invariant across domains and cannot handle distributional shifts between domains.

**Unique Challenges in the Multi-Domain Setting**: When the distribution of latent confounders $U$ varies across domains (latent shift), the causal effect of $X$ on $Y$ differs across domains, precluding direct application of single-domain proxy methods.

**Scarcity of Target Domain Data**: In the target domain, only the proxy variable $W$ may be observed, with $X$ and $Y$ unavailable, necessitating the transfer of causal information from source domains.

**Key Distinction from Prior Work**: Tsai et al. [2024] require observing both $W$ and $X$ in the target domain and target conditional mean prediction rather than the interventional distribution. This paper requires only $W$ in the target domain and aims to estimate the interventional distribution $Q(Y|\text{do}(x))$.

**Practical Application**: For example, studying the causal effect of website ranking on consumer choice — hotel attributes ($U$) influence ranking ($X$) and clicks ($Y$), price ($W$) serves as a proxy for $U$, and different hotels constitute different domains.

## Method

### Overall Architecture

The data-generating process considered in this paper is described by a structural causal model (SCM). Source domains provide observations $(E, W, X, Y)$, while the target domain provides only $W$. The core idea exploits the invertibility of the conditional distribution matrix $P(W|E,x)$ of proxy variable $W$ across domains, establishing the identifiability formula:

$$q(y|\text{do}(x)) = P(y|E,x) \cdot P(W|E,x)^\dagger \cdot Q(W)$$

where $\dagger$ denotes the right Moore–Penrose pseudoinverse.

### Key Designs

#### Module 1: Identifiability Theory

- **Function**: Proves under what conditions the interventional distribution $q(y|\text{do}(x))$ in the target domain can be uniquely determined from observable data.
- **Mechanism**: Starting from the covariate adjustment formula $q(y|\text{do}(x)) = \sum_u q(y|u,x) \cdot q(u)$, the unobservable $U$ is replaced by observables via matrix decomposition. The key step exploits the cross-domain invariance of $P(W|U)$ (modularity assumption) to rewrite $Q(y|U,x)$ in a form that depends only on $P(W|U)$, after which $U$ is marginalized out.
- **Design Motivation**: Assumption 1 requires $\text{rank}(P(W|E,x)) \geq k_U$, i.e., the conditional distribution of proxy $W$ across domains must be sufficiently "diverse" so that variation in $U$ induced by domain shift is fully reflected through $W$. This is a prerequisite for the existence of the pseudoinverse. The assumption is not only sufficient but also (in this setting) necessary — counterexamples can be constructed when it is violated, causing identifiability to fail.
- **Extensions**: Theorem 1 extends to continuous $X$ and $Y$, to settings with additional observed covariates $Z$ (enabling CATE identification), and to more general causal graphs allowing a direct $E \to X$ edge.

#### Module 2: Causal Parametrisation Estimator

- **Function**: Explicitly parameterizes the conditional probability matrices of all causal mechanisms in the SCM; the interventional distribution is computed after obtaining parameters via maximum likelihood estimation.
- **Mechanism**: Parameter $\theta$ comprises all entries of $P(U|E)$, $Q(U)$, $P(W|U)$, $P(X|U)$, and $P(y|U,W,x)$. A softmax transformation maps unconstrained logit parameters to valid probabilities. Using the formula from Proposition 3,
$$q(y|\text{do}(x)) = \text{diag}(P(y|U,W,x) \cdot P(W|U)) \cdot Q(U),$$
the estimated $\hat{\theta}$ is substituted to obtain $\hat{q}_{C,n}$.
- **Design Motivation**: Although some parameters in $\theta$ are not individually identifiable (overparameterization), the interventional distribution as a function of $\theta$ is identifiable. Proposition 4 establishes consistency — even if $\hat{\theta}$ itself does not converge to the true value, the induced observable distribution converges, which is sufficient to guarantee consistent estimation of causal effects.
- **Computational Note**: Requires non-convex optimization for MLE, incurring relatively higher computational cost.

#### Module 3: Reduced Parametrisation Estimator

- **Function**: Estimates only the minimal set of parameters required to compute Eq. 4, avoiding the computational burden of overparameterization.
- **Mechanism**: Directly based on the identifiability formula $q(y|\text{do}(x)) = P(y|E,x) \cdot P(W|E,x)^\dagger \cdot Q(W)$, the parameter vector $\eta$ contains only joint/conditional probabilities that can be directly estimated from data via empirical frequencies (e.g., $P(W,x,E)$, $P(y,x,E)$, $P(x,E)$, $Q(W,e_T)$). The estimator is $\hat{q}_{R,n}(y|\text{do}(x)) = h(\hat{\eta})$, where $h$ maps empirical probabilities to the interventional distribution.
- **Design Motivation**: Avoids the non-convex optimization in the causal parametrization. More importantly, Proposition 5 establishes asymptotic normality, enabling confidence interval construction via the delta method:
$$\hat{\sigma}^2 = \nabla h(\hat{\eta})^\top \cdot \hat{\Sigma} \cdot \nabla h(\hat{\eta}),$$
yielding $(1-\alpha)$-level asymptotic confidence intervals.
- **Practical Consideration**: Estimates and confidence intervals may fall outside $[0,1]$ and require clipping.

### Loss & Training

- **Causal Parametrisation**: Maximizes the conditional likelihood
$$L(\theta) = \prod p_\theta(y,x,w|e)^{n(y,x,w,e)} \cdot \prod q_\theta(w)^{n(w)},$$
with softmax-constrained parameter space and gradient-based optimization (non-convex).
- **Reduced Parametrisation**: No optimization required; $\hat{\eta} = (1/n)\sum \eta^i$ is estimated directly from empirical frequencies, with low computational complexity.
- **Condition Number Monitoring**: $\kappa(P(W|E,x))$ can be estimated from data ($\hat{\kappa} \approx \kappa$); when $\hat{\kappa}$ is large, the estimate is flagged as unreliable.

## Key Experimental Results

### Main Results

**Table 1: Simulation — Mean Absolute Error of Point Estimates ($n=20000$, $k_E=3$, $M=10$, $N=5$)**

| Method | Mean Absolute Error | Note |
|--------|---------------------|------|
| Oracle (interventional data) | Lowest | Uses unobservable interventional distribution data |
| Causal Estimator | 0.040 | Proposed causal parametrisation estimator |
| Reduced Estimator | 0.058 | Proposed reduced parametrisation estimator, faster |
| NoAdj (no adjustment) | Large bias | Directly uses observational distribution |
| NoAdj* (target domain, no adjustment) | Large bias | Uses target domain data (unavailable in practice) |
| WAdj ($W$ adjustment) | Large bias | Adjusts for $W$ in place of $U$ (invalid) |
| WAdj* (target domain $W$ adjustment) | Large bias | Uses target domain data (unavailable in practice) |

**Table 2: Hotel Ranking Real Data — Estimates of $q(Y=1|\text{do}(X=1))$ (25 source domains, 18 target domains)**

| Method | Mean Absolute Error | Median CI Length |
|--------|---------------------|-----------------|
| Reduced Estimator | **0.044** | **0.14** |
| NoAdj | 0.051 | 0.17 |
| NoAdj* | 0.080 | — |
| WAdj | 0.053 | — |
| WAdj* | 0.075 | — |

### Key Findings

1. Both estimators achieve substantially lower absolute error than all non-Oracle baselines, validating the practical effectiveness of the identifiability theory.
2. Estimation error grows with the condition number $\kappa(P(W|E,x))$, which can be accurately estimated from data ($\hat{\kappa} \approx \kappa$), providing a practical reliability diagnostic tool.
3. Coverage of asymptotic confidence intervals is close to the nominal level (95%), and interval length decreases with sample size, validating consistency.
4. On real Expedia hotel search data, the confidence intervals of the Reduced Estimator overlap with Oracle intervals across all 18 target domains.

## Highlights & Insights

1. **Solid Theoretical Contributions**: Identifiability of the interventional distribution is established in the extreme setting where only proxy variable $W$ is observed in the target domain; Assumption 1 is shown to be necessary, admitting no weaker alternative.
2. **Two Complementary Estimators**: The Causal Estimator achieves slightly lower error but requires non-convex optimization; the Reduced Estimator provides analytic asymptotic confidence intervals with high computational efficiency. Practitioners may select between them based on their requirements.
3. **Diagnosability**: The condition number $\kappa$ can be estimated directly from data, allowing users to assess estimation reliability prior to deployment.
4. **Compatibility with Continuous Variables**: Although the core derivations are presented in the discrete setting, the results hold when $X$ and $Y$ are continuous; continuous $W$ can be handled via appropriate discretization.

## Limitations & Future Work

1. $U$ is required to be discrete with a pre-specified or estimated support size $k_U$, which is often difficult to determine in practice.
2. Assumption 1 requires sufficiently many "diverse" source domains ($k_E \geq k_U$), which may not hold when the number of source domains is limited.
3. The Causal Estimator requires solving a non-convex MLE, subject to local optima; the Reduced Estimator may produce estimates outside $[0,1]$ in small-sample regimes.
4. The framework is currently restricted to discrete $W$ (continuous $W$ is only discussed via discretization in the appendix), and systematic treatment of high-dimensional proxy variables is lacking.
5. Real-data experiments are validated only in the hotel ranking scenario, without broader evaluation across application domains.

## Related Work & Insights

- **Proximal Causal Inference [Miao et al., 2018; Tchetgen et al., 2024]**: Requires two proxy variables and assumes causal effects are invariant across domains. This paper requires only one proxy combined with multi-domain data.
- **Domain Adaptation with Proxies [Tsai et al., 2024]**: Requires observing both $W$ and $X$ in the target domain and targets predictive performance. This paper requires only $W$ and targets the interventional distribution.
- **Latent Variable Models [Louizos et al., 2017; Wang & Blei, 2019]**: Explicitly estimate $U$ before adjustment. This paper bypasses estimation of $U$.
- **Transportability [Bareinboim & Pearl, 2013]**: Identifies causal effects based purely on graphical criteria. This paper additionally leverages informativeness assumptions on proxies.

## Rating
- Novelty: ⭐⭐⭐⭐ — Introduces a novel setting in multi-domain proximal causal inference where only $W$ is observed in the target domain; theoretical contributions are outstanding.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Simulations cover consistency, coverage, and condition number sensitivity; real-data experiments include Oracle comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ — Notation is clear, theorem-proof structure is rigorous, and figures and tables are informative.
- Value: ⭐⭐⭐⭐ — Establishes new identifiability results and a practical estimation framework with significant implications for causal inference and domain adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](../../AAAI2026/causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)

</div>

<!-- RELATED:END -->
