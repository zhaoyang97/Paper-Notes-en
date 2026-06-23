---
title: >-
  [Paper Note] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification
description: >-
  [ICML 2026][Causal Inference][Paper Note] This paper provides a unified "distribution-class" framework, characterizing the necessary and sufficient conditions (Condition 1) for the identifiability of the Average Treatment Effect (ATE) across the entire population under selection bias. It proves that this condition is satisfied under c-overlap propensity scores
tags:
  - ICML 2026
  - Causal Inference
date: 2026-05-08
content_hash: 835ead5bb8d6aba5
---
# Towards a Holistic Understanding of Selection Bias for Causal Effect Identification

**Conference**: ICML 2026  
**arXiv**: [2605.13430](https://arxiv.org/abs/2605.13430)  
**Code**: https://github.com/EvieQ01/causal_effect_id_selection_bias  
**Area**: Causal Inference / Selection Bias / ATE Identifiability  
**Keywords**: Selection Bias, ATE Identifiability, Truncated Statistics, Propensity Score, Distribution Class

## TL;DR
This paper provides a unified "distribution-class" framework, characterizing the necessary and sufficient conditions (Condition 1) for the identifiability of the Average Treatment Effect (ATE) across the entire population under selection bias. It proves that this condition is satisfied under c-overlap propensity scores and common distributions such as polynomial exponential families, Gaussian, Laplace, Pareto, and Log-normal. Two estimators, MLE and Score Matching, are proposed with correction using a selection function $\beta(x,y,t)$, which significantly outperform IPW and polynomial regression in synthetic and All of Us semi-synthetic experiments.

## Background & Motivation

**Background**: Selection bias is ubiquitous in observational studies—large-scale biobanks exhibit "healthy volunteer bias," and clinical trial volunteers systematically differ from those who decline. Existing work characterizing ATE identifiability under selection bias generally follows two routes: (i) **Graphical model route** (Bareinboim & Pearl, Correa et al. 2019) provides graphical criteria like selection-backdoor, but **must know where the selection occurs in the DAG**; (ii) **SEM route** (Zhang et al. 2016) assumes Additive Noise Models (ANM) with non-Gaussian noise and **can only handle outcome-dependent selection**.

**Limitations of Prior Work**: (a) In practice, it is difficult to accurately know the selection node's position; (b) parametric assumptions of non-Gaussianity and ANM are strong and violate practical scenarios involving Gaussian or common heavy-tailed distributions; (c) the two routes are incompatible, with no unified language for comparison.

**Key Challenge**: Characterizing ATE identifiability requires joint constraints on the "selection mechanism + propensity score + covariate-outcome distribution," but existing results place constraints either on the graph structure or function forms, lacking a unified language characterized solely by the **probability class**.

**Goal**: To provide a necessary and sufficient condition that **does not rely on graph positions or require ANM**, covering all existing identifiable scenarios while extending to new regions of identifiability, and to design practical ATE estimation algorithms.

**Key Insight**: The authors observe that identifiability can be achieved by imposing a "separability condition" on the triple $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$, such that if the ATEs of two different distributions are unequal, their observed components must differ. Furthermore, **truncated statistics** (Daskalakis et al. 2021) provides tools to extrapolate from truncated samples to the original distribution, perfectly addressing the "completely masked" regions under deterministic selection.

**Core Idea**: Replace graphical criteria and ANM assumptions with a **separability condition on distribution-class triples (Condition 1)**, and implement identifiability into practical algorithms using a **joint estimation of truncated statistics and the selection function $\beta$**.

## Method

### Overall Architecture
The paper addresses the question: when selection bias distorts the observed distribution $P(V\mid S=1)$, when can the population ATE still be recovered, and how can it be calculated? The authors decompose this into theoretical and algorithmic layers. The theoretical layer rewrites "identifiability" from DAG dependency to a distribution-class separability condition (Condition 1) depending only on the triple $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$, instantiated for common distribution families under both deterministic and nondeterministic selection. The algorithmic layer translates this theory into a runnable estimator: first, a sub-region $\mathcal{B}$ with overlap is identified via propensity scores, where the conditional outcome density and an unknown selection function $\beta$ are jointly estimated. Finally, $1/\hat{\beta}$ reweighting is used to extrapolate back to the population ATE.

### Key Designs

**1. Distribution-class Separability Condition: Moving Identifiability from Graphs to Distribution Families**

A major limitation of previous results is that identifiability is tied to the DAG topology—either requiring knowledge of the selection node $S$ (graphical route) or assuming additive non-Gaussian noise (SEM route). This paper uses a different language: proposing conditions directly on the triple $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$. Condition 1 requires that for any two compatible distributions $(P, Q)$, if their ATEs are unequal ($\tau_{P_{xy(t)}} \neq \tau_{Q_{xy(t)}}$), there must exist some point $(x,y,t)$ where the observed densities are unequal, i.e., $\alpha_P(x,y,t)\, P_{t\mid x}\, P_{xy(t)} \neq \alpha_Q(\cdot)\, Q_{t\mid x}\, Q_{xy(t)}$, where $\alpha_P = P_{s\mid xyt} / P(s)$. Intuitively, "different ATE $\Rightarrow$ different observations," which excludes confounding cases where different worlds produce the same observed distribution. Theorem 3.1 proves this is the necessary and sufficient condition for ATE identifiability from $P(V\mid S=1)$. Its advantage is covering both directions: Corollary 3.9-3.14 translate graphical criteria like selection-backdoor, selection-backdoor-ext, outcome-dependent selection, and S-id into special cases, while including atypical cases that graphical criteria cannot handle.

**2. Instantiation under Common Distribution Families: Telling Practitioners if Data is Identifiable**

Condition 1 is abstract, so the authors instantiate it into recognizable distribution families. For **deterministic selection** (where $P_{s\mid xyt}=0$ at some $(x,y,t)$, entirely masking observations), Proposition 3.3 shows that if the propensity score satisfies c-overlap ($c < p(t,x) < 1-c$), the conditional outcome density is $P_{y(t)\mid x} \propto e^{f(x,y)}$, covariate marginal is $P_x \propto e^{g(x)}$, and $f,g$ are polynomials (denoted $\mathbb{P}_{xy(t)}^{C^\infty}$), Condition 1 holds. This relies on truncated statistics (Daskalakis et al. 2021) to extrapolate from truncated samples. For **nondeterministic selection** ($P_{s\mid xyt} > d > 0$ everywhere), Proposition 3.5 provides a lighter condition: $\mathbb{P}_{xy(t)}$ only needs to belong to one of four families: Gaussian, Laplace, Pareto, or Log-normal. This step is significant as it overturns the "non-Gaussian noise" restriction of Zhang et al. 2016—Gaussian noise is identifiable here, and the polynomial exponential family covers most practical distributions across light/heavy tails.

**3. MLE / Score Matching Estimators with Selection Function $\beta$: Learning the Unknown Selection Mechanism**

While theory guarantees identifiability, the algorithm must handle the "unknown selection mechanism." This paper explicitly encodes it as a function $\beta_\phi(x,y,t)$ to be learned jointly with the outcome density $P_\theta(y\mid x,t)$. The observed conditional density satisfies $\tilde{p}(y\mid x) \propto p(y\mid x)\cdot \beta(x,y,t)$; thus, estimating and removing $\beta$ recovers the population density. Two routes are provided: The MLE route minimizes negative log-likelihood in $\mathcal{B}$: $L(\theta) = -\sum_i \big(\log \hat{P}(y_i\mid x_i, t_i) + \log \hat{\beta}(\cdot) + \log \hat{P}(x_i) + \log \hat{e}(x_i)\big)$, with a regularization $\lambda \sum_i \log\|\hat{\beta}\|_2^2$ pulling $\log\hat{\beta}$ toward 0 to reduce uncertainty. The Score Matching route avoids the normalization constant of high-dimensional densities: taking the gradient w.r.t $y$ gives $\psi(x,y,t) = s_\theta(x,y) + \nabla_y \log\beta_\phi(x,y,t)$. Since the partition function only depends on $x,t$, it disappears after derivation. The Hyvärinen objective $\tfrac{1}{2}\|\psi\|^2 + \mathrm{tr}(\nabla_y \psi)$ with $-\lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2$ is then optimized. Both routes use $1/\hat{\beta}$ reweighting to obtain $\hat{P}_{pop}(x)$ and output $\hat{\tau}_P = \mathbb{E}_{x \sim \hat{P}_{pop}}[\mathbb{E}[Y\mid x, 1] - \mathbb{E}[Y\mid x, 0]]$. Compared to IPW (ignores selection) or polynomial extrapolation (fails for nondeterministic bias), the $+\beta$ corrected version handles both, and Score Matching makes high-dimensional density estimation tractable by removing the partition function.

### Loss & Training
The final MLE objective is $L(\theta) + \lambda \sum_i \log\|\hat{\beta}(x_i, y_i, t_i)\|_2^2$; the Score Matching objective is $\mathcal{L}(\theta, \phi) = \mathbb{E}_{\mathcal{D}_{obs}} \big[\tfrac{1}{2}\|\psi\|^2 + \mathrm{tr}(\nabla_y \psi) - \lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2\big]$. Usually $\beta \to 1$ (maximizing the likelihood that observed samples were selected), with $\lambda_1, \lambda_2$ as regularization hyperparameters.

## Key Experimental Results

### Main Results
Synthetic Data: $N=5000$ generated, approx. $3000$ after selection, box plots over 5 repetitions. Settings cover additive / multiplicative $\times$ Gaussian / Laplace noise (4 groups) with combined deterministic + nondeterministic selection.

| Dataset | Noise / Functional Form | IPW | Polynomial | MLE | MLE+$\beta$ | SM | SM+$\beta$ |
|--------|----------------|-----|------------|-----|-------------|----|----|
| Synthetic | Additive Gaussian | High Bias | Med Bias | Low Bias | **Min Bias** | Low Bias | **Min Bias + Low Var** |
| Synthetic | Additive Laplace | High Bias | Med Bias | Low Bias | **Min Bias** | Low Bias | **Min Bias + Low Var** |
| Synthetic | Multiplicative Gaussian | High Bias | Significant Bias | Low Bias | **Min Bias** | Low Bias | **Min Bias** |
| Synthetic | Multiplicative Laplace | High Bias | Significant Bias | Low Bias | **Min Bias** | Low Bias | **Min Bias** |

Semi-synthetic All of Us: T2D as exposure, BMI as covariate, outcome $Y = f(T, X) + E$. Ours significantly reduces bias but does not eliminate it (explained by weak overlap due to low propensity scores $\sim 0.05$).

### Ablation Study

| Configuration | Handles Deterministic | Handles Nondeterministic | Primary Error |
|------|---------------------|------------------------|----------|
| IPW | ❌ | ❌ | High |
| Polynomial | ✅ (Extrapolation) | ❌ | Medium |
| MLE | ✅ | Partial ✅ | Low |
| MLE+$\beta$ | ✅ | ✅ | Minimal (Higher Var)|
| SM | ✅ | Partial ✅ | Low |
| SM+$\beta$ | ✅ | ✅ | Minimal (Lowest Var)|

### Key Findings
- **+$\beta$ correction is always optimal regarding bias**, but leads to higher variance—a typical bias-variance trade-off from estimating the selection function.
- **SM+$\beta$ has lower variance than MLE+$\beta$**: Score matching does not compute partition functions and is insensitive to density scaling, making it numerically more stable while maintaining similar bias.
- **Polynomial fails under multiplicative noise**: It can only fit mean regression of additive structures; multiplicative noise violates this, whereas MLE/SM are robust as they estimate the full density.
- **Residual bias in All of Us** stems from weak overlap (propensity $\sim 0.05$), suggesting real-world data requires regularization or priors to stabilize extrapolation.

## Highlights & Insights
- **Unifying graphical criteria + ANM into distribution-class language**: Corollary 3.9-3.14 translate selection-backdoor, selection-backdoor-ext, outcome-dependent, and S-id as special cases of Condition 1. This "unify all plus add more" approach is exemplary.
- **Introduction of truncated statistics to causal identification**: Truncated statistics, previously used for distribution learning under censoring, is bridged here by treating deterministic selection as "truncation on $\beta=0$ regions."
- **Score matching incorporates $\beta$ into gradients**: Differentiating w.r.t $y$ removes the partition function, making high-dimensional conditional density estimation tractable. This trick is reusable for counterfactual generation and latent variable models.
- **"No need to locate the selection node" is highly practical**: Researchers often know the set of variables affecting $S$ but not the exact DAG; this framework bypasses that pain point.

## Limitations & Future Work
- **Condition 1 is necessary/sufficient but relies on distribution-class characterization**: It is hard for practitioners to verify if data strictly belongs to a polynomial exponential family $\mathbb{P}_{xy(t)}^{C^\infty}$.
- **Residual bias in All of Us**: Theoretical identifiability holds, but in finite samples with poor coverage, variance and extrapolation error dominate.
- **Unobserved confounding**: The paper assumes $Y(t) \perp T \mid X$, which is far from reality in EHR/biobank data with hidden confounders; integration with IV or proxy variables is needed.
- **Semi-synthetic validation**: Outcomes are self-generated; fully real-world validation against RCT gold standards is required.
- **Future directions**: (a) Doubly robust / TMLE approaches to reduce variance; (b) generalising to multi-treatment/continuous settings; (c) combining with graph-discovery to detect unidentifiable selection patterns.

## Related Work & Insights
- **vs Correa et al. 2019**: They require knowing the $S$ node position in the DAG and meeting backdoor criteria; Ours needs no graph structure, only distribution classes.
- **vs Zhang et al. 2016**: They require ANM + non-Gaussian noise + outcome-dependence; Ours allows Gaussian/Laplace/Pareto/Log-normal and arbitrary selection dependence.
- **vs Bareinboim & Pearl 2012**: Their transportability/recoverability relies on identifying the incoming edge structure of $S$; ours is "graph-agnostic."
- **vs Cai et al. 2025**: They handle covariate overlap violation; this paper handles selection-distorted distributions. Cai et al. identify sub-population ATE, but not full population ATE.
- **Insight**: The strategy of "abstracting one layer up to the distribution family" can generalize to missing-at-random or measurement error scenarios. The pipeline of truncated statistics + score matching + $\beta$ estimation is also portable to RLHF or preference learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unified framework for graphical/SEM routes using distribution classes; innovative use of truncated statistics.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage with synthetic/semi-synthetic, but space for fully real-world benchmarks remains.
- Writing Quality: ⭐⭐⭐⭐ Clear mapping of corollaries, but notation is dense; Section 3 has a steep curve for non-causal experts.
- Value: ⭐⭐⭐⭐⭐ A unified "graph-independent" framework with valid algorithms for biobank and clinical data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Latent Variable Causal Discovery under Selection Bias](../../ICML2025/causal_inference/latent_variable_causal_discovery_under_selection_bias.md)
- [\[ICML 2025\] Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants](../../ICML2025/causal_inference/causal_effect_identification_in_lvlingam_from_higher-order_cumulants.md)
- [\[ICML 2026\] Causal Modeling of Selection in Evolution](causal_modeling_of_selection_in_evolution.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)

</div>

<!-- RELATED:END -->
