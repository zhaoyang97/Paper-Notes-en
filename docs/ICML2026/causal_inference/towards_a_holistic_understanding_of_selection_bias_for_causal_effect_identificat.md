---
title: >-
  [Paper Note] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification
description: >-
  [ICML 2026][Causal Inference][Selection Bias] This paper proposes a unified "distribution-class" framework characterizing the necessary and sufficient condition (Condition 1) for population-level Average Treatment Effect…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Selection Bias"
  - "ATE Identifiability"
  - "Truncated Statistics"
  - "Propensity Score"
  - "Distribution Class"
date: 2026-05-08
content_hash: 86682c5b373fe9fd
---

# Towards a Holistic Understanding of Selection Bias for Causal Effect Identification

**Conference**: ICML 2026  
**arXiv**: [2605.13430](https://arxiv.org/abs/2605.13430)  
**Code**: https://github.com/EvieQ01/causal_effect_id_selection_bias  
**Area**: Causal Inference / Selection Bias / ATE Identifiability  
**Keywords**: Selection Bias, ATE Identifiability, Truncated Statistics, Propensity Score, Distribution Class

## TL;DR
This paper proposes a unified "distribution-class" framework characterizing the necessary and sufficient condition (Condition 1) for population-level Average Treatment Effect (ATE) identifiability under selection bias. It demonstrates that common distributions—including polynomial exponential families, Gaussian, Laplace, Pareto, and Log-normal—satisfy this condition under c-overlap propensity scores. The authors introduce MLE and Score Matching estimators corrected by a selection function $\beta(x,y,t)$, which significantly outperform IPW and polynomial regression in synthetic and "All of Us" semi-synthetic experiments.

## Background & Motivation

**Background**: Selection bias is ubiquitous in observational studies; large-scale biobanks exhibit "healthy volunteer bias," and clinical trial participants differ systematically from decliners. Existing works characterizing ATE identifiability under selection bias generally follow two paths: (i) **Graphical Model Route** (Bareinboim & Pearl, Correa et al. 2019), which provides graphical criteria like selection-backdoor but **requires knowing exactly where selection occurs in the DAG**; (ii) **SEM Route** (Zhang et al. 2016), which assumes Additive Noise Models (ANM) and non-Gaussian noise, **handling only outcome-dependent selection**.

**Limitations of Prior Work**: (a) It is practically difficult to pinpoint the selection node location; (b) parametric assumptions of non-Gaussianity and ANM are strong and violate practical scenarios involving Gaussian or common heavy-tailed distributions; (c) the two routes are incompatible, with no framework capable of comparing them in a unified language.

**Key Challenge**: Characterizing ATE identifiability requires joint constraints on the "selection mechanism + propensity score + covariate-outcome distribution." However, existing results impose constraints either on graph topology or functional forms, lacking a unified language characterized solely through **probability classes**.

**Goal**: To provide a necessary and sufficient condition that **does not depend on graphical positions or require ANMs**, while covering all existing identifiable scenarios and extending to new identifiable regions, accompanied by practical ATE estimation algorithms.

**Key Insight**: The authors observe that identifiability can be achieved by imposing an appropriate "separability condition" on the triplet $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$, such that if ATEs differ between two distributions, their observational parts must also differ. Furthermore, **truncated statistics** (Daskalakis et al. 2021) provide tools to extrapolate from truncated samples to the original distribution, perfectly addressing "completely masked" regions under deterministic selection.

**Core Idea**: Replace graphical criteria and ANM assumptions with a **separability condition on the distribution-class triplet (Condition 1)**, and implement identifiability via **truncated statistics combined with joint estimation of the selection function $\beta$**.

## Method

### Overall Architecture
The paper consists of two layers: The **Theoretical Layer** provides the necessary and sufficient condition (Theorem 3.1, centered on Condition 1) for population ATE identifiability and instantiates it for common distribution families under both deterministic and nondeterministic selection (Propositions 3.3 / 3.5). The **Algorithmic Layer** utilizes a three-stage pipeline for ATE estimation: first estimating propensity scores to locate the "overlap" sub-region $\mathcal{B}$, then using MLE or Score Matching on $\mathcal{B}$ to jointly estimate the conditional outcome density $\hat{P}(y\mid x,t)$ and the selection function $\hat{\beta}(x,y,t)$, and finally performing $1/\hat{\beta}$ reweighting to extrapolate to the full population and calculate $\hat{\tau}_P$.

### Key Designs

1.  **Distribution-class Based Identifiability Condition (Condition 1 + Theorem 3.1)**:
    - **Function**: Characterizes the necessary and sufficient condition for population ATE identifiability using a "global separability" condition, **independent of the selection node's position in the DAG**.
    - **Mechanism**: For any pair of compatible distributions $(P, Q)$ in the triplet $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$, if their ATEs differ ($\tau_{P_{xy(t)}} \neq \tau_{Q_{xy(t)}}$), there must exist some point $(x,y,t)$ where the observed density $\alpha_P(x,y,t) P_{t\mid x} P_{xy(t)} \neq \alpha_Q(\cdot) Q_{t\mid x} Q_{xy(t)}$, where $\alpha_P = P_{s\mid xyt} / P(s)$. Theorem 3.1 proves this is the necessary and sufficient condition for "ATE identifiability from $P(V\mid S=1)$."
    - **Design Motivation**: Traditional graphical criteria tie "identifiability" strictly to DAG topology. This condition applies directly to distribution classes, **covering graphical criteria (Corollary 3.9-3.14 incorporate selection-backdoor, selection-backdoor-ext, outcome-dependent selection, and S-id) while breaking through atypical cases that graphical criteria cannot handle**.

2.  **Identifiability Instantiation under Common Distribution Families (Proposition 3.3 + 3.5)**:
    - **Function**: Translates abstract Condition 1 into directly usable distribution families, informing practitioners whether their data is identifiable.
    - **Mechanism**: (i) Under **Deterministic selection** (where $P_{s\mid xyt}=0$ at some points), Condition 1 is satisfied if $\mathbb{P}_{t\mid x}$ meets c-overlap ($c < p(t,x) < 1-c$) and the conditional outcome density follows $P_{y(t)\mid x} \propto e^{f(x,y)}$ with marginal $P_x \propto e^{g(x)}$ (where $f, g$ are polynomials, denoted $\mathbb{P}_{xy(t)}^{C^\infty}$)—utilizing truncated statistics (Daskalakis et al. 2021) for extrapolation. (ii) Under **Nondeterministic selection** ($P_{s\mid xyt} > d > 0$), it is sufficient if $\mathbb{P}_{xy(t)}$ belongs to one of four major distribution families: Gaussian, Laplace, Pareto, or Log-normal.
    - **Design Motivation**: (ii) Challenges the "must be non-Gaussian" assumption of Zhang et al. 2016—**Gaussian distributions are identifiable** in this framework. (i) Extends identifiability to the polynomial exponential family, covering most practical distributions and remaining compatible with both light-tailed and heavy-tailed distributions.

3.  **MLE / Score Matching Estimators with Selection Function $\beta$ (Algorithm 1)**:
    - **Function**: Encodes the "unknown selection mechanism" as a learnable function $\beta_\phi(x,y,t)$ during the estimation phase, learned jointly with the outcome density $P_\theta(y\mid x,t)$.
    - **Mechanism**: The observed conditional density satisfies $\tilde{p}(y\mid x) \propto p(y\mid x) \cdot \beta(x,y,t)$. **MLE route**: Minimizes negative log-likelihood $L(\theta) = -\sum_i (\log \hat{P}(y_i\mid x_i, t_i) + \log \hat{\beta}(\cdot) + \log \hat{P}(x_i) + \log \hat{e}(x_i))$ within $\mathcal{B}$, adding a regularizer $\lambda \sum_i \log\|\hat{\beta}\|_2^2$ such that $\log\hat{\beta} \to 0$ to resolve indeterminacy. **Score Matching route**: Taking the gradient with respect to $y$ yields $\psi(x,y,t) = s_\theta(x,y) + \nabla_y \log\beta_\phi(x,y,t)$ (the hard-to-calculate partition function depends only on $x,t$ and vanishes). Optimization uses the Hyvärinen objective $\frac{1}{2}\|\psi\|^2 + \mathrm{tr}(\nabla_y \psi)$ plus $-\lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2$. ATE $\hat{\tau}_P$ is output via $1/\hat{\beta}$ reweighting.
    - **Design Motivation**: Traditional IPW ignores selection, and "polynomial extrapolation" only works for deterministic bias. The $\beta$-corrected version **handles both types of selection simultaneously**, and the Score Matching route avoids high-dimensional normalization constants by using $\nabla_y$.

### Loss & Training
Final MLE objective: $L(\theta) + \lambda \sum_i \log\|\hat{\beta}(x_i, y_i, t_i)\|_2^2$;  
Final Score Matching objective: $\mathcal{L}(\theta, \phi) = \mathbb{E}_{\mathcal{D}_{obs}} [\frac{1}{2}\|\psi\|^2 + \mathrm{tr}(\nabla_y \psi) - \lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2]$.  
Typically $\beta \to 1$ (maximizing the likelihood of "observed samples being selected"), with $\lambda_1, \lambda_2$ acting as regularization hyperparameters.

## Key Experimental Results

### Main Results
Synthetic data: $N=5000$ generated, approx. $3000$ after selection, 5 repetitions shown in boxplots. Settings cover additive/multiplicative × Gaussian/Laplace noise groups, with simultaneous deterministic + nondeterministic selection.

| Dataset | Noise / Functional Form | IPW | Polynomial | MLE | MLE+$\beta$ | SM | SM+$\beta$ |
|--------|----------------|-----|------------|-----|-------------|----|----|
| Synthetic | Additive Gaussian | High Bias | Medium Bias | Low Bias | **Min. Bias** | Low Bias | **Min. Bias + Low Var** |
| Synthetic | Additive Laplace | High Bias | Medium Bias | Low Bias | **Min. Bias** | Low Bias | **Min. Bias + Low Var** |
| Synthetic | Multiplicative Gaussian | High Bias | Sig. Bias (Fail Extrap.) | Low Bias | **Min. Bias** | Low Bias | **Min. Bias** |
| Synthetic | Multiplicative Laplace | High Bias | Sig. Bias | Low Bias | **Min. Bias** | Low Bias | **Min. Bias** |

Semi-synthetic All of Us: T2D as exposure, BMI as covariate, outcome $Y = f(T, X) + E$; the proposed method significantly reduces bias but does not eliminate it (attributed to weak overlap from propensity scores near $0.05$).

### Ablation Study

| Configuration | Handles Deterministic | Handles Nondeterministic | Primary Error |
|------|---------------------|------------------------|----------|
| IPW | ❌ | ❌ | Large |
| Polynomial | ✅ (Extrap.) | ❌ | Medium |
| MLE | ✅ | Partial ✅ | Small |
| MLE+$\beta$ | ✅ | ✅ | Minimal (Higher Var) |
| SM | ✅ | Partial ✅ | Small |
| SM+$\beta$ | ✅ | ✅ | Minimal (Lowest Var) |

### Key Findings
- **$\beta$-correction is consistently optimal for bias reduction**, but incurs higher variance—a typical bias-variance trade-off cost for estimating the selection function.
- **SM+$\beta$ shows lower variance than MLE+$\beta$**: Score matching is insensitive to density scale factors (no partition function calculation), leading to higher numerical stability.
- **Polynomial regression fails under multiplicative noise**: It can only fit additive mean-regression structures; multiplicative noise violates these assumptions, whereas MLE/SM are robust as they estimate the full density.
- **Residual bias in All of Us** stems from weak overlap (propensity $\sim 0.05$), suggesting that real-world data requires regularization or additional priors to stabilize extrapolation.

## Highlights & Insights
- **Unification of graphical criteria and ANM assumptions**: Corollaries 3.9-3.14 translate selection-backdoor, selection-backdoor-ext, outcome-dependent, and S-id as special cases of Condition 1. This "unify-then-extend" presentation is exemplary.
- **Truncated statistics in causal identification**: Previously used for high-dimensional distribution learning with censoring, truncated statistics are used here to treat deterministic selection as "truncation on regions where $\beta=0$."
- **Score matching incorporates $\beta$ into the gradient**: Taking the gradient with respect to $y$ makes the partition function vanish, making the estimation of high-dimensional conditional densities tractable—a trick reusable in counterfactual generation.
- **"No need to locate selection nodes" is a significant practical advantage**: Researchers often know the set of variables affecting $S$ but not the exact DAG structure. This framework bypasses that pain point.

## Limitations & Future Work
- **Practical verification of distribution classes**: It is difficult for practitioners to confirm their data strictly belongs to a polynomial exponential family. Propositions for Gaussian/Laplace/Pareto/Log-normal are more practical but remain parametric.
- **Residual bias in All of Us**: When selection is strong and coverage is poor, the framework's theoretical identifiability may be overshadowed by sample variance and extrapolation error.
- **Unobserved confounding**: The paper assumes $Y(t) \perp T \mid X$. Extending this to combine with IVs, proxy variables, or negative controls is a future direction for EHR/biobank data.
- **Real-world validation**: In the semi-synthetic experiments, $Y$ is generated. Fully real-world validation would require comparison against RCT gold standards.
- **Refinement ideas**: (a) Incorporate doubly robust/TMLE to reduce variance; (b) Extend to multi-treatment and continuous settings; (c) Use the framework to constrain graph search spaces by detecting unidentifiable selection patterns.

## Related Work & Insights
- **vs. Correa et al. 2019 (selection-backdoor)**: They require the location of $S$ in the DAG and backdoor compliance; ours requires only distribution-class compliance with Condition 1.
- **vs. Zhang et al. 2016 (ANMs + Non-Gaussianity)**: They require ANMs and non-Gaussian noise for outcome-dependent selection; we allow Gaussian/Laplace etc., and selection can depend on any observed variables.
- **vs. Bareinboim & Pearl 2012 / Bareinboim et al. 2014**: Their transportability/recoverability frameworks depend on graphical criteria and in-edge structures of $S$; we provide a "graph-agnostic" equivalent.
- **vs. Cai et al. 2025 (ATE identification under overlap violation)**: They handle covariate support non-overlap; we handle selection-distorted observed distributions. Mechanism differs, but both use distribution-class conditions.
- **Insight**: The strategy of "abstracting up one level to distribution classes" can be extended to MAR or measurement error scenarios. The joint estimation pipeline is also portable to preference learning for handling selective feedback bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies graphical and SEM routes via distribution-class language and introduces truncated statistics to causal ID.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong coverage across synthetic noise groups, though room for more real-world baseline comparison.
- Writing Quality: ⭐⭐⭐⭐ Strong clarity in mapping Condition 1 to Corollaries, though notation density is high in Section 3.
- Value: ⭐⭐⭐⭐⭐ A unified framework independent of graph location that expands identifiable regions and provides practical algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](../../NeurIPS2025/causal_inference/an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)

</div>

<!-- RELATED:END -->
