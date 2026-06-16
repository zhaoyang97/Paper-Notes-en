---
title: >-
  [Paper Note] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification
description: >-
  [ICML 2026][Causal Inference][Paper Note] This paper provides a unified "distribution-class" framework characterizing the necessary and sufficient condition (Condition 1) for population-wide Average Treatment Effect (ATE) identifiability under selection bias. It proves that this condition is satisfied under c-overlap propensity scores for common distributions
tags:
  - ICML 2026
  - Causal Inference
date: 2026-05-08
content_hash: 3eb7a98351dd67fa
---
# Towards a Holistic Understanding of Selection Bias for Causal Effect Identification

**Conference**: ICML 2026  
**arXiv**: [2605.13430](https://arxiv.org/abs/2605.13430)  
**Code**: https://github.com/EvieQ01/causal_effect_id_selection_bias  
**Area**: Causal Inference / Selection Bias / ATE Identifiability  
**Keywords**: Selection Bias, ATE Identifiability, Truncated Statistics, Propensity Score, Distribution Classes

## TL;DR
This paper provides a unified "distribution-class" framework characterizing the necessary and sufficient condition (Condition 1) for population-wide Average Treatment Effect (ATE) identifiability under selection bias. It proves that this condition is satisfied under c-overlap propensity scores for common distributions such as polynomial exponential families, Gaussian, Laplace, Pareto, and Log-normal. Two estimators, MLE and Score Matching, are proposed with selection function $\beta(x,y,t)$ correction, significantly outperforming IPW and polynomial regression in synthetic and "All of Us" semi-synthetic experiments.

## Background & Motivation

**Background**: Selection bias is ubiquitous in observational studies—large-scale biobanks suffer from "healthy volunteer bias," and clinical trial volunteers differ systematically from decliners. Existing work characterizing ATE identifiability under selection bias generally follows two paths: (i) **Graphical Model Path** (Bareinboim & Pearl; Correa et al. 2019) provides graphical criteria like selection-backdoor, but **must know exactly which node in the DAG selection occurs at**; (ii) **SEM Path** (Zhang et al. 2016) assumes Additive Noise Models (ANM) with non-Gaussian noise, which **can only handle outcome-dependent selection**.

**Limitations of Prior Work**: (a) In practice, it is difficult to precisely locate the selection node; (b) parametric assumptions of non-Gaussian noise + ANM are strong and violate practical scenarios involving Gaussian or common heavy-tailed distributions; (c) the two paths are incompatible, and no unified language exists to compare them.

**Key Challenge**: Characterizing ATE identifiability requires joint constraints on the triplet $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$. However, existing results impose constraints either on the graph structure or on functional forms, lacking a unified language characterized solely by **probability classes**.

**Goal**: To provide a necessary and sufficient condition that **does not depend on graph locations or require ANM**, covering all existing identifiable cases while extending to new identifiable regions, and to design practical ATE estimation algorithms.

**Key Insight**: The authors observe that by imposing a "separability" condition on the triplet $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$—where two different distributions with unequal ATEs must differ in their observed parts—identifiability is guaranteed. Furthermore, **truncated statistics** (Daskalakis et al. 2021) provide tools to extrapolate from truncated samples back to the original distribution, perfectly addressing "total masking" regions under deterministic selection.

**Core Idea**: Replace graphical criteria and ANM assumptions with a **separability condition on distribution-class triplets (Condition 1)**, and operationalize identifiability through the **joint estimation of truncated statistics and the selection function $\beta$**.

## Method

### Overall Architecture
The paper addresses the problem of when and how the population ATE can be recovered from the distorted observed distribution $P(V\mid S=1)$. The approach is split into theoretical and algorithmic layers. The theoretical layer redefines "identifiability" from DAG-dependent structures to a distribution-class separability condition (Condition 1) based on the triplet $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$, instantiated for common distribution families under both deterministic and nondeterministic selection mechanisms. The algorithmic layer implements this theory as a runnable estimator: it uses propensity scores to identify an overlap sub-region $\mathcal{B}$, jointly estimates the conditional outcome density and an unknown selection function $\beta$ within it, and finally extrapolates back to the population ATE using $1/\hat{\beta}$ reweighting.

### Key Designs

**1. Distribution-Class Separability Condition: Moving Identifiability from Graphs to Probability Classes**

The limitation of existing results is that "identifiability" is tied to DAG topology—either requiring knowledge of where selection node $S$ sits or assuming non-Gaussian additive noise. This paper directly imposes Condition 1 on the triplet $(\mathbb{P}_{t\mid x}, \mathbb{P}_{xy(t)}, \mathbb{S})$. It requires that for any two compatible distributions $(P, Q)$, if their ATEs are unequal ($\tau_{P_{xy(t)}} \neq \tau_{Q_{xy(t)}}$), there must exist a point $(x,y,t)$ where their observed densities differ: $\alpha_P(x,y,t)\, P_{t\mid x}\, P_{xy(t)} \neq \alpha_Q(\cdot)\, Q_{t\mid x}\, Q_{xy(t)}$, where $\alpha_P = P_{s\mid xyt} / P(s)$. Intuitively, "different ATE $\Rightarrow$ different observations," which excludes the confounding case where "two worlds with different ATEs produce the same observed distribution." Theorem 3.1 proves this separability is the necessary and sufficient condition for ATE identifiability from $P(V\mid S=1)$. Corollaries 3.9-3.14 translate graphical criteria like selection-backdoor, selection-backdoor-ext, outcome-dependent selection, and S-id into special cases of this condition, while also incorporating atypical cases that graphical criteria cannot handle.

**2. Instantiations in Common Distribution Families: Determining Identifiability in Practice**

As Condition 1 is abstract, the authors instantiate it for specific distribution families. For **deterministic selection** (where $P_{s\mid xyt}=0$ at some $(x,y,t)$ and observations are totally masked), Proposition 3.3 shows that if the propensity score satisfies c-overlap ($c < p(t,x) < 1-c$), the conditional outcome density follows $P_{y(t)\mid x} \propto e^{f(x,y)}$, the covariate marginal follows $P_x \propto e^{g(x)}$, and $f,g$ are polynomials (denoted $\mathbb{P}_{xy(t)}^{C^\infty}$), then Condition 1 holds. This relies on truncated statistics (Daskalakis et al. 2021) to extrapolate from truncated samples. For **nondeterministic selection** ($P_{s\mid xyt} > d > 0$ everywhere), Proposition 3.5 provides a lighter condition: $\mathbb{P}_{xy(t)}$ only needs to belong to one of four families: Gaussian, Laplace, Pareto, or Log-normal. This step refutes Zhang et al. (2016)'s restriction that "noise must be non-Gaussian"—within this framework, Gaussian is identifiable, while the polynomial exponential family covers most practical distributions with both light and heavy tails.

**3. MLE / Score Matching Estimators with Selection Function $\beta$: Learning Unknown Selection Mechanisms**

While theory guarantees identifiability, the algorithm must handle the unknown selection mechanism. This is explicitly encoded as a learned function $\beta_\phi(x,y,t)$, jointly learned with the outcome density $P_\theta(y\mid x,t)$. The observed conditional density follows $\tilde{p}(y\mid x) \propto p(y\mid x)\cdot \beta(x,y,t)$; by estimating $\beta$ and dividing it out, the population density can be restored. The MLE approach minimizes the negative log-likelihood within $\mathcal{B}$ with a regularizer $\lambda \sum_i \log\|\hat{\beta}\|_2^2$ to pull $\log\hat{\beta}$ toward 0. The Score Matching approach avoids the normalization constant of high-dimensional densities: taking the gradient w.r.t $y$ yields $\psi(x,y,t) = s_\theta(x,y) + \nabla_y \log\beta_\phi(x,y,t)$. Since the partition function depends only on $(x,t)$, it vanishes after differentiation. This is optimized using the Hyvärinen objective combined with $-\lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2$. Both routes use $1/\hat{\beta}$ reweighting to obtain $\hat{P}_{pop}(x)$ and output $\hat{\tau}_P = \mathbb{E}_{x \sim \hat{P}_{pop}}[\mathbb{E}[Y\mid x, 1] - \mathbb{E}[Y\mid x, 0]]$. Unlike IPW which ignores selection, the $+\beta$ corrected version handles both selection types, and Score Matching makes high-dimensional conditional density estimation tractable.

### Loss & Training
The final MLE objective is $L(\theta) + \lambda \sum_i \log\|\hat{\beta}(x_i, y_i, t_i)\|_2^2$. The final Score Matching objective is $\mathcal{L}(\theta, \phi) = \mathbb{E}_{\mathcal{D}_{obs}} \big[\tfrac{1}{2}\|\psi\|^2 + \mathrm{tr}(\nabla_y \psi) - \lambda_1 \log\beta_\phi + \lambda_2 \|\beta_\phi\|^2\big]$. Usually $\beta \to 1$ (maximizing the likelihood that observed samples were indeed selected), with $\lambda_1, \lambda_2$ as regularization hyperparameters.

## Key Experimental Results

### Main Results
Synthetic Data: $N=5000$ generated, $\sim 3000$ post-selection, 5 repetitions shown via boxplots. Settings cover additive/multiplicative models with Gaussian/Laplace noise, applying both deterministic and nondeterministic selection.

| Dataset | Noise / Functional Form | IPW | Polynomial | MLE | MLE+$\beta$ | SM | SM+$\beta$ |
|--------|----------------|-----|------------|-----|-------------|----|----|
| Synthetic | Additive Gaussian | High Bias | Moderate Bias | Low Bias | **Low Bias** | Low Bias | **Low Bias + Low Var** |
| Synthetic | Additive Laplace | High Bias | Moderate Bias | Low Bias | **Low Bias** | Low Bias | **Low Bias + Low Var** |
| Synthetic | Multiplicative Gaussian | High Bias | Significant Bias | Low Bias | **Low Bias** | Low Bias | **Low Bias** |
| Synthetic | Multiplicative Laplace | High Bias | Significant Bias | Low Bias | **Low Bias** | Low Bias | **Low Bias** |

Semi-synthetic All of Us: T2D as exposure, BMI as covariate, outcome $Y = f(T, X) + E$ synthesized. The proposed method significantly reduces bias but does not eliminate it (explained by weak overlap with propensity scores $\sim 0.05$).

### Ablation Study

| Configuration | Handles Deterministic | Handles Nondeterministic | Main Error |
|------|---------------------|------------------------|----------|
| IPW | ❌ | ❌ | High |
| Polynomial | ✅ (Extrapolation) | ❌ | Medium |
| MLE | ✅ | Partial ✅ | Small |
| MLE+$\beta$ | ✅ | ✅ | Smallest (higher variance) |
| SM | ✅ | Partial ✅ | Small |
| SM+$\beta$ | ✅ | ✅ | Smallest (lowest variance) |

### Key Findings
- **+$\beta$ correction is always superior in bias reduction**, but results in higher variance—estimating the selection function introduces uncertainty, a classic bias-variance trade-off.
- **SM+$\beta$ has lower variance than MLE+$\beta$**: Score matching does not require the partition function and is less sensitive to density scale factors, making it numerically more stable.
- **Polynomial regression fails under multiplicative noise**: It can only fit the mean regression of additive structures; multiplicative noise violates this assumption, whereas density-based MLE/SM is robust.
- **Residual bias in All of Us** stems from weak overlap (propensity $\sim 0.05$), suggesting that real-world data requires stronger regularization or priors for extrapolation.

## Highlights & Insights
- **Unifying Graphical Criteria and ANM in a Distribution-Class Language**: Corollaries 3.9-3.14 translate various graphical criteria into Condition 1. This "unify-then-extend" presentation is a major highlight.
- **Introduction of Truncated Statistics to Causal Identification**: Previously used for learning high-dimensional distributions with censoring, deterministic selection is treated here as "truncation on regions where $\beta=0$."
- **Score Matching incorporates the $\beta$ function into the gradient**: Canceling the partition function via $\nabla_y$ makes high-dimensional conditional outcome density estimation tractable.
- **Graph-Agnostic Identifiability**: The framework is practical because researchers often know the set of variables affecting $S$ but not the exact DAG structure.

## Limitations & Future Work
- **Theoretical Necessity vs. Practical Verification**: Users may find it difficult to verify if their data strictly follows the $\mathbb{P}_{xy(t)}^{C^\infty}$ polynomial exponential family. Proposition 3.5's families are more practical but still parametric.
- **Residual Bias in All of Us**: In cases of extreme selection and poor coverage, theoretical identifiability may be overshadowed by finite-sample variance and extrapolation errors.
- **Unobserved Confounding**: The assumption $Y(t) \perp T \mid X$ persists; future work could integrate IVs, proxy variables, or negative controls.
- **Real-world Validation**: Since $Y$ was synthetic in the All of Us experiments, fully real-world validation against RCT gold standards is needed.
- **Future Directions**: (a) Incorporating doubly robust/TMLE ideas to reduce variance; (b) extending to continuous treatments; (c) combining with graph discovery to constrain search spaces based on identifiability.

## Related Work & Insights
- **vs. Correa et al. 2019 (selection-backdoor)**: Their method requires knowing $S$'s position in a DAG; Ours is graph-agnostic and covers selection-backdoor as a special case.
- **vs. Zhang et al. 2016 (ANM + Non-Gaussian)**: They require additive noise and non-Gaussianity; Ours allows Gaussian/Laplace and flexible selection dependencies.
- **vs. Cai et al. 2025 (Overlap Violation)**: They address covariate support non-overlap; this paper addresses distribution distortion by selection mechanisms.
- **Insight**: The "distribution-class" strategy can be generalized to missing-at-random and measurement error scenarios. The score matching + $\beta$ pipeline is also portable to RLHF for handling selective feedback bias.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifies graphical/SEM paths into a distribution-class framework and introduces truncated statistics to causal ID.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers various noise and selection types, though more large-scale real-world baseline comparisons could be added.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mapping between Condition 1 and corollaries, though notation is dense in Section 3.
- **Value**: ⭐⭐⭐⭐⭐ High practical value for biobank/clinical data by bypassing precise DAG requirements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Latent Variable Causal Discovery under Selection Bias](../../ICML2025/causal_inference/latent_variable_causal_discovery_under_selection_bias.md)
- [\[ICML 2025\] Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants](../../ICML2025/causal_inference/causal_effect_identification_in_lvlingam_from_higher-order_cumulants.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)

</div>

<!-- RELATED:END -->
