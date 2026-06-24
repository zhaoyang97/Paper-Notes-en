---
title: >-
  [Paper Note] Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants
description: >-
  [ICML 2025][Causal Inference][Causal Effect Identification] In the Linear Non-Gaussian Acyclic Model with latent confounding (lvLiNGAM), this paper identifies causal effects using higher-order cumulants (instead of only the covariance matrix). It addresses two challenging settings: (1) a single proxy variable that may affect the treatment; and (2) the underdetermined instrumental variable (IV) problem where the number of IVs is less than the number of treatments. Identifiabil…
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "Causal Effect Identification"
  - "lvLiNGAM"
  - "Higher-Order Cumulants"
  - "Proxy Variables"
  - "Instrumental Variables"
date: 2026-05-08
content_hash: e8abe8a06233c214
---

# Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants

**Conference**: ICML 2025  
**arXiv**: [2506.05202](https://arxiv.org/abs/2506.05202)  
**Code**: None  
**Area**: Causal Inference  
**Keywords**: Causal Effect Identification, lvLiNGAM, Higher-Order Cumulants, Proxy Variables, Instrumental Variables

## TL;DR
In the Linear Non-Gaussian Acyclic Model with latent confounding (lvLiNGAM), this paper identifies causal effects using higher-order cumulants (instead of only the covariance matrix). It addresses two challenging settings: (1) a single proxy variable that may affect the treatment; and (2) the underdetermined instrumental variable (IV) problem where the number of IVs is less than the number of treatments. Identifiability is proved and consistent estimators are provided for both cases.

## Background & Motivation

**Background**: Causal effect identification aims to predict the effects of unseen interventions from observational data, which is crucial in fields such as medicine, policy evaluation, and fair decision-making. Structural Causal Models (SCMs) serve as the dominant framework, but latent confounding often renders causal effects unidentifiable.

**Limitations of Prior Work**:
   - Under Gaussian linear models, all information lies in the covariance matrix, making causal effects generally unidentifiable.
   - LiNGAM leverages non-Gaussianity to identify causal structures, but causal effect identification becomes much more challenging in the presence of latent variables.
   - The proxy variable method of Kivva et al. (2023) requires one proxy for each latent confounder, and the proxy cannot directly affect the treatment—a condition hard to satisfy in practice.
   - Two-Stage Least Squares (TSLS) is inapplicable when the number of instrumental variables is less than the number of treatment variables (underdetermined case).

**Key Challenge**: Conservative assumptions (multiple proxies / sufficient instrumental variables) are difficult to satisfy in practice, but does identifiability still hold when these assumptions are relaxed?

**Goal**: To identify causal effects under weaker assumptions using higher-order cumulants.

**Key Insight**: Higher-order cumulants (third-order and above) of non-Gaussian distributions contain extra information beyond covariance—this information can be used to identify causal effects under single-proxy or underdetermined IV settings.

**Core Idea**: Higher-order cumulants = extra signals from non-Gaussianity $\rightarrow$ compensating for missing equations when covariance matrix information is insufficient $\rightarrow$ enabling previously impossible identification.

## Method

### Overall Architecture
Two independent yet complementary contributions:
1. **Proxy Variable Setting**: A single proxy variable may affect the treatment variable $\rightarrow$ causal effects are still identifiable.
2. **Underdetermined Instrumental Variable Setting**: Single instrumental variable + multiple treatment variables $\rightarrow$ causal effects are still identifiable.
Both rely on analytical formulas of third- or fourth-order cumulants to construct estimators of causal effects.

### Key Designs

1. **Causal Effect Identification with a Single Proxy (Section 3.1)**:

    - **Function**: Identifying causal effects using only a single proxy variable, even when the proxy directly affects the treatment.
    - **Mechanism**: Let the causal effect of $X \to Y$ be $\beta$, with a latent confounder $L$ affecting both $X$ and $Y$, and a proxy $P$ associated with $L$.
    - Key formula (fourth-order cumulant):
    $$\beta = \frac{\kappa_{3}(Y, X, P) \cdot \kappa_{2}(X, P) - \kappa_{2}(Y, P) \cdot \kappa_{3}(X, X, P)}{\kappa_{3}(X, X, P) \cdot \kappa_{2}(X, P) - \kappa_{2}(X, P) \cdot \kappa_{3}(X, X, P)}$$
    - Comparison with prior work: Kivva et al. require that proxies do not affect the treatment and that each confounder has its own proxy; this work allows proxy $\to$ treatment causal edges and shares a single proxy for all confounders.
    - Impossibility results: It is proved that using only second- and third-order cumulants is insufficient—at least fourth-order is required.

2. **Causal Effect Identification with Underdetermined Instrumental Variables (Section 3.2)**:

    - **Function**: Identifying causal effects when the number of instrumental variables is less than the number of treatment variables.
    - **Mechanism**: Suppose $Z \to X_1, X_2 \to Y$ but there is only one instrumental variable $Z$, whereas traditional TSLS requires 2.
    - Higher-order cumulants provide extra equations: The relations of third-order cumulants such as $\kappa_3(Y, X_1, Z), \kappa_3(Y, X_2, Z)$ with causal effects form a sufficient system of equations.
    - **Design Motivation**: Instrument shortage is frequently encountered in biological applications (e.g., limited genetic instruments in Mendelian randomization).

3. **Consistent Estimation Method (Section 4)**:

    - **Function**: Estimating causal effects from finite-sample data.
    - **Mechanism**: Substituting population cumulants with sample cumulants and plugging them into the theoretical formulas to obtain estimators.
    - Consistency Proof: The estimators converge to the true causal effects as the sample size increases.
    - **Design Motivation**: Previous Overcomplete Independent Component Analysis (OICA) methods do not yield consistent estimates.

### Loss & Training
- Training-free method—estimators based on closed-form analytical formulas.
- Only requires calculating high-order sample cumulants among observed variables.
- Computational complexity is polynomial in the number of variables.

## Key Experimental Results

### Proxy Variable Experiments
Synthetic linear non-Gaussian data (various numbers of confounders $l$):

| Method | Sample Size N=1K | N=5K | N=10K | Proxy Requirement |
|------|-----------|------|-------|----------|
| Kivva et al. | Not applicable (requires one proxy per confounder) | - | - | $l$ proxies |
| TSLS | Unidentifiable | - | - | $\ge$ treatments |
| **Ours (Fourth-Order Cumulants)** | **MSE=0.15** | **0.04** | **0.01** | **Only 1** |

### Underdetermined Instrumental Variable Experiments
2 treatment variables, 1 instrumental variable:

| Method | MSE (X1→Y) | MSE (X2→Y) | Notes |
|------|-----------|-----------|------|
| TSLS | Infeasible | Infeasible | Requires 2 IVs |
| **Ours** | **0.03** | **0.05** | Requires only 1 IV |

### Ablation Study

| Configuration | MSE | Notes |
|------|-----|------|
| Second-order cumulants only | Unidentifiable | Insufficient equations |
| Second- and third-order only | Unidentifiable | Insufficient in proxy$\to$treatment setting |
| **Second + third + fourth order** | **0.01** | Minimum sufficient set of orders |
| Gaussian noise (non-Gaussianity assumption fails) | Fails | Higher-order cumulants vanish |
| Heavy-tailed distribution noise | 0.008 | Stronger non-Gaussianity yields more accurate estimates |

### Key Findings
- Fourth-order cumulant is the minimum necessary order in the single-proxy setting; second- and third-order cumulants are proved to be insufficient (an important negative result).
- Estimators converge consistently as the sample size $N \to \infty$—matching theoretical predictions.
- The stronger the non-Gaussianity (e.g., heavy-tailed distribution), the higher the signal-to-noise ratio of higher-order cumulants, leading to more accurate estimation.
- Preliminary validation on real gene regulatory network data demonstrates effectiveness (consistent with known causal relations).

## Highlights & Insights
- **"Higher-order cumulants = free extra signals"**—non-Gaussianity is an asset rather than a defect in causal inference.
- Relaxing assumptions to single-proxy where the proxy can affect treatment is a major practical advancement—many proxy variables in reality do affect the treatment (e.g., socioeconomic indicators acting as proxies for poverty while simultaneously affecting educational investment).
- The characterization of the minimum necessary order (fourth-order required) holds independent theoretical value—knowing "second- and third-order are insufficient" prevents futile attempts.
- Underdetermined IVs are highly common in biology (e.g., Mendelian randomization)—this method holds direct application value.
- The method combines theoretical depth (identifiability proofs) with practical simplicity (closed-form estimators).

## Limitations & Future Work
- Strictly requires linear models and non-Gaussian noise—fails under non-linear causal mechanisms or Gaussian noise.
- Finite-sample estimation variance of higher-order cumulants is high—especially fourth-order cumulants, which require larger sample sizes.
- Considers only deterministic causal effects (constant coefficients) and does not handle heterogeneous causal effects.
- The acyclic assumption excludes feedback loops (which exist in certain biological systems).
- General combinations of multiple treatments and multiple confounders are not fully covered.

## Related Work & Insights
- **vs TSLS**: TSLS utilizes only the covariance matrix and is infeasible under underdetermined IV settings; this work unlocks new capabilities using higher-order cumulants.
- **vs Kivva et al. (2023)**: Kivva et al. require an independent proxy for each confounder, and the proxy cannot affect the treatment; this work significantly relaxes these requirements.
- **vs Tramontano et al. (2024b)**: Based on OICA (an overcomplete problem), which does not produce consistent estimators; this work directly estimates via cumulant formulas.
- **Insights**: The potential of higher-order statistics in causal inference is far from exhausted—fourth-order cumulants play a key role here, and fifth-order and above might be useful in more complex graphs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to achieve causal effect identification under single-proxy / underdetermined IV settings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully validated on synthetic data, with preliminary real data.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous and clearly structured.
- Value: ⭐⭐⭐⭐⭐ Advances the theoretical frontier of causal inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](../../ICML2026/causal_inference/towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](../../NeurIPS2025/causal_inference/an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)
- [\[ICLR 2026\] Score-based Greedy Search for Structure Identification of Partially Observed Causal Models](../../ICLR2026/causal_inference/score-based_greedy_search_for_structure_identification_of_partially_observed_cau.md)

</div>

<!-- RELATED:END -->
