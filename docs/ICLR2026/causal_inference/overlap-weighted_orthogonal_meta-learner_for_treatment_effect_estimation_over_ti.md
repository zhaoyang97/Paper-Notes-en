---
title: >-
  [Paper Note] Overlap-Weighted Orthogonal Meta-Learner for Treatment Effect Estimation over Time
description: >-
  [ICLR 2026][Causal Inference][Heterogeneous Treatment Effects] This paper proposes the WO-learner (overlap-weighted orthogonal meta-learner), which focuses estimation on samples that are truly likely to receive the target intervention sequences by applying an "overlap weight" to training samples. Combined with a Neyman-orthogonal weighted population risk function, it maintains stability in low-overlap scenarios where "overlap probability decays exponentially with the predicti…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "Heterogeneous Treatment Effects"
  - "Neyman Orthogonality"
  - "Overlap Weights"
  - "Meta-Learner"
  - "Time-Varying Confounding"
date: 2026-05-08
content_hash: ae198a2582243b68
---

# Overlap-Weighted Orthogonal Meta-Learner for Treatment Effect Estimation over Time

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0Xi3WDwd5w](https://openreview.net/forum?id=0Xi3WDwd5w)  
**Code**: https://github.com/konstantinhess/wo_learner_timeseries  
**Area**: Causal Inference / Time-series Treatment Effect Estimation  
**Keywords**: Heterogeneous Treatment Effects, Neyman Orthogonality, Overlap Weights, Meta-Learner, Time-Varying Confounding

## TL;DR
This paper proposes the WO-learner (overlap-weighted orthogonal meta-learner), which focuses estimation on samples that are truly likely to receive the target intervention sequences by applying an "overlap weight" to training samples. Combined with a Neyman-orthogonal weighted population risk function, it maintains stability in low-overlap scenarios where "overlap probability decays exponentially with the prediction horizon." It outperforms existing meta-learners across synthetic, semi-synthetic, and real-world datasets.

## Background & Motivation
**Background**: In scenarios such as personalized medicine, researchers aim to estimate **heterogeneous treatment effects (HTE)** from patient trajectories—specifically, the Conditional Average Treatment Effect (CATE) $\mu_t^{\bar a,\bar b}(\bar h_t)=\mathbb{E}[Y_{t+\tau}[a_{t:t+\tau}]-Y_{t+\tau}[b_{t:t+\tau}]\mid \bar H_t=\bar h_t]$, which represents the difference in outcomes if a future treatment sequence $\bar a$ is followed versus sequence $\bar b$. In time-series settings, **time-varying confounding** must be corrected (where future covariates are influenced by past treatments and also affect future treatment assignments); otherwise, bias will persist regardless of sample size. The most common tools are **model-agnostic meta-learners**, which decouple the estimation strategy for confounding correction from the neural network backbone (e.g., HA, RA, IPW, DR, IVW).

**Limitations of Prior Work**: Almost all these meta-learners assume **sufficient treatment overlap**, meaning every target treatment sequence has a non-zero and non-negligible probability of being observed. However, in time-series settings, the propensity score for a sequence of length $\tau+1$ is the **product** of step-wise propensity scores $\prod_j \pi_j$. Consequently, the probability of observing a specific sequence **decays exponentially** with the prediction horizon (Figure 1). In these low-overlap regions, methods relying on inverse propensity weighting (IPW, DR) suffer from extreme weights (dividing by numbers near zero), leading to variance explosion. Regression-based methods (RA) fail because the lack of samples in these regions leads to poorly learned response surfaces. IVW attempts to bypass this, but its weights are not orthogonal, allowing propensity score errors to propagate as first-order bias across all time steps.

**Key Challenge**: There is a fundamental tension between the "sparse sample support" in low-overlap regions and the requirement for "reliable estimation" in those same regions. Forcing unbiased estimation where data is scarce inevitably leads to variance explosion, while common heuristic propensity clipping introduces uncontrollable bias.

**Goal**: To design a meta-learner that is stable in low-overlap regions (resilient to extreme inverse propensity weights), achieves first-order robustness to nuisance function misspecification (Neyman orthogonality), and remains entirely model-agnostic.

**Key Insight**: Since low-overlap regions lack data support, **one should not force precise estimation there**. Instead, the authors use a data-driven weight to **proactively bias the estimation target toward samples with higher overlap (or propensity) and more reliable information**. This systematically suppresses the influence of regions where stable estimation is impossible, rather than relying on post-hoc clipping.

**Core Idea**: Replace the unweighted risk objective with an **overlap-weighted oracle risk**, then orthogonalize it into a **Neyman-orthogonal weighted population risk**. Weighting is used instead of clipping to combat low overlap, and orthogonalization is used instead of plug-in estimation to combat nuisance error.

## Method

### Overall Architecture
The WO-learner is a two-stage, cross-fitting meta-learner aiming to learn a second-stage function $\hat g_\theta(\bar H_t)$ that approximates CATE (or CAPO). The recipe is as follows: split the data into two halves; use one half to estimate a set of **nuisance functions** (response functions $\mu_j$, propensity scores $\pi_j$, and **weight functions** $\omega_j$ derived from the product of propensities). Use the other half with these estimates to construct **WO pseudo-outcomes $\xi_t$** and a corresponding random weight $\rho_t$, then minimize a **weighted empirical risk** $\hat L(\hat g_\theta;\eta)$ to obtain the final estimator.

The primary difference from existing meta-learners lies in the risk function used for the second-stage fitting: the WO-learner employs a **weighted, Neyman-orthogonal population risk**. Weighting ensures estimation focuses on high-overlap samples (combating low overlap), while orthogonality ensures nuisance errors enter the final estimation only as second-order terms (combating misspecification). The design is supported by three theoretical guarantees: the weighted risk indeed minimizes the weighted oracle risk (Theorem 4.3), its minimizer correctly adjusts for time-varying confounding (Corollary 4.4), and the risk is Neyman-orthogonal to all nuisance functions (Theorem 4.5).

### Key Designs

**1. Overlap/Propensity Weight Function: Biasing estimation toward samples "likely to receive target treatment"**

Low overlap destabilizes existing methods because they treat all samples equally, allowing those nearly impossible treatment sequences—magnified by extreme inverse propensity weights—to dominate. The authors define the **propensity weight** for CAPO as $\omega_j^{\bar a}(\bar h_\ell)=\mathbb{E}\big[\prod_{k=j}^{t+\tau}\pi_k^{\bar a}(\bar H_k)\mid \bar H_\ell=\bar h_\ell\big]=p(A_{j:t+\tau}=a_{j:t+\tau}\mid \bar H_\ell=\bar h_\ell)$, the probability of following the target sequence from time $j$ onwards. For CATE, the **overlap weight** is the product of weights for two sequences: $\omega_j^{\bar a,\bar b}=\omega_j^{\bar a}\,\omega_j^{\bar b}$. Intuitively, the overlap weight upweights samples that have a high probability of receiving **both** $\bar a$ and $\bar b$, and downweights samples likely to follow only one. This guides estimation to regions where counterfactuals have data support, avoiding extreme weights at the source. Notably, when $\tau=0$ (static setting), the overlap weight coincides with that of the R-learner (Nie & Wager, 2021), making the WO-learner a non-trivial extension of the R-learner to time-series.

**2. Weighted Population Risk: Using weighted oracle risk as an optimizable target**

Weights must be implemented as a risk function that corrects for time-varying confounding. The authors first define the ideal **weighted oracle risk**:
$$L^*(g;\eta^\circ)=\frac{1}{\mathbb{E}[\omega_t^\circ(\bar H_t)]}\,\mathbb{E}\Big[\omega_t^\circ(\bar H_t)\big(\mu_t^\circ(\bar H_t)-g(\bar H_t)\big)^2\Big],$$
which performs weighted regression on the true value $\mu_t^\circ$. Since $\mu_t^\circ$ is unknown, the authors derive an **estimable weighted population risk** (Theorem 4.3):
$$L(g;\eta^\circ)=\frac{1}{\mathbb{E}[\omega_t^\circ(\bar H_t)]}\,\mathbb{E}\Big[\rho_t^\circ(\bar Z_{t+\tau})\big(\xi_t^\circ(\bar Z_{t+\tau})-g(\bar H_t)\big)^2\Big],$$
which can be computed using only observed data and nuisance estimates but shares the same minimizer as the oracle risk. The random weight $\rho_t$ satisfies $\mathbb{E}[\rho_t^\circ(\bar Z_{t+\tau})\mid \bar H_t]=\omega_t^\circ(\bar H_t)$, acting as an "unbiased randomized version" of the overlap weight. Corollary 4.4 proves that since positivity ensures $\omega_t^\circ>0$, the minimizer is $g=\mu_t^\circ$, thus **correctly adjusting for time-varying confounding**.

**3. WO Pseudo-outcomes and Neyman Orthogonality: Insulating the final estimate from nuisance errors**

Embedding nuisance estimates ($\pi_j, \mu_j, \omega_j$) in the risk risks contaminating the final HTE. Existing IPW/RA/IVW methods suffer because nuisance errors enter the pseudo-outcome as **first-order bias**, which accumulates exponentially in time-series. The authors derive specific **WO pseudo-outcomes** by orthogonalizing the weighted oracle risk. For CATE, it takes the form:
$$\xi_t^{\bar a,\bar b}(\bar Z_{t+\tau})=\mu_t^{\bar a,\bar b}(\bar H_t)+\frac{\omega_t^{\bar a,\bar b}(\bar H_t)}{\rho_t^{\bar a,\bar b}(\bar Z_{t+\tau})}\Big(\gamma_t^{\bar a,\bar b}(\bar Z_{t+\tau})-\mu_t^{\bar a,\bar b}(\bar H_t)\Big),$$
where its structure is derived partly from DR pseudo-outcomes $\gamma_t$ and partly from the orthogonalization of the overlap weights. Theorem 4.5 proves this risk is Neyman-orthogonal to **all** nuisance functions by showing the pathwise derivative's **cross-second derivative with respect to any nuisance function is zero** ($D_{h_j}D_g L=0$). Practically, nuisance estimation errors enter the HTE only as lower-order terms, providing local robustness. This is critical in time-series where cascaded nuisance estimates would otherwise amplify errors through chain-rule propagation.

**4. Model-Agnostic Two-Stage Cross-Fitting: Applicable to any backbone**

As a "recipe," WO-learner is transparent to specific network architectures. Training (Algorithm 1) uses sample splitting $\lambda\in(0,1)$: one part $D^\eta$ estimates nuisance $\hat\eta^\circ$, while the other part $D^g$ evaluates and constructs $\hat\gamma_t, \hat\rho_t, \hat\xi_t$, minimizing the empirical weighted risk:
$$\hat L(\hat g_\theta;\eta^\circ)=\frac{1}{\sum_i \hat\omega_t^\circ(\bar H_{t,i})}\sum_{i=1}^{\lfloor\lambda n\rfloor}\hat\rho_t^\circ(\bar Z_{t+\tau,i})\big(\hat\xi_t^\circ(\bar Z_{t+\tau,i})-\hat g_\theta(\bar H_{t,i})\big)^2$$
via gradient descent. Weights are estimated recursively using the expectation pull-out property (Eq. 18).

### Loss & Training
The core objective is the weighted empirical risk $\hat L(\hat g_\theta;\eta^\circ)$. Key strategies: 1) Two-stage cross-fitting to prevent nuisance overfishing. 2) Shared architectures (e.g., Transformers) across all meta-learners for fair comparison. 3) Standardized training with 5 random seeds.

## Key Experimental Results

### Main Results
Evaluation was conducted on synthetic ($D_\gamma, D_\pi, D_\mu, D_N$), semi-synthetic (MIMIC-III based), and real-world datasets using CATE RMSE. Comparison targets included HA / RA / IPW / DR / IVW.

Results for low-overlap dataset $D_\gamma$ (higher $\gamma$ means lower overlap):

| Overlap $\gamma$ | HA | RA | IPW | DR | IVW | WO (Ours) | Gain |
|------|------|------|------|------|------|------|------|
| 0.5 (High) | 0.17 | 0.10 | 0.09 | 0.06 | 0.06 | **0.03** | 54.4% |
| 1.0 | 0.19 | 0.11 | 0.10 | 0.06 | 0.05 | **0.02** | 58.4% |
| 4.0 | 0.36 | 0.12 | 0.70 | 0.26 | 0.62 | **0.10** | 13.6% |
| 5.0 (Low) | 0.22 | 0.11 | 0.33 | 0.17 | 0.17 | **0.05** | 50.2% |

As $\gamma$ increases, IPW/DR/IVW RMSE and variance deteriorate significantly (IPW hits 0.70±0.76 at $\gamma=4$), whereas WO remains low and stable. On semi-synthetic MIMIC-III data (low overlap, complex confounding), WO was the **only** method stable across all horizons—IVW exploded to 879.80±1243.54 at horizon=4, while WO remained at 0.17±0.07.

### Ablation Study
Synthetic datasets isolated specific difficulties to verify the two mechanisms (overlap weighting / Neyman orthogonality):

| Challenge | Comparison | Key Finding |
|------|------|------|
| $D_\gamma$ Low Overlap | vs IPW/DR/IVW | Overlap weights prevent variance explosion; gain up to 58.4%. |
| $D_\pi$ Complex Propensity | vs IPW/DR/IVW | Propensity error no longer amplifies exponentially at large horizons. |
| $D_\mu$ Complex Response | vs RA | Orthogonality to response functions; WO resists degradation as dimension increases. |
| $D_N$ Low Samples | vs All | Nuisance error propagates only as second-order; WO is stable across sample sizes. |
| LSTM Backbone | — | Conclusions hold regardless of backbone (model-agnosticism). |

### Key Findings
- **Weights handle overlap; Orthogonality handles misspecification**: $D_\gamma$ demonstrates the value of weighting, while $D_\mu$ shows the value of orthogonality. Both are necessary for semi-synthetic stability.
- **Inverse propensity methods are fragile in time-series**: Explosive decay of weights makes IPW/DR/IVW highly unstable as the horizon increases.
- **RA is more stable than IPW in low overlap** (due to lack of inverse weights) but fails when response functions are complex, whereas WO remains robust.

## Highlights & Insights
- **Weighting instead of clipping**: Clipping thresholds cannot be calibrated without counterfactual outcomes. WO uses data-driven overlap weights to determine sample reliability—a cleaner approach applicable to any scenario where inverse propensity variance is an issue.
- **Generalization of R-learner**: WO reduces to R-learner at $\tau=0$, providing a solid theoretical foundation as a natural extension of orthogonal learning to sequential settings.
- **Breaking the error chain**: In time-series, nuisance errors are cascaded. Neyman orthogonality reduces this amplification from first-order to second-order, a design principle highly relevant for sequential causal inference.
- **True Model-Agnosticism**: Consistent performance across Transformer and LSTM backbones ensures the strategy can be integrated into any existing time-series model.

## Limitations & Future Work
- The method relies on the standard identification assumptions (consistency, positivity, sequential ignorability), which may be violated in real data by unobserved confounding.
- Overlap weights focus estimation on high-overlap regions, essentially trading precision in low-overlap areas for global stability; the cost of this trade-off requires further study.
- The weight function $\omega_j$ still requires estimation; the behavior of this estimate in extremely low-overlap finite-sample settings could be further characterized.
- Evaluation relies on ground-truth HTE from synthetic/semi-synthetic data; more real-world deployment validation is needed.

## Related Work & Insights
- **vs DR-learner**: DR is orthogonal but relies on inverse propensity weighting, causing variance explosion in low-overlap time-series. WO incorporates DR structures but adds overlap weighting and re-orthogonalizes.
- **vs IVW-learner**: IVW uses inverse variance weighting but is **not** orthogonal, allowing propensity errors to propagate. WO's weights are orthogonalized, ensuring only second-order error propagation.
- **vs RA-learner**: RA is relatively stable (no inverse weights) but fails with complex/high-dimensional responses because it lacks orthogonality; WO remains robust to both.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First meta-learner to combine overlap weighting and Neyman orthogonality for time-series.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across diverse synthetic cases and semi-synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression from motivation to theoretical derivation and empirical proof.
- **Value**: ⭐⭐⭐⭐ Addresses the critical pain point of low overlap in longitudinal data with a model-agnostic solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Overlap-Adaptive Regularization for Conditional Average Treatment Effect Estimation](overlap-adaptive_regularization_for_conditional_average_treatment_effect_estimat.md)
- [\[ICLR 2026\] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)
- [\[ICML 2026\] Rank-Learner: Orthogonal Ranking of Treatment Effects](../../ICML2026/causal_inference/rank-learner_orthogonal_ranking_of_treatment_effects.md)
- [\[ICLR 2026\] Modeling Interference for Treatment Effect Estimation in Network Dynamic Environment](modeling_interference_for_treatment_effect_estimation_in_network_dynamic_environ.md)
- [\[ICLR 2026\] Matching without Group Barrier for Heterogeneous Treatment Effect Estimation](matching_without_group_barrier_for_heterogeneous_treatment_effect_estimation.md)

</div>

<!-- RELATED:END -->
