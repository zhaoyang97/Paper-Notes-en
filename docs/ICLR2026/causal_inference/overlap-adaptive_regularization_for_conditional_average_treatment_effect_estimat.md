---
title: >-
  [Paper Note] Overlap-Adaptive Regularization for Conditional Average Treatment Effect Estimation
description: >-
  [ICLR 2026][Causal Inference][Paper Note] Addressing the persistent challenge of learning in "low-overlap regions" for Conditional Average Treatment Effect (CATE) estimation, this paper proposes Overlap-Adaptive Regularization (OAR). The regularization strength of the second-stage model in two-stage meta-learners varies inversely with the overlap weight $\nu(x
tags:
  - ICLR 2026
  - Causal Inference
date: 2026-05-08
content_hash: d2b6d76c0cd67dda
---
# Overlap-Adaptive Regularization for Conditional Average Treatment Effect Estimation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HMMSnGgYOy](https://openreview.net/forum?id=HMMSnGgYOy)  
**Code**: https://github.com/Valentyn1997/OAR  
**Area**: Causal Inference / CATE Estimation / Meta-learners  
**Keywords**: Conditional Average Treatment Effect, overlap weights, adaptive regularization, Neyman orthogonality, meta-learners

## TL;DR
Addressing the persistent challenge of learning in "low-overlap regions" for Conditional Average Treatment Effect (CATE) estimation, this paper proposes Overlap-Adaptive Regularization (OAR). The regularization strength of the second-stage model in two-stage meta-learners varies inversely with the overlap weight $\nu(x)$ (stronger regularization for lower overlap). It further introduces dOAR, a debiased version that maintains Neyman orthogonality, consistently outperforming "constant regularization" across multiple (semi-)synthetic datasets.

## Background & Motivation
**Background**: Estimating CATE $\tau(x)=\mathbb{E}[Y[1]-Y[0]\mid X=x]$ from observational data is a core task in causal machine learning, directly used in personalized medicine to predict patient responses to specific treatments. Current SOTA methods are **two-stage Neyman orthogonal meta-learners** (DR-learner, R-learner, IVW-learner): the first stage estimates nuisance functions $\eta=(\mu_0,\mu_1,\pi)$, and the second stage projects the pseudo-outcomes $\phi(Z,\eta)$ onto a target model class $\mathcal{G}$. Their advantages include being model-agnostic and exhibiting first-order insensitivity to nuisance estimation errors due to Neyman orthogonality.

**Limitations of Prior Work**: Meta-learner performance is constrained by the degree of **overlap**, represented by $\nu(x)=\pi(x)(1-\pi(x))$ based on the propensity score—specifically, whether patients with similar covariates receive different treatments. In medicine, overlap is often violated: certain patient types may receive only one treatment per clinical guidelines. Consequently, counterfactual samples are sparse in these low-overlap regions, making CATE extremely difficult to learn. Existing strategies are suboptimal: (1) **Retargeting** incorporates overlap weights into the error term to truncate or downweight low-overlap regions, but this leaves the model behavior uncontrolled in those areas or causes it to estimate a different causal estimand (R-/IVW-learners estimate Weighted Average Treatment Effect, WATE, rather than CATE in low-overlap regions). (2) **Constant Regularization (CR)** applies a uniform reduction in CATE heterogeneity across the entire space regardless of overlap.

**Key Challenge**: Pseudo-outcomes in low-overlap regions exhibit immense variance (due to poor first-stage extrapolation or exploding inverse propensity scores), necessitating stronger regularization. Conversely, high-overlap regions have sufficient samples and require model flexibility. CR using a single $\lambda$ cannot balance both—when paired with a DR-learner, "underfitting in low-overlap regions and overfitting in high-overlap regions occur simultaneously," while with R-/IVW-learners, high regularization leads to WATE.

**Core Idea**: Make regularization strength **adaptive to overlap**—by setting the regularization function $\lambda(\nu)$ proportional to the inverse overlap $1/\nu$, such that regularization is strong in low-overlap regions (enforcing simplicity and smoothness) and weak in high-overlap regions (retaining flexibility). This is the first work to directly incorporate overlap weights into the meta-learner's **regularization term** rather than the error term.

## Method

### Overall Architecture
OAR maintains the skeleton of two-stage meta-learners but replaces the regularization term in the second-stage target risk. Recalling the general form of Neyman orthogonal risk:

$$\mathcal{L}(g,\eta)=\underbrace{\mathbb{E}\big[\rho(A,\pi(X))(\phi(Z,\eta)-g(X))^2\big]}_{\text{Error term }E}+\underbrace{\Lambda(g;P(X))}_{\text{Regularization term }\Lambda}$$

where $\rho(A,\pi(X))\ge 0$ is the debiasing weight and $\phi(Z,\eta)$ is the pseudo-outcome satisfying $\mathbb{E}[\phi(Z,\eta)\mid X=x]=\tau(x)$. DR-/R-/IVW-learners differ only in their choices of $w, \rho, \phi$. Traditional CR sets the regularization to a constant independent of overlap (e.g., $\Lambda=\lambda\|\beta\|_2^2$). OAR's mechanism is to replace the constant $\lambda$ with an overlap-dependent function $\lambda(\nu(x))$, providing specific implementations for different model classes (parametric/non-parametric) and adding a debiasing correction to restore Neyman orthogonality.

### Key Designs

**1. Overlap-Adaptive Regularization Function: Letting Strength $\propto 1/\nu$**

Addressing the "one-size-fits-all" limitation of CR, OAR defines the regularization term as $\Lambda_{\text{OAR}}=\Lambda(g;P(X,A);\lambda(\nu(X)))$, requiring $\lambda(\nu)>0$ and $\lambda(\nu)\propto 1/\nu$. The intuition is: as $\nu(x)\to 0$ (low overlap), $\lambda(\nu)\to\infty$, forcing the model to be simpler/smoother; as $\nu(x)\to 1/4$ (perfect overlap, $\pi=0.5$), $\lambda(\nu)\to 0$, leaving maximum flexibility. The authors provide three candidate functions:

$$\lambda_m(\nu)=\tfrac{1}{4}\nu^{-1}-1,\quad \lambda_{\log}(\nu)=-\log(4\nu),\quad \lambda_{m2}(\nu)=\tfrac{1}{16}\nu^{-2}-1$$

Named multiplicative, log, and squared multiplicative respectively, with increasing penalty intensity. A fundamental difference from retargeting is: retargeting **downweights overlap in the error term** ($\mathbb{E}[\rho\mid X]=\nu$), whereas OAR **upweights overlap in the regularization term**. Furthermore, Proposition 1 proves that the average OAR regularization $\mathbb{E}[\lambda(\nu(X))]$ equals or is bounded by the $f$-divergence between $P(X)$ and $P(X\mid A=a)$, requiring only propensity score estimation without estimating distribution distances in high-dimensional $X$.

**2. Implementation in Parametric Models: OAR Noise Regularization and OAR Dropout**

For parametric models $\mathcal{G}=\{g(\cdot;\beta,c)\}$ like linear models or neural networks, the authors adapt "noise injection" techniques to be overlap-adaptive. **OAR Noise Regularization** adds Gaussian noise $\xi\sim\mathcal{N}(0,\sqrt{\lambda(\nu(X))}^2)$ with variance proportional to inverse overlap, i.e., $\sigma^2\propto 1/\nu(x)$. For linear models, the explicit form (Prop 2) is $E+\|\beta\|_2^2\,\mathbb{E}[\rho(A,\pi(X))\lambda(\nu(X))]$, equivalent to a ridge regression with constant $\mathbb{E}[\rho\cdot\lambda(\nu)]$. **OAR Dropout** drops neurons with probability $p(\nu)=\lambda(\nu)/(\lambda(\nu)+1)\in(0,1)$. The linear explicit form (Prop 3) is a quadratic form $E+\beta^\top\mathrm{diag}(\Sigma_{\rho(\cdot,\pi)}\cdot\lambda(\nu))\beta$, equivalent to scaling features by $\tilde X_j=X_j/\sqrt{\mathbb{E}[\rho\cdot\lambda(\nu)\cdot X_j^2]}$ before ridge regression. 

**3. Debiased OAR (dOAR): One-Step Correction for Neyman Orthogonality**

Original OAR depends on estimated overlap weights $\hat\nu(x)$, which can be biased by first-order errors in $\hat\pi$. The authors construct dOAR using one-step bias correction based on efficient influence functions (IF). The correction term $C^\diamond$ (Eq. 10–11) ensures dOAR is first-order insensitive to errors in $\hat\pi$, restoring **Neyman orthogonality** for the entire learner.

**4. Non-parametric Extensions and Theoretical Guarantees**

For Kernel Ridge Regression (KRR), OAR sets the regularization as a weighted RKHS norm $\Lambda_{\text{OAR}}=\|\sqrt{\lambda(\nu)}\,g\|_{\mathcal{H}_K}^2$. Theoretically, Prop 5 uses bias-variance decomposition to show that under a "low overlap-low heterogeneity inductive bias (LOLH-IB)," OAR/dOAR reduces variance compared to CR without significantly increasing bias, providing formal support for adaptive regularization.

### Loss & Training
Two-stage implementation: Stage 1 estimates nuisance $\hat\eta$ using cross-validated neural networks. Stage 2 fits the target network using the empirical risk $\hat{\mathcal{L}}$ or KRR closed-form solution. To ensure a fair comparison with CR, the authors scale the regularization function $\tilde\lambda(\nu)$ such that its average regularization equals the constant $\lambda$. The **multiplicative function $\lambda_m$** is recommended as the default.

## Key Experimental Results

### Main Results
Evaluated on four (semi-)synthetic datasets using out-of-sample rPEHE ($\text{rPEHE}_{\text{out}}$).

| Dataset | Scale | Key Observation |
|--------|------|----------|
| IHDP | $n=672+75,\,d_x=25$ | Severe overlap violation; OAR/dOAR versions achieve the best results across all meta-learners. |
| ACIC 2016 | $n=4802,\,d_x=82$ (77 subsets) | dOAR significantly outperforms CR in over half of the datasets under DR-learner. |
| HC-MNIST | $d_x=784+1$ | Natural low overlap in high dimensions; OAR/dOAR outperform CR in most cases. |

HC-MNIST (Multiplicative $\lambda_m/p_m$, rPEHE_out, lower is better):

| Learner | Method | Noise reg. $\lambda=0.25$ | Dropout $p=0.3$ |
|--------|------|------|------|
| DR | CR | 0.711 | 0.727 |
| DR | OAR | 0.696 (−0.015) | 0.713 (−0.014) |
| DR | dOAR | 0.684 (−0.027) | 0.705 (−0.021) |
| IVW | CR | 1.028 | 1.117 |
| IVW | OAR | 0.984 (−0.044) | 1.061 (−0.056) |

### Ablation Study
ACIC 2016 (DR-learner, % of datasets where OAR/dOAR significantly outperforms CR):

| Function | Method | Noise reg. $\lambda=0.05$ | Dropout $p=0.3$ |
|----------|------|------|------|
| $\lambda_m$ | OAR | 31.17% | 41.56% |
| $\lambda_m$ | dOAR | 57.14% | 70.13% |
| $\lambda_{m2}$ | dOAR | 76.62% | 64.94% |

### Key Findings
- **Debiasing (dOAR) is almost always superior to original OAR**: The win-rate of dOAR is significantly higher, confirming the importance of one-step correction.
- **DR-learner is the best partner**: OAR/dOAR + DR-learner is consistently effective as it balances high variance with adaptive regularization strength.
- **Multiplicative function is most robust**: Supported by theory ($\lambda_m$ is a stable approximation of the optimal $\nu^{-1/3}$) and its equivalence to R-learner under KRR.

## Highlights & Insights
- **Repositioning Regularization**: Unlike existing work that places overlap weights in the error term (retargeting), this is the first to place them in the regularization term.
- **"Causalizing" Classical Regularization**: Adapts dropout/noise regularization into overlap-adaptive versions, assigning a per-region "effective $l_2$."
- **Debiasing Heuristics**: Using EIF to transform a heuristic trick into an estimator with theoretical guarantees (Neyman orthogonality).
- **Explicit Inductive Bias**: Identifies the LOLH-IB assumption—low overlap should favor simpler models—making it a transparent and practical inductive bias.

## Limitations & Future Work
- **Dependency on LOLH-IB**: If low-overlap regions actually contain high CATE heterogeneity, OAR may smooth out real signals.
- **Reliance on (Semi-)synthetic Data**: Evaluation requires counterfactual ground truth, which is unavailable in pure observational studies.
- **Scaling to aligned CR**: Selecting the overall OAR strength without ground truth remains an open problem.

## Related Work & Insights
- **vs. Retargeting (R-/IVW-learner)**: Retargeting changes the estimand to WATE in low-overlap areas; OAR provides ATE by constraining generalization smoothness in those regions.
- **vs. Representation Balancing (e.g., CFR/TARNet)**: Balancing uses distribution distances as average regularization; OAR achieves a similar effect using only propensity scores, which is more computationally efficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Overlap-Weighted Orthogonal Meta-Learner for Treatment Effect Estimation over Time](overlap-weighted_orthogonal_meta-learner_for_treatment_effect_estimation_over_ti.md)
- [\[ICLR 2026\] IGC-Net for Conditional Average Potential Outcome Estimation Over Time](igc-net_for_conditional_average_potential_outcome_estimation_over_time.md)
- [\[ICLR 2026\] Matching without Group Barrier for Heterogeneous Treatment Effect Estimation](matching_without_group_barrier_for_heterogeneous_treatment_effect_estimation.md)
- [\[ICLR 2026\] Modeling Interference for Treatment Effect Estimation in Network Dynamic Environment](modeling_interference_for_treatment_effect_estimation_in_network_dynamic_environ.md)
- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)

</div>

<!-- RELATED:END -->
