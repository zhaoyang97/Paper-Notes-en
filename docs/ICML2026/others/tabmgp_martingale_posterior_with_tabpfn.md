---
title: >-
  [Paper Note] TabMGP: Martingale Posterior with TabPFN
description: >-
  [ICML 2026][Martingale Posterior] This work treats TabPFN, a pre-trained tabular Transformer, directly as the prediction rule for a Martingale Posterior (MGP). By performing in-context forward rolling sampling…
tags:
  - "ICML 2026"
  - "Martingale Posterior"
  - "TabPFN"
  - "Tabular Foundation Models"
  - "Generalized Bayes"
  - "Credible Sets"
date: 2026-05-08
content_hash: 5c03f93bd5081323
---

# TabMGP: Martingale Posterior with TabPFN

**Conference**: ICML 2026  
**arXiv**: [2510.25154](https://arxiv.org/abs/2510.25154)  
**Code**: Not yet released  
**Area**: Self-supervised / Tabular Foundation Models / Bayesian Uncertainty  
**Keywords**: Martingale Posterior, TabPFN, Tabular Foundation Models, Generalized Bayes, Credible Sets

## TL;DR
This work treats TabPFN, a pre-trained tabular Transformer, directly as the prediction rule for a Martingale Posterior (MGP). By performing in-context forward rolling sampling, it obtains credible sets for parameters $\theta$ under arbitrary loss functions. This approach avoids manual design of priors/likelihoods and tuning of hyper-parameters, while outperforming manual MGP and classical Bayesian methods in both coverage and credible set area across 30 real/synthetic scenarios.

## Background & Motivation

**Background**: Classical Bayesian inference provides uncertainty for parameters $\theta$ but requires explicit specification of priors and likelihoods. The Martingale Posterior (MGP, Fong et al. 2023) replaces the prior-likelihood pair with a "prediction rule" $(P_i)_{i\ge 0}$ and uses a loss function $\ell(z, \theta)$ to define the functional of interest $\theta(F)=\arg\min_\vartheta \int \ell(z, \vartheta)\,\mathrm{d}F(z)$, bypassing prior specification.

**Limitations of Prior Work**: Existing MGP literature almost exclusively uses "handcrafted" prediction rules (Bayesian bootstrap, bivariate copula, autoregressive GP, vine copula, etc.). Each introduces one or more smoothing/bandwidth hyperparameters that must be retuned for every dataset. Furthermore, they perform well only in low-dimensional settings or specific distribution families, struggling with the complex structures of modern tabular data.

**Key Challenge**: Handcrafted prediction rules prevail because the community treats the "strict martingale property $\mathbb{E}[P_{i+1}(A)\mid Z_{1:i}]=P_i(A)$" as a necessary condition and designs every new rule accordingly. The authors argue that the martingale property is a sufficient, but not necessary, condition for the existence of $F_\infty$. Over-emphasizing it hinders the integration of high-capacity predictors.

**Goal**: Can a foundation model (TabPFN), pre-trained on large-scale synthetic tabular data to approximate the Bayesian PPD, be used directly as the prediction rule for MGP? This would (i) eliminate manual design, (ii) leverage pre-trained coverage capabilities, and (iii) empirically provide near-nominal coverage even if the strict martingale property is violated.

**Key Insight**: TabPFN possesses three natural characteristics that align with MGP: ① In-context learning, outputting a predictive distribution $y\mid x$ for new data without fine-tuning; ② Row-permutation invariance, avoiding the need to manually average over permutations as in copulas; ③ A training objective that approximates the Bayesian PPD, which is the ideal prediction rule in MGP.

**Core Idea**: Use TabPFN to provide the conditional distribution $Y\mid X$ and Bayesian bootstrap for the marginal distribution of $X$. Map "forward rolling sampling + loss minimization" to autoregressive inference in the Transformer to obtain $\theta(F_N^{(l)})$ as an approximate posterior sample of $\theta(F_\infty)\mid z_{1:n}$.

## Method

### Overall Architecture
Input: Observed data $z_{1:n}=(x_i, y_i)_{i=1}^n$, loss function $\ell(z, \theta)$, rolling length $N$ (typically $N=n+T, T=500$), number of samples $L$.  
Output: $L$ approximate posterior samples $\{\theta^{(l)}\}_{l=1}^L \sim \theta(F_\infty)\mid z_{1:n}$, used to construct the credible set $\widehat{C}_{1-\alpha}(z_{1:n})$.

The pipeline consists of three stages:
1. **Forward Rolling**: For each $l \in \{1, \dots, L\}$, generate $z_{n+1:N}^{(l)}$ autoregressively starting from $z_{1:n}$. $x_{i+1}^{(l)}$ is drawn from the empirical distribution of $x_{1:i}^{(l)}$ (Bayesian bootstrap), and $y_{i+1}^{(l)} \sim \mathrm{TabPFN}(\cdot \mid x_{i+1}^{(l)}, z_{1:i}^{(l)})$.
2. **Risk Minimization**: For each rollout, form an empirical measure $F_N^{(l)}=\tfrac1N\sum_{i=1}^N\delta_{z_i^{(l)}}$ and compute $\theta^{(l)}=\arg\min_\theta\sum_i\ell(z_i^{(l)}, \theta)$.
3. **Credible Set**: Approximate the $(1-\alpha)$ joint credible set using the covariance trace and ellipsoidal approximation of $\{\theta^{(l)}\}$.

All $L$ paths are independent, allowing for natural parallelization.

### Key Designs

1. **TabPFN as Prediction Rule + Bayesian Bootstrap for Covariate Marginals**:
    - **Function**: Replaces handcrafted MGP prediction rules with a pre-trained Transformer for the $Y\mid X$ conditional distribution and uses bootstrap for the $X$ marginal.
    - **Mechanism**: In supervised settings, MGP requires the joint distribution of $(X, Y)$. Since modeling high-dimensional $X$ distributions is difficult, the authors follow the "joint method" of Fong et al. (2023)—TabPFN handles the conditional $y_{i+1}\sim P_i(\cdot\mid x_{i+1}, z_{1:i})$ while $x_{i+1}\sim \mathrm{Empirical}(x_{1:i})$. This precisely utilizes TabPFN’s strength (supervised prediction) while delegating its weakness (unconditional covariate modeling) to bootstrap.
    - **Design Motivation**: Traditional copula rules require explicit averaging over permutations and manual bandwidth tuning. TabPFN is inherently row-permutation invariant and parameter-free, eliminating the two biggest engineering pain points of the copula framework.

2. **Dropping "Strict Martingale Property" in Favor of Empirical Diagnostics**:
    - **Function**: Uses three empirical metrics to judge whether TabMGP provides valid uncertainty, rather than relying on formal martingale proofs.
    - **Mechanism**: The authors acknowledge that TabPFN satisfies neither the strict martingale condition $\mathbb{E}[P_{i+1}(A)\mid Z_{1:i}]=P_i(A)$ nor the relaxed a.c.i.d. (almost conditionally identically distributed) condition. However, they argue these are sufficient but not necessary for $F_\infty$ to exist. They use three diagnostics: (a) **Path Stability**—monitoring if $\mathbb{E}_{F_N}[\tfrac1p\|\theta(F_n)-\theta(F_N)\|_1]$ plateaus as $N$ increases; (b) **Frequentist Coverage**—checking if the $(1-\alpha)$ set covers the population risk minimizer $\theta(F^\star)$; (c) **Posterior Contraction**—ensuring the set tightens as $n$ increases.
    - **Design Motivation**: Since the martingale property of high-capacity neural networks cannot be proven with current tools, it is more practical to validate effectiveness via empirical diagnostics than to reject the model based on unprovable conditions.

3. **Generating Parameter Posteriors instead of Predictive Distributions**:
    - **Function**: TabMGP outputs posterior samples for $\theta(F_\infty)\mid z_{1:n}$, serving any scientific quantity defined by a loss function, rather than just predictive distributions for new samples.
    - **Mechanism**: MGP is the convergence of BPI (Bayesian Predictive Inference) and GB (Generalized Bayes). BPI replaces priors/likelihoods with prediction rules; GB changes the inference target from "parameters under likelihood" to "minimizers of arbitrary loss." TabMGP implements both: it uses TabPFN for the rule and arbitrary $\ell(z, \theta)$ (e.g., squared loss for linear regression, cross-entropy for logistic) for the target.
    - **Design Motivation**: The "scientific estimator" $\theta$ of interest is often unrelated to the Transformer's implicit latent model. Providing only predictive distributions prevents making credible sets for $\theta$. The "functional posterior" structure of MGP fills this gap.

### Loss & Training
TabMGP **has no training phase**—TabPFN is pre-trained on large-scale synthetic data and acts as an inference engine. The loss function $\ell$ is specified by the user at inference time: $\ell(x, y, \theta)=(y-[1\ x^\top]\theta)^2$ for linear regression or $\ell(x, y, \theta)=-\log\Pr(y=k)$ (softmax) for $K$-class classification. Key hyperparameters: rolling length $T=500$ (up to $1000$ for slow convergence) and number of independent rollouts $L$ (default $100\sim 1000$).

## Key Experimental Results

### Main Results
Over 30 setups (11 synthetic + 19 real datasets from OpenML/UCI) with a target coverage of 0.95. Below is a selection for linear regression; "Rate" should be close to 0.95 and "Size" (trace of posterior covariance) should be small.

| Setup | TabMGP Rate / Size | BB Rate / Size | Copula Rate / Size | Bayes Rate / Size | Asymptotic Rate / Size |
|-------|--------------------|----------------|--------------------|-------------------|------------------------|
| $\mathcal{N}(0,1)$ | **1.00 / 0.45** | 0.55 / 0.09 | 0.99 / 0.35 | 1.00 / 0.65 | 1.00 / 1.31 |
| $t_3$ (Heavy tail) | **1.00 / 0.48** | 0.66 / 0.14 | 0.97 / 0.35 | 0.98 / 0.65 | 0.98 / 1.31 |
| heterosc. $s_3$ | **1.00 / 0.33** | 0.53 / 0.02 | 1.00 / 0.37 | 1.00 / 0.65 | 1.00 / 1.31 |
| concrete (Real) | 0.91 / **0.06** | 0.80 / 0.05 | 1.00 / 0.12 | 0.87 / 0.05 | 1.00 / 0.10 |
| airfoil (Real) | 0.96 / **0.08** | 0.93 / 0.05 | 0.97 / 0.11 | 0.96 / 0.06 | 1.00 / 1.21 |
| energy (Real) | **1.00 / 0.04** | 0.80 / 0.01 | 1.00 / 0.06 | — | — |

### Ablation Study
| Config / Diagnosis | Key Metric | Description |
|--------------------|------------|-------------|
| TabMGP $T=500$ | All 30 setups plateau | Path stability $\mathbb{E}[\tfrac1p\|\theta(F_n)-\theta(F_N)\|_1]$ converges within $T=500$. |
| TabMGP $T=1000$ | Slow setups plateau | No path divergence observed, indirectly suggesting $F_\infty$ exists. |
| Martingale Detection | Visual deviation from Martingale/a.c.i.d. | TabPFN is not strictly martingale or a.c.i.d., yet coverage remains near nominal. |
| Alt. Baseline (Copula+TabPFN init) | Worse than TabMGP in most setups | Confirms that retaining TabPFN as the rule is better than using it merely as an initialization. |

### Key Findings
- TabMGP's coverage is most stable: $\ge 0.97$ in all synthetic scenarios; BB severely under-covers (lack of forward diversity at $n=20$), while Bayes/Asymptotic over-cover with massive sets due to failed asymptotic approximations at low $n$.
- TabMGP posteriors often exhibit skewness and multimodality, whereas BB/Copula/Bayes are nearly Gaussian—indicating the Transformer's implicit model captures non-Gaussian structural information.
- Copulas excel in near-Gaussian settings but break down when data deviates (severe under-coverage on kin8nm, over-coverage on quake), showing sensitivity to structural assumptions. TabMGP is more robust due to pre-training.

## Highlights & Insights
- **"Martingale property is sufficient, not necessary" is the pivotal breakthrough**: By using empirical diagnostics to decouple "unprovable theoretical conditions" from "observable practical properties," the authors legally integrate high-capacity predictors into the framework.
- **The "Divide and Conquer" approach for TabPFN + Bayesian Bootstrap**: Delegating the hard task of marginal covariate modeling to bootstrap while letting TabPFN handle conditional modeling preserves invariance and rationality while avoiding out-of-distribution $X$ issues for TabPFN.
- **Non-Gaussian posterior shapes as a free benefit**: While manual MGP and Bayes often output Gaussian-like sets, TabMGP provides skewed or multimodal distributions, which are more faithful to real data.
- **Structural Row Invariance = Engineering Discount**: Copula MGP requires explicit averaging over permutations, which is slow and complex; TabPFN reduces this cost to zero at the architecture level.

## Limitations & Future Work
- No strict theoretical guarantees; only empirical "plateaus" provide weak evidence for $F_\infty$, which may be insufficient for scenarios requiring formal proof of coverage.
- TabPFN has a maximum in-context length ($10^3\sim 10^4$ rows). Rolling sampling beyond this limit may degrade or require chunking strategies.
- Evaluated only on linear interpretable models and small-to-medium $n$. Loss minimization for high-dimensional non-linear $\theta$ (e.g., neural network weights) is inherently ill-posed; whether TabMGP scales here is unverified.
- Using bootstrap for $X$ might limit forward diversity in scenarios with extremely unbalanced or sparse covariate distributions. 

## Related Work & Insights
- **vs. Fong et al. (2023) Original MGP**: They use manual copula rules to guarantee the martingale property but require bandwidth tuning; Ours uses pre-trained TabPFN, sacrificing strict martingale property for zero-tuning and cross-dataset robustness.
- **vs. Nagler & Rügamer (2025)**: They also use TabPFN but revert to the copula framework upon finding it non-martingale; Ours keeps TabPFN as the rule, asserting that non-martingaleness does not impede practical validity.
- **vs. Classical Bayes (diffuse $\mathcal{N}(0,10^2)$ prior)**: Classical methods are prior-dominated at low $n/p$. TabMGP uses pre-trained knowledge as an implicit prior, yielding higher robustness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to implement "foundation model as prediction rule" in MGP, systematically challenging the necessity of the martingale property.
- Experimental Thoroughness: ⭐⭐⭐⭐ 30 setups + 5 baselines + 3 diagnostics; however, $\theta$ is limited to low-dimensional linear models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear conceptual layering (BPI, GB, MGP, TabMGP); Algorithm 1 and Figure 1 make forward sampling very clear.
- Value: ⭐⭐⭐⭐⭐ Bridges TabPFN with the Bayesian toolbox; high engineering significance for immediate use by practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)
- [\[ICML 2026\] Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)
- [\[ICML 2026\] Position: Age Estimation Models Do Not Process Biometric Data](position_age_estimation_models_do_not_process_biometric_data.md)
- [\[ICML 2026\] From Generalist to Specialist Representation](from_generalist_to_specialist_representation.md)
- [\[ICML 2026\] CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations.md)

</div>

<!-- RELATED:END -->
