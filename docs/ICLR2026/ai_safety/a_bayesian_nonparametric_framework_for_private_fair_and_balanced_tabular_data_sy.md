---
title: >-
  [Paper Note] A Bayesian Nonparametric Framework for Private, Fair, and Balanced Tabular Data Synthesis
description: >-
  [ICLR 2026][AI Safety][Differential Privacy] This paper embeds a conditional VAE-GAN generator into a Bayesian Nonparametric Learning (BNPL) framework. It utilizes a Dirichlet process for global privacy, a copula base measure for column-wise local privacy, BNP mutual information regularization for fairness, and KL divergence for class balance. It represents the first unified framework with theoretical guarantees to simultaneously handle privacy, fairness…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Fairness"
  - "Class Balance"
  - "Bayesian Nonparametrics"
  - "Dirichlet Process"
  - "VAE-GAN"
  - "Mutual Information Regularization"
date: 2026-05-08
content_hash: dfc1c67bf728385a
---

# A Bayesian Nonparametric Framework for Private, Fair, and Balanced Tabular Data Synthesis

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=j0czDrEnFc](https://openreview.net/forum?id=j0czDrEnFc)  
**Code**: TBD  
**Area**: Trustworthy Machine Learning / Privacy / Fairness / Tabular Data Synthesis  
**Keywords**: Differential Privacy, Fairness, Class Balance, Bayesian Nonparametrics, Dirichlet Process, VAE-GAN, Mutual Information Regularization  

## TL;DR
This paper embeds a conditional VAE-GAN generator into a Bayesian Nonparametric Learning (BNPL) framework. It utilizes a Dirichlet process for global privacy, a copula base measure for column-wise local privacy, BNP mutual information regularization for fairness, and KL divergence for class balance. It represents the first unified framework with theoretical guarantees to simultaneously handle privacy, fairness, and class imbalance constraints while naturally supporting non-binary sensitive attributes.

## Background & Motivation
**Background**: Tabular data synthesis must balance utility with privacy (individual protection, usually via Differential Privacy), fairness (equal treatment across protected groups), and class balance (uniform representation of categories). Existing methods like GANs (e.g., CTGAN) maintain patterns but lack endogenous privacy and flexible class ratio control; DP-SGD methods often amplify imbalance due to non-uniform noise; PATE-based methods scale poorly due to multiple teacher discriminators; and VAE variants (e.g., OVAE) lack formal, tunable privacy guarantees.

**Limitations of Prior Work**: (1) Fairness models (TabFairGAN, DECAF, FairGAN) almost exclusively support binary protected attributes, failing on multi-class sensitive features (e.g., 8 ethnic groups). (2) Privacy and fairness often conflict; privacy noise distorts group-level statistics (especially for small groups), while fairness constraints add complexity to an already noise-perturbed optimization. (3) Most works address only one of the three constraints, lacking a truly joint treatment.

**Key Challenge**: The tension between privacy noise addition and the need for accurate group statistics for fairness, compounded by the requirement to scale to non-binary attributes while maintaining theoretical privacy budgets.

**Goal**: Construct a scalable generation framework with theoretical guarantees that jointly ensures privacy, fairness, and class balance within a single model, naturally supporting non-binary sensitive attributes.

**Core Idea**: **Replace standard training with BNPL**. The generator is wrapped within Dirichlet Process posterior resampling. Privacy stems from "resampling weight randomness + column-wise perturbation of the copula base measure." Fairness is achieved via a "BNP-form lower bound regularization on the mutual information between generated results and sensitive attributes." Class balance is enforced via a "KL divergence penalty between generated class ratios and a uniform distribution." All three are jointly optimized through a unified loss.

## Method

### Overall Architecture
The method is named **CBNP-VAECGAN** (Conditional Bayesian NonParametric VAE-code-GAN). The training set is viewed as samples from a Dirichlet Process $F\sim\mathrm{DP}(a, H_{\text{Pert}})$. $N$ privacy-protected samples are resampled from the finite approximation of the posterior $F^{\mathrm{Pos}}\sim\mathrm{DP}(a+n, H^*)$. These posterior samples are fed into a VAE-code-GAN conditioned on protected attributes, with the loss function incorporating fairness MI regularization and class balance KL regularization. Privacy is injected at the resampling stage via two levels: global (random weights) and local (copula column-wise noise). Fairness and balance are implemented as regularizers during network training.

```mermaid
flowchart TD
    A[Real Table D=(X,Y,S)] --> B[Construct DirP Base Measure H_Pert<br/>Copula Column-wise Local Privacy]
    B --> C[DirP Posterior Finite Approximation<br/>Random Weights J → Global Privacy]
    C --> D[Posterior Samples D^Pos<br/>Quantile Transform for Cont. + One-hot for Disc.]
    D --> E[Conditional VAE-code-GAN<br/>Conditioned on Protected Attribute S]
    E --> F[Utility Loss<br/>Reconstruction + Adversarial + MMD]
    E --> G[DirPMINE Fairness Regularizer<br/>min MI(Ỹ, S)]
    E --> H[Class Balance KL Regularizer<br/>D_KL(Class Ratio, Uniform)]
    F --> I[Joint Optimization<br/>L_Util + λ_F L_Fair + Σλ_B L_Balance]
    G --> I
    H --> I
    I --> J[Synthetic Table<br/>Inverse Quantile Transform + Argmax Decoding]
```

### Key Designs

**1. Global Privacy via Dirichlet Process Resampling: Utilizing weight randomness as a privacy mechanism.** Unlike DP-SGD which adds noise to gradients, this paper places the privacy source at the data resampling step. The posterior DirP, using the Ishwaran-Zarepour finite approximation, is written as $F^{\mathrm{Pos}}_{D^{\mathrm{Pos}}_{1:N}}(\cdot)=\sum_{i=1}^N J^{(a+n)}_{i,N-1}\,\mathbb{I}_{D^{\mathrm{Pos}}_i}(\cdot)$, where weights $(J_{1},\dots,J_{N})\sim\mathrm{Dir}((a+n)/N,\dots)$ are stochastic. This weight randomness perturbs the empirical distribution shape around the base measure. Proposition 1 proves this "Dirichlet mechanism" satisfies $(\epsilon_{\text{glo}}, \delta_{\text{glo}})$-differential privacy for $b$-adjacent empirical distributions, with $\epsilon_{\text{glo}}$ explicitly given by the ratio of beta functions and concentration parameter $a+n$. Corollary 1 further indicates that as $a\to\infty$, $\epsilon_{\text{glo}}\to 0$ and $\delta_{\text{glo}}\xrightarrow{p}0$, providing a knob to slide the privacy-utility tradeoff by adjusting $a$.

**2. Local Privacy via Copula Base Measure: Tailored mechanisms for continuous and discrete columns while preserving dependency.** While global weights protect the distribution "shape," the locations (sample points) in the DirP approximation remain unprotected. Thus, the base measure is constructed as a copula in three steps: first, fit continuous columns with Normal $H^{(C)}_{i_C}=\mathcal{N}(\hat\mu, \hat\sigma^2)$ and discrete columns with Categorical $H^{(D)}_{i_D}=\mathrm{Cat}(K,\hat p)$ as marginal priors (preventing prior-data conflict). Second, apply the Analytic Gaussian Mechanism (AGM) to continuous marginals and the Randomized Response Mechanism (RRM) to discrete marginals. Finally, estimate the correlation matrix $\hat R$ using Pearson/Spearman coefficients and combine the perturbed marginals into a joint base measure $H_{\text{Pert}}(t)=C_{\hat R}\big(F_{\mathrm{AGM},1}(t_1),\dots,F_{\mathrm{RRM},N_D}(t_d)\big)$ using a semi-Gaussian copula. This tailors privacy strength (via budgets $\epsilon^{(C)}_{\text{loc}}, \epsilon^{(D)}_{\text{loc}}$) while preserving feature dependencies. Notably, **the sensitive attribute $S$ is never generated, and its privacy budget is set to $\epsilon_{N_D}=\infty$** to prevent noise from erasing minority groups.

**3. Fairness via DirPMINE Mutual Information Regularization: Turning "output independence from sensitive attributes" into a trainable objective.** Statistical Parity (SP) requires $\Pr(Y=1|S=0)=\Pr(Y=1|S=1)$, which for multi-class $S$ is equivalent to $\mathrm{MI}(Y,S)=0$. Direct MI estimation is difficult. This paper uses the Donsker-Varadhan Lower Bound (DVLB) to write MI as a variational maximum of a discriminator network $T_\upsilon$, replacing expectations with DirP random weights:

$$\mathrm{MI}^{\mathrm{DV}}_{\mathrm{DirP}}(\tilde{Y}^{\mathrm{Pos}}, S^{\mathrm{Pos}})=\max_{\upsilon}\Big\{\sum_{r=1}^N J^{(a+n)}_{r,N-1} T_\upsilon(\tilde Y_r, S_r)-\ln\sum_{r=1}^N J^{(a+n)}_{r,N-1} e^{T_\upsilon(\tilde Y_r, S_{\pi(r)})}\Big\},$$

where $\pi$ is a random permutation of $[N]$ to construct the marginal product distribution. Adding this as a regularizer $\lambda_F L_{\text{Fair}}$ forces the generated output to "decouple" from sensitive attributes in high dimensions. Since the local privacy mechanism is applied only once before fairness optimization and uses zero/light budget for sensitive attributes/results, it avoids distorting group-conditional statistics for small populations.

**4. KL Divergence Class Balance Regularization: A differentiable objective for uniform representation.** Balance for protected attributes is achieved via conditional input (where user-specified ratios for $S$ can be set). For non-protected discrete features $i_D$, a uniform distribution $U_{i_D}\sim\mathrm{Cat}(K, 1/K)$ is defined. The objective minimizes $D_{\mathrm{KL}}(\tilde D^{\mathrm{Pos}(D)}_{i_D}, U_{i_D})=\sum_j \tilde p^{\mathrm{Pos}}_{i_D j}\ln(\tilde p^{\mathrm{Pos}}_{i_D j} K_{i_D})$, estimated using the semi-BNP DVLB. The final loss combines these components with weights $\lambda_F, \lambda_{B_{i_D}}\in[0,1]$: $L_{\text{Util}}+\lambda_F L_{\text{Fair}}+\sum_{i_D} \lambda_{B_{i_D}} L_{\text{Balance}_{i_D}}$.

## Key Experimental Results

Datasets: Adult (Income vs. Gender), COMPAS (Risk vs. Race; $Y$ 3-class, $S$ 8-class, where standard fairness baselines are inapplicable), and Bank Marketing. Utility is measured by MMD and Accuracy/F1 across three classifiers (DTC/LR/MLP).

### Main Results (COMPAS: Maintaining conditional distribution while eliminating bias)

| Ethnic Group | High Risk Prob. ($\lambda_F=0$, No Fairness) | High Risk Prob. ($\lambda_F=1$, With Fairness) |
|---|---|---|
| African-American | 0.146 | 0.103 |
| Asian | 0.058 | 0.101 |
| Caucasian | 0.081 | 0.137 |
| Oriental | 0.062 | 0.110 |

Without fairness constraints, the probability of being labeled "High Risk" for African-Americans (0.146) vs. Asians (0.058) shows significant disparity. Fairness regularization effectively levels these probabilities across ethnicities.

### Ablation Study (COMPAS: $\mathrm{MI}_{\text{True}}=0.0259$, $F1_{\text{True}}=0.913$, $\mathrm{Acc}_{\text{True}}=0.901$)

| Balanced Variables | $\lambda_F$ | MI | MMD | F1 (DTC) | Acc (DTC) |
|---|---|---|---|---|---|
| None | 0 | 0.0263 | — | 0.921 | 0.894 |
| None | 1 | ~0 | 1e−5 | 0.903 | 0.886 |
| Sex/Marital/Language | 1 | ~0 | 1e−5 | 0.917 | 0.883 |

With $\lambda_F=1$, MI drops nearly to zero while MMD remains minimal and utility (F1/Acc) stays close to the real data. Even with additional balancing of three categorical variables, utility loss is negligible.

### Key Findings
- **Tunable Privacy Knob**: Larger $a$ and more private columns decrease utility, but with very small $a$ (e.g., $10^{-6}$), the model maintains high utility even under strong local budgets.
- **Fairness Without Utility Sacrifice**: SP and MI metrics on Adult outperform FairGAN and TabFairGAN, particularly exceeding the SOTA DECAF.
- **Support for Non-binary Attributes**: The 8 races in COMPAS are handled naturally by conditioning, where standard baselines fail.
- **Coexistence of Balance, Fairness, and Utility**: Balancing 'Sex' or 'Marital' status does not compromise utility or fairness.

## Highlights & Insights
- **Shifting Privacy "Source" to Resampling**: Using Dirichlet process weights for global DP and copula base measures for local DP is an innovative approach with explicit $(\epsilon, \delta)$ formulations.
- **Joint Constraints over Layering**: Utility, fairness, and balance are coupled via a unified loss and BNP resampling, unlike existing "modular" approaches.
- **Resolving Privacy-Fairness Tension**: Zero budget for sensitive attributes and light budgets for results minimize the distortion of group-level statistics for small populations.
- **Multi-class Fairness**: Utilizing Mutual Information rather than pairwise SP comparisons allows expansion to non-binary attributes at almost zero additional cost.

## Limitations & Future Work
- The fairness regularizer is based on standard MI (corresponding to SP). Scaling to Equalized Odds or Equal Opportunity would require Conditional MI, which is theoretically more complex to implement for generators.
- Evaluation is primarily on Adult and COMPAS; both have been criticized recently regarding their suitability for fairness evaluation.
- The framework is VAE-GAN based; while the authors suggest injecting Dirichlet processes into LLMs for similar guarantees, this remains future work.
- The privacy-utility tradeoff requires manual selection of column budgets and concentration $a$, lacking an automated selection mechanism.

## Related Work & Insights
- **Privacy Tabular Synthesis**: DP-SGD, PATE, OVAE – The paper notes these often amplify imbalance or lack formal guarantees.
- **Fair Generation**: FairGAN, DECAF, TabFairGAN – Limited by binary sensitive attributes, whereas this work uses MI regularization.
- **Privacy + Fairness Intersection**: PF-WGAN, PreFair – This work goes further by adding class balance and providing a unified theoretical foundation.
- **BNP Foundations**: Dirichlet Processes (Ferguson 1973), Finite Approximation (Ishwaran-Zarepour 2002), and BNPL (Fong 2019).
- **Insight**: When joint "Privacy + Fairness + Balance" is required, rather than patching architectures, it is more effective to integrate constraints into the resampling and regularization stages using Bayesian Nonparametrics.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Innovative use of DP resampling for global DP and BNP mutual information for fairness.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers standard datasets and ablation studies, though the datasets themselves are aging and lack massive-scale SOTA comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Solid derivation of mechanisms and theorems, though notation-heavy.
- **Value**: ⭐⭐⭐⭐ Provides a rare "all-in-one" solution for trustworthy tabular synthesis with practical potential for regulated or data-scarce scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](../../ICML2026/ai_safety/differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[ICLR 2026\] Private Rate-Constrained Optimization with Applications to Fair Learning](private_rate-constrained_optimization_with_applications_to_fair_learning.md)
- [\[ICLR 2026\] A Fair Bayesian Inference through Matched Gibbs Posterior](a_fair_bayesian_inference_through_matched_gibbs_posterior.md)
- [\[CVPR 2026\] Image-based Outlier Synthesis With Training Data](../../CVPR2026/ai_safety/image-based_outlier_synthesis_with_training_data.md)
- [\[ICLR 2026\] MUSE: Model-Agnostic Tabular Watermarking via Multi-Sample Selection](muse_model-agnostic_tabular_watermarking_via_multi-sample_selection.md)

</div>

<!-- RELATED:END -->
