---
title: >-
  [Paper Note] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression
description: >-
  [NeurIPS 2025][transfer learning] This paper proposes a two-step Transfer MNI (TM) method that enhances generalization of benign overfitting in overparameterized high-dimensional linear regression via a "preserve target…
tags:
  - "NeurIPS 2025"
  - "transfer learning"
  - "benign overfitting"
  - "high-dimensional linear regression"
  - "minimum-norm interpolation"
  - "covariate shift"
  - "model shift"
date: 2026-05-08
content_hash: 69a21161e152a93a
---

# Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.15337](https://arxiv.org/abs/2510.15337)  
**Authors**: Yeichan Kim (Yonsei University), Ilmun Kim (KAIST), Seyoung Park (Yonsei University)
**Code**: Not released  
**Area**: Others
**Keywords**: transfer learning, benign overfitting, high-dimensional linear regression, minimum-norm interpolation, covariate shift, model shift

## TL;DR

This paper proposes a two-step Transfer MNI (TM) method that enhances generalization of benign overfitting in overparameterized high-dimensional linear regression via a "preserve target signal + transfer source knowledge in the null space" mechanism. Non-asymptotic excess risk bounds are derived under both model shift and covariate shift, and a "free lunch" covariate shift regime is identified.

## Background & Motivation

### State of the Field
Transfer learning improves target task performance by leveraging source task knowledge, with numerous successes in high-dimensional regression (e.g., LASSO and its variants, GLMs, nonparametric regression). Meanwhile, overparameterized models such as the minimum $\ell_2$-norm interpolator (MNI) exhibit surprisingly benign overfitting in the $n < p$ regime—achieving zero training error while generalizing well.

### Limitations of Prior Work
- Existing transfer learning methods rely on **explicit regularization** (e.g., LASSO), whereas benign overfitting depends on **implicit regularization**; the intersection of these two regimes is largely unexplored.
- Mallinar et al. (2021) studied an OOD setting without using target samples for training; Wu et al. (2023) proposed an SGD-based approach but did not consider model shift and focused experiments on the underparameterized regime.
- Song et al. (2024) proposed pooled-MNI, but pooling multi-source data is extremely sensitive to distributional shift.
- Precise characterizations of positive transfer conditions, optimal source sample sizes, and maximum achievable improvement for MNI-based transfer learning are lacking.

### Core Problem
Can transfer learning further enhance the already strong out-of-sample generalization of overparameterized interpolators in high-dimensional linear regression?

## Method

### Problem Setup
Consider one target task and $Q$ source tasks in overparameterized linear regression ($n_q \leq p$):

$$\mathbf{y}^{(q)} = \mathbf{X}^{(q)} \boldsymbol{\beta}^{(q)} + \boldsymbol{\epsilon}^{(q)}, \quad q \in [Q]_0$$

where $q=0$ denotes the target task. Distributional shift is characterized by two components:
- **Model shift**: contrast vector $\boldsymbol{\delta}^{(q)} = \boldsymbol{\beta}^{(q)} - \boldsymbol{\beta}^{(0)}$
- **Covariate shift**: structural differences in covariance matrices $\boldsymbol{\Sigma}^{(q)}$ (simultaneous diagonalizability not required)

### Transfer MNI (TM): A Two-Step Procedure
1. **Pre-training**: Train source MNI $\hat{\boldsymbol{\beta}}_M^{(q)}$ on the $q$-th source dataset.
2. **Fine-tuning**: Subject to interpolating the target data, minimize the Euclidean distance to the source MNI:

$$\hat{\boldsymbol{\beta}}_{TM}^{(q)} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}_M^{(q)}\| : \mathbf{X}^{(0)}\boldsymbol{\beta} = \mathbf{y}^{(0)} \right\}$$

### "Preserve + Transfer" Mechanism
The TM estimator admits an elegant decomposition:

$$\hat{\boldsymbol{\beta}}_{TM}^{(q)} = \underbrace{\hat{\boldsymbol{\beta}}_M^{(0)}}_{\text{target signal}} + \underbrace{(\mathbf{I}_p - \mathbf{H}^{(0)}) \hat{\boldsymbol{\beta}}_M^{(q)}}_{\text{null-space knowledge transfer}}$$

- In the target row space $\mathcal{S}_0$, the signal learned by the target MNI is **preserved** (benign overfitting ensures predictive accuracy).
- Source information is **transferred** only in the target null space $\mathcal{S}_0^\perp$, where target samples carry no information.
- Core trade-off: knowledge transfer inevitably induces **variance inflation** $\mathcal{V}_\uparrow^{(q)} > 0$, but positive transfer is achieved if the bias reduction sufficiently offsets this inflation.

### Theorem 1: Exact Analysis under Isotropic Covariance
Under $\boldsymbol{\Sigma}^{(0)} = \boldsymbol{\Sigma}^{(q)} = \mathbf{I}_p$ with Gaussian design, the expected bias and variance admit exact closed-form expressions. Define the shift-to-signal ratio SSR and signal-to-noise ratio SNR:

$$\text{SSR}_q = \frac{\|\boldsymbol{\delta}^{(q)}\|^2}{\|\boldsymbol{\beta}^{(0)}\|^2}, \quad \text{SNR}_q = \frac{\|\boldsymbol{\beta}^{(0)}\|^2}{\sigma_q^2}$$

**Positive transfer condition** (Corollary 1, necessary and sufficient):

$$\text{SSR}_q < 1 \quad \text{and} \quad \text{SNR}_q(1 - \text{SSR}_q) > \frac{p}{p - (n_q + 1)}$$

**Optimal source sample size**: $n_q^* = p - 1 - \sqrt{p(p-1)/[\text{SNR}_q(1 - \text{SSR}_q)]}$; increasing source samples beyond $n_q^*$ strictly degrades transfer performance (strict concavity).

### Theorem 2: Non-Asymptotic Analysis under Benign Covariates
Under general sub-Gaussian covariates and an effective rank condition (Assumption 2), high-probability upper bounds on TM bias and variance inflation are established. The bias bound depends on the model contrast $\|\boldsymbol{\delta}^{(q)}\|^2$ and the effective rank ratio $r_0/n_q$; variance inflation is the product of a benign term $\Upsilon_q$ and $\psi_0$.

### Free Lunch Covariate Shift (Corollary 2)
When source covariance eigenvalues are uniformly scaled as $\boldsymbol{\Lambda}^{(q)} = \alpha \boldsymbol{\Lambda}^{(0)}$ ($\alpha > 1$):
- The bias upper bound remains unchanged (independent of $\alpha$).
- Variance inflation decreases by a factor of $\alpha$.

This constitutes a "free lunch": variance reduction is obtained at no additional bias cost. Only alignment of the top $\tau^*$ high-signal eigenvectors is required.

### WTM: Weighted Integration of Informative Sources
1. Use $K$-fold cross-validation ($K=5$) to detect informative sources: compare CV loss of each TM against that of the target MNI.
2. Assign adaptive weights proportional to the inverse CV loss, and form a weighted combination of all TM estimators detected as positively transferring.
3. WTM automatically filters out negative-transfer sources and aggregates multiple positive-transfer sources.

## Key Experimental Results

### Experiment 1: Benign Overfitting Setting (3 sources, $n_0=25$, $n_q=75$, $S=500$)

| Method | Model shift only, SSR=(0, 0.3, 0.6) | + Covariate shift, SSR=0.3 | + Free lunch, $\alpha=8$ |
|--------|--------------------------------------|---------------------------|--------------------------|
| Target MNI (baseline) | Slowly decreasing with $p$ | Slowly decreasing with $p$ | Slowly decreasing with $p$ |
| Pooled-MNI | Completely collapses (highly sensitive to shift) | Completely collapses | Completely collapses |
| TM (individual sources) | Outperforms baseline even at SSR=0.6 | TM(3) exhibits negative transfer | TM(3) recovers to baseline level |
| WTM (ensemble) | **Consistently optimal**, surpasses all individual TMs | Automatically filters negative-transfer sources; **consistently optimal** | All TMs converge faster; **WTM optimal** |

Key finding: Pooled-MNI fails catastrophically under distributional shift; WTM effectively identifies and excludes negative-transfer sources via CV detection.

### Experiment 2: Harmless Interpolation Setting (2 sources, isotropic $\mathbf{I}_p$, $S=10$, SSR=0.4)

| Method | No covariate shift | Free lunch, $\alpha=8$ (original $n_2^*$) | Free lunch, $\alpha=8$ (adjusted $n_2^*$) |
|--------|--------------------|------------------------------------------|------------------------------------------|
| Target MNI | Excess risk converges to 10 | Excess risk converges to 10 | Excess risk converges to 10 |
| Pooled-MNI | Eventually surpasses baseline but converges slowly | Performance improves | Performance improves |
| SGD transfer | Significantly lags behind TM | Lags behind TM | Lags behind TM |
| TM (optimal $n_2^*$) | **Significantly outperforms** baseline and competitors | Further improves | Uses $\text{SNR}_\alpha$ to transfer more samples; **best** |
| WTM | **Optimal** (surpasses individual TM) | Further improves | **Optimal** |

Key finding: Under free lunch covariate shift, the corrected $\text{SNR}_\alpha = \alpha\|\boldsymbol{\beta}^{(0)}\|^2/\sigma^2$ allows a larger optimal source sample size, yielding additional performance gains. All settings use 50 independent replications, plotting excess risk against $p \in \{300, 400, \ldots, 1000\}$.

## Highlights & Insights

- **"Preserve + Transfer" mechanism**: TM preserves MNI signal in the target row space and transfers source knowledge only in the null space—theoretically elegant and practically robust.
- **Necessary and sufficient positive transfer condition**: This work provides, for the first time, a precise characterization of positive transfer conditions for MNI-based transfer learning (SSR < 1 and sufficiently large SNR), along with a closed-form optimal source sample size.
- **Free lunch covariate shift**: Full alignment of source and target eigenvectors is unnecessary; aligning only the top $\tau^*$ high-signal directions suffices to obtain free variance reduction.
- **Adaptive ensemble WTM**: CV-based informative source detection combined with inverse-CV-loss weighting achieves consistent optimality across all settings.
- **Clear advantage over pooled-MNI**: The late-fusion architecture of TM is naturally robust to distributional shift, whereas pooled-MNI collapses under shift.

## Limitations & Future Work

- **No theoretical guarantees for WTM**: The consistency of CV-based informative source detection (i.e., high-probability guarantee that $\mathcal{I} = \hat{\mathcal{I}}$) remains an open problem.
- **Loose variance inflation upper bound**: In the non-simultaneously diagonalizable case, the upper bound contains a $(\lambda_p^{(q)})^{-1}$ term that may be large; finer covariance structure analysis is needed.
- **Restricted to linear regression**: The MNI analysis assumes a linear model; discussion of transfer learning for benign overfitting in deep networks is limited to a preliminary NTK-level treatment.
- **Limitation of isotropic analysis**: The exact results in Theorem 1 and Corollary 1 apply only to isotropic covariance with Gaussian design.
- **Multi-source correlations not exploited**: Correlations among source tasks are not modeled; WTM simply applies independent per-source training followed by weighted aggregation.
- **RKHS extension unverified**: The paper discusses a Transfer MNI extension to minimum-RKHS-norm interpolators, but theoretical analysis and experiments remain incomplete.

## Related Work & Insights

- **Bartlett et al. (2020)**: Established the non-asymptotic theoretical foundation for single-task MNI benign overfitting (effective rank condition); the present work inherits and extends this framework to transfer learning.
- **Song et al. (2024) Pooled-MNI**: Pools all data to train a single MNI, which is highly sensitive to distributional shift; the late-fusion architecture of TM is substantially more robust.
- **Mallinar et al. (2021)**: Studied an OOD setting without using target data for training; this work fine-tunes using target data.
- **Wu et al. (2023) SGD transfer**: Pre-training and fine-tuning via SGD, without considering model shift and focusing on the underparameterized regime; the present work targets the overparameterized setting.
- **Tahir et al. (2024)**: Quantifies model shift via cosine similarity but relies on explicit regularization; this work provides exact analysis within an implicit regularization framework.
- **Tian & Feng (2024)**: CV-based source transferability detection for GLMs (explicit regularization); this work extends the idea to the benign overfitting setting.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of MNI-based transfer learning; the decomposition mechanism is elegant and the free lunch covariate shift concept is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both benign overfitting and harmless interpolation settings, multiple shift combinations, and 50 independent replications.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical framework is complete and well-structured, progressing from isotropic to general covariance in a clear and coherent manner.
- Value: ⭐⭐⭐⭐ — Fills a theoretical gap in overparameterized transfer learning, though the restriction to linear models limits broader practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[NeurIPS 2025\] Computable Universal Online Learning](computable_universal_online_learning.md)
- [\[NeurIPS 2025\] Product Distribution Learning with Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)

</div>

<!-- RELATED:END -->
