---
title: >-
  [Paper Note] Price of Quality: Sufficient Conditions for Sparse Recovery using Mixed-Quality Data
description: >-
  [ICLR 2026][learning_theory][LASSO] This work investigates the sufficient conditions for the sample size required for sparse signal support recovery when observations originate from two heteroscedastic noise sources ("small amount of high-quality + large amount of low-quality" data). It proposes a quantitative metric, "Price of Quality" $\gamma$ (represe
tags:
  - ICLR 2026
  - learning_theory
  - LASSO
date: 2026-05-08
content_hash: 20c7ab9f9361774f
---
# Price of Quality: Sufficient Conditions for Sparse Recovery using Mixed-Quality Data

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1PIfB5w05x](https://openreview.net/forum?id=1PIfB5w05x)  
**Code**: To be confirmed  
**Area**: Learning Theory / High-dimensional Statistics / Compressed Sensing  
**Keywords**: Sparse Recovery, Mixed-Quality Data, Heteroscedastic Noise, LASSO, Information-Theoretic Threshold

## TL;DR
This work investigates the sufficient conditions for the sample size required for sparse signal support recovery when observations originate from two heteroscedastic noise sources ("small amount of high-quality + large amount of low-quality" data). It proposes a quantitative metric, "Price of Quality" $\gamma$ (representing how many low-quality samples a single high-quality sample is worth), and reveals a counter-intuitive contrast: information-theoretic thresholds change sensitively with the data quality structure, whereas the LASSO algorithmic threshold depends only on the average noise and is surprisingly robust to data heterogeneity.

## Background & Motivation

**Background**: Sparse recovery is a core problem in high-dimensional statistics and compressed sensing. Given an $s$-sparse high-dimensional signal $\beta^\star \in \mathbb{R}^p$ known a priori, the goal is to recover the support set $S^\star = \{i: \beta^\star_i \neq 0\}$ through random linear projections $Y = X\beta^\star + Z$ corrupted by Gaussian noise. Classical theory has characterized two phase transition thresholds: the information-theoretic threshold $n_{\mathrm{INF}} = 2s\log(p/s)/\log s$ (below which no method can recover the signal) and the algorithmic threshold $n_{\mathrm{ALG}} = 2s\log(p-s)+s+1$ (above which LASSO can recover the signal in polynomial time). However, these results are mostly established under the **homoscedastic** assumption: the noise variance of all observations is a constant $\sigma^2$.

**Limitations of Prior Work**: Real-world data is often of "mixed quality." A small batch of high-quality observations (e.g., expert labels, calibrated sensors) has a small noise variance $\sigma_1^2$, while a large batch of low-quality observations (e.g., weak labels from LLMs, web-crawled corpora) has a larger noise variance $\sigma_2^2 > \sigma_1^2$. The standard technique in homoscedastic theory—normalizing variance by dividing the entire model by $\sigma$—fails under heteroscedasticity because no single $\sigma$ can normalize both data blocks simultaneously. Thus, a fundamental question remains unanswered: under this heteroscedastic structure, exactly how many high-quality samples $n_1$ and low-quality samples $n_2$ are required to recover the signal?

**Key Challenge**: High-quality data is expensive and low-quality data is cheap; whether they can substitute each other—and at what rate—depends on a critical piece of information: whether the decoder knows the noise variance of each sample. The authors distinguish two settings: the **agnostic setting**, where the decoder lacks source information and treats all observations as coming from a homogeneous model (common in web-scale corpora or citizen science data); and the **informed setting**, where the decoder knows the noise variance and can perform reweighting (common in multi-center clinical trials or medical imaging with confidence scores).

**Goal**: To provide sufficient conditions for $(n_1, n_2)$ for both information-theoretic and algorithmic recovery under heteroscedastic noise, and to quantify the substitution rate of the two types of data.

**Key Insight**: The authors observe that sufficient conditions can be expressed in a linear trade-off form $\alpha_1 n_1 + \alpha_2 n_2 > n^\star$. The ratio of coefficients $\alpha_1/\alpha_2$ naturally defines "how many low-quality samples one high-quality sample is worth."

**Core Idea**: Formalize the substitution rate as the **Price of Quality** $\gamma := \alpha_1/\alpha_2$, and systematically characterize its behavior under both agnostic and informed settings across different Signal-to-Noise Ratio (SNR) regimes. Simultaneously, generalize classical LASSO threshold results to the heteroscedastic agnostic setting.

## Method

This paper is a purely theoretical work. The core consists of sufficient (and necessary) conditions provided by three theorems and an asymptotic analysis of the "Price of Quality" across SNR regimes. The logic follows: establish a model for heteroscedastic sparse recovery, derive thresholds for IT and algorithmic lines under both agnostic/informed settings, and finally contrast these thresholds to show the "IT sensitivity vs. algorithmic robustness."

### Overall Architecture

The problem model is fixed as follows: measurement matrix elements $X_{ij} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$, noise $Z = \Sigma W$, where $W \sim \mathcal{N}(0, I_n)$, and $\Sigma = \mathrm{diag}(\sigma_1 I_{n_1}, \sigma_2 I_{n_2})$ is a block-diagonal matrix. Crucially, the authors **do not assume** $\sigma_1^2, \sigma_2^2$ are constants, allowing them to scale arbitrarily with $p, s$. Thus, in addition to the overall $\mathrm{SNR} = s/\sigma_{\mathrm{avg}}^2$, high-quality $\mathrm{SNR}_1 = s/\sigma_1^2$ and low-quality $\mathrm{SNR}_2 = s/\sigma_2^2$ are defined to delineate three regimes: high SNR ($\sigma_2^2 = o(s)$), low $\mathrm{SNR}_2$-high $\mathrm{SNR}_1$, and low SNR ($\sigma_1^2 = \omega(s)$). Here $\sigma_{\mathrm{avg}}^2 := (n_1\sigma_1^2 + n_2\sigma_2^2)/n$ is the sample-weighted average noise variance.

The analysis unfolds along two orthogonal axes: **Information-Theoretic (IT) vs. Algorithmic** and **Agnostic vs. Informed**. Three main theorems fill this 2×2 grid: Theorem 1 (IT + agnostic), Theorem 2 (IT + informed), and Theorem 3 (Algorithmic + agnostic, i.e., LASSO). Each sufficient condition is reduced to the form "a linear combination of sample sizes $> n^\star$," from which $\gamma$ is derived and asymptotically expanded.

### Key Designs

**1. Price of Quality $\gamma$: Formulating the Substitution Rate as a Coefficient Ratio**

This is the central conceptual axis. All sufficient conditions take the linear form $\alpha_1 n_1 + \alpha_2 n_2 \geq (1+\varepsilon)n^\star$. If $(n_1, n_2)$ satisfies the condition, then $(n_1 - 1,\, n_2 + \alpha_1/\alpha_2)$ also satisfies it. Thus, the definition is:

$$\gamma\left(s, \sigma_1^2, \sigma_2^2\right) := \frac{\alpha_1}{\alpha_2}.$$

This represents how many low-quality samples are equivalent to one high-quality sample while maintaining recovery guarantees.

**2. Uniformly Bounded $\gamma$ in Agnostic Setting: One High-Quality Sample Worth at Most Two Low-Quality Samples**

In the agnostic setting, the decoder uses unweighted Least Squares $\hat\beta \in \arg\min_{\beta \in \mathcal{B}_{p,s}} \|Y - X\beta\|_2^2$. Theorem 1 provides the sufficient condition:

$$n_1 \log\!\left(1 + \frac{\delta(2\sigma_2^2 - \sigma_1^2)s}{2\sigma_2^4}\right) + n_2 \log\!\left(1 + \frac{\delta s}{2\sigma_2^2}\right) \geq (1+\varepsilon)\, n^\star,$$

where $\delta$ is the allowed support error proportion. The price of quality is:

$$\gamma = \frac{\log\!\big(1 + \delta(2\sigma_2^2 - \sigma_1^2)s/(2\sigma_2^4)\big)}{\log\!\big(1 + \delta s/(2\sigma_2^2)\big)} > 1.$$

The key conclusion is that $\gamma$ is **uniformly bounded**: in the high $\mathrm{SNR}_2$ regime, $\gamma \simeq 1$; in the low $\mathrm{SNR}_2$ regime, $\gamma \simeq 2 - \sigma_1^2/\sigma_2^2 < 2$. Even as the noise variance ratio becomes extreme, a high-quality sample never substitutes for more than two low-quality samples because the decoder cannot exploit the low variance of high-quality data.

**3. Unbounded $\gamma$ in Informed Setting: Variance Awareness Skyrockets Value**

In the informed setting, the decoder uses the variance-rescaled MLE $\hat\beta_{\mathrm{MLE}} \in \arg\min_{\beta} \|\Sigma^{-1}(Y - X\beta)\|_2^2$. Theorem 2's sufficient condition becomes:

$$n_1 \log\!\left(1 + \frac{\delta s}{2\sigma_1^2}\right) + n_2 \log\!\left(1 + \frac{\delta s}{2\sigma_2^2}\right) \geq (1+\varepsilon)\, n^\star,$$

with the price of quality:

$$\gamma = \frac{\log\!\big(1 + \delta s/(2\sigma_1^2)\big)}{\log\!\big(1 + \delta s/(2\sigma_2^2)\big)}.$$

Unlike the agnostic case, $\gamma$ **can be arbitrarily large**. In the low SNR regime, $\gamma \simeq \sigma_2^2/\sigma_1^2$. High-quality samples become infinitely precious as the noise ratio increases because the rescaled MLE optimally utilizes the high-quality data.

**4. LASSO Algorithmic Threshold depends only on Average Noise**

Theorem 3 studies whether polynomial-time LASSO can recover the signed support. The conclusion is surprising: in the heteroscedastic agnostic setting, the LASSO phase transition threshold is **identical to the homoscedastic case**. If $n > (1+\varepsilon)n_{\mathrm{ALG}}$ and the regularization parameter $\lambda_p$ satisfies:

$$\frac{n\lambda_p^2}{\sigma_{\mathrm{avg}}^2 \log(p-s)} \to +\infty, \qquad \frac{1}{\rho}\left[\lambda_p\sqrt{s} + \sqrt{\frac{\sigma_{\mathrm{avg}}^2 \log s}{n}}\right] \to 0,$$

recovery succeeds. The condition $n_{\mathrm{ALG}} = 2s\log(p-s)+s+1$ **does not depend** on specific $\sigma_1^2, \sigma_2^2$, and only involves $\sigma_{\mathrm{avg}}^2$. For LASSO, high and low-quality samples contribute equally in terms of the number of samples needed, meaning the Price of Quality is identically 1.

### Loss & Training
The three estimators correspond to three loss functions:
- **Agnostic IT**: Unweighted Least Squares $\|Y-X\beta\|_2^2$.
- **Informed IT**: Variance-rescaled MLE $\|\Sigma^{-1}(Y-X\beta)\|_2^2$.
- **Algorithmic Agnostic**: $\ell_1$-regularized LASSO, with $\lambda_p$ selected based on Proposition 4.1.

## Key Experimental Results

This is a theoretical paper without empirical experiments. "Key results" are presented via asymptotic expressions of thresholds and the Price of Quality.

### Main Results Comparison

| Setting | Estimator | Sufficient Condition (Linear combination $\geq (1+\varepsilon)n^\star$) | Price of Quality $\gamma$ |
|------|--------|------------------------------------------------|------------------|
| IT · Agnostic (Thm 1) | Unweighted LS | $n_1\log\!\big(1+\frac{\delta(2\sigma_2^2-\sigma_1^2)s}{2\sigma_2^4}\big)+n_2\log\!\big(1+\frac{\delta s}{2\sigma_2^2}\big)$ | Bounded, $1 < \gamma < 2$ |
| IT · Informed (Thm 2) | Rescaled MLE | $n_1\log\!\big(1+\frac{\delta s}{2\sigma_1^2}\big)+n_2\log\!\big(1+\frac{\delta s}{2\sigma_2^2}\big)$ | Unbounded, $\to +\infty$ |
| Algorithmic · Agnostic (Thm 3) | LASSO | $n > n_{\mathrm{ALG}} = 2s\log(p-s)+s+1$ (No $\sigma_i^2$ dependence) | Constant $=1$ |

### Price of Quality Asymptotics by SNR Regime

| SNR Regime | Agnostic $\gamma$ (Thm 1) | Informed $\gamma$ (Thm 2) |
|----------|---------------------------|----------------------------|
| High SNR ($\sigma_2^2 = o(s)$) | $\simeq 1$ | $\simeq \log(s/\sigma_1^2)/\log(s/\sigma_2^2)$ |
| Low $\mathrm{SNR}_2$-High $\mathrm{SNR}_1$ | (Intermediate) | $\Theta(\log\mathrm{SNR}_1/\log\mathrm{SNR}_2) \to +\infty$ |
| Low SNR ($\sigma_1^2 = \omega(s)$) | $\simeq 2 - \sigma_1^2/\sigma_2^2 < 2$ | $\simeq \sigma_2^2/\sigma_1^2$ (Unbounded) |

### Key Findings
- **Contrast between IT sensitivity and Algorithmic robustness**: The substitution rate $\gamma$ varies wildly for information-theoretic thresholds based on whether the source is known, but LASSO's threshold is completely "blind" to the heterogeneity beyond the average noise level $\sigma_{\mathrm{avg}}^2$.
- **"Knowledge" is the switch for High-Quality Premium**: High-quality data's value is capped at 2x if the variance is unknown but can be infinitely high if known and used for reweighting.
- The agnostic IT condition is sufficient but **not guaranteed to be tight** due to Chernoff relaxation, whereas the informed IT and LASSO conditions are **sharp** under the Gaussian design.

## Highlights & Insights
- **Quantifying "Data Quality" as a Scalar $\gamma$**: Transforms a vague intuition into a closed-form, computable metric across different SNR regimes.
- **Robustness Inversion**: Contrary to the common belief that algorithmic thresholds are more fragile, here the algorithmic threshold (LASSO) is more robust to data heterogeneity than the IT threshold.
- **Reweighting is Power**: Rescaling by $\Sigma^{-1}$ is the mechanism that pushes high-quality data value from "bounded" to "unbounded," suggesting significant gains in modeling noise variance for hybrid training.

## Limitations & Future Work
- **Agnostic IT Bound Lack of Tightness**: The sufficient condition might be loose due to the三次方程 (cubic equation) relaxation; exact characterization remains open.
- **Missing Informed LASSO Analysis**: Analyzing the phase transition for rescaled LASSO is difficult due to the interaction between $\Sigma^{-1}$ and $X$, which breaks the Wishart structure.
- **Idealized Assumptions**: The work relies on Gaussian designs, exact sparsity, and additive Gaussian noise from only two sources.
- **Binary Signal Assumption**: IT analysis assumes $\beta^\star \in \{0,1\}^p$, which simplifies calculation but limits direct applicability to general amplitude signals.

## Related Work & Insights
- **vs. Wainwright (2009)**: Generalizes classical LASSO phase transitions from homoscedastic to heteroscedastic agnostic settings using Gram–Schmidt/QR decomposition and Haar measure properties.
- **vs. Gamarnik & Zadik (2022)**: Builds upon their work on IT/algorithmic thresholds in the homoscedastic case, extending the MLE analysis to heteroscedasticity.
- **vs. Weak Supervision Literature**: While most work in weak supervision is empirical, this work provides a rigorous theoretical framework for substitution rates between "strong" and "weak" labels in sparse recovery.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First characterizing sufficient conditions for mixed-quality sparse recovery and revealing the robustness contrast.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical, covering comprehensive SNR regimes (though some bounds are not tight).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, well-organized 2×2 framework.
- Value: ⭐⭐⭐⭐ Solid theoretical insights with clear practical implications for reweighting based on uncertainty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis](robustness_of_probabilistic_models_to_low-quality_data_a_multi-perspective_analy.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] Why Less is More (Sometimes): A Theory of Data Curation](why_less_is_more_sometimes_a_theory_of_data_curation.md)
- [\[ICLR 2026\] SVD Provably Denoises Nearest Neighbor Data](svd_provably_denoises_nearest_neighbor_data.md)
- [\[ICLR 2026\] Residual Feature Integration is Sufficient to Prevent Negative Transfer](residual_feature_integration_is_sufficient_to_prevent_negative_transfer.md)

</div>

<!-- RELATED:END -->
