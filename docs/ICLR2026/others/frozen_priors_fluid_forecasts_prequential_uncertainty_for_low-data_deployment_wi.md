---
title: >-
  [Paper Note] Frozen Priors, Fluid Forecasts: Prequential Uncertainty for Low-Data Deployment with Pretrained Generative Models
description: >-
  [ICLR 2026][Others][Paper Note] Targeting low-data deployment scenarios where "only dozens of real samples are available at launch," this paper proposes a "forecast-first" uncertainty quantification (UQ) framework. It uses a **unique Dirichlet mixture schedule** to fuse the empirical distribution with a frozen pretrained generative model into a time-
tags:
  - ICLR 2026
  - Others
date: 2026-05-08
content_hash: 710a401ff43ac7f7
---
# Frozen Priors, Fluid Forecasts: Prequential Uncertainty for Low-Data Deployment with Pretrained Generative Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3FCHmUPmhe](https://openreview.net/forum?id=3FCHmUPmhe)  
**Code**: [https://github.com/Aalto-QuML/Prequential](https://github.com/Aalto-QuML/Prequential)  
**Area**: Probabilistic Methods / Uncertainty Quantification (UQ)  
**Keywords**: Uncertainty Quantification, Prequential Inference, Martingale Posterior, Frozen Generative Models, Low-data Deployment, Dirichlet Contraction

## TL;DR
Targeting low-data deployment scenarios where "only dozens of real samples are available at launch," this paper proposes a "forecast-first" uncertainty quantification (UQ) framework. It uses a **unique Dirichlet mixture schedule** to fuse the empirical distribution with a frozen pretrained generative model into a time-consistent (martingale) forecast stream. Calibrated intervals for long-term values of operational metrics are then provided via **Martingale Posterior (MP) resampling**—all without retraining or density evaluation. On GPT-2 / CIFAR-10 / SVHN, it achieves approximately 90% coverage with only 20 samples (while bootstrap achieves only 37%).

## Background & Motivation
**Background**: Machine learning systems often encounter very few real samples upon deployment (staged rollouts, safety gating, and rate limiting intentionally keep the initial sample size $n$ low). However, operators need to answer early questions like "What percentage will trigger alarms in the long run?" or "What is the long-term mean of this metric?" A common practice is to "borrow stability" from a generative model $Q_\phi$ (flow models, diffusion, autoregressive LMs, GANs) pretrained on similar data.

**Limitations of Prior Work**: Standard UQ methods fail under small $n$. Frequentist intervals target unknown population parameters and ignore the actual deployment forecasting rules, becoming overly conservative at small sample sizes. Bayesian posteriors typically require continuous refitting, which contradicts "frozen deployment." Conformal prediction provides per-sample coverage rather than uncertainty regarding long-term rates under fixed rules. Direct use of model likelihood is unreliable under distribution shift (flow models may even assign higher likelihood to OOD samples).

**Key Challenge**: Operators care about **long-term operational metrics under deployment rules** $\theta_\infty$, not the population parameters of the true distribution $\theta(F^\star)$. Almost all classical UQ methods estimate the latter. How to fuse "synthetic data $Q_\phi$" and "sequentially arriving real data $F^\star$" in a mathematically rigorous and time-consistent manner remains an unaddressed question.

**Goal**: To provide an out-of-the-box UQ tool for frozen generators + linear metrics (mean, tail probability, NLL) that requires no retraining or density evaluation, while also answering "when to retrain."

**Core Idea**: **[Forecast-first + Martingale Consistency]** Uncertainty is attributed to "unseen future outcomes" rather than distribution parameters. The empirical distribution and the frozen model are blended via $P_i=(1-\lambda_i)\hat F_i+\lambda_i Q_\phi$. It is proved that only one specific schedule, $\lambda_i=\alpha/(i+\alpha)$ (Dirichlet/Pólya pseudo-counts), ensures that the forecast sequence $\theta(P_i)$ for any bounded score is a martingale. This causes initial contraction toward the model and automatic fading back to empirical behavior as data accumulates. **Note**: The authors explicitly state this is a "surrogate forecasting system"; the theory holds for the surrogate process $\theta(P_i)$ and its operational target $\theta_\infty$, without claiming that real data follows this forecast.

## Method

### Overall Architecture
The framework is a unidirectional "forecast-first" pipeline: First, the empirical distribution and frozen generator are fused into a time-consistent forecast stream $P_i$ using the unique Dirichlet mixture rule. Next, the only hyperparameter—pseudo-count $\alpha$—is determined using a small-sample minimax criterion. Then, future forecasts are simulated via Martingale Posterior (forward resampling or Dirichlet-mean closed-form shortcut), tracking only the score values to obtain calibrated intervals for the long-term metric $\theta_\infty$. Finally, finite-time drift bounds provide decisions on "when to stop simulation" and "when to retrain."

```mermaid
flowchart LR
    A[Few real samples Y_1..n] --> B[Empirical distribution F_hat_i]
    Q[Frozen generator Q_phi] --> C
    B --> C[Dirichlet mixture<br/>P_i=(1-λ_i)F_hat+λ_i Q_phi<br/>λ_i=α/(i+α)]
    C --> D[Minimax selection α<br/>α*=σ²/Δ²]
    D --> E[Martingale Posterior MP resampling<br/>Simulate future tracking only h]
    E --> F[θ_∞ Calibrated interval]
    F --> G[Finite-time drift bound<br/>Decide horizon M / When to retrain]
```

### Key Designs

**1. Dirichlet mixture is the unique time-consistent forecast rule: Translating "today's forecast is the best estimate of tomorrow's forecast" into a mathematical constraint.** The foundation is Theorem 1: among all "empirical-model affine mixtures with predictable weights," if scalar prequential consistency $\mathbb{E}\big[\int h\,dP_i \mid \mathcal F_{i-1}\big]=\int h\,dP_{i-1}$ is required (i.e., the forecast is "stationary in expectation" for every bounded score $h$), the weights **must** take the Dirichlet/Pólya pseudo-count schedule $\lambda_i=\alpha/(i+\alpha)$. This step solidifies the intuition (borrowing stability from $Q_\phi$ early, fading to empirical data later) as the unique solution. The immediate reward is that under this rule, the process $\theta(P_i)=\int h\,dP_i$ for any bounded (or $L^2$) score is a martingale, converging almost surely and in $L^2$ to the long-term value $\theta_\infty$ by Doob’s convergence theorem. The conditional mean is exactly the familiar contraction point $\mathbb{E}[\theta_\infty\mid\mathcal F_n]=\frac{n}{n+\alpha}\int h\,d\hat F_n+\frac{\alpha}{n+\alpha}\int h\,dQ_\phi$. Martingality is broken if $\lambda$ is fixed without decay or if $\phi$ is updated online.

**2. Small-sample minimax yields closed-form $\alpha$: Making the bias-variance tradeoff explicit as a single knob.** Under small $n$, error stems from two sources: sampling noise of the empirical plug-in $\theta(\hat F_n)$ and the potential bias of the frozen model $Q_\phi$ relative to the true functional. The authors define an ambiguity set $\mathcal G(\sigma^2,\Delta)=\{F^\star:\mathrm{Var}_{F^\star}[h]\le\sigma^2,\ |\theta(Q_\phi)-\theta(F^\star)|\le\Delta\}$ and perform worst-case risk minimization for the shrinkage estimator $\hat\theta_\lambda=(1-\lambda)\theta(\hat F_n)+\lambda\theta(Q_\phi)$. Theorem 2 provides the exact minimax weight $\lambda^\star=\frac{a}{a+\Delta^2}$ (where $a=\sigma^2/n$), which corresponds exactly to the pseudo-count $\alpha^\star=\sigma^2/\Delta^2$ (independent of $n$). In practice, since $\sigma^2$ and $\Delta$ are unknown, the sample variance $\hat\sigma^2$ is used, and bias is upper-bounded by $\hat\Delta=|\theta(Q_\phi)-\bar z|+t_n$ using an empirical Bernstein safety margin $t_n$, resulting in a data-driven $\hat\alpha=\mathrm{clip}(\hat\sigma^2/\hat\Delta^2;\alpha_{\min},\alpha_{\max})$. Proposition 3 proves this estimated weight approximates the oracle minimax risk with high probability, differing only by second-order terms.

**3. Martingale Posterior resampling: Likelihood-free, score-only lightweight UQ.** The uncertainty of the long-term metric is characterized by the Martingale Posterior $\Pi_{\mathrm{MP}}(\cdot\mid\mathcal F_n):=\mathrm{Law}(\theta_\infty\mid\mathcal F_n)$, approximated via forward prequential resampling (Algorithm 1): each replica maintains a score pool, running sum, and count. It uses Bernoulli trials with $\lambda=\alpha/(i-1+\alpha)$ to decide whether to "sample a new score from model $Q_\phi$" or "resample with replacement from the historical score pool." After advancing to horizon $M$, $\mathrm{sum}/\mathrm{count}$ serves as one sample of $\theta_\infty$. This process stores no inputs, calculates no densities, and is GPU-friendly. For **linear metrics** ($\theta(F)=\int h\,dF$), a Dirichlet-mean closed-form shortcut exists (Remark 5): directly sample $(w_0,\dots,w_n)\sim\mathrm{Dirichlet}(\alpha,1,\dots,1)$, set $Z_0\sim H_0$, and $\theta^{(b)}=w_0Z_0+\sum_i w_i z_i$. This is identically distributed to forward simulation but significantly reduces computation.

**4. Finite-time drift bounds: Turning "where to stop simulation" and "when to retrain" into auditable rules.** Theorem 6 (Freedman-type) provides an anytime deviation bound for bounded scores $\|h\|_\infty\le H$ under the Dirichlet schedule: $\sup_{t\ge n}|\theta(P_t)-\theta(P_n)|\le H\sqrt{\frac{2\log(2/\delta)}{n+\alpha}}+\frac{2H}{3(n+\alpha+1)}\log\frac{2}{\delta}$ holds with probability $1-\delta$. Drift is controlled only by the score magnitude $H$ and effective sample size $n+\alpha$, contracting at $O((n+\alpha)^{-1/2})$. The MP simulation horizon $M$ is chosen such that the "theoretical drift bound < MP Monte Carlo error" for a safe stop. The same logic supports retraining decisions (Section 7 / Proposition 7): the minimax risk $R^\star(a,\Delta)=\frac{a\Delta^2}{a+\Delta^2}$ estimates the per-use gain of reducing mismatch from $\Delta$ to $\Delta^+$ via retraining. This gain multiplied by future uses $H$ is compared against retraining cost $C_{rt}$. An audit-friendly trigger is activated when $H\sum_k w_k(R^\star(a,\Delta_k)-R^\star(a,\Delta_k^+))\ge C_{rt}$.

## Key Experimental Results

### Main Results
Three settings were evaluated, all using frozen generators + linear metrics. Baselines included non-parametric bootstrap (NPB), Bayesian bootstrap (BB), and Jackknife (JK). Ours included DWS (Dirichlet Weighted Shrinkage with minimax $\alpha$) and MP (Martingale Posterior, using the Dirichlet-mean shortcut for linear metrics).

| Setting | Data/Model | Score $h$ | Target | Key Result |
|------|-----------|----------|------|----------|
| Language (ID) | GPT-2 (117M) / WikiText-2 | Per-token NLL | $\mathbb{E}[h]$ | ~90% coverage at $n_0=20$; NPB only achieves ~37% |
| Vision (ID) | CIFAR-10 Pretrained Generator | CLIP-rarity | $\mathbb{E}[h]$ | Methods converge; close to nominal coverage |
| Vision (OOD) | SVHN (Strong Shift) | CLIP-rarity | $\mathbb{E}[h]$ | DWS has best calibration at small $n$, near 90%; others under-cover |
| Toy | Two Moons | Alarm rate / Mean score | — | Validates contraction and fading behavior |

Coverage is defined as the proportion of runs where the 90% predictive interval (targeting $\theta_\infty$ under deployment rules) contains the large-sample reference from an independent truth pool. On GPT-2, DWS remained closest to the nominal value in the small $n$ range (approx. 0.90 at $n_0=20$, stable to $n_0=100$), whereas NPB/JK significantly under-covered for $n_0\le50$.

### Ablation Study

| Comparison | Observation | Conclusion |
|------|-----------|----------|
| Remove minimax $\alpha$ (plain Dirichlet-mean) | Under-coverage at very small $n$ | Minimax $\alpha$ is key for small-sample calibration |
| $n_0=50$ vs $n_0=3000$ (MP draws) | At 50, mean is between empirical and model; band is wide. At 3000, fades to empirical; band tightens | Automatically narrows while maintaining calibration as $n_0+\alpha$ grows |
| Across data/sample sizes | DWS/MP intervals narrow steadily with $n_0$ | Becomes both sharper and remains calibrated |

### Key Findings
Three reasons for success at small sample sizes: (i) **Coherent pseudo-counts**—the unique predictable affine mixture makes $\theta(P_i)$ a martingale, stabilizing early phases while ensuring fading; (ii) **Minimax $\alpha$**—a single data-driven knob balances sampling variance and model mismatch, suppressing the under-coverage of bootstrap at minimal $n$; (iii) **The Right Target**—MP/Dirichlet-mean quantifies the uncertainty of the operational limit $\theta_\infty$ under deployment rules, which is the quantity operators truly care about.

## Highlights & Insights
- **Clarification of "Surrogate Forecasting"**: The authors repeatedly emphasize that the theory holds only for $\theta(P_i)$ and $\theta_\infty$ (the surrogate process) and does not claim to approximate the true population parameter $\theta(F^\star)$. This conceptual shift from "population parameter" to "operational target" is why it outperforms frequentist tools at small $n$.
- **Elegant Uniqueness Result**: Converting the simple intuition that "forecasts don't move on average" into the unique Dirichlet schedule (Theorem 1) ensures the mixture rule is necessitated by coherence rather than ad hoc tuning.
- **Single Knob, Closed-form Solution**: $\alpha^\star=\sigma^2/\Delta^2$ is independent of $n$ and adaptive to data, making engineering implementation extremely simple ("single pass over observables").
- **Likelihood-free + GPU Friendly**: It only requires the ability to sample from $Q_\phi$ and evaluate $h$. It treats flows, diffusion, autoregressive, GANs, and EBMs equally and handles both domain and semantic shifts.
- **Unified Retraining Decision**: Using the minimax risk difference multiplied by usage frequency vs. cost provides a stakeholders-friendly, auditable trigger.

## Limitations & Future Work
- **Surrogate vs. Reality**: The method explicitly waives guarantees for $\theta(F^\star)$. If an operator strictly requires population parameters (rather than long-term metrics under deployment rules), frequentist tools should be prioritized.
- **Dependency on Frozen Assumption**: Any online update to parameters, decoding, MCMC, or prompts breaks $\mathcal{F}_n$-measurability and coherence, invalidating the framework. This limits its use in continual learning or online adaptation systems.
- **Focus on Linear Metrics**: The closed-form Dirichlet-mean shortcut only holds for functionals linear in $F$. Nonlinear or path-dependent targets still require forward MP simulation, which is more computationally intensive.
- **Drift Bound and Bounded Scores**: The finite-time bounds and horizon selection assume $\|h\|_\infty\le H$. Unbounded scores (e.g., some heavy-tailed NLLs) require additional handling.
- **Not for Per-sample Coverage**: Conformal prediction remains the standard for per-example guarantees. Experimental scale was relatively small/medium (GPT-2 117M, CIFAR/SVHN); validation on larger models and industrial deployments is left for the future.

## Related Work & Insights
- **Prequential/Forecast-first Statistics**: The prequential perspective of Dawid (1984) and the review of Fortini & Petrone (2025), which place uncertainty on unobserved outcomes, form the philosophical foundation.
- **Martingale Posteriors**: Fong et al. (2023) proposed approximating $\mathrm{Law}(\theta_\infty\mid\mathcal F_n)$ via prequential resampling; this paper combines it with a frozen generator mixture rule and provides a closed-form shortcut.
- **Dirichlet/Pólya Sequences**: The pseudo-count structure from Blackwell & MacQueen (1973) and Ferguson (1973) is the mathematical source of the coherence uniqueness result.
- **Contrast with Classical UQ**: Conformal prediction (Vovk et al. 2005; Angelopoulos & Bates 2023) and the failure of flow model OOD likelihoods (Nalisnick et al. 2019; Ren et al. 2019; Kirichenko et al. 2020) outline the problem boundaries addressed here.
- **Insight**: When the deployment goal is "what happens long-term under fixed rules," it is more efficient to perform rigorous UQ on the "prediction process induced by the deployment rules" than to struggle with estimating the true distribution. This shift from "truth" to "operational surrogate" is transferable to various low-data launch scenarios like recommendation, risk control, and safety gating.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Integrates the uniqueness of "coherence ⇔ Dirichlet schedule," closed-form minimax $\alpha$ for small samples, and likelihood-free Martingale Posterior UQ into a self-consistent framework. The shift to "operational surrogate targets" is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers language/vision ID/OOD and toy problems with comparisons to various bootstrap baselines. Small $n$ coverage advantage is significant, but model scales are small/medium, and industrial-scale validation is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with helpful "At a glance / Why this matters / What breaks" guides in each section. Theories and intuitions are well-balanced; the high density of formulas may be challenging for those without a probabilistic background.
- **Value**: ⭐⭐⭐⭐ Directly addresses the real-world pain point of "deploying with dozens of samples." It is out-of-the-box, requires no retraining, and provides a retraining decider, offering high value for UQ practices in low-data deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exploring State-Space Models for Data-Specific Neural Representations](exploring_state-space_models_for_data-specific_neural_representations.md)
- [\[NeurIPS 2025\] Improving Forecasts of Suicide Attempts for Patients with Little Data](../../NeurIPS2025/others/improving_forecasts_of_suicide_attempts_for_patients_with_little_data.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](measuring_uncertainty_calibration.md)
- [\[ICLR 2026\] GoR: A Unified and Extensible Generative Framework for Ordinal Regression](gor_a_unified_and_extensible_generative_framework_for_ordinal_regression.md)
- [\[ICML 2026\] Spatial Priors via Space Filling Curves for Small and Limited Data Vision Transformers](../../ICML2026/others/spatial_priors_via_space_filling_curves_for_small_and_limited_data_vision_transf.md)

</div>

<!-- RELATED:END -->
