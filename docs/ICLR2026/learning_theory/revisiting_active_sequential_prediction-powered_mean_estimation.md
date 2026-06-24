---
title: >-
  [Paper Note] Revisiting Active Sequential Prediction-Powered Mean Estimation
description: >-
  [ICLR 2026][Learning Theory][Active Sequential Mean Estimation] This paper revisits "active sequential prediction-powered mean estimation": it provides **non-asymptotic, anytime-valid**, data-dependent confidence bounds for an estimator that previously only had asymptotic guarantees. By employing Follow-the-Regularized-Leader (FTRL) online learning to select query probabilities per round, the theory and experiments jointly demonstrate that when query probabilities are blind t…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Active Statistical Inference"
  - "Prediction-Powered Inference (PPI)"
  - "Active Sequential Mean Estimation"
  - "Non-asymptotic Analysis"
  - "Freedman's Inequality"
  - "FTRL"
  - "Query Probability"
date: 2026-05-08
content_hash: 7ff20078c9e05d7a
---

# Revisiting Active Sequential Prediction-Powered Mean Estimation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Iw0tMeLed8](https://openreview.net/forum?id=Iw0tMeLed8)  
**Code**: Provided with supplementary materials (Jupyter notebook for reproducing Figure 1, etc.)  
**Area**: Learning Theory / Active Statistical Inference / Prediction-Powered Inference (PPI)  
**Keywords**: Active Sequential Mean Estimation, Non-asymptotic Analysis, Freedman's Inequality, FTRL, Query Probability

## TL;DR
This paper revisits "active sequential prediction-powered mean estimation": it provides **non-asymptotic, anytime-valid**, data-dependent confidence bounds for an estimator that previously only had asymptotic guarantees. By employing Follow-the-Regularized-Leader (FTRL) online learning to select query probabilities per round, the theory and experiments jointly demonstrate that when query probabilities are blind to **current** covariates, the optimal strategy is simply to have the query probability converge to the budget upper bound $T_b/T$, while sophisticated uncertainty weighting yields negligible additional gain.

## Background & Motivation

**Background**: Mean estimation is a classic inference task recently revitalized under the "Active Statistical Inference / Prediction-Powered Inference (PPI)" framework. Given a large set of unlabeled samples $x_1, \dots, x_T$, the goal is to estimate the label mean $\mu_y = \mathbb{E}[y_t]$. However, ground-truth labels $y_t$ are expensive with a limited budget (requiring $\mathbb{E}[T_{\text{lab}}] \le T_b$, where $T_b \ll T$). Meanwhile, a continuously updated black-box prediction model $f_t(\cdot)$ is available. Zrnic & Candes (2024) (hereafter ZC24) proposed deciding whether to query the truth at each round with probability $\pi_t(x_t)$: if queried, use the true label; otherwise, use the model prediction, resulting in the unbiased estimator:

$$\hat w = \frac{1}{T} \sum_{t=1}^{T} \Big( f_t(x_t) + \big( y_t - f_t(x_t) \big) \frac{\xi_t}{\pi_t(x_t)} \Big), \quad \xi_t \sim \text{Bernoulli}(\pi_t(x_t)).$$

**Limitations of Prior Work**: The optimal strategy in ZC24 is proportional to prediction uncertainty $\pi^{\text{opt}}_t \propto \sqrt{\mathbb{E}[(y_t - f_t(x_t))^2 \mid \mathcal{F}_{t-1}]}$. However, since the true distribution is unknown, practice requires using an uncertainty predictor $u_t(x_t)$ to approximate $|y_t - f_t(x_t)|$, followed by clipping based on the budget. Due to potential approximation errors and budget waste, ZC24 linearly mixes this with a uniform strategy $\pi^{\text{unif}} = T_b/T$ as $\pi^{(\lambda)}_t = (1-\lambda)\pi_t + \lambda\pi^{\text{unif}}_t$, setting $\lambda=0.5$. Crucially, **this estimator previously lacked non-asymptotic anytime confidence bounds, possessing only asymptotic normality**.

**Key Challenge**: Upon running ZC24's public implementation and scanning different mixing coefficients $\lambda$, the authors observed a counter-intuitive phenomenon: $\lambda=1$ (pure uniform, **completely ignoring** uncertainty) yielded confidence interval (CI) widths comparable to or even slightly narrower than $\lambda=0.5$ (Figure 1). This suggests that the "sophisticated" design of adjusting query probability based on current covariate uncertainty is fragile and potentially unnecessary in practice.

**Goal**: (1) Supplement the sequential estimator with non-asymptotic analysis, providing data-dependent bounds as observations accumulate; (2) Theoretically explain why the uncertainty component appears "useless" and provide an online learning scheme for query probability with no-regret guarantees.

**Core Idea**: The paper formulates the sequential estimation as an **online update**, using Freedman's inequality to characterize its convergence. It then models "selecting query probability" as an online convex optimization problem **independent of the current covariate**, solved via FTRL. The sublinear regret of FTRL forces query probabilities to converge to the budget upper bound $\tau = T_b/T$, theoretically validating that "uniform is optimal."

## Method

### Overall Architecture
Rather than proposing new networks, this work addresses the **statistical/online learning problem** of active sequential mean estimation through two efforts: first, rewriting the ZC24 estimator as step-by-step online updates and proving a non-asymptotic bound; second, designing an online rule to **actively control the variance term within this bound**. These threads form a closed loop: for each sample $x_t \rightarrow$ FTRL provides query probability $p_t$ (based on history, not current $x_t$) $\rightarrow$ Decide whether to query via $\text{Bernoulli}(p_t) \rightarrow$ Update the unbiased estimate $w_{t+1}$, and update the prediction model $f_t$ and approximate oracle per batch. The final output is the $(1-\alpha)$ confidence interval $\mathrm{CI}_\alpha = \big[ w_{T+1} \pm z_{1-\alpha/2} \hat\sigma / \sqrt{T} \big]$.

The theoretical pipeline is: estimation error $|w_{t+1} - \mu_y|$ is controlled by a term involving **cumulative conditional variance** $S_t = \sum_{s \le t} \sigma_s^2$ (Theorem 1); the only term in the conditional variance affected by query probability is $\frac{1}{p_t} \mathbb{E}[(y_t - f_t(x_t))^2 \mid \mathcal{F}_{t-1}]$ (Lemma 2); thus, "selecting $p_t$ to minimize the bound" is reduced to an online optimization problem (Section 5).

### Key Designs

**1. Rewriting sequential active estimation as online updates with anytime non-asymptotic bounds via Freedman's Inequality**

Addressing the gap of "only asymptotic guarantees," the authors rewrite the estimator as $w_{t+1} = w_t + \frac{1}{T}g_t$, where $g_t := f_t(x_t) + (y_t - f_t(x_t)) \frac{\xi_t}{\pi_t(x_t)}$ satisfies $\mathbb{E}[g_t] = \mu_y$, with initial value $w_1 = 0$. Since $g_t - \mu_y$ is a martingale difference sequence, the authors apply Freedman’s martingale concentration inequality (Lemma 1), which provides a deviation bound **adaptive to cumulative conditional variance** rather than worst-case variance. Theorem 1 states: for any $\delta \in (0, 1/e)$, with probability at least $1-\delta$, for all $t \in [T]$:

$$|w_{t+1} - \mu_y| \le \frac{2\max\{2\sqrt{S_t}, (G+|\mu_y|)\sqrt{\log(\log(T)/\delta)}\}\sqrt{\log(\log(T)/\delta)}}{T} + \Big(1-\frac{t}{T}\Big)|\mu_y|,$$

where $S_t = \sum_{s=1}^{t}\sigma_s^2$, $\sigma_s^2 = \mathbb{E}[(g_s - \mu_y)^2 \mid \mathcal{F}_{s-1}]$, and $|g_t| \le G$. This bound is "uniform over $t$" and consists of two parts: a burn-in term $(1-t/T)|\mu_y|$ dominating early on ($t \ll T$), and a primary term dominating after burn-in that yields $|w_{t+1} - \mu_y| = O(1/\sqrt{t})$. Its value lies in explicitly linking error magnitude to $S_t$, suggesting that **lowering cumulative conditional variance $S_t$ can lead to faster convergence**.

**2. Conditional variance decomposition + Approximate Oracle: Reducing decisions to a single online-optimizable scalar**

To minimize $S_t$, Lemma 2 proves that the **only** term dependent on the query probability $\pi_t(x_t)$ is $\mathbb{E}\big[ (y_t - f_t(x_t))^2 \frac{1}{\pi_t(x_t)} \mid \mathcal{F}_{t-1} \big]$. When the policy only depends on history ($\pi_t(x_t) = p_t$ is $\mathcal{F}_{t-1}$-measurable), this simplifies to $\frac{1}{p_t} \mathbb{E}[(y_t - f_t(x_t))^2 \mid \mathcal{F}_{t-1}]$. Larger $p_t$ reduces variance but is limited by the budget—a clean trade-off. Since $\mathbb{E}[(y_t - f_t(x_t))^2 \mid \mathcal{F}_{t-1}]$ is unknown, the authors introduce an **approximate oracle** $\Phi_t(x_t) \in \mathbb{R}_+$ satisfying:

$$\frac{1}{c_1}\Phi_t(x_t) \le \mathbb{E}\big[ (y_t - f_t(x_t))^2 \mid \mathcal{F}_{t-1} \big] \le c_0 \Phi_t(x_t), \quad c_0, c_1 > 0.$$

In practice, $\Phi_t$ is implemented via linear regression on squared residuals $(f_t(x_t) - y_t)^2$, updated per batch.

**3. FTRL for Online Query Probability: Closed-form solution forcing $p_t \to \tau = T_b/T$**

The authors define the per-round loss as $\tilde\ell_t(p) = \Phi_t(x_t)/p$ (convex on $(0,1]$) and use FTRL to select $p_t$ within the feasible region $[\beta, \tau]$:

$$p_t \leftarrow \arg\min_{p \in [\beta, \tau]} \gamma \theta_{t-1} p + \frac{1}{2} p^2, \qquad \theta_{t-1} := -\sum_{s=1}^{t-1} \frac{\Phi_s(x_s)}{p_s^2}.$$

Lemma 3 provides the closed-form $p_t = \max\{\beta, \min\{\tau, -\gamma \theta_{t-1}\}\}$. The upper bound $\tau = T_b/T$ enforces the budget, while the lower bound $\beta > 0$ ensures minimal exploration. Since $\Phi_s(\cdot) \ge 0$, the loss $\Phi_s/p$ is always minimized at $p=\tau$. Thus, to maintain sublinear regret, $p_t$ **must eventually converge to the upper bound $\tau = T_b/T$**. This confirms the counter-intuitive observation: **when query probability is blind to the current covariate $x_t$, the optimal strategy is a constant probability $T_b/T$**.

### Loss & Training
- **Estimator Update**: $w_{t+1} = w_t + \frac{1}{T}\big( f_t(x_t) + (y_t - f_t(x_t)) \frac{\xi_t}{p_t} \big)$.
- **FTRL Loss**: $\tilde\ell_t(p) = \Phi_t(x_t)/p$ with regularization $R(p) = \frac{1}{2} p^2$.
- **Model/Oracle Update**: Labeled data is split into two disjoint subsets $D^\clubsuit, D^\spadesuit$ to update $f_{t+1}$ and $u_{t+1}/\Phi_{t+1}$ separately, preventing the oracle from underestimating uncertainty on seen data.

## Key Experimental Results

### Main Results (CI Width)
The authors compared **FTRL** (Ours), ZC24 (mixed policy), and a **uniform sampling baseline** (which uses a fixed predictor $f$).

| Dataset | Metric | FTRL (Ours) | ZC24 Mixed | Uniform Baseline |
| :--- | :--- | :--- | :--- | :--- |
| Politeness Score | Interval Width | Comparable to ZC24 | Baseline for comparison | Significantly wider |
| Wine Rating | Interval Width | Comparable to ZC24 | Baseline for comparison | Significantly wider |
| Post-Election Survey | Interval Width | Comparable to ZC24 | Baseline for comparison | Significantly wider |
| Coverage | Valid Coverage | High | High | High |

Conclusion: FTRL and ZC24 yield **comparable widths**, and both significantly outperform the fixed-predictor uniform baseline.

### Ablation Study (Mixing Coefficient $\lambda$)
Scanning $\lambda \in \{0.05, 0.1, 0.5, 0.8, 1.0\}$ for ZC24 reveals that $\lambda=1$ (pure uniform, ignoring uncertainty) provides widths equal to or **slightly narrower** than $\lambda=0.5$.

### Key Findings
- **Uncertainty components are fragile**: Fully weighting the uniform policy ($\lambda=1$) does not degrade performance and is sometimes better.
- **Gains stem from "model updates," not "smart querying"**: FTRL converges to $T_b/T$ (same as uniform sampling) yet still outperforms the fixed-predictor baseline. The improvement comes from continuously updating $f$ and the oracle.

## Highlights & Insights
- **Deriving uniform optimality via no-regret**: By modeling query probability selection as FTRL, the regret geometry naturally identifies $T_b/T$ as the attractor.
- **Anytime-valid non-asymptotic bounds**: Freedman's inequality provides guarantees for stream/sequential settings that were previously only available asymptotically.
- **Practical heuristic**: If an active sampling scheme seems to require sample-level uncertainty, first test a version blind to the current sample. If results are similar, complex uncertainty modeling can be avoided.

## Limitations & Future Work
- **Blindness Assumption**: "Uniform is optimal" specifically holds when query probabilities are blind to $x_t$. If $p_t$ can be conditioned on $x_t$ (as theoretically intended by ZC24), gains might still exist.
- **Oracle Assumptions**: Theorem 2 assumes $\Phi_t \le B$ and a constant-factor sandwich approximation of true conditional variance.
- **Distribution Setting**: Analysis is limited to scalar labels and stationary data streams. Extensions to non-stationary or high-dimensional settings remain future work.

## Related Work & Insights
- **vs ZC24**: Ours adds anytime-valid non-asymptotic bounds and replaces fixed-mixing heuristics with the FTRL framework.
- **vs PPI / PPI++**: While PPI uses fixed data for bias correction, this work focuses on the **query probability** selection in an active, online update setting.

## Rating
- Novelty: ⭐⭐⭐⭐☆
- Experimental Thoroughness: ⭐⭐⭐⭐☆
- Writing Quality: ⭐⭐⭐⭐☆
- Value: ⭐⭐⭐⭐☆

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multiple-Prediction-Powered Inference](multiple-prediction-powered_inference.md)
- [\[ICLR 2026\] Mean Estimation from Coarse Data: Characterizations and Efficient Algorithms](mean_estimation_from_coarse_data_characterizations_and_efficient_algorithms.md)
- [\[ICLR 2026\] Revenue Maximization under Sequential Price Competition via the Estimation of s-Concave Demand Functions](revenue_maximization_under_sequential_price_competition_via_the_estimation_of_s-.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Multiple-Prediction-Powered Inference](multiple-prediction-powered_inference.md)
- [\[ICLR 2026\] Mean Estimation from Coarse Data: Characterizations and Efficient Algorithms](mean_estimation_from_coarse_data_characterizations_and_efficient_algorithms.md)
- [\[ICLR 2026\] Revenue Maximization under Sequential Price Competition via the Estimation of s-Concave Demand Functions](revenue_maximization_under_sequential_price_competition_via_the_estimation_of_s-.md)
- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)

</div>

<!-- RELATED:END -->
