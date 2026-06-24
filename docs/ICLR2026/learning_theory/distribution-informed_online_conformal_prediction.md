---
title: >-
  [Paper Note] Distribution-informed Online Conformal Prediction
description: >-
  [ICLR2026][Learning Theory][Online Conformal Prediction] This paper proposes COP (Conformal Optimistic Prediction), which adds an "optimistic correction" step beyond the traditional reactive updates in online conformal prediction. By using the estimated CDF of non-conformity scores as a "hint" for the next step, the method produces narrower prediction intervals when predictable patterns exist in the data. Meanwhile, it maintains distribution-free finite-sample coverage guaran…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Conformal Prediction"
  - "Online Learning"
  - "Uncertainty Quantification"
  - "Online Conformal Prediction"
  - "Optimistic Gradient Descent"
  - "Coverage Guarantee"
  - "Regret Bound"
  - "Time Series"
date: 2026-05-08
content_hash: 5d4f76a9278d5072
---

# Distribution-informed Online Conformal Prediction

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=I69SaLbwqZ](https://openreview.net/forum?id=I69SaLbwqZ)  
**Code**: https://github.com/creator-xi/Conformal-Optimistic-Prediction  
**Area**: Learning Theory / Conformal Prediction / Online Learning / Uncertainty Quantification  
**Keywords**: Online Conformal Prediction, Optimistic Gradient Descent, Coverage Guarantee, Regret Bound, Time Series

## TL;DR
This paper proposes COP (Conformal Optimistic Prediction), which adds an "optimistic correction" step beyond the traditional reactive updates in online conformal prediction. By using the estimated CDF of non-conformity scores as a "hint" for the next step, the method produces narrower prediction intervals when predictable patterns exist in the data. Meanwhile, it maintains distribution-free finite-sample coverage guarantees, ensuring that long-term coverage is not compromised even if the CDF estimation is inaccurate.

## Background & Motivation
**Background**: Conformal Prediction (CP) is a mainstream tool for uncertainty quantification. Under the assumption of "data exchangeability," it can construct a prediction set that contains the true value with a specified probability $1-\alpha$ in a model-agnostic and distribution-free manner. However, real-world time series exhibit distribution shifts and temporal correlations, violating exchangeability. This led to the development of online conformal prediction (online CP): pioneered by ACI, it integrates adversarial online learning into the CP framework, dynamically adjusting thresholds using gradient updates such as $\hat q_{t+1}=\hat q_t+\eta(\mathrm{err}_t-\alpha)$ to pursue only long-term coverage $\lim_{T\to\infty}\frac1T\sum_t \mathbf 1\{Y_t\notin \hat C_t\}=\alpha$. Subsequent works like Conformal PID, ECI, decay-OGD, and expert aggregation have further improved step sizes and update rules.

**Limitations of Prior Work**: These online learning methods treat the environment as **fully adversarial**, correcting errors reactively based only on whether the "previous step was correct," relying on the cancellation of positive and negative errors to meet the coverage target. Consequently, the threshold $q_t$ oscillates violently around the ground truth, and prediction intervals are generally **too conservative and wide**, wasting predictable information inherent in the sequence. Another line of research (SPCI, Lee et al., HOP-CPT) directly estimates the score distribution or weights historical scores to generate intervals. While they utilize temporal structure, they either rely on assumptions like model consistency/distribution smoothness (not distribution-free) or lack finite-sample coverage guarantees, making them prone to overfitting.

**Key Challenge**: There is a tension between "utilizing predictable patterns to narrow intervals" and "obtaining coverage guarantees without distribution assumptions." Purely reactive methods maintain guarantees but are too wide, while pure distribution estimation methods are narrow but tethered entirely to the accuracy of the estimation.

**Goal**: To design an online algorithm that dynamically selects thresholds to achieve long-term coverage without making any distributional assumptions about the data, while keeping the intervals as narrow as possible.

**Key Insight**: The authors observe that the reactive update in online CP, $\hat q_{t+1}=\hat q_t-\eta\nabla\ell_{1-\alpha}(s_t-\hat q_t)$, is essentially Online (Sub)Gradient Descent (OGD) on the quantile loss. In online optimization, "Optimistic Online Gradient Descent" (OOGD) reduces regret by incorporating a **hint** (a prediction of the next gradient) into the objective. If the "estimated score distribution" can be injected as this hint, one can leverage structural information while retaining the coverage guarantees provided by OGD.

**Core Idea**: In addition to the reactive update at each step, an **optimistic correction** step is added based on the estimated CDF. By using $\hat F_{t+1}(\hat q_{t+1})-(1-\alpha)$ as a hint to anticipate the next threshold movement, the algorithm is analyzed through the lens of OOGD to simultaneously tighten both the regret and coverage bounds.

## Method

### Overall Architecture
COP takes sequentially arriving data $\{(X_t,Y_t)\}$ and a base predictor $\hat f_t$ as input, and outputs prediction intervals $\hat C_t(X_t)=[\hat f_t(X_t)-q_t,\ \hat f_t(X_t)+q_t]$ for each time step. At each step, it maintains **two sets of radii**: a "main radius" $\hat q_t$ and a "corrected radius" $q_t$ used for the actual interval.

The complete workflow at time $t$ is: ① Observe $X_t$ and output the interval using the current corrected radius $q_t$; ② Observe the true label $Y_t$, calculate the non-conformity score $s_t$ (e.g., $s_t=|Y_t-\hat f_t(X_t)|$) and the error indicator $\mathrm{err}_t=\mathbf 1\{s_t>q_t\}$; ③ Update the main radius $\hat q_{t+1}=\hat q_t+\eta(\mathrm{err}_t-\alpha)$ using OGD; ④ Estimate the CDF $\hat F_{t+1}$ using scores from a recent window; ⑤ Apply an optimistic correction based on the main radius to obtain $q_{t+1}=\hat q_{t+1}-\lambda_{t+1}(\hat F_{t+1}(\hat q_{t+1})-(1-\alpha))$. Note that the indicator uses $\mathrm{err}_t=\mathbf 1\{s_t>q_t\}$ (based on the actual corrected radius), not $\mathbf 1\{s_t>\hat q_t\}$ as in previous methods—this integrates the correction step into the coverage feedback loop.

The key is that the main radius $\hat q$ is responsible for "maintaining long-term coverage" (standard OGD guarantee), while the corrected radius $q$ is responsible for "narrowing the interval using distribution information while maintaining coverage." Since COP reduces to standard OGD when $\lambda_{t+1}=0$, it is a strict superset of OGD. This is a purely online, first-order iterative method without multi-module coordination or complex pipeline branching.

### Key Designs

**1. Optimistic Correction Step based on Estimated CDF: Injecting Predictable Structure without Breaking Guarantees**

To address the conservatism and excessive width of reactive updates, COP adds a correction after the main update. The premise is: if the conditional distribution of $s_t\mid S_{t-1}$ is roughly stationary, the goal of CP is approximately equivalent to finding the $(1-\alpha)$-quantile of $s_t$. Thus, the corrected radius is found by solving a quadratically regularized objective: $q_{t+1}=\arg\min_q \mathbb E_{s_{t+1}}[\ell_{1-\alpha}(s_{t+1}-q)\mid S_t]+\frac{1}{2\lambda_{t+1}}\|q-\hat q_{t+1}\|_2^2$. Since the gradient $\nabla_q L_{t+1}(q)=F_{t+1}(q)-(1-\alpha)$ lacks a closed-form solution for this implicit update, the authors use a first-order linear approximation of $L_{t+1}$ at $\hat q_{t+1}$ and replace the true CDF $F_{t+1}$ with the estimate $\hat F_{t+1}$, resulting in the closed-form correction:

$$q_{t+1}=\hat q_{t+1}-\lambda_{t+1}\big(\hat F_{t+1}(\hat q_{t+1})-(1-\alpha)\big).$$

The intuition is clear: if the estimated CDF shows that the cumulative probability at the current radius is higher than the target $1-\alpha$ (over-coverage), the radius is pushed down; otherwise, it is expanded. Unlike SPCI or Lee et al., which directly compute $\hat F^{-1}(1-\alpha)$, COP **does not require inverting the estimated CDF**, avoiding numerical errors and computational costs. Furthermore, it only "takes a small step from the OGD output" rather than committing entirely to the estimate, which is why it is both narrow and stable. By default, an empirical CDF is used: $\hat F_{t+1}(\hat q_{t+1})=\frac1w\sum_{i=t-w+1}^{t}\mathbf 1\{s_i\le \hat q_{t+1}\}$ (window $w=100$).

**2. OOGD Perspective and Distribution-aware Hint: Trust via a Scale Factor**

The two-step update of COP can be strictly rewritten in the form of Optimistic Online Gradient Descent: first $\hat q_{t+1}=\arg\min_q\{\eta\langle\nabla\ell_{1-\alpha}(s_t-q_t),q\rangle+\frac12\|q-\hat q_t\|^2\}$, then $q_{t+1}=\arg\min_q\{\eta\langle M_{t+1},q\rangle+\frac12\|q-\hat q_{t+1}\|^2\}$. Here $M_{t+1}$ is the optimistic term (hint) in OOGD. While classical OOGD often uses the previous gradient as a hint, COP uses a **distribution-aware hint**:

$$M_{t+1}=\big(\hat F_{t+1}(\hat q_{t+1})-(1-\alpha)\big)\cdot \lambda_{t+1}/\eta,$$

which characterizes the potential direction of distribution shift for the next score. The ratio $\lambda_{t+1}/\eta\le 1$ is defined as the scale factor, directly expressing "the degree of trust in $\hat F_{t+1}$." If the base model is strong and scores are highly stochastic, $\lambda$ is reduced (falling back to reactive mode); if $\hat F$ is reliable, the scale factor remains near 1. Unifying this under the OOGD framework allows the use of established theories where "the more accurate the hint, the smaller the regret."

**3. Joint Regret-Coverage Bound: Proving that a Good Hint Tightens Both**

Online CP usually examines coverage and dynamic regret separately, and one does not typically imply the other. This paper derives a **joint bound** from the OOGD perspective (Theorem 1, constant step size):

$$\underbrace{\tfrac1T\sum_t[\ell_t(q_t)-\ell_t(u_t)]}_{\text{Regret}}+\underbrace{\tfrac{\eta(1-2\alpha)}{4}\big(\tfrac{\sum_t\mathrm{err}_t}{T}-\alpha\big)}_{\text{Coverage}}\le \tfrac{\eta}{T}\sum_t\|\alpha-\mathrm{err}_t-M_t\|_2^2+\underbrace{\sum_t\tfrac{1}{2\eta}\big(\|u_t-\hat q_t\|^2-\|u_{t-1}-\hat q_t\|^2\big)}_{\text{Environmental Non-stationarity}}.$$

The term $\|\alpha-\mathrm{err}_t-M_t\|^2$ on the right side indicates that **as long as $M_t$ is chosen close to $F_t(\hat q_t)-(1-\alpha)$, both the regret and coverage bounds are simultaneously lowered**. This corresponds to the COP choice when the scale factor is 1. This bound theoretically explains "why injecting a distribution hint is beneficial," serving as a core theoretical contribution beyond empirical observation.

**4. Distribution-free Finite-sample Coverage + Asymptotic Consistency**

The coverage guarantee of COP does not depend on the accuracy of $\hat F$. Proposition 2 provides a finite-sample bound $\big|\frac1T\sum_t\mathrm{err}_t-\alpha\big|\le \frac{B+(2+6M)\eta}{T\eta}$ (for bounded scores/hints), and the algorithm **does not need to know** the specific bounds of $B,M$. As $T\to\infty$, long-term coverage $\frac1T\sum\mathrm{err}_t\to\alpha$ is achieved. Theorem 2 extends this to dynamic step sizes $\eta_t$. Furthermore, Theorem 3 proves that when scores are i.i.d. and the step size satisfies the Robbins–Monro conditions, $q_t$ **converges** to the true $(1-\alpha)$-quantile $q^*$. This overcomes the oscillation inherent in fixed-step OGD, even with the addition of bounded optimistic terms.

### Mechanism Example
Suppose the target coverage $1-\alpha=90\%$, the main radius is $\hat q_{t+1}=8.5$, and the scale factor $\lambda_{t+1}=0.5$. If 95 out of the last 100 scores are $\le 8.5$, then the empirical CDF $\hat F_{t+1}(8.5)=0.95$, and the hint $=0.95-0.90=0.05>0$, indicating over-coverage. The correction yields $q_{t+1}=8.475$, slightly narrowing the interval. The next step observes if $s_{t+1}$ falls within the interval to get $\mathrm{err}_{t+1}$, which is fed back into the main update.

### Loss & Training
COP requires no additional model training. There are three core hyperparameters: base learning rate $\eta$, scale factor $\lambda=0.5$, and window length $w=100$. It adopts the adaptive step size from previous work: $\eta_t=\eta\cdot(\max\{s_{t-w+1..t}\}-\min\{s_{t-w+1..t}\})$. The optimization target is the $(1-\alpha)$ quantile loss $\ell_{1-\alpha}(q)=(\mathbf 1\{q>0\}-\alpha)q$.

## Key Experimental Results

### Main Results
Target coverage $1-\alpha=90\%$. Comparison with 7 SOTAs: ACI, OGD, SF-OGD, decay-OGD, Conformal PID, ECI, LQT(fixed) across three base predictors (Prophet / AR / Theta). The table below excerpts results under Prophet for two synthetic datasets.

| Dataset | Method | Coverage (%) | Mean Width | Median Width |
|--------|------|-----------|----------|----------|
| Changepoint | ACI | 89.9 | ∞ | 8.20 |
| Changepoint | decay-OGD | 90.0 | 8.30 | 8.22 |
| Changepoint | ECI | 89.9 | 8.16 | 8.25 |
| Changepoint | **COP** | 89.8 | 8.29 | 8.44 |
| Dist. Drift | ACI | 89.9 | ∞ | 6.69 |
| Dist. Drift | decay-OGD | 90.6 | 7.64 | 6.95 |
| Dist. Drift | ECI | 90.0 | 7.27 | 6.98 |
| Dist. Drift | **COP** | 90.6 | **7.07** | **6.89** |

COP maintains ~90% coverage while achieving the tightest or near-tightest width, particularly under distribution drift where the mean width (7.07) is lower than all baselines.

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| $\lambda=0$ (OGD) | Wider intervals | Removal of optimistic correction reverts to purely reactive mode. |
| ECDF vs KDE | Similar | Choice of estimator is robust. |
| Inaccurate CDF | Coverage maintained | Matches Prop. 2; poor estimation only reverts to the baseline. |
| Scale factor $\lambda$ | Affects shrinkage | Larger $\lambda$ increases trust in estimates and narrowing. |

### Key Findings
- Narrowing originates from the optimistic correction step: $\lambda=0$ results in the wider intervals of OGD, proving the contribution of the distribution hint.
- Coverage is robust to estimation quality: even with poor CDF estimation, long-term coverage remains near 90%.
- Universal across base predictors: COP maintains performance across Prophet, AR, and Theta predictors.

## Highlights & Insights
- **Reinterpreting online CP as OOGD bridges empirical improvement and theoretical guarantees**: By recognizing reactive updates as OGD on quantile loss, the optimistic correction naturally becomes the OOGD hint, allowing the use of established regret analyses.
- **Clean design of the scale factor**: $\lambda/\eta\le 1$ simultaneously captures "trust in estimates" and the "interpolation between reactive and distributional modes."
- **Joint Regret-Coverage Bound**: Proves that a proper hint reduces both metrics, transforming intuition into a formal theorem.
- **Avoiding CDF inversion** is an engineering high point that saves computation and avoids numerical instability.

## Limitations & Future Work
- The "same sign" assumption in Proposition 1 is relatively strong, though a revised version is provided in the appendix.
- Asymptotic consistency (Theorem 3) relies on the **i.i.d.** assumption, which diverges from the strongly correlated reality of time series.
- The window length $w$ and scale factor $\lambda$ remain hyperparameters that require manual tuning.
- Validated only on 1D interval-based scores; extension to classification or multidimensional outputs remains to be explored.

## Related Work & Insights
- **vs ACI / OGD / decay-OGD**: These are purely reactive, treating the environment as adversarial. COP adds distribution-aware correction to narrow intervals and achieve convergence under i.i.d. conditions.
- **vs Conformal PID / ECI**: PID requires training a separate scorecaster, which might introduce variance. COP directly utilizes the estimated CDF without extra models.
- **vs SPCI / Lee et al.**: These rely on distribution smoothness/consistency and require CDF inversion. COP is distribution-free, uses the estimate only as a hint, and avoids inversion.

## Rating
- Novelty: ⭐⭐⭐⭐ Restating online CP as OOGD with distribution hints is a clean and theoretically supported perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across 9 datasets and 3 base predictors with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and smooth transition from theory to algorithm.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play, theoretically sound, and interval-narrowing method for time-series uncertainty quantification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)
- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)
- [\[ICML 2026\] Enhancing Conformal Prediction via Class Similarity](../../ICML2026/learning_theory/enhancing_conformal_prediction_via_class_similarity.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Adaptive Conformal Prediction via Mixture-of-Experts Gating Similarity](adaptive_conformal_prediction_via_mixture-of-experts_gating_similarity.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)
- [\[ICLR 2026\] Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting](conformal_prediction_with_corrupted_labels_uncertain_imputation_and_robust_re-we.md)
- [\[ICML 2026\] Enhancing Conformal Prediction via Class Similarity](../../ICML2026/learning_theory/enhancing_conformal_prediction_via_class_similarity.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)

</div>

<!-- RELATED:END -->
