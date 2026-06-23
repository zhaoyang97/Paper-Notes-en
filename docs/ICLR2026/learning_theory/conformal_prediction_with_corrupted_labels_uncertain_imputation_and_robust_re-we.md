---
title: >-
  [Paper Note] Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting
description: >-
  [ICLR 2026][learning_theory][Paper Note] Aiming at scenarios where training labels are corrupted by noise or missing and key features are unavailable at test time (privileged information), this paper first proves the precise conditions under which existing Privileged Conformal Prediction (PCP) remains valid despite inaccurate weight estimation. It then propos
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 0d259e1ad5707580
---
# Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=ztEKLEUNKS](https://openreview.net/forum?id=ztEKLEUNKS)  
**Code**: https://github.com/Shai128/ui  
**Area**: Learning Theory / Conformal Prediction / Uncertainty Quantization  
**Keywords**: Conformal Prediction, Corrupted Labels, Privileged Information, Weighted Conformal, Uncertain Imputation

## TL;DR
Aiming at scenarios where training labels are corrupted by noise or missing and key features are unavailable at test time (privileged information), this paper first proves the precise conditions under which existing Privileged Conformal Prediction (PCP) remains valid despite inaccurate weight estimation. It then proposes a new method, UI, that relies on "backfilling labels with uncertainty" instead of weights. Finally, it takes the union of Naive CP, PCP, and UI to obtain a TriplyRobust calibration scheme that is valid as long as one of the underlying assumptions holds.

## Background & Motivation
**Background**: Conformal Prediction (CP) is a general tool for providing "statistical guarantees" for any predictive model. It uses a hold-out calibration set to calculate non-conformity scores and takes a quantile as a threshold to construct a prediction set for test points that covers the true label with a user-specified probability $1-\alpha$ (e.g., 90%). The premise of this guarantee is that the calibration and test data are **exchangeable (i.i.d.)**.

**Limitations of Prior Work**: In reality, training labels are often corrupted—either noisy or missing. Once labels are corrupted, only "clean samples" can be used to calculate scores. However, the distribution of clean samples $P_{X,Y\mid M=0}$ often differs from the test distribution $P_{X,Y}$, breaking exchangeability. Naive CP (performing CP only on observed labels) leads to **under-coverage**. Using MEPS medical data, the paper demonstrates that Naive CP fails to achieve 90% coverage after labels are randomly deleted based on feature-related probabilities.

**Key Challenge**: Distribution shift caused by corruption is essentially a form of **covariate shift**. The standard approach is Weighted Conformal Prediction (WCP), which re-weights scores by the likelihood ratio $w(z)=\mathrm{d}P_{\text{test}}/\mathrm{d}P_{\text{train}}(z)$ to restore exchangeability. However, WCP requires all features to be available at test time to compute weights. In practice, features that explain "why a label is corrupted" (e.g., income, race, annotator quality) are often unavailable at test time due to privacy or missingness—these features, visible only during training, are **Privileged Information (PI)**. Existing PCP can provide valid prediction sets without test PI, but it relies on a strict premise: the **weights $w$ must be ground truth**. In MEPS experiments, PCP also under-covers once estimated weights are used.

**Goal**: In the setting of "corrupted labels + missing privileged features at test time," this paper aims to answer two questions: (1) How sensitive is PCP to weight errors? (2) Can weight estimation be bypassed entirely if it is unreliable?

**Key Insight**: The authors found that PCP's validity is not monotonically related to weight accuracy; rather, it is strongly correlated with whether Naive CP itself over-covers or under-covers. On the other hand, if PI is not used to explain the corruption indicator $M$ but can instead **predict the label $Y$ itself**, one can take a different path: directly backfilling the corrupted labels.

**Core Idea**: Replace "weight re-weighting (PCP)" with "uncertainty-preserving label backfilling (UI)" to counter distribution shift from corrupted labels, and combine both paths with Naive CP into a TriplyRobust union that is valid if at least one assumption holds.

## Method
The paper splits the problem into two cases based on the role of PI: Case 1 where PI is an explanatory variable for the corruption indicator $M$ (PCP + robustness analysis), and Case 2 where PI is a proxy for the label $Y$ (UI). Finally, both are combined with Naive CP into a TriplyRobust scheme.

### Overall Architecture
Let the training samples be $\{(X_i, \tilde{Y}_i, Z_i, M_i)\}_{i=1}^n$, where $X_i$ are observed features, $\tilde{Y}_i$ are possibly corrupted labels, $Z_i$ is privileged information, and $M_i\in\{0,1\}$ is the corruption indicator ($M_i=0$ means $\tilde{Y}_i=Y_i$ is clean, $M_i=1$ means corrupted/missing). Only $X_{\text{test}}$ is provided at test time. The goal is to construct a prediction set satisfying marginal coverage $P(Y_{\text{test}}\in C(X_{\text{test}}))\ge 1-\alpha$. The challenge is the shift between $P_{X,Y\mid M=0}$ and $P_{X,Y}$. The key assumption is $(X,Y)\perp M\mid Z$: given PI, corruption is independent of the true data.

The work proceeds along three lines: ① **Robustness characterization** of weight-dependent PCP, providing precise intervals where coverage is maintained despite weight errors; ② Proposal of **UI**, which avoids weights by backfilling labels; ③ **Union** of Naive CP, PCP, and UI into TriplyRobust, which is valid if at least one of the three sets of assumptions holds.

### Key Designs

**1. PCP Robustness Characterization: Mapping weight accuracy to intervals bound by Naive CP coverage status**

The PCP procedure involves running a WCP sub-routine for each calibration point as a test point using clean samples to get thresholds $Q(Z_i)$, then taking the $(1-\beta)$ empirical quantile of these thresholds as the final $Q_{\text{PCP}}$ ($\beta\in(0,\alpha)$), bypassing test PI. Theorem 1 guarantees its validity under **true weights**. This paper’s contribution is asking: what happens when weights are inaccurate? The authors first consider constant errors $\tilde{w}_i := w_i+\delta$ (Theorem 2). The conclusion is unexpectedly non-monotonic—**whether PCP is valid depends on whether Naive CP over-covers or under-covers**: if Naive CP over-covers ($Q_{\text{CP}}>Q_{\text{WCP}}$), PCP remains valid even with poor weights ($\delta\ge 0$). However, if Naive CP under-covers ($Q_{\text{CP}}<Q_{\text{WCP}}$), $\delta$ must fall into a narrow interval $\left(-\tfrac{W_{n+1}}{n+1},\,0\right)$ to maintain coverage $P(Y_{\text{test}}\in C_{\text{PCP}})\ge 1-\alpha-\varepsilon$. Theorem 3 extends this to **sample-wise varying errors** $\tilde{w}_i=w_i+\delta_i$.

This characterization differs from prior worst-case analyses—it shows that PCP can remain valid even with significant weight errors, replacing the intuition that "weights must be very accurate" with "examine Naive CP coverage status + whether error falls in a computable interval." On synthetic data, the empirical validity intervals match the theoretical boundaries, explaining why PCP fails on MEPS when errors fall outside the narrow interval.

**2. Uncertain Imputation (UI): Using PI for "noisy backfilling" to bypass weight estimation**

When PI is a strong proxy for the label $Y$ (e.g., high-resolution images or detailed reports available only during training), UI estimates labels instead of weights. Data is split into training $I_1$, calibration $I_2$, and reference $I_3$. Two models are trained: $\hat{f}(x)$ using only $X$, and $\hat{g}(x,z)$ using both $X,Z$. Residuals $E_i = Y_i-\hat{g}(X_i,Z_i)$ are calculated on the reference set to collect conditional residual pools $\mathcal{E}(z)=\{E_i: i\in I_3, Z_i=z, M_i=0\}$. For backfilling, clean samples keep their true labels, while corrupted samples are filled with the predicted value **plus a random error drawn from the residual pool**:

$$\bar{Y}_i = \begin{cases} Y_i & M_i=0\\ \hat{g}(X_i,Z_i)+E(Z_i) & M_i=1\end{cases}$$

Adding the random residual $E(Z_i)$ instead of filling the mean is the core insight: filling the mean (Naive Imputation) **artificially reduces label variance**, leading to narrow prediction sets and under-coverage. "Uncertainty-preserving backfilling" replicates the dispersion of true labels. Leveraging the result that CP is robust to dispersive (variance-increasing) noise, the backfilled scores $\bar{S}_i=S(X_i,\bar{Y}_i;\hat{f})$ provide a valid threshold $Q_{\text{UI}}$. Theorem 4 proves that if $\hat{g}$ is sufficiently accurate, UI satisfies $P(Y_{\text{test}}\in C_{\text{UI}})\ge 1-\alpha$.

**3. TriplyRobust: Union of three prediction sets, "OR" rather than "AND" logic**

PCP and UI rely on **complementary** assumptions—PCP requires accurate $M\mid Z$ estimation, while UI requires accurate $Y\mid Z$ estimation. Naive CP is valid if the base model $\hat{f}$ is ideal. The authors take the union of the three sets:

$$C_{\text{TriplyRobust}}(X_{\text{test}}) = C_{\text{Naive CP}}(X_{\text{test}}) \cup C_{\text{PCP}}(X_{\text{test}}) \cup C_{\text{UI}}(X_{\text{test}})$$

Theorem 5 guarantees that as long as **at least one** of the three sets of assumptions holds (accuracy in $Y\mid X$, $M\mid Z$, or $Y\mid Z$), TriplyRobust achieves nominal coverage. The union naturally ensures coverage only increases, and experiments show it is **not overly conservative**—as long as one component is an oracle, coverage returns to nominal levels without excessive set expansion.

### Loss & Training
The method is a calibration-layer solution and does not introduce new training losses. Experiments use CQR (Conformalized Quantile Regression) as the non-conformity score with a target coverage of $1-\alpha=90\%$. The training set fits the model, the validation set is used for early stopping, and the calibration set is used for calibration (UI further splits a reference set). Results are averaged over 30 random trials.

## Key Experimental Results

### Main Results
The authors validated the method using three types of experiments: TriplyRobust robustness on synthetic data, UI's advantage over PCP when weights are hard to estimate, and "missing response" on 5 real regression benchmarks (Facebook1/2, Bio, House, Meps19).

| Experiment | Setting | Key Phenomenon |
|------|------|----------|
| Synthetic (Hard Weight Estimation) | $Z$ strongly predicts $Y$, corruption mechanism is hard to estimate | PCP fails to reach 90% due to inaccurate weights; UI remains stable at 90% based on $Y\mid X,Z$. |
| Real (Missing Response) | 5 benchmarks, 20% labels deleted, most $Y$-correlated feature used as PI | Naive CP / Naive Imputation are too narrow and undercover; PCP (true/est weight) and UI consistently reach 90%. |
| Synthetic (TriplyRobust) | QR / PCP / UI each take degenerate or oracle variants | Under-covers only when all are degenerate; valid and not overly conservative if any one is an oracle. |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|----------|
| Naive CP (observed labels only) | Under-coverage | Clean distribution $\neq$ test distribution; exchangeability broken. |
| Naive Imputation (mean filling) | Too narrow, under-coverage | Mean filling artificially reduces label variance. |
| UI (prediction + random residual) | Reaches 90% | Uncertainty-preserving backfilling restores true dispersion. |
| TriplyRobust (all degenerate) | Under-coverage | All three sets of assumptions fail. |
| TriplyRobust (any oracle) | Reaches 90%, not conservative | "OR" logic provides a safety net without over-expansion. |

### Key Findings
- **Contrast between Naive Imputation and UI**: Both perform backfilling, but the random residual is the deciding factor for validity. Mean filling under-covers; adding residuals works.
- **PCP validity is tied to Naive CP coverage status**: PCP remained valid with estimated weights in real experiments because errors fell within the theoretical intervals defined in Theorems 2/3.
- **TriplyRobust union is not overly conservative**: Although a union of three sets seems wide, coverage stays near 90% when one component is accurate, making it a "cheap insurance."

## Highlights & Insights
- **Reframing weight accuracy**: The insight that PCP sensitivity depends on Naive CP's over/under-coverage explains why weight errors do not always lead to failure, avoiding a dogmatic pursuit of perfect weights.
- **"Uncertainty-preserving backfilling" as a transferable trick**: In any task involving "missing value imputation + reliable downstream uncertainty," mean imputation shrinks variance. Adding random residuals from a conditional pool preserves dispersion, useful for regression calibration or semi-supervised learning.
- **"OR" vs "AND" assumptions**: TriplyRobust uses a union to combine complementary assumptions, a robust design pattern that replaces betting on a single model assumption with a multi-hypotheses safety net.

## Limitations & Future Work
- **Theoretical Dependency**: Validity conditions for WCP/PCP rely on true weights, which are unavailable in practice; estimating these conditions from data remains an open problem.
- **UI Assumptions**: UI requires that features and responses are independent of corruption given PI, and that label variation depends primarily on PI—assumptions that may be challenged by high-dimensional or continuous $Z$.
- **Future Work**: Extending theoretical guarantees to multiple-annotator settings and integrating privileged information into ambiguity-aware calibration methods.

## Related Work & Insights
- **vs WCP (Tibshirani et al. 2019)**: WCP uses likelihood ratio weighting for covariate shift but requires weights (all features) at test time. This paper addresses missing test-time PI.
- **vs PCP (Feldman & Romano 2024)**: PCP uses WCP as a sub-routine to bypass test PI but relies on true weights. This paper provides precise robustness bounds for weights and introduces UI to bypass weights entirely.
- **vs CP under Label Noise (Einbinder et al. 2023; Sesia et al. 2024)**: Prior work showed CP is robust to dispersive additive noise. UI leverages this by intentionally "backfilling with noise" rather than precise imputation.
- **vs Worst-case Weight Analysis**: Existing works often focus on worst-case bounds. Theorems 2/3 in this paper provide more granular intervals linked to Naive CP coverage status.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Non-monotonic characterization of PCP + Uncertain Imputation + TriplyRobust union are all novel and well-integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive synthetic and 5 real benchmarks; however, tasks are concentrated on regression/missing response.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured between theory and intuition; uses MEPS and Figure 1 effectively to motivate the problem.
- Value: ⭐⭐⭐⭐ Provides a practical and grounded solution for reliable uncertainty quantification under corrupted labels and missing test features.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)
- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICML 2026\] Enhancing Conformal Prediction via Class Similarity](../../ICML2026/learning_theory/enhancing_conformal_prediction_via_class_similarity.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)

</div>

<!-- RELATED:END -->
