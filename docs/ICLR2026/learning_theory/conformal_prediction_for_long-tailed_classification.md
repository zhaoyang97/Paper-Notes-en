---
title: >-
  [Paper Note] Conformal Prediction for Long-Tailed Classification
description: >-
  [ICLR2026][Learning Theory][Long-tailed Classification] To address the dilemma in long-tailed classification where prediction sets are either small but miss rare classes or have good coverage but are excessively large, this paper proposes two conformal prediction methods that maintain marginal coverage guarantees: a new scoring function, PAS (Prevalence-Adjusted Softmax, which optimally trades off set size and macro-coverage), and a new procedure…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Uncertainty Quantification"
  - "Conformal Prediction"
  - "Long-tailed Classification"
  - "Class-conditional Coverage"
  - "Prediction Sets"
  - "Coverage-size Trade-off"
date: 2026-05-08
content_hash: 4a2fda768df701d3
---

# Conformal Prediction for Long-Tailed Classification

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=8L83ZbFDjk](https://openreview.net/forum?id=8L83ZbFDjk)  
**Code**: https://github.com/tiffanyding/long-tail-conformal  
**Area**: Learning Theory / Uncertainty Quantification / Conformal Prediction  
**Keywords**: Conformal Prediction, Long-tailed Classification, Class-conditional Coverage, Prediction Sets, Coverage-size Trade-off

## TL;DR
To address the dilemma in long-tailed classification where prediction sets are either small but miss rare classes or have good coverage but are excessively large, this paper proposes two conformal prediction methods that maintain marginal coverage guarantees: a new scoring function, PAS (Prevalence-Adjusted Softmax, which optimally trades off set size and macro-coverage), and a new procedure, INTERP-Q (Linearly interpolating Classwise and Standard quantile thresholds to slide along the trade-off). These methods significantly improve the trade-off between set size and class-conditional coverage on Pl@ntNet-300K (1081 classes) and iNaturalist-2018 (8142 classes).

## Background & Motivation
**Background**: Conformal Prediction (CP) is a class of distribution-free uncertainty quantification methods that augment point predictions with "prediction sets." Instead of outputting a single potentially incorrect label, CP provides a **set likely to contain the true label**, allowing users (e.g., plant identification app users) to verify candidates. The core of CP involves choosing a threshold $q$ for a scoring function $s(x,y)$ using a calibration set to guarantee coverage. **Standard CP** only guarantees **marginal coverage** $P(Y\in C(X))\ge 1-\alpha$, while **Classwise CP** (an instance of Mondrian conformal) guarantees $P(Y\in C(X)\mid Y=y)\ge 1-\alpha$ for every class.

**Limitations of Prior Work**: Many real-world classification tasks (plant/animal identification, medical diagnosis) are **extreme long-tailed**, where common classes have thousands of images while rare classes have very few. Crucially, stakeholders often **care more about rare classes** (endangered species, rare malignant tumors), yet most rare class samples are used for training, leaving few or zero holdout samples for calibration. In this setting, existing CP methods force a binary choice: Standard CP yields **small sets but poor rare-class coverage** (e.g., 421 out of 1081 classes in Pl@ntNet have coverage below 50%), while Classwise CP ensures good coverage for all classes but produces **sets too large to be usable** (averaging 780 candidates). Clustered CP reverts to Standard CP for rare classes that cannot be clustered.

**Key Challenge**: In long-tailed settings, there is an inherent trade-off between **small set size** and **class-conditional coverage**. The scarcity of calibration samples for rare classes makes "per-class coverage guarantees" nearly infeasible.

**Goal**: To provide a tunable "set size $\leftrightarrow$ class-conditional coverage" trade-off curve that is superior to both Standard and Classwise CP, while **maintaining marginal coverage guarantees**.

**Key Insight**: The authors approach this from two angles. First, by **relaxing the objective**—instead of insisting on per-class coverage, they optimize "average class-conditional coverage" (macro-coverage) and theoretically derive the optimal prediction set shape for the set size vs. macro-coverage trade-off. Second, by **backing off from Classwise CP**—softening the strict Classwise objective towards Standard CP using an interpolation parameter, allowing users to select a point on the trade-off curve.

**Core Idea**: Use prevalence-adjusted $\hat p(y\mid x)/\hat p(y)$ instead of $\hat p(y\mid x)$ for scoring (PAS) to optimize macro-coverage for a fixed set size. Simultaneously, use quantile linear interpolation (INTERP-Q) to connect Classwise and Standard CP into a tunable curve.

## Method

### Overall Architecture
All methods follow the same CP meta-algorithm (Algorithm 1): prediction sets are defined as **threshold sets** $C(X;q)=\{y\in\mathcal{Y}: s(X,y)\le q_y\}$, where a threshold $q_y$ is applied to the score of each class $y$. The methods differ only in the **score function $s$** and the **threshold function $\hat q$**.

- Standard CP: Arbitrary score + uniform threshold across classes $\hat q_{\text{STAND}}=(\hat q,\dots,\hat q)$, where $\hat q$ is the $(1-\alpha)$ empirical quantile of all calibration scores, yielding marginal coverage $1-\alpha$.
- Classwise CP: Arbitrary score + per-class thresholds $\hat q_y^{\text{CW}}$ (quantiles calculated using only $n_y$ calibration points for class $y$), yielding per-class coverage $1-\alpha$.
- **APPROACH I (Ours)**: Modify the score function. Maintain the Standard thresholding method but replace the softmax score with **PAS / WPAS** to optimize (weighted) macro-coverage without losing marginal coverage.
- **APPROACH II (Ours)**: Modify the threshold function. Keep the softmax score but use a **linear interpolation** of Standard and Classwise thresholds $\hat q^{\text{IQ}}$, resulting in the **INTERP-Q** family.

The baseline score uses the softmax score $s_{\text{softmax}}(x,y)=1-\hat p(y\mid x)$ (i.e., the LAC score), where smaller scores represent higher confidence.

### Key Designs

**1. PAS Score: Adjusting softmax with class prevalence to shift the "Optimal Set" from marginal to macro-coverage**

The pain point is that Standard CP optimizes for "maximizing marginal coverage given a fixed set size." Marginal coverage is the sum of class-conditional coverage **weighted by class prevalence** $\text{MarginalCov}(C)=\sum_y p(y)\,\text{CondCov}(C,y)$, which places most weight on high-frequency classes, systematically ignoring rare classes. This paper focuses on **macro-coverage**, the **unweighted** average of class-conditional coverage $\text{MacroCov}(C)=\frac{1}{|\mathcal{Y}|}\sum_y \text{CondCov}(C,y)$.

The authors solve a population optimization problem: minimize expected set size under a macro-coverage constraint (and its dual). Proposition 1 shows the optimal set thresholds the **density ratio**:

$$C^*(x)=\{y\in\mathcal{Y}: p(y\mid x)/p(y)\ge t\}.$$

The key insight: while the classic "minimum set size + marginal/classwise coverage" solution thresholds $p(y\mid x)$ (Neyman-Pearson), the **macro-coverage** optimal solution thresholds $p(y\mid x)/p(y)$—posterior divided by class prevalence. Intuitively, since $p(y)$ is small for rare classes, this division "boosts" their scores, making them more likely to enter the set. In practice, true densities are replaced by estimates $\hat p(y\mid x)$ (classifier softmax) and $\hat p(y)$ (empirical distribution of training labels), defining:

$$s_{\text{PAS}}(x,y)=-\hat p(y\mid x)/\hat p(y),$$

where PAS stands for prevalence-adjusted softmax. By plugging this into the Standard CP threshold $\hat q$, the prediction set $\hat C(x)=\{y:\hat p(y\mid x)/\hat p(y)\ge t\}$ **inherits the marginal coverage guarantee of Standard CP** while (approximately) optimally trading off set size and macro-coverage. Note: PAS improves this trade-off but does not provide a direct macro-coverage guarantee.

**2. WPAS: Generalizing "Equal-weighted Macro-coverage" to "Weighted Macro-coverage" to protect specific classes**

While macro-coverage treats classes equally, users often prioritize specific classes (e.g., endangered species). Given user weights $\omega(y)$ summing to 1, weighted macro-coverage is defined as $\text{MacroCov}_\omega(C)=\sum_y \omega(y)\,\text{CondCov}(C,y)$. Taking $\omega(y)=|\mathcal{Y}|^{-1}$ yields macro-coverage; $\omega(y)=p(y)$ yields marginal coverage. Proposition 2 shows the optimal set thresholds $\omega(y)\cdot p(y\mid x)/p(y)$, leading to:

$$s_{\text{WPAS}}(x,y)=-\omega(y)\,\hat p(y\mid x)/\hat p(y),$$

which also uses the Standard threshold. In endangered species experiments, setting the weight of critical species to $\lambda\ge 1$ times that of common species increases their class-conditional coverage with only a **modest increase** in average set size and minimal impact on other classes.

**3. INTERP-Q: Linearly interpolating Classwise and Standard quantiles to slide the trade-off curve**

APPROACH II modifies thresholds instead of scores. INTERP-Q ("interpolated quantile") defines the threshold for each class as a weighted average of the Global Standard threshold $\hat q$ and the Classwise threshold $\hat q_y^{\text{CW}}$:

$$\hat q_y^{\text{IQ}}=\tau\,\hat q_y^{\text{CW}}+(1-\tau)\,\hat q,\quad \tau\in[0,1].$$

For classes with too few samples where $\hat q_y^{\text{CW}}=\infty$, the value is replaced by 1 (the maximum softmax score) before interpolation. $\tau=0$ is Standard CP; $\tau=1$ is Classwise CP. Proposition 3 provides a marginal coverage lower bound of $1-2\alpha$, which is theoretically near-tight. However, in empirical tests, real-world data does not exhibit the pathological score distributions required to reach this bound, so INTERP-Q empirical coverage remains near $1-\alpha$.

The utility lies in a **non-linear** phenomenon: linear interpolation of thresholds does not lead to linear interpolation of set sizes. At $\tau=1$, sets are enormous (780 on Pl@ntNet; 7430 on iNaturalist), but dropping $\tau$ to 0.99 causes the average set size to plummet to 7.6 and 55.8, respectively. This occurs because rare class softmax scores are highly skewed toward 1, making quantiles extremely sensitive to $\tau$. This allows users to flexibly choose a position on the "small set $\leftrightarrow$ class-conditional coverage" curve while maintaining marginal coverage.

### A Complete Example
Using Pl@ntNet-300K with a target marginal coverage of 90%:
- **Standard CP** produces small sets (average size 1.57) but **421 out of 1081** plant species have coverage below 50% (rare classes are systematically missed).
- **Classwise CP** reduces "classes with coverage < 50%" to **0**, but the average set size explodes to **780** (unusable for manual verification).
- **Standard + PAS** (Ours) yields an average set size of **2.57** (only slightly larger than Standard) while cutting the number of classes with coverage < 50% **by half to 180**.
- **INTERP-Q** (Ours) behaves similarly, with the added benefit of being able to slide the trade-off by adjusting $\tau$.

## Key Experimental Results

### Main Results
Datasets: Pl@ntNet-300K (1081 classes), iNaturalist-2018 (8142 classes). Base model: ResNet-50. Calibration set: 70% of validation data. 

| Metric | Definition | Direction |
|--------|------|------|
| FracBelow50% | Proportion of classes with coverage $\le 50\%$ | Lower is better |
| UnderCovGap | Mean under-coverage gap $\frac{1}{\|\mathcal{Y}\|}\sum_y \max(1-\alpha-\hat c_y,0)$ | Lower is better |
| MacroCov | Average class-conditional coverage $\frac{1}{\|\mathcal{Y}\|}\sum_y \hat c_y$ | Higher is better |
| MarginalCov | Marginal coverage (must be $\ge 1-\alpha$) | Threshold met |
| Average set size | Mean number of elements in $C(X)$ | Lower is better |

Pl@ntNet-300K (Target 90% Marginal Coverage):

| Method | Avg Set Size | Classes with Cov < 50% (/1081) | Note |
|--------|------|----------|------|
| Standard | 1.57 | 421 | Smallest set, but rare classes fail |
| Classwise | 780 | 0 | Perfect per-class coverage, unusable set |
| **Standard + PAS** | **2.57** | **180** | Minimal size increase, rare class coverage drastically improved |
| **INTERP-Q** | Tunable via $\tau$ | Tunable via $\tau$ | Slides the trade-off via one parameter |

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| INTERP-Q, $\tau=1$ | Size 780 / 7430 | Equivalent to Classwise, massive sets |
| INTERP-Q, $\tau=0.99$ | Size 7.6 / 55.8 | Slight threshold change results in huge size reduction (non-linear) |
| Standard + WPAS ($\lambda\uparrow$) | Targeted Cov↑ | Increasing target weights boosts coverage for specific classes with moderate size cost |
| Standard + PAS (Pareto) | Optimal across $\alpha$ | No method achieves smaller size and better macro-coverage simultaneously |

### Key Findings
- **Direct optimization is more effective**: Adjusting $\alpha$ in Standard CP to indirectly improve class-conditional coverage is sub-optimal; explicitly optimizing macro/class-conditional coverage is consistently better.
- **Avoid Classwise CP in long-tail settings**: Ours achieves comparable class-conditional/macro coverage with much smaller sets.
- **Standard + PAS is Pareto optimal**: It is a simple and powerful baseline that no other method dominates in the size-conditional coverage space.
- **INTERP-Q non-linearity is useful**: Changing $\tau$ from 1 to 0.99 drops set size by orders of magnitude due to the extreme skew of softmax scores for rare classes.

## Highlights & Insights
- **Switching objectives with a single division**: Dividing the posterior by the class prevalence ($\hat p(y\mid x)/\hat p(y)$) shifts the CP objective from marginal to macro-coverage. This has theoretical Neyman-Pearson optimality yet requires only one extra line of code.
- **Decoupling Guarantees from Optimization**: PAS changes the score to optimize an objective while reusing the Standard threshold to "piggyback" on marginal coverage guarantees. This paradigm is transferable to any CP task requiring new coverage objectives.
- **Threshold interpolation as a simple knob**: INTERP-Q does not require redesigned scores or retrained models; it uses a linear combination of existing thresholds to provide a continuous range of options.
- **Targeted WPAS**: Encoding "I care more about endangered species/rare cancers" directly into the score provides a clean interface for injecting domain priorities into uncertainty quantification.

## Limitations & Future Work
- **PAS lacks direct macro-coverage guarantees**: It only guarantees marginal coverage; the macro-coverage improvement depends on the quality of $\hat p(y\mid x)$ and $\hat p(y)$ estimates.
- **INTERP-Q theoretical gap**: The worst-case marginal coverage is $1-2\alpha$. Although empirical results are near $1-\alpha$, this depends on the assumption that real-world score distributions are not pathological.
- **Evaluation on truncated sets**: Long-tail test sets are also long-tailed, making it hard to reliably estimate rare-class coverage. The authors use truncated versions (100 samples per class) for evaluation, which may differ from deployment distributions.
- **Future Directions**: Combining the two approaches, designing direct macro-coverage guarantees for PAS, and studying the theoretical impact of prevalence estimation errors.

## Related Work & Insights
- **vs. Standard CP**: Standard CP thresholds $p(y\mid x)$ for minimum size under marginal coverage; PAS thresholds $p(y\mid x)/p(y)$ for macro-coverage, greatly improving rare class performance.
- **vs. Classwise CP (Mondrian)**: Classwise guarantees coverage per class but yields huge sets in long-tailed data; INTERP-Q softens this via interpolation to achieve manageable set sizes.
- **vs. Clustered CP**: Clustered depends on grouping classes with similar score distributions; the proposed methods do not rely on clustering and work for extremely rare classes.
- **vs. APS / RAPS / SAPS**: These focus on X-conditional coverage and generally produce larger sets; this paper uniquely targets the under-studied class-conditional coverage in long-tailed settings.

## Rating
- Novelty: ⭐⭐⭐⭐ Deriving density-ratio optimal sets for macro-coverage and the threshold interpolation process is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple large-scale datasets and baselines, though dependent on truncated test sets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, logical progression from theory to method, and intuitive examples.
- Value: ⭐⭐⭐⭐ Highly practical tools for long-tailed uncertainty quantification; PAS is trivial to implement and deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Adaptive Conformal Prediction via Mixture-of-Experts Gating Similarity](adaptive_conformal_prediction_via_mixture-of-experts_gating_similarity.md)
- [\[ICML 2026\] Enhancing Conformal Prediction via Class Similarity](../../ICML2026/learning_theory/enhancing_conformal_prediction_via_class_similarity.md)
- [\[ICLR 2026\] Softmax is not Enough (for Adaptive Conformal Classification)](softmax_is_not_enough_for_adaptive_conformal_classification.md)

</div>

<!-- RELATED:END -->
