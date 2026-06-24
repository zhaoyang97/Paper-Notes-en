---
title: >-
  [Paper Note] On the Bayes Inconsistency of Disagreement Discrepancy Surrogates
description: >-
  [ICLR 2026][Learning Theory][Disagreement discrepancy] This paper proves that existing surrogate losses used for "disagreement discrepancy" are **not Bayes consistent** in multiclass ($K>2$) settings—optimizing the surrogate does not necessarily optimize the true objective. Accordingly, it designs a new disagreement loss $-\log(1-\sigma(s)_y)$, which, combined with cross-entropy, yields the **first provably Bayes consistent** surrogate. This surrogate is more reliable for dow…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Distribution Shift Generalization"
  - "Disagreement discrepancy"
  - "surrogate loss"
  - "Bayes consistency"
  - "distribution shift"
  - "error bounds"
date: 2026-05-08
content_hash: b855dc9cd41ab61f
---

# On the Bayes Inconsistency of Disagreement Discrepancy Surrogates

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=VwCyRQJ51H](https://openreview.net/forum?id=VwCyRQJ51H)  
**Code**: Reuses experimental code from Rosenfeld & Garg (2023); no independent repository provided.  
**Area**: Learning Theory / Distribution Shift Generalization  
**Keywords**: Disagreement discrepancy, surrogate loss, Bayes consistency, distribution shift, error bounds

## TL;DR
This paper proves that existing surrogate losses used for "disagreement discrepancy" are **not Bayes consistent** in multiclass ($K>2$) settings—optimizing the surrogate does not necessarily optimize the true objective. Accordingly, it designs a new disagreement loss $-\log(1-\sigma(s)_y)$, which, combined with cross-entropy, yields the **first provably Bayes consistent** surrogate. This surrogate is more reliable for downstream tasks like error bound estimation and harmful shift detection.

## Background & Motivation

**Background**: To evaluate and improve model reliability under distribution shift, an emerging approach is "disagreement discrepancy." It measures how much the disagreement between a **critic model** $f$ and a **reference model** $h$ changes from a source distribution $S$ to a target distribution $T$. Maximizing this quantity over the critic is used to: **upper bound** model error on unlabeled target data (Rosenfeld & Garg, 2023), construct **statistical tests for harmful shifts** (Ginsberg et al., 2023, Detectron), and train **more robust ensembles** (Pagliardini et al., 2023, D-BAT).

**Limitations of Prior Work**: The true disagreement discrepancy objective involves the **zero-one loss** $\ell_{zo}(y,y')=\mathbb{1}_{y\neq y'}$, which is non-differentiable and cannot be optimized directly via gradients. Consequently, existing methods substitute it with differentiable **surrogate losses**—cross-entropy is uniformly used for agreement, while disagreement losses vary (RG23 uses $\ell^{RG}_{dis}$, Ginsberg uses $\ell^{GLK}_{dis}$). However, a critical question ignored by this research line is: **does optimizing the surrogate objective truly equate to optimizing the real disagreement discrepancy?** This is not a hypothetical concern—Mishra & Liu (2025) reported training instabilities, suggesting current surrogates may be inadequate.

**Key Challenge**: Classical surrogate consistency theories (Zhang 2004; Bartlett et al. 2006) are established for a **single risk** (classification risk on one distribution). Disagreement discrepancy, however, is a **difference of risks** over two different distributions using different losses: $\alpha R[\ell_{zo},h_2,Af](T)-R[\ell_{zo},h_1,Af](S)$. These risks are **coupled** during optimization, meaning existing theories cannot be applied to each term independently. Thus, whether these surrogates are consistent remained unanswered.

**Goal**: (1) Extend the consistency framework from "single-risk classification" to the "difference of risks" objective; (2) determine if existing surrogates are Bayes consistent; (3) if inconsistent, design a consistent surrogate.

**Key Insight**: Utilizing a pseudo-loss to **decouple the "difference of risks" into a sum of two disjoint risks**, enabling pointwise analysis of the optimal critic. This machinery is used to both prove lower bounds (showing existing surrogates are inconsistent) and upper bounds (showing the new surrogate is consistent), deriving a new disagreement loss based on the criterion of "pointwise alignment with the zero-one loss."

## Method

### Overall Architecture

The paper addresses whether surrogate optimization is faithful to the true objective through three steps. **Step one** expresses the generalized disagreement discrepancy (Definition 1) as:

$$d_\alpha[h,f](S,T) = \alpha R[\ell_{zo},h_2,Af](T) - R[\ell_{zo},h_1,Af](S)$$

Where $Af(x)=\arg\max_i f(x)_i$ (with tie-breaking by minimum index) and $\alpha>0$ balances the terms. RG23 sets $h_1=h_2$ and $\alpha=1$, while Ginsberg sets $h_1$ as the true label function, $h_2$ as the model under test, and $\alpha\approx 1/N$. The surrogate objective is Definition 2: $\hat d_\alpha=R[\ell_{agr},h_1,f](S)+\alpha R[\ell_{dis},h_2,f](T)$, designed for **minimization** (the inverse of maximizing the true objective).

**Step two** involves a critical mathematical reformulation: rewriting this "difference" as a sum of two **pointwise disjoint** risks, allowing classical single-risk theory to be applied segment-wise. **Step three** uses this reformulated version for two opposing purposes: proving a **lower bound** on the optimality gap for existing surrogates to show they are inconsistent for $K>2$, and proving an **upper bound** for the proposed surrogate to show it is consistent. As a purely theoretical and empirical validation paper, the core lies in its proofs rather than a visual pipeline.

### Key Designs

**1. Decoupling the "risk difference" into a sum of disjoint risks via pseudo-loss**

The challenge is that disagreement discrepancy involves **subtraction** of risks across distributions and losses. The authors introduce source/target densities $p_S, p_T$ and split the input space based on which density is larger, constructing two loss functionals:

$$L_1[\ell_1,\ell_2](x,y,z)=\mathbb{1}_{p_S(x)\ge p_T(x)}\Big(\ell_1(x,y_1,z)+\tfrac{p_T(x)}{p_S(x)}\ell_2(x,y_2,z)\Big)$$

$$L_2[\ell_1,\ell_2](x,y,z)=\mathbb{1}_{p_S(x)<p_T(x)}\Big(\tfrac{p_S(x)}{p_T(x)}\ell_1(x,y_1,z)+\ell_2(x,y_2,z)\Big)$$

Thus $d_\alpha[h,f](S,T)=R[\ell_1,h,Af](S)+R[\ell_2,h,Af](T)$, where the pseudo-losses are $\ell_1=L_1[-\ell_{zo},\alpha\ell_{zo}]$ and $\ell_2=L_2[-\ell_{zo},\alpha\ell_{zo}]$ (and similarly for surrogates using $\ell_{agr}, \alpha\ell_{dis}$). This reformulation has a **vital** property: for any input $x$, **exactly one** of $\ell_1(x,\cdot)$ or $\ell_2(x,\cdot)$ is non-zero. Density ratios act as weights to "flatten" the distributions into one integral, while the indicator ensures terms do not overlap, allowing for **pointwise optimization** of $f$.

**2. Proving prior surrogates are not Bayes consistent via optimality-gap lower bounds**

With decoupling, the authors reverse Zhang’s (2004) upper bound framework to develop an optimality gap **lower bound** for single risks (Appendix A), then apply it to disagreement discrepancy (Theorem 4). The intuition is: there exists an input region where the **optimal critic of the surrogate is inconsistent with the optimal critic of the true objective**. Specifically, for $K>2$, on a restricted subspace

$$\mathcal{X}'=\Big\{x: h_1(x)=h_2(x),\ p_T(x)>0,\ \lambda+\delta\le \tfrac{p_S(x)}{\alpha p_T(x)}\le 1-\delta\Big\}$$

there exists a convex function $\zeta$ continuous at 0 such that:

$$\sup_{f'}d_\alpha[h,f'](S,T)-d_\alpha[h,f](S,T)\ \ge\ \zeta\big(\hat d_\alpha[\cdot](S|_{\mathcal{X}'},T|_{\mathcal{X}'})-\inf_{f'}\hat d_\alpha[\cdot]\big)$$

where $\zeta(0)=\frac{\delta}{1-\delta}\mathbb{1}_{S(\mathcal{X}')>0}+\alpha\delta\,\mathbb{1}_{T(\mathcal{X}')>0}$. Crucially, **$\zeta(0)$ can be strictly greater than 0**. As long as source or target has positive measure on $\mathcal{X}'$, even if the surrogate is optimized perfectly (gap of 0), a non-vanishing gap remains in the true objective. This violates the consistency condition in Definition 3; thus, Corollary 5 concludes that RG23 and Ginsberg surrogates are **not Bayes consistent** for $K>2$.

**3. Symmetric disagreement loss aligned with zero-one loss for the first consistent surrogate**

The root of inconsistency is the misalignment of optima. The authors derive a new disagreement loss (Eq. 9):

$$\ell^{Ours}_{dis}(x,y,s)=-\log\big(1-\sigma(s)_y\big)$$

The design is **strictly symmetric** to the cross-entropy agreement loss $-\log\sigma(s)_y$: while cross-entropy pushes $\sigma(s)_y$ toward 1 (encouraging agreement), this loss pushes $\sigma(s)_y$ toward 0 (encouraging disagreement). More importantly, it **only requires $\sigma(s)_y\to 0$ without prescribing how remaining probability is distributed**—matching the pointwise behavior of the true disagreement loss $-\mathbb{1}_{y\ne A(s)}$. In contrast, $\ell^{RG}_{dis}$ and $\ell^{GLK}_{dis}$ impose extra structure on other logits, causing the optimum to deviate. Using Zhang's framework and the decoupling, the authors prove (Theorem 6) that this surrogate is **Bayes consistent for all $K\ge 2$**.

### Loss & Training
The final surrogate objective is a combination of cross-entropy $\ell_{agr}=\ell_{ce}$ for agreement and $\ell^{Ours}_{dis}=-\log(1-\sigma(s)_y)$ for disagreement, minimized over the critic $f$. In experiments, the critic is implemented as an adjustable linear layer atop the frozen model $h$.

## Key Experimental Results

Validation focuses on two downstream tasks: error bounds under covariate shift and harmful shift detection.

### Main Results: Error Bound Estimation (Natural + Adversarial Target Data)

Replicating RG23's setup across 11 vision shift benchmarks and 5 training methods (130 "shift-model" pairs). Since the true maximum discrepancy is uncalculatable, the "maximum value attained by any surrogate" is used as a reference.

| Setup | Metric | Ours | Comparison | Conclusion |
|--------|------|------|----------|------|
| Natural Shift 130 pairs | Prop. of max discrepancy reached | ≈80% | RG23 (Rem.) | One-sided Wilcoxon $p=1.8\times10^{-11}$ |
| Bound Calibration | Observed vs Desired violation rate $\delta$ | Closer to $y=x$ | RG23 deviates more | Both high at small $\delta$ |
| Adversarial Target $f=0\%$ | Rank-1 Proportion | 87.5% | RG23 / GLK23 | $p=3.3\times10^{-4}$ |
| Adversarial Target $f=50\%$ | Rank-1 Proportion | 100% | RG23 / GLK23 | $p<6.0\times10^{-10}$ |

Underestimating true discrepancy results in error bounds that "look tighter but are invalid." Ours consistently recovers larger discrepancies, making the bounds more reliable.

### Ablation Study: Harmful Shift Detection (Detectron / UCI-HD)

Using the Detectron framework on UCI Heart Disease (keeping the original 5 classes). XGBoost is used as the critic with target sample sizes $N\in\{10,20,50\}$.

| $N$ | GLK23 AUC-ROC | Ours AUC-ROC |
|------|------|------|
| 10 | $0.821^{+0.025}_{-0.025}$ | $0.908^{+0.019}_{-0.017}$ |
| 20 | $0.913^{+0.016}_{-0.019}$ | $0.984^{+0.005}_{-0.006}$ |
| 50 | $0.995^{+0.003}_{-0.002}$ | $1.000^{+0.000}_{-0.000}$ |

Non-overlapping 95% confidence intervals demonstrate that theoretical consistency translates into higher statistical power.

### Key Findings
- Inconsistency only emerges when $K>2$: For binary classification, all three surrogates are equivalent, explaining why this went unnoticed in binary benchmarks.
- Adversarial target data is a harsher stress test: As the attack proportion increases, the lead of Ours becomes more pronounced, showing that surrogate failure is amplified under difficult distributions.
- Calibration rates are still high at small $\delta$ for both methods, suggesting that even with consistent surrogates, the assumptions of the downstream application (the error bound itself) may be violated.

## Highlights & Insights
- **Decoupling the "risk difference" into disjoint risks** is the lever for the paper: Using density weighting and indicators to ensure only one term is non-zero allows pointwise optimization. This trick could be reused for any objective involving a difference of risks (domain adaptation, invariance regularization).
- **A unified optimality-gap machine for both proof directions**: The lower bound disproves existing surrogates ($\zeta(0)>0$), while the upper bound proves the new one ($\xi(0)=0$). The symmetry is logically clean.
- **Symmetric design criterion**: $-\log(1-\sigma(s)_y)$ only lowers the target class probability, mirroring the behavior of $-\mathbb{1}_{y\ne A(s)}$. Inferring the surrogate from the pointwise optimal behavior of the true loss is a valuable heuristic.

## Limitations & Future Work
- The analysis is restricted to **Bayes consistency** (optimization over all measurable functions); $H$-consistency for restricted hypothesis classes (like deep nets) remains an open problem.
- Consistency is an **asymptotic** guarantee (infinite data/capacity); finite-sample guarantees require estimation error bounds, whereas this paper only addresses calibration error.
- Even if the surrogate is consistent, **downstream assumptions may fail**. Error bounds were still breached in adversarial experiments, suggesting caution in deployment.

## Related Work & Insights
- **vs Rosenfeld & Garg (2023)**: They used critics to bound target error with $\ell^{RG}_{dis}$; Ours proves this is inconsistent for $K>2$, potentially leading to invalid "tight" bounds.
- **vs Ginsberg et al. (2023) Detectron**: They used disagreement cross-entropy for testing; Ours shows this is also inconsistent and demonstrates higher power in multiclass settings.
- **vs Mishra & Liu (2025)**: They empirically reported training instability; Ours provides the theoretical root cause (inconsistency).
- **vs $H\Delta H$-divergence (Ben-David et al. 2010)**: Disagreement discrepancy is an operationalized single-critic version; Ours fills the gap regarding surrogate consistency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First expansion of consistency theory to "risk difference" objectives and first consistent surrogate for this task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of benchmarks and adversarial tests, though the critic class is relatively simple.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical narrative with well-motivated structures.
- Value: ⭐⭐⭐⭐⭐ Fundamental contribution fixing a theoretical flaw shared across multiple research lines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PAC-Bayes Bounds for Cumulative Loss in Continual Learning](pac-bayes_bounds_for_cumulative_loss_in_continual_learning.md)
- [\[ICML 2026\] Realizable Bayes-Consistency for General Metric Losses](../../ICML2026/learning_theory/realizable_bayes-consistency_for_general_metric_losses.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICLR 2026\] Why Ask One When You Can Ask k? Learning-to-Defer to the Top-k Experts](why_ask_one_when_you_can_ask_k_learning-to-defer_to_the_top-k_experts.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)

</div>

<!-- RELATED:END -->
