---
title: >-
  [Paper Note] Robust Decision Making with Partially Calibrated Forecasts
description: >-
  [ICLR 2026][Learning Theory][Calibration] When a predictor satisfies only "partial calibration" (weaker than full calibration), this paper characterizes the optimal decision rule through a minimax robust lens. The optimal rule is an optimal response to the "worst-case distribution allowed by the calibration constraints." Furthermore, it proves that once calibration strength reaches "decision calibration"—a **computable weak condition**—the optimal robust rule collapses into "…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Decision Theory"
  - "Calibration"
  - "Decision Calibration"
  - "Minimax Robust Decision Making"
  - "H-Calibration"
  - "Dual Characterization"
date: 2026-05-08
content_hash: 5fe204292bd58fcb
---

# Robust Decision Making with Partially Calibrated Forecasts

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=mHRuCmc9lo](https://openreview.net/forum?id=mHRuCmc9lo)  
**Area**: Learning Theory / Decision Theory  
**Keywords**: Calibration, Decision Calibration, Minimax Robust Decision Making, H-Calibration, Dual Characterization

## TL;DR
When a predictor satisfies only "partial calibration" (weaker than full calibration), this paper characterizes the optimal decision rule through a minimax robust lens. The optimal rule is an optimal response to the "worst-case distribution allowed by the calibration constraints." Furthermore, it proves that once calibration strength reaches "decision calibration"—a **computable weak condition**—the optimal robust rule collapses into "trust the prediction and play the direct optimal response," perfectly aligning with the semantics of full calibration.

## Background & Motivation

**Background**: In trustworthy machine learning, calibration is a core objective, largely due to its clean decision-theoretic semantics. If a predictor $f$ is **fully calibrated**—meaning $E[Y\mid f(X)=v]=v$ for any predicted value $v$—then regardless of the underlying distribution or the decision-maker's utility function, the **optimal strategy** among all "prediction-to-action" mappings is to "trust the prediction as truth." This is the plug-in optimal response $a_{\mathrm{BR}}(f(x))=\arg\max_{a}u(a,f(x))$, which formulates the fundamental justification for why calibration is "trustworthy."

**Limitations of Prior Work**: Full calibration is only feasible for low-dimensional outputs. When the output $Y\in[0,1]^d$ is high-dimensional (e.g., multi-class classification), the sample complexity required to calibrate an existing predictor without damaging its accuracy grows **exponentially** with dimension $d$. In practice, models from neural networks to LLMs systematically deviate from calibration. To bypass this, literature proposes various **weaker, more computable** forms of calibration (e.g., top-label calibration, decision calibration). However, these **lose the decision-theoretic guarantee** of full calibration; under weak calibration, the decision-maker lacks a principled way to determine how much to trust the prediction.

**Key Challenge**: Decision-makers face two extremes. One is "aggressive": treating the prediction as truth and playing the plug-in optimal response. The other is "conservative": completely distrusting the prediction and playing a fixed minimax safe strategy $\arg\max_a\min_y u(a,y)$, which treats the prediction as zero-information. While full calibration makes the former optimal, it is unattainable; a principled compromise between these extremes for weak calibration remains missing.

**Goal**: Given a predictor guaranteed to satisfy some form of weak calibration, how should a **conservative** decision-maker map predictions to actions to maximize expected utility under the worst-case scenario among all distributions compatible with said calibration guarantee?

**Key Insight**: View the prediction $f(x)$ as a **constraint** on the "true instance-conditional output distribution." Upon observing $f(x)$, the decision-maker considers an **ambiguity set** $Q$ of all "candidate realities" compatible with the prediction and the calibration guarantee. The "volume" of this set is determined by the calibration strength: $Q$ collapses to the prediction itself under full calibration and expands as calibration weakens. The decision rule should **adaptively adjust its degree of conservatism** based on $Q$.

**Core Idea**: This paper re-envisions decision-making under partially calibrated forecasts through a minimax robust optimization lens—performing an optimal response against the worst-case candidate reality allowed by the calibration constraints, thereby building a principled bridge between "complete trust" and "total conservatism."

## Method

### Overall Architecture

This paper does not study "how to train a predictor for calibration," but rather "how to decide given a calibration guarantee." The logic follows three layers: first, parameterize calibration strength using a family of **test functions $H$**, unifying full calibration and decision calibration as special cases of $H$-calibration; second, formulate decision-making with partially calibrated forecasts as a minimax problem and provide a closed-form dual characterization of the optimal robust rule; third, specialize this characterization to specific $H$ classes to derive meaningful conclusions (notably the "collapse" result under decision calibration).

Formal Setup: $(X,Y)\sim D$, where $X$ represents features and $Y\in[0,1]^d$ represents outputs. $A$ is the action set and $u(a,y)$ is the utility. The decision-maker uses a strategy $a(\cdot):[0,1]^d\to A$ to map predictions to actions, with performance measured by $E_{(X,Y)\sim D}[u(a(f(X)),Y)]$. A key assumption is that the **utility $u(a,v)$ is linear** in the second variable $v$ (naturally satisfied in multi-class, risk-neutral settings where $u(a,v)=\sum_k v_k\,U(a,k)$).

Given an $H$-calibrated predictor $f$, define the **ambiguity set**

$$Q=\Big\{q:[0,1]^d\to[0,1]^d \ \Big|\ E\big[h(f(X))\cdot(q(f(X))-f(X))\big]=0,\ \forall h\in H\Big\},$$

where $q(v):=E[Y\mid f(X)=v]$ is the true conditional expectation. $Q$ collects all candidate conditional expectations compatible with the fact that "$f$ is $H$-calibrated." The **robust decision rule** maximizes the expected utility in the worst case over $Q$:

$$a_{\mathrm{robust}}(\cdot)=\arg\max_{a(\cdot)}\ \min_{q\in Q}\ E\big[u(a(f(X)),q(f(X)))\big].$$

This rule possesses an **interpolation property**: when $H$ contains all functions, $Q=\{q(v)=v\}$ and $a_{\mathrm{robust}}$ reduces to the optimal response $a_{\mathrm{BR}}$. When $H$ is empty, $Q$ contains all functions and the rule collapses to a constant minimax safe strategy. As $H$ becomes richer, $Q$ shrinks, and conservatism decreases.

### Key Designs

**1. H-Calibration: Using Test Functions as Continuous Knobs for Calibration Strength**

Weak calibrations often lack a unified decision-theoretic interpretation because they are defined disparately. This paper unifies them: $f$ is $H$-calibrated if and only if for every test function $h\in H$:

$$E\big[h(f(X))\cdot(Y-f(X))\big]=0.$$

The intuition is that the prediction residual $Y-f(X)$ is **uncorrelated** with any "test" $h(f(X))$ in $H$. When $H$ includes all bounded measurable functions, this is equivalent to full calibration. Small $H$ implies weak constraints. This framework allows "calibration strength" to be adjusted continuously via the complexity of $H$, subsuming concepts like top-label and decision calibration.

**2. Dual Characterization: Optimal Robust Rule = Optimal Response to an "Adversarially Distorted Distribution"**

Solving the minimax problem directly is typically difficult. By restricting $H=\mathrm{span}\{h_1,\dots,h_k\}$ to be finite-dimensional, $H$-calibration becomes $k$ linear moment constraints. Theorem 3.1 provides a dual closed-form structure for the saddle point $(a_{\mathrm{robust}},q^\star)$: there exist multipliers $\lambda^\star=(\lambda_1^\star,\dots,\lambda_k^\star)$ such that for almost every prediction $v=f(x)$, the worst-case distribution is

$$q^\star(v)\in\arg\min_{p\in[0,1]^d}\Big\{\mathrm{val}(p)+p\cdot\sum_{i=1}^k h_i(v)\lambda_i^\star\Big\},\qquad \mathrm{val}(p)=\max_{a\in A}u(a,p),$$

and the optimal robust action is the optimal response to it: $a_{\mathrm{robust}}(v)\in\arg\max_{a}u(a,q^\star(v))$. This implies the optimal strategy is **always an optimal response**, not to the original forecast $f(x)$, but to an **adversarially distorted distribution** $q^\star$ with a distortion $s^\star(v)=\sum_i h_i(v)\lambda_i^\star$. This is **point-wise computable**, requiring only low-dimensional optimization at each $v$.

**3. Sharp Phase Transition in Decision Calibration: Weak Conditions Restore "Complete Trust"**

One might expect the robust rule to **gradually** shift from "completely conservative" to "optimal response" as $H$ grows. This paper proves a **sharp phase transition**. For a fixed problem, define the decision regions $R_a=\{v:u(a,v)\ge u(a',v)\ \forall a'\}$ and the **decision calibration class** $H_{\mathrm{dec}}=\{\mathbf 1_{R_a}:a\in A\}$. Theorems 4.1/4.2 show that if $f$ is $H_{\mathrm{dec}}$-calibrated, the adversarial distortion **entirely vanishes** ($q^\star(v)=v$ a.e.), and the rule **collapses back to the plug-in optimal response** $a_{\mathrm{robust}}(v)=\arg\max_a u(a,v)$.

Mechanism: Decision calibration ensures the expected utility of $a_{\mathrm{BR}}$ is **invariant** to the adversary's choices within $Q$. The adversary cannot weaken $a_{\mathrm{BR}}$, making its worst-case utility equal to its nominal utility, thus ensuring minimax optimality. This upgrades decision calibration from "no swap regret" to full minimax optimality, requiring a test class of size only $|A|$ (number of actions), which is much more computable than full calibration.

**4. "Free" Calibration from Training Pipelines: MSE Self-Orthogonality and Binning**

Many $H$-calibrations are **structurally satisfied for free**. **Self-orthogonality in MSE** (Prop 4.4): a model $f_\theta(X)=Wz_\phi(X)$ trained to a first-order stationary point of Mean Squared Error automatically satisfies $E[f_\theta(X)(Y-f_\theta(X))^\top]=0$, meaning it is $H$-calibrated for $H=\{h(v)=v\}$. Similarly, **binning calibration** (Prop 4.5) used in post-processing (e.g., histogram binning) produces a worst-case distribution $q^\star(v)$ that is constant per bin. These allow the framework to be applied to existing models without intervention.

### Loss & Training

The paper focuses on how to make decisions rather than training. However, it notes that since MSE with a linear head ensures self-orthogonality (Prop 4.4), standard regression models already satisfy the necessary constraints to apply the robust rule. The robust strategy's multipliers are estimated using an independent calibration split.

## Key Experimental Results

Experiments compare the plug-in optimal response $a_{\mathrm{BR}}$ and the robust rule $a_{\mathrm{robust}}$ on two regression datasets with $H=\{h(v)=v\}$. Evaluations cover **nominal performance** (i.i.d. test set) and **adversarial performance** (the worst-case distribution compatible with the $H$-calibration constraint).

- **Bike Sharing (UCI)**: $Y$ is normalized daily rentals; $A=\{0.8, 1.0, 1.2\}$ (capacity multipliers).
- **California Housing**: $Y$ is normalized house prices; $A=\{0.6, 0.75, 0.90\}$ (investment multipliers).

### Main Results

Average utility on test set (adversary respects $H$-calibration):

| Dataset | i.i.d. Plug-in | i.i.d. Robust | Worst·Plug-in | Worst·Robust | Plug-in Worst·Plug-in | Plug-in Worst·Robust |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Bike Sharing (UCI) | 0.474 | 0.463 | 0.402 | **0.410** | 0.393 | **0.412** |
| California Housing | 0.216 | 0.207 | 0.160 | **0.164** | 0.155 | **0.166** |

### Key Findings
- Results align with the theory: Under **calibration-preserving distribution shifts**, the robust rule outperforms the plug-in response.
- The cost of robustness under i.i.d. conditions is **very mild** (e.g., 0.474 to 0.463 in Bike Sharing).
- The robust rule never performs worse than the plug-in rule on its own worst-case distribution, validating the saddle-point properties.
- Since $H=\{h(v)=v\}$ is satisfied for any MSE-trained model with a linear head, this robust rule is effectively "free" to implement.

## Highlights & Insights
- **Unified Lens via Minimax**: The framework unifies "complete trust" and "total conservatism" as endpoints on a spectrum parameterized by $H$, answering "how much to trust" weak forecasts on principled grounds.
- **Sharp Phase Transition**: It is surprising that conservatism does not scale linearly with calibration; instead, once the $|A|$ basic decision constraints are met, the optimal response becomes fully optimal instantly.
- **Upgraded Semantics**: It elevates decision calibration from "no swap regret" (excluding improvements of the form $\phi(a_{\mathrm{BR}})$) to minimax optimality (excluding **any** strategy improvement).
- **Transferable Trick**: Identifying structural properties like MSE self-orthogonality as "free" moment constraints for robust optimization is a powerful motif for any downstream task requiring distribution robustness.

## Limitations & Future Work
- **Risk Neutrality**: The framework currently requires utilities $u(a,v)$ to be linear in $v$. Risk-averse utilities sensitive to variance fall outside this scope.
- **Predictor Profiling**: The decision-maker must know which $H$-calibration the predictor satisfies.
- **Scale**: Experiments are limited to small-scale regression and 1D outputs. Scalability in high-dimensional multi-class settings remains to be empirically tested.
- **Future Work**: Extending the framework to non-linear utilities via basis expansion and visualizing adversarial distortions $s^\star(v)$ as explainable signals for conservatism.

## Related Work & Insights
- **vs. Rothblum & Yona (2023)**: While they study regret minimization under calibration, they are restricted to 1D binary outputs and near-full calibration. This paper handles **qualitatively weaker** calibration necessary for high dimensions.
- **vs. Zhao et al. (2021) / Noarov et al. (2023)**: These works provide regret bounds for plug-in rules under weak calibration. This paper proves that decision calibration is actually sufficient for minimax optimality, a stronger semantic guarantee.
- **vs. Robust Optimization (DRO)**: This is the first work to apply minimax robust decision techniques specifically to the posterior decision problem of "high-dimensional partially calibrated forecasts."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Unifies partial calibration via minimax; "sharp phase transition" is a deep insight)
- Experimental Thoroughness: ⭐⭐⭐ (Solid theory, but experiments are on small datasets with 1D outputs)
- Writing Quality: ⭐⭐⭐⭐⭐ (Excellent logical flow and clear visual intuition of the theory)
- Value: ⭐⭐⭐⭐ (Provides a computable answer for trustworthy decision-making and establishes decision calibration as a critical design target)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Decision Making with Generative Action Sets](online_decision_making_with_generative_action_sets.md)
- [\[ICLR 2026\] Trained on Tokens, Calibrated on Concepts: The Emergence of Semantic Calibration in LLMs](trained_on_tokens_calibrated_on_concepts_the_emergence_of_semantic_calibration_i.md)
- [\[ICLR 2026\] Online Decision-Focused Learning](online_decision-focused_learning.md)
- [\[ICLR 2026\] Conformalized Decision Risk Assessment](conformalized_decision_risk_assessment.md)
- [\[ICLR 2026\] Noise Tolerance of Distributionally Robust Learning](noise_tolerance_of_distributionally_robust_learning.md)

</div>

<!-- RELATED:END -->
