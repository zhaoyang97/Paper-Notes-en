---
title: >-
  [Paper Note] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications
description: >-
  [ICML2025][Learning Theory][Transductive learning] This paper proposes the Transductive Local Complexity (TLC) framework, extending classical local Rademacher complexity to the transductive learning setting. It achieves excess risk bounds that are almost consistent with inductive learning (differing only by logarithmic factors) and resolves a decade-old open problem.
tags:
  - "ICML2025"
  - "Learning Theory"
  - "Transductive learning"
  - "Generalization bounds"
  - "Local Rademacher complexity"
  - "VC dimension"
  - "Kernel learning"
  - "Concentration inequalities"
  - "Sampling without replacement"
date: 2026-05-08
content_hash: aeeea08ed2282f5f
---

# Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications

**Conference**: ICML2025  
**arXiv**: [2309.16858](https://arxiv.org/abs/2309.16858)  
**Code**: None  
**Area**: Learning Theory  
**Keywords**: Transductive learning, Generalization bounds, Local Rademacher complexity, VC dimension, Kernel learning, Concentration inequalities, Sampling without replacement

## TL;DR

This paper proposes the Transductive Local Complexity (TLC) framework, extending classical local Rademacher complexity to the transductive learning setting. It achieves excess risk bounds that are almost consistent with inductive learning (differing only by logarithmic factors) and resolves a decade-old open problem.

## Background & Motivation

In transductive learning, the learner has access to both labeled training data and unlabeled test data, aiming to predict the labels of the test data. Obtaining tight generalization bounds is a core problem in statistical learning theory.

**Classical Results in Inductive Learning**: Local Rademacher complexity (LRC) provides sharp excess risk bounds for empirical risk minimizers in inductive learning:

$$\mathcal{E}(\hat{f}) \leq \Theta\left(r^* + \frac{x}{n}\right)$$

where $r^*$ is the fixed point of a certain empirical process, and $n$ is the training sample size. LRC has achieved minimax rates in tasks like nonparametric regression and classification.

**Difficulties in Transductive Learning**: The best existing result [Tolstikhin et al., 2014] provides an excess risk bound of:

$$\mathcal{E}(\hat{f}) \leq \Theta\left(\frac{n}{u}r_m^* + \frac{n}{m}r_u^* + \frac{1}{m} + \frac{1}{u}\right)$$

where $m, u$ are the sizes of training and test data, respectively, and $n = m + u$. This bound can diverge when $m$ or $u$ is much smaller than $n$, showing a fundamental gap compared to inductive learning bounds. The root cause is the lack of effective concentration inequalities for the supremum of empirical processes under sampling without replacement.

**Core Problem**: Can a transductive learning framework based on local complexity be constructed such that the excess risk bounds match or closely approach those of inductive learning?

## Method

### 1. Problem Setting

Given the full sample $\mathbf{X}_n = \{\vec{\mathbf{x}}_i\}_{i=1}^n$, the training features $\mathbf{X}_m$ are obtained by uniformly sampling $m$ elements from $\mathbf{X}_n$ without replacement, leaving the remaining $u$ elements as the test features $\mathbf{X}_u$. The test-train process is defined as:

$$g(\mathcal{H}) \coloneqq \sup_{h \in \mathcal{H}} \left(\mathcal{L}_u(h) - \mathcal{L}_m(h)\right)$$

### 2. Transductive Complexity (TC)

Four transductive complexities are defined, taking $\mathfrak{R}_u^+$ as an example:

$$\mathfrak{R}_u^+(\mathcal{H}) \coloneqq \mathbb{E}\left[\sup_{h \in \mathcal{H}} R_u^+ h\right], \quad R_u^+ h \coloneqq \mathcal{L}_u(h) - \mathcal{L}_n(h)$$

Key Property: TC can be upper-bounded by the inductive Rademacher complexity (Theorem 3.1):

$$\max\{\mathfrak{R}_u^+(\mathcal{H}), \mathfrak{R}_u^-(\mathcal{H})\} \leq 2\mathfrak{R}_u^{(\text{ind})}(\mathcal{H})$$

### 3. Concentration Inequality for the Test-Train Process (Core Technical Contribution)

**Theorem 4.1**: For a bounded function class $\mathcal{H}$ ($|h(i)| \leq H_0$), with probability at least $1 - e^{-x} - \delta$:

$$g(\mathcal{H}) \leq \mathbb{E}[g(\mathcal{H})] + 4\sqrt{\frac{10rx}{N_{u,m,\delta}}} + 2\sqrt{2}\inf_{\alpha > 0}\left(\frac{\mathfrak{R}_{\min\{u,m\}}^+(\mathcal{H}^2)}{\alpha} + \frac{\alpha x}{N_{u,m,\delta}}\right) + \frac{4H_0^2 x}{N_{u,m,\delta}}$$

where $N_{u,m,\delta} = \frac{\min\{u,m\}}{\log_2(4\min\{u,m\}/\delta)}$.

**Core Proof Innovations**:

- Discovered the **combinatorial property** of the test-train process (Lemma 5.4): changes in test-train loss for each loss function $h$ are always a difference between a pair of elements.
- A new proof strategy that **applies the exponential Efron-Stein inequality twice**: first deriving an upper bound on the upper variance $V_+(g)$, then using this upper bound in synergy with the empirical process on $\mathcal{H}^2$ to derive the concentration inequality.
- Decomposed sampling without replacement into independent random variable sampling via the RANDPERM algorithm.

### 4. TLC Excess Risk Bound (Main Theorem)

**Theorem 4.11** (Core Result): Under standard assumptions, with probability at least $1 - 3e^{-x} - 3\delta$:

$$\mathcal{E}(\hat{f}_m) \leq c_0 r_{u,m} + \frac{4Bc_2 r^*}{K_0} + \frac{c_3 x}{N_{u,m,\delta}}$$

where $r_{u,m}$ and $r^*$ are the fixed points of sub-root functions, both of which converge to 0 at a fast rate under standard learning models. The surrogate variance operator is defined as:

$$\tilde{T}_n(h) \coloneqq \inf_{f_1,f_2 \in \mathcal{F}: \ell_{f_1} - \ell_{f_2} = h} 2B\mathcal{L}_n(\ell_{f_1} - \ell_{f_n^*}) + 2B\mathcal{L}_n(\ell_{f_2} - \ell_{f_n^*})$$

## Theoretical Results

| Result | Best Existing Bound | Ours | Gain |
|------|-----------|--------|------|
| General Excess Risk Bound | $\Theta(\frac{n}{u}r_m^* + \frac{n}{m}r_u^* + \frac{1}{m} + \frac{1}{u})$ (Can diverge) | $\Theta(r_{u,m} + r^* + \frac{\log_2(\min\{u,m\}/\delta) \cdot x}{\min\{u,m\}})$ (Always converges) | Eliminates $n/u, n/m$ factors |
| Realizable Transductive Learning with Finite VC-dimension | $\Theta(\frac{d^{(\text{VC})} \log(ne/d^{(\text{VC})})}{m})$ ($\log n$ gap) | $\Theta(\frac{d^{(\text{VC})} \log(me/d^{(\text{VC})})}{m})$ ($\log m$ gap) | $\log n \to \log m$, resolving a decade-old open problem |
| Minimax Lower Bound Matching | Differing from the lower bound $\Theta(d^{(\text{VC})}/m)$ by $\log n$ | Differing from the lower bound by $\log m$ | Almost optimal |
| Transductive Kernel Learning | Diverges when $m = o(\sqrt{n})$ or $u = o(\sqrt{n})$ | No such constraint, always converges | Significant improvement |
| Concentration Inequality (Corollary 4.3) | Both versions in [Tolstikhin 2014] diverge under certain $u/n$ ratios | Always converges to 0 under standard models | Uniformly outperforms both existing versions |

## Highlights & Insights

1. **Decade-Old Open Problem Resolved**: Narrows the logarithmic gap of realizable transductive learning with finite VC-dimension from $\log n$ to $\log m$, differing from the minimax lower bound by only a $\log m$ factor.
2. **Unified Framework**: Results under the TLC framework cover those of [Yang, 2022] in the unbalanced regime ($m \gg u^2$ or $u \gg m^2$) as a special case, without being subject to such restrictions.
3. **Profound Technical Innovation**: The strategy of applying the exponential Efron-Stein inequality twice presents a brand-new approach for handling complexities under sampling without replacement.
4. **Bridge to Inductive Learning**: Theorem 3.1 establishes a symmetrization inequality relationship between transductive complexity and inductive Rademacher complexity.
5. **Surrogate Variance Operator**: Introducing this operator provides an appropriate localization metric for peeling strategies.

## Limitations & Future Work

1. **Logarithmic Gap Not Completely Eliminated**: A gap of factor $\log m$ remains compared to the minimax lower bound; the full tightness of the optimal bound remains unresolved.
2. **Purely Theoretical Work**: Lacks experimental validation; the tightness of the bounds has not been tested on practical transductive learning algorithms.
3. **Technical Conditions**: Requires a technical assumption of $H_0 \geq 2\sqrt{2}$ (though this can be met via redefining $H_0$).
4. **Scope of Assumption 1**: Although standard, it requires specific convexity and Lipschitz conditions on the loss function.
5. **Computational Feasibility**: Practical computation of the fixed points $r_{u,m}$ and $r^*$ can be difficult for complex function classes.

## Related Work & Insights

- **[Tolstikhin et al., 2014]**: Local complexity analysis of transductive kernel learning, which this work directly improves.
- **[Bartlett et al., 2005]**: Classic work on LRC in inductive learning, which this work extends to the transductive setting.
- **[Darnstädt et al., 2013]**: Existing optimal bounds for finite VC-dimension transductive learning; this work improves its logarithmic factor.
- **[Boucheron et al., 2005]**: Exponential Efron-Stein inequality, the core technical tool of this work.
- **[El-Yaniv and Pechyony, 2009]**: Rademacher complexity framework for transductive learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Brand-new TLC framework, resolving a decade-old open problem
- Experimental Thoroughness: ⭐⭐ — Purely theoretical work, no experiments
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, explicit summaries of main results, but heavy on proof details
- Value: ⭐⭐⭐⭐⭐ — Significant advancement in the field of learning theory, establishing a close connection between transductive and inductive learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](../../ICLR2026/learning_theory/sharp_asymptotic_theory_for_q-learning_with_textttld2z_learning_rate_and_its_gen.md)
- [\[ICLR 2026\] Quantitative Bounds for Length Generalization in Transformers](../../ICLR2026/learning_theory/quantitative_bounds_for_length_generalization_in_transformers.md)
- [\[ICLR 2026\] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs](../../ICLR2026/learning_theory/an_improved_model-free_decision-estimation_coefficient_with_applications_in_adve.md)
- [\[ICML 2025\] Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](improved_and_oracle-efficient_online_ell_1-multicalibration.md)
- [\[ICLR 2026\] Efficient Testing for Correlation Clustering: Improved Algorithms and Optimal Bounds](../../ICLR2026/learning_theory/efficient_testing_for_correlation_clustering_improved_algorithms_and_optimal_bou.md)

</div>

<!-- RELATED:END -->
