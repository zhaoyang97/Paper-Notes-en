---
title: >-
  [Paper Note] Envy-Free Allocation of Indivisible Goods via Noisy Queries
description: >-
  [ICML 2026][Envy-free allocation] This paper establishes the first sample complexity benchmarks for the problem of "finding an envy-free allocation using noisy valuation queries." In a setting with two agents, additive Gaussian noise, $m$ items, and an optimal negative envy gap $\Delta$, the authors prove a tight bound of $\widetilde{\Theta}(m^{2.5}/\Delta^2)$ (when $\Delta\gg m^{1/4}$). The upper bound is achieved by a polynomial-time algorithm using non-adaptive queries and…
tags:
  - "ICML 2026"
  - "Envy-free allocation"
  - "noisy queries"
  - "Gaussian noise"
  - "Multi-armed bandits"
  - "sample complexity"
date: 2026-05-08
content_hash: c9836cb71704106a
---

# Envy-Free Allocation of Indivisible Goods via Noisy Queries

**Conference**: ICML 2026  
**arXiv**: [2602.06361](https://arxiv.org/abs/2602.06361)  
**Code**: None  
**Area**: Multi-agent / Fair Allocation / Algorithmic Game Theory  
**Keywords**: Envy-free allocation, noisy queries, Gaussian noise, Multi-armed bandits, sample complexity

## TL;DR
This paper establishes the first sample complexity benchmarks for the problem of "finding an envy-free allocation using noisy valuation queries." In a setting with two agents, additive Gaussian noise, $m$ items, and an optimal negative envy gap $\Delta$, the authors prove a tight bound of $\widetilde{\Theta}(m^{2.5}/\Delta^2)$ (when $\Delta\gg m^{1/4}$). The upper bound is achieved by a polynomial-time algorithm using non-adaptive queries and a single-item threshold rule, while the lower bound holds for adaptive queries and arbitrary computational time.

## Background & Motivation
**Background**: Fair allocation of indivisible goods is a core problem at the intersection of economics and algorithmic game theory. The most common fairness criterion is **Envy-Freeness** (EF): each agent perceives their own bundle to be at least as good as anyone else's. Since exact EF allocations often do not exist, extensive literature focuses on relaxations like EF1 or EFX. A **default premise** in most work is that each agent's utility $u_i^\nu$ for every item is **known exactly** or can be queried precisely.

**Limitations of Prior Work**: In practice, this assumption rarely holds—valuations may be estimated via stochastic simulations, represent an average of a population, or stem from a noisy evaluation process. Once "precise utility" is replaced by "noisy observations," almost all existing EF/EF1 algorithms lose their guarantees. Existing work on fair allocation "with queries" (Oh et al. 2021; Bu et al. 2024; Plaut–Roughgarden 2020) assumes noiseless queries and often queries **bundles** rather than single items. Li et al. 2025 introduced noise but assumed discrete noise where "repeated queries + majority voting" could recover exact values.

**Key Challenge**: Under **additive Gaussian noise**, exact utilities can never be obtained regardless of the number of queries; one only attains estimates with a $1/\sqrt{q}$ convergence rate. Precisely estimating all items would lead to exponential sample complexity. However, EF fundamentally requires "partitioning correctly" rather than "estimating accurately." This necessitates new algorithms and lower bound techniques to characterize the gap between the two.

**Goal**: (1) Formalize the problem of "EF allocation under noisy queries"; (2) Provide tight sample complexity bounds in terms of $m$, $\Delta$, and noise variance $\sigma^2$; (3) Ensure the upper bound is achievable in polynomial time and the lower bound holds for adaptive algorithms with arbitrary time.

**Key Insight**: The "optimal negative envy gap" $\Delta=-\mathrm{OptEnvy}$ is used as the difficulty parameter, analogous to the reward gap in multi-armed bandits. A larger $\Delta$ implies a more "generous" optimal allocation, making it easier to find an EF solution. This gap-based perspective allows for a standard sample complexity form of $q\propto \mathrm{poly}(m)/\Delta^2$.

**Core Idea**: Instead of estimating every item to $O(\Delta/m)$ precision, the algorithm allocates items directly via a **single-item threshold rule** $cv_i^a - v_i^b$. By carefully selecting the threshold constant $c$, the envy on both sides is "automatically balanced." This reduces the naive $\widetilde{O}(m^3/\Delta^2)$ bound to $\widetilde{O}(m^{2.5}/\Delta^2)$. The lower bound utilizes a hard instance constructed from four types of items (two types of "slight preferences" and two types of "common preferences") to prove that $m^{2.5}$ is unimprovable.

## Method

### Overall Architecture
The paper provides two dual contributions:

1.  **Problem Formalization**: $m$ items are allocated between two agents $a,b$, with utilities $u_i^\nu\in[0,1]$. Each query returns $y_{i,t}^\nu\sim\mathcal{N}(u_i^\nu,\sigma^2)$. The goal is to output an allocation $\widehat{\mathcal{A}}$ with minimum queries $q$ such that $\mathrm{Envy}(\widehat{\mathcal{A}})\le 0$ with high probability. The difficulty is parameterized by the optimal negative envy $\Delta>0$, where $\mathrm{OptEnvy}\le -\Delta$.
2.  **Mechanism**: A three-stage technical roadmap: Naive baseline (§3) $\rightarrow$ Improved upper bound (§4) $\rightarrow$ Matching lower bound (§5). The paper provides versions for both constant noise variance and general $\sigma$ (§6), and extends results to proportional fairness (§7).

### Key Designs

**1. Naive Repeated Sampling Baseline: Exposing the bottleneck ($q=\widetilde{O}(m^3/\Delta^2)$)**

To understand the improvement, one must first calculate the cost of a naive approach. Uniformly allocating $q$ queries across $m$ items yields $\tau=q/m$ samples per item. Taking the average $v_i^\nu=\tfrac{1}{\tau}\sum_t y_{i,t}^\nu$, one then exhaustively searches for an allocation $\mathcal{A}$ that maximizes $\min\{v_{\mathcal{A}_a}^a-v_{\mathcal{A}_b}^a,\,v_{\mathcal{A}_b}^b-v_{\mathcal{A}_a}^b\}$. Using bundle confidence intervals (CI), a width $\le \Delta$ requires $\tau\gtrsim \sigma^2 m^2/\Delta^2$, resulting in $q=\widetilde{O}(m^3/\Delta^2)$. This baseline reveals two weaknesses: the bundle CI width scales by the worst-case $O(m\cdot\sigma/\sqrt{\tau})$, wasting a factor of $m$, and the search is exponential.

**2. Improved Upper Bound: Single-item Threshold + Adaptive Threshold Constant $c$ ($q=\widetilde{O}(m^{2.5}/\Delta^2)$)**

The key insight is that EF only requires "correct partitioning," not "precise estimation," so $O(\Delta/m)$ precision per item is unnecessary. In the small $\Delta$ regime where $q\ge m$, the algorithm still samples uniformly to get $v_i^\nu$ but allocates using the single-item threshold rule (Eq. 4): item $i$ goes to $a$ if $cv_i^a>v_i^b$, otherwise to $b$. Since $cv_i^a-v_i^b$ is Gaussian, the allocation probability per item can be precisely characterized. By weighting the envy on both sides, it is proven that a threshold $c$ exists such that the true envy for both $a\to b$ and $b\to a$ is negative. This $c$ depends only on estimated values $v_i^\nu$. In the large $\Delta$ regime where $q<m$, $q$ items are sampled once while the rest are allocated randomly with probability $1/2$; the Central Limit Theorem (CLT) ensures that the $O(\sqrt{m})$ fluctuations of the unqueried items do not overwhelm the negative envy of the queried ones. This avoids the $m$ loss from worst-case error accumulation.

**3. Matching Lower Bound: 4-Item-Type Hard Instance + Multiple Hypothesis Testing ($\Omega(m^{2.5}/\Delta^2)$)**

To prove that $m^{2.5}$ cannot be improved even by adaptive algorithms with arbitrary time, a sophisticated hard instance is designed. Table 1 uses four item types: items $i\le m/2$ are slightly preferred by $a$ with probability $1/2$ (utility $\tfrac{1}{2}\pm\varepsilon$) and inversely by $b$, contributing to the "difficulty of identifying individual types." Items $i>m/2$ are commonly preferred/disliked by both with probability $1/2$ (utility $\tfrac{1}{2}\pm\gamma$), contributing "random fluctuations of $\Theta(\sqrt{m})$ in the unqueried part." Using Le Cam/KL-style multiple hypothesis testing, the authors prove any algorithm with $q\le O(m^{2.5}/\Delta^2)$ fails with constant probability. While simple $\tfrac{1}{2}\pm\varepsilon$ instances only yield $m^2/\Delta^2$, adding "common preference" items scales the fluctuation from $\Theta(\varepsilon\sqrt{m})$ to $\Theta(\sqrt{m})$, matching the upper bound.

### Loss & Training
This is an **algorithmic analysis** paper; there is no differentiable loss. Key hyperparameters at the algorithmic level include the threshold constant $c$ (selected adaptively) and the sampling pattern (uniform repeated sampling for $q\ge m$ vs. random sub-sampling for $q<m$). The theoretical analysis is built on sub-Gaussian tail bounds for Gaussian noise, KL/Le Cam testing, and a proof chain linking "allocation probability—envy integrals—CLT."

## Key Experimental Results

### Main Results Comparison Table
The core result is the sample complexity bounds for constant noise variance.

| Result | Condition | Queries $q$ | Remarks |
|------|----------|------------|------|
| Naive Sampling Upper Bound (Thm.1) | Any $\Delta>0$ | $O(m^3\log(m/\delta)/\Delta^2)$ | Exhaustive search, exponential time |
| **Ours: Main Upper Bound (Thm.2)** | $m^{1/4}\log^2 m\le\Delta\le Cm$ | $\widetilde{O}(m^{2.5}/\Delta^2)$ | Poly-time, non-adaptive queries |
| **Ours: Main Lower Bound (Thm.3)** | $\Delta\in(1,m/2)$ | $\widetilde{\Omega}(m^{2.5}/\Delta^2)$ | Holds for adaptive + arbitrary time |
| Naive Hard Instance Lower Bound (§5) | — | $\max\{m/\Delta^{1.5},m^2/\Delta^2\}$ | Strictly weaker than ours |

The primary conclusion: In the "broad regime" of $\Delta\gg m^{1/4}\log^2 m$, the bounds match up to logarithmic factors. The **true sample complexity** is $\widetilde{\Theta}(m^{2.5}/\Delta^2) = \widetilde{\Theta}\!\left(\sqrt{m}/(\Delta/m)^2\right)$, explicitly separating the "normalized gap" $\Delta/m$.

### Refined Bounds for General Noise

| Regime | Upper Bound | Lower Bound | Tightness |
|------|--------------|--------------|-----------|
| $q\ge m$ (Thm.5) | $q=m\lceil\sigma^2(15 m^{3/2}\log m/\Delta^2+\log^2 m)\rceil$ | $O(\sigma^2 m^{2.5}/\Delta^2)$ | Tight up to $\log m$ |
| $q<m$ (Thm.102) | $\max\{160^2 m^4\sigma^4\log^2 m/\Delta^4,\,160\sigma m^{5/2}\log^2 m/\Delta^2\}$ | $O(\sigma m^{2.5}/\Delta^2)$ | Tight up to $(\log m)^2$ |

### Key Findings
- **Source of the $\sqrt{m}$ factor**: Intuitively, $q\propto \sqrt{m}/(\Delta/m)^2$, where the denominator is the standard bandit gap squared, and the numerator $\sqrt{m}$ reflects the CLT-style error accumulation when items are bundled.
- **When not to query every item**: For $\Delta\gg m^{3/4}$, the algorithm only needs $q<m$ queries. The unqueried part is handled by random allocation and CLT, which is cheaper than querying everything once.
- **Lower bound insights for algorithm design**: Naive $\tfrac{1}{2}\pm\varepsilon$ instances are insufficient. One must include "common preference" items to amplify fluctuations, suggesting that items valued similarly by both parties are the true drivers of variance in decision-making.
- **Prop = EF (n=2, additive)**: For two agents with additive utilities, EF is equivalent to proportional fairness; hence, all results extend to proportionality.

## Highlights & Insights
- Successfully bridges the "EF problem in AGT" with the "sample complexity framework in statistical estimation/bandits" by treating $\Delta$ as a reward gap.
- The **single-item threshold + adaptive constant $c$** is an elegant design: it avoids bundle-level estimation yet uses a global degree of freedom to cancel systematic bias between agents.
- The "common preference item" trick in the lower bound is subtle and clever, revealing an auxiliary structure that does not change the optimal envy direction but significantly impacts the algorithm's decision variance.
- The upper bound is achieved via **non-adaptive** queries (Thm.2), while the lower bound covers **adaptive** algorithms, showing that adaptivity provides little advantage here.

## Limitations & Future Work
- **Two agents only**: Extending this to $n>2$ agents would require the threshold rule to expand to $\binom{n}{2}$ pairs, making the optimal statistic unclear.
- **Model assumptions**: Focuses on **additive utilities** and **additive Gaussian noise**. Sub-Gaussian extensions are likely possible but non-additive utilities (complements/substitutes) would require entirely new algorithms.
- **Gap constraint**: The assumption $\Delta\ge m^{1/4}\log^2 m$ may reflect a fundamental transition point where algorithms must switch from item-wise thresholds to bundle-level decisions.
- **No EF1/EFX relaxations**: This work focuses on strict EF. Future work on EF1/EFX under noisy queries would be beneficial as solutions always exist, potentially removing the $\Delta$ dependency.

## Related Work & Insights
- **vs. Oh et al. 2021 / Bu et al. 2024 (Noiseless bundle queries)**: Their queries provide exact bundle comparisons, achieving EF1 with logarithmic queries. This paper handles single-item Gaussian noise where exact values are never known.
- **vs. Li et al. 2025 (Discrete noise)**: Their noise model allows recovery of exact values via majority voting, essentially reducing back to a noiseless problem. This paper's Gaussian model is truly "non-asymptotic" and noisy.
- **vs. Procaccia et al. 2024 / Sinha et al. 2023 (Online bandit allocation)**: Those works focus on items arriving over time and fairness in expectation. This paper is offline and requires high-probability EF guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](../../ACL2025/others/laquer_localized_attribution.md)
- [\[ICLR 2026\] Learning in Prophet Inequalities with Noisy Observations](../../ICLR2026/others/learning_in_prophet_inequalities_with_noisy_observations.md)
- [\[ICML 2025\] Generation from Noisy Examples](../../ICML2025/others/generation_from_noisy_examples.md)
- [\[CVPR 2026\] Debiased Sample Selection for Learning with Noisy Labels](../../CVPR2026/others/debiased_sample_selection_for_learning_with_noisy_labels.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)
- [\[ICML 2025\] Generation from Noisy Examples](../../ICML2025/others/generation_from_noisy_examples.md)
- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](../../ACL2025/others/laquer_localized_attribution.md)

</div>

<!-- RELATED:END -->
