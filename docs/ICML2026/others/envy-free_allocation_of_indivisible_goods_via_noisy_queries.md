---
title: >-
  [Paper Note] Envy-Free Allocation of Indivisible Goods via Noisy Queries
description: >-
  [ICML 2026][Envy-free allocation] This paper establishes the first sample complexity benchmark for finding envy-free (EF) allocations using noisy valuation queries. For two agents, additive Gaussian noise, $m$ items…
tags:
  - "ICML 2026"
  - "Envy-free allocation"
  - "Noisy queries"
  - "Gaussian noise"
  - "Multi-armed bandits"
  - "Sample complexity"
date: 2026-05-08
content_hash: f045411e899ddb6d
---

# Envy-Free Allocation of Indivisible Goods via Noisy Queries

**Conference**: ICML 2026  
**arXiv**: [2602.06361](https://arxiv.org/abs/2602.06361)  
**Code**: None  
**Area**: Multi-agent / Fair Allocation / Algorithmic Game Theory  
**Keywords**: Envy-free allocation, Noisy queries, Gaussian noise, Multi-armed bandits, Sample complexity

## TL;DR
This paper establishes the first sample complexity benchmark for finding envy-free (EF) allocations using noisy valuation queries. For two agents, additive Gaussian noise, $m$ items, and an optimal negative envy gap $\Delta$, the query complexity is shown to have a tight bound of $\widetilde{\Theta}(m^{2.5}/\Delta^2)$ (when $\Delta\gg m^{1/4}$). The upper bound is achieved by non-adaptive queries with a poly-time single-item threshold algorithm, while the lower bound holds for adaptive queries and arbitrary computation time.

## Background & Motivation
**Background**: Fair allocation of indivisible goods is a core problem at the intersection of economics and algorithmic game theory. The most common fairness metric is **Envy-Free** (EF): every agent believes their bundle is no worse than anyone else's. Since exact EF often does not exist, extensive research has relaxed this to EF1 or EFX. Most existing literature operates under the **default premise** that each agent's utility $u_i^\nu$ for each item is **exactly known** or can be precisely queried.

**Limitations of Prior Work**: This assumption often fails in reality—valuations might be estimated via stochastic simulations, represent group averages, or the evaluation process itself may be noisy. When "precise utility" is replaced with "noisy observations," almost all existing EF/EF1 algorithms lose their guarantees. Existing "fair allocation with queries" work (Oh et al. 2021; Bu et al. 2024; Plaut–Roughgarden 2020) assumes noise-free queries and typically queries **bundles** rather than individual items. Li et al. 2025 introduced noise but assumed "repeated queries + majority voting" could recover exact values (discrete noise with constant flip probability).

**Key Challenge**: Under **additive Gaussian noise**, exact utilities cannot be obtained regardless of the number of queries; one only receives estimates converging at $1/\sqrt{q}$. Attempting to estimate all items precisely would cause an exponential explosion in sample size. However, EF only requires "allocating correctly" rather than "estimating accurately," necessitating new algorithms and lower-bound techniques to characterize this gap.

**Goal**: (1) Formalize the problem of "EF allocation under noisy queries"; (2) Provide tight sample complexity bounds in terms of $m$, $\Delta$, and noise variance $\sigma^2$; (3) Ensure upper bounds are poly-time computable and lower bounds hold for adaptive, arbitrary-time algorithms.

**Key Insight**: The paper uses the "negative envy gap of the optimal allocation" $\Delta = -\mathrm{OptEnvy}$ as a "gap parameter" for problem difficulty, similar to the reward gap in multi-armed bandits. A larger $\Delta$ implies the optimal allocation is more "forgiving," making EF easier to find. This gap-centric perspective allows for standard sample complexity forms such as $q\propto \mathrm{poly}(m)/\Delta^2$.

**Core Idea**: Instead of estimating every item to $O(\Delta/m)$ precision, items are allocated directly using a **single-item threshold rule** such as $cv_i^a - v_i^b$, where the threshold constant $c$ is carefully chosen to "automatically balance" envy between sides. This reduces the naive $\widetilde{O}(m^3/\Delta^2)$ to $\widetilde{O}(m^{2.5}/\Delta^2)$. The lower bound uses a hard instance constructed with four categories of items (two types of "subtle preferences" + two types of "joint preferences") to prove that $m^{2.5}$ is unimprovable.

## Method

### Overall Architecture
Two dual contributions:

1.  **Problem Formalization**: $m$ goods allocated between agents $a, b$ with valuations $u_i^\nu\in[0,1]$. Each query returns $y_{i,t}^\nu\sim\mathcal{N}(u_i^\nu,\sigma^2)$. The goal is to output an allocation $\widehat{\mathcal{A}}$ that satisfies $\mathrm{Envy}(\widehat{\mathcal{A}})\le 0$ with high probability using minimal queries $q$. Difficulty is parameterized by optimal negative envy $\Delta>0$, where $\mathrm{OptEnvy}\le -\Delta$.
2.  **Three-stage Technical Approach**: Naive baseline (§3) $\to$ Improved upper bound (§4) $\to$ Matching lower bound (§5). The paper provides versions for both constant noise variance and general $\sigma$ (§6), extending results to proportional fairness (§7).

### Key Designs

1.  **Naive Repeated Sampling Baseline — Identifying Bottlenecks ($q=\widetilde{O}(m^3/\Delta^2)$)**:
    - **Function**: Serves as a control group to demonstrate the cost of naive approaches and pinpoint the waste of factor $m$.
    - **Mechanism**: Distributes $q$ queries uniformly over $m$ items, sampling each $\tau=q/m$ times to get average $v_i^\nu=\tfrac{1}{\tau}\sum_t y_{i,t}^\nu$. It then performs a **brute-force search** over all allocations $\mathcal{A}$ to find the one maximizing $\min\{v_{\mathcal{A}_a}^a-v_{\mathcal{A}_b}^a,\,v_{\mathcal{A}_b}^b-v_{\mathcal{A}_a}^b\}$. Summing single-item confidence intervals yields a bundle confidence width requiring $\le \Delta$, which implies $\tau\gtrsim \sigma^2 m^2/\Delta^2$, hence $q=\widetilde{O}(m^3/\Delta^2)$.
    - **Design Motivation**: Exposed two weaknesses: (i) Bundle confidence width scales by "worst-case" $O(m\cdot\sigma/\sqrt{\tau})$, wasting a factor of $m$; (ii) Brute-force search has exponential complexity. Subsequent upper bounds fix these specific points.

2.  **Improved Upper Bound — Single-item Threshold + Adaptive Constant $c$ ($q=\widetilde{O}(m^{2.5}/\Delta^2)$)**:
    - **Function**: Reduces sample complexity from $m^3$ to $m^{2.5}$ and converts the allocation rule to polynomial time.
    - **Mechanism**: In the "small $\Delta$" regime ($q \ge m$), it still uses uniform sampling for estimates $v_i^\nu$, but adopts a **single-item threshold** $(\mathrm{eq.4})$: assign to $a$ if $cv_i^a>v_i^b$, else to $b$. Since $cv_i^a-v_i^b$ is Gaussian, the allocation probability of each item is precisely characterized. By summing envy on both sides with appropriate weights, it proves the existence of a threshold $c$ such that true envy for both $a \to b$ and $b \to a$ is simultaneously negative. This $c$ depends only on estimates $v_i^\nu$ and is computable. In the "large $\Delta$" regime ($q < m$), it samples $q$ items once and assigns the rest randomly with $p=1/2$. Central limit arguments show that the $\sqrt{m}$ fluctuations of the unsampled portion do not override the negative envy of the sampled portion.
    - **Design Motivation**: Replaces the "worst-case error accumulation" of $m$ in the naive analysis with "precise characterization of allocation probabilities under a threshold rule." Using $c$ as a balance instead of a fixed $c=1$ avoids failure modes where one agent's valuations are globally higher, effectively using one degree of freedom to cancel heterogeneity.

3.  **Matching Lower Bound — 4-Category Hard Instance + Multi-hypothesis Testing ($\Omega(m^{2.5}/\Delta^2)$)**:
    - **Function**: Proves $m^{2.5}/\Delta^2$ is tight even for adaptive queries and arbitrary computation time.
    - **Mechanism**: Table 1 designs four types of items: (i) items for $i\le m/2$ are weakly preferred by $a$ with probability $1/2$ at $\tfrac{1}{2}\pm\varepsilon$ and inversely by $b$; (ii) items for $i>m/2$ are **jointly preferred/disliked** by both with $\tfrac{1}{2}\pm\gamma$ with probability $1/2$. The first group contributes the difficulty of identifying individual types, while the second contributes random fluctuations $\Theta(\sqrt{m})$ from unsampled parts. Using Le Cam/KL-style multi-hypothesis testing, it proves the algorithm fails with constant probability when $q\le O(m^{2.5}/\Delta^2)$.
    - **Design Motivation**: Original "only $\tfrac{1}{2}\pm\varepsilon$" hard instances only reach $m/\Delta^{1.5}$ or $m^2/\Delta^2$, which is looser than the upper bound. Adding "joint preference" items stretches random fluctuations from $\Theta(\varepsilon\sqrt{m})$ to $\Theta(\sqrt{m})$, filling the gap and yielding a tight bound within logarithmic factors.

### Loss & Training
As this is an **algorithmic analysis** paper, there is no differentiable loss. Key hyperparameters at the algorithmic level include the threshold constant $c$ (adaptively selected from estimates to balance envy) and the sampling pattern (uniform repeated sampling if $q \ge m$, random sub-sampling once if $q < m$). Theoretical analysis relies on sub-Gaussian tail bounds for Gaussian noise, KL/Le Cam multi-hypothesis testing, and a "probability-envy integration-central limit" proof chain.

## Key Experimental Results

### Main Results Comparison Table
The core contributions are the sample complexity upper and lower bounds (for constant noise variance), summarized below:

| Result | Conditions | Queries $q$ | Remarks |
| :--- | :--- | :--- | :--- |
| Naive Upper Bound (Thm.1) | Any $\Delta>0$ | $O(m^3\log(m/\delta)/\Delta^2)$ | Brute-force allocation, exponential time |
| **Ours Main Upper Bound (Thm.2)** | $m^{1/4}\log^2 m\le\Delta\le Cm$ | $\widetilde{O}(m^{2.5}/\Delta^2)$ | Poly-time, non-adaptive queries |
| **Ours Main Lower Bound (Thm.3)** | $\Delta\in(1,m/2)$ | $\widetilde{\Omega}(m^{2.5}/\Delta^2)$ | Holds for adaptive + arbitrary-time algorithms |
| Naive Hard Instance (§5 Discussion) | — | $\max\{m/\Delta^{1.5},m^2/\Delta^2\}$ | Strictly weaker, justifies 4-category instance |

The most critical conclusion: in the "wide regime" of $\Delta\gg m^{1/4}\log^2 m$, the bounds match within logarithmic factors, showing the **true sample complexity** $= \widetilde{\Theta}(m^{2.5}/\Delta^2) = \widetilde{\Theta}\!\left(\sqrt{m}/(\Delta/m)^2\right)$, explicitly isolating the "normalized gap" $\Delta/m$.

### Key Findings
- **Origin of the $\sqrt{m}$ factor**: Intuitively, $q\propto \sqrt{m}/(\Delta/m)^2$. The denominator $(\Delta/m)^2$ is the standard bandit-style "inverse square of the normalized gap," while the numerator $\sqrt{m}$ reflects the Central Limit Theorem accumulation of errors when bundling $m$ items.
- **When querying every item is unnecessary**: When $\Delta\gg m^{3/4}$, the algorithm only needs to query $q<m$ items. Random allocation + CLT for the unsampled part is cheaper than querying every item once.
- **Lower bound insights for algorithm design**: Naive $\tfrac{1}{2}\pm\varepsilon$ instances are insufficient. Including "joint preference" items to amplify random fluctuations is necessary to reach $m^{2.5}$, suggesting algorithms cannot ignore items that are expensive/cheap for both parties.
- **Proportional Fairness = EF (n=2, additive)**: For two agents with additive utilities, proportional fairness is equivalent to "envy $\le$ half of the total." Thus, all results translate to proportional fairness.

## Highlights & Insights
- Successfully bridges "EF problems in AGT" and "sample complexity frameworks in statistical estimation/bandits" by treating $\Delta$ as a gap, mapping fairness thresholds to statistical exploration difficulty.
- **Single-item threshold + Adaptive constant $c$** is an elegant design: it avoids bundle-level estimation but uses a global degree of freedom to cancel systematic bias, making the analysis technically deep yet natural.
- The lower bound trick of "adding joint preference items to amplify fluctuation" reveals an auxiliary structure that does not affect the optimal envy direction but significantly impacts the algorithm's decision variance.
- The upper bound is achieved with **non-adaptive** queries (Thm.2), while the lower bound allows **adaptive** algorithms, providing a tight match that confirms adaptivity does not fundamentally reduce complexity here.

## Limitations & Future Work
- Results currently cover **two agents**. Extending the threshold rule to $\binom{n}{2}$ ratios for $n>2$ agents is non-trivial and remains an open problem.
- Considers only **additive utilities** and **additive Gaussian noise**. Extensions to sub-Gaussian noise should follow similar lines but require re-deriving concentration inequalities; non-additive utilities (complements/substitutes) likely require entirely new algorithms.
- The assumption **$\Delta\ge m^{1/4}\log^2 m$**: As noted in Remark 2, this may reflect a fundamental shift where algorithms must transition from item-by-item thresholding to bundle-level decision-making for smaller gaps.
- No **EF1 / EFX relaxations**: The paper focuses on strict EF. Future work extending sample complexity to EF1/EFX is practically valuable since they always exist and could avoid the $\Delta$ assumption.

## Related Work & Insights
- **vs. Oh et al. 2021 / Bu et al. 2024 (Exact bundle queries)**: They query precise values or comparisons on bundles, where EF1 takes logarithmic queries. This paper uses single-item Gaussian noise where repeated queries never yield precision, changing the difficulty structure.
- **vs. Li et al. 2025 (Round-robin + Noise)**: Their noise is discrete (constant flip), where majority voting recovers exact values. This makes their problem essentially noise-free in the limit; additive Gaussian noise breaks this property.
- **vs. Procaccia et al. 2024 / Sinha et al. 2023 (Online bandit allocation)**: They focus on streaming items and fairness in expectation; this paper focuses on offline, high-probability EF achievement with sample complexity bounds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes the "EF allocation under noisy queries" framework and maps it to bandit theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical; the three-stage proof (naive/improved/lower bound) is complete with refined bounds for varying noise levels.
- Writing Quality: ⭐⭐⭐⭐⭐ Provides strong intuition in the main text, details in appendix, and clear boundaries in Remarks 1-2.
- Value: ⭐⭐⭐⭐ Opens a new direction for group decision-making, crowdsourced allocation, and A/B test-driven resource distribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](../../ICLR2026/others/noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)
- [\[ICML 2025\] Generation from Noisy Examples](../../ICML2025/others/generation_from_noisy_examples.md)

</div>

<!-- RELATED:END -->
