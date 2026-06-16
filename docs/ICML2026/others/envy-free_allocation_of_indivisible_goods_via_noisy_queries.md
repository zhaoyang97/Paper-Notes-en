---
title: >-
  [Paper Note] Envy-Free Allocation of Indivisible Goods via Noisy Queries
description: >-
  [ICML 2026][Others][Paper Note] This paper establishes the first sample complexity benchmarks for the problem of finding envy-free (EF) allocations using noisy valuation queries. For two agents under additive Gaussian noise with $m$ items and an optimal negative envy gap $\Delta$, the paper proves a tight query complexity bound of $\widetilde{\Theta}
tags:
  - ICML 2026
  - Others
date: 2026-05-08
content_hash: 1f9a852b1be22e0d
---
# Envy-Free Allocation of Indivisible Goods via Noisy Queries

**Conference**: ICML 2026  
**arXiv**: [2602.06361](https://arxiv.org/abs/2602.06361)  
**Code**: None  
**Area**: Multi-agent Systems / Fair Allocation / Algorithmic Game Theory  
**Keywords**: Envy-Free (EF) Allocation, Noisy Queries, Gaussian Noise, Multi-Armed Bandits, Sample Complexity

## TL;DR
This paper establishes the first sample complexity benchmarks for the problem of finding envy-free (EF) allocations using noisy valuation queries. For two agents under additive Gaussian noise with $m$ items and an optimal negative envy gap $\Delta$, the paper proves a tight query complexity bound of $\widetilde{\Theta}(m^{2.5}/\Delta^2)$ (when $\Delta\gg m^{1/4}$). The upper bound is achieves by a non-adaptive querying strategy combined with a polynomial-time single-item threshold algorithm, while the lower bound holds for adaptive queries and arbitrary computational time.

## Background & Motivation
**Background**: Fair allocation of indivisible goods is a core problem at the intersection of economics and algorithmic game theory. The most common criterion is **Envy-Freeness** (EF), where each agent perceives their own share to be at least as valuable as any other agent's. Since exact EF often does not exist, extensive research has relaxed this to EF1 or EFX. Most existing literature operates under a **default premise**: each agent’s utility $u_i^\nu$ for every item $i$ is **exactly known** or can be precisely queried.

**Limitations of Prior Work**: This assumption often fails in practice—valuations are typically estimated via stochastic simulations, averaged across groups, or are inherently noisy. Once "exact utilities" are replaced by "noisy observations," almost all existing EF/EF1 algorithms lose their guarantees. Prior "query-based" fair allocation studies (Oh et al. 2021; Bu et al. 2024; Plaut–Roughgarden 2020) assume noiseless queries and typically query **bundles** rather than individual items. Li et al. 2025 introduced noise but assumed that "repeated queries + majority voting" could recover exact values (discrete noise with constant flip probability).

**Key Challenge**: Under **additive Gaussian noise**, exact utilities cannot be recovered regardless of the number of queries; one only obtains estimates with $1/\sqrt{q}$ convergence. Estimating all items with sufficient precision would lead to an exponential explosion in sample size. However, EF only requires "correct allocation" rather than "accurate estimation," necessitating new algorithms and lower-bound techniques to characterize this gap.

**Goal**: (1) Formalize the "EF allocation under noisy queries" problem; (2) Provide tight sample complexity bounds in terms of $m$, $\Delta$, and noise variance $\sigma^2$; (3) Ensure the upper bound is computationally efficient and the lower bound covers adaptive, non-limited algorithms.

**Key Insight**: The research adopts the "negative envy gap of the optimal allocation" $\Delta = -\mathrm{OptEnvy}$ as a "gap parameter" for problem difficulty, analogous to the reward gap in multi-armed bandits. A larger $\Delta$ indicates a more "robust" optimal allocation, making EF easier to find. This gap-based perspective allows formulating sample complexity in the standard form $q \propto \mathrm{poly}(m)/\Delta^2$.

**Core Idea**: Instead of estimating every item to $O(\Delta/m)$ precision, the algorithm directly allocates items based on a **single-item threshold rule** $cv_i^a - v_i^b$. By carefully selecting the threshold constant $c$, the envy on both sides is "automatically balanced." This reduces the naive $\widetilde{O}(m^3/\Delta^2)$ to $\widetilde{O}(m^{2.5}/\Delta^2)$. The lower bound uses a hard instance constructed with four types of items (two types of "subtle preference" + two types of "common preference") to prove that the $m^{2.5}$ factor is optimal.

## Method

### Overall Architecture
The paper makes dual contributions:

1.  **Problem Formalization**: $m$ items are allocated between two agents $a$ and $b$, with valuations $u_i^\nu \in [0,1]$. Each query returns $y_{i,t}^\nu \sim \mathcal{N}(u_i^\nu, \sigma^2)$. The goal is to output an allocation $\widehat{\mathcal{A}}$ that satisfies $\mathrm{Envy}(\widehat{\mathcal{A}}) \le 0$ with high probability using minimal queries $q$. The difficulty is parameterized by the optimal negative envy $\Delta > 0$, where $\mathrm{OptEnvy} \le -\Delta$.
2.  **Three-Tier Technique**: A naive baseline (§3) $\rightarrow$ an improved upper bound (§4) $\rightarrow$ a matching lower bound (§5). The paper provides versions for both constant noise variance and general $\sigma$ (§6), extending the conclusions to proportional fairness (§7).

### Key Designs

**1. Naive Repeated Sampling Baseline: Identifying the Bottleneck ($q = \widetilde{O}(m^3/\Delta^2)$)**

To understand the improvement, the cost of the naive approach must be quantified. Distribute $q$ queries uniformly across $m$ items, sampling each $\tau = q/m$ times to get the average $v_i^\nu = \tfrac{1}{\tau}\sum_t y_{i,t}^\nu$. Then, brute-force search over all allocations $\mathcal{A}$ to maximize $\min\{v_{\mathcal{A}_a}^a - v_{\mathcal{A}_b}^a, v_{\mathcal{A}_b}^b - v_{\mathcal{A}_a}^b\}$. Accumulating single-item confidence intervals yields a bundle confidence width $\le \Delta$, leading to $\tau \gtrsim \sigma^2 m^2/\Delta^2$ and thus $q = \widetilde{O}(m^3/\Delta^2)$. This baseline reveals two weaknesses: the bundle confidence width scales at the worst-case rate $O(m \cdot \sigma/\sqrt{\tau})$, losing an $O(m)$ factor, and the search is exponentially complex.

**2. Improved Upper Bound: Single-Item Threshold + Adaptive Threshold Constant $c$ ($q = \widetilde{O}(m^{2.5}/\Delta^2)$)**

The crucial insight is that EF requires "getting the allocation right" rather than "getting the values right," so $O(\Delta/m)$ precision per item is unnecessary. For the small $\Delta$ regime where $q \ge m$, $v_i^\nu$ is still estimated via uniform sampling, but allocation follows a single-item threshold (Eq. 4): give item $i$ to $a$ if $cv_i^a > v_i^b$, else to $b$. Since $cv_i^a - v_i^b$ is Gaussian, the allocation probabilities can be precisely characterized. By weighting the envy on both sides and choosing $c$ to balance them, the algorithm ensures true envy for both $a \to b$ and $b \to a$ is negative. This $c$ depends only on estimates $v_i^\nu$, making the algorithm computable. For the large $\Delta$ regime ($q < m$), the algorithm samples $q$ items once and assigns the rest randomly with $p=1/2$, relying on the Central Limit Theorem (CLT) to ensure that the $\sqrt{m}$ fluctuations of the unsampled portion do not overwhelm the negative envy of the sampled portion. Remark 2 explains that $\Delta \ge m^{1/4}\log^2 m$ is an inherent limit for smoothness control rather than just a technical artifact.

**3. Matching Lower Bound: 4-Type Item Hard Instance + Multiple Hypothesis Testing ($\Omega(m^{2.5}/\Delta^2)$)**

To prove that $m^{2.5}$ is unimprovable even for adaptive algorithms with unlimited time, a sophisticated hard instance is required. Table 1 uses four types of items: the $i \le m/2$ segment contains items where either $a$ or $b$ has a slight preference $\tfrac{1}{2} \pm \varepsilon$, contributing to the "difficulty of identifying individual types." The $i > m/2$ segment includes items where both agents commonlly prefer or dislike items by $\tfrac{1}{2} \pm \gamma$, contributing a "random fluctuation $\Theta(\sqrt{m})$ in the unsampled part." Using Le Cam / KL-style multiple hypothesis testing, the paper proves the algorithm fails with constant probability when $q \le O(m^{2.5}/\Delta^2)$. Previous "slight preference only" instances could only prove $m/\Delta^{1.5}$ or $m^2/\Delta^2$; including "common preference" items scales the fluctuations from $\Theta(\varepsilon\sqrt{m})$ to $\Theta(\sqrt{m})$, closing the gap.

### Loss & Training
This work is an **algorithmic analysis** and does not use differentiable loss functions. Key algorithmic hyperparameters include the threshold constant $c$ (adaptively selected to balance envy) and the sampling pattern (uniform repetition for $q \ge m$, random sub-sampling for $q < m$). Theoretical analysis is built on sub-Gaussian tail bounds for additive Gaussian noise, KL/Le Cam hypothesis testing, and a proof chain of "allocation probability $\rightarrow$ envy integration $\rightarrow$ CLT."

## Key Experimental Results

### Main Results Table
The core complexity results (for constant noise variance) are summarized below.

| Result | Conditions | Query Complexity $q$ | Note |
|------|----------|------------|------|
| Naive Sampling Upper Bound (Thm.1) | Any $\Delta > 0$ | $O(m^3\log(m/\delta)/\Delta^2)$ | Brute-force search, exponential time |
| Main Upper Bound (Thm.2) | $m^{1/4}\log^2 m \le \Delta \le Cm$ | $\widetilde{O}(m^{2.5}/\Delta^2)$ | Poly-time, non-adaptive queries |
| Main Lower Bound (Thm.3) | $\Delta \in (1, m/2)$ | $\widetilde{\Omega}(m^{2.5}/\Delta^2)$ | Holds for adaptive/unlimited algorithms |
| Naive Hard Instance Lower Bound (§5) | — | $\max\{m/\Delta^{1.5}, m^2/\Delta^2\}$ | Strictly weaker than the main lower bound |

The key takeaway: in the "broad regime" of $\Delta \gg m^{1/4}\log^2 m$, the bounds match up to logarithmic factors. The **true sample complexity** is $\widetilde{\Theta}(m^{2.5}/\Delta^2) = \widetilde{\Theta}\left(\sqrt{m}/(\Delta/m)^2\right)$, explicitly isolating the "normalized gap" $\Delta/m$.

### Refined Bounds under General Noise Levels

| Regime | Upper Bound | Lower Bound | Tightness |
|------|------|------|------|
| $q \ge m$ (Thm.5, Appx.C.3) | $q = m\lceil\sigma^2(15 m^{3/2}\log m/\Delta^2 + \log^2 m)\rceil$ | $O(\sigma^2 m^{2.5}/\Delta^2)$ (at $\Delta=O(m^{3/4})$) | Tight up to $\log m$ |
| $q < m$ (Thm.102, Appx.C.4) | $\max\{160^2 m^4\sigma^4\log^2 m/\Delta^4, 160\sigma m^{5/2}\log^2 m/\Delta^2\}$ | $O(\sigma m^{2.5}/\Delta^2)$ (at $\Delta=\omega(m^{3/4})$) | Tight up to $(\log m)^2$ |

### Key Findings
- **Origin of the $\sqrt{m}$ Factor**: Intuitively, $q \propto \sqrt{m}/(\Delta/m)^2$. The denominator $(\Delta/m)^2$ is the bandit-style squared inverse of the normalized gap, and the numerator $\sqrt{m}$ reflects the CLT-based error accumulation when bundling $m$ items.
- **When Every Item Doesn't Need to be Queried**: When $\Delta \gg m^{3/4}$, the algorithm only needs to query $q < m$ items. The unsampled portion is handled by random allocation and CLT, which is cheaper than querying every item once.
- **Lower Bound Insights for Algorithm Design**: The naive $\tfrac{1}{2} \pm \varepsilon$ instance is insufficient. Adding "common preference" items to amplify random fluctuations is necessary to reach $m^{2.5}$, suggesting that items "expensive/cheap for both" significantly affect decision variance.
- **Proportional Fairness = EF (n=2, additive)**: For two agents with additive utilities, proportionality is equivalent to EF; thus, all results extend to proportional fairness.

## Highlights & Insights
- Successfully bridges **EF problems in AGT** with the **sample complexity framework of statistical estimation/bandits**, mapping the "fairness threshold" to "exploration difficulty."
- **Single-Item Threshold + Adaptive $c$** is an elegant design: it avoids bundle-level estimation by using a global degree of freedom to cancel systematic bias between agents.
- The "common preference" trick in the lower bound reveals an auxiliary structure that affects algorithmic variance without changing the optimal envy direction. This construction likely serves as a template for extending results to $n > 2$ agents.
- The upper bound is achieved via **non-adaptive** queries (Thm.2), yet matches the lower bound for **adaptive** algorithms, providing high pedagogical value for complexity analysis.

## Limitations & Future Work
- **Two Agents Only**: The threshold rule would need to expand from one ratio to $\binom{n}{2}$ ratios for $n > 2$ agents; extending this is an open problem.
- **Additive Valuations/Gaussian Noise**: Extending to sub-Gaussian noise is straightforward but requires reworking concentration inequalities; non-additive utilities (complements/substitutes) would require entirely new algorithms.
- **Constraint $\Delta \ge m^{1/4}\log^2 m$**: This may represent a fundamental phase transition where algorithms must switch from "item-by-item" to "bundle-level" decision making.
- **No EF1/EFX Relaxations**: This paper focuses on strict EF. Sample complexity for EF1/EFX would be more practical since these always exist, potentially removing the $\Delta$ assumption.

## Related Work & Insights
- **vs. Oh et al. 2021 / Bu et al. 2024 (Noiseless Bundle Queries)**: They query exact bundle values/comparisons, achieving EF1 in logarithmic queries. This paper uses noisy single-item queries where even repetition doesn't yield exact values, representing a different difficulty structure.
- **vs. Li et al. 2025 (Round-Robin + Noise)**: Their noise is a constant flip probability where majority voting recovers exact values, essentially making it a noiseless problem asymptotically. Additive Gaussian noise in this paper destroys that property.
- **vs. Procaccia et al. 2024 (Online Bandit Allocation)**: They focus on sequential arrival and fairness in expectation. This paper is offline and requires high-probability EF, focusing on sample complexity bounds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes the "noisy query EF" framework with a perfect bandit-theoretic alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical; complete with naive/improved upper bounds, matching lower bounds, and refined general noise analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ High readability, providing intuition in the main text and technical rigor in the appendices.
- Value: ⭐⭐⭐⭐ Opens a new direction for noisy fair allocation, with applications in group decision-making and crowdsourcing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)
- [\[ICML 2025\] Generation from Noisy Examples](../../ICML2025/others/generation_from_noisy_examples.md)
- [\[CVPR 2026\] Debiased Sample Selection for Learning with Noisy Labels](../../CVPR2026/others/debiased_sample_selection_for_learning_with_noisy_labels.md)

</div>

<!-- RELATED:END -->
