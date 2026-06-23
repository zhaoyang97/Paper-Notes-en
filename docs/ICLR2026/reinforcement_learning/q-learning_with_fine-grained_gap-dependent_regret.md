---
title: >-
  [Paper Note] Q-Learning with Fine-Grained Gap-Dependent Regret
description: >-
  [ICLR 2026][Reinforcement Learning][model-free RL] Focusing on model-free RL for episodic tabular MDPs, this paper proposes a fine-grained analysis framework that "separately counts visits for optimal and sub-optimal state-action pairs." It provides the first fine-grained gap-dependent regret upper bound involving individual gaps $\Delta_h(s,a)$ for UCB-Hoeffding and s
tags:
  - ICLR 2026
  - Reinforcement Learning
  - model-free RL
  - gap-dependent regret
  - UCB-Hoeffding
  - tabular MDP
date: 2026-05-08
content_hash: 9efb128850ed1a3b
---
# Q-Learning with Fine-Grained Gap-Dependent Regret

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fE0RJto3Na](https://openreview.net/forum?id=fE0RJto3Na)  
**Code**: Included in supplementary material (synthetic experiments)  
**Area**: Reinforcement Learning / Theory / Regret Analysis  
**Keywords**: model-free RL, gap-dependent regret, UCB-Hoeffding, multi-step bootstrap, tabular MDP

## TL;DR
Focusing on model-free RL for episodic tabular MDPs, this paper proposes a fine-grained analysis framework that "separately counts visits for optimal and sub-optimal state-action pairs." It provides the first fine-grained gap-dependent regret upper bound involving individual gaps $\Delta_h(s,a)$ for UCB-Hoeffding and subsequently fixes two defects in the truncation and martingale difference conditions of AMB—the only prior non-UCB algorithm—introducing two improved versions: ULCB-Hoeffding and Refined AMB.

## Background & Motivation
**Background**: In episodic tabular MDPs with $S$ states, $A$ actions, and $H$ steps per episode, model-free methods (learning value functions directly with memory scaling linearly with state count) are the primary practical choice. The minimax regret lower bound under $K$ episodes and $T=KH$ steps is $\Omega(\sqrt{H^2SAT})$, which most UCB-type algorithms match in terms of the $\sqrt{T}$ worst-case bound.

**Limitations of Prior Work**: In practice, when positive sub-optimal gaps exist (where the optimal action is better than others by a margin), algorithm performance is significantly better than worst-case bounds, leading to gap-dependent regret. However, existing model-free gap-dependent results are "coarse": Yang et al. (2021) proved a bound of $\tilde{O}(H^6SA/\Delta_{\min})$ for UCB-Hoeffding, which relies on the global minimum gap $\Delta_{\min}$ and a coarse $SA/\Delta_{\min}$ term rather than individual gaps $\Delta_h(s,a)$, making it loose.

**Key Challenge**: The key to obtaining fine-grained bounds is controlling the "cumulative weighted estimation error" of Q-estimates. However, the visit patterns of optimal versus sub-optimal state-action pairs are vastly different—sub-optimal pairs are typically visited only $\hat{O}(\log T)$ times, while optimal pairs may be visited infinitely. Existing analyses treat all $(s,a)$ pairs equally, inevitably relaxing the bound and being overly conservative regarding the dependence on $1/\Delta_{\min}$. Furthermore, AMB, the only non-UCB algorithm to achieve a fine-grained bound (via ULCB + multi-step bootstrap), contains theoretical flaws, leaving it questionable whether fine-grained bounds can be strictly proven in a non-UCB framework.

**Goal**: To answer this open question: Can a strict, fine-grained gap-dependent regret upper bound with better dependence on $1/\Delta_{\min}$ be established for model-free RL using individual gaps $\Delta_h(s,a)$?

**Key Insight**: Starting from the observation that "optimal versus sub-optimal visit frequencies are highly unbalanced," the authors advocate for **separate treatment** of the two types of pairs in the analysis rather than uniform relaxation. This aligns with bounds already achieved in the model-based setting, formulated as $\tilde{O}\big(\sum_{\Delta_h(s,a)>0}\frac{1}{\Delta_h(s,a)}+\frac{|Z_{\mathrm{opt}}|}{\Delta_{\min}}+SA\big)\mathrm{poly}(H)$.

**Core Idea**: Replace uniform relaxation with an induction-based analysis framework that controls cumulative weighted estimation errors on a "per-$(s,a)$, per-step $h$" basis, thereby precisely attributing regret to each individual gap.

## Method

### Overall Architecture
The paper is essentially an analytical framework paper. Total logic follows: "Establish a general framework -> Use the framework to solve an old algorithm -> Use the framework to fix a buggy algorithm."

The first step establishes the **Fine-Grained Analysis Framework**: An identity applicable to any algorithm is used to write expected regret as a sum of weighted visits. Then, the "cumulative weighted estimation error" is recursively propagated per state-action pair and step, followed by induction over steps to obtain fine-grained visit upper bounds. The second step applies this framework to the classic **UCB-Hoeffding** (Jin et al. 2018), yielding its first fine-grained gap-dependent bound (Theorem 3.1). The third step turns to **AMB**, the only non-UCB algorithm: after identifying two critical flaws in algorithm design (improper truncation) and theoretical analysis (violation of martingale difference conditions), the authors provide two fixed versions—**ULCB-Hoeffding**, which removes multi-step bootstrap and keeps only confidence interval elimination, and **Refined AMB**, which keeps multi-step bootstrap but completes the unbiasedness proof. Both versions achieve strict fine-grained bounds and empirically outperform the original AMB.

Since the method centers on mathematical analysis and algorithm rewriting rather than a multi-stage data pipeline, no architecture diagram is provided.

### Key Designs

**1. Fine-Grained Analysis Framework: Separate Counting of Optimal and Sub-optimal Pairs**

Addressing the "loose bounds due to uniform relaxation" pain point, the framework begins with an identity from Lemma 3.1 valid for any algorithm:
$$\mathbb{E}[\mathrm{Regret}(T)] = \mathbb{E}\Big[\sum_{h=1}^{H}\sum_{s,a}\Delta_h(s,a)\,N^{K+1}_h(s,a)\Big],$$
where $\Delta_h(s,a):=V^\star_h(s)-Q^\star_h(s,a)$ is the sub-optimal gap and $N^{K+1}_h(s,a)$ is the visit count. Controlling regret is equivalent to controlling weighted visits, which can be bounded by the cumulative weighted estimation error $\sum_k \omega^k_h(Q^k_h-Q^\star_h)(s^k_h,a^k_h)$ (using the optimism corollary $(Q^k_h-Q^\star_h)(s^k_h,a^k_h)\ge\Delta_h(s^k_h,a^k_h)$).

The key innovation is **no longer treating all $(s,a)$ pairs as a single group**: the authors define recursive weights $\omega_h(k',h')$ for each pair at each step $h'$ and maintain individual contribution terms $\sqrt{H^3\|\omega_h(\cdot,h')\|_\infty\|\omega_h(\cdot,h',s,a)\|_1\,\iota}$ (Lemma 3.2, 3.3), rather than applying Cauchy–Schwarz to all pairs simultaneously as in Yang et al. (2021). During induction (Lemma 3.4), pairs are partitioned into the optimal set $Z_{\mathrm{opt},h}$ ($\Delta_h(s,a)=0$) and the sub-optimal set $Z_{\mathrm{sub},h}$ ($\Delta_h(s,a)>0$). Cauchy–Schwarz is applied per-step for optimal pairs and cross-step for sub-optimal pairs. This specifically exploits the fact that "sub-optimal pairs are only visited $\hat{O}(\log T)$ times," allowing regret to be decomposed into the sum of reciprocals of individual gaps rather than being dominated by the global $1/\Delta_{\min}$.

**2. Solving UCB-Hoeffding: The First Fine-Grained Upper Bound**

UCB-Hoeffding uses standard Bellman updates (learning rate $\eta_t=(H+1)/(H+t)$, Hoeffding bonus $b_t=2\sqrt{H^3\iota/t}$) with greedy action selection. The algorithm itself requires no modification; the difficulty lies solely in the analysis. Applying the framework from Design 1, Theorem 3.1 gives the expected regret bound:
$$O\Big(\sum_{h=1}^{H}\sum_{\Delta_h(s,a)>0}\frac{H^5\log(SAT)}{\Delta_h(s,a)}+\sum_{h=1}^{H}\frac{H^3\big(\sum_{t=h+1}^{H}\sqrt{|Z_{\mathrm{opt},t}|}\big)^2\log(SAT)}{\Delta_{\min,h}}+SAH^3\Big),$$
where $Z_{\mathrm{opt},h}=\{(s,a):\Delta_h(s,a)=0\}$. This can be further relaxed to $\sum_{\Delta_h(s,a)>0}\frac{H^5\log(SAT)}{\Delta_h(s,a)}+\frac{H^5|Z_{\mathrm{opt}}|\log(SAT)}{\Delta_{\min}}+SAH^3$. This is strictly superior to the $\tilde{O}(H^6SA/\Delta_{\min})$ from Yang et al. (2021): in an ideal case with only one sub-optimal triplet (at $h=H$), it reduces to $\tilde{O}(H^5/\Delta_{\min})$; in the worst case where all gaps equal $\Delta_{\min}$, it gracefully degrades to match old bounds. Its dependence on $|Z_{\mathrm{opt}}|/\Delta_{\min}$ also matches the lower bounds of Simchowitz & Jamieson (2019) and Xu et al. (2021) (up to polynomial factors of $H$).

**3. ULCB-Hoeffding: UCB Improvement with Confidence Interval Elimination and No Multi-step Bootstrap**

The difficulty in analyzing AMB stems from multi-step bootstrapping. The authors first propose a "subtractive" improvement: ULCB-Hoeffding maintains both an upper bound $\overline{Q}^k_h$ and a lower bound $\underline{Q}^k_h$ for $Q^\star_h$ using **standard Bellman updates** (upper bound plus bonus, lower bound minus bonus), avoiding multi-step bootstrapping. Exploration relies on two mechanisms: first, maintaining a candidate action set $A^k_h(s)$, where any action satisfying $\overline{Q}^{k+1}_h(s,a)<\underline{V}^{k+1}_h(s)$ is deemed sub-optimal and eliminated (since $Q^\star_h(s,a)\le\overline{Q}^{k+1}_h(s,a)<\underline{V}^{k+1}_h(s)\le\underline{Q}^{k+1}_h(s,a')\le Q^\star_h(s,a')$ exists for some $a'$); second, selecting the action with the **widest confidence interval** $\arg\max_{a\in A^k_h(s)}(\overline{Q}^k_h-\underline{Q}^k_h)(s,a)$, i.e., the action with the highest uncertainty. Because update rules return to standard Bellman, the framework from Design 1 applies directly. Theorem 4.2 proves it achieves the **exact same** fine-grained bound as UCB-Hoeffding, while Theorem 4.1 ensures it remains $O(\sqrt{H^4SAT\iota})$ in the worst case. This demonstrates that the ULCB principle alone, without multi-step bootstrapping, is sufficient for fine-grained guarantees.

**4. Refined AMB: Correcting Truncation and Martingale Difference for Unbiased Multi-step Bootstrapping**

This step addresses AMB's two major flaws. The first is **improper truncation at the algorithmic level**: AMB truncates Q-estimates at $H$ and $0$ within the multi-step bootstrap update (Algorithm 1, lines 13–14), which breaks the recursive structure linking Q-estimates to historical V-estimates (Eq. (A.5)), the core of proving optimism/pessimism. The second is **violation of martingale difference conditions in analysis**: AMB splits the Q-function into "cumulative rewards along fixed optimal actions" and "rewards collected from the first undetermined optimal action." When controlling deviation with Azuma–Hoeffding, it ignores the randomness of the bootstrap step, incorrectly centering the estimator on its unconditional expectation rather than its conditional expectation, violating the martingale difference condition.

Refined AMB retains both ULCB and multi-step bootstrapping with four corrections: (a) moving truncation from the Q-estimate to the corresponding V-estimate to preserve the recursive structure; (b) strictly proving the sum of the two multi-step bootstrap estimators is an unbiased estimate of $Q^\star$ via **state-wise decomposition** of conditional expectations and induction over steps (Core technique of Theorem 4.3 / Appendix G.3); (c) centering estimators on true conditional expectations to restore the martingale condition; (d) jointly analyzing the concentration of both estimators to tighten confidence intervals and halve the bonus constant. Ultimately, Refined AMB achieves $O\big(\sum_{\Delta_h(s,a)>0}\frac{H^5\log(SAT)}{\Delta_h(s,a)}+\frac{H^5|Z_{\mathrm{mul}}|\log(SAT)}{\Delta_{\min}}\big)$, where $Z_{\mathrm{mul}}$ is the set of optimal triplets where multiple optimal actions exist, consistently outperforming the original AMB empirically.

## Key Experimental Results

Experiments are conducted in synthetic environments with rewards $r_h(s,a)\sim\mathrm{Unif}[0,1]$ and transition kernels sampled uniformly from the probability simplex. The four algorithms—AMB, Refined AMB, UCB-Hoeffding, and ULCB-Hoeffding—are compared by plotting $\mathrm{Regret}(T)/\log(K+1)$ against $K$ (10 trajectories per algorithm, median + 10th/90th percentile bands).

### Main Results (Relative performance across four scales)

| Scale $(H,S,A,K)$ | Performance Ranking (Regret Low to High) | Key Observation |
|-------------------|------------------------------------------|-----------------|
| $(2,3,3,10^5)$ | UCB-Hoeffding < ULCB-Hoeffding ≈ Refined AMB < AMB | Curves (except AMB) flatten as $K$ increases |
| $(5,5,5,6\times10^5)$ | Same as above | Reflects logarithmic growth, consistent with theory |
| $(7,8,6,5\times10^6)$ | Same as above | Improved versions consistently outperform original AMB |
| $(10,15,10,2\times10^7)$ | Same as above | UCB-Hoeffding is overall best |

### Comparison of Key Findings

| Configuration | Key Observation | Explanation |
|---------------|-----------------|-------------|
| UCB-Hoeffding | Lowest overall regret | Standard Bellman + framework analysis; best empirical performance |
| ULCB / Refined AMB | Comparable; both better than AMB | The two fixed versions show similar performance |
| AMB (Original) | Regret curve does not flatten | Bonus constant $c=2$ (concentrating two estimators separately) leads to looser bounds |

### Key Findings
- Except for AMB, the $\mathrm{Regret}(T)/\log(K+1)$ curves for all algorithms flatten as $K$ increases, indicating logarithmic regret growth consistent with fine-grained gap-dependent theory.
- Refined AMB reduces the bonus constant from $c=2$ to $c=1$ (using joint concentration once instead of once per estimator), which is the direct reason it empirically outperforms the original AMB.
- UCB-Hoeffding performs best across all scales, suggesting that in these tabular tasks, standard Bellman + greedy selection is not necessarily inferior to complex multi-step bootstrapping and is often better due to smaller bonuses.

## Highlights & Insights
- **"Separate counting of visit frequencies" is the core theme**: Exploiting the asymmetry that "sub-optimal pairs are visited $O(\log T)$ times while optimal pairs can be visited infinitely" by switching from uniform relaxation to per-$(s,a)$, per-step induction is the key technical lever for moving from coarse to fine-grained bounds.
- **Converting regret to visit counts via an identity** (Lemma 3.1, valid for any algorithm) allows the "regret bound proof" to be cleanly reduced to a "visit count bound proof," making the framework easily transferable from UCB-Hoeffding to ULCB-Hoeffding.
- **Rigorous verification of prior work**: The authors precisely identify that AMB's truncation destroys recursive structures and that centering on unconditional expectations violates martingale difference conditions—providing both the "bug report" and a "working replacement"—a high-value contribution to the theory community.
- **State-wise conditional expectation decomposition for multi-step bootstrap unbiasedness** (Theorem 4.3) is a reusable technique for any algorithm analysis involving multi-step bootstrapping.

## Limitations & Future Work
- Restricted to tabular, finite-state episodic MDPs; extension of the fine-grained framework to function approximation or continuous states is not addressed.
- Polynomial dependence on $H$ (e.g., $H^5$) in the regret bound may still be loose, leaving a polynomial gap with lower bounds.
- Small-scale synthetic MDP experiments lack validation on larger state spaces or real-world tasks; it is unclear if "UCB-Hoeffding is empirically best" holds in more complex environments.
- The complete algorithm and unbiasedness proof for Refined AMB are placed in the appendix, with only informal versions provided in the main text (Theorem 4.3), requiring a search in Appendix G for implementation details.

## Related Work & Insights
- **vs Yang et al. (2021)**: They provide a coarse $\tilde{O}(H^6SA/\Delta_{\min})$ for UCB-Hoeffding depending on global $SA/\Delta_{\min}$; this work changes it to individual gaps $\sum 1/\Delta_h(s,a)$, performing no worse in all cases and significantly better in ideal ones.
- **vs AMB from Xu et al. (2021)**: AMB was the first non-UCB algorithm claiming a fine-grained bound but had flaws in truncation and martingale difference; this work identifies these and provides ULCB-Hoeffding (no bootstrap) and Refined AMB (fixed bootstrap) as strictly provable alternatives.
- **vs Model-based fine-grained work (Simchowitz & Jamieson 2019; Dann et al. 2021; Chen et al. 2025)**: Model-based methods already achieved bounds like $\sum 1/\Delta_h(s,a)+|Z_{\mathrm{opt}}|/\Delta_{\min}+SA$; this work brings the same level of granularity to the **model-free** setting for the first time and matches UCB-type lower bounds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First fine-grained gap-dependent bound for model-free UCB-Hoeffding and rectifies long-standing unnoticed flaws in AMB.
- Experimental Thoroughness: ⭐⭐⭐ Sufficient for a theory paper but limited to small synthetic environments.
- Writing Quality: ⭐⭐⭐⭐ Clear logic across framework, algorithms, and fixes, though many details are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Provides a generalizable framework for model-free gap-dependent analysis and a reusable technique for proving bootstrap unbiasedness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)
- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[CVPR 2026\] Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](../../CVPR2026/reinforcement_learning/specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [\[ICML 2026\] Data- and Variance-dependent Regret Bounds for Online Tabular MDPs](../../ICML2026/reinforcement_learning/data-_and_variance-dependent_regret_bounds_for_online_tabular_mdps.md)
- [\[ICLR 2026\] Bridging the Performance-Gap Between Target-Free and Target-Based Reinforcement Learning](bridging_the_performance-gap_between_target-free_and_target-based_reinforcement_.md)

</div>

<!-- RELATED:END -->
