---
title: >-
  [Paper Note] Diversified Multinomial Logit Contextual Bandits
description: >-
  [ICLR2026][Learning Theory][MNL bandit] This paper directly embeds "combinatorial diversity" into the selection probabilities of the Multinomial Logit (MNL) model. It proposes the DMNL contextual bandit model and designs OFU-DMNL, a white-box UCB algorithm that eliminates the need for a black-box optimization oracle. By constructing assortments through item-wise greedy selection with $O(NK)$ per-round overhead, the authors prove a $(1-\frac{1}{e+1})$-approximate regret bound…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Online Learning"
  - "Combinatorial Bandits"
  - "MNL bandit"
  - "diversity"
  - "submodular functions"
  - "approximate regret"
  - "UCB"
date: 2026-05-08
content_hash: fcd4117b1e0cf2e0
---

# Diversified Multinomial Logit Contextual Bandits

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=swwelQtLRn](https://openreview.net/forum?id=swwelQtLRn)  
**Code**: To be confirmed  
**Area**: Learning Theory / Online Learning / Combinatorial Bandits  
**Keywords**: MNL bandit, diversity, submodular functions, approximate regret, UCB

## TL;DR
This paper directly embeds "combinatorial diversity" into the selection probabilities of the Multinomial Logit (MNL) model. It proposes the DMNL contextual bandit model and designs OFU-DMNL, a white-box UCB algorithm that eliminates the need for a black-box optimization oracle. By constructing assortments through item-wise greedy selection with $O(NK)$ per-round overhead, the authors prove a $(1-\frac{1}{e+1})$-approximate regret bound of $\tilde{O}(d\sqrt{T/K})$.

## Background & Motivation

**Background**: In sequential combinatorial recommendation (e.g., e-commerce product slots, streaming playlists, app store sections), a platform displays an assortment of items in each round. The user selects at most one item, and the platform learns online based on feedback. The classic model linking "displayed assortment $\to$ user choice probability" is the Multinomial Logit (MNL): each item has a relevance utility $x_{ti}^\top\theta^*$, and users choose within the set (plus a "no-choice" outside option) according to a softmax distribution. The advantages of MNL include solvable combinatorial optimization (simply selecting the top-$K$ items by relevance under uniform revenue) and clean statistical guarantees, which have led to numerous works on MNL assortment bandits and their contextual variants.

**Limitations of Prior Work**: Practical product selection rarely relies on relevance alone—**diversity** within the assortment is equally critical. A set of near-duplicate items results in cannibalization, where additional slots provide almost no extra value. Conversely, assortments with complementary categories, brands, or styles are often preferred. However, standard MNL choice probabilities are only derived from individual item utilities; **interactions between items in a set (cannibalization due to similarity or complementary effects) are not modeled at all**.

**Key Challenge**: There is an inherent trade-off between relevance and diversity—a diverse but irrelevant set will still perform poorly, so diversity cannot replace relevance; both must be considered simultaneously. Another modeling route is submodular/combinatorial bandits, which use a submodular reward function to characterize diversity (diminishing marginal returns, encouraging coverage). However, these models define rewards as set functions, **losing the core feedback mechanism of assortment scenarios—where users select at most one item based on a structured discrete choice model**—and they fail to couple relevance utility and diversity effects into a single probabilistic choice model. This creates a modeling gap between "diversity-aware assortment selection" and "relevance-based, choice-model-driven dynamic assortment learning."

**Goal**: This paper aims to characterize both relevance and diversity within a single probabilistic choice model while providing end-to-end statistical and computational guarantees. The difficulty lies in the fact that once diversity is embedded into choice probabilities, the "exact combinatorial optimization" property of classic MNL is lost—the optimal set may require an exhaustive search of $\binom{N}{K}$, which is computationally infeasible even if parameters are known. Many combinatorial bandit works bypass this by assuming a "$\gamma$-approximation black-box oracle," but this assumption is often unrealistic and masks the source of the approximation.

**Key Insight**: The authors observe that the MNL reward function itself possesses a monotonic submodular structure over sets. Item-wise greedy selection for submodular functions naturally yields a $(1-\frac1e)$ approximation. This means one can **avoid relying on a black-box oracle by constructing a transparent item-wise greedy oracle** and directly proving its approximation ratio. By further exploiting the specific structure of MNL, the $(1-\frac1e)$ ratio can be improved to $(1-\frac{1}{e+1})$.

**Core Idea**: A general submodular diversity function $g_t(S)$ is used to **modulate the weight of the MNL outside option**, formalizing the relevance-diversity trade-off within the choice probability (the DMNL model). Then, an item-wise optimistic construction is used to replace the black-box oracle, achieving white-box, provably approximate, and computationally efficient online learning.

## Method

### Overall Architecture
The objective of the DMNL bandit is to receive item features $X_t=\{x_{t1},\dots,x_{tN}\}$ and a diversity measure $g_t$ at each round $t$, then select a set $S_t$ of size $\le K$ to minimize the cumulative regret over $T$ rounds. The process consists of three components: (1) **DMNL Choice Model**, which embeds submodular diversity into MNL probabilities to define the choice probability and expected reward; (2) **Item-wise Greedy Approximation Oracle**, which proves that greedy construction achieves a $(1-\frac{1}{e+1})$ approximation—better than general submodular bounds—under known parameters; (3) **OFU-DMNL Online Algorithm**, which uses Online Mirror Descent to estimate parameters, constructs optimistic rewards, and builds the set via item-wise optimistic construction, triggering adaptive exploration when the diversity parameter's confidence is insufficient. The resulting approximate regret upper bound matches the lower bound of MNL bandits.

### Key Designs

**1. DMNL Choice Model: Modulating the outside option with submodular diversity to embed the relevance-diversity trade-off.**

Standard MNL probabilities depend only on single-item utilities, failing to capture set interactions. Ours introduces a monotonic submodular diversity function $g_t:S\to\mathbb{R}_{\ge0}$ that acts solely on the "outside option" ($i=0$). The choice probability is defined as:
$$p_t(i\mid S,\theta^*,\lambda^*)=\frac{\exp(x_{ti}^\top\theta^*)}{\exp(-\lambda^* g_t(S))+\sum_{j\in S}\exp(x_{tj}^\top\theta^*)},$$
The outside option probability is $p_t(0\mid S)=\frac{\exp(-\lambda^* g_t(S))}{\exp(-\lambda^* g_t(S))+\sum_{j\in S}\exp(x_{tj}^\top\theta^*)}$, where $\theta^*$ is the relevance parameter and $\lambda^*$ is the diversity parameter (assuming $\lambda^*>l>0$). The intuition is clear: if two sets have identical item utilities, the set with the higher diversity score $g_t(S)$ will suppress the weight of the outside option $\exp(-\lambda^* g_t(S))$, thereby increasing the choice probability of each item in the set and raising the expected reward. The expected reward for set $S$ is $R_t(S,\theta^*,\lambda^*)=\sum_{i\in S}p_t(i\mid S,\theta^*,\lambda^*)$. This paper focuses on the **uniform revenue** ($r_{ti}=1$) setting, where increasing diversity is directly equivalent to increasing the total click probability. This model strictly generalizes standard MNL, which is recovered when $g_t$ is constant across all sets.

**2. Item-wise Greedy Approximation Oracle: Improving the submodular bound from $1-\frac1e$ to $1-\frac{1}{e+1}$ via MNL structure.**

With diversity, the optimal set requires a $\binom{N}{K}$ search, making approximation necessary. Instead of a black-box oracle, this paper explicitly constructs an item-wise greedy approach: at step $k$, add the item with the maximum marginal gain $a_k=\arg\max_{a}R_t(\{a_1,\dots,a_{k-1}\}\cup\{a\})$. The key is proving that $R_t(S)$ is monotonically submodular. By writing the reward as $R_t(S)=\frac{\exp(f_t(S))}{1+\exp(f_t(S))}$, where $f_t(S)=\log\sum_{i\in S}\exp(x_{ti}^\top\theta^*)+\lambda^* g_t(S)$ is the sum of LogSumExp (submodular) and a submodular $g_t$ (thus submodular), the composition with the non-decreasing concave function $\frac{e^x}{1+e^x}$ maintains submodularity. This ensures the $(1-\frac1e)$ bound. Using the specific structure of MNL rewards, Theorem 1 proves a stronger bound: $R_t(S^{\text{greedy}})\ge\frac{\psi_0(1+\psi_0^\alpha)}{\psi_0^\alpha(1+\psi_0)}R_t(S^*)$, where $\alpha=\frac{e}{e-1}$ and $\psi_0$ solves $x^\alpha=\alpha x+\alpha-1$; the lower bound is $1-\frac{1}{e+1}$, strictly better than the $1-\frac1e$ bound for general submodular functions.

**3. OFU-DMNL: White-box UCB with augmented features, Online Mirror Descent, and item-wise optimistic construction.**

By combining unknown parameters into $w^*=[\theta^*,\lambda^*]$ and augmenting features with diversity as $z_{ti}(S)=[x_{ti},g_t(S)]$, the DMNL probability takes the single-parameter MNL form $p_t(i\mid S,w^*)=\frac{\exp(z_{ti}(S)^\top w^*)}{1+\sum_{j\in S}\exp(z_{tj}(S)^\top w^*)}$. Parameter estimation follows Lee & Oh (2024): Online Mirror Descent on the MNL loss $\ell_t(w)=-\sum_i y_{ti}\log p_t(i\mid S_t,w)$ updates $w_{t+1}$, and a Hessian-based cumulative matrix $H_t$ provides the confidence radius $\|w_t-w^*\|_{H_t}\le\alpha_t$. This allows the construction of optimistic utility $\text{ucb}(z_{ti}(S))=z_{ti}(S)^\top w_t+\alpha_t\|z_{ti}(S)\|_{H_t^{-1}}$ and optimistic reward $\tilde R_t(S)$. Assortments are then built via item-wise optimistic construction $a_k=\arg\max_a\tilde R_t(\{a_1,\dots,a_{k-1}\}\cup\{a\})$ with a per-round cost of $O(NK)$.

**4. Adaptive Exploration: Random exploration when diversity parameter confidence is insufficient.**

Since $\theta$ and $\lambda$ are estimated jointly via augmented features, their individual uncertainties are coupled. If the joint confidence width for $\lambda^*$ is not tight enough, the "optimism" of the greedy construction isn't guaranteed. Thus, a trigger condition is added: if $\|[0_d,1]\|_{H_t^{-1}}\ge\frac{\nu}{\hat\lambda_t}\alpha_t$ (meaning uncertainty in the diversity direction $[0_d,1]$ is too high), the algorithm performs uniform random exploration for one round. The authors prove that the number of such exploration rounds is at most $O(\sqrt{d}\log T)$, preserving the overall regret order.

### Loss & Training
Parameters are estimated by minimizing the MNL loss $\ell_t(w)=-\sum_{i\in S_t}y_{ti}\log p_t(i\mid S_t,w)$ via Online Mirror Descent:
$$w_{t+1}=\arg\min_{w\in\mathcal W}\Big\{\langle\nabla\ell_t(w_t),w\rangle+\tfrac{1}{2\eta}\|w-w_t\|_{\tilde H_t}^2\Big\},$$
where $\mathcal W=\{w:\|w\|_2\le1\}$, $\tilde H_t=H_t+\eta G_t(w_t)$, $H_t=\Lambda I_{d+1}+\sum_{s<t}G_s(w_{s+1})$, and $G_t(w)$ is the Hessian induced by MNL probabilities. Hyperparameters are set as $\alpha_t=O(\sqrt{d\log t}\log K)$, $\eta=\frac12\log(K+1)+2$, $\Lambda=84\sqrt{(d+1)\eta}$, and $\nu=\frac\omega2$, without requiring knowledge of $l$.

## Key Experimental Results

### Main Results
Synthetic experiments: Contextual features are sampled from $\mathcal N(0_d,I_d)$ and clipped to $[-1/\sqrt d,1/\sqrt d]^d$. Items are assigned categories, and the diversity function uses "exponentially decaying category coverage." Diversity parameter $\lambda^*$ is fixed at various values, and $\theta^*$ is sampled uniformly such that $\|\theta^*\|_2+\lambda^*=1$. Baseline comparisons include UCB-MNL, TS-MNL, OFU-MNL+, as well as two modified versions: OFU-MNL-DR (standard MNL model but with an additive submodular reward term needing a tuned $\lambda$) and OFU-DMNL-FULL (exact DMNL with exhaustive search per round, cost approx $O((eN/K)^K)$).

| Setting (N, K, d, λ*) | Cumulative Regret Performance | Runtime |
|--------|------|------|
| N=10/20, K=5, d=5, λ*=0.4 | OFU-DMNL significantly lower than pure MNL and OFU-MNL-DR | OFU-DMNL ≪ OFU-DMNL-FULL |
| N=50/100, K=5/10, d=5, λ*=0.4| OFU-DMNL regret comparable to exhaustive OFU-DMNL-FULL | OFU-DMNL $O(NK)$, FULL explodes with N, K |

Core conclusion: OFU-DMNL outperforms baselines that ignore or manually tune diversity and **approaches the regret of the exhaustive optimal solution while maintaining a significantly lower runtime**.

### Ablation Study

| Comparator | Difference from Ours | Description |
|------|---------|------|
| OFU-MNL+ | Does not model diversity | Suffers in DMNL environments due to ignoring diversity |
| OFU-MNL-DR | Std MNL + additive submodular reward | Requires manual weight tuning; diversity not coupled in choice model |
| OFU-DMNL-FULL| Exact DMNL with exhaustive search | Regret acts as an upper reference; computationally infeasible |
| OFU-DMNL (Ours)| Item-wise optimistic + joint learning $\lambda$ | Parameter-free, $O(NK)$, approximately optimal |

### Key Findings
- Across different relevance-diversity ratios controlled by $\lambda$, OFU-DMNL remains robust, indicating it adapts to various trade-off regimes.
- Compared to the hand-tuned OFU-MNL-DR, jointly learning $\lambda^*$ eliminates manual tuning and modeling diversity within the selection probability (rather than as a reward add-on) better fits user behavior.
- Regret is nearly identical to the exhaustive FULL version, quantitatively validating that the item-wise optimistic construction is both statistically and computationally efficient.

## Highlights & Insights
- **Diversity in Probabilities, Not Rewards**: Placing $g_t$ on the outside option weight ensures "higher diversity $\to$ less attractive outside option $\to$ higher choice probability for items." This preserves the discrete choice mechanism and naturally models the relevance-diversity trade-off.
- **White-box vs. Black-box Oracle**: By proving the submodularity of DMNL rewards (composition of LogSumExp and concave functions), the greedy approach becomes a provable oracle. Leveraging MNL structure to improve the ratio to $1-\frac{1}{e+1}$ is a significant theoretical contribution.
- **Unified Estimation with Augmented Features**: Folding the diversity parameter into the feature vector $z_{ti}(S)$ allows the use of minimax-optimal MNL estimators, ensuring that the regret $\tilde O(d\sqrt{T/K})$ does not worsen in order despite learning an extra parameter.
- **Adaptive Exploration for Coupled Parameters**: When uncertainty between $\theta$ and $\lambda$ cannot be decoupled, using a specific confidence criterion in the diversity direction is a practical way to ensure optimism.

## Limitations & Future Work
- **Limited to Uniform Revenue**: Ours only studies $r_{ti}=1$, as high-revenue items dominate the optimal set in non-uniform settings, potentially conflicting with diversity goals.
- **Strong Structural Assumptions**: Requires non-degeneracy and $\omega$-strictly submodular assumptions. In reality, diversity measures may not be strictly submodular, and the assumption $\lambda^*>l>0$ excludes cases where diversity is irrelevant.
- **Approximation vs. Exact Optimality**: Because exhaustive search is infeasible, the target is $\gamma$-approximate regret, resulting in an inherent $\gamma$-factor gap compared to exact regret.
- **Synthetic Data**: Validation is limited to synthetic data; the performance on real-world recommendation datasets and with different submodular measures remains to be tested.

## Related Work & Insights
- **vs. Classic MNL Assortment Bandits**: Those models rely on top-$K$ selection. Ours adds diversity to the probability, necessitating approximation, but matching their regret order of $\tilde O(d\sqrt{T/K})$.
- **vs. Submodular/Combinatorial Bandits**: Such models often use set-defined rewards and black-box oracles. Ours uses discrete choice feedback (only the final choice is observed) and constructs a white-box oracle.
- **vs. OFU-MNL-DR**: DR uses an additive diversity term. Ours couples diversity within the choice model and learns it jointly, proving superior in both self-consistency and experimental results.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to embed diversity directly into MNL choice probabilities and replace black-box oracles with a provable white-box greedy approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive synthetic experiments including exhaustive comparisons, though real-world data is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Very clear progression from model to approximation, algorithm, and regret analysis.
- Value: ⭐⭐⭐⭐ Provides a theoretically grounded and computationally efficient paradigm for diversity-aware assortment optimization under uncertainty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Queue Length Regret Bounds for Contextual Queueing Bandits](queue_length_regret_bounds_for_contextual_queueing_bandits.md)
- [\[ICLR 2026\] Efficient Best-of-Both-Worlds Algorithms for Contextual Combinatorial Semi-Bandits](efficient_best-of-both-worlds_algorithms_for_contextual_combinatorial_semi-bandi.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](../../ICML2026/learning_theory/optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICLR 2026\] Contextual Multi-Armed Bandits with Minimum Aggregated Revenue Constraints](contextual_multi-armed_bandits_with_minimum_aggregated_revenue_constraints.md)

</div>

<!-- RELATED:END -->
