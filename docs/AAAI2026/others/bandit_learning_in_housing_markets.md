---
title: >-
  [Paper Note] Bandit Learning in Housing Markets
description: >-
  [AAAI 2026][multi-armed-bandit] This paper is the first to introduce the multi-armed bandit (MAB) framework into housing markets (one-sided matching markets). It defines regret based on the core solution concept, proposes a decentralized ETC algorithm and a centralized UCB algorithm, proves a decentralized regret upper bound of $\mathcal{O}(N\log T / \Delta_{\min}^2)$ along with a matching lower bound, and establishes order-optimality.
tags:
  - AAAI 2026
  - multi-armed-bandit
  - housing-market
  - matching
  - online-learning
  - game-theory
date: 2026-05-08
content_hash: d77f69d85470faef
---

# Bandit Learning in Housing Markets

**Conference**: AAAI 2026
**arXiv**: [2511.12629](https://arxiv.org/abs/2511.12629)
**Code**: None
**Area**: Other
**Keywords**: multi-armed-bandit, housing-market, matching, online-learning, game-theory

## TL;DR

This paper is the first to introduce the multi-armed bandit (MAB) framework into housing markets (one-sided matching markets). It defines regret based on the core solution concept, proposes a decentralized ETC algorithm and a centralized UCB algorithm, proves a decentralized regret upper bound of $\mathcal{O}(N\log T / \Delta_{\min}^2)$ along with a matching lower bound, and establishes order-optimality.

## Background & Motivation

### The Housing Market Model

The housing market is a classical one-sided matching market model introduced by Shapley and Scarf (1974). In this model, $N$ agents each own an indivisible object ("house") and hold strict preference orderings over all objects. The goal is to find a **core-stable allocation**, i.e., one in which no subset of agents can all improve by trading internally. The well-known Top Trading Cycle (TTC) algorithm efficiently computes the unique core matching when preferences are known.

### Core Pain Point: Unknown Preferences

Traditional housing market research assumes each agent has complete knowledge of their preferences over all objects, but this assumption often fails in practice:

- **Rental exchange platforms** (e.g., LeaseSwap NYC): tenants have not personally experienced all listings.
- **Kidney exchange**: compatibility between organ pairs must be confirmed incrementally through medical testing.
- **NFT trading**: traders face uncertainty about the value of digital assets they have not yet experienced.

A common feature of these settings is that agents can learn preferences gradually through **repeated short-term interactions** (e.g., viewings, compatibility tests, temporary permits) with immediate feedback, making them naturally suited to the bandit learning paradigm.

### Distinction from Prior Work

Existing work on online matching learning primarily focuses on **two-sided matching markets** (bipartite matching, e.g., stable matching under the Gale-Shapley framework), where both sides have preferences and multiple stable matchings may exist. This paper studies **one-sided matching markets**, which differ in three key respects:

1. Only agents have preferences over objects; objects have no preferences.
2. Collision rule: when multiple agents compete for the same object, **all** receive zero reward (rather than selecting the best agent according to object preferences).
3. The core matching is **unique**, so the regret definition is unambiguous.
4. The benchmark is the game-theoretic "core" rather than a maximum-weight matching.

## Method

### Problem Formulation

**Model setup:** $N$ players $\mathcal{N} = \{p_1, \ldots, p_N\}$, $N$ arms $\mathcal{A} = \{a_1, \ldots, a_N\}$. Player $p_i$ initially owns arm $a_i$. The utility matrix $\bm{U}(i,j) \in [0,1]$ encodes player $p_i$'s preference for arm $a_j$ (assumed to be strictly distinct).

**Interaction protocol:** At each round $t$, every player selects an arm. If arm $a_j$ is chosen by exactly one player, that player receives a 1-subgaussian random reward with mean $\bm{U}(i,j)$; if multiple players select the same arm, a collision occurs and all receive reward 0.

**Core regret definition:** Using the unique TTC output $\mu^*$ as the benchmark, the core regret of player $p_i$ is:

$$Reg_i(T) = T \cdot \bm{U}(i, \mu^*(p_i)) - \mathbb{E}\left[\sum_{t=1}^T X_i(t)\right]$$

### Overall Architecture

The paper proposes two algorithms corresponding to decentralized and centralized market settings, respectively:

| Dimension | Decentralized ETC (Algorithm 1) | Centralized UCB (Algorithm 2) |
|---|---|---|
| Requires platform | No | Yes (aggregates preference rankings and computes matching) |
| Requires knowledge of $T$ | Yes | No (anytime algorithm) |
| Regret upper bound | $\mathcal{O}(N\log T / \Delta_{\min}^2)$ | $\mathcal{O}(N^2\log T / \Delta_{\min}^2)$ |
| Order-optimality | Yes (matches lower bound) | Loose by a factor of $N$ |

### Algorithm 1: Decentralized Explore-then-YRMH-IGYT

The algorithm proceeds in two phases:

**Phase 1 — Exploration and Preference Learning:** Consists of sub-phases $\ell = 1, 2, \ldots$, each of length $2^\ell + N$ rounds.

- **Exploration stage** ($2^\ell$ rounds): Players pull arms in a round-robin fashion. Since each player has a unique index, round-robin guarantees **zero collisions**. Players use collected samples to update utility estimates $\hat{\bm{U}}(i,j)$ and UCB/LCB confidence intervals.
- **Communication stage** ($N$ rounds): Players use their initially owned arms as broadcast channels to implicitly exchange information. Player $p_j$ submits a request to player $p_i$'s arm if $p_j$ has already learned its preference ranking, and abstains otherwise. When $p_i$ observes requests from all $N$ players, it infers that all players have completed preference learning and triggers entry into Phase 2.
- **Termination condition:** For all arm pairs $(a_j, a_{j'})$, the condition $\text{LCB}_{i,\sigma_k} > \text{UCB}_{i,\sigma_{k+1}}$ holds, i.e., non-overlapping confidence intervals certify the complete ranking.

**Phase 2 — YRMH-IGYT Decentralized Core Matching:** Players execute a decentralized version of the You Request My House, I Get Your Turn mechanism using their learned preference rankings $\sigma_i$:

1. The unassigned player with the smallest index initiates a proposal chain, requesting their most preferred available arm.
2. The player receiving the proposal in turn requests their own most preferred available arm, forming a chain.
3. When a proposal "loops back" to a player already in the chain, a top trading cycle is identified.
4. All players in the cycle lock in their assignments, and their owned arms are removed from the market.
5. The process repeats until all arms are assigned.

Phase 2 requires at most $N^2$ rounds (at most $N$ epochs, each requiring at most $N$ rounds to identify one cycle).

### Algorithm 2: Centralized Anytime UCB

This algorithm requires no knowledge of the time horizon $T$. At each round:

1. Each player $p_i$ computes a preference ranking based on the UCB index $u^{(t)}(i,j) = \hat{\bm{U}}^{(t)}(i,j) + \sqrt{3\log t / (2T_{i,j}(t-1))}$.
2. The ranking is submitted to a central platform.
3. The platform runs the offline TTC algorithm to compute the current core matching $\mu_t$.
4. Each player pulls their assigned arm and updates their estimate.

The key advantage is **adaptivity** and **collision-free operation** (coordinated by the platform), at the cost of an additional factor of $N$ in regret.

### Regret Lower Bound Construction

For the class of **Single Top Trading Cycle Bandit (STTCB)** instances (where all $N$ players form one large cycle), the KL-divergence method yields:

- For any uniformly consistent policy $\pi$ and any player $p_i$: $\liminf_{T\to\infty} Reg_i(T;\bm{\nu},\pi) / \log T \geq \Omega(N/\Delta_{\min}^2)$.
- Core construction: a player with preference gap $\Delta_{\min}$ requires at least $\Omega(\log T / \Delta_{\min}^2)$ rounds of exploration to identify the optimal arm; the exploration of $N-1$ such players inevitably causes collisions for a specific player, accumulating to $\Omega(N\log T / \Delta_{\min}^2)$ regret.

## Key Experimental Results

This is a purely theoretical paper with no experimental data. The core theoretical results are summarized below.

**Table 1: Summary of Regret Bounds**

| Setting | Algorithm | Regret Upper Bound | Anytime | Requires $T$ |
|---|---|---|---|---|
| Decentralized | ETC + YRMH-IGYT | $\mathcal{O}(N\log T / \Delta_{\min}^2)$ | No | Yes |
| Centralized | UCB + TTC | $\mathcal{O}(N^2\log T / \Delta_{\min}^2)$ | Yes | No |
| Decentralized lower bound | Any consistent policy | $\Omega(N\log T / \Delta_{\min}^2)$ | — | — |

**Table 2: Summary of Technical Lemmas**

| Lemma | Content | Key Conclusion |
|---|---|---|
| Lemma 1 | Probability of bad concentration events | $\mathbb{P}(\mathcal{F}) \leq 2N^2 / T$ |
| Lemma 2 | Convergence speed of Phase 2 | Core matching found in at most $N^2$ rounds |
| Lemma 3 | Duration of Phase 1 | At most $\ell_{\max} = \log(192N\log T / \Delta_{\min}^2)$ sub-phases |
| Lemma 4 | STTCB regret decomposition | Decomposes regret into collision term and suboptimal arm pull term |

## Highlights & Insights

1. **Natural problem formulation:** The paper elegantly bridges the game-theoretic concept of the core with the online learning framework of bandits. The uniqueness of the core matching ensures an unambiguous regret definition, avoiding the benchmark selection issue that arises from multiple stable matchings in two-sided markets.

2. **Collision-free exploration via round-robin:** By exploiting unique indices from initial endowments to design a round-robin strategy, the algorithm completely avoids collisions during exploration — a clean and effective design.

3. **Implicit communication mechanism:** A single bit of information ("whether I have learned my preferences") is conveyed by whether a player submits a request to another's arm, requiring no explicit communication channel and enabling global synchronization in a decentralized setting.

4. **Lower bound construction:** The paper exploits the structure of a single TTC — where each player's exploration necessarily causes collisions for another specific player — to prove the inevitability of collisions. This is a novel element in multi-player bandit lower bound proofs.

5. **Precise centralized vs. decentralized tradeoff:** The decentralized algorithm requires knowledge of $T$ but is order-optimal; the centralized algorithm is anytime but incurs an extra factor of $N$. This tradeoff arises because the platform eliminates collisions but introduces additional slack due to "blocking triplets."

## Limitations & Future Work

1. **Purely theoretical, no experimental validation:** The absence of simulation experiments makes it difficult to assess finite-sample performance or the practical impact of constant factors.

2. **Strict preference assumption is strong:** The requirement that $\bm{U}(i,j) \neq \bm{U}(i,j')$ for all $j \neq j'$ is restrictive; the regret bound degrades trivially when $\Delta_{\min} = o(1/\sqrt{T})$. The authors acknowledge that indifference requires new methods.

3. **Decentralized algorithm requires knowledge of $T$:** Although a doubling trick can address this, it leads to non-monotone instantaneous regret, which may undermine agent trust.

4. **Centralized algorithm regret is loose by a factor of $N$:** The gap between $\mathcal{O}(N^2)$ and $\Omega(N)$ remains open.

5. **Strong modeling assumptions:** Requirements such as globally known arm IDs, a shared clock, and all players knowing the set of available arms may be difficult to satisfy in practical decentralized systems.

6. **Strategic manipulation is not discussed:** Whether TTC's strategy-proofness remains intact when preferences must be learned has not been analyzed.

## Related Work & Insights

- **Multi-player bandit learning:** Works such as Tibrewal et al. (2019), Mehrabian et al. (2020), and Shi et al. (2021) focus on system regret with maximum-weight matching as the benchmark. The key distinction here is the use of the "core" as benchmark and the focus on individual regret.
- **Bandits in two-sided matching markets:** Liu et al. (2020, 2021) and Kong and Li (2023) study learning stable matchings in two-sided markets, with different collision rules and benchmark definitions.
- **Insight:** Using game-theoretic solution concepts (core, stable matching, etc.) as benchmarks for bandit learning is a promising direction that can be generalized to other cooperative game settings, such as coalition formation and cost sharing.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale:** The problem formulation is novel, and integrating the core concept from housing markets with the bandit framework constitutes a genuine theoretical contribution. The matching upper and lower bounds for the decentralized algorithm constitute a strong result. However, the absence of any experimental validation, the restrictive strict preference assumption, and the $N$-factor gap in the centralized algorithm make this a predominantly theoretical work whose practical impact remains to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)
- [\[AAAI 2026\] Why Isn't Relational Learning Taking Over the World?](why_isnt_relational_learning_taking_over_the_world.md)
- [\[AAAI 2026\] Learning Network Dismantling Without Handcrafted Inputs](learning_network_dismantling_without_handcrafted_inputs.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] ParaMETA: Towards Learning Disentangled Paralinguistic Speaking Styles Representations](parameta_towards_learning_disentangled_paralinguistic_speaking_styles_representa.md)

</div>

<!-- RELATED:END -->
