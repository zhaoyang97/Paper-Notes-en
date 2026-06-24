---
title: >-
  [Paper Note] Perturbing Best Responses in Zero-Sum Games
description: >-
  [AAAI 2026][Reinforcement Learning][Zero-sum games] This paper investigates the introduction of stochastic perturbations into best-response oracles (BROs) for zero-sum games. It proves that Stochastic Fictitious Play (SFP) achieves an expected iteration count of $O(\frac{\log n}{\varepsilon^2})$ with respect to the number of pure strategies $n$, and proposes the Stochastic Double Oracle (SDO) algorithm, which achieves logarithmic convergence under specific game structures.
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Zero-sum games"
  - "Nash equilibrium"
  - "perturbed best response"
  - "fictitious play"
  - "double oracle"
date: 2026-05-08
content_hash: efac5b1471bc7176
---

# Perturbing Best Responses in Zero-Sum Games

**Conference**: AAAI 2026
**arXiv**: [2511.12523](https://arxiv.org/abs/2511.12523)  
**Code**: [github](https://github.com/geoborek/perturbing-best-responses)  
**Area**: Reinforcement Learning
**Keywords**: Zero-sum games, Nash equilibrium, perturbed best response, fictitious play, double oracle

## TL;DR

This paper investigates the introduction of stochastic perturbations into best-response oracles (BROs) for zero-sum games. It proves that Stochastic Fictitious Play (SFP) achieves an expected iteration count of $O(\frac{\log n}{\varepsilon^2})$ with respect to the number of pure strategies $n$, and proposes the Stochastic Double Oracle (SDO) algorithm, which achieves logarithmic convergence under specific game structures.

## Background & Motivation

Computing Nash Equilibria (NE) in large-scale two-player zero-sum games is a central computational challenge. Fictitious Play (FP) and Double Oracle (DO) are two classical algorithms based on best-response oracles, but both require $\Omega(n)$ iterations in the worst case to find an $\varepsilon$-NE. While it is theoretically known that an $\varepsilon$-NE of logarithmic support size always exists (Althöfer 1994), Hazan & Koren 2016 proved that any randomized BRO algorithm requires at least $\Omega(\sqrt{n}/\log^3 n)$ iterations.

**Core Motivation**: To overcome the lower bound of BROs, the authors propose "augmenting" the BRO by adding random perturbations to utility values prior to computing best responses. This Perturbed Best-Response Oracle (PBRO) can be understood as allowing the algorithm to explore suboptimal but potentially valuable strategies with some probability, thereby avoiding the linear iteration trap inherent to deterministic algorithms.

**Key Observation**: Deterministic DO and FP are slow because they can be guided by adversarially designed best responses, progressively enumerating all pure strategies. Perturbation breaks this deterministic pattern, enabling the algorithm to "jump" directly to critical strategies.

## Method

### Overall Architecture

The paper proposes two randomized algorithmic variants:
1. **Stochastic Fictitious Play (SFP)**: At each iteration of FP, the deterministic best response is replaced by a perturbed best response.
2. **Stochastic Double Oracle (SDO)**: At each iteration of DO, the perturbed best response is used to expand the strategy subset.

The termination condition for both algorithms is unaffected by perturbation — unperturbed best-response values are still used to determine whether an $\varepsilon$-NE has been reached.

### Key Designs

#### 1. Perturbed Best Response (PBRO)

The core idea is to add i.i.d. random noise to each component of the utility vector before computing the best response:

$$\widetilde{\text{BR}}_r(\mathbf{q}) \in \arg\min_{i \in [m]} \pi_i(\mathbf{M}\mathbf{q} - \mathbf{u})$$

where $\mathbf{u}$ is a random perturbation vector. A key mathematical connection is established: when perturbations follow a Gumbel distribution $G(0, \beta)$, by the Gumbel-max trick (Lemma 1), the perturbed best response is equivalent to sampling from $\text{softmax}(\mathbf{M}\mathbf{q}/\beta)$. This connection establishes an equivalence between SFP and the Randomized Exponential Weights Forecaster (REWF).

**Design Motivation**: Gumbel perturbations naturally introduce a mechanism of "selecting strategies with probability proportional to utility quality" rather than always selecting the absolute best response. The parameter $\beta$ controls the exploration–exploitation trade-off.

#### 2. Logarithmic Convergence Proof for SFP

**Theorem 3**: For a matrix game $\mathbf{M} \in [0,1]^{m \times n}$ with $m \leq n$, SFP with Gumbel perturbations computes an $\varepsilon$-NE within $O(\frac{\log n}{\varepsilon^2})$ expected iterations.

Proof sketch:
- SFP with Gumbel perturbations is mapped to a setting where both players use REWF.
- The regret bound of REWF (Corollary 2) is applied with parameter choices $\eta = \sqrt{8\ln n / T}$ and $\beta = \frac{2 + \sqrt{2\ln n}}{\varepsilon\sqrt{8\ln n}}$.
- It is shown that within $T = (\frac{2+\sqrt{2\ln n}}{\varepsilon})^2$ iterations, the sum of both players' regrets does not exceed $\varepsilon$ with probability at least $1/2$.
- Repeated execution yields a logarithmic expected iteration count.

#### 3. Logarithmic Convergence of SDO on Specific Games

For two families of hard game instances proposed by Zhang & Sandholm (2024):

**Example 1 (Matrix $\mathbf{L}$)**: Both players choose a number; the player with the larger number wins. Deterministic DO requires $n$ iterations, but SDO with uniform perturbations halves the expected minimum index at each iteration (Lemma 4), yielding $O(\log n)$ convergence via drift analysis (Theorem 6).

**Example 2 (Matrix $\mathbf{U}$)**: A variant of $\mathbf{L}$ with a unique best response. Perturbation ensures the expected index does not exceed $\frac{3}{4}k$ (Lemma 8), again achieving $O(\log n)$ convergence (Theorem 7).

#### 4. Efficient Perturbation Mechanism

For games with internal structure (e.g., POSGs), directly perturbing the full utility matrix is infeasible. The paper proposes *clustered perturbation*: terminal states are grouped into clusters, and the same random perturbation value is applied to each cluster:

$$\widetilde{\text{BR}}_r(\mathbf{q}) = \arg\min_{i \in [m]} \pi_i\left(\left(\mathbf{L} + \sum_{k=1}^{K} z_k \mathbf{B}_k\right)\mathbf{q}\right)$$

where $\mathbf{B}_k$ is the mask matrix for the $k$-th cluster. This reduces perturbation complexity from $O(n^2)$ to $O(K)$, where $K$ is the number of terminal states.

### Loss & Training

This paper presents theoretical and algorithmic contributions and does not involve deep learning training. Key algorithmic design choices include:
- SFP uses Gumbel perturbation parameter $\beta = \frac{2+\sqrt{2\ln n}}{\varepsilon\sqrt{8\ln n}}$
- SDO uses uniform perturbation $\mathcal{U}(-1/2, 1/2)$ or $\mathcal{U}(-1, 1)$
- Termination condition: $\text{BRVal}_c(\mathbf{p}) - \text{BRVal}_r(\mathbf{q}) \leq \varepsilon$

## Key Experimental Results

### Main Results

SFP experiments ($\varepsilon=0.1$):

| Game Type | FP | AFP | SFP | SAFP |
|---------|-----|-----|-----|------|
| Random matrix $(n=200)$ | ~200 | ~80 | ~200 | ~200 |
| Matrix $\mathbf{U}^\top$ $(n=200)$ | 200 (linear) | 200 (linear) | ~25 (log) | ~25 (log) |
| Morra game $(f=15,\ n=225)$ | >200 | >200 | ~35 | ~35 |

SDO experiments ($\varepsilon=0.1$):

| Game Type | DO | SDO |
|---------|-----|-----|
| Matrix $\mathbf{L}$ $(n=200)$ | 200 | ~15 |
| Matrix $\mathbf{U}^\top$ $(n=200)$ | 200 | ~20 |
| Morra game $(f=15)$ | ~150 | ~25 |
| Colonel Blotto (5 battlefields) | >100 | ~20 |

### Ablation Study

| Perturbation Type | Matrix $\mathbf{L}$ (SDO) | Matrix $\mathbf{U}^\top$ (SDO) | Notes |
|---------|-------------|-------------|------|
| No perturbation (DO) | $n$ iters | $n$ iters | Linear worst case |
| Uniform $\mathcal{U}(-1,1)$ | ~$\log n$ | ~$\log n$ | Logarithmic convergence |
| Efficient clustered perturbation | Slightly above direct | Slightly above direct | Still far better than DO |
| Random matrix game | No improvement or degradation | — | Perturbation ineffective on random games |

### Key Findings

1. **Logarithmic convergence of SFP**: The first proof that SFP achieves logarithmic iteration complexity in the number of pure strategies, significantly outperforming FP's $O(n)$.
2. **Conditional effectiveness of SDO**: SDO achieves logarithmic convergence on specific structured games, but its complexity in the general case remains an open problem.
3. **Negative result for random games**: Perturbation not only fails to accelerate convergence on random matrix games but may cause degradation — because the best responses of random games are themselves inherently random.
4. **Validation on path-planning games**: On path-planning games with internal structure, efficient perturbation likewise reduces iteration counts.

## Highlights & Insights

- **Elegant connection between Gumbel-softmax and game theory**: The Gumbel-max trick establishes an equivalence between PBRO and REWF, enabling the use of online learning regret bounds to prove convergence of game-theoretic algorithms.
- **Elegant application of drift analysis**: Drift theory from evolutionary computation is applied to analyze SDO's convergence, showcasing the power of cross-domain technique transfer.
- **Practical considerations**: The efficient perturbation scheme requires perturbing only terminal-state rewards and can be directly integrated into existing POSG solvers.
- **Theoretical rigor**: All major results are accompanied by complete proofs, without reliance on asymptotic assumptions.

## Limitations & Future Work

1. **General complexity of SDO unresolved**: Whether SDO always achieves sublinear or logarithmic iterations on arbitrary matrix games remains an open problem.
2. **Failure on random games**: The ineffectiveness of perturbation on random matrix games indicates that the method depends on game structure.
3. **Perturbation parameter selection**: The optimal choice of $\beta$ depends on the normalization range of the game, which may require tuning in practice.
4. **Limitations of efficient perturbation**: Clustered perturbation is not fully equivalent to direct utility perturbation, leaving a gap in theoretical guarantees.
5. **Limited experimental scale**: Experiments are conducted up to $n \approx 1000$; larger-scale games have not been validated.

## Related Work & Insights

- Closely related to the PSRO (Policy Space Response Oracles) line of work; the perturbation idea may be applied to multi-agent reinforcement learning.
- Regularized AFP variants (PU and OMWU) can also achieve $O(\log n)$, but require maintaining probability distributions over all strategies, making them unsuitable for large games.
- Insight: The perturbation idea can be extended to broader settings such as extensive-form game solving and multi-player games.

## Rating

- Novelty: ⭐⭐⭐⭐ — Perturbing BROs is not new per se, but the logarithmic complexity proof and efficient perturbation scheme represent significant contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Theoretical validation is sufficient, but experiments on truly large-scale application scenarios are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematically rigorous, logically clear, with complete proofs.
- Value: ⭐⭐⭐⭐ — Addresses an important theoretical problem in game theory with potential implications for multi-agent RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](../../ICML2026/reinforcement_learning/global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](../../ICML2025/reinforcement_learning/solving_zero-sum_convex_markov_games.md)
- [\[ICML 2026\] Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](../../ICML2026/reinforcement_learning/bilevel_optimization_over_saddle_points_of_zero-sum_markov_games.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[AAAI 2026\] Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits](beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)

</div>

<!-- RELATED:END -->
