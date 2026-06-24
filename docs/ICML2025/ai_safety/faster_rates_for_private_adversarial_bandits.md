---
title: >-
  [Paper Note] Faster Rates for Private Adversarial Bandits
description: >-
  [ICML 2025][AI Safety][Differential Privacy] Proposes a simple and efficient non-private to private reduction framework for the differentially private adversarial bandits problem. By utilizing batched losses and Laplace noise, it achieves an O(√(KT/ε)) regret bound, proves for the first time a separation between central DP and local DP for this problem, and provides the first private bandits with expert advice algorithm.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Differential Privacy"
  - "Adversarial Bandits"
  - "Bandits with Expert Advice"
  - "Online Learning"
  - "Laplace Mechanism"
date: 2026-05-08
content_hash: c1923b0330410be0
---

# Faster Rates for Private Adversarial Bandits

**Conference**: ICML 2025  
**arXiv**: [2505.21790](https://arxiv.org/abs/2505.21790)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Differential Privacy, Adversarial Bandits, Bandits with Expert Advice, Online Learning, Laplace Mechanism

## TL;DR

Proposes a simple and efficient non-private to private reduction framework for the differentially private adversarial bandits problem. By utilizing batched losses and Laplace noise, it achieves an O(√(KT/ε)) regret bound, proves for the first time a separation between central DP and local DP for this problem, and provides the first private bandits with expert advice algorithm.

## Background & Motivation

In the adversarial bandit problem, a learner faces a sequence of loss functions chosen by an adversary over T rounds. In each round, the learner selects an action and observes only the loss of the chosen action, with the goal of minimizing regret (the difference between its cumulative loss and that of the best fixed action in hindsight).

**Privacy Requirements**: Applications such as releasing the optimal treatment in medical trials without leaking patient information, or recommending ads in online advertising without exposing user preferences, require DP guarantees.

**Limitations of Prior Work**:
- Agarwal & Singh (2017) and Tossou & Dimitrakakis (2017): regret bound of O(√(KT·log(KT))/ε)
- These algorithms satisfy the stronger notion of **local DP**, where the regret becomes linear (vacuous) when ε < 1/√T.
- Previously, it was unknown whether central DP could achieve better utility, and how small ε could be to still obtain sublinear regret.

**Core Problem**:
1. Is there a separation in achievability between central DP and local DP in adversarial bandits?
2. How small can ε be while still achieving sublinear regret?

## Method

### Overall Architecture

**Generic Reduction Theorem (Theorem 1)**: Converts any non-private bandit algorithm B into an ε-DP algorithm A_τ.

The core idea is extremely simple:
1. Divide the T rounds into T/τ batches of size τ.
2. Within each batch, repeatedly play the same action prescribed by B.
3. At the end of each batch, compute the average of the losses within the batch.
4. Add Laplace(1/(ετ)) noise to the average loss.
5. Feed the noisy losses back to B.

**Regret Guarantee**: R_{A_τ}(T,K) ≤ τ · R̃_B(T/τ, K, 1/(ετ)) + τ

Where R̃_B is the regret of B on losses corrupted with Laplace noise. The key is that batching reduces the noise scale by a factor of τ.

### Key Designs

**1. Capability of Handling Negative Losses**

Since the losses can become negative and unbounded after adding Laplace noise, the base algorithm B must be capable of handling heavy-tailed/negative losses. This work leverages the existing heavy-tailed bandit algorithm HTINF (Huang et al., 2022), which employs a modified FTRL and can handle losses with a bounded second moment.

**2. FTRL Reduction (Corollary 2)**

Setting B = HTINF (α=2, σ=√6) and τ = ⌈1/ε⌉ yields:
- **ε-DP Regret Bound**: O(√(KT)/√ε + 1/ε)
- Valid for all ε ∈ [1/T, 1]
- Sublinear as long as ε > 1/T!

**3. Bandits with Expert Advice**

Three schemes cover different parameter regimes:

| Scheme | Regret Bound | Optimal Regime |
|------|--------|---------|
| Corollary 8 (FTRL→expert) | O(√(NT)/√ε) | N ≤ K |
| Theorem 9 (Private EXP4) | O(√(KT·log N)·log(KT)/ε) | N ≥ K, ε ≥ K/N |
| Theorem 10 (Batched EXP4) | O(N^{1/6}K^{1/2}T^{2/3}/ε^{1/3} + N^{1/2}/ε) | N ≥ K, ε ≤ 1/√T |

**4. Private Batched EXP4 (Algorithm 2)**

Specially designed for high-dimensional experts and high-privacy requirements:
- Batches the expert weight updates
- Adds Laplace noise (with magnitude scaled with √N) to the IPS estimates of expert losses
- Employs the same expert weight distribution across all rounds within a batch
- Updates the weights using the noisy losses after each batch

### Loss & Training

- **Standard Setting**: Loss functions ℓ_t: [K] → [0,1]. After the adversary's choice, the learner only observes the loss of the selected action.
- **Oblivious Adversary**: The loss sequence is determined before the game starts (for utility guarantees), though the privacy guarantees also hold against an adaptive adversary.
- **Noise Mechanism**: Laplace mechanism with sensitivity 1/τ (representing the maximum change in the batched average loss).

## Key Experimental Results

### Main Results

This is a purely theoretical work with no empirical experiments. The core results are improvements in theoretical upper bounds:

| Problem | Prev. SOTA | Ours | Gain |
|------|---------|---------|------|
| Adversarial Bandits | √(KT·log(KT))/ε | **√(KT)/√ε** | For all ε ≤ 1 |
| Experts (N ≤ K) | N/A | √(NT)/√ε | First result |
| Experts (N ≥ K, Low Privacy) | N/A | √(KT·log N)·log(KT)/ε | First result |
| Experts (N ≥ K, High Privacy) | N/A | N^{1/6}K^{1/2}T^{2/3}/ε^{1/3} | First result |

### Ablation Study

Theoretical analysis of behaviors in different ε regimes:
- When ε > 1/√T: Both ours and prior work are sublinear, but ours is superior.
- When 1/T < ε < 1/√T: **Only ours is sublinear**, while prior work is linear.
- When ε < 1/T: Both are linear.

### Key Findings

1. **Central DP vs. Local DP Separation**: Central DP can still achieve sublinear regret when ε = o(1/√T), whereas local DP cannot.
2. **Oblivious vs. Adaptive Adversary Separation**: Against an adaptive adversary, regret must be linear when ε = o(1/√T).
3. **Difficulty in Privatizing EXP3**: The probability P_t of the IPS estimator leaks information, causing the per-round privacy loss to grow with T.
4. **Regret Lower Bound for Batched EXP3**: Lemma 11 proves that any algorithm satisfying certain conditions must have an Ω(√(T/ε)) regret bound.

## Highlights & Insights

1. **The Power of Minimalist Reduction**: The entire approach is simply "batching + adding Laplace noise," yet the theoretical performance is profound. Batching naturally reduces sensitivity, and establishing the link to heavy-tailed bandit algorithms is a key insight.
2. **Handling Negative Losses equals Handling Privacy Noise**: This equivalence is a brand-new discovery—previously, there was no overlap between the heavy-tailed bandit literature and the DP literature.
3. **Complementary Nature of Three Expert Advice Algorithms**: Different parameter regimes (N vs. K, magnitude of ε) have different optimal algorithms, covering the complete parameter space.
4. **Elegance of Lower Bound Analysis**: The "two-instance" construction in Lemma 11—where a DP algorithm requires 1/ε exploration steps to "detect" changes in the loss structure—elegantly explains the origin of the √ε factor in the rate.
5. **First to Establish Separation between Central and Local DP in the Bandit Setting**

## Limitations & Future Work

1. **Limited to Oblivious Adversary**: Utility guarantees do not apply to adaptive adversaries, which require different DP definitions and analyses.
2. **No Additive Separation**: Regret bounds of the form O(√(KT) + poly(K)/ε) (i.e., additive division of T and ε) are not achieved.
3. **Vacuous Regret for Bandits with Experts when ε ≤ 1/√T and N ≥ T**: Indicates that new techniques are still required for this parameter regime.
4. **Lack of Lower Bounds**: The optimal regret bound for adversarial bandits under central DP remains an open problem.
5. **Purely Theoretical**: Lacks experimental validation. In real-world applications, batching may introduce latency and loss of adaptability.

## Related Work & Insights

- **Classic Works in Adversarial Bandits** [Auer et al., 2002]: EXP3 algorithm, minimax regret of Θ(√(TK)).
- **HTINF Heavy-Tailed FTRL** [Huang et al., 2022]: The key underlying algorithm for handling heavy-tailed losses.
- **Private Full-Information Online Learning** [Agarwal & Singh, 2017; Asi et al., 2023]: Additive separation has already been achieved under the full-information setting.
- **Batched Regret** [Arora et al., 2012]: Theorem 5 serves as the foundation for batched analysis.
- **EXP4** [Auer et al., 2002]: Baseline for bandits with expert advice.
- Significantly advances the theoretical research on the privacy-utility trade-off in online learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Minimalist approach with profound theoretical insights, establishing the first separation result.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical with no experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-structured with elegantly presented results.
- Value: ⭐⭐⭐⭐⭐ — Highly significant for the theory of private online learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](../../NeurIPS2025/ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)

</div>

<!-- RELATED:END -->
