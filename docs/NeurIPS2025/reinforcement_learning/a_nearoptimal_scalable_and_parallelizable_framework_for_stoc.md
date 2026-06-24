---
title: >-
  [Paper Note] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond
description: >-
  [NeurIPS 2025][Reinforcement Learning][adversarial corruptions] This paper proposes BARBAT, an improvement over the classical BARBAR algorithm. By fixing epoch lengths and adjusting failure probabilities per epoch, BARBAT reduces the regret of stochastic multi-armed bandits under adversarial corruptions from $O(\sqrt{K}C)$ to the near-optimal $O(C)$ (eliminating the $\sqrt{K}$ factor), and successfully extends to multi-agent, graph bandit, combinatorial semi-bandit…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "adversarial corruptions"
  - "multi-armed bandits"
  - "elimination-based algorithm"
  - "regret bound"
  - "parallelizable"
date: 2026-05-08
content_hash: a2c28fe7d6a06752
---

# A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond

**Conference**: NeurIPS 2025
**arXiv**: [2502.07514](https://arxiv.org/abs/2502.07514)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: adversarial corruptions, multi-armed bandits, elimination-based algorithm, regret bound, parallelizable

## TL;DR
This paper proposes BARBAT, an improvement over the classical BARBAR algorithm. By fixing epoch lengths and adjusting failure probabilities per epoch, BARBAT reduces the regret of stochastic multi-armed bandits under adversarial corruptions from $O(\sqrt{K}C)$ to the near-optimal $O(C)$ (eliminating the $\sqrt{K}$ factor), and successfully extends to multi-agent, graph bandit, combinatorial semi-bandit, and batched bandit settings.

## Background & Motivation
Multi-armed bandits (MAB) are among the most fundamental problems in online learning. With growing concerns about safety, researchers have begun studying the "adversarial corruption" setting, where an adversary can tamper with reward observations. Existing approaches fall into two main categories:

1. **FTRL-based methods** (Follow-the-Regularized-Leader): Achieve optimal regret in both stochastic and adversarial environments, but require solving a constrained convex optimization problem at each round. This incurs substantial computational overhead in settings such as semi-bandits and is difficult to parallelize to multi-agent or batched settings.
2. **Elimination-based methods** (e.g., BARBAR): Computationally efficient and easily parallelizable, but BARBAR's regret contains an $O(\sqrt{K}C)$ term, with a redundant $\sqrt{K}$ factor that does not match the lower bound $\Omega(C)$.

The core challenge is: can elimination-based methods achieve near-optimal regret on par with FTRL? This is an open problem posed by Gupta et al.

## Core Problem
In the stochastic MAB problem under adversarial corruptions, how can one design an elimination-based algorithm whose regret is near-optimal—i.e., where the corruption-dependent term is $O(C)$ rather than $O(\sqrt{K}C)$—while retaining the computational efficiency and parallelizability of elimination-based methods?

## Method

### Overall Architecture
BARBAT (**B**ad **A**rms get **R**ecourse, **B**est **A**rm gets **T**rust) is an improved version of BARBAR. The core mechanism operates within each epoch as follows:
- **Input**: Current surviving arm set and estimated suboptimality gaps from the previous round
- **Process**: Sample arms according to preset probabilities and accumulate reward observations
- **Output**: Updated empirical reward estimates and gap estimates

The overall structure iterates over epochs; at the end of each epoch, the suboptimality gap of each arm is re-estimated, progressively focusing on the optimal arm.

### Key Designs

1. **Data-independent epoch length**: In BARBAR, epoch length depends on data from the previous round (i.e., estimated gaps), which allows an adversary to indirectly extend the next epoch's length by concentrating attacks within a single epoch, resulting in $O(\sqrt{K}C)$ additional regret. BARBAT instead uses a data-independent epoch length $N_m = \lceil K \lambda_m 2^{2(m-1)} \rceil$, preventing the adversary from manipulating epoch length through corruption. Surplus samples (i.e., $N_m - \sum_{k \neq k_m} n_k^m$) are entirely allocated to the currently estimated best arm $k_m$, embodying the "Best Arm gets Trust" principle.

2. **Epoch-varying failure probabilities**: BARBAR uses a globally uniform failure probability $\delta$, whereas BARBAT assigns a distinct $\delta_m = 1/(K \zeta_m)$ to each epoch $m$. This refined design eliminates the need for prior knowledge of the time horizon $T$ while keeping the total failure probability well controlled.

3. **Intuition for why BARBAR incurs $O(\sqrt{K}C)$**: Gupta et al. construct a counterexample: if the adversary concentrates the entire corruption budget $C = N_c$ within epoch $c$, BARBAR loses all previously accumulated information, and the next epoch length becomes $N_{c+1} = O(KC)$, with each arm sampled approximately $O(C)$ times uniformly, generating $O((K-1)C)$ regret. Since BARBAT fixes the epoch length to $N_m = K \lambda_m 2^{2(m-1)}$, an adversary must spend $C = N_c = K\lambda 2^{2(c-1)}$—a factor of $K$ more corruption budget than attacking BARBAR—to corrupt an entire epoch, thus limiting the corruption cost to $O(C)$.

### Extensions

BARBAT demonstrates strong scalability and parallelizability, and is extended to four additional settings:

- **MA-BARBAT (Multi-agent)**: $V$ agents collaborate by broadcasting their rewards at the end of each epoch, reducing per-agent regret by a factor of $V$: $R_v(T) = O(C/V + \sum_{\Delta_k>0} \log^2(VT) / (V\Delta_k))$. Communication cost is only $O(V \log(VT))$.
- **BB-BARBAT (Batched bandits)**: Adapts to the constraint of $L$ batches by setting epoch lengths to $O(T^{m/(L+1)})$; this is the first work to study batched bandits under adversarial corruptions and provides matching lower bounds.
- **SOG-BARBAT (Strongly observable graph bandits)**: Exploits the out-domination set of the graph structure to reduce the number of arms that need to be sampled, using the OODS algorithm to compute this set efficiently. The gap-dependent term in regret depends only on the $O(\alpha \ln(K/\alpha))$ arms with the smallest gaps.
- **DS-BARBAT ($d$-set semi-bandits)**: Handles combinatorial action spaces where the optimal action is a set of $d$ arms, achieving regret $O(dC + \sum_{k=d+1}^{K} \log^2(T)/\Delta_k)$.

### Loss & Training
This is a theoretical paper and does not involve neural network training. Key technical tools include:
- Chernoff–Hoeffding inequalities to control reward estimation error
- Freedman's martingale concentration inequality to handle the corruption component
- Induction to establish recursive upper and lower bounds on gap estimates
- Definition of "offset level" $D_m$ and "discounted offset rate" $\rho_m$ to unify the analysis of corruption and failure events

## Key Experimental Results

| Setting | Configuration | BARBAT Time | Strongest Baseline Time | BARBAT Advantage |
|---------|--------------|-------------|------------------------|-----------------|
| Multi-agent MAB (K=12) | V=10, T=50000 | 0.13 s | IND-FTRL: 1.53 s | ~12× speedup |
| Multi-agent MAB (K=16) | V=10, T=50000 | 0.13 s | IND-FTRL: 1.91 s | ~15× speedup |
| $d$-set semi-bandit (K=12) | d=3, T=50000 | 0.36 s | HYBRID: 14.63 s | ~40× speedup |
| $d$-set semi-bandit (K=16) | d=4, T=50000 | 0.37 s | LBINF\_GD: 12.69 s | ~34× speedup |

Across all settings, BARBAT's cumulative regret curves at high corruption levels (C=5000) are substantially lower than those of baseline methods, empirically validating the theoretical improvement in the corruption term.

### Ablation Study
- The paper primarily demonstrates robustness differences by comparing different corruption levels (C=2000 vs. C=5000).
- The $V$-fold regret reduction from collaboration in MA-BARBAT is verified experimentally.
- The computational efficiency advantage is most pronounced in the semi-bandit setting (~40× speedup), as FTRL requires solving expensive convex optimization problems there.

## Highlights & Insights
- **Distilled key insight**: A simple modification—fixing epoch lengths—eliminates the $\sqrt{K}$ factor and resolves the core flaw of BARBAR, namely that an adversary can indirectly manipulate epoch length through corruption.
- **Eliminates prior knowledge of $T$**: By setting $\delta_m$ per epoch, the method requires no foreknowledge of the time horizon, which is highly advantageous for practical deployment.
- **Exemplary scalable design**: A single core framework naturally adapts to four distinct bandit variants, with only local modifications required for each extension.
- **Overwhelming computational efficiency**: A ~40× speedup over FTRL in the semi-bandit setting, stemming from the elimination of per-round convex optimization.
- **No unique optimal arm assumption**: FTRL-based methods (beyond MAB) require this assumption, whereas the BARBAT family does not.

## Limitations & Future Work
- The regret contains a $\log^2(T)$ factor rather than the optimal $\log(T)$; the authors explicitly identify closing this gap as a future direction.
- A gap remains between the upper and lower bounds for batched bandits; the optimal regret is still an open problem.
- The multi-agent variant requires centralized communication (each agent broadcasts to all others); decentralized or communication-constrained scenarios are not considered.
- Although the out-domination set computation in graph bandits runs in polynomial time, its actual complexity is $O(|V|^2|E|)$.
- The paper does not extend to linear bandits and other important structured settings; the authors list this as future work.

## Related Work & Insights

| Method | Corruption Term | Computational Efficiency | Parallelizable | No Unique Optimal Arm Required |
|--------|----------------|------------------------|---------------|-------------------------------|
| BARBAR | $O(\sqrt{K}C)$ | ✓ | ✓ | ✓ |
| Tsallis-FTRL | $O(C)$ | ✗ | ✗ | ✗ (beyond MAB) |
| Shannon-FTRL | $O(C)$ | ✓ (closed-form) | ✓ | ✓ |
| **BARBAT** | $O(C)$ | ✓ | ✓ | ✓ |

Core distinction: BARBAT is the only elimination-based method that simultaneously achieves near-optimal regret, computational efficiency, parallelizability, and freedom from the unique optimal arm assumption. Shannon-FTRL admits closed-form solutions for MAB, but still requires solving an optimization problem in combinatorial settings such as semi-bandits.

The **epoch-fixing idea** is transferable to other epoch-based online learning algorithms that need to resist adversarial attacks. The **discounted offset rate** analysis technique offers an elegant approach to handling the cumulative effect of historical corruptions and is worth adopting more broadly.

## Rating
- Novelty: ⭐⭐⭐⭐ The core ideas (fixed epochs + epoch-varying $\delta$) are concise yet targeted, fully resolving an open problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three settings, multiple baselines, and runtime comparisons, though larger-scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly structured; the intuitive explanation of why BARBAR incurs $\sqrt{K} \cdot C$ is exceptionally well written, and the proofs are complete.
- Value: ⭐⭐⭐⭐ Resolves an open problem with a unified framework, though the research area is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games](near-optimal_quantum_algorithms_for_computing_coarse_correlated_equilibria_of_ge.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](../../ICLR2026/reinforcement_learning/near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)

</div>

<!-- RELATED:END -->
