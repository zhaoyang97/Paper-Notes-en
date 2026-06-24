---
title: >-
  [Paper Note] Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games
description: >-
  [NeurIPS 2025][Reinforcement Learning][quantum algorithms] This work presents the first quantum algorithms for computing correlated equilibria (CE) and coarse correlated equilibria (CCE) in multi-player general-sum games. By quantizing the multi-scale MWU framework and introducing a unified QRAM scheme, the paper achieves a near-optimal query complexity of $\tilde{O}(m\sqrt{n})$ in both the number of players $m$ and actions $n$, along with matching quantum lower bounds.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "quantum algorithms"
  - "game theory"
  - "correlated equilibrium"
  - "multiplicative weights update"
  - "query complexity"
date: 2026-05-08
content_hash: 50b4fa443f71b391
---

# Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games

**Conference**: NeurIPS 2025
**arXiv**: [2510.16782](https://arxiv.org/abs/2510.16782)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: quantum algorithms, game theory, correlated equilibrium, multiplicative weights update, query complexity

## TL;DR
This work presents the first quantum algorithms for computing correlated equilibria (CE) and coarse correlated equilibria (CCE) in multi-player general-sum games. By quantizing the multi-scale MWU framework and introducing a unified QRAM scheme, the paper achieves a near-optimal query complexity of $\tilde{O}(m\sqrt{n})$ in both the number of players $m$ and actions $n$, along with matching quantum lower bounds.

## Background & Motivation

**Background**: Quantum algorithms for two-player zero-sum games have been extensively studied, achieving a $\sqrt{n}$ quantum speedup ($\tilde{O}(\sqrt{n}/\varepsilon^{2.5})$). Classical equilibrium computation for multi-player general-sum games is also well-established. However, quantum equilibrium computation for multi-player games remains entirely unexplored.

**Limitations of Prior Work**:
   - Nash equilibrium is PPAD-hard, motivating the study of the more tractable CE/CCE concepts.
   - The classical optimal CE query complexity is $\tilde{O}(mn(\log mn)^{O(1/\varepsilon)})$ (Peng–Rubinstein '23); classical CCE requires $\tilde{O}(mn/\varepsilon^2)$.
   - In multi-player settings, the joint action space of size $n^m$ causes naïve quantization to incur exponential QRAM overhead.

**Key Challenge**: Quantizing multi-scale MWU requires amplitude encoding of loss vectors. Standard methods demand a QRAM of size $\Omega(n^m)$ to store frequency tables — an exponential blowup.

**Key Insight**: Design a unified QRAM that stores historical action samples (rather than frequency tables), and construct the amplitude encodings required by all MWU subroutines from a single QRAM.

**Core Idea**: For CE, a quantum Gibbs sampler accelerates the exponential update steps in multi-scale MWU. For CCE, the ghost iteration technique extends the zero-sum quantum framework to the multi-player setting. A unified QRAM eliminates exponential overhead throughout.

## Method

### Overall Architecture
Two independent algorithms are proposed. (1) **CE algorithm**: quantizes the Peng–Rubinstein multi-scale MWU framework by replacing the classical $O(n)$ queries per round with a quantum Gibbs sampler, achieving a $\sqrt{n}$ speedup. (2) **CCE algorithm**: extends the Bouland et al. quantum framework for zero-sum games to the multi-player setting using ghost iterations, achieving $\tilde{O}(1/\varepsilon^2)$-round convergence. Both algorithms share a unified QRAM design.

### Key Designs

1. **Quantum Multi-Scale MWU for CE**:

    - **Function**: Computes an $\varepsilon$-correlated equilibrium.
    - **Mechanism**: Classical multi-scale MWU requires $\Omega(n)$ queries per round to compute loss vectors and perform exponential updates. The quantization proceeds in two steps: (i) constructing amplitude encodings of loss vectors from QRAM, and (ii) sampling from the exponential distribution via a quantum Gibbs sampler — yielding a total query cost of $\tilde{O}(m\sqrt{n}(\log mn)^{O(1/\varepsilon)})$.
    - **Design Motivation**: Running $O(1/\varepsilon)$ MWU instances in parallel would naïvely require $O(1/\varepsilon)$ independent QRAMs under standard approaches.

2. **Quantized Grigoriadis–Khachiyan Framework for CCE**:

    - **Function**: Computes an $\varepsilon$-coarse correlated equilibrium.
    - **Mechanism**: The GK algorithm is specifically chosen (over faster MWU variants) because its regret bound is **independent** of the number of players $m$ — a critical property for achieving query complexity linear in $m$. Ghost iterations are used to prove convergence in the multi-player setting.
    - **Design Motivation**: Optimistic/cautious MWU variants offer better $\varepsilon$-dependence but incur regret that grows polynomially in $m$, causing total query complexity to scale super-linearly in $m$.

3. **Unified QRAM Scheme**:

    - **Function**: Uses a single QRAM storing all historical action samples to avoid exponential storage overhead.
    - **Mechanism**: Instead of storing frequency vectors (requiring $n^m$ space), the raw sample sequences are stored. Each MWU subroutine constructs the required amplitude encoding by superposing over different subsets of these samples. A gate-level analysis shows that the QRAM requires only $m \log n \cdot (\log mn)^{O(1/\varepsilon)}$ gates.
    - **Design Motivation**: This is the key technical innovation enabling the reduction from exponential to polynomial overhead.

4. **Quantum Lower Bounds**:

    - **Function**: Establishes quantum query lower bounds of $\Omega(m\sqrt{n})$ for computing CE/CCE.
    - **Mechanism**: Reduces $m$ independent unstructured search instances to CE/CCE computation in an $m$-player game. Combining the $\Omega(\sqrt{n})$ lower bound for unstructured search with a direct product theorem yields $\Omega(m\sqrt{n})$.
    - **Design Motivation**: The lower bounds match the upper bounds in both $m$ and $n$, establishing near-optimality of the algorithms.

### Loss & Training
This is a purely theoretical work analyzed within the query complexity framework; no machine learning training is involved.

## Key Experimental Results

### Theoretical Complexity Comparison

| Problem | Classical Query | Quantum Query (Upper Bound) | Quantum Lower Bound | Optimal in $m$, $n$? |
|---------|----------------|----------------------------|---------------------|----------------------|
| $\varepsilon$-CE | $mn(\log mn)^{O(1/\varepsilon)}$ | $m\sqrt{n}(\log mn)^{O(1/\varepsilon)}$ | $\Omega(m\sqrt{n})$ | ✓ |
| $\varepsilon$-CCE | $mn/\varepsilon^2$ | $m\sqrt{n}/\varepsilon^{2.5}$ | $\Omega(m\sqrt{n})$ | ✓ |

### Time Complexity

| Problem | Quantum Query | Quantum Time | Source of Gap |
|---------|--------------|--------------|---------------|
| $\varepsilon$-CE | $m\sqrt{n}(\log)^{O(1/\varepsilon)}$ | $m^2\sqrt{n}(\log)^{O(1/\varepsilon)}$ | QRAM gate overhead ($\times m$) |
| $\varepsilon$-CCE | $m\sqrt{n}/\varepsilon^{2.5}$ | $m^2\sqrt{n}/\varepsilon^{4.5}$ | QRAM + Gibbs sampling overhead |

### Key Findings
- The query upper and lower bounds for both CE and CCE match exactly in $m$ and $n$ (up to polylogarithmic factors), establishing near-optimality of the algorithms.
- The quantum speedup is primarily a $\sqrt{n}$ (Grover-style) acceleration in $n$; speedups in $m$ and $\varepsilon$ are limited.
- The unified QRAM is the central implementation technique — it not only reduces storage but serves as the mechanism enabling multiple MWU instances to share quantum resources.
- The $\varepsilon$-dependence for CCE ($\varepsilon^{-2.5}$) is worse than the classical bound ($\varepsilon^{-2}$) — a direct cost of quantizing GK; improvement may be possible by quantizing the RVU framework.
- Open problem: Can the time complexity be reduced to $\tilde{O}(m\sqrt{n})$, matching the query complexity? The main difficulty is that each round requires sampling strategies for all $m$ players, necessitating QRAM access within the Gibbs sampler.

## Highlights & Insights
- **Multi-player games + quantum computation** constitutes an entirely new interdisciplinary direction — extending quantum advantage from two-player zero-sum to general-sum multi-player settings is a non-trivial generalization.
- **The choice of GK over faster MWU variants for CCE** reflects a deep algorithmic insight: GK's regret bound is independent of $m$, which is crucial in the multi-player setting. Although the $\varepsilon$-dependence worsens, the scaling in $m$ and $n$ remains near-optimal — a canonical trade-off in the question of "which dimension to optimize."
- The **unified QRAM** elegantly stores samples rather than frequencies, compressing exponential storage to polynomial, while supporting amplitude encoding construction for arbitrary MWU instances. This technique may be of independent value for other quantum online learning algorithms.
- The **lower bound proof** is concise yet powerful — reduction to unstructured search via a direct product theorem is a standard but effective tool.

## Limitations & Future Work
- The $\varepsilon$-dependence for CCE ($\varepsilon^{-2.5}$) is worse than the classical bound by $\varepsilon^{-0.5}$ — quantizing optimistic MWU may improve this, though its higher-order smoothness properties are sensitive to Gibbs sampling noise.
- Time complexity exceeds query complexity by a factor of $m$, arising from the need to run Gibbs sampling for all $m$ players per round.
- Only normal-form games are considered — quantum algorithms for Bayesian games and extensive-form games remain open problems.
- The quantum algorithms assume access to an efficient QRAM, whose physical implementation remains controversial.

## Related Work & Insights
- **vs. quantum algorithms for zero-sum games** (Li et al., Bouland et al.): This work generalizes those results to multi-player general-sum settings; the core challenge is the exponential growth of QRAM overhead from $O(n)$ to $O(n^m)$.
- **vs. Peng–Rubinstein (classical CE)**: The quantum algorithm replaces exponential updates in their framework with Gibbs sampling, achieving a $\sqrt{n}$ speedup.
- **vs. quantum games** (Eisert et al.): Entirely distinct — that line of work studies quantum equilibria under quantum strategies, whereas this paper uses quantum computation to accelerate the computation of classical equilibria in classical games.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First study of quantum equilibrium computation for multi-player general-sum games, opening a new research direction.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work with no numerical experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ High theoretical depth, clear problem formulation, and well-articulated technical contributions.
- Value: ⭐⭐⭐⭐ Lays the groundwork for the intersection of quantum optimization and game theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies](certifying_concavity_and_monotonicity_in_games_via_sum-of-squares_hierarchies.md)
- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](../../ICML2025/reinforcement_learning/solving_zero-sum_convex_markov_games.md)
- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](../../ICLR2026/reinforcement_learning/near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)

</div>

<!-- RELATED:END -->
