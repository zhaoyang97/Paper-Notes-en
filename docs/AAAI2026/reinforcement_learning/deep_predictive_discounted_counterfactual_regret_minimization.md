---
title: >-
  [Paper Note] Deep (Predictive) Discounted Counterfactual Regret Minimization
description: >-
  [AAAI 2026][Reinforcement Learning][Counterfactual Regret Minimization] This paper proposes two model-free neural CFR algorithms, VR-DeepDCFR+ and VR-DeepPDCFR+…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Counterfactual Regret Minimization"
  - "Imperfect Information Games"
  - "Nash Equilibrium"
  - "Neural Network Approximation"
  - "Variance Reduction"
date: 2026-05-08
content_hash: 9410de4b6e4e32e1
---

# Deep (Predictive) Discounted Counterfactual Regret Minimization

**Conference**: AAAI 2026
**arXiv**: [2511.08174](https://arxiv.org/abs/2511.08174)  
**Code**: [rpSebastian/DeepPDCFR](https://github.com/rpSebastian/DeepPDCFR)  
**Area**: Reinforcement Learning
**Keywords**: Counterfactual Regret Minimization, Imperfect Information Games, Nash Equilibrium, Neural Network Approximation, Variance Reduction

## TL;DR

This paper proposes two model-free neural CFR algorithms, VR-DeepDCFR+ and VR-DeepPDCFR+, which integrate advanced tabular CFR variants (DCFR+/PDCFR+) into neural network approximation frameworks for the first time. Through bootstrapped cumulative advantage estimation, discounted clipping mechanisms, and baseline variance reduction, the proposed methods achieve faster convergence in standard imperfect information games.

## Background & Motivation

Imperfect information games (IIGs) provide a foundational framework for modeling multi-player strategic interactions with hidden information, with the core objective of computing (approximate) Nash equilibria. The CFR (Counterfactual Regret Minimization) family of algorithms represents one of the most successful approaches for solving IIGs, converging to NE by iteratively minimizing cumulative counterfactual regret.

Several tabular CFR variants have been proposed to accelerate convergence:
- **CFR+**: Clips negative cumulative regrets + linearly weighted average strategy
- **DCFR**: Applies discounting to cumulative regrets
- **DCFR+**: Combines the advantages of CFR+ and DCFR
- **PDCFR+**: Exploits the predictability of regret to accelerate convergence

However, existing neural CFR methods (e.g., DeepCFR, DREAM) primarily approximate vanilla CFR or LinearCFR behavior and **cannot effectively incorporate more advanced CFR variants**. The core difficulty is that DCFR+ and PDCFR+ updates rely on bootstrapping from the previous iteration's cumulative counterfactual regrets, whereas conventional neural CFR methods fit from scratch using all iterations' samples in a replay buffer — the two approaches are architecturally incompatible. At a deeper level, counterfactual values are expected utilities weighted by opponent reach probabilities; these unnormalized values vary enormously in magnitude across information sets, making them difficult for networks to learn effectively.

## Method

### Overall Architecture

The key mechanism of the proposed algorithms is to **replace cumulative counterfactual regrets with cumulative advantages**. An advantage is defined as the counterfactual regret divided by the opponent reach probability: $r_i^t(I,a) = \pi_{-i}^{\sigma^t}(I) \cdot A_i^{\sigma^t}(I,a)$. Advantages exhibit a more uniform numerical scale, enabling better neural network learning and generalization.

The overall pipeline (per iteration):
1. Collect $K$ episodes via outcome sampling
2. Compute variance-reduced sampled advantages using the value network
3. Update the cumulative advantage network via bootstrapping
4. Apply discounting and clipping to the cumulative advantages to emulate DCFR+/PDCFR+ behavior
5. Derive the new strategy from cumulative advantages via regret matching

### Key Designs

**1. Bootstrapped Cumulative Advantage Estimation**

Traditional DeepCFR maintains a replay buffer retaining samples from all iterations, refitting cumulative regrets from scratch. This is incompatible with the bootstrap update of DCFR+. Instead, the buffer is cleared each iteration; only current-iteration samples are used, with bootstrapping performed using the output of the previous iteration's network.

The adjusted sampled counterfactual value is:
$$\check{v}_i^{\sigma^t}(I,a|z) = \frac{\pi^{\sigma^t}(z[I]a,z) \cdot u_i(z)}{\pi^{\xi^t}(z[I],z)}$$

Key Theorem 2 establishes that the expectation equals the advantage: $\mathbb{E}[\check{r}_i^t(I,a)|z \in Z_I] = A_i^{\sigma^t}(I,a)$

The training loss for the cumulative advantage network $R(I,a|\theta_i^t)$ is based on bootstrapping:

$$\mathcal{L}(\theta_i^t) = \mathbb{E}_{(I,\check{r}) \sim \mathcal{B}_{V,i}}\left[\sum_a \left(R(I,a|\theta_i^{t-1}) + \check{r}(I,a) - R(I,a|\theta_i^t)\right)^2\right]$$

**2. Approximate DCFR+**

Discounting and clipping operations are incorporated into the bootstrap loss:

$$\mathcal{L}(\theta_i^t) = \mathbb{E}\left[\sum_a \left(\max(R(I,a|\theta_i^{t-1}), 0) \cdot \frac{(t-1)^\alpha}{(t-1)^\alpha + 1} + \check{r}(I,a) - R(I,a|\theta_i^t)\right)^2\right]$$

- $\max(\cdot, 0)$ clips negative cumulative advantages (inspired by CFR+, reducing the cost of incorrect actions)
- $(t-1)^\alpha / ((t-1)^\alpha + 1)$ is a discount factor (inspired by DCFR, down-weighting inaccurate early estimates)

**3. Approximate PDCFR+**

Building on DCFR+, a prediction mechanism is added: an instantaneous advantage network $r(I,a|\phi_i^t)$ is additionally trained to estimate the current-iteration advantages, which is then used to predict the cumulative advantages for the next iteration:

$$\max\left(R(I,a|\theta_i^t), 0\right) \cdot \frac{t^\alpha}{t^\alpha + 1} + r(I,a|\phi_i^t)$$

The predicted cumulative advantages are used to compute the new strategy via regret matching, exploiting the slow variation (predictability) of counterfactual regrets across iterations.

**4. Baseline Variance Reduction**

Single-episode sampling introduces high variance. A historical value network $Q(h,a|w^t)$ is introduced as a baseline function (inspired by DREAM):

$$\bar{v}_i^{\sigma^t}(I,a|z) = \begin{cases} Q_i(h,a|w^{t-1}) + \frac{\bar{v}_i(I'|z) - Q_i(h,a|w^{t-1})}{\xi^t(I,a)} & \text{if } a = \hat{a} \\ Q_i(h,a|w^{t-1}) & \text{otherwise} \end{cases}$$

For the sampled action, an importance-sampling correction accounts for the discrepancy between the value network prediction and the actual sample; for unsampled actions, the value network estimate is used directly.

### Loss & Training

Three networks are trained jointly:
- **Cumulative advantage network** $R$: bootstrap + discounted clipping loss
- **Instantaneous advantage network** $r$ (PDCFR+ only): standard regression loss
- **Historical value network** $Q$: DQN-style TD loss, trained off-policy
- **Average strategy network** $\Pi$: weighted regression loss with weights $(t/T)^\gamma$ to assign higher importance to later-iteration strategies

## Key Experimental Results

### Main Results (Convergence to Equilibrium)

Seven model-free neural algorithms are compared by exploitability convergence speed across 8 standard IIGs. Key comparisons:

| Method | Kuhn Poker | Leduc Poker | Liar's Dice | Other 5 Games |
|--------|-----------|-------------|-------------|---------------|
| QPG/RPG | Converges to 0.01 only on Kuhn | Poor | Poor | Poor |
| NFSP | Slow | Slow | Slow | Moderate |
| OS-DeepCFR | Moderate | Moderate | Moderate | Moderate |
| DREAM | Fast | Fast | Fast | Fast |
| **VR-DeepDCFR+** | **Fastest** | **Fastest** | **Fastest** | **Fastest on most** |
| **VR-DeepPDCFR+** | **Fastest** | **Fastest** | **Fastest** | **Fastest on most** |

VR-DeepDCFR+ and VR-DeepPDCFR+ achieve the fastest convergence in most games.

### Large-Scale Poker Head-to-Head Evaluation

Evaluated on Flop Hold'em Poker (FHP) against 5 rule-based agents of varying styles, with 20,000 hands per matchup:

| Method | Average Reward (chips/hand) |
|--------|-----------------------------|
| OS-DeepCFR | -7.8 ± 1.4 |
| DREAM | -2.0 ± 3.1 |
| **VR-DeepDCFR+** | **11.6 ± 1.2** |
| **VR-DeepPDCFR+** | **11.3 ± 0.9** |

In professional poker, a margin of 5 chips per hand is considered a significant skill gap. The proposed methods win 11+ chips per hand on average, substantially outperforming other neural CFR approaches.

### Ablation Study

Starting from VR-DeepPDCFR+, three components are ablated across 4 IIGs:
- Removing bootstrapped cumulative advantages → degrades to DeepCFR-like behavior, slower convergence
- Removing advanced CFR variant (discounting + clipping) → degrades to approximate vanilla CFR
- Removing baseline variance reduction → increased variance, unstable training

All three components contribute positively to performance.

### Key Findings

1. Cumulative advantages exhibit lower variance than cumulative counterfactual regrets, leading to more stable network training
2. Bootstrapping eliminates the storage overhead of large replay buffers and the computational cost of full retraining
3. VR-DeepDCFR+ has comparable runtime to DREAM (the main difference lies in the loss formulation) while converging faster
4. Both methods use identical hyperparameters across all games, demonstrating strong generalization

## Highlights & Insights

1. **Core insight of replacing regrets with advantages**: Counterfactual regret equals opponent reach probability times advantage; dividing by opponent reach probability normalizes the value scale, resolving a fundamental bottleneck in neural CFR
2. **Elegant combination of bootstrapping and discounted clipping**: Each iteration requires only current-iteration samples combined with the previous network's outputs, faithfully emulating advanced CFR variants
3. **Theoretical rigor**: Theorems 1 and 2 provide unbiasedness proofs for the sampling estimators, grounding the algorithmic design in solid theory
4. **Engineering simplicity**: Runtime is comparable to DREAM with minimal additional overhead, making the approach highly practical

## Limitations & Future Work

1. The advantage prediction in PDCFR+ assumes slow variation between iterations (predicting the next iteration using the current one); RNNs could be explored to capture temporal dependencies
2. Validation is limited to two-player zero-sum games; extension to multi-player or non-zero-sum settings remains unexplored
3. The quality of value network training directly impacts variance reduction; value network accuracy may become a bottleneck in large games
4. Although hyperparameters $\alpha$ and $\gamma$ are unified across games, theoretical guidance for selecting their optimal values is lacking

## Related Work & Insights

- **DeepCFR / OS-DeepCFR**: Pioneering work on neural network approximation of CFR, but limited to vanilla/Linear CFR
- **DREAM**: Introduces value function baselines for variance reduction; this work inherits that approach and extends it to advanced variants
- **ESCHER**: Directly computes regrets using value functions but incurs high training cost; this work adopts DREAM's lightweight alternative
- **DCFR+ / PDCFR+**: State-of-the-art tabular CFR advances; this work achieves their neural network approximation for the first time
- **Conceptual inspiration**: Decoupling core operations of advanced algorithm variants (discounting, clipping, prediction) into modular components implementable within neural network loss functions

## Rating

- Novelty: ⭐⭐⭐⭐ (First effective integration of DCFR+/PDCFR+ into neural CFR)
- Technical Depth: ⭐⭐⭐⭐⭐ (Solid theoretical proofs, algorithmic design, and variance analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐ (8 games + large-scale poker evaluation + ablations, comprehensive coverage)
- Writing Quality: ⭐⭐⭐⭐ (Sufficient background coverage, clear derivations)
- Value: ⭐⭐⭐⭐ (Open-source code; strong head-to-head performance on FHP is practically meaningful)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits](beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)
- [\[AAAI 2026\] Discounted Cuts: A Stackelberg Approach to Network Disruption](discounted_cuts_a_stackelberg_approach_to_network_disruption.md)
- [\[NeurIPS 2025\] Simultaneous Swap Regret Minimization via KL-Calibration](../../NeurIPS2025/reinforcement_learning/simultaneous_swap_regret_minimization_via_kl-calibration.md)
- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[NeurIPS 2025\] Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](../../NeurIPS2025/reinforcement_learning/comparing_uniform_price_and_discriminatory_multi-unit_auctions_through_regret_mi.md)

</div>

<!-- RELATED:END -->
