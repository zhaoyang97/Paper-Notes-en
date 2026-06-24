---
title: >-
  [Paper Note] Optimal and Practical Batched Linear Bandit Algorithm
description: >-
  [ICML 2025][Reinforcement Learning][Batched Linear Bandit] By deeply integrating the **arm elimination strategy** with **regularized G-optimal design**, BLAE is the first to **simultaneously** achieve minimax optimal regret (up to a $\log T$ factor) under both large-$K$ and small-$K$ regimes in the batched linear bandit problem, while maintaining the minimal batch complexity of $\mathcal{O}(\log\log T)$ and outstanding practical performance.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Batched Linear Bandit"
  - "Arm Elimination"
  - "Regularized G-Optimal Design"
  - "Minimax Optimal"
  - "Limited Adaptivity"
date: 2026-05-08
content_hash: e404003be5462897
---

# Optimal and Practical Batched Linear Bandit Algorithm

**Conference**: ICML 2025  
**arXiv**: [2507.08438](https://arxiv.org/abs/2507.08438)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Bandit  
**Keywords**: Batched Linear Bandit, Arm Elimination, Regularized G-Optimal Design, Minimax Optimal, Limited Adaptivity

## TL;DR

By deeply integrating the **arm elimination strategy** with **regularized G-optimal design**, BLAE is the first to **simultaneously** achieve minimax optimal regret (up to a $\log T$ factor) under both large-$K$ and small-$K$ regimes in the batched linear bandit problem, while maintaining the minimal batch complexity of $\mathcal{O}(\log\log T)$ and outstanding practical performance.

## Background & Motivation

### Problem Definition

Stochastic linear bandit is a classic framework for sequential decision-making: an agent repeatedly selects from $K$ arms embedded in a $d$-dimensional feature space, receiving a noisy linear reward $r_t = \langle x_t, \theta^* \rangle + \eta_t$ in each round. The goal is to minimize cumulative regret $\mathcal{R}(T) = \mathbb{E}\left[\sum_{t=1}^T \left(\langle x^*, \theta^* \rangle - \langle x_t, \theta^* \rangle\right)\right]$.

Under the **batched** setting, the $T$ rounds are partitioned into $B$ disjoint batches, and policy updates only occur at batch boundaries. This directly models the practical constraint in scenarios such as recommendation systems and clinical trials where policies cannot be frequently updated.

### Limitations of Prior Work

Although several algorithms can theoretically achieve near-optimal regret, they each suffer from various pain points in practice:

**Computational Infeasibility**: The algorithms of Ruan et al. (2021), Hanna et al. (2023), and Zhang et al. (2025) originate from the contextual (time-varying feature) setting and contain subroutines tailored for time-varying scenarios. They cannot exploit the static structure in plain linear bandits with fixed features, making them unrunnable even on medium-scale problems.

**Excessive Batches**: The rare-switching OFUL of Abbasi-Yadkori et al. (2011) requires $\mathcal{O}(d\log T)$ batches, and Esfandiari et al. (2021) requires $\mathcal{O}(\log T)$ batches—which are still too many.

**Premature Arm Elimination**: The method of Ren et al. (2024) often mistakenly eliminates the optimal arm in early stages and is highly sensitive to the hyperparameter $\gamma$. An improper choice can cause the size of the third batch to explode to $\mathcal{O}((\log T)^{1+\gamma})$, consuming the entire time horizon.

**Regime Blind Spot**: Prior to this work, no algorithm could **simultaneously** match the respective regret lower bounds under both large-$K$ ($K \geq \Omega(e^d)$) and small-$K$ ($K \leq \mathcal{O}(e^d)$) regimes.

**Core Problem**: **Can we design a batched linear bandit algorithm that is both theoretically minimax optimal and practically superior?**

## Method

### Overall Architecture: BLAE Algorithm

The core idea of BLAE (Batched Linear bandit Algorithm with Elimination) is to seamlessly combine **batch-wise arm elimination** with **regularized G-optimal experimental design**. The algorithm runs in $\mathcal{O}(\log\log T)$ batches, executing the following steps in each batch:

1. **Construct Confidence Sets**: Estimate $\hat{\theta}$ based on current data and build a confidence ellipsoid for the parameters.
2. **Arm Elimination**: Prune "definitely suboptimal" arms using confidence upper and lower bounds—if the upper bound of an arm's expected reward is lower than the lower bound of the current best arm, it is permanently eliminated.
3. **Regularized G-Optimal Design**: Solve a regularized G-optimal design problem on the set of surviving arms to obtain the optimal allocation ratios for each arm.
4. **Proportional Sampling**: Allocate the pulling budget of each arm in this batch according to the designed ratios, and collect observational data.
5. **Parameter Update**: Update $\hat{\theta}$ and the information matrix $V$ using the new data.

The batch size grows doubly exponentially (with a recurrence structure of $T_m \propto T^{2^{-m}}$), limiting the total number of batches to $\mathcal{O}(\log\log T)$.

### Key Designs

#### Regularized G-Optimal Design

Standard G-optimal design aims to minimize the maximum prediction variance across all arms:

$$\min_{\pi \in \Delta_K} \max_{k \in [K]} \| x^{(k)} \|^2_{H(\pi)^{-1}}, \quad H(\pi) = \sum_k \pi_k x^{(k)} x^{(k)\top}$$

However, under the batched setting, the set of surviving arms shrinks batch by batch, and the information matrix can become ill-conditioned. BLAE introduces **regularization**:

$$H_\lambda(\pi) = \sum_k \pi_k x^{(k)} x^{(k)\top} + \lambda I$$

The regularization parameter $\lambda$ ensures that the information matrix is always well-defined and allows for analyzing smooth transitions from one batch to the next, avoiding warm-start issues.

#### Cross-Batch Optimal Design Analysis

Traditional analysis treats each batch as an independent optimal design problem. The key technical innovation of BLAE lies in the **cross-batch optimal design analysis** (Lemma 3): by using the information matrix accumulated in the previous batch as the regularization term for the next batch, it establishes a fine-grained recurrence relation of information accumulation across batches, thereby obtaining tighter bounds than batch-wise independent analyses.

#### Precise Concentration Inequalities

To simultaneously handle the dependency structures induced by batching and regularization, the authors develop two new concentration inequalities (Lemma 1, 2) that respectively characterize:

- The upper bound of the deviation of the regularized least-squares estimator $\hat{\theta}$ from the true parameter $\theta^*$.
- The precise control of the variance term of the information matrix under the G-optimal design allocation.

These technical tools are applicable to both batched and non-batched bandit research.

#### Efficient Unit Sphere Covering

To handle the large-$K$ regime ($K \geq \Omega(e^d)$), the algorithm requires an efficient $\epsilon$-covering of the unit sphere to restrict the search space. Lemma 8 provides an improved upper bound on the covering size, which both guarantees theoretical bounds and improves practical performance.

### Theoretical Analysis

**Theorem 1 (Main Theorem)**: Under $\mathcal{O}(\log\log T)$ batches, the worst-case cumulative regret of BLAE satisfies:

$$\mathcal{R}(T) = \mathcal{O}\left(\sqrt{dT}\left(\sqrt{\log(KT)} \wedge \sqrt{d + \log T}\right) \log\log T\right)$$

Key characteristics of this bound:

- **small-$K$ regime** ($K \leq \mathcal{O}(e^d)$): The regret is $\mathcal{O}(\sqrt{dT\log K} \cdot \log\log T)$, matching the lower bound $\Omega(\sqrt{dT\log K})$ (Zhou, 2019).
- **large-$K$ regime** ($K \geq \Omega(e^d)$): The regret is $\mathcal{O}(d\sqrt{T} \cdot \log\log T)$, matching the lower bound $\Omega(d\sqrt{T})$ (Dani et al., 2008).
- **Unification of two regimes**: It automatically adapts to both regimes through the minimum operation $\wedge$, without needing prior knowledge of the relationship between $K$ and $d$.

## Key Experimental Results

### Regret Bounds and Batch Complexity Comparison

| Method | Worst-case Regret | Batch Complexity | Practical Feasibility |
|------|---------|-----------|-----------|
| Abbasi-Yadkori et al. (2011) | $\mathcal{O}(d\sqrt{T}\log T)$ | $\mathcal{O}(d\log T)$ | Excessive batches |
| Esfandiari et al. (2021) | $\mathcal{O}(\sqrt{dT\log(KT)})$ | $\mathcal{O}(\log T)$ | Relatively many batches |
| Ruan et al. (2021) | $\mathcal{O}(\sqrt{dT\log(dKT)\log d}\log\log T)$ | $\mathcal{O}(\log\log T)$ | High computation cost |
| Hanna et al. (2023) | $\mathcal{O}(d\sqrt{T\log T}\log\log T)$ | $\mathcal{O}(\log\log T)$ | High computation cost |
| Ren et al. (2024) | $\mathcal{O}(\sqrt{dT\log(KT)}\log\log T)$ | $\mathcal{O}(\log\log T)$ | Prone to premature optimal arm elimination |
| Zhang et al. (2025) | $\mathcal{O}(\sqrt{dT\log(dKT)\log T}\log(dT)\log\log T)$ | $\mathcal{O}(\log\log T)$ | High computation cost |
| **BLAE (Ours)** | $\mathcal{O}(\sqrt{dT}(\sqrt{\log(KT)} \wedge \sqrt{d+\log T})\log\log T)$ | $\mathcal{O}(\log\log T)$ | **Low computational overhead** |
| Lower bound (Dani et al., 2008) | $\Omega(d\sqrt{T})$ | — | large-$K$ |
| Lower bound (Zhou, 2019) | $\Omega(\sqrt{dT\log K})$ | — | small-$K$ |

### Numerical Experiment Performance Comparison

The paper conducts extensive numerical experiments under various settings (different combinations of $d$, $K$, and $T$). BLAE consistently and substantially outperforms existing methods in all tested scenarios:

| Experimental Dimension | BLAE Performance | Baseline Performance | Advantage Description |
|---------|----------|------------|---------|
| Fixed $d=5, K=20$ | Lowest cumulative regret | Ren et al. exhibits significantly higher regret | Ren eliminates the optimal arm prematurely |
| Increasing $K$ (large-$K$) | Regret grows steadily | Regret of most methods increases sharply | G-optimal design effectively controls variance |
| Increasing $d$ | Regret grows moderately | Ruan/Hanna/Zhang suffer computational timeouts | Low computational overhead advantage becomes prominent |
| Different values of $T$ | Consistent with the theoretical $\sqrt{T}$ trend | Some methods deviate from theoretical expectations | Consistency between theory and practice |
| Actual number of batches | Very few ($\sim\log\log T$) | Esfandiari requires dozens of times more batches | True low adaptivity |
| Running time | Seconds | Ruan/Zhang can take up to hours | Performance gap is evident even on medium-scale problems |

## Highlights & Insights

- **Theory-Practice Unification**: BLAE is the first batched linear bandit algorithm that achieves both minimax optimal regret (in both regimes) and outstanding practical performance, breaking the common dilemma of "theoretically optimal $\neq$ practically useful."
- **Regime Adaptivity**: It naturally adapts to both large-$K$ and small-$K$ regimes through the minimum operation $\wedge$, requiring no prior knowledge or regime detection.
- **Computational Feasibility**: It avoids heavy subroutines customized for contextual settings (such as phased estimation-policy updates). By fully taking advantage of the static structure under the fixed feature setting, it achieves low computational overhead.
- **Crucial Role of Regularization**: Regularization is not just a technical trick (to assure matrix invertibility), but also the core lever for implementing cross-batch information accumulation analysis—an insight that can be generalized to other online learning problems.
- **Robustness of Elimination Policy**: Compared to the aggressive elimination of Ren et al., BLAE's elimination is more conservative. Combined with the variance control of G-optimal design, it effectively avoids the fatal error of prematurely discarding the optimal arm.

## Limitations & Future Work

1. **Fixed Feature Assumption**: BLAE is designed and optimized for fixed features (plain linear bandit) and does not directly handle time-varying features in contextual settings. Extending it to time-varying features requires additional mechanisms.
2. **$\log\log T$ Factor**: Although the batch complexity has reached the optimal $\mathcal{O}(\log\log T)$ level, the regret bound still contains a $\log\log T$ factor. Whether it can be completely eliminated remains an open question.
3. **Finite Arm Set**: The algorithm assumes a finite arm set $K < \infty$. For linear bandits with continuous arm spaces (e.g., $\mathcal{A} = \{x : \|x\| \leq 1\}$), discretization via covering nets is required, which may introduce additional loss.
4. **Solving G-Optimal Design**: It requires solving a convex optimization problem, which, though far less computationally heavy than rival methods, could still become a bottleneck in scenarios with extremely large $K$.
5. **Noise Model**: It assumes $\sigma$-sub-Gaussian noise; its applicability to heavy-tailed noise or heteroscedastic scenarios remains to be validated.

## Related Work & Insights

- **Abbasi-Yadkori et al. (2011)**: Proposed OFUL and its rare-switching variant, laying the foundation of batched linear bandit research, but with a batch complexity of $\mathcal{O}(d\log T)$.
- **Esfandiari et al. (2021)**: Achieved $\mathcal{O}(\log T)$ batches under the adversarial features setting for the first time, but the regret bound is sub-optimal in the large-$K$ regime.
- **Ruan et al. (2021) / Zhang et al. (2025)**: Achieved $\mathcal{O}(\log\log T)$ batches, but their high computational complexity makes them practically infeasible.
- **Ren et al. (2024)**: The closest competitor, achieving optimal regret under small-$K$ but is sub-optimal under large-$K$, with unstable practical performance.
- **Dani et al. (2008) / Zhou (2019)**: Respectively established regret lower bounds under large-$K$ and small-$K$ regimes, providing baselines for optimality.
- **Insight**: The idea of using regularized G-optimal design as an exploration strategy in the batched setting is generalizable, and can be extended to areas such as batched contextual bandits and batched Bayesian optimization. The paradigm of "lightweight algorithm design + fine-grained analysis" can serve as a reference pathway to bridge the theory-practice gap.

## Rating

| Dimension | Score (1-10) | Description |
|------|-----------|------|
| Novelty | 8 | Original fusion design of regularized G-optimal design + arm elimination; achieving unified optimality under both regimes is a first. |
| Technical Depth | 9 | Solid technical tools, including cross-batch optimal design analysis and new concentration inequalities. |
| Experimental Thoroughness | 8 | Comprehensive comparison across multiple dimensions; the comparison of computational efficiency is particularly persuasive. |
| Practical Value | 8 | Low computational overhead + strong practical performance; can be directly deployed in recommendation, clinical trial, and similar scenarios. |
| Writing Quality | 8 | Clear problem motivation, good correspondence between theory and experiments, and Table 1 is clear at a glance. |
| **Overall** | **8.2** | A perfect unification of theory and practice, simultaneously achieving two-regime optimality and practicality for the first time in batched linear bandits. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates](../../ICML2026/reinforcement_learning/practical_and_optimal_algorithm_for_linear_contextual_bandits_with_rare_paramete.md)
- [\[ICML 2025\] Actor-Critics Can Achieve Optimal Sample Efficiency](actor-critics_can_achieve_optimal_sample_efficiency.md)
- [\[ICML 2025\] Principal-Agent Bandit Games with Self-Interested and Exploratory Learning Agents](principal-agent_bandit_games_with_self-interested_and_exploratory_learning_agent.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](../../NeurIPS2025/reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](../../NeurIPS2025/reinforcement_learning/generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)

</div>

<!-- RELATED:END -->
