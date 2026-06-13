---
title: >-
  [Paper Note] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning
description: >-
  [NeurIPS 2025][Robotics][offline RL] This paper proposes the RTZ-VI-LCB algorithm for offline robust two-player zero-sum Markov games (RTZM G). By combining pessimistic robust value iteration with Bernstein-style penalti…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "offline RL"
  - "robust MDP"
  - "zero-sum Markov game"
  - "sample complexity"
  - "pessimism"
date: 2026-05-08
content_hash: 05a03963f2fc6108
---

# Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2512.00352](https://arxiv.org/abs/2512.00352)  
**Code**: None  
**Area**: Robust Reinforcement Learning / Offline RL
**Keywords**: offline RL, robust MDP, zero-sum Markov game, sample complexity, pessimism

## TL;DR
This paper proposes the RTZ-VI-LCB algorithm for offline robust two-player zero-sum Markov games (RTZM G). By combining pessimistic robust value iteration with Bernstein-style penalties, it achieves a near-optimal sample complexity of $O(C_r^* \cdot H^4 \cdot S \cdot (A+B) / \varepsilon^2)$, significantly improving upon the prior best result of $O(H^5 \cdot S^2 \cdot AB / \varepsilon^2)$ in terms of dependence on both the state space and the action space.

## Background & Motivation

### State of the Field

**Background**: Offline RL learns policies from a fixed dataset without online interaction. Robust RL further accounts for environmental uncertainty, where the transition kernel may vary within an uncertainty set. The combination of these two settings forms the emerging direction of offline robust RL.

**Limitations of Prior Work**: (a) Existing offline robust RL methods exhibit redundant dependence on $S$ (number of states) and $AB$ (joint action space size) in their sample complexity bounds; (b) distribution coverage measures are overly conservative — standard concentrability coefficients are excessively pessimistic for worst-case analysis.

**Key Challenge**: How can one simultaneously handle distributional shift and environmental uncertainty under partial coverage, while obtaining tight sample complexity bounds?

**Key Insight**: The paper introduces *robust unilateral clipped concentrability* as a tighter distribution measure, and employs two-stage subsampling to suppress statistical dependencies, thereby achieving information-theoretically optimal sample bounds.

**Core Idea**: A model-based approach — first estimate the uncertainty set of the transition kernel, then perform pessimistic (LCB) value iteration over this set, using Bernstein inequalities to obtain tighter statistical estimates.

## Method

### Overall Architecture
- **Input**: Offline dataset ($N$ tuples of $(s, a_1, a_2, r, s')$), uncertainty set radius $\rho$
- **Algorithm**: RTZ-VI-LCB (Robust Two-player Zero-sum Value Iteration with Lower Confidence Bound)
- **Output**: Near-optimal robust policy $\hat{\pi}$

### Key Designs

1. **Robust Value Iteration with Pessimistic Estimation**

    - **Function**: Performs worst-case optimization over the transition kernel within the estimated uncertainty set, while applying LCB penalties to state-action pairs with insufficient coverage.
    - **Mechanism**: $\hat{Q}_h(s,a,b) = \hat{r}_h(s,a,b) + \min_{P \in \hat{U}(s,a,b)} P \hat{V}_{h+1} - \Gamma_h(s,a,b)$, where $\hat{U}$ is the estimated uncertainty set and $\Gamma_h$ is the Bernstein penalty term.
    - **Design Motivation**: The pessimism principle prevents overly optimistic value estimates in regions with insufficient data.

2. **Robust Unilateral Clipped Concentrability**

    - **Function**: Defines a tighter distribution coverage measure $C_r^*$.
    - **Mechanism**: Coverage is required only for the state-action distribution of the target policy (rather than all policies), and clipping is applied to eliminate extreme values in density ratios.
    - **Comparison with Prior Measures**: Strictly smaller than the standard concentrability coefficient, permitting more relaxed data collection conditions.

3. **Two-Stage Subsampling**

    - **Function**: Randomly partitions the dataset into two disjoint subsets, used separately for uncertainty set construction and value iteration.
    - **Design Motivation**: Suppresses statistical dependencies between estimation errors and value functions, simplifying the theoretical analysis.

### Loss & Training
- **Uncertainty set construction**: Based on the empirical transition kernel $\hat{P}_h$ and TV/KL/chi-squared balls.
- **Bernstein penalty**: $\Gamma_h(s,a,b) = c \sqrt{\frac{\hat{\text{Var}} V_{h+1}}{N_h(s,a,b)}} + \frac{H}{N_h(s,a,b)}$
- Pure offline learning; no online interaction is required.

## Key Experimental Results

### Sample Complexity Comparison

### Main Results

| Method | Sample Complexity | State Dependence | Action Dependence |
|--------|------------------|-----------------|------------------|
| Prev. SOTA | $O(H^5 S^2 AB / \varepsilon^2)$ | $S^2$ | $AB$ |
| **RTZ-VI-LCB** | $O(C_r^* H^4 S (A+B) / \varepsilon^2)$ | $S$ | $A+B$ |
| Information-theoretic lower bound | $\Omega(C_r^* H^3 S (A+B) / \varepsilon^2)$ | $S$ | $A+B$ |

### Key Findings
- The sample complexity matches the information-theoretic lower bound in $S$ and $A+B$ up to an $H$ factor.
- The algorithm is optimal under both high- and low-uncertainty regimes.
- The robust unilateral clipped concentrability coefficient is strictly smaller than the standard coefficient.

## Highlights & Insights
- **First optimal sample complexity for offline RTZM G**: Achieves information-theoretic optimality in the $S$ and $A+B$ dimensions.
- **New distribution measure**: Robust unilateral clipped concentrability is a general-purpose tool potentially applicable to other offline RL problems.
- **Unified treatment of high and low uncertainty regimes**: A single algorithmic framework achieves optimal performance in both extremes.

## Limitations & Future Work
- **Tabular setting only**: Assumes a finite state-action space; function approximation is not considered.
- **Gap of an $H$ factor**: Whether the remaining gap relative to the lower bound can be closed remains open.
- **Restriction to two-player zero-sum games**: Although extensions to multi-player general-sum games are discussed, the core analysis is limited to the two-player zero-sum setting.
- **Limited empirical validation**: The contribution is primarily theoretical, with minimal experimental verification.

## Related Work & Insights
- **vs. Blanchet et al. (2024)**: Their sample complexity is $O(S^2 AB)$; this work improves it to $O(S(A+B))$.
- **vs. Shi et al. (2022)**: They analyze offline robust single-agent RL; this work extends the framework to two-player zero-sum games.
- **vs. Cui & Du (2022)**: They study offline (non-robust) Markov games; this work incorporates environmental robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ New distribution measure and optimal sample complexity bounds
- Experimental Thoroughness: ⭐⭐ Primarily theoretical; limited empirical evaluation
- Writing Quality: ⭐⭐⭐⭐ Theorems are clearly stated; technically rigorous
- Value: ⭐⭐⭐⭐ Significant theoretical advancement for offline robust RL

## Additional Remarks
- The theoretical analysis framework and technical tools developed in this paper offer insights for adjacent research areas.
- The core contribution lies in a deeper theoretical understanding, providing a foundation for subsequent practical improvements.
- The paper is methodologically complementary to other NeurIPS 2025 papers published concurrently.
- The paper's exposition of problem motivation and technical approach is worth studying.
- Readers are encouraged to consult the appendix for complete experimental details and proofs.

## Further Reading
- This research direction is closely related to several active topics in the AI community.
- The rigor of the theoretical results provides a solid mathematical foundation for subsequent empirical studies.
- The methodology can be generalized to broader problem settings.
- Follow-up work from the same research group is worth monitoring.
- For newcomers to the theoretical direction, the proof sketch section provides an excellent technical roadmap.
- From a methodological perspective, the paper demonstrates how careful mathematical modeling can reduce complex problems to analytically tractable frameworks.

## Technical Notes
- The proof of the core theorem relies on multi-step refined concentration inequality analysis and properties of the Bellman equation.
- The choice between pessimism and optimism in algorithm design is a central consideration in offline RL theory.
- Logarithmic factors hidden in the theoretical bounds are non-negligible in practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)

</div>

<!-- RELATED:END -->
