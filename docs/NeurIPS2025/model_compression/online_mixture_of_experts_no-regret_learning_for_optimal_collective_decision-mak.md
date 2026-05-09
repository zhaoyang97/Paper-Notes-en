---
title: >-
  [Paper Note] Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making
description: >-
  [NeurIPS 2025][Model Compression][online learning] This paper proposes an Online Mixture of Experts (OMoE) framework comprising two algorithms — UCB-Successive Elimination and Online Weighted Majority Voting — with theoretical no-regret guarantees, and applies them to the online dynamic aggregation of LLM experts.
tags:
  - NeurIPS 2025
  - Model Compression
  - online learning
  - mixture of experts
  - bandit
  - no-regret
  - LLM ensemble
date: 2026-05-08
content_hash: b13895f39b885c30
---

# Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making

**Conference**: NeurIPS 2025
**arXiv**: [2510.21788](https://arxiv.org/abs/2510.21788)
**Code**: None
**Area**: Online Learning / Ensemble Methods
**Keywords**: online learning, mixture of experts, bandit, no-regret, LLM ensemble

## TL;DR
This paper proposes an Online Mixture of Experts (OMoE) framework comprising two algorithms — UCB-Successive Elimination and Online Weighted Majority Voting — with theoretical no-regret guarantees, and applies them to the online dynamic aggregation of LLM experts.

## Background & Motivation
**Background**: Mixture of Experts (MoE) has been widely studied in offline settings, but systematic research on dynamically aggregating multiple expert outputs for optimal collective decision-making in online settings remains lacking.

**Limitations of Prior Work**: Traditional MoE assumes fixed gating/routing mechanisms and cannot adapt to dynamic changes in expert quality under streaming data; multi-armed bandit frameworks assume independent arms and fail to exploit the combinatorial structure among experts.

**Key Challenge**: The classical exploration–exploitation trade-off — trying new expert combinations versus leveraging known optimal ones — is considerably more complex in the aggregation dimension than standard bandit problems.

**Key Insight**: Expert aggregation is modeled as a contextual bandit problem, combining UCB and weighted voting effectively.

**Core Idea**: MoE is reformulated as an online bandit learning problem, yielding theoretically grounded no-regret expert aggregation algorithms.

## Method

### Overall Architecture
Given $K$ experts and context $x_t$, at each round the algorithm selects a subset of experts (a committee) and aggregation weights, then observes the accuracy of the aggregated output as feedback. The objective is to minimize cumulative regret $R_T = \sum_{t=1}^T [r^* - r_t]$.

### Key Designs

1. **Algorithm 1: UCB-Successive Elimination (UCB-SE)**

    - **Function**: Progressively eliminates suboptimal expert committees via UCB confidence bounds.
    - **Mechanism**: Maintains UCB estimates $\hat{\mu}_k + \sqrt{\frac{2\ln t}{n_k}}$ for each candidate committee.
    - **Elimination Rule**: A committee is eliminated when its UCB falls below the LCB of the current best committee.
    - **Regret Bound**: $R_T = O(\sqrt{KT \log T})$
    - **Novelty**: Efficient pruning reduces exploration waste.

2. **Algorithm 2: Online Weighted Majority Voting (OWMV)**

    - **Function**: Dynamically assigns voting weights to experts based on their historical performance.
    - **Update Rule**: $w_k^{(t+1)} = w_k^{(t)} \cdot \beta^{\mathbb{1}[\text{expert } k \text{ incorrect}]}$
    - The decay factor $\beta \in (0,1)$ penalizes incorrect experts.
    - **Regret Bound**: $R_T = O(\sqrt{T \log K})$
    - **Novelty**: Eliminates the need to enumerate all committee combinations.

3. **LLM Application: Online Expert LLM Aggregation**

    - **Scenario**: Dynamic aggregation of $K$ specialized LLMs (code, reasoning, creative writing, etc.).
    - Expert weights or the optimal committee are updated after each query.
    - **Aggregation**: Weighted voting combined with top-$k$ selection.

### Theoretical Guarantees
- **Theorem 1 (UCB-SE)**: For $K$ experts and $T$ rounds, $R_T \leq O\!\left(\sum_{k \neq k^*} \frac{\log T}{\Delta_k}\right)$, where $\Delta_k$ denotes the suboptimality gap.
- **Theorem 2 (OWMV)**: $R_T \leq O(\sqrt{T \log K})$, consistent with Hedge/MW algorithms but applicable under bandit feedback.

## Key Experimental Results

### Main Results — Synthetic Data: Cumulative Regret Comparison ($K=10$, $T=10000$)

| Algorithm | Cumulative Regret | Convergence Round |
|-----------|-------------------|-------------------|
| Uniform Random | 2847 | — |
| EXP3 | 892 | ~3000 |
| UCB1 | 743 | ~2500 |
| **UCB-SE (Ours)** | **531** | **~1800** |
| **OWMV (Ours)** | **487** | **~1500** |

### LLM Expert Aggregation — Accuracy ($K=5$ Expert LLMs)

| Method | MMLU↑ | GSM8K↑ | HumanEval↑ | Avg.↑ |
|--------|-------|--------|------------|-------|
| Best Single Expert | 68.2 | 71.5 | 62.8 | 67.5 |
| Fixed Uniform Weights | 65.1 | 68.3 | 60.2 | 64.5 |
| Oracle (Hindsight Optimal) | 73.8 | 76.2 | 68.1 | 72.7 |
| **UCB-SE** | **71.2** | **73.8** | **65.9** | **70.3** |
| **OWMV** | **72.1** | **74.5** | **66.7** | **71.1** |

### Ablation Study — Effect of Number of Experts $K$ (OWMV)

| $K$ | Regret at $T=1000$ | Regret at $T=5000$ | Regret at $T=10000$ |
|-----|--------------------|--------------------|---------------------|
| 3 | 87 | 195 | 312 |
| 5 | 134 | 289 | 421 |
| 10 | 198 | 387 | 487 |
| 20 | 276 | 512 | 634 |

### Key Findings
- OWMV achieves the lowest regret and fastest convergence across all settings.
- UCB-SE is particularly efficient when the number of experts is small, owing to effective pruning.
- In the LLM aggregation setting, OWMV surpasses the best single expert by 3–4% and approaches the Oracle upper bound.
- The regret growth rate is consistent with the theoretical prediction of $O(\sqrt{T \log K})$.
- From approximately round 1000 onward, OWMV stably identifies the optimal expert committee.

## Highlights & Insights
- **Dual Theoretical-Applied Contribution**: The paper provides both rigorous regret bound proofs and practical application to LLM aggregation.
- **Algorithmic Simplicity**: OWMV is straightforward to implement with negligible additional computational overhead.
- **Dynamic LLM Routing**: The framework offers a theoretical foundation for expert selection during large model inference.

## Limitations & Future Work
- The current framework assumes stationary expert quality over time; non-stationary settings require extensions.
- The aggregation mechanism in LLM experiments is relatively simple (majority voting); token-level mixing may be more effective.
- Theoretical guidance for selecting committee size is lacking.
- The relationship with MoE routing methods (e.g., Switch Transformer) warrants further investigation.

## Related Work & Insights
- **Hedge (Freund & Schapire 1997)**: Classic weighted majority voting algorithm.
- **EXP3 (Auer et al. 2002)**: Online learning under bandit feedback.
- **Switch Transformer (Fedus et al. 2022)**: Sparse MoE routing.
- **Insight**: No-regret theory can be applied to online optimization of MoE routing.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel bandit perspective on MoE
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic and LLM settings
- Writing Quality: ⭐⭐⭐⭐ Algorithm presentation is clear
- Value: ⭐⭐⭐⭐ Practical value for dynamic LLM aggregation

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Geometry of Decision Making in Language Models](geometry_of_decision_making_in_language_models.md)
- [\[NeurIPS 2025\] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP](multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)
- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)

<!-- RELATED:END -->
