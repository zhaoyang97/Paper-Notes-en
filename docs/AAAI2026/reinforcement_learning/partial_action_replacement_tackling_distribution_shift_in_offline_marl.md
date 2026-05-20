---
title: >-
  [Paper Note] Partial Action Replacement: Tackling Distribution Shift in Offline MARL
description: >-
  [AAAI 2026][Reinforcement Learning][Offline Multi-Agent Reinforcement Learning] This paper proposes the Partial Action Replacement (PAR) principle, theoretically proving that under a factorized behavior policy…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Offline Multi-Agent Reinforcement Learning"
  - "Distribution Shift"
  - "Partial Action Replacement"
  - "Conservative Q-Learning"
  - "Uncertainty Estimation"
date: 2026-05-08
content_hash: 670557d88ccae6f3
---

# Partial Action Replacement: Tackling Distribution Shift in Offline MARL

**Conference**: AAAI 2026
**arXiv**: [2511.07629](https://arxiv.org/abs/2511.07629)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Offline Multi-Agent Reinforcement Learning, Distribution Shift, Partial Action Replacement, Conservative Q-Learning, Uncertainty Estimation

## TL;DR

This paper proposes the Partial Action Replacement (PAR) principle, theoretically proving that under a factorized behavior policy, distribution shift grows linearly with the number of deviating agents (rather than exponentially in the joint action space). Building on this, the SPaCQL algorithm is developed to dynamically weight different PAR operators via Q-ensemble uncertainty, achieving substantial improvements over all baselines on Random and Medium-Replay datasets.

## Background & Motivation

Offline Multi-Agent Reinforcement Learning (Offline MARL) faces core challenges rooted in the combinatorial explosion inherent to multi-agent systems:

**Curse of dimensionality in joint action spaces**: Any finite dataset provides only sparse coverage over all possible action combinations, forcing learning algorithms to reason about the value of countless out-of-distribution (OOD) joint actions. Standard Q-learning is highly susceptible to failure in this regime—function approximators (e.g., neural networks) can assign arbitrarily high Q-values to unseen OOD actions, steering agents toward divergent policies.

**Geometric intuition**: An offline dataset constitutes a sparse set of known points in a high-dimensional joint action space. Standard Q-learning updates require evaluating joint actions where all agents follow the new policy, effectively querying Q-values far from any known data point and forcing large, unreliable extrapolations.

**Core insight of partial action replacement**: If only one or a few agents' actions are modified while the remaining agents' actions are drawn from the dataset, the queried joint action differs from known data by only one coordinate—requiring only minor local extrapolation rather than a leap into unknown territory. A key prerequisite is that data are collected by independently acting agents (factorized behavior policies), which is common in practice (independent human demonstrations, independently trained agents, decentralized systems, etc.).

**Stability–coordination trade-off**: Replacing only a single agent's action (ICQL-QS) is stable but may miss valuable multi-agent coordination; replacing all agents' actions is effective but risky. An adaptive approach is needed to balance this trade-off.

## Method

### Overall Architecture

Building on the PAR principle, two algorithms are proposed:
1. **ICQL-QS** (Individual CQL with Q-sharing): A conservative baseline that replaces only one agent's action per update.
2. **SPaCQL** (Soft-Partial Conservative Q-Learning): The main contribution, adaptively mixing Bellman operators with different replacement counts.

### Key Designs

1. **ICQL-QS: A stable but conservative baseline**: For each agent $i$, an individual Bellman operator $\mathcal{T}_i^{\text{ind}}$ is defined—agent $i$'s next action $a_i'$ is sampled from its learned policy $\pi_i$, while all other agents' actions $a_{-i}'$ are drawn from the dataset $\mathcal{D}$:

   $\mathcal{T}_i^{\text{ind}} Q(s, \boldsymbol{a}) := \mathbb{E}_{(s', \boldsymbol{a}'_{-i}) \sim \mathcal{D}, a'_i \sim \pi_i(\cdot|s')}[r + \gamma Q(s', a'_i, \boldsymbol{a}'_{-i})]$

   The loss includes a TD error term and a CQL conservative regularization term. Although seemingly "myopic" (considering only single-agent deviations at a time), the shared Q-function provides implicit coupling: each update furnishes learning signals for all possible joint actions. **Proposition 1 proves** that the gradient of ICQL-QS is equivalent to stochastic gradient descent on the centralized averaged Bellman operator $\mathcal{T}^{ai} = \frac{1}{n}\sum_{i=1}^{n}\mathcal{T}_i^{\text{ind}}$.

2. **SPaCQL: Adaptive mixture of partial backups**: The core observation is that no single fixed backup strategy is optimal across all data qualities. Random datasets favor conservative single-agent updates; Expert datasets may require coordinated multi-agent deviations.

   Define $n$ base Bellman operators $\{\mathcal{T}^{(k)}\}_{k=1}^{n}$, where $\mathcal{T}^{(k)}$ replaces exactly $k$ agents' actions (selected uniformly at random):

   $\mathcal{T}^{(k)} Q(s, \boldsymbol{a}) := \mathbb{E}_{s' \sim \mathcal{D}, \boldsymbol{a}'^{(k)}}[r + \gamma Q(s', \boldsymbol{a}'^{(k)})]$

   The SPaCQL operator is a convex combination of these base operators: $\mathcal{T}^{SP} Q = \sum_{k=1}^{n} w_k \mathcal{T}^{(k)} Q$. As a convex combination of $\gamma$-contractions, $\mathcal{T}^{SP}$ is itself guaranteed to be a $\gamma$-contraction.

3. **Uncertainty-based adaptive weights**: High ensemble disagreement implies insufficient data coverage, so high-risk deviations should be down-weighted. Uncertainty is measured via the variance of a Q-function ensemble:

   $u_k = \sqrt{\text{Var}_j[Q_{\theta_j}(s', \boldsymbol{a}'^{(k)})]}$

   Weights are normalized inverse uncertainties: $w_k = \frac{1/u_k}{\sum_k 1/u_k}$

   Configurations with high uncertainty (large $k$, multi-agent deviation) receive low weight, while those with low uncertainty receive high weight.

   Target value construction: $Y_{SP} = r + \gamma \sum_{k=1}^{n} w_k \min_j Q_j^{tar}(s', \boldsymbol{a}'^{(k)})$

### Loss & Training

Full loss: $\mathcal{L}(\theta) = \mathbb{E}_\mathcal{D}[(Q_\theta(s, \boldsymbol{a}) - Y_{SP})^2] + \xi_c$

where the conservative regularization term $\xi_c = \alpha \sum_{i=1}^{n} \lambda_i (\mathbb{E}_{a_i \sim \pi_i}[Q_\theta(s, a_i, \boldsymbol{a}_{-i})] - \mathbb{E}_\mathcal{D}[Q_\theta(s, \boldsymbol{a})])$ follows CFCQL.

Implementation: 10 Q-network ensemble, 5 random seeds, all hyperparameters consistent with the CFCQL paper.

## Theoretical Contributions

The paper's most significant theoretical contribution is a rigorous proof of the superiority of partial action replacement:

**Lemma 1 (Linear divergence bound)**: For any subset $S \subseteq \{1,...,n\}$, $W_1(d^{(S)}, d^{(\varnothing)}) \leq \frac{\gamma}{1-\gamma} \sum_{i \in S} \text{TV}(\pi_i, \mu_i)$
→ Distribution shift grows linearly with the number of deviating agents, not exponentially in the joint space.

**Theorem 1 (Tight value error bound)**: $|V^\pi - \hat{V}^\pi| \leq \varepsilon_{\text{Subopt}} + \varepsilon_{\text{FQI}} + \frac{4\gamma}{(1-\gamma)^2} \sum_{i=1}^{n} \text{TV}(\pi_i, \mu_i)$
→ Under single-agent deviation, the third term reduces to $\frac{4\gamma}{(1-\gamma)^2} \text{TV}(\pi_k, \mu_k)$, strictly improving the joint TV bound.

**Theorem 2 (Extension to correlated behavior policies)**: Introducing the maximum excess correlation $\kappa$, the error bound acquires an additive constant $\kappa$ independent of $n$, while maintaining linear scaling.

**Theorem 3 (SPaCQL error bound)**: The error scales with the effective number of deviating agents $k_{eff} = \sum_k w_k \cdot k$, which the algorithm adaptively regulates.

## Key Experimental Results

### Main Results

Evaluation on MPE (Cooperative Navigation, Predator-Prey, World) and MaMujoco (Half-Cheetah).

| Task | Dataset | OMAR | MACQL | IQL | CFCQL | DoF | **SPaCQL** |
|------|---------|------|-------|-----|-------|-----|-----------|
| CN | Random | 34.4 | 45.6 | 5.5 | 62.2 | 35.9 | **78.2±14** |
| CN | Med-R | 37.9 | 25.5 | 10.8 | 52.2 | 57.4 | **71.9±13.2** |
| CN | Expert | 114.9 | 12.2 | 103.7 | 112 | **136.4** | 111.9 |
| PP | Random | 11.1 | 25.2 | 1.3 | 16.5 | - | **89.4±13.7** |
| PP | Med-R | 47.1 | 11.9 | 23.2 | 71.1 | 65.4 | **75.0±12.7** |
| World | Random | 5.9 | 11.7 | 2.9 | 68 | 13.1 | **94.3±7.4** |
| World | Med-R | 42.9 | 13.2 | 41.5 | 73.4 | 58.6 | **105.2±11.1** |
| Half-C | Random | 13.5 | 5.3 | 7.4 | 39.7 | - | **43.8±4.9** |
| Half-C | Med-R | 57.7 | 37.0 | 58.8 | 59.5 | - | **66.1±3.4** |

SPaCQL achieves the best performance on **all Random and Med-R datasets**, winning 10 out of 16 tasks.

### Ablation Study

| Configuration | CN-Rand | CN-Expert | World-Rand | World-Expert | Notes |
|--------------|---------|-----------|------------|-------------|-------|
| CFCQL | 62.2 | **112** | 68 | **119.7** | Full joint update |
| ICQL-QS | 77.7 | 97.2 | 89.9 | 106.5 | Single-agent replacement only |
| SPaCQL | **78.2** | 111.9 | **94.3** | 112.3 | Adaptive mixture |

ICQL-QS vs. CFCQL validates the core trade-off:
- ICQL-QS leads substantially on Random data (greater benefit from partial replacement)
- CFCQL is marginally superior on Expert data (joint updates needed to capture coordination)
- SPaCQL approaches optimality on both ends

### Key Findings

1. **Dominant advantage on Random/Med-R datasets**: SPaCQL scores 94.3 vs. CFCQL's 68 (+38.7%) on World-Random, and 89.4 vs. 16.5 (+441.8%) on PP-Random. When agent behaviors are independent or weakly coordinated, the advantage of partial action replacement is remarkably pronounced.

2. **Comparable performance on Expert datasets**: On high-quality coordinated data, SPaCQL matches the best baselines, indicating that adaptive weights correctly increase $w_n$ (more joint deviations).

3. **Visualization of adaptive weights**: On Random data, $w_1$ (single-agent deviation) dominates; on Expert data, $w_2, w_3$ increase—fully consistent with theoretical expectations.

4. **Uncertainty estimation validation**: ICQL-QS Q-value estimation uncertainty is consistently lower than CFCQL on Random data, with both being similar on Expert data.

5. **Theoretical validation of linear scaling**: Partial replacement is not a heuristic compromise but carries rigorous theoretical guarantees—distribution shift and value error both scale linearly.

## Highlights & Insights

- **Theoretical contributions surpass algorithmic ones**: The greatest value of this paper lies in Lemma 1 and Theorems 1–3, which formally prove that "the curse of dimensionality may be overstated in offline MARL"—a finding of methodological significance for the entire field.
- **Implicit coordination in ICQL-QS** (Proposition 1): What appears to be independent updates is in fact equivalent to gradient descent on a centralized objective—this insight dispels the intuitive concern that partial replacement cannot support coordination.
- **Extension to correlated behavior policies** (Theorem 2): Real-world data are rarely fully independent; by introducing maximum excess correlation $\kappa$ as an additive penalty term, the theoretical guarantees remain valid.
- **Algorithmic simplicity**: SPaCQL is essentially an "uncertainty-weighted convex combination of multiple Bellman operators"—conceptually clear, straightforward to implement, and theoretically well-grounded.

## Limitations & Future Work

1. Theoretical analysis assumes finite state-action spaces and i.i.d. transitions, which may not hold when neural networks and trajectory data are used in practice.
2. Theorem 1 assumes Q-functions are $2/(1-\gamma)$-Lipschitz, which requires additional technical guarantees such as spectral normalization for neural networks.
3. Uncertainty estimation relies solely on Q-ensemble variance; more informative metrics may exist.
4. SPaCQL fails to lead on all Expert datasets, indicating that full joint updates still hold an advantage on high-quality coordinated data.
5. Validation is limited to simple environments (MPE, MaMujoco), leaving a substantial gap relative to real-world multi-agent scenarios.

## Related Work & Insights

- **CFCQL** (Shao et al. 2023): The multi-agent extension of CQL most directly related to this work—it also employs partial replacement but only for regularization rather than target value computation, and lacks adaptive weighting.
- **CQL** (Kumar et al. 2020): The single-agent conservative Q-learning method that underpins this paper's theoretical analysis.
- **IQL** (Kostrikov et al. 2022): Implicit Q-learning, which avoids OOD action queries.
- **DoF** (Li et al. 2025): Diffusion-model-based offline MARL, strong on Expert data.
- **SAC-N / EDAC** (An et al. 2021): Use of Q-ensemble variance for uncertainty estimation—the inspiration for SPaCQL's weight design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Formalization of partial action replacement + linear scaling proof + adaptive mixture operator; outstanding theoretical contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive coverage across 4 tasks × 4 dataset types, intuitive weight visualization, though environment complexity is limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous and clear theoretical derivations, well-motivated exposition, Figures 1/2 intuitively convey the geometric intuition)
- Value: ⭐⭐⭐⭐⭐ (Provides a more optimistic theoretical perspective on offline MARL; contributions at the methodological level)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL](../../NeurIPS2025/reinforcement_learning/oryx_a_scalable_sequence_model_for_many-agent_coordination_in_offline_marl.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](object-centric_latent_action_learning.md)
- [\[AAAI 2026\] Efficient Multiagent Planning via Shared Action Suggestions](efficient_multiagent_planning_via_shared_action_suggestions.md)
- [\[AAAI 2026\] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)
- [\[AAAI 2026\] SafeMIL: Learning Offline Safe Imitation Policy from Non-Preferred Trajectories](safemil_learning_offline_safe_imitation_policy_from_non-preferred_trajectories.md)

</div>

<!-- RELATED:END -->
