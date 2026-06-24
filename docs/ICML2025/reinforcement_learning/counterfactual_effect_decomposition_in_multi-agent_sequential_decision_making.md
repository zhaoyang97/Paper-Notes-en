---
title: >-
  [Paper Note] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making
description: >-
  [ICML 2025][Reinforcement Learning][Counterfactual reasoning] This paper proposes a bi-level causal decomposition framework that systematically decomposes the Total Counterfactual Effect (TCFE) of an action in multi-agent sequential decision-making into the "effect propagated through agent behavior" (tot-ASE) and the "effect propagated through state transitions" (r-SSE), and further attributes them to individual agents and state variables using Shapley values and Intrinsic Ca…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Counterfactual reasoning"
  - "causal explanation"
  - "multi-agent MDP"
  - "Shapley value"
  - "explainability"
date: 2026-05-08
content_hash: 9f401d4d0815c66e
---

# Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making

**Conference**: ICML 2025  
**arXiv**: [2410.12539](https://arxiv.org/abs/2410.12539)  
**Code**: [GitHub](https://github.com/stelios30/cf-effect-decomposition)  
**Area**: Reinforcement Learning  
**Keywords**: Counterfactual reasoning, causal explanation, multi-agent MDP, Shapley value, explainability

## TL;DR

This paper proposes a bi-level causal decomposition framework that systematically decomposes the Total Counterfactual Effect (TCFE) of an action in multi-agent sequential decision-making into the "effect propagated through agent behavior" (tot-ASE) and the "effect propagated through state transitions" (r-SSE), and further attributes them to individual agents and state variables using Shapley values and Intrinsic Causal Contribution (ICC), respectively.

## Background & Motivation

In multi-agent sequential decision-making scenarios (such as human-AI collaborative medical decision-making), counterfactual reasoning is a core tool for the retrospective analysis of decision impacts. Given an actual trajectory, one wants to know "how would the outcome change if a different action had been taken at a certain timestamp" — this is the **Total Counterfactual Effect (TCFE)**.

However, TCFE itself is only a scalar and cannot explain **why** and **how** the effect arises:
- Is the effect propagated by altering the behaviors of other subsequent agents?
- Or is it propagated by altering environmental state transitions?
- Which agent contributes the most to the effect?
- Which state variables are the most critical?

Traditional causal mediation analysis decomposes effects by enumerating causal paths. However, in multi-agent MDPs (MMDPs), the causal paths from actions to outcomes grow exponentially, and many paths lack intuitive operational meanings. Therefore, a decomposition method naturally compatible with the structured nature of MMDPs is required.

## Method

### Overall Architecture

The core of this work is a **Bi-level Decomposition framework**:

**Level 1 — Causal Explanation Formula**: TCFE is decomposed into two components with clear physical interpretations:

$$\text{TCFE} = \text{tot-ASE} - \text{r-SSE}$$

- **tot-ASE (Total Agent-Specific Effect)**: The effect propagated solely through changes in the behavior of all subsequent agents (while state transitions maintain the "original" mechanism).
- **r-SSE (Reversed State-Specific Effect)**: The effect that is "lost or gained" assuming all agents have already acted counterfactually, but state transitions are not affected by the intervention.

This formula is a generalization of Pearl's (2001) classic causal mediation formula to multi-agent MDPs.

**Level 2a — ASE-SV**: Attributes tot-ASE further to each individual agent using Shapley values.

**Level 2b — r-SSE-ICC**: Attributes r-SSE further to each state variable using the Intrinsic Causal Contribution (ICC).

### Key Designs

#### Formal Foundation: MMDP-SCM

Constructing the MMDP-SCM $\langle \mathbf{V}, \mathbf{U}, P(\mathbf{u}), \mathcal{F} \rangle$ by integrating multi-agent MDPs with Structural Causal Models (SCMs):

- $\mathbf{V}$: Observed variables (all state and action variables)
- $\mathbf{U}$: Mutually independent noise variables (capturing randomness)
- $\mathcal{F}$: Structural equations, state transition $S_t := f^S(S_{t-1}, \mathbf{A}_{t-1}, U^{S_t})$, agent policy $A_{i,t} := f^{A_i}(S_t, U^{A_{i,t}})$

Each noise instance $\mathbf{u}$ uniquely determines a trajectory $\tau$, and counterfactuals are defined through interventions on structural equations.

#### Level 1: Decomposition of tot-ASE and r-SSE

**tot-ASE** is defined via "natural intervention": applying natural interventions to the actions of all subsequent agents, letting them take the actions they would naturally take in the counterfactual world, and measuring the difference in outcomes.

**r-SSE** is defined via "reversed" state-specific effects: assuming all agents have already acted counterfactually, but the state $S_{t+1}$ remains unaffected by the intervention, and measuring the "lost" effect.

The key theorem (Theorem 3.3) proves that $\text{TCFE} = \text{tot-ASE} - \text{r-SSE}$ always holds. Note: Intuitively, one might assume TCFE = tot-ASE + SSE, but the authors experimentally demonstrate that this does not always hold, and the correct decomposition requires using r-SSE instead of SSE.

#### Level 2a: ASE-SV (Shapley Value Decomposition of tot-ASE)

Utilizing the concept of Agent-Specific Effect (ASE) — where the $\mathbf{N}$-specific effect measures the portion of the intervention effect propagated solely through the subset of agents $\mathbf{N}$. This is used to construct a cooperative game, calculating each agent's contribution score with the Shapley value:

$$\phi_j = \sum_{S \subseteq \{1,...,n\} \setminus \{j\}} w_S \cdot [\text{ASE}^{S \cup \{j\}} - \text{ASE}^S]$$

**Axiomatic Guarantee** (Theorem 5.3): ASE-SV is the unique attribution method that simultaneously satisfies the following four properties:
1. **Efficiency**: The sum of all agents' contributions equals tot-ASE
2. **Invariance**: Agents that do not contribute receive a score of zero
3. **Symmetry**: Agents that contribute equally receive the same score
4. **Contribution Monotonicity**: Scores depend only on marginal contributions and are monotonic

#### Level 2b: r-SSE-ICC (Intrinsic Causal Contribution Decomposition of r-SSE)

Utilizing the concept of Intrinsic Causal Contribution (ICC) from Janzing et al. (2024), this quantifies the reduction in uncertainty of r-SSE by each state variable:

$$\text{ICC}(S_k \to \Delta Y | \tau) = \text{Unc}^{<S_k} - \text{Unc}^{\leq S_k}$$

where $\text{Unc}$ is the expectation of the conditional variance. Intuitive meaning: if knowing the counterfactual value of state $S_k$ significantly reduces the uncertainty in estimating r-SSE, then $S_k$ has a large contribution to r-SSE.

The final attribution score allocates r-SSE proportionally according to the relative ICC (with efficiency guaranteed by Theorem 4.2).

### Loss & Training

This work belongs to a **causal inference analysis framework** rather than training new models, thus involving no traditional loss functions. The core computational workflow is:

1. **Abduction**: Given an observed trajectory $\tau$, sample noise from the posterior distribution $P(\mathbf{u}|\tau)$
2. **Action**: Apply the target intervention in the MMDP-SCM
3. **Prediction**: Roll out forward to obtain counterfactual outcomes

In the experiments, 100 posterior samples are used to estimate counterfactual effects, and 20 additional samples are used for conditional variance estimation. The policy networks in the environments are trained via deep RL (for the Gridworld experiments), but this is not the core contribution of this paper.

## Key Experimental Results

### Main Results

The authors validate their method on two environments:

| Environment | Task Description | Number of Agents | Time Steps | Main Validation |
|------|---------|---------|---------|---------|
| Gridworld (LLM-assisted) | Two Actors + LLM Planner for item delivery | 3 (2 Actors + 1 Planner) | ~20 steps | Correctness of the decomposition formula, explainability of ASE-SV, precise attribution of r-SSE-ICC |
| Sepsis Simulator | Clinician + AI joint treatment of ICU patients | 2 (Clinician + AI) | 20 rounds | Plausibility of ASE-SV variation with trust, r-SSE-ICC sparsity |

**Gridworld Key Results**:
- Verified that $\text{TCFE} \neq \text{tot-ASE} + \text{SSE}$, but $\text{TCFE} = \text{tot-ASE} - \text{r-SSE}$ holds (Theorem 3.3).
- ASE-SV attributes tot-ASE entirely to Actor 2 (the subject of intervention), with Actor 1 and Planner scoring zero — matching expectations.
- r-SSE-ICC precisely identifies 4 state variables as having significant contributions, which exactly correspond to the time steps when Actor 2 crosses colored regions.

**Sepsis Key Results**:
- Screened 8,728 alternative actions with TCFE $\ge$ 0.8 from 600 failed trajectories.
- Individual clinician-specific effect + AI-specific effect $\neq$ tot-ASE, with a discrepancy of up to 95%, whereas ASE-SV consistently guarantees efficiency.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Trust level $\mu$ from low to high | Clinician ASE-SV score $\downarrow$, AI score $\uparrow$ | Higher trust $\rightarrow$ clinician overrides AI decisions less $\rightarrow$ contribution decreases |
| Trust level $\mu \rightarrow 1$ (Complete trust) | Clinician score $\rightarrow 0$, AI score $\rightarrow$ All | AI bears full responsibility when in complete control |
| Round difference 4-10 | Gini coefficient concentrated between 0.6-0.9 | r-SSE-ICC only attributes to a few key states, demonstrating sparsity |
| Noise monotonicity hypothesis testing | Robust decomposition results | High robustness against violations of the hypothesis |
| Estimation error analysis | Stable estimation achieved with 100 samples | Standard errors are within a reasonable range |

### Key Findings

1. **Rigor of the Decomposition Formula**: The intuitive "$\text{TCFE} = \text{tot-ASE} + \text{SSE}$" does not always hold (verified by counterexamples in Gridworld); the correct formula is $\text{TCFE} = \text{tot-ASE} - \text{r-SSE}$.
2. **Meaningful Zeroing Mechanism of ASE-SV**: Non-contributing agents are scored zero for two distinct reasons — (a) lack of response to intervention (e.g., Actor 1), or (b) responsive but unable to affect the state (e.g., Planner).
3. **Sparse Attribution of r-SSE-ICC**: In Sepsis, regardless of the trajectory length, the Gini coefficient exceeds 0.6 for most trajectories, meaning only a few critical state variables dominate the r-SSE.
4. **Intuitive Responsibility Allocation in Sepsis**: The clinician bears approximately 73.5% of the direct state-related responsibility under a certain trust level.

## Highlights & Insights

1. **Ingenious Framework Design**: The Level 1 decomposition separates effects according to the two core components of MMDPs (agent behavior vs. environmental state), possessing clearer operational meaning than traditional path mediation analysis.
2. **Axiomatic Guarantee**: The uniqueness theorem of ASE-SV provides a solid theoretical foundation for the attribution method, free from heuristics.
3. **r-SSE Instead of SSE**: The discovery that "reversed" state-specific effects are required for correct decomposition is seemingly counterintuitive but backed by rigorous mathematical guarantees.
4. **LLM + RL Experimental Design**: Adopting the LLM Planner + RL Actor architecture in Gridworld demonstrates the applicability of the framework in modern AI systems.
5. **Practical Value of Sparse Attribution**: The sparsity of r-SSE-ICC implies that in practice, one only needs to infer the counterfactual values of a few key states to accurately estimate the effect.

## Limitations & Future Work

1. **Computational Complexity**: ASE-SV requires enumerating agent subsets ($2^n$ complexity), and r-SSE-ICC requires calculating conditional variance state-by-state, leading to a significant overhead when the number of agents is large or the time step is long.
2. **Noise Monotonicity Assumption**: Although experiments demonstrate robustness against violations of the assumption, this assumption may fail in practice, and a partial identification version is currently lacking.
3. **Limited Experimental Scale**: The number of agents is $\le 3$ and the time steps are $\le 20$ in both experimental settings, lacking validation in large-scale multi-agent scenarios.
4. **Assumed Access to Causal Models**: The framework assumes access to the structural causal model of the environment, whereas in reality, models typically need to be learned from data.
5. **Focus on Discrete/Categorical MMDP-SCMs**: Extensions to continuous state spaces have not yet been addressed.

## Related Work & Insights

- **Causal Mediation Analysis** (Pearl 2001, VanderWeele 2016): This work can be viewed as generalizing classical mediation formulas to the structure of multi-agent MDPs.
- **Agent-Specific Effects** (Triantafyllou et al. 2024): This work directly builds upon this concept to further achieve complete effect decomposition.
- **Application of Shapley Values in Causal Inference** (Janzing et al. 2024, Heskes et al. 2020): Applying the game-theoretic concept of fair division to causal attribution.
- **Counterfactuals in Explainable AI**: This work provides richer explanations than a single TCFE for counterfactual questions such as "what if the agent had acted otherwise."
- **Insights**: The framework can be directly applied to behavioral auditing in LLM Agent systems — analyzing blame allocation for decision failures during collaborative execution among multiple LLM Agents.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ - First to systematically decompose counterfactual effects in multi-agent MDPs, exhibiting strong theoretical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ - Explainability of the method is verified in two environments, but the scale and diversity are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ - The paper is well-structured; definitions, theorems, and experiments are tightly interconnected, with examples integrated throughout.
- Value: ⭐⭐⭐⭐ - Highly significant from a theoretical perspective for multi-agent responsibility attribution, though practical applications require further refinement.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bayesian Ensemble for Sequential Decision-Making](../../ICLR2026/reinforcement_learning/bayesian_ensemble_for_sequential_decision-making.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](../../ICML2026/reinforcement_learning/multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)

</div>

<!-- RELATED:END -->
