---
title: >-
  [Paper Note] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs
description: >-
  [AAAI 2026][Reinforcement Learning][Neurosymbolic AI] This paper proposes DeepProofLog (DPrL), a neurosymbolic system grounded in stochastic logic programs that introduces neural network parameterization at each proof step and establishes a formal mapping between SLD resolution and MDPs. This enables dynamic programming and reinforcement learning techniques to be applied for efficient inference and learning, substantially improving the scalability of neurosymbolic systems.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Neurosymbolic AI
  - Stochastic Logic Programs
  - Markov Decision Processes
  - Dynamic Programming
  - Policy Gradient
date: 2026-05-08
content_hash: 6deaec851b581ec8
---

# DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs

**Conference**: AAAI 2026
**arXiv**: [2511.08581](https://arxiv.org/abs/2511.08581)
**Code**: [DeepProofLog/DPrL-AAAI](https://github.com/DeepProofLog/DPrL-AAAI)
**Area**: Reinforcement Learning
**Keywords**: Neurosymbolic AI, Stochastic Logic Programs, Markov Decision Processes, Dynamic Programming, Policy Gradient

## TL;DR

This paper proposes DeepProofLog (DPrL), a neurosymbolic system grounded in stochastic logic programs that introduces neural network parameterization at each proof step and establishes a formal mapping between SLD resolution and MDPs. This enables dynamic programming and reinforcement learning techniques to be applied for efficient inference and learning, substantially improving the scalability of neurosymbolic systems.

## Background & Motivation

Neurosymbolic AI (NeSy) integrates neural networks with symbolic reasoning to improve accuracy, interpretability, and generalization. Mainstream NeSy systems typically perform inference in two stages: (1) symbolic proof search to enumerate all proofs of a query, and (2) probabilistic inference over those proofs to compute the probability that the query holds. However, the latter corresponds to weighted model counting, a #P-hard problem.

Two existing classes of acceleration approaches both have limitations:

**Sampling/variational methods** (e.g., Scallop, A-NeSI): approximate the probabilistic inference stage but do not address the computational cost of proof search.

**Task-specific solutions**: require substantial manual engineering and lack generality.

More fundamentally, existing approaches do not address the computational cost of **finding proofs themselves**, rendering them infeasible for large-scale relational reasoning.

Furthermore, two existing stochastic logic program frameworks have expressiveness limitations:
- **SLP** (standard stochastic logic programs): clause probabilities are fixed scalars, independent of query context.
- **DeepStochLog**: conditions clause probabilities on neural networks but requires a fixed output domain and conditions only on partial goals.

## Method

### Overall Architecture

The core shift in DPrL is to **parameterize the resolution process rather than the logic program itself** (e.g., clause weights). Concretely, at each step of SLD resolution, a neural network computes transition probabilities—given the current goal $G_i$, it scores all possible next goals $G_{i+1}$:

$$p_\lambda(G_{i+1} | G_i) = \frac{\exp(f_\lambda(G_i, G_{i+1}))}{\sum_{G \in \mathcal{N}(G_i)} \exp(f_\lambda(G_i, G))}$$

where $\mathcal{N}(G)$ is the set of all possible next goals after resolving the leftmost atom, and $f_\lambda$ computes a compatibility score between embeddings of the current and candidate goals.

The **formal mapping to an MDP** is the central theoretical contribution of the paper:

- **States** $\mathcal{S}$: initial query ∪ intermediate subgoals ∪ {True, False}
- **Actions** $\mathcal{A}(G)$: all applicable (clause, unifier) pairs
- **Transition function**: deterministic, $P(G'|G_i, C, \theta) = \mathbf{1}\{G' = \text{res}(G_i, C, \theta)\}$
- **Reward**: given only at terminal states; +1 if a positive query proof succeeds, −1 if a negative query proof succeeds
- **Policy**: $\pi_\lambda(a|G_i) = p_\lambda(G_{i+1}|G_i)$, i.e., the resolution transition model

### Key Designs

**1. Hierarchical Goal Representation**

To make symbolic goals amenable to neural processing, a three-level embedding hierarchy is constructed:

- **Sub-symbolic embeddings**: all entities (predicates, constants, variables, images, etc.) share learnable embeddings $\mathbf{e}_v \in \mathbb{R}^d$
- **Atom embeddings**: $\mathbf{e}_A = f_{\text{atom}}(\mathbf{e}_r, \mathbf{e}_{t_1}, \ldots, \mathbf{e}_{t_n})$, computed by any neural network or knowledge graph embedding model
- **Goal embeddings**: $\mathbf{e}_G = f_{\text{agg}}(\mathbf{e}_{A_1}, \ldots, \mathbf{e}_{A_n})$, aggregating atom embeddings
- **Compatibility score**: $f_\lambda(G, G') = \mathbf{e}_G \cdot \mathbf{e}_{G'}$ (inner product)

This design conditions the policy on the **complete current goal** rather than only a partial subgoal, enabling more informed decisions at each step.

**2. Exact Inference — Dynamic Programming**

When the symbolic goal space is manageable, DP is used to exactly solve the MDP value function:

$$V^{\pi_\lambda}(G, y) = \sum_{a \in \mathcal{A}(G)} \pi_\lambda(\text{res}(G,a), G) \cdot V^{\pi_\lambda}(\text{res}(G,a), y)$$

with boundary condition $V^{\pi_\lambda}(G,y) = R(G,y)$ for terminal goals.

A key theorem establishes that maximizing the MDP's expected return $J(\pi_\lambda)$ is equivalent to minimizing the standard NeSy learning objective $\mathcal{L}(\lambda)$:

$$J(\pi_\lambda) = \sum_{i=1}^N (2y^{(i)} - 1) \cdot p_{\text{success}}^{\pi_\lambda}(q^{(i)})$$

**3. Approximate Inference — Policy Gradient**

When the goal space is not enumerable, PPO (Proximal Policy Optimization) is used for approximate inference:
- The neural policy is optimized using the standard clipped surrogate objective.
- A value network of the same architecture estimates returns and reduces variance.
- The structure of the logic program naturally constrains the valid action set, reducing sampling variance.

**4. Universal Approximation (Proposition 1)**

DPrL is shown to approximate any probability distribution over SLD derivations to arbitrary precision, establishing the model's expressive power.

### Loss & Training

The learning objective is:

$$\mathcal{L}(\lambda) = \sum_{(q^{(i)}, y^{(i)}) \in \mathcal{D}} \ell(p_{\text{success}}^\lambda(q^{(i)}), y^{(i)})$$

where $\ell(p, y) = (1 - 2y) \cdot p$ is a linear loss that maximizes proof success probability for positive queries and minimizes it for negative ones.

In DP mode, the value function is optimized directly via automatic differentiation. In PG mode, PPO with a value function baseline is used.

## Key Experimental Results

### MNIST Addition (Scalability Test)

The task is to predict the sum of two $N$-digit MNIST digit sequences given only the final sum as supervision.

| N | Reference | DeepStochLog | DeepSoftLog | **DPrL(DP)** | Scallop | A-NeSI | **DPrL(PG)** |
|---|-----------|-------------|-------------|-------------|---------|--------|-------------|
| 4 | 92.9% | 92.7% | 93.5% | **94.0%** | 92.3% | 92.6% | 93.5% |
| 15 | 75.8% | T/O | 77.1% | **80.8%** | T/O | 75.9% | 76.9% |
| 100 | 15.4% | T/O | T/O | **37.8%** | T/O | T/O | 0.0% |
| 200 | 2.4% | T/O | T/O | **23.2%** | T/O | T/O | 0.0% |
| 500 | 0.009% | T/O | T/O | **6.0%** | T/O | T/O | 0.0% |

DPrL(DP) is the only method that scales to $N \geq 100$. In terms of training time, DPrL requires only 25 minutes at $N=4$, compared to 141 minutes for DeepStochLog.

### Knowledge Graph Completion

| Method | Family MRR | Family H@1 | WN18RR MRR | WN18RR H@1 |
|--------|-----------|-----------|-----------|-----------|
| ComplEx | 0.964 | 0.938 | 0.479 | 0.438 |
| RotatE | 0.968 | 0.943 | 0.833 | 0.787 |
| R2N_SG | 0.985 | 0.981 | 0.724 | 0.699 |
| DCR_SG | 0.975 | 0.956 | 0.817 | 0.754 |
| SBR_SG | 0.981 | 0.966 | 0.840 | 0.784 |
| **DPrL (PG)** | **0.986** | **0.979** | 0.834 | 0.784 |

DPrL achieves the best MRR and H@1 on Family, and matches the best-performing method (SBR_SG) on WN18RR.

### Ablation Study

- DP vs. PG: DP is more accurate on small goal spaces (94.0% vs. 93.5% at $N=4$), while PG offers greater computational flexibility.
- On the Family test set: naive grounding requires instantiating 18,071 / 45,466 / 181,095 groundings at depth 1/2/3, whereas DPrL requires only 12,334 ground evaluations.

### Key Findings

1. DPrL is the only NeSy system capable of scaling MNIST addition to $N \geq 100$.
2. On KG completion, DPrL produces fully interpretable proof trees (e.g., the *uncle* relation is derived via *father* → *brother*).
3. Unlike grounding-based methods that require manual hyperparameter tuning, DPrL automatically learns an optimal exploration strategy.
4. The MDP mapping allows flexible selection between exact (DP) and approximate (PG) inference depending on the size of the goal space.

## Highlights & Insights

1. **Theoretical bridge from SLP to MDP**: The mapping is not merely conceptual but mathematically rigorous—the policy exactly corresponds to resolution transition probabilities, and value function optimization is provably equivalent to the standard NeSy objective. This unlocks the full RL toolbox (DP, PPO, etc.) for symbolic reasoning.
2. **Advantages of goal-level parameterization**: Conditioning on the complete goal rather than individual atoms or terms provides richer context for decision-making at each step.
3. **Interpretable inference**: Each prediction corresponds to a complete proof tree, offering substantially greater transparency than the black-box outputs of other neural methods.
4. **Unexpected efficiency of DP**: Many NeSy benchmarks that appear combinatorially explosive in fact yield manageable state spaces after encoding, making exact DP both efficient and precise.

## Limitations & Future Work

1. Clause selection is restricted to the leftmost atom—while this ensures completeness, it may be suboptimal; more flexible selection strategies could prune failing derivations more quickly.
2. In PG mode, accuracy on MNIST addition with $N \geq 100$ drops to 0%—policy gradient methods fail to escape suboptimal solutions in extremely large combinatorial spaces.
3. The current framework uses standard policy gradient; more advanced RL techniques (e.g., value-based methods) could be explored.
4. Automatic restructuring of logic programs to reduce the search space has not been investigated.

## Related Work & Insights

- **DeepStochLog**: The closest predecessor; DPrL advances beyond it along three dimensions: (1) conditioning on the complete current goal, (2) supporting dynamic output domains, and (3) achieving superior scalability.
- **MINERVA/DeepPath**: RL-based path search on knowledge graphs, but restricted to KG-specific structures; DPrL, grounded in SLPs, is applicable to any domain expressible in logic.
- **Automated Theorem Proving (ATP)**: ATP seeks any valid proof, whereas NeSy requires probabilistically weighting multiple proofs and aligning them with sub-symbolic signals—a fundamentally different objective.
- Conceptual takeaway: **mapping structured reasoning processes to MDPs** is a general paradigm extensible to other formal reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The SLP→MDP mapping is a highly original theoretical contribution)
- Technical Depth: ⭐⭐⭐⭐⭐ (Rigorous formal definitions, universal approximation theorem, and equivalence proofs)
- Experimental Thoroughness: ⭐⭐⭐⭐ (MNIST addition scalability tests and KG completion cover diverse scenarios)
- Writing Quality: ⭐⭐⭐⭐ (Logical structure, intuitive examples)
- Value: ⭐⭐⭐⭐ (Open-source code; interpretable KG reasoning has practical significance)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](deep_predictive_discounted_counterfactual_regret_minimization.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[AAAI 2026\] Efficient Multiagent Planning via Shared Action Suggestions](efficient_multiagent_planning_via_shared_action_suggestions.md)

<!-- RELATED:END -->
