---
title: >-
  [Paper Note] Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs
description: >-
  [NeurIPS 2025][Reinforcement Learning][retrosynthesis planning] This work reformulates retrosynthesis planning as a worst-path optimisation problem in tree-structured MDPs — the value of a synthesis tree is determined by…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "retrosynthesis planning"
  - "tree-structured MDP"
  - "worst-path optimisation"
  - "self-imitation learning"
  - "search-free inference"
date: 2026-05-08
content_hash: 5563972fd6dce65b
---

# Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs

**Conference**: NeurIPS 2025
**arXiv**: [2509.10504](https://arxiv.org/abs/2509.10504)  
**Code**: [GitHub](https://github.com/MianchuWang/InterRetro)  
**Area**: Reinforcement Learning / Retrosynthesis Planning
**Keywords**: retrosynthesis planning, tree-structured MDP, worst-path optimisation, self-imitation learning, search-free inference

## TL;DR

This work reformulates retrosynthesis planning as a worst-path optimisation problem in tree-structured MDPs — the value of a synthesis tree is determined by its weakest path, since any dead-end path renders the entire tree invalid. The proposed method, InterRetro, optimises this worst-path objective via weighted self-imitation learning, achieving 100% success rate on Retro*-190, reducing path length by 4.9%, and attaining 92% of full performance with only 10% of training data.

## Background & Motivation

**Background**: Retrosynthesis planning aims to decompose a target molecule step-by-step into purchasable building blocks, forming a synthesis tree. Single-step prediction accuracy has reached human-level performance, yet multi-step planning still relies on heuristic search methods (e.g., MCTS, A*) requiring hundreds of model calls, resulting in substantial computational cost.

**Limitations of Prior Work**: (1) Search-based methods (MCTS/Retro*) demand intensive real-time computation, with hundreds of model calls per molecule, limiting large-scale applicability. (2) Fine-tuning methods (e.g., PDVN) improve policies by imitating successful search trajectories, but the model adapts to the molecular distribution encountered during search rather than the direct inference distribution, leading to performance degradation without search. (3) Most critically, existing methods typically optimise the **average** performance across all paths, neglecting the **worst-case sensitivity** of synthesis trees: a single path that fails to reach purchasable building blocks invalidates the entire synthesis tree.

**Key Challenge**: Successful synthesis requires **all** leaf nodes to be purchasable compounds, yet existing optimisation objectives focus on average performance rather than the worst path.

**Goal**: Eliminate the need for search at inference time while guaranteeing high-quality synthesis routes — requiring both a more appropriate optimisation objective and a search-free policy improvement method.

**Key Insight**: Address the optimisation objective directly — model the "weakest link" as the worst path, replacing the average cumulative return with a discounted worst-path return.

**Core Idea**: A synthesis route is only as strong as its weakest path — worst-path optimisation directly improves the most failure-prone decomposition steps, coupled with self-imitation learning to enable search-free inference.

## Method

### Overall Architecture

InterRetro operates as an agent interacting with a tree-structured MDP: (1) **Exploration** — recursively decompose the target molecule using the current policy to construct synthesis trees; (2) **Extraction** — identify successful subtrees within the synthesis tree and collect all branching decisions; (3) **Learning** — update the policy and value network via weighted self-imitation learning. This process iterates until policy convergence.

### Key Designs

1. **Worst-Path Objective**:
    - Function: Defines the value of a synthesis tree as the worst return among all root-to-leaf paths.
    - Mechanism: Reward function $r(s) = 1$ if molecule $s$ is a purchasable building block, otherwise 0. Path return is $\gamma^T r(s_T)$, where discount factor $\gamma \in (0,1)$ penalises longer paths. The synthesis tree objective is $J(\pi) = \mathbb{E}_{\tau \sim \pi}[\min_{p \in P(\tau)} \sum_{t=0}^T \gamma^t r(s_t)]$. A single failing path reduces the entire tree's value to zero.
    - Design Motivation: 98.6% of reactions in USPTO-50k involve at most 3 reactants, so synthesis tree quality is primarily determined by depth (worst-path length) rather than breadth.

2. **Bellman Optimality Equations in Tree MDPs**:
    - Function: Derive the recursive relationship and optimality conditions for the value function under the worst-path objective.
    - Mechanism: Q-function recursion $Q^\pi(s,a) = r(s) + \gamma(1-r(s))\min_{s' \in \mathcal{T}(s,a)} V^\pi(s')$ — the key distinction is the use of $\min$ rather than $\sum$ to aggregate child nodes. The Bellman optimality operator is proven to be a contraction mapping, guaranteeing unique existence of $V^*$ and convergence of value iteration.
    - Design Motivation: Standard MDPs aggregate via summation or expectation, but chemical reactions produce multiple child states (molecules), necessitating $\min$ to capture the "weakest-link" effect.

3. **Weighted Self-Imitation Learning**:
    - Function: Imitate past successful decisions using advantage-based weights, ensuring chemical feasibility.
    - Mechanism: Policy is constrained to the support of the pretrained single-step model $\pi^0$: $\Pi = \{\pi \mid \pi(a|s)=0 \text{ whenever } \pi^0(a|s)=0\}$. The update objective is $\mathcal{L}(\theta) = -\mathbb{E}[\exp_{clip}(\beta A_\phi(s,a))\log\pi_\theta(a|s)]$, assigning higher weights to reactions with higher advantage. Theoretical proof guarantees $V^{\pi^{i+1}}(s) \geq V^{\pi^i}(s)$ (monotonic improvement).
    - Design Motivation: Support constraint ensures all proposed reactions remain chemically valid; advantage weighting avoids blind imitation and focuses learning on high-quality decisions.

### Loss & Training

Value network loss (Bellman TD): $\mathcal{L}(\phi) = \mathbb{E}[(V_\phi(s) - (r(s) + \gamma(1-r(s))\min_{s'} V_{\phi^-}(s')))^2]$, where $V_{\phi^-}$ denotes the target network. Policy network loss: $\mathcal{L}(\theta) = -\mathbb{E}[\exp_{clip}(\beta A_\phi(s,a))\log\pi_\theta(a|s)]$. Training uses a FIFO experience replay buffer (capacity 20K branches), 6 parallel exploration processes, collecting 36 synthesis trees per iteration before performing 5 gradient updates.

## Key Experimental Results

### Main Results

| Benchmark | Model Calls | InterRetro | PDVN | Retro* | MCTS |
|-----------|------------|------------|------|--------|------|
| Retro*-190 | 500 | **100.0%** | 98.95% | 75.26% | 62.63% |
| Retro*-190 | Direct Gen. | **95.78%** | - | 20.00% | 20.00% |
| ChEMBL-1000 | 500 | **97.50%** | 83.50% | 74.70% | 71.90% |
| GDB17-1000 | 500 | **99.50%** | 26.90% | 7.50% | 4.50% |

### Ablation Study

| Configuration | Retro*-190 Success Rate | Notes |
|---------------|------------------------|-------|
| Worst-path objective (default) | 100.0% | Ours |
| Average-path objective | ~96% | Ignores weakest link |
| No self-imitation (pretrained only) | 16.84% | Direct inference from pretrained policy |
| 10% training data | ~92% | Strong sample efficiency |
| Full training data | 100.0% | Full convergence |

### Key Findings
- **100% success rate**: First method to achieve perfect success on Retro*-190, with path lengths 4.9% shorter than search-based methods.
- **Search-free inference**: Direct Generation mode achieves 95.78% (vs. 20% for search-based baselines).
- **Exceptional sample efficiency**: 92% performance attained with only 10% of training data.
- **Dominant advantage on GDB17-1000**: InterRetro 99.5% vs. PDVN 26.9% — particularly pronounced on challenging molecules.

## Highlights & Insights
- The worst-path optimisation formulation is a highly natural model of the problem — the "weakest-link" effect is indeed the central challenge in retrosynthesis.
- Theoretical contributions are rigorous: existence and uniqueness of the optimal solution, monotonic improvement guarantees, and complete derivation of Bellman optimality equations.
- Practical significance is substantial: eliminating search at inference time reduces retrosynthesis from minutes to milliseconds.
- Attaining 92% performance with 10% of data suggests the worst-path objective leverages experience more efficiently than alternative objectives.

## Limitations & Future Work
- Performance is bounded by the quality of the pretrained single-step model (Graph2Edits) — correct reactions outside the pretrained model's support cannot be recovered.
- Training requires approximately 48 hours (single A5000 GPU), which is acceptable but leaves room for optimisation.
- Evaluation is limited to the USPTO-50k training set; real-world reactions may exceed the coverage of this dataset.
- The tree MDP formulation assumes deterministic reaction recommendations, whereas real chemical reactions involve uncertainty (side reactions, yields).

## Related Work & Insights
- **vs. PDVN**: PDVN also improves search policies via self-imitation but optimises average return and still relies on search at inference; InterRetro optimises the worst path and is entirely search-free.
- **vs. MCTS/Retro***: Search-based methods require extensive model calls (500), whereas InterRetro achieves higher success rates via direct generation.
- **vs. DreamRetroer**: Comparable performance, but InterRetro does not depend on any external model augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The worst-path optimisation framework offers an entirely new perspective for retrosynthesis, with complete and elegant theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks, multiple model call budgets, thorough ablations, and the first method to achieve 100% success rate.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear; theory and algorithm are presented in a coherent and well-structured manner.
- Value: ⭐⭐⭐⭐⭐ — Search-free inference combined with 100% success rate represents significant practical value for computer-aided molecular synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](bandit_and_delayed_feedback_in_online_structured_prediction.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] The Path Not Taken: RLVR Provably Learns Off the Principals](the_path_not_taken_rlvr_provably_learns_off_the_principals.md)

</div>

<!-- RELATED:END -->
