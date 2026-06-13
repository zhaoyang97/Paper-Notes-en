---
title: >-
  [Paper Note] Learning to Search and Searching to Learn for Generalization in Planning
description: >-
  [ICML 2026][Reinforcement Learning][Generalized Planning] This paper proposes GSP: a "self-improving generalized planner" that wraps Weighted A* best-first search and Q-learning in the same loop…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Generalized Planning"
  - "WA* Search"
  - "Q-learning"
  - "R-GNN"
  - "Zero-shot Size Generalization"
date: 2026-05-08
content_hash: 3c270e18bc7a5bc0
---

# Learning to Search and Searching to Learn for Generalization in Planning

**Conference**: ICML 2026  
**arXiv**: [2605.25720](https://arxiv.org/abs/2605.25720)  
**Code**: https://github.com/maichmueller/generalized-search-for-planning  
**Area**: Reinforcement Learning / Classical Planning / Relational GNNs  
**Keywords**: Generalized Planning, WA* Search, Q-learning, R-GNN, Zero-shot Size Generalization  

## TL;DR
This paper proposes GSP: a "self-improving generalized planner" that wraps Weighted A* best-first search and Q-learning in the same loop, using Relational Graph Neural Networks (R-GNN) to represent $Q_\theta(s,a)$. By training only on small-scale instances, it achieves zero-shot generalization to instances over ten times larger than those seen during training (e.g., from $\leq 30$ blocks to 488 blocks in Blocksworld). It simultaneously sets new coverage records on multiple IPC benchmarks, Sokoban, PushWorld, and The Witness, outperforming DRL baselines based on real-time search.

## Background & Motivation
**Background**: Classical planning (PDDL) naturally provides a "domain-instance" separation: the domain defines fixed predicates and action schemas, while instances vary in initial values, goals, and the number of objects. This clean structure makes it an ideal laboratory for studying "combinatorial generalization." Recent works like Lifted HER, WL-features, and Distincter have attempted to generate "generalized policies" or "generalized heuristics" through DRL/supervised learning, enabling a single learned network to solve instances of arbitrary size.

**Limitations of Prior Work**: (1) DRL approaches (Rivlin 2020, Ståhlberg 2023a/2026, etc.) follow real-time (agent-based) search, which is extremely inefficient for exploration in sparse-reward, dead-end-prone, or long-horizon problems (e.g., Sokoban, Floortile); (2) Supervised approaches (WL-f, Horčik, Distincter) use optimal planners to pre-generate $h^\ast(s)$ for supervised regression, skipping the self-improving loop where "better search leads to better training samples leads to stronger heuristics"; (3) Hindsight Experience Replay (HER) is effective in domains with decomposable subgoals, but many planning goals cannot be meaningfully decomposed, rendering HER insufficient for real-time search.

**Key Challenge**: In classical planning, the model is fully known, which suggests that best-first searches like A*/WA*/GBFS should be utilized. However, DRL often forces real-time search due to the inertia of the RL framework, effectively discarding powerful tools. Furthermore, while the feedback loop of "better heuristics lead to faster search, and faster search leads to superior samples" has existed in LRTA*/RTDP, it has traditionally been tied to a single state space rather than generalized across a "family of instances."

**Goal**: Construct a self-improving search-learning loop where an untrained $Q_\theta$ learns generalized heuristics from WA* search data on small instances that can directly generalize across scales. At test time, the model can be used greedily as a policy or to accelerate search using the same WA*.

**Key Insight**: Since the model is known, best-first search is treated as the "explorer" and R-GNN as the "cross-scale transferer." R-GNN performs message passing over the set of objects, naturally supporting the transition from 29 blocks in training to 488 blocks in testing.

**Core Idea**: Replace real-time search with WA* for exploration and use Q-learning combined with search-derived "return lower bounds" for supervision. Build the entire loop on a schema-shared R-GNN to obtain a $Q_\theta(s,a)$ capable of generalizing across instances.

## Method

### Overall Architecture
GSP maintains a $Q_\theta(s,a)$ parameterized by an R-GNN and a replay buffer $\mathcal D$. In each iteration: (1) Sample an instance $\mathcal E$ from the training pool; (2) Run WA* search using $f(s,a)=g(s)+w\,Q_\theta(s,a)$ ($w=2$), where nodes are state-action pairs $(s,a)$; (3) Store dead-ends, samples on the goal path, and search-derived return lower bounds $\underline R$ into the buffer; (4) Asynchronously sample mini-batches from the buffer to run Q-learning (with a target network) on $Q_\theta$; (5) Dynamically schedule instances across three pools: "unsolved / solved (nodes = optimal length) / satisficed (solved but non-optimal)," with exponentially increasing weights to ensure most computational effort is spent on medium-difficulty instances that are "just solvable but not yet optimal"—the primary source of learning signals.

### Key Designs

1. **WA* Exploration + Search-derived Return Lower Bound**:
    - **Function**: Reintroduces the common sense of using best-first search in model-known scenarios into RL and uses the actual solution length found by best-first search as a lower-bound supervision for $Q$.
    - **Mechanism**: Rewards use unit step costs $r=-1$ without discounting; the node cumulative return $g(s)$ represents the negative depth. The frontier expands based on the maximum $f(s,a)=g(s)+w\,Q_\theta(s,a)$. Three types of transitions are distinguished: dead-ends receive a fixed penalty $R_\bot$, goal paths are backtracked to label each $(s_t,a_t)$ with the actual return-to-go $\underline R$ as a lower bound for the optimal return, and non-terminal states follow standard Q-learning bootstrapping. The critical training target is $y=\max\{\underline R,\,\hat y(s,a)\}$, where $\hat y(s,a)=-1+\max_{a'} Q_\theta(s',a')$. This prevents bootstrapping from pulling the target lower than the solution already found by search, serving as a vital stabilizer.
    - **Design Motivation**: Real-time search has poor sample efficiency under sparse rewards. Since the model is known, WA* is allowed to complete an entire solution path to explicitly feed the "successful traversal" information back into $Q$.

2. **Schema-shared R-GNN with Action Atoms for $Q_\theta$**:
    - **Function**: Enables a single readout to work across different object counts and action schemas, supporting zero-shot cross-scale transfer.
    - **Mechanism**: Each state-goal pair is encoded as a set of relations $\mathcal R_{s,g}=\{p(\bar o)\in s\cup g\}$. For each applicable grounded action $a=A(\bar o)$, a dedicated action object $o_a$ and atom $A(o_a,\bar o)$ are introduced into the message-passing graph. Each predicate $p$ has its own $\mathrm{Comb}_p$ MLP to encode positional roles into messages. Objects aggregate messages via smoothmax, and updates use a shared $\mathrm{Comb}_U$ with residuals. After $L$ layers, a single schema-shared MLP predicts $Q_\theta(s,a)$ based on $[X_L(o_a)\,\|\,\bar X(s,g)]$. Since $\mathrm{MLP}_Q$ is shared across all action schemas, differences in action types are expressed only through message passing and $X_L(o_a)$. The model learns "scoring principles" rather than schema-specific scorers, which is why it can solve Blocksworld with 488 blocks.
    - **Design Motivation**: Traditional DRL/MLPs hardcode the action space into the output layer, which breaks when instances change. R-GNN + action atoms maintain permutation equivariance at the object set level, naturally supporting varying object counts.

3. **Three-pool Instance Scheduling (Unsolved / Solved / Satisficed)**:
    - **Function**: Automatically allocates training computation to the "most informative instances."
    - **Mechanism**: Each training instance is categorized based on the latest WA* result: satisficed (solution found but nodes expanded > solution length), solved (solution found with nodes = length, indicating high heuristic confidence), or unsolved. Sampling weights increase exponentially from satisficed $\to$ unsolved $\to$ solved. The intent is to prioritize training on instances where the "heuristic can find a solution but is not yet direct"—the contrast between sub-optimal and optimal paths provides the best fuel for improving $Q$.
    - **Design Motivation**: Uniform sampling wastes time on "perfectly solved instances" (zero gradient) or "completely unsolvable instances" (zero signal).

### Loss & Training
All experiments share a single set of hyperparameters: embedding dimension $d=32$, smoothmax aggregation; R-GNN learning rate $10^{-4}$, readout $10^{-3}$; 1 learner + 5 parallel search workers with a 60-second budget per search and worker batch of 256; FIFO replay buffer capacity of 40 batches; target network updates every 10 full passes of the replay buffer (Mnih et al. 2015 style). Training duration is 12 hours, with training dynamic curves reported for the first 180 minutes.

## Key Experimental Results

### Main Results (2023 IPC Learning Track Coverage and Expanded Nodes)
| Domain (Train $\to$ Test Scale) | GSPπ (greedy) | GSP$_{\mathrm{WA^*}}$ | Lifted HER | LAMA | Remarks |
|--------|------|------|------|------|------|
| Blocksworld (29 $\to$ 488) | Strong (cross-scale holds) | Further Improvement | Weak | Domain-independent baseline | Trained on $\leq 30$ blocks, tested on 488 |
| Transport (34 $\to$ 453) | Strong | Strong | Weaker | Medium | Suitable for search heuristics |
| Sokoban (11×11, b=3 $\to$ 99×99, b=79) | Still solves many | Significantly > greedy | Degenerated | Limited | Classic PSPACE-hard problem |
| Spanner (28 $\to$ 833) | High Coverage | Near 100% | Medium | Medium | Classic schema generalization benchmark |
| Childsnack (51 $\to$ 1326) | High | High | Weak | Limited | Hardest instance has 4.67e7 applicable actions |
| Satellite/Miconic/Ferry/Rovers | Generally Leading | Generally Leading | Close in some domains | Medium | Learned heuristics take over search |
| Floortile | Average | Average | Weak | Average | Training loop failed to solve all train instances (limitation) |

> Note: The table entries correspond to qualitative summaries of the "Cov./Steps" columns in the original paper; GSP achieves coverage equal to or better than all baselines in both GSPπ and GSP$_{\mathrm{WA^*}}$ modes and remains competitive in plan length.

### Ablation Study
| Comparison | Result | Explanation |
|------|------|------|
| GSP$_{\mathrm{WA^*}}$ vs. GSPπ | WA* mode significantly stronger on hard problems like Sokoban | Learned $Q$ serves as both a policy and a heuristic |
| GSP vs. AlphaZero ($\alpha_0$, same R-GNN) | GSP globally stronger | MCTS is disadvantaged in single-goal pathfinding/sparse-reward puzzles |
| GSP vs. Lifted HER | Significantly leading in most cross-domain settings | Best-first exploration + return lower bound > real-time search + HER |
| GSP vs. WL-f / Horčik / Distincter | Competitive or better without requiring pre-generated $h^\ast$ | Self-improving vs. Supervised |
| Training Curves: Blocks/Transport/Satellite vs. Floortile | Former shows nodes $\downarrow$ and solve rate $\uparrow$; Floortile stuck with high expansion | Validates "search $\to$ learn $\to$ smaller search" loop/exposes failure cases |
| Puzzle Domains at Test: PushWorld / The Witness | GSP still solvable, DRL baselines fail largely | Cross-domain transferable |

### Key Findings
- **Zero-shot size transfer** is the most striking result: Blocksworld was only trained on instances with $\leq 30$ blocks but solved 488-block instances without search (greedy policy); Sokoban was trained on 11×11 and tested on 99×99.
- **$y=\max\{\underline R,\hat y\}$ is the stabilizer**: Using the search-found solution length as a hard lower bound for bootstrapping avoids the underestimation pitfalls of Q-learning under large-scale sparse rewards.
- **R-GNN + action atoms** enable a single readout to handle all action schemas, which is key to generalizing to unknown scales. Switching to MLP/AR output heads would almost inevitably require retraining for scaled instances.
- **MCTS loses to WA* in generalized planning**: The AlphaZero style is strong in two-player zero-sum games or dense rewards, but for single-goal pathfinding and sparse-reward puzzles, MCTS local simulation lacks the directionality of global best-first search.
- **Failure Case: Floortile**: The training loop failed to resolve all training instances, and node expansions remained high. This indicates that if initial search cannot find a signal within 12 hours, the self-improving loop cannot start.

## Highlights & Insights
- **Re-partitioning the roles of RL and search**: When the model is known, real-time search should not be used for exploration. Treating WA* as the explorer and Q-learning as the estimator is a much cleaner paradigm than "RL treating the model as a black box." This principle can generalize to any model-based RL setting.
- **Search-derived lower bounds are nearly free labels**: Every solved path provides $\underline R$ for every $(s,a)$ on that path. This turns "successful search" directly into supervised data, a fundamental reason for the strong heuristics.
- **Three-pool instance scheduling** is a valuable engineering trick: Stratifying training samples by "difficulty-informativeness" provides a free speed-up for any curriculum-light training setup.
- **R-GNN + action atoms** provide a universal framework for "object permutation equivariance + schema sharing." Any domain with clear relational structures (chemical reactions, ECS games, combinatorial optimization) can adopt this: using a class of objects to occupy the "action slot," allowing a single readout to evaluate heterogeneous actions.
- **Retaining dual usage at test time**: The same $Q_\theta$ works as a greedy policy (no search) and a WA* heuristic (with search), providing a "pair of tools" from one training session, which is highly beneficial for varying online inference budgets.
- **The failure of Floortile is an honest control**: It informs the reader that the loop only turns when the initial search can occasionally reach the goal, suggesting future work could incorporate lightweight supervised pre-training or adaptive search budgets to bootstrap difficult domains.

## Limitations & Future Work
- **Reliance on known models**: Environments with stochasticity, partial observability, or raw sensory inputs cannot be directly applied, as WA* requires explicit transition functions and applicable action sets.
- **Sensitivity to initial search feasibility**: In domains like Floortile, where initial 60s searches rarely find the goal, the loop stalls. Smarter warm-starts (e.g., hybrid supervised pre-training) or adaptive budgets are needed.
- **Multiplicative rewards/speed are not optimized**: Unlike works such as RL for LLMs, GSP does not treat the "computation budget" as an optimizable target. Future work could include node expansions in the return.
- **GNN Expressivity Upper Bound**: Relational GNNs based on WL-equivalence cannot distinguish certain graph isomorphism pairs, which may limit policy performance in symmetrical or dense domains. Symmetry pruning (e.g., from Distincter) could complement this framework.
- **Training duration is still significant**: 12 hours per domain with 5 workers is non-negligible cost. Scaling to massive domain libraries requires more efficient batching and parallelization schemes.

## Related Work & Insights
This work follows the long tradition of "search-learning symbiosis" (LRTA*/RTDP) but extends it to the new dimension of "learning a generalized $Q$ across a family of instances." Compared to Lifted HER (Ståhlberg & Geffner 2026), the core difference is the replacement of real-time search + HER with best-first search. Compared to the AlphaZero family, it removes the dependency on MCTS, better fitting the structure of single-goal long-horizon puzzles. Compared to supervised heuristic learning (Chen 2025, Horčik 2025, Bai 2025), its main advantage is not requiring an optimal planner for $h^\ast(s)$ labels and possessing a self-improving loop. It suggests a unified "model-known RL" path: for any decision problem where the model is fully known (manually modeled robotics, CO, formal reasoning), one should reconsider the workflow of "best-first search for exploration + learning a cross-instance $Q_\theta$."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Surprising Difficulty of Search in Model-Based Reinforcement Learning](the_surprising_difficulty_of_search_in_model-based_reinforcement_learning.md)
- [\[ICML 2026\] What Does Reinforcement Learning for Visual Tool Use Actually Learn?](what_does_vision_tool-use_reinforcement_learning_really_learn_disentangling_tool.md)
- [\[ICML 2026\] Laplacian Representations for Decision-Time Planning](laplacian_representations_for_decision-time_planning.md)
- [\[ICML 2026\] Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](safety_generalization_under_distribution_shift_in_safe_reinforcement_learning_a_.md)
- [\[ICML 2026\] ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization](asap_exploiting_the_satisficing_generalization_edge_in_neural_combinatorial_opti.md)

</div>

<!-- RELATED:END -->
