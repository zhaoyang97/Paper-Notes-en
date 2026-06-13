---
title: >-
  [Paper Note] MASPOB: Multi-Agent Prompt Optimization via GNN Surrogate + LinUCB + Coordinate Ascent
description: >-
  [ICML 2026][Multi-Agent][MAS] MASPOB treats multi-agent system (MAS) prompt optimization as budget-constrained black-box optimization. It utilizes a GAT surrogate model to capture prompt coupling under workflow topology…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "MAS"
  - "prompt optimization"
  - "GAT"
  - "LinUCB"
  - "coordinate ascent"
  - "black-box optimization"
date: 2026-05-08
content_hash: 1d09bd20d9d247ce
---

# MASPOB: Multi-Agent Prompt Optimization via GNN Surrogate + LinUCB + Coordinate Ascent

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2603.02630](https://arxiv.org/abs/2603.02630)  
**Code**: https://github.com/HZ1008/MASPOB  
**Area**: Multi-Agent / Prompt Optimization / Bayesian Optimization  
**Keywords**: MAS, prompt optimization, GAT, LinUCB, coordinate ascent, black-box optimization

## TL;DR
MASPOB treats multi-agent system (MAS) prompt optimization as budget-constrained black-box optimization. It utilizes a GAT surrogate model to capture prompt coupling under workflow topology, employs LinUCB in the embedding space to calculate epistemic uncertainty, and uses coordinate ascent to decompose joint search into sequential single-agent problems. This reduces complexity from $\mathcal{O}(\prod |\mathcal{P}_i|)$ to $\mathcal{O}(\sum |\mathcal{P}_i|)$. Across 6 benchmarks (QA/Code/Math), it achieves an average of 80.58, outperforming MIPRO (78.87), AFlow (78.52), and IO (68.56).

## Background & Motivation

**Background**: LLM multi-agent systems (MAS) enable multiple specialized agents to collaborate on complex tasks. MAS performance depends not only on the LLM itself but also on the workflow topology and agent prompts. While frameworks like AFlow and GPTSwarm explore automated topology, many real-world workflows are fixed due to expert validation and security audits, making prompt optimization the only feasible lever for improvement.

**Limitations of Prior Work**: Prompt optimization in MAS is a combined black-box problem presenting a triple challenge: (1) Expensive evaluation: a single metric measurement requires running the full end-to-end MAS with multiple LLM calls; (2) Topology-induced coupling: changes in an upstream agent's prompt alter the input distribution for downstream agents, making the objective non-decomposable; (3) Combinatorial explosion: the joint prompt space for $N$ agents is a Cartesian product.

**Key Challenge**: Existing prompt optimizers are either single-agent (OPRO/PromptBreeder/Instinct) and ignore topology by optimizing independently, or multi-stage but topology-agnostic (MIPRO using TPE) by modeling dependencies implicitly. Given a budget of 50 evaluations and an exponentially growing prompt space, they are sample-inefficient and miss high-quality coordinated combinations.

**Goal**: Simultaneously address the three challenges: sample-efficient exploration, topology-aware modeling, and scalable combinatorial search.

**Key Insight**: Reframe prompt optimization as a contextual bandit problem. UCB provides the exploration/exploitation balance; a GNN serves as a surrogate to capture inter-agent dependencies; coordinate ascent decomposes combinatorial optimization into sequential single-agent problems, reducing complexity from $O(\prod |\mathcal{P}_i|)$ to $O(\sum |\mathcal{P}_i|)$.

**Core Idea**: A three-part strategy. GAT message passing provides topology-aware $\mu(c)$; the information matrix $\mathbf{M}$ calculates uncertainty $\sigma(c) = \sqrt{\Phi(c)^\top \mathbf{M}^{-1} \Phi(c)}$; UCB $= \mu(c) + \alpha \sigma(c)$ guides the search; and coordinate ascent optimizes single-agent prompts sequentially each round.

## Method

### Overall Architecture

The MAS workflow is modeled as a DAG $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with $N$ agents, where each agent $a_i$ is bound to a prompt $p_i \in \mathcal{P}_i$. The objective is to find $c^* = \arg\max_c s(c)$ within $T = 50$ evaluation budgets. The process has three stages: (1) Warmup: $T_0$ rounds of random sampling + initial GAT training; (2) Main loop of Coordinate ascent + UCB for $T - T_0$ rounds, selecting the prompt with the highest UCB for each agent based on the current $c^*$ and evaluating the new $c$; (3) Updating the GAT, information matrix, and incumbent. The text encoder uses Qwen3-Embedding-8B, and the LLM backbone is GPT-4o-mini.

### Key Designs

1. **GAT Surrogate Model for Topology-induced Coupling**:

    - **Function**: Explicitly model the chain of "prompt changed → downstream agent input distribution changed → task performance changed" within the prediction.
    - **Mechanism**: Each agent is a node with feature $\Phi(p_i)$, and edges represent workflow dependencies plus self-loops. Multi-head GAT message passing is performed as $\mathbf{h}_i^{(l+1)} = \|_{k=1}^K \sigma(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(k)} \mathbf{W}^{(l,k)} \mathbf{h}_j^{(l)})$, where attention $\alpha_{ij}^{(k)}$ is computed via leaky-ReLU + softmax. Finally, mean pooling + MLP yields $\mu(c)$.
    - **Design Motivation**: OPRO/PromptBreeder treat MAS as a black box; MIPRO models dependencies implicitly and weakly. GAT explicitly encodes workflow topology, and its attention mechanism learns the relative importance of specific edges.

2. **LinUCB + Information Matrix for Epistemic Uncertainty**:

    - **Function**: Balance exploitation (high GAT predicted scores) and exploration (unseen combinations) under a budget of 50.
    - **Mechanism**: Maintain an information matrix $\mathbf{M} \in \mathbb{R}^{Nd \times Nd}$ initialized to $\lambda \mathbf{I}$, updated as $\mathbf{M} \leftarrow \mathbf{M} + \Phi(c) \Phi(c)^\top$ after each evaluation. Combined embedding $\Phi(c) = [\Phi(p_1); \dots; \Phi(p_N)] \in \mathbb{R}^{Nd}$; uncertainty $\sigma(c) = \sqrt{\Phi(c)^\top \mathbf{M}^{-1} \Phi(c)}$; UCB $= \mu(c) + \alpha \sigma(c)$.
    - **Design Motivation**: Since LLM evaluation is expensive, sample efficiency is mandatory. The information matrix is a classic tool from LinUCB; applying it to prompt embeddings provides a natural novelty signal more intelligent than $\epsilon$-greedy.

3. **Coordinate Ascent + UCB for Linear Scaling**:

    - **Function**: Reduce calculating UCB for all $\prod |\mathcal{P}_i|$ combinations to calculating $|\mathcal{P}_i|$ UCB values per agent.
    - **Mechanism**: Starting from the current best $c^*$, agents are updated sequentially: $p_i^* \leftarrow \arg\max_{p \in \mathcal{P}_i} \mathrm{UCB}(p_1^*, \dots, p_{i-1}^*, p, p_{i+1}^*, \dots, p_N^*)$. Each UCB evaluation only requires a GAT forward pass without running the MAS, incurring almost zero cost; full MAS evaluation is performed only once at the end of each round.
    - **Design Motivation**: UCB+GNN forward passes are cheap while MAS evaluations are expensive—distinguishing between "expensive" and "cheap" operations allows for clever compute allocation.

## Key Experimental Results

### Main Results: 6 Benchmarks (GPT-4o-mini, Average of 3 runs)

| Method | HotpotQA | DROP | HumanEval | MBPP | GSM8K | MATH | Average |
|------|---|---|---|---|---|---|---|
| IO | 60.36 | 53.09 | 89.31 | 69.11 | 87.80 | 51.71 | 68.56 |
| CoT | 67.62 | 58.27 | 89.57 | 69.89 | 88.34 | 52.47 | 71.03 |
| ReAct | 65.61 | 67.25 | 87.79 | 66.08 | 88.91 | 52.61 | 71.38 |
| PromptBreeder | 68.76 | 71.85 | 88.80 | 70.38 | 91.97 | 52.13 | 73.98 |
| Instinct | 69.92 | 71.90 | 90.08 | 70.23 | 92.64 | 52.40 | 74.53 |
| AFlow | 73.42 | 79.48 | 91.09 | 79.96 | 93.36 | 53.83 | 78.52 |
| MIPRO | 74.37 | 79.13 | 91.35 | 80.65 | 92.80 | 54.90 | 78.87 |
| **MASPOB** | **75.43** | **82.28** | **94.15** | **80.65** | **93.90** | **57.05** | **80.58** |

MASPOB achieves SOTA in 5 out of 6 tasks, with an average of 80.58 vs MIPRO's 78.87 (+1.71) and IO's 68.56 (+12.02).

### Complex Topology Experiments

On larger topologies automatically generated by AFlow (e.g., HotpotQA with 8 agents, DROP with 7, HumanEval with 7), MASPOB remains the top performer, with its advantage increasing as the number of agents grows—validating the specific value of explicit topology modeling with GATs.

### Key Findings

- **Average +12% relative to IO**: Prompt optimization overall provides a +12 score gain; MASPOB adds another +1.7 points over MIPRO.
- **Maximum improvement on MATH**: Mathematical reasoning requires multi-step logic chains, where topology coupling is strongest; MASPOB shows a +2.15 gain over MIPRO on MATH.
- **HumanEval at 94.15**: The ceiling for code generation is pushed to 94.15, which is particularly effective for high-value industrial scenarios.
- **Superiority under identical budget**: When all methods are restricted to a 50-evaluation budget, MASPOB's sample efficiency provides a genuine advantage.

## Highlights & Insights

- **Clear mapping of three challenges to three mechanisms**: expensive sampling → UCB; topology coupling → GAT; combinatorial space → coordinate ascent. Each component solves a specific pain point.
- **GNN as surrogate for prompt optimization**: Injecting workflow topology as an inductive bias into the surrogate is a methodological innovation.
- **LinUCB in prompt embedding space**: Transferring contextual bandit tools to LLM prompt optimization eliminates the need for manual $\epsilon$ tuning.
- **Coordinate ascent + cheap UCB forward**: Separating "expensive evaluations" from "cheap surrogate forwards" is a smart allocation of computational resources.
- **6 benchmarks across 3 domains**: Covers QA, Code, and Math, demonstrating task-agnostic versatility.
- **Adaptable to other multi-agent systems**: The approach can be generalized to any scenario requiring sample-efficient optimization over a structured combinatorial design space.

## Limitations & Future Work

- **Dependence on fixed workflows**: Assumes the topology is pre-designed and only optimizes prompts—though this fits the problem's premise.
- **Limited GAT training data**: 50 total evaluation samples might be insufficient for a GNN, potentially leading to overfitting; the selection of warmup $T_0$ requires tuning.
- **Prompt embedding quality depends on Qwen3-Embedding**: Fixed encoding might limit the expressiveness of the GAT.
- **Lack of comparison with DSPy / TextGrad**: Missing RL-style and gradient-based baselines.
- **Single backbone (GPT-4o-mini)**: Marginal gains under stronger LLMs remain unknown.
- **Uncertain long-horizon stability**: Whether the advantage persists beyond a 500+ evaluation budget is unknown.
- **Missing theoretical regret analysis**: Regret bounds for LinUCB in combinatorial spaces are not provided.

## Related Work & Insights

- **vs OPRO / PromptBreeder / Instinct**: These are single-agent and ignore topology; MASPOB uses GAT for explicit modeling.
- **vs MIPRO**: Uses multi-stage TPE where dependency is implicit; MASPOB is explicit.
- **vs AFlow / GPTSwarm**: These optimize topology; Ours optimizes prompts. The two are orthogonal and can be combined.
- **vs Bayesian Optimization**: Classic BO uses GP surrogates; MASPOB's GNN is better suited for structured combinatorial spaces.
- **Insights**: (1) GNN-surrogate + UCB can be applied to any sample-efficient black-box optimization in structured discrete spaces; (2) coordinate ascent paired with cheap surrogate forwarding is a universal trick for combinatorial problems; (3) workflow topology is a source of inductive bias that should be integrated into downstream modeling like prompt optimization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of GNN surrogate + LinUCB + coordinate ascent applied to MAS prompt optimization is a methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across 6 benchmarks in 3 domains + 7 baselines + complex topology experiments; lacks RL-style and varied LLM backbones.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivation (triple challenges), corresponding methods, and reproducible pseudo-code.
- Value: ⭐⭐⭐⭐⭐ Directly addresses MAS deployment pain points (fixed workflows + prompt optimization). Open-source code lowers the barrier, providing practical value for production LLM agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ICML 2026\] CoOT: Learning to Coordinate In-Context with Coordination Transformers](coot_learning_to_coordinate_in-context_with_coordination_transformers.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](../../ACL2026/multi_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)

</div>

<!-- RELATED:END -->
