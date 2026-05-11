---
title: >-
  [Paper Note] Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models
description: >-
  [ICLR 2026][Reinforcement Learning][Multi-Commodity Flow] This paper proposes Pram, the first framework to leverage multimodal language models (MLMs) for solving multi-commodity flow (MCF) problems. It decomposes the ori…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Multi-Commodity Flow"
  - "Multimodal Language Models"
  - "Multi-Agent Reinforcement Learning"
  - "Network Optimization"
  - "Partitioned Solving"
date: 2026-05-08
content_hash: b951087c991e3290
---

# Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models

**Conference**: ICLR 2026
**arXiv**: [2602.11057](https://arxiv.org/abs/2602.11057)
**Code**: [GitHub](https://github.com/Y-debug-sys/Pram)
**Area**: Reinforcement Learning
**Keywords**: Multi-Commodity Flow, Multimodal Language Models, Multi-Agent Reinforcement Learning, Network Optimization, Partitioned Solving

## TL;DR

This paper proposes Pram, the first framework to leverage multimodal language models (MLMs) for solving multi-commodity flow (MCF) problems. It decomposes the original problem into subproblems via partitioning and employs multi-agent reinforcement learning (MARL) to coordinate global consistency across subproblems. Theoretical convergence to the optimal solution is proven, and empirical results show that Pram is 1–2 orders of magnitude faster than LP solvers while achieving near-optimal performance.

## Background & Motivation

The multi-commodity flow (MCF) problem is a fundamental topic in network flow and combinatorial optimization, with broad applications in transportation, communications, and logistics. Problem objectives include minimizing maximum link utilization (MLU), maximizing throughput, and maximizing concurrent flow.

Existing approaches face two major limitations:

**Scalability bottleneck of LP solvers**: LP solving has complexity approximately $\mathcal{O}(d^{2.3729})$; when the variable scale reaches millions, runtime becomes prohibitively long (on the order of hours), and accurate future demand prediction is required.

**Limitations of ML-based methods**: (a) Specialized networks such as GNNs and RL agents incur high engineering costs and require repeated hyperparameter tuning; (b) generalization to unseen environments is poor; (c) output dimensionality grows quadratically with the number of nodes ($\mathcal{O}(|\mathcal{V}|^2)$), and the curse of dimensionality persists.

The core insight is **divide and conquer**: decomposing MCF into subproblems reduces the number of variables by a factor of $k^2$, while the strong mathematical reasoning and generalization capabilities of MLMs can replace specialized networks without frequent retraining.

## Method

### Overall Architecture

Pram (Partitioned Resource Allocation with MLMs) consists of three core modules:

1. **Partition Module**: Groups nodes by source, decomposing the MCF problem into $|\mathcal{V}|$ subproblems.
2. **Agent Module**: A shared MLM backbone processes each subproblem.
3. **Adaptation Module**: Learns global coordination via MARL.

### Key Design 1: Multimodal Problem Partitioning

Partitioning is performed at the source-node level, reducing model complexity from $\mathcal{O}(|\mathcal{V}|^2)$ to $\mathcal{O}(|\mathcal{V}|)$. Each subproblem receives inputs in two modalities:

- **Visual modality**: The routing links from the source node to all other nodes are rendered as a subgraph and fed into the MLM via a CLIP visual encoder.
- **Text modality**: Each demand is paired with a subtask description (source node information, historical rolling-average demand, etc.) via subtask-aware prompting.

The rationale for adopting MLMs as the agent backbone is twofold: (1) large-scale pretraining endows MLMs with emergent mathematical reasoning and strong generalization; (2) MLMs naturally handle multimodal image–text inputs, obviating the need for hand-crafted GNN/RNN architectures for complex structures.

### Key Design 2: Lightweight Multi-Agent Adaptation Framework

All subproblem agents share the MLM backbone but receive distinct observations, and coordinate via the following mechanisms:

**Communication mechanisms**:

- **Intra-model**: **LoRA** low-rank matrices are used to adapt the MLM's attention weights, introducing only a negligible number of additional parameters.
- **Extra-model**: Inspired by ICL, learnable **"global context"** embeddings serve as input prompt prefixes. The context parameters act as queries and are aligned with the frozen tokenizer embedding matrix via multi-head cross-attention, from which global information is extracted.

**Policy training (MARL)**:

- Counterfactual policy gradient is employed for fine-tuning.
- The single-step nature of MCF is exploited—actions (flow assignments) do not affect future states, so the expected return reduces to the immediate reward $R(s,a)$.
- The advantage function for each agent: $A_i(s,a) = R(s,a) - \sum_{a_i'} \pi_\theta(a_i'|s_i) R(s,(a_{-i},a_i'))$
- Policy gradient: $g = \mathbb{E}_\pi[\sum_i A_i(s,a) \nabla_\theta \log \pi_\theta(a_i|s_i)]$
- The counterfactual baseline is approximated via Monte Carlo sampling.

### Theoretical Guarantees

**Theorem 1 (GD Solves MCF)**: The MCF objective is convex/concave with respect to path weights, and there exists a step size $\eta > 0$ such that gradient descent converges to the optimum in a finite number of steps.

**Lemma 1 (MARL Convergence)**: Under bounded reward and Hessian assumptions, the policy iteration of Pram converges and the expected gradient tends to zero.

**Theorem 2 (Pram Implements GD)**: A constant-depth, constant-width adapted MLM can simulate multi-step gradient descent updates in its forward pass; that is, the MLM implicitly performs optimization in the token space via the ICL mechanism.

## Key Experimental Results

### Main Results: Real-World Datasets

| Method | MLU (↓) | Total Flow (↑) | Concurrent Flow (↑) | Requires True Demand |
|--------|---------|----------------|---------------------|----------------------|
| LP (Gurobi) | Optimal baseline | Optimal baseline | Optimal baseline | Yes |
| Pram | **2nd** (outperforms LP in some cases) | **2nd** | **2nd** | No |
| DRL | Poor (unstable training) | Poor | Poor | No |
| POP | Near worst | Near worst | Near worst | Yes |
| LP-top | Close to LP but unstable | Close to LP | Close to LP | Yes |

Key finding: Pram even surpasses LP on the MLU metric (21% lower on CERNET, 45% lower on GÉANT), consistent with the stronger convexity of the MLU objective.

### Main Results: Large-Scale Datasets (100–800 Nodes)

| Topology | Nodes | Pram Time | LP Time | Speedup | Pram vs. LP Performance |
|----------|-------|-----------|---------|---------|--------------------------|
| GtsCe | ~100 | Fast | Slow | ~10× | >90% |
| Colt | ~150 | Fast | Slow | ~50× | >90% |
| Kdl | 754 | <25s | ~2500s | **100×** | >90% |

On the largest topology (754 nodes, 1.9 million path weights), Pram is 100× faster than LP. On average, Pram outperforms HARP by 6.1%/16.6%/24.8% (MLU/throughput/concurrent flow) and Aether by 17.2%/7.3%/13.5%.

### Ablation Study

| Variant | Description | Effect |
|---------|-------------|--------|
| w/o MLM | Replaced with GNN+FC | Significant performance drop, especially on MLU |
| w/o Context | Global context embeddings removed | Performance degradation |
| w/o LoRA | Low-rank adapters removed | Performance degradation |
| w/o MARL | Direct end-to-end fine-tuning | Performance degradation |
| w/o Partition | No partitioning applied | Marginally better on small networks; infeasible on large networks (31.6 GB) |

### Key Findings

- **Strong generalization**: Performance degradation is <10% under link failures and <15% under traffic fluctuations with $\alpha=2$.
- **Parameter efficiency**: LoRA and context parameters do not grow with network scale; the unpartitioned variant has a parameter count approaching full fine-tuning.
- Visualization shows that the learned context embeddings are strongly correlated with MCF-relevant vocabulary (Flow, Demand, Capacity).

## Highlights & Insights

1. The **partitioning + MLM** combination is synergistic: partitioning resolves dimensional explosion while MLMs provide reasoning and generalization.
2. **Theoretical coherence**: MCF convexity → GD convergence → MLM simulates GD → MARL convergence, forming a logically complete chain.
3. **Engineering practicality**: The framework is objective-agnostic, integrates seamlessly into mainstream flow allocation systems, and the code is open-sourced.
4. Counterfactual policy gradient is naturally suited to the single-step nature of MCF, avoiding the credit assignment difficulties associated with multi-step RL.

## Limitations & Future Work

1. Fine-tuning remains resource-intensive even when truncating the backbone to only the first 8 layers.
2. The visual encoding scheme may introduce bias, as the rendering style of subgraphs affects information fidelity.
3. The current work focuses on static demand allocation; dynamic online scenarios remain unexplored.
4. Partition granularity is fixed at the source-node level; adaptive partitioning strategies may further improve efficiency.

## Related Work & Insights

- The partitioning idea is consistent with POP (Cohen et al. 2021), but the LP sub-solver is replaced by an MLM.
- The combination of LoRA and ICL context provides a lightweight paradigm for adapting pretrained large models to domain-specific optimization tasks.
- This work offers a theoretically grounded new instance of the "LLM as optimizer" paradigm.
- The MARL counterfactual gradient is generalizable to other multi-agent coordination problems with single-step decision-making.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Transitive RL: Value Learning via Divide and Conquer](transitive_rl_value_learning_via_divide_and_conquer.md)
- [\[ICLR 2026\] InFOM: Intention-Conditioned Flow Occupancy Models](infom_intention_flow_occupancy.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)
- [\[ICLR 2026\] TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models](tpru_advancing_temporal_and_procedural_understanding_in_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
