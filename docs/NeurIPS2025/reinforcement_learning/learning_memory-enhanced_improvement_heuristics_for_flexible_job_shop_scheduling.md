---
title: >-
  [Paper Note] Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling
description: >-
  [NeurIPS 2025][Reinforcement Learning][flexible job shop scheduling] This paper proposes MIStar—the first deep reinforcement learning (DRL)-based improvement heuristic framework for the Flexible Job Shop Scheduling Problem (FJSP). Key innovations include a directed heterogeneous disjunctive graph representation, a Memory-enhanced Heterogeneous Graph Neural Network (MHGNN), and a parallel greedy search strategy. MIStar consistently outperforms handcrafted improvement heuristics and state-of-the-art constructive DRL methods on both synthetic datasets and public benchmarks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - flexible job shop scheduling
  - improvement heuristics
  - memory-enhanced GNN
  - heterogeneous disjunctive graph
  - parallel greedy search
date: 2026-05-08
content_hash: 91c004f5032bb0f3
---

# Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling

**Conference**: NeurIPS 2025
**arXiv**: [2603.02846](https://arxiv.org/abs/2603.02846)
**Code**: None
**Area**: Combinatorial Optimization / Reinforcement Learning / Scheduling
**Keywords**: flexible job shop scheduling, improvement heuristics, memory-enhanced GNN, heterogeneous disjunctive graph, parallel greedy search

## TL;DR

This paper proposes MIStar—the first deep reinforcement learning (DRL)-based improvement heuristic framework for the Flexible Job Shop Scheduling Problem (FJSP). Key innovations include a directed heterogeneous disjunctive graph representation, a Memory-enhanced Heterogeneous Graph Neural Network (MHGNN), and a parallel greedy search strategy. MIStar consistently outperforms handcrafted improvement heuristics and state-of-the-art constructive DRL methods on both synthetic datasets and public benchmarks.

## Background & Motivation

**FJSP is a core scheduling problem in Industry 4.0**: Flexible job shop scheduling allows each operation to be processed on multiple compatible machines, better reflecting the personalized and dynamic production requirements of smart manufacturing compared to classical JSP. However, the many-to-one operation-machine relationship significantly increases solution complexity.

**Inherent limitations of constructive DRL methods**: Existing DRL approaches (e.g., HGNN, DANIEL) predominantly adopt constructive strategies, incrementally assigning operations to machines to build a complete schedule. However, the disjunctive graph representation of partial solutions is incomplete (missing edges between unscheduled operations), and encoding processing progress information (e.g., current machine load, job status) explicitly is difficult, leading to distorted state representations and limited solution quality.

**Improvement methods start from complete solutions with no information loss**: Improvement methods use complete scheduling solutions as MDP states, naturally avoiding the information loss inherent in partial solutions. The disjunctive graph under a complete solution accurately encodes all topological and ordering relationships among operations, making it more suitable for high-performance scheduling.

**Existing improvement DRL methods support only non-flexible JSP**: Zhang et al.'s L2S series targets JSP exclusively, using a conventional disjunctive graph (without machine nodes) and the N5 neighborhood structure, which cannot handle machine reassignment in FJSP. Direct extension faces three key challenges: (a) state representations must capture complex operation-machine relationships; (b) the action space must jointly cover operation reordering and machine reassignment; (c) flexibility enlarges the solution space, exacerbating the risk of local optima.

**Memory mechanisms can alleviate local optima**: Leveraging historical search trajectories to enhance current decision-making is an effective strategy for escaping local optima. However, existing approaches such as MARCO are limited to simple binary optimization problems, and their information aggregation schemes are ill-suited for the complex constraints of FJSP.

## Method

### Overall Architecture (MIStar)

Given an FJSP instance, an initial solution is sampled using DANIEL and converted into a directed heterogeneous disjunctive graph. MHGNN extracts state features from this graph. The policy network samples multiple candidate actions from the $\text{N}_{\text{opt2}}$ neighborhood, evaluates them in parallel, and executes the best action to update the current solution. This process iterates until a preset number of search steps is reached, retaining the global best solution found.

### MDP Formulation

- **State $s_t$**: A complete scheduling solution, described by a 9-dimensional feature vector per operation and a 4-dimensional feature vector per machine.
- **Action $a_t = [O_m, O_n, M_k]$**: Remove operation $O_m$ from its current machine and insert it before operation $O_n$ in the processing sequence of machine $M_k$ (based on the $\text{N}_{\text{opt2}}$ neighborhood structure).
- **Reward**: Composed of two terms—$r_{gain} = \max(C_{\max}(s_t^*) - C_{\max}(s_{t+1}), 0)$ measures improvement relative to the historical best solution; $r_{penalty}$ penalizes redundant exploration based on similarity to historical states. The total reward is $r_t = r_{gain} - r_{penalty}$, dominated by makespan improvement in early stages and gradually shifting toward encouraging diversity in later stages.

### Directed Heterogeneous Disjunctive Graph

Traditional disjunctive graphs contain only operation nodes and cannot capture machine states in FJSP. This paper proposes $\overrightarrow{\mathcal{H}} = (\mathcal{O}, \mathcal{M}, \mathcal{C}, \mathcal{E})$:

- **Operation nodes $\mathcal{O}$**: Include all operations and virtual source/sink nodes.
- **Machine nodes $\mathcal{M}$**: Explicitly model the state of each machine.
- **Conjunctive arcs $\mathcal{C}$**: Encode precedence constraints between consecutive operations within a job.
- **Directed hyper-arcs $\mathcal{E}$**: $E^k = (M_k, O^{k1}, O^{k2}, \ldots, O^{kn_k})$ encodes the complete processing order of operations on machine $M_k$. Unified directed edges enable unambiguous differentiation between distinct scheduling solutions.

### Memory-Enhanced Heterogeneous Graph Neural Network (MHGNN)

Features are extracted in three components:

**1. Operation node embeddings**: Dual-channel encoding.
- **Topological channel (GIN)**: A Graph Isomorphism Network encodes structural information—$\mu_{O_{ij}}^l = \text{MLP}^l((1+\epsilon^l) \cdot \mu_{O_{ij}}^{l-1} + \sum_{U \in N(O_{ij})} \mu_U^{l-1})$. GIN exhibits strong discriminative power for non-isomorphic graphs.
- **Semantic channel (GAT)**: A Graph Attention Network captures differentiated semantics from job-predecessor and machine-predecessor neighbors—$\tau_{O_{ij}}^l = \text{GAT}^l(\tau_{O_{ij}}^{l-1}, \{\tau_U^{l-1} | U \in N(O_{ij})\})$.
- Outputs from both channels are concatenated to yield operation embedding $h_{O_{ij}} \in \mathbb{R}^{2q}$.

**2. Machine node embeddings**: A heterogeneous GAT aggregates information from operations in each machine's processing sequence, capturing machine load. Independent linear transformations for different node types are used to compute attention coefficients.

**3. Historical action embeddings (memory module)**:
- Each step stores a state-action pair $(s_t, a_t)$, where the state is simplified to the operation-machine association matrix $\bm{L}_t$ (retaining only processing order, discarding node attributes).
- Similarity between the current state and historical states is computed via Frobenius inner product: $\omega_{t,t'} = \langle L_t, L_{t'} \rangle_F$.
- KNN retrieves the $K$ most similar historical actions; a **soft voting mechanism** aggregates them dimension-wise—separately voting for the best candidate among $O_m$, $O_n$, and $M_k$ by frequency weighted by similarity.
- This avoids the semantic invalidity that arises from directly averaging discrete indices.

### Decision-Making and Action Space Compression

Since machine assignments are fixed in an FJSP solution (given action $[O_m, O_n, M_k]$, $O_n$ must reside on $M_k$), the third dimension can be omitted, compressing the action space from $O(|\mathcal{O}|^2 |\mathcal{M}|)$ to $O(|\mathcal{O}|^2)$. The policy network outputs an operation scoring matrix; infeasible actions are masked via the $\text{N}_{\text{opt2}}$ neighborhood before softmax normalization yields the action probability distribution.

### Parallel Greedy Search Strategy

At each step, $P$ candidate actions are sampled and evaluated in parallel for makespan improvement, and the best action is executed. A single iteration explores $P$ solutions, achieving high-quality results with fewer iterations and significantly reducing search time. Training employs the $n$-step PPO algorithm.

## Key Experimental Results

### Synthetic Datasets

Two distributions: SD1 (compact distribution, initial solutions close to optimal) and SD2 (sparse distribution, large optimization headroom). Training is conducted on 4 small-scale instances and generalized to larger scales.

| Scale | Dataset | DANIEL(S) | HGNN(S) | GD-400 | BI-400 | **MIStar-400** | OR-Tools |
|-------|---------|-----------|---------|--------|--------|----------------|----------|
| 10×5 | SD2 | 11.96% | 46.94% | 10.40% | 7.18% | **4.98%** | 96% optimal |
| 20×10 | SD2 | 19.01% | 112.76% | 18.34% | 15.27% | **13.21%** | 1% optimal |
| 30×10 | SD2 | 9.28% | 109.95% | 8.87% | 6.82% | **5.27%** (generalized) | 0% optimal |
| 40×10 | SD2 | -4.53% | 94.11% | -5.08% | -6.81% | **-7.26%** (generalized) | 0% optimal |

→ MIStar consistently outperforms constructive DRL and handcrafted improvement heuristics at all scales, with runtime only 1/3–1/5 that of BI/FI methods.

### Public Benchmarks (Hurink & Brandimarte)

The model is trained on SD2 at 10×5 scale and transferred zero-shot to each benchmark:

| Benchmark | GD-400 Gap | BI-400 Gap | **MIStar-400 Gap** |
|-----------|------------|------------|--------------------|
| mk (Brandimarte) | 3.72% | 3.44% | **2.96%** |
| la (rdata) | 3.08% | 3.24% | **2.37%** |
| la (edata) | 6.96% | 6.94% | **6.83%** |
| la (vdata) | 0.53% | 0.64% | **0.24%** |

→ MIStar achieves the smallest gap across all benchmarks, with stable runtime that does not grow with machine flexibility.

### Large-Scale Instances (up to 1,500 operations)

| Scale | OR-Tools (1h) | MIStar-200 |
|-------|---------------|------------|
| 50×15 | 39.05% gap, 60min | 44.60% gap, 10.6min |
| 100×10 | 62.22% gap, 60min | 66.03% gap, 15.9min |
| **50×30** | **infeasible** | **feasible solution**, 60min |

→ On the extremely large 50×30 instances, OR-Tools completely fails to find a feasible solution, while MIStar still produces valid results, demonstrating strong scalability.

### Ablation Study

- The combination of the memory module and parallel greedy search yields the best overall performance.
- Parallel greedy search significantly improves search efficiency and mitigates the risk of premature convergence to local optima.
- The memory module refines policy decisions and improves solution quality.
- The parallel scale $P$ requires balancing runtime and solution quality.

## Highlights & Insights

- **First DRL-based improvement heuristic framework for FJSP**: Fills the gap in extending DRL improvement methods from JSP to FJSP, with a learned policy that generalizes across problem scales.
- **Directed hyper-arcs precisely encode machine processing sequences**: Compared to undirected edges in traditional disjunctive graphs, directed hyper-arcs unambiguously distinguish different scheduling solutions, resolving the core challenge of state representation in FJSP.
- **Elegant memory module design**: The operation-machine association matrix simplifies storage; the Frobenius inner product enables efficient similarity computation; soft voting avoids the semantic invalidity of averaging discrete action indices—all tailored to FJSP's complex constraints.
- **Action space compression**: Exploiting FJSP constraints reduces the space from $O(|\mathcal{O}|^2|\mathcal{M}|)$ to $O(|\mathcal{O}|^2)$, accelerating convergence.
- **Parallel greedy strategy balances efficiency and quality**: Exploring $P$ candidate solutions per iteration dramatically reduces the required number of iterations, with runtime only 1/3–1/5 that of rule-based methods.

## Limitations & Future Work

1. **Parallel scale $P$ is a hyperparameter**: The optimal $P$ depends on instance characteristics; an adaptive adjustment mechanism could be explored in future work.
2. **Locality of the $\text{N}_{\text{opt2}}$ neighborhood**: Only single-operation removal and reinsertion along the critical path are considered, precluding large-scale restructuring and retaining some dependence on initial solution quality.
3. **Dependence on DANIEL for initial solution generation**: Initial solution quality directly affects the starting point of the search; when the initial solution is already near-optimal (e.g., SD1), the improvement margin is limited.
4. **Memory module space overhead**: Storing state matrices at each step incurs growing storage and retrieval costs over long search trajectories.
5. **Single-objective optimization (makespan only)**: Multi-objective scheduling (e.g., machine utilization, total tardiness) is not considered, whereas practical production scenarios involve more complex requirements.
6. **Large gaps persist on large-scale instances**: A gap of 44.6% on 50×15 instances indicates that further work is needed before practical deployment.

## Related Work & Insights

- **L2S (Zhang et al.)**: The pioneering DRL improvement method for JSP and the direct predecessor of MIStar, using the N5 neighborhood and a conventional disjunctive graph. This paper extends and substantially upgrades that framework to FJSP.
- **HGNN (Song et al.)**: A representative constructive DRL method based on a heterogeneous disjunctive graph; MIStar adopts its heterogeneous graph concept but replaces undirected edges with directed hyper-arcs.
- **DANIEL**: A dual-attention constructive method used in this paper as the initial solution generator and baseline.
- **MARCO**: A pioneer in memory-augmented RL for combinatorial optimization, but limited to simple binary problems; this paper's soft voting aggregation mechanism is better suited to FJSP's complex constraints.
- **Insight**: The combination paradigm of improvement methods + memory mechanisms + heterogeneous graph representations is broadly applicable to other complex scheduling and combinatorial optimization problems (e.g., scheduling variants, vehicle routing problems).

## Rating

- Novelty: ⭐⭐⭐⭐ — First extension of improvement-based DRL framework to FJSP; the combination of directed heterogeneous disjunctive graphs and memory-enhanced GNN is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of synthetic datasets, public benchmarks, large-scale instances, and ablation studies with rich baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, sufficient technical detail, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐ — Makes a significant contribution to both FJSP solving and the DRL improvement methods community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers](blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](../../ICLR2026/reinforcement_learning/shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](../../ICLR2026/reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)

</div>

<!-- RELATED:END -->
