---
title: >-
  [Paper Note] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding
description: >-
  [ICLR2026][Graph Learning][MAPF] This paper proposes HMAGAT, which replaces the pairwise message passing of GNNs with a directed hypergraph attention network to model group interactions in multi-agent pathfinding, surpassing a 85M-parameter SOTA model using only 1M parameters and 1% of the training data.
tags:
  - ICLR2026
  - Graph Learning
  - MAPF
  - Hypergraph Neural Networks
  - Attention Mechanism
  - Imitation Learning
  - Group Interaction
date: 2026-05-08
content_hash: d2d9450e558870b3
---

# Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding

**Conference**: ICLR2026
**arXiv**: [2602.06733](https://arxiv.org/abs/2602.06733)
**Code**: [GitHub](https://github.com/proroklab/HMAGAT)
**Area**: Graph Learning
**Keywords**: MAPF, Hypergraph Neural Networks, Attention Mechanism, Imitation Learning, Group Interaction

## TL;DR
This paper proposes HMAGAT, which replaces the pairwise message passing of GNNs with a directed hypergraph attention network to model group interactions in multi-agent pathfinding, surpassing a 85M-parameter SOTA model using only 1M parameters and 1% of the training data.

## Background & Motivation

### State of the Field

**Background**:
1. Multi-Agent Pathfinding (MAPF) requires multiple agents to reach their respective goals without collisions; optimal solving is NP-hard.
2. Existing learning-based methods (GNNs, Transformers) model only pairwise interactions and fail to capture group dynamics arising from simultaneous multi-agent interactions.
3. In high-density scenarios, attention dilution in GNNs is particularly severe: large numbers of irrelevant agents dilute the attention weights of critical interactions.
4. MAPF is inherently a group problem — optimality and completeness can only be achieved by modeling the global joint state space.
5. Hypergraphs are naturally suited for modeling group interactions, yet have not been applied to complex, tightly coupled MAPF scenarios.
6. Existing hypergraph methods only handle small-scale (~10 agents) problems; whether they can scale to large-scale, highly coupled settings remains an open question.

## Method

**HMAGAT Architecture**: CNN encoder → Hypergraph Attention Network (HGNN) layers × 3 → MLP decoder

- **Directed Hyperedge Design**: Single-head, multi-tail structure — multiple tail nodes (influencers) → single head node (influenced agent), naturally modeling "multiple agents jointly influencing one agent's decision."
- **Two-Level Attention Mechanism**:
  - Tail-to-hyperedge attention $\alpha_{ej}$: mean of head node features as query; tail node features + hyperedge features as key-value.
  - Hyperedge-to-head attention $\alpha_{ie}$: head node as query; hyperedge representation as key-value.
  - Softmax normalization at each level is restricted to its own scope, preventing cross-level dilution.
- **Hypergraph Construction Strategies**:
  - Lloyd hypergraph: Voronoi partitioning with soft boundaries for overlapping groupings; suited for medium-scale scenarios.
  - k-means hypergraph: random point diffusion + clustering; complexity $O(k|V|)$; suited for large graphs.
  - Shortest-distance hypergraph: constructed based on shortest-path distances between agents; suited for obstacle-dense environments.
  - All strategies generate hyperedge features (relative position coordinates + Manhattan distance).

**Training Pipeline**:
- Expert demonstration trajectories collected on 21K instances using the lacam3 solver (vs. MAPF-GPT's 3.75M instances — 178× fewer).
- Cross-entropy loss for imitation learning, with optional DAgger online expert correction.
- Post-training quality improvement: fine-tuning on medium-difficulty instances to improve solution quality (rather than success rate alone).
- RL temperature sampling module: a small model dynamically adjusts the softmax temperature $\tau \in [0.5, 1.0]$ to promote more deterministic policies.

## Key Experimental Results

### Main Results

| Metric | HMAGAT (1M, 21K instances) | MAPF-GPT (85M, 3.75M instances) | MAGAT (GNN) |
|--------|---------------------------|----------------------------------|-------------|
| Parameters | ~1M (1.2%) | 85M (100%) | ~1M |
| Training Data | 21K (~0.56%) | 3.75M (100%) | 21K |
| Dense Warehouse Success Rate | **75%+** | <11% | ~5% |
| ost003d Large Map | ✓ Scalable | ✗ OOM | ✓ |
| Small Map Avg. SoC | **Best** | Competitive | Worse |

## Ablation Study & In-Depth Analysis

| Component | Dense Warehouse Success Rate | Notes |
|-----------|------------------------------|-------|
| Full HMAGAT | **39.8%** | Complete model |
| HGNN → GNN (MAGAT) | 2.3% | Hypergraph layer is critical — degrading to pairwise interaction causes performance collapse |
| w/o post-training | ~30% | Post-training yields ~10% gain in medium/high-density scenarios |
| w/o temperature sampling | ~35% | Temperature sampling reduces uncertain actions |
| Lloyd → k-means hypergraph | ~37% | k-means slightly weaker but computationally cheaper; preferred for large graphs |

### Attention Dilution Analysis
- In GNNs, as neighborhood density increases, attention weights of critical interactions are diluted by large numbers of irrelevant agents.
- The authors demonstrate that with $k$ critical neighbors among $n$ total neighbors, the attention allocated by a GNN to critical neighbors is approximately $k/n$, which decays as $n$ grows.
- HGNNs confine attention scope via hyperedges — each hyperedge contains only a small set of relevant agents, preventing external dilution.
- Validated through a handcrafted scenario: in a three-agent intersection setting, the GNN fails to capture three-way coordination while the HGNN succeeds.

### Large-Scale Scalability

| Map | HMAGAT | MAPF-GPT | Notes |
|-----|--------|----------|-------|
| ost003d (194×194) | ✓ Scalable | ✗ Cannot run | MAPF-GPT's Transformer runs OOM on large maps |
| Warehouse (high density) | **75%+** | <11% | HMAGAT substantially outperforms in high-density scenarios |

## Highlights & Insights
- **Inductive bias > data volume / parameter count** is the central conclusion — 1% of parameters and 1% of data outperform an 85M-parameter model, demonstrating that on problems with clear structure, the correct architectural prior (hypergraph) is more effective than brute-force data scaling.
- The formalization and experimental validation of **attention dilution** provides a clear explanation for GNN failure in high-density scenarios.
- The **handcrafted scenario** as a "minimal failure case" is highly convincing — it intuitively demonstrates why three-agent joint coordination cannot be decomposed into three pairwise interactions.
- **Practical hypergraph construction strategies**: the Lloyd hypergraph based on Voronoi partitioning and the k-means hypergraph suited for large-scale settings offer flexible options for real-world deployment.

## Limitations & Future Work
- Hypergraph construction strategies rely on preset communication radius $R^{\text{comm}}$ and color count — the optimal hypergraph structure is not learned adaptively.
- Evaluation is limited to grid environments; extension to continuous spaces or more complex topologies (e.g., 3D environments) remains unexplored.
- Imitation learning still depends on the quality of the expert solver — a suboptimal expert yields a suboptimal learned policy.
- Hypergraph construction introduces additional computational overhead; scalability at very large agent counts (>1000) requires further verification.
- Hyperedge size (number of agents per hyperedge) is implicitly determined by the construction strategy; an explicit analysis of optimal hyperedge size is absent.

## Related Work & Insights
- **vs. MAGAT (Li et al.)**: Under an identical framework (CNN + message passing + MLP), replacing only the GNN layers with HGNN layers directly demonstrates that the hypergraph is the sole factor driving performance gains.
- **vs. MAPF-GPT (Andreychuk et al.)**: An 85M-parameter GPT-style model trained on 3.75M instances is surpassed by HMAGAT with 1M parameters and 21K instances — model design matters far more than brute-force scaling.
- **vs. SCRIMP (Wang et al.)**: SCRIMP models communication via Transformers, which are fundamentally all-to-all pairwise attention mechanisms and are equally susceptible to attention dilution in high-density scenarios.
- **vs. HyperComm (Zhu et al.)**: HyperComm first applied hypergraphs to multi-agent communication but validated only on simple scenarios with ~10 agents; HMAGAT is the first to extend hypergraph-based methods to large-scale, tightly coupled MAPF.
- **Broader Implications**: The approach of modeling group interactions via hypergraphs is generalizable to traffic management, robotic swarm coordination, and protein–protein interaction networks — any domain requiring modeling of many-body effects.

## Rating
- Novelty: ⭐⭐⭐⭐ Hypergraph + MAPF is a first; the inductive bias argument is compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across multiple maps, ablations, attention analysis, and handcrafted scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures; the three-panel design of Figure 1 is particularly effective.
- Value: ⭐⭐⭐⭐ Broadly instructive for discussions on inductive bias in multi-agent learning.

### Overall Assessment
The most compelling aspect of this paper is not the performance numbers per se, but the extreme contrast — 1M vs. 85M parameters and 21K vs. 3.75M training instances — which directly demonstrates that on problems with clear structure, the correct inductive bias (hypergraph modeling of group interactions) is more effective than brute-force scaling of data and parameters. This serves as a valuable counterpoint to the "scaling-is-all-you-need" mindset prevalent in multi-agent learning, graph neural networks, and the era of large models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[ICLR 2026\] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)
- [\[AAAI 2026\] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](../../AAAI2026/graph_learning/assemble_your_crew_automatic_multi-agent_communication_topol.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)

</div>

<!-- RELATED:END -->
