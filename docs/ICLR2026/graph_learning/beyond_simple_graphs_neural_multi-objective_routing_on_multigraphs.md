---
title: >-
  [Paper Note] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs
description: >-
  [ICLR 2026][Graph Learning][Multi-objective routing] This paper presents GMS, the first neural combinatorial optimization routing method for multigraphs, comprising two variants: GMS-EB, which performs edge-level autoregressive construction directly on the multigraph, and GMS-DH, a dual-head approach that learns to prune the multigraph before performing node-level routing. GMS achieves near-LKH performance on asymmetric multi-objective TSP and CVRP while being tens of times faster.
tags:
  - ICLR 2026
  - Graph Learning
  - Multi-objective routing
  - multigraph
  - graph neural networks
  - autoregressive construction
  - Pareto optimization
date: 2026-05-08
content_hash: 3a44b243aa7f25b9
---

# Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs

**Conference**: ICLR 2026
**arXiv**: [2506.22095](https://arxiv.org/abs/2506.22095)
**Code**: [https://github.com/filiprydin/GMS](https://github.com/filiprydin/GMS)
**Area**: Combinatorial Optimization / Graph Neural Networks / Vehicle Routing
**Keywords**: Multi-objective routing, multigraph, graph neural networks, autoregressive construction, Pareto optimization

## TL;DR
This paper presents GMS, the first neural combinatorial optimization routing method for multigraphs, comprising two variants: GMS-EB, which performs edge-level autoregressive construction directly on the multigraph, and GMS-DH, a dual-head approach that learns to prune the multigraph before performing node-level routing. GMS achieves near-LKH performance on asymmetric multi-objective TSP and CVRP while being tens of times faster.

## Background & Motivation

**State of the Field**: Deep learning-based combinatorial optimization solvers have made significant progress on vehicle routing problems (VRPs). Existing methods (Kool et al., POMO, MatNet, etc.) match or even surpass classical heuristics on TSP and CVRP. However, all existing neural solvers assume problems are defined on **simple graphs**—i.e., at most one edge between each pair of nodes.

**Limitations of Prior Work**: In the real world, multigraphs are more accurate representations—multiple edges with different attributes (e.g., different travel times and distances) may exist between the same pair of nodes. Operations research has shown that multigraph modeling can yield 5–10% cost improvements, yet existing neural methods cannot handle multigraphs for two reasons: (i) Transformer encoders are not suited to encoding multigraph structures; (ii) decoding must simultaneously select both node ordering **and** edges, whereas existing methods only select nodes.

**Root Cause**: Routing on multigraphs is substantially more complex than on simple graphs—one must decide not only the visitation order of nodes but also which edge to take between each pair. In a multi-objective setting, different edges offer different trade-offs across objectives, making it impossible to determine the optimal edge selection a priori.

**Paper Goals**: How to design a neural routing solver that handles multigraph inputs while supporting multi-objective optimization.

**Starting Point**: The authors propose two complementary strategies—(1) direct edge-level autoregressive construction on the multigraph; (2) a learned pruning strategy that simplifies the multigraph into a simple graph before node-level routing. Both variants use the Graph Edge Attention Network (GREAT) to process multigraph structures.

**Core Idea**: Encode multigraph edge embeddings via a GNN, then achieve multi-objective multigraph routing through either edge-level autoregression or a dual-head decoder combining non-autoregressive pruning with autoregressive routing.

## Method

### Overall Architecture
The input is a directed multigraph $G$ with $M$ edges per node pair, each edge carrying a multi-dimensional cost vector. The goal is to find the Pareto front—the set of solutions not dominated across all objectives. Multi-objective problems are decomposed into subproblems via Chebyshev scalarization, each corresponding to a preference vector $\lambda$. Both GMS variants share a GREAT-based encoder but differ in their decoding strategies.

### Key Designs

1. **GREAT Encoder (Graph Edge Attention Network)**:

    - **Function**: Encodes the multigraph into edge embeddings while preserving the distinguishability of parallel edges.
    - **Mechanism**: Each layer consists of attention sublayers that separately aggregate incoming and outgoing edge embeddings via attention scores $\alpha'$, $\alpha''$ to form temporary node representations $x_i = (\sum_{l \in E^+(i)} \alpha'_{il} W'_1 e_l \| \sum_{l' \in E^-(i)} \alpha''_{il'} W''_1 e_{l'})$, then update edge embeddings by concatenating head and tail node representations: $e'_l = W_2(x_{\text{start}(l)} \| x_{\text{end}(l)})$. Residual connections ensure parallel edges retain distinct embeddings.
    - **Design Motivation**: Standard Transformers cannot distinguish multiple edges between the same node pair. GREAT operates directly on edges and preserves per-edge identity through residual connections.

2. **GMS-EB (Edge-level Autoregressive Construction)**:

    - **Function**: Directly predicts the next edge to traverse on the multigraph (rather than the next node).
    - **Mechanism**: After the encoder produces edge embeddings, an edge-level Multi-Pointer decoder selects the next edge $\pi_t$ at each step with probability $p_{\theta(\lambda)}(\pi_t | \pi_{1:t-1}, s)$. The preference $\lambda$ is incorporated via an MLP hypernetwork that generates decoder weights $\theta_{\text{dec}}(\lambda) = \text{MLP}(\lambda)$; the encoder is preference-agnostic.
    - **Design Motivation**: Edge-level decoding is straightforward and learns edge selection and node ordering end-to-end. However, memory complexity is $\mathcal{O}(MN^4)$ (vs. $\mathcal{O}(N^3)$ for node-level), limiting scalability.

3. **GMS-DH (Dual-Head Decoding)**:

    - **Function**: A selection decoder first learns to prune the multigraph into a simple graph; a routing decoder then constructs the route.
    - **Mechanism**: **Stage 1 (Non-autoregressive pruning)**—the selection decoder, placed after an intermediate GREAT layer, independently selects one edge to retain per node pair with probability $q_{\tilde{\theta}(\lambda)}(l | i, j, s)$. **Stage 2 (Autoregressive routing)**—pruned edge embeddings are aggregated into node embeddings, passed through additional Transformer layers, and then decoded via a preference-conditioned Multi-Pointer decoder for node-level routing. Weights for both stages are generated by MLP hypernetworks.
    - **Design Motivation**: Decomposing the problem into "edge selection" and "route sequencing" subtasks reduces the complexity of the node-level decoder. The main encoder body is preference-agnostic and need only be run once, with multiple preferences sharing a single encoding pass, substantially improving inference efficiency.

### Loss & Training

- **Multi-objective REINFORCE**: GMS-EB uses the standard POMO framework; the reward is the negative Chebyshev scalarized cost $R_\lambda(\pi) = -\max_i\{\lambda_i|f_i(\pi) - z_i^*|\}$.
- **GMS-DH Dual-Head Training**: The selection head and routing head each have independent rewards and baselines. The routing head reward is $R^{(2)}_\lambda(\pi) = R_\lambda(\pi)$; the selection head reward approximates the optimal route using $K_2$ samples from the routing head: $R^{(1)}_\lambda(\mathcal{E}) \approx \max_{k=1,...,K_2} R_\lambda(\pi_k)$.
- **Curriculum Learning**: Training begins on small graphs and progressively scales up. For GMS-DH, the routing head is pre-trained on simple graphs to ensure approximate accuracy during early selection head training.

## Key Experimental Results

### Main Results
Evaluated on MOTSP, MOCVRP (simple graphs) and MGMOTSP, MGMOCVRP (multigraphs), using TMAT/XASY (simple graph) and FLEX/FIX (multigraph) distributions, at scales of 50/100 nodes. The metric is normalized hypervolume (HV).

| Problem | Method | HV (50 nodes) | Gap | HV (100 nodes) | Gap | Inference Time |
|---------|--------|--------------|-----|----------------|-----|----------------|
| MOTSP-TMAT | LKH (exact) | 0.58 | 0.00% | 0.63 | 0.00% | 6–29m |
| MOTSP-TMAT | MBM (aug) | 0.52 | 10.2% | 0.58 | 9.17% | 27s–2.4m |
| MOTSP-TMAT | **GMS-EB (aug)** | **0.57** | **1.14%** | **0.63** | **1.13%** | 3.3m–29m |
| MOTSP-TMAT | **GMS-DH (aug)** | **0.57** | **1.76%** | **0.62** | **2.19%** | 28s–2.6m |
| MOCVRP-XASY | GMS-EB (aug) | 0.75 | 0.00% | 0.80 | 0.00% | 4.9m–48m |
| MGMOTSP-FLEX2 | GMS-EB (aug) | 0.89 | 1.50% | 0.93 | 1.47% | 7.5m–1.2h |
| MGMOTSP-FLEX2 | GMS-DH (aug) | 0.89 | 1.45% | 0.93 | 1.61% | 1.5m–6.6m |
| MGMOTSP-FLEX2 | GMS-DH PP | 0.77 | 14.24% | 0.83 | 12.53% | 33s–2.3m |

### Ablation Study

| Configuration | MGMOTSP HV (50) | Notes |
|---------------|-----------------|-------|
| GMS-DH (full) | 0.89 (FLEX2) | Learned edge selection + routing |
| GMS-DH PP (pre-pruning) | 0.77 (FLEX2) | Pre-prune via linear scalarization; HV drops 14% |
| GMS-DH Simple (simple pruning) | 0.83 (FLEX2 MGMOTSPTW) | Replace selection head with simple function; drops 7% |
| MBM (pre-pruning) | 0.86–0.87 (FLEX2) | MatNet + pre-pruning; unstable performance |

### Zero-Shot Generalization (200 nodes; trained on ≤100 nodes)

| Problem | Method | HV (200 nodes) | Gap |
|---------|--------|----------------|-----|
| MOTSP-TMAT | LKH | 0.67 | 0.00% |
| MOTSP-TMAT | GMS-EB | 0.66 | 1.86% |
| MOTSP-TMAT | GMS-DH | 0.65 | 3.59% |
| MGMOTSP-FLEX2 | GMS-EB | 0.95 | 1.74% |
| MGMOTSP-FLEX2 | GMS-DH | 0.95 | 2.27% |

### Key Findings
- GMS-EB achieves the best performance across all problems (close to LKH) but incurs higher memory and runtime overhead.
- GMS-DH performs slightly worse but is 5–10× faster at inference, making it more suitable for large-scale applications.
- Pre-pruning (GMS-DH PP) causes severe performance degradation (10–15%), highlighting the critical importance of explicitly modeling multigraph structure.
- On MGMOTSPTW (a complex problem with time windows), GMS-EB/DH substantially outperform all baselines; the advantage of learned edge selection over simple pruning is even more pronounced.
- MOEAs (evolutionary algorithms) are significantly weaker than neural methods across all settings, especially at 100 nodes.
- Zero-shot generalization to 200 nodes incurs only minimal performance degradation (HV gap < 4%).

## Highlights & Insights
- **The dual-head NAR pruning + AR routing design is particularly elegant**: it decomposes the multigraph problem into two complementary tasks—the selection head runs only once (non-autoregressively), while the routing head operates efficiently on the simplified simple graph. This architectural decomposition simultaneously addresses the tension between expressiveness and computational efficiency.
- **Preference-agnostic encoder design**: The GREAT encoder does not depend on the preference $\lambda$; a single encoding pass per instance serves inference across all preferences. The MLP hypernetwork only modulates decoder weights, substantially reducing computation for multi-preference inference.
- **Counterintuitive failure of pre-pruning**: Even though Proposition 1 provides a theoretical guarantee that pre-pruning via linear scalarization incurs no optimality loss for MOTSP, combining pre-pruning with GMS-DH still leads to severe performance degradation in practice. The authors conjecture that pre-pruning shifts the data distribution, making the model harder to train.

## Limitations & Future Work
- The $\mathcal{O}(MN^4)$ complexity of GMS-EB limits applicability to large-scale problems; the authors themselves acknowledge the need for more scalable solutions.
- Experiments consider at most 2 parallel edges per node pair (FLEX2/FIX2); scenarios with more parallel edges are not evaluated.
- Main experiments are conducted on bi-objective problems only; tri-objective results are relegated to the appendix, and scalability to higher-dimensional objectives is not thoroughly validated.
- Training requires large numbers of randomly generated instances (200 epochs × 100K instances), resulting in substantial training costs.
- The curriculum learning strategy (small-to-large graphs) may not generalize consistently across different problem distributions.

## Related Work & Insights
- **vs. MatNet/POMO**: MatNet+MP (MBM) serves as a naive multigraph extension (prune then solve) but exhibits unstable performance on certain distributions (e.g., ~20% gap on XASY100 MOCVRP). GMS achieves better robustness through explicit multigraph architectural design.
- **vs. LKH/HGS**: Classical exact solvers yield the best quality but are 10–100× slower. GMS provides a practical real-time alternative within a 1–3% HV gap.
- **vs. MOEAs**: Evolutionary algorithms substantially underperform neural methods on these problems (gap 10–60%), demonstrating the advantage of learned inductive biases over unguided search.
- The dual-head AR+NAR paradigm is transferable to other combinatorial optimization problems that require simultaneous discrete selection and sequential construction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First extension of neural combinatorial optimization to multigraphs; both edge-level autoregression and dual-head decoding are novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple problems (TSP/CVRP/TSPTW), distributions (TMAT/XASY/FLEX/FIX), and scales (50/100/200); ablation and generalization studies are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; the theoretical analysis in Proposition 1 provides principled motivation for the method design.
- **Value**: ⭐⭐⭐⭐ — Fills a gap in neural routing for multigraphs with practical relevance to logistics and transportation domains.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](../../AAAI2026/graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[ICLR 2026\] Revisiting Node Affinity Prediction in Temporal Graphs](revisiting_node_affinity_prediction_in_temporal_graphs.md)

<!-- RELATED:END -->
