---
title: >-
  [Paper Note] Learning to Insert for Constructive Neural Vehicle Routing Solver
description: >-
  [NeurIPS 2025][Optimization][Vehicle Routing Problem] This paper proposes L2C-Insert, the first learning-based insertion construction paradigm for neural combinatorial optimization. By allowing nodes to be inserted at any feasible position within a partial solution—rather than appended only to its end—the method significantly improves construction quality and flexibility for TSP/CVRP.
tags:
  - NeurIPS 2025
  - Optimization
  - Vehicle Routing Problem
  - Neural Combinatorial Optimization
  - Insertion-based Construction
  - TSP
  - CVRP
date: 2026-05-08
content_hash: ddf04247b29420e0
---

# Learning to Insert for Constructive Neural Vehicle Routing Solver

**Conference**: NeurIPS 2025
**arXiv**: [2505.13904](https://arxiv.org/abs/2505.13904)
**Code**: [GitHub](https://github.com/CIAM-Group/L2C_Insert)
**Area**: Optimization
**Keywords**: Vehicle Routing Problem, Neural Combinatorial Optimization, Insertion-based Construction, TSP, CVRP

## TL;DR

This paper proposes L2C-Insert, the first learning-based insertion construction paradigm for neural combinatorial optimization. By allowing nodes to be inserted at any feasible position within a partial solution—rather than appended only to its end—the method significantly improves construction quality and flexibility for TSP/CVRP.

## Background & Motivation

**Background**: Neural combinatorial optimization (NCO) methods leverage neural networks to automatically learn heuristic strategies for solving VRPs, with constructive approaches receiving the most attention. Existing methods almost universally adopt an appending paradigm—sequentially appending unvisited nodes to the end of the current solution.

**Limitations of Prior Work**: The appending paradigm imposes strict operational constraints: new nodes can only be added to the tail of the solution, preventing more effective modifications. Early greedy decisions frequently lead to suboptimal outcomes such as route crossings.

**Key Challenge**: The appending paradigm works well in NLP seq2seq frameworks, but the solution space structure of VRP differs fundamentally from language sequences—solution quality depends heavily on the global positional relationships among nodes rather than local ordering.

**Goal**: To explore the potential of an insertion-based paradigm in NCO, and to design learning-driven insertion strategies, training schemes, and inference techniques.

**Key Insight**: Building on well-established insertion heuristics from operations research, this work integrates them with neural network learning to overcome the limitations of traditional hand-crafted strategies.

**Core Idea**: The construction process is reformulated from "selecting which node to append" to "selecting where to insert a node into the partial solution," with attention mechanisms used to learn optimal insertion positions.

## Method

### Overall Architecture

L2C-Insert adopts an Encoder-Decoder architecture: the Encoder maps node features to embedding vectors, and the Decoder maintains three types of embeddings at each step—the current node, unvisited nodes, and insertion positions within the partial solution—computing insertion probabilities via multi-layer attention.

### Key Designs

1. **Position Embedding**: Insertion positions in the partial solution are represented by horizontally concatenating the embeddings of adjacent visited nodes. → The embedding for position $(\pi_i, \pi_{i+1})$ is $\mathbf{e}_{\pi_i} = W_2[\mathbf{h}_{\pi_i}, \mathbf{h}_{\pi_{i+1}}]$. → The $2d$-dimensional concatenation is projected to $d$ dimensions for consistency. → Unlike node selection embeddings, these encode the characteristics of the "gap" between nodes.

2. **Multi-layer Attention Decoder**: The current node embedding, unvisited node embeddings, and position embeddings are jointly fed into $L$ stacked attention layers. → Each layer applies self-attention and a feed-forward network. → Insertion probabilities over all positions are computed via linear projection followed by masked softmax: $p_i = \text{Softmax}(X_i^{(L)} W_f + b_f)$. → Illegal positions (current node and unvisited node positions) are masked to $-\infty$.

3. **Supervised Learning Training Scheme**: Given an instance with annotated optimal solution $\pi^*$, an initial node is randomly selected, and the remaining unvisited nodes are re-inserted into their target positions in the order defined by the annotated solution. → Target positions are defined by pairs of adjacent visited nodes in the annotated solution. → Loss function: $\mathcal{L}(\theta) = -\log p_\theta(\vec{e}^* \mid \vec{e}_{\pi_{1:l}}, \pi^*)$. → Supervised learning is preferred over RL because SL is more stable and effective under a heavy decoder.

4. **Insertion-based Local Reconstruction**: An iterative optimization technique applied at inference time. → **Destruction phase**: Distance-based proximity destruction is used (rather than conventional sequence-position-based destruction); a node is randomly selected and its $\alpha$ nearest neighbors are removed. → **Reconstruction phase**: The model re-inserts the removed nodes into the partial solution. → Key advantage: Distance-based destruction more effectively escapes local optima—nodes that are geometrically close but far apart in sequence order are difficult to co-consider under sequence-based destruction, but are naturally captured by distance-based destruction.

### Loss & Training

- Optimizer: Adam/AdamW
- Initial learning rate $1 \times 10^{-4}$, decayed by 0.97 per epoch
- Trained on 1 million 100-node instances for 50 epochs (TSP) / 15 epochs (CVRP)
- Inference uses greedy rollout with iterative local reconstruction

## Key Experimental Results

### Main Results (TSP)

| Method | TSP100 Gap | TSP1K Gap | TSP10K Gap |
|------|:-:|:-:|:-:|
| LKH3 | 0.000% | 0.000% | 0.000% |
| POMO aug×8 | 0.138% | 40.577% | OOM |
| LEHD RRC1000 | 0.0016% | 0.731% | 12.709% |
| BQ bs16 | 0.010% | 1.354% | OOM |
| **L2C-Insert (II=200)** | **0.0014%** | **0.952%** | **4.117%** |
| **L2C-Insert (II=1000)** | **0.0002%** | **0.481%** | **2.079%** |

### Main Results (CVRP)

| Method | CVRP100 Gap | CVRP1K Gap | CVRP10K Gap |
|------|:-:|:-:|:-:|
| HGS | 0.000% | 0.000% | 0.000% |
| LEHD RRC1000 | 0.420% | 3.028% | 9.916% |
| INViT-3V | 6.282% | 13.230% | 21.892% |
| **L2C-Insert (II=1000)** | **0.413%** | **4.836%** | **8.646%** |

### Ablation Study: Append vs. Insert

| Method | TSP100 Gap | TSP1K Gap | TSP10K Gap |
|------|:-:|:-:|:-:|
| L2C-Append (II=1000) | 0.00144% | 0.974% | 12.881% |
| **L2C-Insert (II=1000)** | **0.00017%** | **0.481%** | **2.079%** |
| Gap Reduction | 88.2% | 50.6% | 83.9% |

### Ablation Study: Destruction Strategy

| Destruction Strategy | TSP1K Gap | TSP10K Gap |
|------|:-:|:-:|
| Sequence-based | 1.182% | 6.980% |
| **Distance-based** | **0.481%** | **2.079%** |

### Key Findings

- **Insertion significantly outperforms appending**: Under an identical architecture, L2C-Insert reduces the optimality gap by 50%–88% compared to L2C-Append.
- **Strong scalability**: A model trained on 100-node instances generalizes directly to instances with up to 100K nodes.
- **Distance-based destruction is critical**: It yields a ~70% improvement over sequence-based destruction on TSP10K.
- **State-of-the-art on real-world benchmarks**: L2C-Insert outperforms all neural solvers across all instance groups in TSPLIB/CVRPLIB.
- **Efficient inference**: TSP with 100K nodes requires only 26 minutes, compared to ~400 hours for LKH3.

## Highlights & Insights

- L2C-Insert is the first work to introduce an insertion-based paradigm into NCO, filling a research gap that has persisted for over a decade.
- The design of position embeddings is elegant—concatenating adjacent node embeddings naturally encodes the characteristics of the intervening "gap."
- The distance-based local reconstruction strategy is naturally compatible with the insertion paradigm and cannot be fully exploited by appending-based methods.
- The supervised training scheme, in which target positions are defined by adjacent visited nodes in the annotated solution, is both conceptually simple and empirically effective.

## Limitations & Future Work

- The node selection strategy relies on a simple nearest-neighbor heuristic rather than being learned—a jointly trained node selection and position selection strategy remains an open direction.
- Validation is limited to TSP and CVRP; extension to a broader range of combinatorial optimization problems has not been explored.
- A hybrid append-insert solver is a promising but unrealized direction.
- The gap relative to HGS on large-scale CVRP instances remains non-trivial (~8% on CVRP100K).
- Training requires annotated optimal solutions, which constrains direct training at larger scales.

## Related Work & Insights

- Pointer Networks (Vinyals et al.) pioneered the application of the seq2seq framework to NCO, but the appending paradigm they established has since become an entrenched assumption in subsequent research.
- LEHD and BQ are strong recent appending-based baselines; L2C-Insert substantially surpasses them through a paradigm shift while maintaining a comparable architecture.
- Classical OR heuristics such as cheapest insertion serve as the conceptual inspiration for this work.
- Unlike improvement-based methods such as NeurOpt, L2C-Insert is a constructive method and can be combined with improvement-based approaches.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneers the insertion-based construction paradigm in NCO with a clear and compelling formulation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 100–100K nodes, synthetic and real-world datasets, with comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with intuitive figures and precise problem definitions
- Value: ⭐⭐⭐⭐⭐ Opens a new paradigm for NCO with consistently state-of-the-art empirical results

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Neural Thermodynamics: Entropic Forces in Deep and Universal Representation Learning](neural_thermodynamics_entropic_forces_in_deep_and_universal_representation_learn.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)

<!-- RELATED:END -->
