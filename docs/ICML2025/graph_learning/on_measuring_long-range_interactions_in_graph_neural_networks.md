---
title: >-
  [Paper Note] On Measuring Long-Range Interactions in Graph Neural Networks
description: >-
  [ICML 2025][Graph Learning][Long-range interactions] This paper formally defines "long-range interactions" in graph tasks from first principles for the first time. It derives a unique range measure $\hat{\rho}_u = \mathbb{E}_{v \sim I_u}[d_G(u,v)]$ that satisfies four axioms. After validating its effectiveness through synthetic experiments, the authors utilize this measure to reveal that the peptides tasks in the LRGB benchmark are actually short-range.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Long-range interactions"
  - "GNN Range Measure"
  - "message passing"
  - "LRGB benchmark"
  - "over-squashing"
date: 2026-05-08
content_hash: d544c53834ba3975
---

# On Measuring Long-Range Interactions in Graph Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2506.05971](https://arxiv.org/abs/2506.05971)  
**Code**: [range-measure](https://github.com/BenGutteridge/range-measure)  
**Area**: Graph Learning  
**Keywords**: Long-range interactions, GNN Range Measure, message passing, LRGB benchmark, over-squashing

## TL;DR

This paper formally defines "long-range interactions" in graph tasks from first principles for the first time. It derives a unique range measure $\hat{\rho}_u = \mathbb{E}_{v \sim I_u}[d_G(u,v)]$ that satisfies four axioms. After validating its effectiveness through synthetic experiments, the authors utilize this measure to reveal that the peptides tasks in the LRGB benchmark are actually short-range.

## Background & Motivation

**Background**: In GNN research, "long-range" is a core topic—numerous works on over-smoothing, over-squashing, and graph rewiring focus on how GNNs can capture distant dependencies. LRGB (Long Range Graph Benchmark) is the standard benchmark for verifying long-range capabilities.

**Limitations of Prior Work**: (1) The concept of "long-range" is vague; existing benchmarks rely only on graph size/diameter and domain heuristics to classify a task as long-range, lacking theoretical support. (2) Synthetic long-range tasks are oversimplified, where simple rewiring can convert them into short-range tasks. (3) LRGB is highly sensitive to hyperparameters, raising doubts about whether its tasks are truly "long-range." (4) The few existing measures (e.g., problem radius, influence score) are either too coarse or lack axiomatic derivations.

**Key Challenge**: The lack of a theoretically grounded, quantifiable measure to answer "exactly how long-range is a task or a model?". Without such a measure, it is impossible to reliably evaluate whether architectural improvements genuinely resolve the long-range problem.

**Goal**: (1) Provide a formal definition of long-range interactions; (2) derive a family of range measures satisfying axiomatic properties; (3) reexamine the LRGB benchmark using the proposed measure.

**Key Insight**: The authors start from the axiomatization of linear tasks: a sound range measure should satisfy four properties: locality, unit interaction, additivity, and homogeneity. From these axioms, the mathematical form of range can be uniquely derived.

**Core Idea**: Use the weighted expectation of "distance $\times$ influence" as the unique axiomatic measure of range, and generalize it from linear tasks to arbitrary non-linear GNNs (via Jacobian/Hessian approximation).

## Method

### Overall Architecture

Given a graph $G$, a graph task $\mathbf{F}$ (which maps node features to outputs), and a distance metric $d_G$ on the graph, the method outputs a range value $\hat{\rho}_u$ for each node $u$, which is then aggregated into a graph-level or dataset-level range. To capture pairwise interaction strength between nodes, a Jacobian (first-order) is used for node-level tasks, and a Hessian (second-order) is used for graph-level tasks.

### Key Designs

1. **Node-level Range Measure (Jacobian-based)**:

    - **Function**: Measures the interaction range of each node in node-level tasks.
    - **Mechanism**: Defines four axioms: locality ($\rho_u$ depends only on interactions received by $u$), unit interaction (the range of a single interaction equals its distance), additivity (ranges of disjoint interactions are additive), and homogeneity (scaling interaction strength scales the range proportionally). These four axioms uniquely dictate: $\rho_u(\mathbf{L}) = \sum_v |\mathbf{L}_{uv}| \cdot d_G(u,v)$. The normalized version replaces Properties 2 and 3 with a weighted average property, yielding: $\hat{\rho}_u = \frac{1}{\sum_v |\mathbf{L}_{uv}|} \sum_v |\mathbf{L}_{uv}| \cdot d_G(u,v)$, which represents the expected distance under the influence distribution $I_u$.
    - **Design Motivation**: The normalized version avoids the issue where "a large number of short-range interactions leads to a high range," ensuring that the range reflects the average interaction distance rather than the total volume of interactions.

2. **Extension to Non-linear GNNs**:

    - **Function**: Generalizes the range of linear tasks to arbitrary differentiable mappings.
    - **Mechanism**: Employs the Jacobian $\frac{\partial \mathbf{F}_u}{\partial \mathbf{x}_v}$ as the measure of interaction strength for a non-linear mapping $\mathbf{F}$. The influence distribution is defined as $I_u(v) = \frac{1}{N_u}\sum_{\alpha,\beta}|\frac{\partial \mathbf{F}_u^\alpha}{\partial \mathbf{x}_v^\beta}|$, and the range is given by $\hat{\rho}_u = \mathbb{E}_{v \sim I_u}[d_G(u,v)]$. This formulation supports multiple input/output channels, and its probabilistic perspective allows for fast randomized approximation.
    - **Design Motivation**: The Jacobian is the best linear approximation, naturally inheriting the uniqueness guarantees from the linear case.

3. **Graph-level Range Measure (Hessian-based)**:

    - **Function**: Measures the pairwise interaction range between nodes in graph-level tasks.
    - **Mechanism**: For graph-level tasks, the Jacobian is only a vector and cannot capture pairwise interactions. Therefore, the Hessian (second-order term of Taylor expansion) is used: $\hat{\eta}_u = \mathbb{E}_{v \sim J_u}[d_G(u,v)]$, where $J_u(v) \propto \sum_{\alpha,\beta,\gamma}|\frac{\partial^2 \mathbf{y}^\gamma}{\partial \mathbf{x}_u^\alpha \partial \mathbf{x}_v^\beta}|$.
    - **Design Motivation**: The second-order derivative naturally captures the degree of feature mixing between nodes, aligning with the mixing measure proposed by Giovanni et al. (2024).

### Range Granularity Levels

Defines the aggregation hierarchy from node-level $\rightarrow$ graph-level $\rightarrow$ dataset-level (by sequential averaging), supporting both transductive and inductive settings.

## Key Experimental Results

### Synthetic Experiments: Range Variation Across Tasks

| Task Type | $k$=1 range | $k$=5 range | Growth Trend |
|---------|-------------|-------------|---------|
| $k$-Dirac | ~1.0 | ~5.0 | Linear growth |
| $k$-Rectangle | ~0.5 | ~2.5 | Linear growth |
| $k$-Power | ~0.5 | ~1.3 | Sublinear growth |

### LRGB Benchmark Evaluation

| Model | vocsuperpixels (F1↑) | range (hop) | peptides-func (AP↑) | range (hop) | peptides-struct (MAE↓) | range (hop) |
|------|---------------------|-------------|---------------------|-------------|----------------------|-------------|
| GCN | Low | ~2 | Good | ~1 | Good | ~1 |
| GINE | Med | ~2.5 | Good | ~1 | Good | ~1 |
| GatedGCN | Med | ~3 | Good | ~1 | Good | ~1 |
| GCN+VN | Med-High | ~5 | Med | ~3 | Med | ~3 |
| GPS | **High** | **~10** | Poor | ~6 | Poor | ~6 |
| GT | High | ~10 | Poor | ~10 | Poor | ~10 |

### Key Findings

- **vocsuperpixels is indeed a long-range task**: Model range positively correlates with performance—GPS (range ~10) performs the best, while pure MPNNs (range ~2-3) perform the worst. A higher range leads to better performance.
- **peptides tasks are not actually long-range**: Model range is **negatively correlated** with performance—the MPNNs perform best with a range of ~1, whereas the higher range of GPS/GT is actually detrimental. This challenges the fundamental premise of the LRGB benchmark.
- During training, the range of MPNNs increases rapidly in the first few epochs and then stabilizes, indicating that the model learns local features first before attempting to learn long-range ones.
- GCN remains the state-of-the-art on Peptides-struct to this day, further confirming that this task is fundamentally local.

## Highlights & Insights

- **Elegance of Axiomatic Derivation**: Starting from four intuitive axioms, the functional form of the range measure is uniquely determined without any heuristic choices. This deductive approach is uncommon in GNN theory and is highly exemplary.
- **Probabilistic Interpretation of "Influence $\times$ Distance"**: The normalized range is essentially the "expected distance under the influence distribution." This is not only conceptually clear but also enables efficient approximation via random sampling, making the metric scalable to large graphs.
- **Challenging the Authority of LRGB**: While the peptides tasks are widely used to validate the superiority of long-range architectures, the proposed range measure demonstrates that MPNNs achieve optimal performance with a range of approximately 1. This implies that many architectures claiming to "solve the long-range problem" might simply be performing better local feature extraction.

## Limitations & Future Work

- The computation of Hessian-based range is highly costly; the paper practically resorts to a Jacobian-based approximation in the LRGB experiments, and the quantitative relationship between the two requires further validation.
- The range measure relies on a local linear approximation (Jacobian/Hessian). For highly non-linear deep GNNs (e.g., GNNs with ReLU), the quality of such local approximations may be unstable.
- The paper predominantly analyzes three tasks in LRGB; extensions to a wider array of real-world datasets (especially molecular property prediction and protein structures) would be more compelling.
- The metric can currently only calculate the range of "already trained models" and cannot predict the range capabilities of an architecture before training.

## Related Work & Insights

- **vs. the problem radius of Alon & Yahav (2020)**: Problem radius is an intrinsic property of the task (the minimum number of hops required for the solution), whereas the proposed range is capable of measuring both the task and the model, backed by a rigorous axiomatic foundation.
- **vs. the over-squashing analysis of Di Giovanni et al. (2023)**: They employ pairwise node sensitivity analysis to study information propagation bottlenecks, which is extended in this work into a systematic family of measures.
- **vs. Liang et al. (2025)**: While they also proposed influence-based metrics to construct new long-range benchmarks, this paper demonstrates that their metrics are special cases within the proposed framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Formally derives a GNN range measure from first-principles axiomatization for the first time, offering outstanding methodological contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic experiments are thoroughly validated, and LRGB analysis is in-depth, though it lacks analysis on more diverse real-world tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The theoretical derivations are elegant and clear, the diagrams are meticulously designed, and the overall readability is exceptionally high.
- **Value**: ⭐⭐⭐⭐⭐ Provides a foundational tool for the GNN community and challenges the core assumptions of mainstream benchmarks, yielding a profound impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics](../../ICLR2026/graph_learning/improving_long-range_interactions_in_graph_neural_simulators_via_hamiltonian_dyn.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[ICML 2025\] A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks](a_cognac_shot_to_forget_bad_memories_corrective_unlearning_for_graph_neural_netw.md)
- [\[ICLR 2026\] Towards Quantifying Long-Range Interactions in Graph Machine Learning: A Large Graph Dataset and a Measurement](../../ICLR2026/graph_learning/towards_quantifying_long-range_interactions_in_graph_machine_learning_a_large_gr.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](../../ICLR2026/graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)

</div>

<!-- RELATED:END -->
