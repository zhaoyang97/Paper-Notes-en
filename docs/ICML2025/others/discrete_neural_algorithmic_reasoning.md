---
title: >-
  [Paper Note] Discrete Neural Algorithmic Reasoning
description: >-
  [ICML2025][Discretization] This paper proposes the Discrete Neural Algorithmic Reasoner (DNAR). By leveraging three core components—feature discretization, hard attention, and separate continuous/discrete data flows—DNAR forces neural networks to execute algorithmic trajectories along a finite set of predefined states. It achieves a **100% perfect test score** on tasks such as BFS, DFS, Dijkstra, Prim, and MIS, and allows for formal proofs of the correctness of the learned al…
tags:
  - "ICML2025"
  - "Discretization"
  - "Graph Neural Networks"
  - "Algorithmic Simulation"
  - "Generalization"
  - "Interpretability"
  - "Hard Attention"
date: 2026-05-08
content_hash: 785bf897ffc0f97d
---

# Discrete Neural Algorithmic Reasoning

**Conference**: ICML2025  
**arXiv**: [2402.11628](https://arxiv.org/abs/2402.11628)  
**Code**: [yandex-research/dnar](https://github.com/yandex-research/dnar)  
**Area**: Other  
**Keywords**: Discretization, Graph Neural Networks, Algorithmic Simulation, Generalization, Interpretability, Hard Attention

## TL;DR

This paper proposes the Discrete Neural Algorithmic Reasoner (DNAR). By leveraging three core components—feature discretization, hard attention, and separate continuous/discrete data flows—DNAR forces neural networks to execute algorithmic trajectories along a finite set of predefined states. It achieves a **100% perfect test score** on tasks such as BFS, DFS, Dijkstra, Prim, and MIS, and allows for formal proofs of the correctness of the learned algorithms.

## Background & Motivation

Neural Algorithmic Reasoning aims to simulate the execution of classical algorithms using neural networks. The CLRS-30 benchmark unifies multiple classical algorithms as operations on graphs, training models to step-by-step simulate algorithmic state transitions (hints).

**Key Challenge**: Current GNN-based neural reasoners generalize poorly to out-of-distribution (OOD) data, especially exhibiting a severe drop in performance when testing graphs are much larger than training ones. The fundamental reasons are:

1. State transitions of classical algorithms are **discrete and deterministic**, which are unaffected by distribution shifts, whereas neural networks operate in continuous latent spaces and accumulate errors easily when facing OOD inputs.
2. Even in the simplest real-value addition cases, gradient optimization struggles to accurately simulate the target computation (Klindt, 2023).
3. For large graphs, softmax attention weights are diluted (annealing), leading to the degradation of message passing.

**Motivation**: Given that the correctness of an algorithm stems from deterministic transitions between finite discrete states, can we enforce the neural reasoner to also maintain discrete states? This would not only enhance generalization but also make the model naturally interpretable, allowing for formal proofs of its correctness.

## Method

### Overall Architecture: Encode-Process-Decode

The model follows the encode-process-decode paradigm. Node and edge features of the input graph $G$ are mapped to high-dimensional vectors by linear encoders, and the processor (a single-layer GNN) recurrently updates the features:

$$X^{t+1}, E^{t+1} = \text{Processor}(X^t, E^t, A, S^t)$$

$$S^{t+1} = \text{ScalarUpdate}(X^t, E^t, A, S^t)$$

where $X^t$ represents node features, $E^t$ represents edge features, $S^t$ is the continuous scalar input, and $A$ is the adjacency matrix.

### Three Core Components

**1. Feature Discretization**

A discrete bottleneck layer is added at the output of the processor to map continuous features to a finite set of states:

$$x_i^{t+1} = \text{Discretize}_{\text{nodes}}(\hat{x}_i^{t+1})$$

$$e_{ij}^{t+1} = \text{Discretize}_{\text{edges}}(\hat{e}_{ij}^{t+1})$$

Implementation details: features are projected into $k$-dimensional state logits, and argmax is applied during inference to produce one-hot discrete states. During training, teacher forcing or Gumbel-Softmax is used (with the temperature annealing from 3.0 to 0.01).

**2. Hard Attention**

Standard softmax attention gets diluted on large graphs. Hard attention ensures each node only attends to the most important neighbors:

$$\alpha_{ij} = \frac{\langle Q_j, K_i + K_{ij} \rangle}{\sqrt{h}}$$

By taking argmax instead of softmax, the set of messages each node can receive is strictly restricted, ensuring consistent behavior across any graph scale.

**3. Separation of Continuous and Discrete Data Flows**

Most algorithms involve continuous inputs (such as edge weights). Direct discretization would lose this information. This paper maintains scalar inputs $S$ separately from discrete states:

- **Read interface**: Scalars solely act as edge priorities in attention for tie-breaking among neighbors with the same discrete state.
- **Update interface**: Scalars are updated through discrete operations (increment/keep/push) to avoid learning high-precision continuous operations:

$$s_i^{t+1} = \text{inc}(x_i^t) + \text{keep}(x_i^t) \cdot s_i^t + \sum_{j \in \mathcal{N}(i)} \text{push}(e_{ji}^t) \cdot s_{ji}^t$$

where $\text{inc}$, $\text{keep}$, and $\text{push}$ are 0-1 discrete functions, obtained by linear projection and discretization of node/edge features.

## Key Experimental Results

### Datasets

The SALSA-CLRS benchmark is used, covering 6 tasks: BFS, DFS, shortest path (Dijkstra), minimum spanning tree (Prim), maximum independent set (MIS), and eccentricity. The training set consists of ER random graphs with $\le 16$ nodes, and the test set scale scales up to 16–1600 nodes (up to a 100x size extrapolation).

### Main Results (SALSA-CLRS)

| Task | Graph Size | GIN (Node/Graph) | PGN (Node/Graph) | **DNAR (Node/Graph)** |
|------|------------|------------------|------------------|---------------------|
| BFS | 16 | 98.8 / 92.5 | 100. / 100. | **100. / 100.** |
| BFS | 1600 | 86.5 / 0.0 | 98.5 / 0.0 | **100. / 100.** |
| DFS | 16 | 41.5 / 0.0 | 82.0 / 19.9 | **100. / 100.** |
| DFS | 1600 | 17.8 / 0.0 | 23.1 / 0.0 | **100. / 100.** |
| SP | 1600 | 36.9 / 0.0 | 84.5 / 0.0 | **100. / 100.** |
| Prim | 1600 | 43.2 / 0.0 | 66.8 / 0.0 | **100. / 100.** |
| MIS | 1600 | 79.2 / 0.0 | 98.9 / 5.2 | **100. / 100.** |
| Ecc. | 1600 | NA / 16.0 | NA / 83.0 | **NA / 100.** |

DNAR achieves 100% node-level and graph-level accuracy on **all tasks and all test scales**.

### CLRS-30 Comparison (Graph Size 64)

| Task | Hint-ReLIC | G-ForgetNet | **DNAR** |
|------|------------|-------------|----------|
| BFS | 99.00 | 99.96 | **100.** |
| DFS | — | — | **100.** |
| Dijkstra | — | — | **100.** |
| Prim | — | — | **100.** |

### Multi-Task Experiments

With a single processor and task-specific encoders/decoders, executing all 6 algorithms simultaneously still achieves **perfect generalization**. Training only requires a single A100 GPU, taking <1h for a single task and 5-6h for multi-task learning.

## Highlights & Insights

1. **Perfect Generalization + Provable Correctness**: This work is the first to achieve 100% test accuracy in neural algorithmic reasoning and to formally prove the correctness of the learned algorithm on any input. This represents a milestone in the field.
2. **Profound Design Philosophy**: Grounded in the insight that "the generalization of algorithms originates from deterministic transitions between discrete states," the authors systematically design three complementary components.
3. **Clever Continuous/Discrete Separation**: Restricting scalars to acting only as priority tie-breakers in attention avoids information loss caused by discretization, while scalar updates are handled elegantly via discrete operations (inc/keep/push).
4. **Extremely Efficient Training**: The model converges within 1000 steps, requiring less than an hour on a single A100 GPU.
5. **Multi-task Capability**: A single architecture perfectly runs 6 different algorithms simultaneously, demonstrating robust versatility.

## Limitations & Future Work

1. **Reliance on Hint Supervision**: The model depends on step-by-step supervision of algorithmic execution trajectories (teacher forcing); exploration under hint-free settings remains in its infancy.
2. **Limited Task Coverage**: Currently, only 6 graph algorithm tasks have been verified, lacking validation on non-graph algorithms like sorting or string matching.
3. **Fixed Scalar Operations**: The three operations (inc/keep/push) might not suffice for more complex algorithms (e.g., scenarios requiring multiplication or division).
4. **Exclusion of Edge-Level Reasoning**: It lacks support for more complex data-flow interactions, such as edge-based reasoning or graph-level hints.
5. **Unclear Practical Scenarios**: The practical value of perfectly simulating classical algorithms remains to be clarified—if an algorithm is already known, why not execute it directly? The authors do not fully discuss the advantageous scenarios of a neural reasoner compared to direct execution.

## Related Work & Insights

- **CLRS-30** (Veličković et al., 2022): The standard benchmark for algorithmic reasoning.
- **SALSA-CLRS** (Minder et al., 2023): A more stringent benchmark for large-scale OOD evaluation.
- **Transformer Programs** (Friedman et al., 2023): Interpretable Transformers that enable trainable-to-readable program translation.
- **RASP/RASP-L** (Weiss et al., 2021; Zhou et al., 2024): Formal languages for Transformer computational models.

**Key Insight**: The path of discretization + provable correctness could provide inspiration for enhancing the reasoning capabilities of LLMs—if reasoning steps can be constrained to verifiable, discrete state spaces, it could improve the reliability of LLMs in mathematical and logical reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Discretization + provable correctness is pioneering in the field of algorithmic reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The perfect results are convincing, but the task coverage could be further expanded.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic, with a complete chain from motivation to method and experiments.
- Value: ⭐⭐⭐⭐ — Highly significant within the algorithmic reasoning subfield, though its practical utility remains to be demonstrated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](../../ACL2025/others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[ICML 2026\] MetaDNS: Enhancing Exploration in Discrete Neural Samplers via Well-Tempered Metadynamics](../../ICML2026/others/metadns_enhancing_exploration_in_discrete_neural_samplers_via_well-tempered_meta.md)
- [\[ICLR 2026\] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](../../ICLR2026/others/neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)
- [\[ACL 2025\] Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](../../ACL2025/others/neodiff_unified_text_diffusion.md)

</div>

<!-- RELATED:END -->
