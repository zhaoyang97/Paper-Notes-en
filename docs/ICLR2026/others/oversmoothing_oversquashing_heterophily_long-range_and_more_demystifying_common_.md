---
title: >-
  [Paper Note] Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning
description: >-
  [ICLR 2026][Others][oversmoothing] This paper systematically examines nine common myths in the field of graph machine learning regarding oversmoothing, oversquashing, homophily/heterophily, and long-range dependencies. By refuting each myth with concise counter-examples, the authors decouple "oversquashing" into two independent concepts—**computational
tags:
  - ICLR 2026
  - Others
  - oversmoothing
  - oversquashing
  - heterophily
  - long-range dependencies
  - message-passing
  - GNN
date: 2026-05-08
content_hash: fca8c9fbb710546a
---
# Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning

- **Conference**: ICLR2026
- **arXiv**: [2505.15547](https://arxiv.org/abs/2505.15547)
- **Code**: [GitHub](https://github.com/AdrianArnaiz/demystifyingGraphML)
- **Area**: Others
- **Keywords**: oversmoothing, oversquashing, heterophily, long-range dependencies, message-passing, GNN

## TL;DR

This paper systematically examines nine common myths in the field of graph machine learning regarding oversmoothing, oversquashing, homophily/heterophily, and long-range dependencies. By refuting each myth with concise counter-examples, the authors decouple "oversquashing" into two independent concepts—**computational bottleneck** and **topological bottleneck**—resolving widespread conceptual confusion in the literature.

## Background & Motivation

**Background**: Deep learning on graphs via message-passing has evolved rapidly. Researchers focus on inherent limitations such as oversmoothing (OSM, node representation convergence), oversquashing (OSQ, information loss during compression), the impact of homophily/heterophily on classification, and long-range information propagation.

**Limitations of Prior Work**: Due to the rapid pace of the field, many "common beliefs" have been widely disseminated and cited without sufficient verification. This has led to conceptual confusion and vague problem definitions, hindering targeted further research.

**Key Challenge**: Assertions in the literature—such as viewing oversmoothing as the fundamental cause of performance degradation, equating oversquashing directly with topological bottlenecks, or treating heterophily as synonymous with "difficulty"—do not hold universally but have become almost universally accepted.

**Goal**: To clearly identify the limitations of these beliefs and refute them through simple counter-examples, enabling researchers to distinguish and precisely define the problems they aim to solve.

**Key Insight**: Rather than critiquing specific works, the paper summarizes nebulous assertions scattered across the literature into nine common beliefs and "demystifies" them using mathematical definitions and counter-examples.

**Core Idea**: "Oversquashing" should be decomposed into two independent issues: **computational bottleneck** (stemming from the exponential expansion of the computation tree) and **topological bottleneck** (stemming from graph connectivity). These two can exist independently or be entirely unrelated.

## Method

### Overall Architecture

This paper does not propose a new model but performs a "conceptual demystification." It summarizes vague assertions found in numerous papers—often treated as established facts but never rigorously verified—into **nine common beliefs**, each refuted by a **concise but formally sufficient counter-example**. These nine beliefs are grouped into three themes: oversmoothing (OSM), homophily/heterophily, and oversquashing (OSQ). Each argument follows the same structure: a mathematical definition, a small graph counter-example, and a take-home message. The core contribution is the disentanglement of "oversquashing" into **computational** and **topological** bottlenecks. The following table provides an overview:

| Theme | Beliefs |
|------|---------|
| OSM | 1. OSM causes performance degradation; 2. OSM is an inherent property of all DGNs |
| Homophily/Heterophily | 3. Homophily is good, heterophily is bad; 4. Long-range propagation is evaluated on heterophilous graphs; 5. Different classes imply different features |
| OSQ | 6. OSQ = Topological bottleneck; 7. OSQ = Computational bottleneck; 8. OSQ is harmful to long-range tasks; 9. Topological bottlenecks are linked to long-range problems |

### Key Designs

The three design points correspond to the three themes in the table, following the paper's structure (OSM → Homophily/Heterophily → OSQ). The OSQ decomposition is the central contribution.

**1. Oversmoothing is neither universal nor the root cause of degradation**

Addressing the beliefs that "OSM causes performance drop" and is "inherent to all deep graph networks," the paper first notes that the metrics used to measure OSM are inconsistent. Literature typically uses Dirichlet Energy (DE) and the Rayleigh Quotient (RQ) to characterize convergence:

$$\mathrm{DE}(\mathbf{H}^\ell) = \mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell), \quad \mathrm{RQ} = \frac{\mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell)}{\|\mathbf{H}^\ell\|_F^2}$$

The crux is that whether OSM occurs and its direction depends heavily on architecture and hyperparameters. In GIN, DE can **explode** rather than decrease under standard settings; multiplying the weight matrix by 2 (i.e., $AX(2W)$) can reverse the DE trend; and DE and RQ often yield contradictory conclusions for the same model. Since changing the aggregation, weight scaling, or metric alters the conclusion, OSM is not universal and lacks a unique definition. Furthermore, even if DE decreases, it does not imply performance failure—nodes of different classes might collapse to their respective points ("beneficial smoothing" phase), while **class separability** remains or even improves. Accuracy drops are more likely caused by vanishing gradients and overfitting.

**2. Decoupling heterophily from task difficulty and long-range dependencies**

This set of beliefs treats "heterophily = difficult" and "long-range tasks only occur on heterophilous graphs" as common sense. The paper refutes these with two counter-examples. First, in a purely heterophilous bipartite graph, a 1-layer sum-based DGN can achieve perfect classification based solely on node degree differences—extreme heterophily can be trivial. Second, on a highly homophilous graph, if the task is to determine if a node is at a distance $>5$ from a specific node, long-range propagation is required despite the graph being homophilous. Together, these show that **long-range tasks and heterophily are orthogonal**; class labels induced by the task cannot be used to infer graph properties, and thus should not be conflated.

**3. Decoupling Oversquashing into Computational and Topological Bottlenecks**

This is the core contribution. While literature defaults OSQ to "graph topological bottlenecks," the paper provides a rigorous definition of computational bottlenecks and proves they are independent. A **computational bottleneck** refers to the size of the multiset of the computation tree expansion $|\mathcal{M}_v^K|$ for node $v$ after $K$ layers:

$$\mathcal{M}_v^K := \mathcal{M}_v^{K-1} \uplus \left\{\biguplus_{u \in \mathcal{M}_v^{K-1}} \mathcal{N}_u\right\}$$

This expands exponentially with the number of layers and is independent of the existence of a **topological bottleneck** (characterized by curvature/spectral gap, inherent to the graph structure). Two counter-examples clarify this: a grid graph has no topological bottlenecks but suffers from severe computational bottlenecks due to computation tree explosion. Conversely, graphs with topological bottlenecks may have minimal computational bottlenecks when the number of layers is small. Distinguishing these has practical implications: current graph rewiring methods improve topological bottlenecks by shortening distances but **worsen** computational bottlenecks for a fixed layer count. In contrast, "message filtering" (learning how much information to exchange/filter) can reduce both computational bottlenecks and sensitivity without altering the graph structure.

### Loss & Training

As an analytical work, this paper does not propose new models. Counter-examples are mostly synthetic small graphs. Experiments involved reproducing standard DGN training (Cross-Entropy loss for node classification) to demonstrate the inconsistency of OSM metrics.

## Key Experimental Results

### Main Results: OSM Metric Behavior across Architectures (Cora, 50 seeds)

| Architecture | W: DE Trend | 2W: DE Trend | W: RQ Trend | 2W: RQ Trend |
|------|-----------|-------------|-----------|-------------|
| GCN | Collapses to 0 | Explodes ↑ | Stable | Stable |
| GAT | Collapses to 0 | Explodes ↑ | Linear Decay | Stable |
| SAGE | Collapses to 0 | Explodes ↑ | Linear Decay | Partially Stable |
| GIN | Explodes ↑ | Explodes ↑↑ | Linear Decay | Partially Stable |

> Conclusion: The same model can show contradictory trends under different metrics. OSM observations are highly dependent on the choice of metric and hyperparameters.

### Ablation Study: Performance Degradation vs. DE with/without Bias

| Configuration | DE with Depth | Accuracy with Depth |
|------|------------|--------------|
| GCN (w/ bias) | Collapses | Decreases |
| GCN (w/o bias) | **Does not collapse** | Also decreases |

> This indicates that performance degradation cannot be attributed solely to OSM—removing bias eliminates DE collapse, yet accuracy still drops, suggesting the real causes are vanishing gradients or overfitting.

### Key Findings

1. **OSM is not universal**: Changing the aggregation function (GIN), fine-tuning weight scaling (2W), or switching metrics (DE vs RQ) changes the conclusion.
2. **Computational Bottleneck ≠ Topological Bottleneck**: Grid graphs have no topological bottlenecks but have effective resistance between diagonal nodes that grows linearly, causing severe computational bottlenecks; graph rewiring improves topology but may worsen the computational bottleneck.
3. **Heterophily ≠ Difficulty**: Purely heterophilous graphs can be perfectly classified by a 1-layer DGN; highly homophilous graphs may still require long-range propagation.

## Highlights & Insights

- Explicitly **decouples** "oversquashing" into computational and topological bottlenecks for the first time, which is highly instructive.
- Counter-examples are concise and formal (e.g., bipartite graph classification, grid graph bottlenecks), making them easy to remember and share.
- The categorization of the nine beliefs serves as a "debugging checklist" or reference manual for graph ML researchers.
- Urges the community to use terminology more precisely and avoid over-generalization in arguments.

## Limitations

- Primarily concept-clearing and counter-example driven; does not propose new solutions or models.
- Most counter-examples are based on synthetic small graphs; real-world large-scale graphs are more complex.
- Being "meta-research" for the graph ML community, its direct utility for other fields is limited.
- "Refuting" some beliefs essentially means showing they "do not always hold" rather than "never hold"—boundary conditions still require deeper exploration.

## Related Work & Insights

- **Oversmoothing**: Cai & Wang (2020) proposed the Dirichlet Energy metric; Roth & Liebig (2024) analyzed the Rayleigh Quotient; Zhang et al. (2025) noted that OSM in untrained networks does not reflect behavior after training.
- **Oversquashing**: Topping et al. (2022) linked it to curvature and Jacobian sensitivity; Errica et al. (2025) proposed message filtering to reduce computational bottlenecks.
- **Heterophily**: Ma et al. (2022) proposed metrics linking heterophily to discriminability; Platonov et al. (2023) built more rigorous heterophily benchmarks.
- **Insights**: Future work should clearly distinguish what is being measured (DE vs RQ vs separability), the type of bottleneck (computational vs topological), and the relationship between task and graph properties.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Systematizes scattered confusion into nine beliefs with formal refutations; strong conceptual contribution.
- **Value**: ⭐⭐⭐ — High value in conceptual clarification but offers no new methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logic, refined counter-examples, and excellent summary tables.
- **Impact**: ⭐⭐⭐⭐ — Likely to have a long-term impact on terminology and problem definition within the graph ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OSIRIS: Bridging Analog Circuit Design and Machine Learning with Scalable Dataset Generation](osiris_bridging_analog_circuit_design_and_machine_learning_with_scalable_dataset.md)
- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[CVPR 2026\] Learning Long-term Motion Embeddings for Efficient Kinematics Generation](../../CVPR2026/others/learning_long-term_motion_embeddings_for_efficient_kinematics_generation.md)
- [\[CVPR 2026\] FedSDR: Federated Graph Learning with Structural Noise Detection and Reconstruction](../../CVPR2026/others/fedsdr_federated_graph_learning_with_structural_noise_detection_and_reconstructi.md)

</div>

<!-- RELATED:END -->
