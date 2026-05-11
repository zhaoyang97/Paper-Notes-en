---
title: >-
  [Paper Note] Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning
description: >-
  This paper systematically examines nine common beliefs in graph machine learning concerning oversmoothing, oversquashing, homophily/heterophily, and long-range dependencies. Through concise counterexamples…
tags:

date: 2026-05-08
content_hash: 3db50d06024f61bc
---

# Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning

- **Conference**: ICLR2026
- **arXiv**: [2505.15547](https://arxiv.org/abs/2505.15547)
- **Code**: [GitHub](https://github.com/AdrianArnaiz/demystifyingGraphML)
- **Area**: Others
- **Keywords**: oversmoothing, oversquashing, heterophily, long-range dependencies, message-passing, GNN

## TL;DR

This paper systematically examines nine common beliefs in graph machine learning concerning oversmoothing, oversquashing, homophily/heterophily, and long-range dependencies. Through concise counterexamples, each belief is refuted. Notably, "oversquashing" is decomposed into two independent concepts—**computational bottleneck** and **topological bottleneck**—thereby clarifying widespread conceptual confusion in the field.

## Background & Motivation

**Background**: Message-passing deep learning on graphs has undergone rapid development, with researchers focusing on its inherent limitations—oversmoothing (OSM, where node representations collapse to similar values), oversquashing (OSQ, where information is compressed and lost), the influence of homophily/heterophily on classification, and long-range information propagation.

**Limitations of Prior Work**: Due to the fast pace of progress, numerous "commonly accepted beliefs" have been widely disseminated and cited without sufficient validation, causing conceptual confusion and ill-defined problem formulations that significantly hinder targeted follow-up research.

**Key Challenge**: Claims such as treating oversmoothing as the fundamental cause of performance degradation, equating oversquashing directly with topological bottlenecks, or identifying heterophily with "difficulty" are not universally valid, yet have become near-consensus conclusions in the literature.

**Goal**: To explicitly identify the limitations of these beliefs, refute them through clear counterexamples, and enable researchers to distinguish and precisely define the problems they seek to address.

**Key Insight**: Rather than criticizing specific works, the paper distills ambiguous claims scattered across the literature into nine common beliefs and systematically "demystifies" each using mathematical definitions and counterexamples.

**Core Idea**: "Oversquashing" should be decomposed into two independent problems: the **computational bottleneck** (arising from the exponential expansion of computation trees) and the **topological bottleneck** (arising from graph connectivity). These two phenomena can exist independently and need not be correlated.

## Method

### Overall Architecture

This paper is a **survey, analysis, and refutation** work organized around nine beliefs across three major themes (OSM, homophily/heterophily, OSQ):

| Theme | Beliefs |
|-------|---------|
| OSM | 1. OSM causes performance degradation; 2. OSM is an inherent property of all DGNs |
| Homophily/Heterophily | 3. Homophily is beneficial, heterophily is harmful; 4. Long-range propagation is evaluated on heterophilic graphs; 5. Different classes imply different features |
| OSQ | 6. OSQ = topological bottleneck; 7. OSQ = computational bottleneck; 8. OSQ is harmful for long-range tasks; 9. Topological bottlenecks correlate with long-range problems |

### Key Designs

#### 1. Oversmoothing Is Neither Universal Nor the Root Cause of Performance Degradation

**Function**: Demonstrates that whether OSM occurs and how it is measured is highly dependent on architectural choices and hyperparameters.

**Mechanism**: Two metrics are used to quantify OSM—Dirichlet Energy (DE) and Rayleigh Quotient (RQ):

$$\mathrm{DE}(\mathbf{H}^\ell) = \mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell), \quad \mathrm{RQ} = \frac{\mathrm{Tr}((\mathbf{H}^\ell)^T \mathbf{L} \mathbf{H}^\ell)}{\|\mathbf{H}^\ell\|_F^2}$$

Experiments show: (i) the DE of GIN **explodes** rather than collapses under standard settings; (ii) simply scaling the weight matrix by 2 ($AX(2W)$) reverses the DE trend; (iii) DE and RQ frequently yield contradictory conclusions on the same model. OSM is therefore neither universal nor uniquely defined.

**Design Motivation**: Even when DE decreases, the **class separability** of node embeddings may be maintained or even improved—nodes within each class first collapse to distinct class-specific points ("beneficial smoothing" phase). Actual performance degradation is more attributable to vanishing gradients and overfitting.

#### 2. Decomposing Oversquashing into Computational and Topological Bottlenecks

**Function**: Provides a rigorous definition of the computational bottleneck and demonstrates that it can exist independently of the topological bottleneck.

**Mechanism**: The **computational bottleneck** is defined via the multiset size of the computation tree $|\mathcal{M}_v^K|$:

$$\mathcal{M}_v^K := \mathcal{M}_v^{K-1} \uplus \left\{\biguplus_{u \in \mathcal{M}_v^{K-1}} \mathcal{N}_u\right\}$$

This grows exponentially with depth, independently of whether the graph exhibits a topological bottleneck. Counterexamples: (a) a grid graph has no topological bottleneck but suffers a severe computational bottleneck; (b) a dumbbell graph with a topological bottleneck has a mild computational bottleneck at shallow depths.

**Design Motivation**: Existing rewiring methods alleviate the topological bottleneck, but adding edges or nodes can **worsen** the computational bottleneck. Conversely, message filtering methods can reduce the computational bottleneck without modifying the graph structure.

#### 3. Decoupling Homophily/Heterophily from Task Difficulty

**Function**: Demonstrates that high heterophily does not imply "difficulty," and that long-range propagation tasks need not occur on heterophilic graphs.

**Mechanism**: (a) In a fully heterophilic bipartite graph, differing node degrees suffice for a 1-layer sum-based DGN to achieve perfect classification; (b) on a highly homophilic graph, the task of determining whether the distance from a node to a specific node exceeds 5 requires long-range propagation—yet the graph is homophilic. Therefore, **long-range tasks ⊥ heterophily**.

### Loss & Training

This paper is an analytical work and proposes no new model. Experiments replicate standard DGN training using cross-entropy loss for node classification.

## Key Experimental Results

### Main Results: OSM Metrics Across Architectures (Cora, 50 seeds)

| Architecture | W: DE Trend | 2W: DE Trend | W: RQ Trend | 2W: RQ Trend |
|--------------|------------|--------------|------------|--------------|
| GCN | Collapse→0 | Explosion↑ | Stable | Stable |
| GAT | Collapse→0 | Explosion↑ | Linear decay | Stable |
| SAGE | Collapse→0 | Explosion↑ | Linear decay | Partially stable |
| GIN | Explosion↑ | Explosion↑↑ | Linear decay | Partially stable |

> Conclusion: The same model can exhibit completely contradictory trends across the two metrics. Observation of OSM is highly sensitive to the choice of measure and hyperparameters.

### Ablation Study: Performance Degradation vs. DE With/Without Bias

| Configuration | DE as Depth Increases | Accuracy as Depth Increases |
|---------------|-----------------------|-----------------------------|
| GCN (with bias) | Collapse | Decreases |
| GCN (without bias) | **No collapse** | Still decreases |

> This demonstrates that performance degradation cannot be attributed to OSM—removing bias eliminates DE collapse, yet accuracy still decreases. The true causes are vanishing gradients and overfitting.

### Key Findings

1. **OSM is not universal**: Changing the aggregation function (GIN), adjusting weight scaling (2W), or switching metrics (DE vs. RQ) all alter the conclusion.
2. **Computational bottleneck ≠ Topological bottleneck**: A grid graph has no topological bottleneck, yet effective resistance between diagonal nodes grows linearly, indicating a severe computational bottleneck; graph rewiring improves the topological bottleneck but may worsen the computational one.
3. **Heterophily ≠ Difficulty**: Fully heterophilic graphs can be perfectly classified by a 1-layer DGN; highly homophilic graphs may still require long-range propagation.

## Highlights & Insights

- The paper is the first to **explicitly decompose** "oversquashing" into computational and topological bottlenecks as independent concepts, offering substantial conceptual contribution.
- Counterexamples are concise and formalized—e.g., perfect classification on bipartite graphs, and severe computational bottleneck on grid graphs despite the absence of topological bottlenecks—making them easy to remember and disseminate.
- The nine-belief structure functions as a "debugging checklist" and can serve as a reference handbook for graph ML researchers.
- The paper calls on the community to use terminology more precisely and avoid overgeneralization.

## Limitations & Future Work

- The work is primarily conceptual clarification and counterexample-driven; no new solutions or models are proposed.
- Most counterexamples are based on small, artificially constructed graphs; behavior on large-scale real-world graphs is more complex.
- As a "meta-research" effort targeting the graph ML community, its direct practical value in other domains is limited.
- Some refutations are essentially arguments that a belief "does not always hold" rather than "never holds"—boundary conditions warrant further investigation.

## Related Work & Insights

- **Oversmoothing**: Cai & Wang (2020) propose Dirichlet Energy as a measure; Roth & Liebig (2024) analyze the Rayleigh Quotient; Zhang et al. (2025) show that OSM in untrained networks does not reflect behavior after training.
- **Oversquashing**: Topping et al. (2022) connect oversquashing to curvature and Jacobian sensitivity; Errica et al. (2025) propose message filtering to reduce the computational bottleneck.
- **Heterophily**: Ma et al. (2022) propose new metrics linking heterophily to node distinguishability; Platonov et al. (2023) construct more rigorous heterophily benchmarks.
- **Insights**: Future work should clearly distinguish the object of measurement (DE vs. RQ vs. separability), the type of problem (computational vs. topological), and the relationship between task requirements and graph properties.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Systematizing scattered confusions into nine formalized beliefs with rigorous refutations represents a strong conceptual contribution.
- **Value**: ⭐⭐⭐ — High value for conceptual clarification, but no new methods are proposed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logically clear, counterexamples are crisp, and summary tables are immediately interpretable.
- **Impact**: ⭐⭐⭐⭐ — Likely to have lasting influence on terminology usage and problem formulation in the graph ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_long_horizon_planning.md)
- [\[AAAI 2026\] Life, Machine Learning, and the Search for Habitability: Predicting Biosignature Fluxes for the Habitable Worlds Observatory](../../AAAI2026/others/life_machine_learning_and_the_search_for_habitability_predicting_biosignature_fl.md)
- [\[NeurIPS 2025\] Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning](../../NeurIPS2025/others/directional_non-commutative_monoidal_structures_for_compositional_embeddings_in_.md)

</div>

<!-- RELATED:END -->
