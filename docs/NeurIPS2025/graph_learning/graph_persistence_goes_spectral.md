---
title: >-
  [Paper Note] Graph Persistence goes Spectral
description: >-
  [NeurIPS 2025][Graph Learning][Persistent Homology] This paper proposes SpectRe — a novel topological descriptor that incorporates graph Laplacian spectral information into persistent homology (PH) graphs. It proves that…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Persistent Homology"
  - "Graph Laplacian Spectrum"
  - "GNN Expressivity"
  - "SpectRe"
  - "Weisfeiler-Leman"
date: 2026-05-08
content_hash: 1d333072d3291dba
---

# Graph Persistence goes Spectral

**Conference**: NeurIPS 2025
**arXiv**: [2506.06571](https://arxiv.org/abs/2506.06571)
**Code**: [GitHub](https://github.com/Aalto-QuML/SpectRe/)
**Area**: Graph Learning / Topological Data Analysis / Graph Representation Learning
**Keywords**: Persistent Homology, Graph Laplacian Spectrum, GNN Expressivity, SpectRe, Weisfeiler-Leman

## TL;DR
This paper proposes SpectRe — a novel topological descriptor that incorporates graph Laplacian spectral information into persistent homology (PH) graphs. It proves that SpectRe is strictly more expressive than either PH or spectral methods alone, establishes a local stability theory, and demonstrates improved GNN graph classification performance on both synthetic and real-world datasets.

## Background & Motivation

**Background**: The expressivity of message-passing GNNs is bounded by the WL hierarchy, preventing them from computing topological properties such as cycles and connectivity. Persistent homology (PH) tracks the evolution of topological features (e.g., merging of connected components, emergence of cycles) via filtration functions, providing information beyond the WL hierarchy.

**Limitations of Prior Work**:
- Classical PH diagrams record only (birth, death) pairs, limiting their expressivity.
- RePHINE enhances PH by appending vertex color information $(\alpha, \gamma)$, but degenerates to ordinary PH on monochromatic graphs — still failing to distinguish certain non-isomorphic graphs.
- PH captures only harmonic information (the kernel of the graph Laplacian = number of connected components), while ignoring the rich non-harmonic spectral information.

**Key Challenge**: PH and spectral methods each have blind spots — PH detects cycles but ignores the spectrum; spectral methods capture global structure but do not track the evolution of topological features. The two are incomparable.

**Goal**: Design a new topological descriptor that integrates spectral information into the PH framework, yielding strictly greater expressivity than either PH or spectral methods alone.

**Key Insight**: Augment the RePHINE persistence tuple with a fifth component $\rho(v)$ — the list of nonzero eigenvalues of the graph Laplacian of the connected component containing vertex $v$ at the moment of its death. This embeds spectral information into the temporal evolution framework of PH.

**Core Idea**: Enrich each persistence pair in persistent homology with the Laplacian spectrum of the corresponding subgraph, achieving strictly stronger graph discriminative power.

## Method

### Overall Architecture
SpectRe = RePHINE + Graph Laplacian Spectrum. Given a colored graph and a filtration function $(f_v, f_e)$, SpectRe produces a quintuple $(b, d, \alpha, \gamma, \rho)$ for each vertex, where $\rho$ is the list of nonzero eigenvalues of the graph Laplacian of the connected component to which the vertex belongs at the moment of its "death" (i.e., when it is merged into another component).

### Key Designs

1. **SpectRe Descriptor**:

   - **Function**: Extends the RePHINE quadruple $(b, d, \alpha, \gamma)$ with a fifth component $\rho(v)$.
   - **Mechanism**: $\rho(v)$ records the nonzero eigenvalues of the graph Laplacian of the connected component containing vertex $v$ at the moment of its death in the edge filtration. For 1-dimensional persistent homology (cycles), $\rho(e)$ records the spectrum of the component in which edge $e$ creates a cycle.
   - **Design Motivation**: Classical PH knows only that two components merge, but not the structural details of each component prior to merging; spectral information precisely characterizes these structures.

2. **Expressivity Analysis**:

   - **SpectRe ≻ RePHINE**: There exist monochromatic graphs (e.g., Figure 3a) that RePHINE cannot distinguish but SpectRe can (due to differing spectra).
   - **SpectRe ≻ LS (Laplacian Spectrum)**: There exist graphs (e.g., Figure 3b) that LS cannot distinguish but SpectRe can (due to the additional information provided by the color-based filtration).
   - **Persistent Laplacians Do Not Further Enhance Expressivity** (Proposition 3.4): Replacing the graph Laplacian spectrum with the persistent Laplacian spectrum of Wang et al. does not increase expressivity.

3. **Stability Theory**:

   - New bottleneck distance metrics are defined for RePHINE and SpectRe, extending the classical PH bottleneck distance to incorporate the $\alpha$, $\gamma$, and $\rho$ components.
   - **RePHINE is Globally Stable**: Small perturbations in the filtration function lead to small changes in the RePHINE diagram.
   - **SpectRe is Locally Stable**: Stability holds under the condition that the filtration function is injective; global stability does not hold in general (counterexample: the spectrum may jump discontinuously when the filtration function crosses a non-injective point). An explicit upper bound on the degree of instability is provided, related to the complexity of the graph.

4. **Integration with GNNs**:

   - SpectRe diagrams are computed at each GNN layer.
   - DeepSets encode the $(b, d, \alpha, \gamma)$ components and $\rho$ separately (a dedicated DeepSet handles the spectral features).
   - The vectorized topological embeddings are concatenated to the graph-level GNN embeddings.

## Key Experimental Results

### Main Results: Synthetic Data (Graph Isomorphism Tests)

| Method | Cayley c-12 | Cayley c-60 | BREC All (400) |
|--------|-------------|-------------|----------------|
| PH0 | 0.67 | 0.69 | 0.03 (12/400) |
| PH1 | 0.95 | 0.78 | 0.41 (164/400) |
| RePHINE | 0.95 | 0.78 | 0.41 |
| LS | - | - | higher |
| **SpectRe** | **1.00** | **1.00** | **best** |

SpectRe achieves 100% discrimination on all Cayley graphs and ranks as the best-performing method on the BREC benchmark.

### Key Findings
- **SpectRe Strictly Improves Discriminative Power**: On monochromatic Cayley graphs, PH and RePHINE fail to fully distinguish non-isomorphic graphs, whereas SpectRe succeeds in all cases.
- **GNN Performance Improves After Integration**: On multiple real-world datasets, SpectRe-augmented GNNs outperform versions augmented with PH or RePHINE alone.
- **Local Stability Suffices in Practice**: Despite the theoretical absence of global stability, no instability issues are observed in experiments.

## Highlights & Insights
- **Theoretically Complete Expressivity Hierarchy**: A full relationship diagram is established: SpectRe ≻ RePHINE ≻ PH, SpectRe ≻ LS, and RePHINE is incomparable with LS.
- **Unification of Harmonic and Non-Harmonic Information**: PH captures harmonic information (Betti numbers = dimension of the Laplacian kernel), while SpectRe additionally captures non-harmonic spectral information, achieving a complete integration of topological and spectral information.
- **Novel Stability Concept**: Classical PH stability theory relies on global bottleneck distances; the local stability notion introduced in this paper is better suited to practical applications.

## Limitations & Future Work
- **Computational Cost of Spectral Computation**: Computing the Laplacian eigenvalues of the corresponding subgraph for each persistence pair may be slow on large graphs or under dense filtrations.
- **Absence of Global Stability**: Although local stability is sufficient in practice, the lack of global stability theoretically limits certain applications.
- **Choice of Filtration Function**: Experiments primarily employ vertex degree and Forman-Ricci curvature as filtration functions; the selection of optimal filtration functions remains an open problem.
- **Imprecise Positioning within the WL Hierarchy**: While SpectRe+GNN is known to surpass 1-WL, its precise position within the $k$-WL hierarchy has not been established.

## Related Work & Insights
- **vs. RePHINE**: SpectRe is strictly more expressive; spectral information resolves the degradation of RePHINE on monochromatic graphs.
- **vs. PH-GNN Methods (e.g., TOGL)**: SpectRe is compatible with existing PH-GNN integration frameworks, requiring only replacement of the PH computation component.
- **vs. Spectral GNN Methods**: SpectRe embeds spectral information within the temporal evolution framework of persistent homology, rather than using it directly as positional encodings in GNNs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of integrating spectral information into PH is both novel and natural, with thorough theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic data comprehensively validates expressivity; real-world data confirms practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[NeurIPS 2025\] GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)
- [\[NeurIPS 2025\] Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Generative Graph Pattern Machine](generative_graph_pattern_machine.md)

</div>

<!-- RELATED:END -->
