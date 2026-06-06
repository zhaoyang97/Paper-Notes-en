---
title: >-
  [Paper Note] Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond
description: >-
  [ICLR 2026][LLM Evaluation][long-range propagation] This paper proposes the ECHO benchmark, comprising 3 synthetic tasks and 2 real-world chemistry tasks grounded in density functional theory (DFT)…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "long-range propagation"
  - "graph benchmark"
  - "over-squashing"
  - "graph transformers"
  - "molecular property prediction"
date: 2026-05-08
content_hash: ce75cdc4e742812e
---

# Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond

**Conference**: ICLR 2026
**arXiv**: [2512.17762](https://arxiv.org/abs/2512.17762)  
**Code**: [GitHub](https://github.com/Graph-ECHO-Benchmark/ECHO)  
**Area**: LLM Evaluation
**Keywords**: long-range propagation, graph benchmark, over-squashing, graph transformers, molecular property prediction

## TL;DR

This paper proposes the ECHO benchmark, comprising 3 synthetic tasks and 2 real-world chemistry tasks grounded in density functional theory (DFT), requiring graph neural networks to propagate information effectively over 17–40 hops. The benchmark systematically evaluates the long-range propagation capabilities of 11 GNN architectures.

## Background & Motivation

**The long-range propagation challenge in GNNs**: Capturing dependencies between distant nodes in a graph is a fundamental challenge in GNN research, closely related to phenomena such as oversmoothing, over-squashing, and vanishing gradients.

**Limitations of existing benchmarks**:
- LRGB (Dwivedi et al., 2022) is the most widely used long-range benchmark, but performance has saturated and some of its tasks have been shown to be inherently local rather than long-range.
- Graph Property Prediction (Gravina et al., 2023) uses small graphs with limited diameter.
- Existing benchmarks primarily target either oversmoothing or over-squashing in isolation, without comprehensively evaluating long-range propagation.

**Demand from real-world applications**: In molecular property prediction, atomic charge distributions and total molecular energies inherently depend on long-range quantum mechanical effects, yet existing molecular benchmarks (ZINC, MoleculeNet) are predominantly short-range tasks.

**Model maturity**: Architectures such as Graph Transformers, multi-hop GNNs, and differential-equation-inspired GNNs have matured, necessitating more challenging benchmarks to differentiate their long-range propagation capabilities.

**Scientific rigor in evaluation**: There is a need for a quantitative definition of "long-range" and for ensuring that tasks genuinely require global information propagation.

## Method

### Overall Architecture

The ECHO (Evaluating Communication over long HOps) benchmark consists of two components:
- **ECHO-Synth**: 3 synthetic tasks — Single-Source Shortest Path (SSSP), Node Eccentricity (ECC), and Graph Diameter (DIAM) — covering 6 graph topologies across 10,080 graphs.
- **ECHO-Chem**: 2 chemistry tasks — atomic charge prediction (ECHO-Charge) and molecular energy prediction (ECHO-Energy) — computed via DFT, covering approximately 170K–196K molecular graphs.

All tasks feature graph diameters in the range of 17–40, ensuring propagation ranges far exceeding those of existing benchmarks.

### Key Designs

1. **Topology Design for Synthetic Tasks**

    - **Function**: Six graph topologies with structural bottlenecks are designed: line, ladder, grid, tree, caterpillar, and lobster graphs.
    - **Mechanism**: Each topology introduces a distinct type of information propagation bottleneck. In line graphs, information must propagate sequentially; in trees, branching points act as bottlenecks; in grids, symmetry is broken via random edge deletion (with probability 20%).
    - **Design Motivation**: Structural bottlenecks exacerbate over-squashing, making tasks intractable for GNNs relying solely on local message passing.

2. **Selection of Graph Property Prediction Tasks**

    - **Function**: Three tasks — SSSP, ECC, and DIAM — are selected.
    - **Mechanism**: These tasks exhibit a progressive degree of dependence on global information. SSSP requires shortest paths from a source node to all others; ECC requires the longest shortest path from each node; DIAM requires the longest shortest path between any two nodes in the entire graph.
    - **Design Motivation**: Classical algorithms such as Dijkstra and Bellman-Ford require full graph traversal to solve these problems, thereby testing whether GNNs can learn to simulate such global traversal.

3. **Chemical Foundation of ECHO-Chem**

    - **Function**: Datasets of atomic charges and molecular energies are constructed based on DFT computations.
    - **Mechanism**: Molecules with diameters between 17 and 40 are filtered from the ChEMBL database; 3D structures are optimized using the GAFF force field, and ground-truth labels are obtained via DFT calculations using the ORCA quantum chemistry package.
    - **Design Motivation**: In quantum mechanics, charge redistribution and molecular energy inherently depend on long-range electron–nucleus and electron–electron interactions, providing a natural source of long-range dependencies. Approximately two months of parallel DFT computation ensures data accuracy.

4. **Minimalist Node Feature Design**

    - **Function**: In synthetic tasks, each node is assigned only a uniformly sampled scalar feature $r \sim \mathcal{U}(0,1)$.
    - **Mechanism**: For SSSP, the source node is additionally marked with a binary indicator.
    - **Design Motivation**: This prevents models from exploiting feature shortcuts, ensuring that performance differences reflect structural awareness.

5. **E(3)-Invariant Molecular Encoding**

    - **Function**: In chemistry tasks, node features consist of atomic number and distance to the molecular centroid; edge features consist of bond type and bond length.
    - **Mechanism**: Spatial encodings are invariant under the E(3) group (rotations, reflections, and translations).
    - **Design Motivation**: This ensures that spatial representations respect the physical symmetries of molecules.

### Loss & Training

- Loss function: $\log_{10}(\text{MSE}(y_{\text{true}} - y_{\text{pred}}))$, using a logarithmic scale due to potentially small prediction values.
- Optimizer: Adam.
- Early stopping: based on validation loss, with a patience of 50 epochs and a maximum of 1,000 epochs.
- Hyperparameter search: Bayesian Optimization (Gaussian Process prior) with 100 trials.
- The optimal configuration is repeated across 4 random seeds, and results are averaged.

## Key Experimental Results

### Main Results

ECHO-Synth — three synthetic tasks (Test MAE, lower is better):

| Model | DIAM ↓ | ECC ↓ | SSSP ↓ |
|------|--------|-------|--------|
| **GRIT** | **1.014** | 5.091 | **0.121** |
| SWAN | 1.121 | 4.840 | 0.896 |
| A-DGN | 1.151 | 4.981 | 1.176 |
| **DRew** | 1.243 | **4.651** | 1.279 |
| GPS | 2.160 | 4.758 | 0.472 |
| GCN | 3.832 | 5.233 | 2.102 |
| GraphCON | 2.969 | 5.474 | 5.734 |

ECHO-Chem — two chemistry tasks (Test MAE, lower is better):

| Model | ECHO-Energy ↓ | ECHO-Charge (×10⁻³) ↓ |
|------|--------------|----------------------|
| **GPS** | **5.257** | 6.182 |
| DRew | 11.325 | 9.086 |
| A-DGN | 12.486 | 6.543 |
| **SWAN** | 12.629 | **6.109** |
| GCN | 28.112 | 8.421 |
| GIN | 47.851 | 10.784 |

### Ablation Study

In-depth analysis of the effect of depth/number of hops on performance:

| Analysis Dimension | Finding |
|---------|------|
| Network depth vs. performance | Deeper networks consistently outperform shallower ones, confirming the long-range nature of the tasks. |
| Graph diameter vs. performance | Larger-diameter graphs incur higher errors, confirming the challenge of long-range propagation. |
| Readout depth | Final performance is independent of readout depth; the bottleneck lies in the propagation layers. |
| Different topologies | Relative model rankings remain consistent across topologies. |
| GPS attention patterns | The highest attention scores are frequently assigned to distant, non-adjacent node pairs in the graph. |

### Key Findings

- **Global attention is critical**: GRIT achieves a MAE of only 0.121 on SSSP, far surpassing GCN's 2.102, confirming that Transformer-style global attention substantially alleviates the limitations of local message passing.
- **Non-dissipative dynamics are effective**: Differential-equation-based non-dissipative GNNs such as A-DGN and SWAN consistently perform well, indicating that preserving signal energy is essential for long-range propagation.
- **Mitigating oversmoothing alone is insufficient**: GraphCON, designed solely to address oversmoothing, performs worst on long-range tasks, demonstrating that long-range propagation requires dedicated mechanisms beyond preventing feature collapse.
- **Accuracy–efficiency trade-off**: Transformer-based models such as GPS achieve strong performance but at high computational cost; A-DGN offers a better cost-performance ratio.
- **Practical significance of chemistry tasks**: Even charge errors on the order of $10^{-4}$ to $10^{-6}$ e can affect downstream molecular modeling.

## Highlights & Insights

- **Rigor in task design**: Synthetic tasks directly correspond to classical graph algorithms (Dijkstra, Bellman-Ford), providing a precise physical interpretation of "long-range."
- **Complementary perspectives**: Synthetic tasks provide controlled theoretical tests, while chemistry tasks demonstrate the practical necessity of long-range propagation (with DFT computations requiring approximately two months).
- **Exposing LRGB's limitations**: ECHO's propagation range (17–40 hops) far exceeds that of LRGB, and its tasks cannot be solved via local substructure counting.
- **Fair experimental design**: All models share a unified backbone (embedding + GNN layers + MLP readout); differences arise solely from core propagation mechanisms.
- **Bayesian hyperparameter search**: With 100 trials per model–dataset pair, the search is more reliable than random or manual tuning.

## Limitations & Future Work

- The scale of synthetic task graphs remains limited (approximately 30 to hundreds of nodes per graph); extension to larger scales is possible.
- Chemistry tasks use only atomic number and distance features, without richer chemical descriptors such as bond angles or dihedral angles.
- Evaluation of graph structure learning (graph rewiring) methods is absent.
- Random node features in ECHO-Synth may be unfavorable for methods based on feature similarity.
- Molecular graph construction does not account for non-covalent interactions (e.g., hydrogen bonds, van der Waals forces), potentially underestimating actual long-range dependencies.
- Additional task types such as link prediction and graph generation could be incorporated for a more comprehensive evaluation of long-range propagation.

## Related Work & Insights

- **Dwivedi et al. (2022)**: The LRGB benchmark, currently the most widely used but already saturated; ECHO significantly surpasses it in propagation range and task difficulty.
- **Gravina et al. (2023)**: The GPP dataset, which uses small graphs and short distances; ECHO employs larger diameters and more diverse topologies.
- **Alon & Yahav (2021)**: Theoretical analysis of over-squashing; ECHO provides an empirical validation platform.
- **Rampášek et al. (2022)**: GPS Graph Transformer; experiments confirm its advantage on long-range chemistry tasks.
- **Insights**: The construction methodology of ECHO-Chem can be extended to other scientific domains requiring long-range modeling, such as protein interactions and crystalline materials; the consistency of performance across topologies points toward future directions in topology-adaptive GNN design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first comprehensive benchmark to explicitly quantify long-range propagation range (17–40 hops); chemistry tasks carry genuine scientific value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 11 models, 5 tasks, Bayesian hyperparameter search, and multiple analytical dimensions — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is well-argued and comparisons with existing benchmarks are clear, though some sections are verbose.
- **Value**: ⭐⭐⭐⭐⭐ Provides a much-needed standardized evaluation platform for GNN long-range propagation research; chemistry tasks hold strong prospects for AI for Science applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](../../ICML2026/llm_evaluation/beyond_trajectory-level_attribution_graph-based_credit_assignment_for_agentic_re.md)
- [\[NeurIPS 2025\] BLINK-Twice: You See But Do You Observe? A Reasoning Benchmark on Visual Perception](../../NeurIPS2025/llm_evaluation/blink-twice_you_see_but_do_you_observe_a_reasoning_benchmark_on_visual_perceptio.md)
- [\[CVPR 2026\] R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII](../../CVPR2026/llm_evaluation/r2g_multi_view_circuit_graph_benchmark_suite_from_rtl_to_gdsii.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)

</div>

<!-- RELATED:END -->
