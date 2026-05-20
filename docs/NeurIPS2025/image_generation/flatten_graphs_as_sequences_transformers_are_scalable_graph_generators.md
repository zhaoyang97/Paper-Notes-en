---
title: >-
  [Paper Note] Flatten Graphs as Sequences: Transformers are Scalable Graph Generators
description: >-
  [NeurIPS 2025][Image Generation][graph generation] This paper proposes AutoGraph, which losslessly flattens graphs into token sequences via Segmented Eulerian Neighborhood Trails (SENT)…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "graph generation"
  - "autoregressive model"
  - "Eulerian trail"
  - "language model"
  - "scalability"
date: 2026-05-08
content_hash: a954763e56bac504
---

# Flatten Graphs as Sequences: Transformers are Scalable Graph Generators

**Conference**: NeurIPS 2025
**arXiv**: [2502.02216](https://arxiv.org/abs/2502.02216)  
**Code**: [AutoGraph](https://github.com/BorgwardtLab/AutoGraph)  
**Area**: Graph Generation
**Keywords**: graph generation, autoregressive model, Eulerian trail, language model, scalability

## TL;DR

This paper proposes AutoGraph, which losslessly flattens graphs into token sequences via Segmented Eulerian Neighborhood Trails (SENT), enabling direct modeling with a decoder-only Transformer. AutoGraph achieves graph generation speeds ~100× faster than diffusion models while reaching state-of-the-art performance on both synthetic and molecular benchmarks.

## Background & Motivation

Graph generation is a fundamental task in drug discovery, protein design, and program synthesis. The core bottlenecks of existing approaches are:

- **Diffusion models**: Require full adjacency matrix operations ($O(n^2)$ memory) and spectral feature computation at each step ($O(n^3)$), severely limiting scalability.
- **Traditional autoregressive models** (e.g., GraphRNN): Sequences are not composed of tokens and require specialized architectures (RNNs), precluding the use of powerful LLMs; moreover, sequence prefixes are not guaranteed to correspond to induced subgraphs, leading to loss of structural information.
- The graph generation field has not benefited from the advances in Transformers and large-scale pretraining to the same extent as text or image generation.

### Solution Approach

**Goal**: ### Core Concept: Segmented Eulerian Neighborhood Trail (SENT)

**Segmented Eulerian Trail (SET)**: Decomposes a graph into multiple trail segments, visiting each edge exactly once.

## Method

### Core Concept: Segmented Eulerian Neighborhood Trail (SENT)

**Segmented Eulerian Trail (SET)**: Decomposes a graph into multiple trail segments, visiting each edge exactly once. However, prefixes of a SET yield subgraphs that are not necessarily induced subgraphs, leading to loss of structural information.

**Segmented Eulerian Neighborhood Trail (SENT)**: Extends SET by appending neighborhood information to each node. Each tuple $w_i = (v_i, A_i)$ encodes node $v_i$ and its neighborhood set $A_i \subseteq V$.

**Key Theorem**: If a SENT $s$ is causal and semi-Hamiltonian, then any prefix of $s$ generates an induced subgraph of $G_s$. This establishes a direct correspondence between graph prefixes and clauses in language modeling.

### Necessary and Sufficient Conditions (Theorem 2.15)

A SENT $s$ is causal and Hamiltonian in graph $G$ if and only if each tuple $w = (v, A_v)$ satisfies:

$$A_v = \mathcal{N}_G(v) \cap V_s(w)$$

That is, the neighborhood set is exactly the graph neighbors of $v$ among previously visited nodes.

### Sampling Algorithm

Based on random path sampling (Algorithm 1):
1. Randomly select a starting node and initialize the trail.
2. If the current node has unvisited neighbors, randomly select one to extend the trail.
3. Otherwise, break the current segment and randomly select a new unvisited node to begin a new segment.
4. For each new node, compute its neighborhood set with respect to previously visited nodes.

Both time and space complexity are $\mathcal{O}(m)$, where $m$ is the number of edges — **optimal linear complexity**.

### Tokenization

1. **Re-indexing**: Nodes are renumbered in order of first appearance, ensuring permutation invariance.
2. **Special tokens**: `/` denotes a segment break; `<` and `>` delimit the neighborhood set.
3. **Attributed graphs**: Node and edge attributes are inserted into the token sequence in an interleaved fashion.

### Autoregressive Modeling

Standard language modeling objective:

$$p(s) = \sum_{i=1}^{n} \log p_\theta(s_i \mid s_1, \ldots, s_{i-1})$$

The LLaMA architecture is adopted (12 layers, hidden dim=768), aligned with GPT-2 smallest. Top-$k$ sampling is used during inference.

## Key Experimental Results

### Synthetic Graph Generation (Planar & SBM)

### Main Results

| Model | Planar Ratio↓ | Planar VUN↑ | SBM Ratio↓ | SBM VUN↑ |
|-------|--------------|-------------|-----------|----------|
| DiGress | 6.1% | 77.5% | 1.8 | 60.0% |
| GruM | 1.8 | 90.0% | 1.5 | 85.0% |
| GEEL | 9.5% | 0.0 | 7.3% | 5.0% |
| **AutoGraph** | **1.5** | **87.5%** | **3.4** | **92.5%** |

- AutoGraph achieves the highest VUN (Valid+Unique+Novel) on the SBM dataset.
- Only AutoGraph and GruM achieve VUN ≥ 80% on both datasets.

### Efficiency Comparison

### Ablation Study

| Model | Inference (s/graph) | Training (h) |
|-------|--------------------:|-------------:|
| DiGress | 7.53% | 5.56% |
| GRAN | 1.25 | 12.3% |
| **AutoGraph** | **0.08** | **1.57** |

AutoGraph is ~100× faster than diffusion models at inference and ~3× faster during training.

### Large-Scale Graphs (Proteins & Point Clouds)

| Model | Proteins Ratio↓ | Point Clouds Ratio↓ |
|-------|----------------|---------------------|
| GRAN | 77.7% | 19.1% |
| DiGress | OOM | OOM |
| **AutoGraph** | Close to training set | Scalable to 5000+ nodes |

Diffusion models run out of memory on large graphs, while AutoGraph scales due to its linear complexity.

### Molecular Generation

AutoGraph achieves state-of-the-art or competitive performance on the QM9 and MOSES molecular benchmarks. It additionally supports **substructure-conditioned generation** (without fine-tuning, analogous to prompting) and **transfer learning**.

## Rating

⭐⭐⭐⭐⭐

**Strengths**:
- Theoretically elegant: The SENT design establishes a rigorous correspondence between graph generation and language modeling, where prefix = induced subgraph.
- Optimal complexity: Sequence length and sampling complexity are linear in the number of edges, significantly better than the quadratic/cubic complexity of diffusion models.
- High practical value: Directly leverages existing LLM infrastructure (LLaMA + HuggingFace) for graph generation.
- Supports substructure-conditioned generation and transfer learning, providing a viable path toward graph foundation models.
- The ~100× inference and ~3× training speedups make large-scale graph generation practically feasible.

**Limitations**:
- The sequence diversity introduced by random path sampling increases learning difficulty and may require more training data.
- Validation is currently limited to undirected graphs; directed graphs, hypergraphs, etc. require further extension.
- VUN on very small graphs (nodes < 10) may be inferior to diffusion models.
- Value:

## Highlights & Insights
- The method is concise and effective, with a clear core mechanism.
- Experimental validation is comprehensive and ablation analysis is thorough.
- The paper provides a novel solution to a key problem in the field.

## Limitations & Future Work
- The method may have limitations under certain conditions; generalizability warrants further investigation.
- Computational efficiency and scalability can be further optimized.
- Integration with additional related methods is worth exploring.

## Related Work & Insights
- **vs. representative methods in the field**: This paper makes a unique methodological contribution that is complementary to existing approaches.
- **vs. traditional methods**: The proposed method achieves significant improvements on key metrics compared to conventional solutions.
- **Insights**: The technical approach offers an important reference point for subsequent related work.

## Background & Motivation

### State of the Field
**Background**: The field addressed in this work has seen significant progress in recent years, with numerous methods proposed to tackle core challenges.

### Limitations of Prior Work
**Limitations of Prior Work**: Existing methods exhibit notable deficiencies in effectiveness, efficiency, or generalizability, limiting their practical applicability.

**Key Challenge**: A fundamental trade-off exists between performance and efficiency, necessitating innovation at the methodological level.

### Research Goals
**Goal**: To propose a novel methodological framework that addresses the core challenges described above.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Video Motion Graphs](../../ICCV2025/image_generation/video_motion_graphs.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](graph_diffusion_that_can_insert_and_delete.md)
- [\[NeurIPS 2025\] Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning](graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](scaling_diffusion_transformers_efficiently_via_μp.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)

</div>

<!-- RELATED:END -->
