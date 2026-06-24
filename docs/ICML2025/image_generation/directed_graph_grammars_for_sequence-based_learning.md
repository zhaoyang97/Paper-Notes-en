---
title: >-
  [Paper Note] Directed Graph Grammars for Sequence-based Learning
description: >-
  [ICML 2025][Image Generation][Directed Acyclic Graphs] This paper proposes DIGGED, which losslessly maps DAGs to unique sequences of production rules via unambiguous context-free graph grammars. Combined with a Transformer decoder, it achieves graph generation, property prediction, and Bayesian optimization, thoroughly outperforming existing methods on neural architecture search, Bayesian networks, and circuit design tasks.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Directed Acyclic Graphs"
  - "Graph Grammars"
  - "Serialized Representation"
  - "Minimum Description Length"
  - "Bayesian Optimization"
date: 2026-05-08
content_hash: e2fea921323cfdcb
---

# Directed Graph Grammars for Sequence-based Learning

**Conference**: ICML 2025  
**arXiv**: [2505.22949](https://arxiv.org/abs/2505.22949)  
**Code**: [https://github.com/shiningsunnyday/induction](https://github.com/shiningsunnyday/induction)  
**Area**: Graph Generation / Graph Representation Learning / Structure Search  
**Keywords**: Directed Acyclic Graphs, Graph Grammars, Serialized Representation, Minimum Description Length, Bayesian Optimization

## TL;DR

This paper proposes DIGGED, which losslessly maps DAGs to unique sequences of production rules via unambiguous context-free graph grammars. Combined with a Transformer decoder, it achieves graph generation, property prediction, and Bayesian optimization, thoroughly outperforming existing methods on neural architecture search, Bayesian networks, and circuit design tasks.

## Background & Motivation

**Background**: Directed acyclic graphs (DAGs) are core data structures for many practical problems, including neural architecture search (NAS), Bayesian network structure learning, and analog circuit design. Currently, mainstream graph generation methods fall into two categories: autoregressive generation (AG), which incrementally adds nodes and edges in a specific order; and sequence decoding (SD), which encodes the adjacency information of a graph into a sequence of node permutations.

**Limitations of Prior Work**: AG methods lack rigor—a single graph can correspond to exponentially many generation sequences, resulting in high decoding ambiguity. SD methods rely on arbitrary node ordering (e.g., topological order, BFS order), and arbitrary sequences cannot guarantee the decoding of valid graphs. Neither approach can simultaneously guarantee one-to-one mapping, determinism, validity, and statelessness.

**Key Challenge**: Graphs are permutation-invariant structures, and forcing a node order introduces redundancy and ambiguity. The lack of a strict bijection from graph space to sequence space prevents sequential models (such as Transformers) from being directly utilized for graph generation.

**Goal**: To design a DAG-to-sequence mapping that simultaneously satisfies five ideal properties: unambiguous (one-to-one), onto, deterministic, valid, and stateless.

**Key Insight**: The authors observe that context-free graph grammars (edNCE grammar) naturally support deterministic derivation and validity guarantees. If a grammar can be made unambiguous with respect to a dataset, each DAG will correspond to a unique derivation sequence, perfectly reducing the graph problem to a sequence problem.

**Core Idea**: Induce an unambiguous edNCE graph grammar from data using the Minimum Description Length (MDL) principle to losslessly compress DAGs into unique sequences of production rules, thereby completely converting graph modeling into sequence modeling.

## Method

### Overall Architecture

DIGGED consists of two phases: **grammar induction** and **sequence modeling**.

1. **Grammar Induction Phase**: Given a DAG dataset $\mathcal{D} = \{H_1, H_2, \ldots\}$, an iterative compression algorithm automatically detects recurring subgraph patterns (motifs), establishes a production rule set $P$, and eliminates ambiguity, ultimately parsing each DAG into a unique sequence of rules.
2. **Sequence Modeling Phase**: Taking the rule sequence as tokens, a VAE framework is constructed—the encoder can be a GNN (DAGNN) or a Transformer, while the decoder utilizes a causal-attention Transformer to autoregressively decode the rule sequence. Once trained, it supports three downstream tasks: unconditional generation, property prediction, and Bayesian optimization.

### Key Designs

1. **Frequent Subgraph Mining**

    - **Function**: Identifies recurring subgraph patterns (motifs) within the current composite graph $H$ (the union of all DAGs).
    - **Mechanism**: Employs Subdue, an approximate FSM library, to quickly identify candidate motifs, and then adopts a parallel subgraph isomorphism algorithm to locate all motif occurrences $D_1, D_2, \ldots, D_K$. Key novelty: for connected components containing non-terminal nodes, it only considers subgraphs containing that non-terminal node, thereby simplifying the parse tree into rooted paths, ensuring a linear parsing structure.
    - **Design Motivation**: A linear parse tree is the foundation for achieving an "unambiguous (one-to-one) mapping." Traditional graph grammars allow branching parse trees where a single graph may have multiple derivations, whereas path-like parse trees eliminate this structural ambiguity.

2. **Compatibility Maximization Solver**

    - **Function**: Finds the largest subset of occurrences for each candidate motif such that they can share the same set of connection instructions.
    - **Mechanism**: edNCE rules must define how to connect a motif to its neighborhood. Neighbor connection patterns at different occurrence locations may conflict. The authors model this as a **maximum clique problem**—each node represents a motif occurrence and its edge redirection scheme, with edges added between compatible schemes, and the maximum clique representing the maximum compatible set. Each node maintains an "inset" (instructions that must be included) and an "outset" (instructions that must be excluded), determining node validity via $v_\text{inset} \cap v_\text{outset} = \emptyset$, and edge compatibility via $(u_\text{inset} \cup v_\text{inset}) \cap (u_\text{outset} \cup v_\text{outset}) = \emptyset$.
    - **Design Motivation**: Naive approaches lead to grammar conflicts due to inconsistent connection patterns. Formulating compatibility checking as a combinatorial optimization problem on a graph guarantees consistency of the induced rules across all occurrence sites.

3. **MDL Selection**

    - **Function**: Out of all candidate motifs and their maximum compatible solutions, selects the one that compresses the data description length the most.
    - **Mechanism**: Utilizes the greedy objective $\max |C|(|D| - 1)$, where $|C|$ is the maximum clique size (number of compatible occurrences) and $|D|$ is the number of motif nodes. Once selected, motif contraction is performed—replacing each occurrence with a non-terminal node. This process iterates until no motif has a compatible occurrence count of $\geq 2$.
    - **Design Motivation**: The MDL principle (Occam's razor) is a classic guiding principle for unsupervised graph compression. Higher compression rates indicate that the discovered patterns are more representative, resulting in a more compact induced grammar.

4. **Disambiguation Procedure**

    - **Function**: Modifies grammar $G \to G'$ to eliminate ambiguous parses within the dataset.
    - **Mechanism**: Although determining whether a general graph grammar is ambiguous is undecidable, all derivations can be enumerated for a given dataset. The authors design a DAG edition of the CYK dynamic programming parsing algorithm, utilizing the **canonical string representation** of DAGs for hash memoization, and accelerating execution by pruning disconnected or cyclic intermediate graphs. After finding all derivations, they model a **maximum hitting set** problem to determine the minimal set of rules to remove, preserving unique parses for as much of the data as possible.
    - **Design Motivation**: Unambiguity is the core guarantee of a "one-to-one mapping." Although NP-hard, the problem can be efficiently solved by leveraging structural properties unique to DAGs (canonicalization, acyclic constraints).

### Loss & Training

Standard VAE training: Maximizing the Evidence Lower Bound (ELBO), which includes the reconstruction loss (cross-entropy on rule sequence) and a KL divergence regularization term. Because the one-to-one mapping is guaranteed, reconstruction is equivalent to exact matching of the rule sequence.

During inference, constrained decoding is performed using the properties of context-free grammars: the first step only allows rules with the start symbol $S$ on the left-hand side, and each subsequent step only allows rules corresponding to the current non-terminal, terminating when no non-terminals remain. Domain-specific constraints (e.g., stability constraints for operational amplifiers in circuits) can also be incorporated to ensure valid generation.

## Key Experimental Results

### Main Results: Unconditional Generation

| Dataset | Method | Accuracy (%) | Validity (%) | Uniqueness (%) | Novelty (%) |
|--------|------|----------|----------|----------|----------|
| ENAS | D-VAE | 99.96 | 100 | 37.26 | 100 |
| ENAS | DAGNN | — | — | — | — |
| ENAS | **DIGGED (GNN)** | **100** | **100** | **98.7** | **99.9** |
| BN | D-VAE | 99.94 | 98.84 | 38.98 | 98.01 |
| BN | GraphRNN | 96.71 | 100 | 27.30 | 98.57 |
| BN | **DIGGED (GNN)** | **100** | **100** | **97.6** | **100** |

DIGGED outperforms the runner-up method by over 50% in uniqueness while maintaining 100% validity and near-100% novelty.

### Circuit Design: Bayesian Optimization FoM

| Method | DAG Validity (%) | Circuit Validity (%) | Novelty (%) | BO Opt. FoM ↑ |
|------|-------------|-------------|----------|------------|
| PACE | 83.12 | 75.52 | 97.14 | 33.27 |
| CktGNN | 98.92 | 98.92 | 92.29 | 33.44 |
| CktGNN (CktBench301) | — | — | — | 190.24 |
| CktBench101 (max) | 100 | 100 | 0 | 326.67 |
| **DIGGED (GNN)** | **100** | **100** | **78.80** | **310.26** |

DIGGED outperforms CktGNN by over 9 times in FoM optimization (310.26 vs 33.44), closely approaching the optimal design in the training set (326.67), while generating 100% valid circuits.

### Ablation Study: Comparison of Sequence Representations

| Encoding Method | Validity (%) | Uniqueness (%) | Novelty (%) | BO Top-1 FoM |
|---------|----------|----------|----------|------------|
| Graph2NS-Default | 80.2 (CKT) | 71.0 | 96.8 | 220.96 |
| Graph2NS-BFS | 0.1 (CKT) | 100 | 100 | — |
| Graph2NS-Random | 0 (CKT) | — | — | — |
| **DIGGED** | **100** | **100** | **78.8** | **306.32** |

Comparison shows: BFS and random ordering fail almost completely to generate valid samples on circuit tasks, whereas DIGGED's grammatical serialization eliminates position dependency, achieving 100% validity and the optimal FoM.

### Key Findings

- **Uniqueness is DIGGED's greatest advantage**: Achieving a uniqueness rate of 98.7% on ENAS vs 37.26% for D-VAE, demonstrating that grammatical serialization almost completely eliminates generation redundancy.
- **GNN encoder outperforms Token encoder**: The latent space constructed by the GNN encoder is better suited for property prediction and generation (the Token encoder achieves a Pearson $r$ of only 0.049 on ENAS), though the Token encoder still validates the feasibility of the serialization approach.
- **Compression rate reflects structural complexity**: Linear structures (ENAS) exhibit the highest compression rate, while dense graphs (BN) show the lowest, though BN achieves a high compression ratio with the fewest rules (845).
- **Description length is negatively correlated with prediction error**: Longer rule sequences contain more structural information, leading to lower evaluation errors (Fig. 5), which implies that MDL compression acts as an explicit information bottleneck mechanism.

## Highlights & Insights

- **Complete reduction of graph problems to sequence problems**: By leveraging unambiguous graph grammars from formal language theory, this work achieves a strict bijection from DAGs to sequences for the first time. This means that all advances of Transformers in sequential tasks (such as pre-training and generation strategies) can be directly applied to graph tasks, launching a brand-new modeling paradigm.
- **MDL as an unsupervised induction objective**: Without requiring labels or external supervision, meaningful graph structural primitives (motifs) are discovered solely by "compressing the dataset." This paradigm can be transferred to any structured data task where reusable components need to be identified.
- **Elegance of constrained decoding**: Context-free grammars naturally support incremental validity checking. In circuit design, stability constraints can be translated into predicates over candidate rules, entirely avoiding post-processing or rejection sampling, which is far more efficient than post-hoc filtering.
- **Compatibility Maximization $\to$ Maximum Clique** is an ingenious modeling formulation. It formalizes transition instruction consistency as a classic combinatorial optimization problem, providing both theoretical guarantees and practical approximation algorithms.

## Limitations & Future Work

- **Relationship between grammar scale and data scale**: Extracting 7504 rules on ENAS leads to an excessively large token dictionary, which degrades the performance of the Token encoder (Pearson $r$ is only 0.049). It is necessary to explore more aggressive grammar compression or hierarchical grammar structures to regulate the rule count.
- **Limited to DAGs**: The method relies on the topological sort and acyclic nature of DAGs (e.g., canonical string representations, intermediate graph pruning) and cannot be directly scaled to general directed or undirected graphs.
- **Subproblems are NP-hard**: Calculations for subgraph isomorphism, maximum clique, and hitting sets restrict scalability. Although approximate algorithms are provided, efficiency remains a bottleneck on large-scale datasets.
- **Undecidability of disambiguation**: Disambiguation can only be performed on the training data, and there is no guarantee that test data will also be unambiguous, creating a risk of generalization issues.
- **Trade-off between compression rate and subgraph complexity** is under-analyzed—there is a lack of theoretical guidance on when to stop compression or what specifies the optimal motif granularity.

## Related Work & Insights

- **vs D-VAE / DAGNN**: D-VAE and DAGNN use GNNs to encode the topology of DAGs, but their decoding still relies on autoregressively adding nodes/edges sequentially. This allows multiple generation sequences for the same graph (violating the one-to-one mapping) and requires tracking intermediate graph states. DIGGED replaces autoregressive decoding with grammatical derivation, making it stateless and unique.
- **vs CktGNN**: CktGNN uses manually defined circuit substructure bases as motifs, showing heavy reliance on domain knowledge. DIGGED automatically induces motifs from data, which is more general and significantly outperforms block-based designs on BO FoM.
- **vs Molecular Graph Grammars (Jin et al. 2018, Sun et al. 2024)**: These works apply graph grammars to molecular generation but face limited decoding capabilities. DIGGED is the first to combine graph grammars with Transformer decoders, leveraging the statelessness of context-free grammars for highly efficient constrained sampling.
- The "graph-to-sequence bijection" concept proposed in this work may inspire the tokenization of structured data (e.g., knowledge graphs, scene graphs) in multimodal VLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to establish a strict bijection from DAGs to sequences from the perspective of formal language theory, presenting an entirely new angle.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three real-world tasks (NAS, BN, circuits) + comprehensive ablations + case studies, though lacking validation on large-scale graph datasets.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theory and clear illustrations, but with a heavy notation system, presenting a steep learning curve.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for graph generation and modeling, though the DAG limitation and scalability issues constrain its immediate impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](../../NeurIPS2025/image_generation/gspn-2_efficient_parallel_sequence_modeling.md)
- [\[NeurIPS 2025\] Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning](../../NeurIPS2025/image_generation/graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)
- [\[ICLR 2026\] Generating Directed Graphs with Dual Attention and Asymmetric Encoding](../../ICLR2026/image_generation/generating_directed_graphs_with_dual_attention_and_asymmetric_encoding.md)
- [\[CVPR 2025\] Parallel Sequence Modeling via Generalized Spatial Propagation Network](../../CVPR2025/image_generation/parallel_sequence_modeling_via_generalized_spatial_propagation_network.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](../../ICCV2025/image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)

</div>

<!-- RELATED:END -->
