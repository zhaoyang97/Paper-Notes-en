---
title: >-
  [Paper Note] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs
description: >-
  [NeurIPS 2025][Medical Imaging][protein representation learning] This paper proposes SSHG (Secondary Structure-based Hierarchical Graph), a framework that constructs two-level hierarchical graph representations from prot…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "protein representation learning"
  - "graph neural networks"
  - "multiscale"
  - "secondary structure"
  - "hierarchical graph"
date: 2026-05-08
content_hash: 48415dbb48ad0b13
---

# Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs

**Conference**: NeurIPS 2025
**arXiv**: [2602.00862](https://arxiv.org/abs/2602.00862)
**Code**: Unavailable
**Area**: Medical Imaging
**Keywords**: protein representation learning, graph neural networks, multiscale, secondary structure, hierarchical graph

## TL;DR
This paper proposes SSHG (Secondary Structure-based Hierarchical Graph), a framework that constructs two-level hierarchical graph representations from protein secondary structure motifs — an intra-motif residue-level graph and an inter-motif global graph — and employs a two-stage GNN to learn local and global features respectively. Theoretical guarantees of maximal expressiveness are provided, with empirical improvements in both accuracy and computational efficiency on enzyme classification and ligand affinity prediction tasks.

## Background & Motivation

Graph neural networks have emerged as powerful tools for protein structure learning, enabling the capture of spatial relationships at the residue level. However, existing GNN approaches face two core challenges:

**Insufficient multiscale modeling.** Proteins are naturally organized hierarchically: primary structure (amino acid sequence) → secondary structure (α-helices, β-sheets, and other motifs) → tertiary/quaternary structure. Existing residue-level GNNs cannot effectively capture features at the secondary structure level. A canonical counterexample is the prion protein: the normal isoform PrP$^\text{C}$ and the pathological isoform PrP$^\text{Sc}$ share identical primary sequences but differ drastically in secondary structure — the normal form is α-helix-rich, while the pathological form converts to a β-sheet-enriched structure, causing fatal neurodegenerative disease. Pure residue-level GNNs cannot distinguish these two states.

**Inefficient modeling of long-range dependencies.** To capture global context, existing methods typically employ large cutoff radii (e.g., 16 Å), generating extremely dense graphs (up to ~15K edges) with substantial computational and memory overhead. Multiscale methods such as HoloProt use surface-based modeling but incur similarly high costs.

The core idea of this paper is to leverage domain knowledge — protein secondary structure — as natural hierarchical nodes: each secondary structure motif (e.g., an α-helix segment) serves as a higher-level node. This design simultaneously exploits biological prior knowledge and geometric information, enabling efficient multiscale protein modeling with very few edges (total edges $< 3N$, where $N$ is the number of residues), backed by theoretical guarantees of maximal expressiveness.

## Method

### Overall Architecture
SSHG is a modular two-stage GNN framework:
1. The DSSP algorithm partitions the protein sequence into secondary structure motifs (α-helices, β-strands, loops, etc.).
2. A two-level hierarchical graph is constructed: intra-structure graphs (residue-level graphs within each motif) and an inter-structure graph (global graph between motifs).
3. A first-stage GNN learns local features within each motif; a second-stage GNN learns global features across motifs.
Any GNN backbone (e.g., GVP-GNN, ProNet, Mamba) can be flexibly selected.

### Key Designs

1. **DSSP-based secondary structure segmentation and hierarchical graph construction**

    - **Function**: Segments the protein sequence into motif subsequences based on secondary structure, and constructs two-level graphs.
    - **Mechanism**: DSSP assigns each residue a secondary structure type token (H = α-helix, E = β-strand, T = turn, etc., 9 types in total); residues with consecutive identical tokens are grouped into the same subsequence $S_i$. An intra-structure graph $\mathcal{G}_i$ is built for each $S_i$ (residues as nodes; edges determined via the SCHull method from α-carbon coordinates). An inter-structure graph $\mathcal{G}$ is constructed from the geometric centers of all $S_i$ (motifs as nodes), with edge features encoding the local frame product $g_i^\top g_j$ representing relative orientations.
    - **Design Motivation**: Secondary structures are well-established functional units in proteins; using them as hierarchical nodes provides both biological grounding and a significant reduction in edge count. The SCHull graph guarantees geometric completeness and sparsity.

2. **Two-stage GNN message passing**

    - **Function**: Learns local intra-motif interactions in the first stage, then global inter-motif relationships in the second stage.
    - Stage 1: $T_1$ rounds of message passing are performed independently on each intra-graph $\mathcal{G}_i$, followed by readout to obtain motif embeddings $\mathbf{s}_i = \text{readout}_1(\{\!\!\{\mathbf{f}_k^{(T_1)} | k \in \mathcal{V}(\mathcal{G}_i)\}\!\!\})$.
    - Stage 2: $T_2$ rounds of message passing are performed on the inter-structure graph $\mathcal{G}$ using $\mathbf{s}_i$ as initial node features, producing global features $\mathbf{s}_{\text{global}} = \text{readout}_2(\{\!\!\{\mathbf{s}_i^{(T_2)} | i \in \mathcal{V}(\mathcal{G})\}\!\!\})$.
    - **Design Motivation**: The two-stage design allows the first-stage GNN to process small graphs (each motif typically contains only tens of residues) with few layers, while the second-stage GNN can capture long-range dependencies in a single layer due to the substantially reduced node count in the motif graph.

3. **Theoretical guarantee: Maximal Expressiveness Theorem (Theorem 4.2)**

    - Under injectivity assumptions on UPD, AGG, and readout, the two-stage SSHG GNN can distinguish any two protein structures that are inequivalent under rigid-body motion.
    - Sparsity guarantee (Proposition 3.2): the total edge count $|\mathcal{E}| + \sum_i |\mathcal{E}_i| < 3N$, where $N$ is the number of residues.
    - **Design Motivation**: This theorem establishes that the hierarchical design does not discard critical structural information — forming the theoretical cornerstone of the entire framework.

### Loss & Training
Loss functions are selected according to the downstream task: cross-entropy for enzyme classification and MSE for ligand affinity prediction. Data augmentation includes adding Gaussian noise to coordinates (std = 0.1), anisotropic scaling (0.9–1.1), and random masking of amino acid types (probability 0.1–0.2).

## Key Experimental Results

### Main Results

**Enzyme Reaction Classification (EC)**

| Method | Test Accuracy (%) | Training Time (s/epoch) | Parameters |
|---|---|---|---|
| GCN | 66.5 | 186 | — |
| GCN+SSHG | **71.2** | **150** | — |
| GVP-GNN | 68.5 | 334 | 1.0M |
| GVP-GNN+SSHG | **73.6** | **236** | 1.0M |
| IEConv | 87.2 | — | 9.8M |
| ProNet-Backbone | 86.4 | 210 | 1.3M |
| ProNet+SSHG | **87.2** | **140** | 1.3M |
| Mamba+SSHG | **88.4** | **157** | 1.5M |

**Ligand Binding Affinity Prediction (LBA)**

| Method | RMSE↓ | Pearson↑ | Spearman↑ | Training Time (s/epoch) |
|---|---|---|---|---|
| HoloProt-Full | 1.464 | 0.509 | 0.500 | 45 |
| ProNet-Backbone | 1.458 | 0.546 | 0.550 | 32 |
| ProNet+SSHG | **1.435** | **0.579** | **0.591** | **24** |
| Mamba+SSHG | **1.399** | **0.614** | **0.610** | 29 |

### Ablation Study

**Graph Construction Strategy Efficiency Comparison (ProNet backbone)**

| Configuration | Avg. Edges | Training Time (s/epoch) | Memory (MiB) | Accuracy (%) |
|---|---|---|---|---|
| cutoff=4 | 1,034 | 138 | 1,290 | 78.1 |
| cutoff=10 | 11,316 | 210 | 14,548 | 86.4 |
| cutoff=16 | 14,881 | 247 | 17,768 | 87.0 |
| **+SSHG** | **1,593** | **140** | **1,818** | **87.2** |

**Two-Stage GNN Parameter Allocation**

| Stage 1 Params | Stage 2 Params | Training Time | Accuracy (%) |
|---|---|---|---|
| 0.69M (balanced) | 0.69M | 140 | 87.2 |
| 1.03M (local-heavy) | 0.34M | 136 | **87.4** |
| 0.34M (global-heavy) | 1.03M | 142 | 87.1 |

### Key Findings
- SSHG delivers simultaneous accuracy gains and training speedups across all evaluated backbone architectures — a rare "win-win" outcome.
- With only 1,593 edges (vs. 14,881 for cutoff=16), SSHG achieves higher accuracy, confirming the effectiveness of hierarchical sparse graphs.
- Memory usage drops from 17,768 MiB to 1,818 MiB (a 90% reduction), which is critical for practical application to large-scale proteins.
- Allocating more parameters to the first stage (local motif modeling) slightly outperforms allocating them to the second stage, indicating that fine-grained local representations are more important.

## Highlights & Insights
- The framework design is highly elegant: protein secondary structure — an existing piece of biological knowledge — is used as a natural hierarchical partition, enabling hierarchical graph construction without any learned components. This paradigm of "domain-knowledge-driven graph construction" is worth generalizing to other structured data domains.
- The paper achieves a seamless integration of theory and experiment: maximal expressiveness and sparsity are formally guaranteed, while experiments consistently demonstrate accuracy improvements and efficiency gains. The edge count upper bound of $< 3N$ in Proposition 3.2 renders the complexity analysis concise and compelling.

## Limitations & Future Work
- Validation is currently limited to enzyme classification and ligand affinity prediction; tasks such as protein fold classification and protein–protein interaction remain to be evaluated.
- The readout function employs mean pooling rather than the injective aggregation required by theory; stronger aggregation schemes may yield further improvements.
- Integration with pretrained protein language models (e.g., ESM-2) has not been explored, representing a promising future direction.
- The secondary structure definition relies on the accuracy of the DSSP algorithm; robustness to predicted structures (e.g., AlphaFold outputs) requires further validation.

## Related Work & Insights
- **vs. HoloProt**: HoloProt achieves multiscale modeling via large-radius graphs and surface-based representations, with a training time of 300 s/epoch vs. 157 s/epoch for SSHG, and accuracy of 78.9% vs. 88.4% for Mamba+SSHG — a substantial margin in favor of SSHG.
- **vs. IEConv**: IEConv acquires multiscale features through hierarchical pooling applied across multiple layers, with 9.8M parameters vs. 1.3M for SSHG; SSHG achieves comparable accuracy at roughly one-seventh the parameter count.
- **vs. ProNet**: As one of SSHG's backbones, ProNet+SSHG reduces training time by 33% while maintaining equivalent parameter count, with equal or improved accuracy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of leveraging secondary structure to construct hierarchical graphs is both intuitively natural and theoretically well-founded, representing a significant contribution to the protein GNN literature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two benchmark tasks, multiple backbones, and comprehensive efficiency analysis are provided, though broader task coverage would strengthen the work.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous, figures are clear, and the logical chain from biological motivation to method design to theoretical guarantees is coherent and complete.
- **Value**: ⭐⭐⭐⭐⭐ — The framework provides a general plug-and-play module from which any GNN backbone can benefit, making it highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[NeurIPS 2025\] Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning](generative_distribution_embeddings_lifting_autoencoders_to_the_space_of_distribu.md)
- [\[ICLR 2026\] Protein Structure Tokenization via Geometric Byte Pair Encoding](../../ICLR2026/medical_imaging/protein_structure_tokenization_via_geometric_byte_pair_encoding.md)

</div>

<!-- RELATED:END -->
