---
title: >-
  [Paper Note] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach
description: >-
  [ICML 2025][Model Compression][Directed Graph Learning] Proposed EDEN (Entropy-driven Digraph Knowledge Distillation), which constructs a Hierarchical Knowledge Tree (HKT) from a data-centric perspective. By measuring directed topological structures and quantifying node mutual information, it reveals the latent associations between topology and node attributes in directed graphs. Serving as a plug-and-play module, it brings an average performance gain of 2-5% to any DiGNN and…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Directed Graph Learning"
  - "Data-centric ML"
  - "Knowledge Distillation"
  - "Information Entropy"
  - "Hierarchical Coding"
date: 2026-05-08
content_hash: bf95ff21e920f5bc
---

# Toward Data-centric Directed Graph Learning: An Entropy-driven Approach

**Conference**: ICML 2025  
**arXiv**: [2505.00983](https://arxiv.org/abs/2505.00983)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: Directed Graph Learning, Data-centric ML, Knowledge Distillation, Information Entropy, Hierarchical Coding

## TL;DR

Proposed EDEN (Entropy-driven Digraph Knowledge Distillation), which constructs a Hierarchical Knowledge Tree (HKT) from a data-centric perspective. By measuring directed topological structures and quantifying node mutual information, it reveals the latent associations between topology and node attributes in directed graphs. Serving as a plug-and-play module, it brings an average performance gain of 2-5% to any DiGNN and achieves SOTA on 14 datasets and 4 downstream tasks.

## Background & Motivation

**Background**: Directed Graph Neural Networks (DiGNNs) have recently attracted attention, including spectral-based methods (MagNet, HoloNet) and spatial-based methods (Dir-GNN, NSTE), which model complex topologies by considering edge directionality. However, existing works primarily focus on model architecture innovation (model-centric), while overlooking deep exploration at the data level.

**Limitations of Prior Work**: Although existing DiGNNs utilize directed edges, they fail to comprehensively mine the rich hidden knowledge in directed graph data—there are massive latent associations between topology and node attributes (e.g., unobserved structural patterns). These data-level limitations result in sub-optimal model-level prediction performance. Furthermore, undirected GNNs struggle on homophily/heterophily-entangled graphs, primarily because they ignore valuable directed topological information.

**Key Challenge**: Data quality determines the upper bound of model performance (data-level limitation $\rightarrow$ model-level sub-optimal), yet existing methods focus effort on the model end, neglecting the potential for improvement from the data side. The topological complexity and variety of node attributes in directed graphs present richer but also harder-to-decode data knowledge.

**Goal**: How to reveal hidden structural knowledge in directed graphs from a data-centric perspective and inject it as a supervision signal into the model training process.

**Key Insight**: Based on hierarchical coding theory—real-world directed graphs are affected by natural noise, and their information entropy $\mathcal{H}$ is determined by topology and attributes. Minimizing $\mathcal{H}$ can reveal the true structure $\mathcal{T}$, while data knowledge $\mathcal{K}$ is hidden within $\mathcal{T}$.

**Core Idea**: Constructing a Hierarchical Knowledge Tree (HKT) to hierarchically encode directed graphs from both topological and attribute perspectives, and injecting the mined knowledge into model training through data-level online knowledge distillation.

## Method

### Overall Architecture

The pipeline of EDEN consists of three steps: (1) **Knowledge Discovery**—constructing a coarse-grained HKT using directed topological structure metrics, and then refining the HKT through neural estimation of node mutual information; (2) **Knowledge Distillation**—personally transferring the knowledge of parent nodes (teachers) in the HKT to child nodes (students) to achieve data-level online KD; (3) **Leaf Node Prediction**—aggregating multi-level representations through random walks on the tree to generate leaf node predictions for downstream tasks.

### Key Designs

1. **Topology-aware Directed Structural Metrics and Coarse-grained HKT Construction**:

    - **Function**: Quantifying topological uncertainty based on the information entropy of directed graphs to construct an initial hierarchical knowledge tree.
    - **Mechanism**: First, defining a one-dimensional structural metric $\mathcal{H}^1(\mathcal{G}) = -\sum_{v \in \mathcal{V}} (\frac{\tilde{d}_v^{in}}{m} \log \frac{\tilde{d}_v^{in}}{m} + \frac{\tilde{d}_v^{out}}{m} \log \frac{\tilde{d}_v^{out}}{m})$, which is then extended to an $h$-dimensional partition tree $\mathcal{H}^h(\mathcal{G}) = \min_{\mathcal{T}: Height(\mathcal{T})=h}\{\mathcal{H}^{\mathcal{T}}_{in} + \mathcal{H}^{\mathcal{T}}_{out}\}$. A greedy algorithm is applied to find the optimal HKT that minimizes uncertainty. Innovation: introducing reverse probability to solve the problem where random walk paths decay drastically when directed graphs are not strongly connected, and adding self-loops for sink nodes.
    - **Design Motivation**: Unlike prior works that solely use forward random walks, experiments reveal that the proportion of complete paths in directed graphs drops sharply after only 5 steps of forward walk. Introducing reverse probability helps capture more distant neighborhood information.

2. **Mutual Information-driven HKT Refinement**:

    - **Function**: Using node attribute information to perform fine-grained refinement on the coarse-grained HKT.
    - **Mechanism**: For each partition $\mathcal{X}_p$, neural estimation of mutual information within and between partitions is defined. Based on the DV representation of f-divergence and GAN-like divergence (Theorems 3.1-3.3), a neural network $F_w$ is trained to estimate the mutual info between a node and its generalized neighborhood, determining whether node affiliation is correct. The criterion function $C(\Omega) = \hat{I}_{GAN}^{(\Omega)}$ is used to discover the most informative subset of nodes in each partition.
    - **Design Motivation**: Partitions obtained solely through topological metrics are too coarse. Node attributes (features and labels) exhibit more complex knowledge patterns in directed graphs, necessitating mutual information-driven refinement to ensure high similarity within the same partition and high distinction between different partitions.

3. **Node-adaptive Knowledge Distillation**:

    - **Function**: Personalizing the transfer of knowledge from parent nodes in the HKT to child nodes.
    - **Mechanism**: Parent node knowledge is generated via weighted aggregation as $\mathbf{X}_p = \mathcal{S}_{\Omega_p} \mathbf{X}_{\Omega_p}$, and then unclear knowledge is filtered out through uncertainty quantification $\mathcal{U}_p = \sigma(\mathcal{Q}_{parent}(-\sum_i \mathbf{X}_{p,i} \log \mathbf{X}_{p,i}))$. The final KD loss is formulation as $\mathcal{L}_{kd} = \|\mathbf{X}_p / \mathcal{U}_p - \mathcal{Q}_{child}(\mathbf{X}_{v_1,v_2})\|_F$. Diverse knowledge $\mathcal{S}_{v_2}$ is introduced to prevent overfitting.
    - **Design Motivation**: Not all parent node knowledge is clear and expressible, and each child node has a unique graph context and knowledge demand, thus requiring personalized transfer rather than uniform distillation.

### Loss & Training

The total training objective is $\mathcal{L} = \mathcal{L}_{\text{cross-entropy}}(\hat{\mathbf{Y}}, \mathbf{Y}) + \alpha \mathcal{L}_{kd}$, where $\alpha$ controls the intensity of KD. Leaf node prediction leverages random walks on the tree to obtain the multi-level representation sequence $\mathcal{P}_{rw}^k$. Node-level tasks prioritize sampling parent/child nodes (multi-level representations), while link-level tasks prioritize sampling sibling nodes (same-level representations). Lightweight design: using Monte Carlo simulation to approximate the optimal HKT, incremental training, and prototype representations to reduce computational overhead, and parameter-free feature propagation instead of learnable GNNs.

## Key Experimental Results

### Main Results (Node-C Accuracy %)

| Model | CoraML | CiteSeer | WikiCS | Tolokers | Empire | Rating | Arxiv |
|------|--------|----------|--------|----------|--------|--------|-------|
| GCNII | 80.8 | 62.5 | 78.1 | 78.5 | 76.3 | 42.3 | 65.4 |
| HoloNet | 82.5 | 64.1 | 79.2 | 79.4 | 78.7 | 44.5 | 67.5 |
| **EDEN** | **84.6** | **65.8** | **81.4** | **81.3** | **81.1** | **46.3** | **69.7** |

### Plug-and-Play Performance (Node-C Accuracy %)

| DiGNN + EDEN | CoraML | CiteSeer | WikiCS | Arxiv | Average Gain |
|-------------|--------|----------|--------|-------|---------|
| Dir-GNN → +EDEN | 82.6→85.9 | 64.5→67.2 | 79.1→82.8 | 66.9→70.5 | **⇑4.68%** |
| HoloNet → +EDEN | 82.5→86.0 | 64.1→67.5 | 79.2→82.6 | 67.5→70.8 | **⇑4.46%** |
| DIMPA → +EDEN | 82.4→85.4 | 64.0→66.9 | 78.8→82.2 | 67.1→69.9 | **⇑4.32%** |
| OptBG → +EDEN | 81.5→82.8 | 62.4→64.6 | 77.9→79.4 | 66.4→67.9 | ↑2.75% |

### Link-Level Tasks (AUC/AP/ACC %)

| Dataset | Task | Best Baseline | EDEN | Gain |
|--------|------|---------|------|------|
| Slashdot | Existence (AUC) | 90.6 (NSTE) | **91.8** | +1.2 |
| Slashdot | Direction (AUC) | 92.2 (NSTE/MagNet) | **93.3** | +1.1 |
| Slashdot | Link-C (ACC) | 85.4 (NSTE) | **87.1** | +1.7 |
| WikiTalk | Existence (AUC) | 94.7 (Dir-GNN) | **95.4** | +0.7 |
| WikiTalk | Direction (AUC) | 90.9 (Dir-GNN) | **91.5** | +0.6 |

### Ablation Study

| Configuration | Tolokers Node-C | Slashdot Existence | Slashdot Direction |
|------|-----------------|--------------------|--------------------|
| EDEN (Full) | 81.33 | 91.82 | 93.29 |
| w/o Diverse Knowledge | 80.90 (-0.43) | 91.40 (-0.42) | 92.96 (-0.33) |
| w/o Personalized Transfer | 80.84 (-0.49) | 91.49 (-0.33) | 93.01 (-0.28) |
| w/o Tree-based Random Walk | 80.67 (-0.66) | 91.16 (-0.66) | 92.77 (-0.52) |
| w/o KD Loss | 80.01 (-1.32) | 90.84 (-0.98) | 92.25 (-1.04) |

### Key Findings
- The KD loss is the most critical component, with an average performance drop of about 1% when removed, indicating that data-level knowledge distillation is indeed the core driver of performance improvement.
- The tree-based random walk is the second most important module, with a drop of 0.66 in Slashdot Existence AUC upon its removal, demonstrating that multi-level representation aggregation significantly assists prediction.
- When used as a plug-and-play module, EDEN brings a more substantial improvement to DiGNNs (Dir-GNN +4.68%) compared to traditional GNNs (OptBG +2.75%), validating the hypothesis that directed graph data contains richer knowledge.
- As an independent method, EDEN outperforms the best baseline for Node-C by an average of 2.78% and improves link-level tasks by 2.24% on average.

## Highlights & Insights

- **Data-centric Graph Learning Paradigm**: EDEN is the first to propose hierarchical data-level knowledge distillation (distinguished from model-level offline KD), converting the insight of "data quality limiting the model's upper bound" into an actionable framework. This perspective can be transferred to any structured data learning scenario.
- **Clever Extension of Directed Topological Metrics**: Incorporating reverse probability solves the random walk decay issue in non-strongly connected directed graphs, and the $h$-dimensional partition tree extends one-dimensional Shannon entropy to hierarchical structural metrics, offering a natural and practical theoretical basis.
- **Model-Agnostic Plug-and-Play Design**: EDEN can be directly attached to any DiGNN as an online KD module without modifying the underlying model architecture, yielding high flexibility for actual deployment.

## Limitations & Future Work

- Scalability remains a bottleneck. Although lightweight solutions have been proposed (Monte Carlo, incremental training, and parameter-free propagation), the practical efficiency on large-scale graphs has not been fully verified.
- The number of layers $h$ in the HKT is a hyperparameter that requires tuning for different graph structures, lacking an adaptive selection mechanism.
- Neural estimation of mutual information requires training an extra neural network (e.g., MLP), increasing overall training complexity.
- Extended experiments were only conducted on homophilous and heterophilous undirected graphs; its applicability to more complex graph types such as dynamic graphs and hypergraphs remains unexplored.

## Related Work & Insights

- **vs Traditional DiGNNs (Dir-GNN, HoloNet)**: These methods utilize directed edges from the model end, while EDEN complements from the data end—the two are orthogonal and complementary, with the combination of EDEN+Dir-GNN yielding the best results.
- **vs Graph Structure Learning (CoGSL)**: CoGSL optimizes graph view generation and fusion through mutual information. EDEN borrows a similar concept but focuses on hierarchical structure discovery rather than view augmentation.
- **vs Graph Contrastive Learning (DGI, GMI)**: DGI/GMI maximize mutual info between nodes and the graph/neighborhood for self-supervision, whereas EDEN's mutual info is used for knowledge tree refinement and knowledge distillation. Though objectives differ, they share information theory tools.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The data-centric and hierarchical coding theory is a novel perspective in graph learning, but the construction of HKT is essentially an information-theoretic reformulation of hierarchical clustering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 14 datasets, 4 downstream tasks, plug-and-play validation, ablation study, and efficiency analysis, making it highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The theoretical derivation is rigorous and detailed, though dense equations might affect readability.
- **Value**: ⭐⭐⭐⭐ The data-centric perspective and plug-and-play characteristics give it practical application value, though the non-substitutability of its core contributions warrants further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs](../../NeurIPS2025/model_compression/codegemm_a_codebook-centric_approach_to_efficient_gemm_in_quantized_llms.md)
- [\[ICML 2025\] Beyond Communication Overhead: A Multilevel Monte Carlo Approach for Mitigating Compression Bias in Distributed Learning](beyond_communication_overhead_a_multilevel_monte_carlo_approach_for_mitigating_c.md)
- [\[ACL 2025\] Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](../../ACL2025/model_compression/prompt_distill_teacher_student.md)
- [\[ACL 2025\] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](../../ACL2025/model_compression/magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)
- [\[ICML 2026\] Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](../../ICML2026/model_compression/beyond_tokens_enhancing_rtl_quality_estimation_via_structural_graph_learning.md)

</div>

<!-- RELATED:END -->
