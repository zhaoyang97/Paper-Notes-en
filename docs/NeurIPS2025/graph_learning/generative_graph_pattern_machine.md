---
title: >-
  [Paper Note] Generative Graph Pattern Machine
description: >-
  [NeurIPS 2025][Graph Learning][graph pre-training] This paper proposes the Generative Graph Pattern Machine (G2PM), a fully message-passing-free generative Transformer framework for graph pre-training. It tokenizes graph…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "graph pre-training"
  - "generative Transformer"
  - "substructure tokenization"
  - "masked substructure modeling"
  - "random walk"
  - "message-passing-free"
date: 2026-05-08
content_hash: 33390181f6977912
---

# Generative Graph Pattern Machine

**Conference**: NeurIPS 2025
**arXiv**: [2505.16130](https://arxiv.org/abs/2505.16130)  
**Authors**: Zehong Wang, Zheyuan Zhang, Tianyi Ma, Chuxu Zhang, Yanfang Ye (University of Notre Dame, University of Connecticut)
**Code**: [https://github.com/Zehong-Wang/G2PM](https://github.com/Zehong-Wang/G2PM)  
**Area**: Graph Learning / Graph Pre-training / Transformer
**Keywords**: graph pre-training, generative Transformer, substructure tokenization, masked substructure modeling, random walk, message-passing-free

## TL;DR

This paper proposes the Generative Graph Pattern Machine (G2PM), a fully message-passing-free generative Transformer framework for graph pre-training. It tokenizes graph instances (nodes/edges/graphs) into substructure sequences via random walks and performs self-supervised pre-training under a Masked Substructure Modeling objective. G2PM comprehensively outperforms existing graph pre-training methods on node/link/graph classification and cross-domain transfer tasks, while exhibiting model and data scaling laws analogous to those observed in NLP and CV.

## Background & Motivation

Transformers have achieved remarkable success in NLP (LLMs) and CV (ViT) through generative pre-training, yet this paradigm has not produced comparable breakthroughs in the graph domain, severely limiting the development of graph foundation models. Existing graph learning predominantly relies on **message-passing GNNs (MPNNs)**, which suffer from the following fundamental bottlenecks:

**Limited expressive power**: Upper-bounded by the 1-WL test

**Over-smoothing**: Node representations converge as the number of layers increases

**Over-squashing**: Information is compressed and lost at bottleneck edges

**Insufficient long-range dependency modeling**: The locality of message passing restricts global information flow

**Poor scalability**: Scaling up model size or data volume fails to yield sustained performance improvements

Furthermore, existing graph pre-training research has primarily adopted **contrastive learning**, which is known to be inferior to generative objectives (as demonstrated by BERT and MAE) in learning generalizable semantic representations.

Transferring generative Transformer pre-training to graphs presents three unique challenges:
- **Lack of sequential structure**: Graphs possess neither the linear structure of text nor the grid structure of images
- **Low semantic granularity**: Nodes and edges are low-level semantic units, unlike the high-level semantics of words or image patches
- **Sequence length**: Treating nodes as tokens causes sequence length to grow linearly with the number of nodes, making the quadratic complexity of Transformers intractable

## Method

### Core Idea: Substructure Sequentialization

The central insight of G2PM is that **meaningful graph substructures** serve as the semantic building blocks of graphs (e.g., triangles in social networks represent stable relationships; benzene rings in molecular graphs encode chemical stability). Any graph instance can thus be represented as a **sequence of substructures**, analogous to word sequences in text or patch sequences in images.

### Random Walk Tokenizer

To avoid the high computational cost of subgraph isomorphism matching (NP-complete), G2PM employs **random walks** to sample substructure patterns on the fly:

- For each graph instance, $k$ unbiased random walks of length $L$ are sampled
- Transition probability: $P(v_{i+1}|v_0,...,v_i) = \mathbb{1}[(v_i,v_{i+1}) \in \mathcal{E}] / D(v_i)$
- Each walk corresponds to a substructure pattern (e.g., [1,2,3,1,4,3] corresponds to a diamond structure)

Substructure encoding: the node feature sequence along a walk $w = [\mathbf{x}_1, ..., \mathbf{x}_m]$ is fed into a **lightweight Transformer encoder** $f$ to produce a substructure embedding $\mathbf{p} = f(w)$.

**Key design decisions**:
- **No positional encoding**: Experiments show positional embeddings yield no consistent benefit and introduce cubic complexity; they are therefore omitted entirely
- **Task-agnostic**: For node tasks, walks originate from the target node; for edge tasks, from the endpoints; for graph tasks, from randomly sampled nodes
- **Elimination of message-passing bottlenecks**: Random walks naturally capture long-range dependencies, possess expressive power beyond 1-WL, and are immune to over-smoothing and over-squashing

### G2PM Backbone

A standard Transformer architecture serves as the backbone for both encoder and decoder. The input is a sequence of substructure tokens $\mathbf{P} = [\mathbf{p}_1, ..., \mathbf{p}_n]$, which are encoded layer by layer through multi-head self-attention and feed-forward networks. Default configuration:

- Hidden dimension 768, 12 attention heads
- Encoder: 3 layers; Decoder: 1 layer
- Approximately 21.3M (encoder) + 7.1M (decoder) parameters

### Pre-training Objective: Masked Substructure Modeling (MSM)

Inspired by BERT/MAE-style masked modeling, but operating on **substructures** rather than words or patches:

1. Sample a substructure sequence $\mathbf{P} = [\mathbf{p}_1, ..., \mathbf{p}_n]$ from the graph
2. Randomly mask a subset of substructures (high masking ratio by default), retaining visible substructures $\mathbf{P}_{vis}$
3. The visible substructures are passed through the encoder to produce contextual representations $\mathbf{H}_{vis}$
4. Learnable [MASK] tokens are inserted at masked positions, and the full sequence is fed to the decoder for reconstruction

**Reconstruction target**: Rather than reconstructing low-level node features, the model reconstructs high-level semantic embeddings produced by an **EMA encoder**:

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \text{is\_masked}(\mathbf{p}_i) \cdot \|\mathbf{r}_i - \text{sg}[\hat{\mathbf{p}}_i]\|_2^2$$

where $\hat{\mathbf{p}}_i = f_{\text{EMA}}(w_i)$ is the output of the EMA encoder, and sg denotes the stop-gradient operator.

### Data Augmentation Strategy

A **mixed augmentation** strategy is adopted, applying one feature-level and one structure-level augmentation simultaneously to each instance:
- **Feature-level**: Feature masking (zeroing out a subset of features), node masking (masking entire nodes)
- **Structure-level**: Substructure corruption (dropping nodes along a walk), substructure injection (replacing nodes with random nodes)

### Downstream Adaptation

After pre-training, the decoder is discarded and a linear prediction head is appended to the encoder. Final predictions are made by mean-pooling over substructure representations.

## Design Space Exploration (Key Ablation Study)

The paper conducts a systematic analysis of the design space, yielding several important insights:

| Design Dimension | Best Choice | Insight |
|---------|---------|------|
| Architecture style | MAE-style (sparse encoder + lightweight decoder) | SimMIM-style performs worse; sparsity encourages the encoder to learn stronger contextual representations |
| Model dimension | 768 (64.5M parameters) | Performance improves consistently with dimension, exhibiting favorable scaling behavior |
| Encoder/decoder depth | Encoder 3 layers / Decoder 1 layer | Deeper decoders lead to overfitting |
| Substructure encoder | Transformer | Outperforms GRU, GIN, and mean pooling |
| Positional encoding | Not used | No consistent benefit; high computational cost |
| Masking ratio | High (~0.7–0.8) | High masking reduces redundant walks and provides a more challenging learning signal |
| Mask token | Learnable | Outperforms fixed/random/sampled tokens |
| Reconstruction target | EMA semantic embeddings | Outperforms low-level features (mean pooling/concatenation) |
| Loss function | L2 loss | Outperforms L1; sensitivity to outliers helps capture subtle semantics |
| EMA momentum | α=0.99, updated every 10 steps | α=0 (no EMA) causes training collapse |
| Augmentation strategy | Mixed augmentation | Single augmentation has limited effect; diversity is critical |

## Key Experimental Results

### Node Classification (Homophilic Graphs)

Evaluated on 7 datasets (from Pubmed to ogbn-products), G2PM achieves the best performance among all self-supervised methods with an average rank of 2.0 (second only to supervised GPM):
- ogbn-arxiv: 72.31% (60M parameters), with performance improving consistently as parameter count increases
- ogbn-products: 80.56% (significantly outperforming GraphMAE2's 79.33%)

### Model/Data Scaling

- **Model scaling**: On ogbn-arxiv, performance improves continuously from 68.2% to 72.3% as parameters scale from 0.4M to 64.5M, whereas GraphMAE/BGRL plateau or decline beyond ~3M parameters
- **Data scaling**: Increasing pre-training data consistently improves performance, while competing methods peak at small data fractions and subsequently decline

### Graph Classification

G2PM achieves an average rank of 1.0 across 7 datasets (5 molecular + 2 social):
- ogbg-HIV: 78.7% (vs. GraphMAE 77.8%)
- IMDB-B: 83.0% (vs. GraphMAE 75.5%, a substantial gain)
- Without pre-training, the average rank drops to 8.3, demonstrating the critical role of pre-training for graph-level tasks

### Link Prediction

G2PM outperforms GCN and GraphMAE on all three datasets: Cora, Pubmed, and ogbl-Collab.

### Cross-Domain Transfer

G2PM demonstrates strong cross-domain transfer capability:
- Arxiv → Products: 81.3% (positive transfer +0.7%)
- Arxiv → HIV: 76.8%
- HIV → Arxiv: 72.6% (positive transfer +0.3%)
- HIV → PCBA: 77.9% (positive transfer +2.3%)

Joint pre-training on Arxiv + FB15K237 + ChemBL achieves performance close to or exceeding graph foundation models (OFA, GFT) across three target domains.

## Theoretical Insight: Substructure Dependencies

The paper analyzes the structural basis for MSM's effectiveness from three perspectives:

1. **Hierarchical dependency**: Smaller substructures (e.g., triangles) are building blocks of larger ones (e.g., cliques); inference can proceed bidirectionally
2. **Functional reinforcement**: Certain substructures co-occur statistically (e.g., high-density triangles correlate with higher-order cliques)
3. **Functional exclusion**: Certain substructures are mutually exclusive (e.g., dense communities are incompatible with star-hub structures)

## Limitations & Future Work

- The current framework uses **unordered** substructure sequences, suited for masked prediction; future work may explore ordered sequences to support **next-token prediction** and further improve scalability
- Random walk serves as an online tokenizer without learnable parameters; future work may design **adaptive, learnable substructure tokenizers**
- On heterophilic graphs (e.g., Actor), expanding the receptive field does not consistently improve performance, suggesting that key signals in certain graphs are inherently local

## Highlights & Insights

1. **Methodological significance**: G2PM demonstrates that graph learning can entirely abandon the message-passing paradigm, achieving state-of-the-art results through pure Transformer + random walk tokenization + generative pre-training, thereby charting a new direction for graph foundation models
2. **Scalability as a core contribution**: This is the first self-supervised method in the graph domain to exhibit model and data scaling laws comparable to those in NLP and CV; while GraphMAE/BGRL plateau or decline beyond 3M parameters, G2PM continues to improve at 60M
3. **Comprehensive design space exploration**: The ablation study yields multiple actionable insights (e.g., MAE outperforms SimMIM, high masking ratios are preferable, positional encodings are unnecessary) that offer valuable guidance for future work
4. **Cross-domain transfer capability** deserves attention: By learning transferable substructure patterns, G2PM achieves positive transfer across domain and task boundaries—a capability that message-passing methods struggle to attain
5. **Relationship to GPM**: G2PM can be viewed as a generative pre-training extension of GPM (a prior work from the same group with ViT-style substructure modeling), upgrading contrastive/supervised learning to generative self-supervised learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] P-DRUM: Post-hoc Descriptor-based Residual Uncertainty Modeling for Machine Learning Potentials](p-drum_post-hoc_descriptor-based_residual_uncertainty_modeling_for_machine_learn.md)
- [\[ACL 2026\] GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion](../../ACL2026/graph_learning/gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](../../ICML2026/graph_learning/generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning](../../AAAI2026/graph_learning/gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)

</div>

<!-- RELATED:END -->
