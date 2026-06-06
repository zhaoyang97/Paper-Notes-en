---
title: >-
  [Paper Note] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization
description: >-
  [AAAI 2026][Graph Learning][Graph Transformer] GT-SNT is proposed to leverage spiking neural networks (SNNs) as a graph node tokenizer. By aggregating multi-step propagated features into compact spike-count embeddings as…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Graph Transformer"
  - "Spiking Neural Networks"
  - "Node Tokenization"
  - "Linear Complexity"
  - "Large-Scale Graphs"
date: 2026-05-08
content_hash: bbfa345a56a83483
---

# GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization

**Conference**: AAAI 2026
**arXiv**: [2504.11840](https://arxiv.org/abs/2504.11840)  
**Code**: [https://github.com/Zhhuizhe/GT-SNT](https://github.com/Zhhuizhe/GT-SNT)  
**Area**: Graph Learning
**Keywords**: Graph Transformer, Spiking Neural Networks, Node Tokenization, Linear Complexity, Large-Scale Graphs

## TL;DR
GT-SNT is proposed to leverage spiking neural networks (SNNs) as a graph node tokenizer. By aggregating multi-step propagated features into compact spike-count embeddings as node tokens, and employing Codebook-Guided Self-Attention (CGSA) to capture global context in linear time, GT-SNT achieves competitive performance on 9 node classification benchmarks while delivering up to 130× inference speedup.

## Background & Motivation

Graph Transformers (GTs) combine message passing with self-attention and achieve strong performance on graph prediction tasks, but suffer from **quadratic complexity**:

- **Full-attention GTs**: $O(N^2)$ complexity is infeasible for large-scale graphs (e.g., ogbn-products with 2.45M nodes)
- **Linear-attention GTs** (NodeFormer, SGFormer): reduce computation but may cause an "over-globalization" problem
- **Node tokenization methods** (GOAT, VQGraph): employ VQ-VAE-style codebooks but suffer from severe **codebook collapse**—a large proportion of codewords in predefined codebooks are never utilized

The authors identify an interesting connection: **SNNs naturally convert continuous high-precision inputs into low-precision, event-driven representations**—which is essentially a "sequence-to-token" transformation. This motivates the core research question: *Can we go beyond treating spiking neurons as low-power units and instead leverage spike representations to build efficient tokenized Graph Transformers?*

**Core Idea**: The spike-firing mechanism of SNNs is used to transform node embedding sequences from multi-step propagation into discrete spike-count vectors as node tokens, enabling graph node tokenization without predefined codebooks and achieving 100% codebook utilization.

## Method

### Overall Architecture

GT-SNT consists of four modules:
1. **SNT (Spiking Node Tokenizer)**: Generates spike-count embeddings and a dynamically reconstructed codebook
2. **Auxiliary MPNN**: Extracts node embeddings containing semantic and local topology information
3. **CGSA (Codebook-Guided Self-Attention)**: Achieves linear-time global aggregation based on spike-count tokens
4. **Classification Head**: A simple fully connected layer

### Key Designs

1. **Spiking Node Tokenization (SNT)**:

    - Function: Encodes graph topology into discrete spike-count vectors
    - Mechanism:
      * Samples a learnable $D$-dimensional random feature matrix $R$
      * Iterates $T$ steps via propagation operator $P$ to collect a node embedding sequence $M = \{M^0, M^1, \ldots, M^T\}$
      * Feeds the sequence into IF/LIF/PLIF spiking neurons, emitting a spike $S^t$ upon reaching threshold $V_{th}$
      * Accumulates spikes to obtain spike-count embeddings $\hat{S} = \sum S^t$
      * Deduplicates $\hat{S}$ to construct codebook $C$ and obtains a one-hot index matrix $U$
    - Design Motivation:
      * The discrete spike-count space is $|\tilde{C}| = (T+1)^D$, but the actual codebook size $B \ll |\tilde{C}|$, avoiding codebook collapse
      * Propagation steps construct sequential inputs (rather than repeating static graphs), reducing additional computational overhead
      * The spiking mechanism naturally injects a locality prior from the graph

2. **Codebook-Guided Self-Attention (CGSA)**:

    - Function: Achieves linear-complexity global attention based on spike-count tokens
    - Mechanism:
      * Queries and Values are derived from node embeddings $H$ of the auxiliary MPNN
      * Keys are generated from codebook $C$ via $U \cdot G$ matrix multiplication, where $G = \text{Norm}(\text{Linear}(C))$
      * Attention is decomposed as $\text{softmax}(Q\hat{C}^T) \cdot U^T \cdot V$, with complexity $O(NBd_v)$ where $B \ll N$
    - Design Motivation: Node-to-token attention reduces the cost of full pairwise node attention; $B \ll N$ guarantees linear growth

3. **Truncation Strategy**:

    - Function: Prevents the codebook from growing unboundedly during early training
    - Mechanism: Codewords are ranked by usage frequency; only the top $B_{\max}$ are retained
    - Design Motivation: Spike patterns are unstable early in training and may produce an excessive number of unique codewords

### Loss & Training

- Standard cross-entropy loss for node classification
- Auxiliary MPNN uses a single-layer GCN (without normalization layers) to avoid overfitting on large-scale graphs
- Projection layers and MLPs are discarded; only the self-attention module and residual connections are retained
- Surrogate gradients address the non-differentiability of the spike function

## Key Experimental Results

### Main Results

**Accuracy (%) on 9 node classification datasets**:

| Model | Cora | CiteSeer | PubMed | Co-CS | Co-Physics | Actor | arXiv | Products |
|------|------|----------|--------|-------|------------|-------|-------|----------|
| GCN | 81.6 | 71.6 | 78.8 | 92.5 | 95.7 | 30.1 | 70.4 | 75.7 |
| SGFormer | 84.5 | 72.6 | 80.3 | 91.8 | 95.9 | 37.9 | 72.6 | 72.6 |
| VQGraph | 81.1 | 74.5 | 77.1 | 93.3 | 95.0 | 38.7 | 72.4 | 78.3 |
| SpikeGT | 82.0 | 70.5 | 71.1 | 92.1 | 95.7 | 36.0 | 70.2 | OOM |
| **GT-SNT** | **84.7** | **74.0** | **80.6** | **93.7** | **96.2** | **39.1** | 72.4 | **74.8** |

**Inference speed and resource comparison**:

| Dataset | Metric | NAGphormer | GOAT | NodeFormer | SGFormer | GT-SNT | Speedup |
|--------|------|-----------|------|-----------|---------|--------|--------|
| Physics | Latency(s) | 1.79 | 10.98 | 0.14 | 0.02 | **0.02** | **130×** (vs avg) |
| Products | Latency(s) | 25.74 | 2416.84 | 41.78 | 24.34 | **20.83** | **30×** (vs avg) |
| CS | Memory(MB) | 3400 | 12490 | 2822 | 1662 | **1638** | Lowest |

### Ablation Study

| Configuration | Description | Key Observation |
|------|------|---------|
| SNT hyperparameters $(D, T)$ | $D=4, T=3$ to $D=7, T=6$ | Excessively small latent space ($D \leq 3$) degrades performance; diminishing returns for $T > 6$ |
| Codebook utilization | GOAT: 10.6%, VQGraph: 16.9% | GT-SNT: **100%**, completely eliminating codebook collapse |
| vs. spike-based methods | SpikingGCN, SpikeNet, SpikeGCL, SpikeGT | GT-SNT achieves an average improvement of **3.9%** |

### Key Findings

- **Complete elimination of codebook collapse**: GOAT and VQGraph exhibit codebook utilization below 50% when the latent space exceeds $2^{10}$, whereas GT-SNT consistently maintains 100% utilization
- **Strong performance on heterophilic graphs**: Achieves 39.1% on Actor (high heterophily), surpassing all baselines
- **A first-of-its-kind application**: SNNs are used as energy-efficient tokenizers for graph data rather than merely as low-power substitutes
- **OOM immunity**: Runs successfully on ogbn-products (2.45M nodes), while SpikingGCN, SpikeGCL, GAT, and SpikeGT all run out of memory

## Highlights & Insights

- **Cross-domain innovation**: Neuromorphic computing (SNN) is elegantly integrated with graph Transformer tokenization, introducing a novel "sequence-to-token" paradigm
- **Theoretical elegance**: Spike-count vectors naturally form a discrete codebook without explicit training or auxiliary losses (e.g., VQ-VAE's commitment loss)
- **Strong practical value**: GT-SNT is a truly deployable linear-time GT for large-scale graphs, with substantial inference acceleration
- **Comprehensive ablation**: Analyses across codebook utilization, latent space size, propagation steps, and energy consumption are thorough and well-grounded

## Limitations & Future Work

- GT-SNT does not outperform VQGraph and SGFormer on arXiv (72.4 vs. 72.4 / 72.6), indicating inconsistent advantages on large-scale graphs of varying homophily
- Evaluation is limited to node classification; effectiveness on graph classification and link prediction remains unexplored
- Random feature matrix initialization may introduce additional variance, requiring multiple runs with averaged results
- Tuning SNN hyperparameters (threshold $V_{th}$, time constant $\tau$, propagation steps $T$) may increase practical difficulty
- The 74.8% accuracy on ogbn-products falls short of VQGraph's 78.3%, suggesting that codebook-guided attention may be insufficient for extremely large-scale graphs

## Related Work & Insights

- The tokenization ideas from VQ-VAE and LFQ adapted to the graph domain; this work proposes a more natural alternative grounded in a physical process (spike firing)
- The linear attention of SGFormer combined with GT-SNT's tokenization may yield complementary benefits
- The application of SNNs to graph data is still nascent; this work demonstrates a direction that goes beyond treating SNNs as low-power drop-in replacements
- The spike-count embedding idea may also generalize to tokenization of other structured data, such as point clouds and molecular graphs

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] Spiking Heterogeneous Graph Attention Networks](spiking_heterogeneous_graph_attention_networks.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](../../ACL2026/graph_learning/evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)
- [\[AAAI 2026\] MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation](motorec_sparse-regularized_multimodal_tokenization_for_cold-start_recommendation.md)

</div>

<!-- RELATED:END -->
