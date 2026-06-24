---
title: >-
  [Paper Note] DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition
description: >-
  [CVPR 2025][Graph Learning][Hypergraph Neural Networks] This paper proposes DVHGNN, a vision backbone network that utilizes multi-scale dilated hypergraphs to capture high-order correlations among image patches. By employing clustering and Dilated Hypergraph Construction (DHGC) to extract multi-scale hyperedges, alongside dynamic hypergraph convolution for adaptive feature exchange, DVHGNN achieves an 83.1% top-1 accuracy on ImageNet-1K with 30.2M parameters…
tags:
  - "CVPR 2025"
  - "Graph Learning"
  - "Hypergraph Neural Networks"
  - "Multi-Scale Dilated Hypergraph"
  - "Vision Backbone"
  - "Dynamic Hypergraph Convolution"
  - "High-Order Correlation"
date: 2026-05-08
content_hash: 653d585801a8f6e0
---

# DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition

**Conference**: CVPR 2025  
**arXiv**: [2503.14867](https://arxiv.org/abs/2503.14867)  
**Code**: None (not provided in the paper)  
**Area**: Graph Learning / Vision Backbone Networks  
**Keywords**: Hypergraph Neural Networks, Multi-Scale Dilated Hypergraph, Vision Backbone, Dynamic Hypergraph Convolution, High-Order Correlation

## TL;DR
This paper proposes DVHGNN, a vision backbone network that utilizes multi-scale dilated hypergraphs to capture high-order correlations among image patches. By employing clustering and Dilated Hypergraph Construction (DHGC) to extract multi-scale hyperedges, alongside dynamic hypergraph convolution for adaptive feature exchange, DVHGNN achieves an 83.1% top-1 accuracy on ImageNet-1K with 30.2M parameters, outperforming ViG-S by 1.0% and ViHGNN-S by 0.6%.

## Background & Motivation

**Background**: Vision GNN (ViG) pioneered representing images as graph structures and processing them with GNNs, but it faces two major challenges. Meanwhile, ViHGNN attempts to use hypergraphs to capture high-order relationships, but gains remain limited.

**Limitations of Prior Work**: (1) ViG relies on KNN graph construction, which yields quadratic computational complexity and is non-learnable, potentially leading to the loss of key information; (2) Ordinary graphs can only model pairwise relationships and fail to capture high-order correlations among multiple nodes; (3) ViHGNN constructs hypergraphs using fuzzy C-means clustering, which lacks multi-scale information and cannot dynamically adapt during the learning process.

**Key Challenge**: There is a need to capture high-order relationships (via hypergraphs), but existing hypergraph construction methods either overlook multi-scale properties or suffer from excessive computational complexity.

**Goal**: How to efficiently construct hypergraph-based visual representations that capture multi-scale high-order relationships.

**Key Insight**: A dual-path hypergraph representation combining clustering (to capture global semantic grouping) and dilated hypergraph construction (to capture multi-scale local spatial relationships).

**Core Idea**: Constructing a multi-scale hypergraph using clustering and local hyperedges with different dilation rates, and performing dynamic hypergraph convolution via cosine similarity and sparsity-aware weights.

## Method

### Overall Architecture
The image is split into $N$ patches acting as vertices. Each block comprises: multi-scale hypergraph construction (clustering hyperedges + dilated hyperedges) $\rightarrow$ two-stage dynamic hypergraph convolution (vertex convolution aggregating to hyperedges $\rightarrow$ hyperedge convolution distributing back to vertices) $\rightarrow$ ConvFFN for enhanced feature transformation. It adopts a hierarchical isotropic structure similar to ViG.

### Key Designs

1. **Clustering + Dilated Hypergraph Construction (DHGC)**:

    - **Function**: Adaptively obtaining multi-scale hyperedge sets
    - **Mechanism**: Dual-path design. Clustering path: map patch features to a similarity space and assign them to $C$ cluster centers via cosine similarity to form semantic-level hyperedges $\mathcal{E}_c$. Dilation path: for the center vertex $v_c$ in each $w \times w$ window, construct local hyperedges with dilation rates $r=1,2,3$ (corresponding to $3 \times 3$, $5 \times 5$, and $7 \times 7$ receptive fields), where each dilated hyperedge is associated with a learnable sparsity-aware weight $w_r$. A region partitioning mechanism (similar to Swin Transformer windows) is introduced to reduce the complexity from $O(NCD)$ to $O(NCD/m)$
    - **Design Motivation**: Clustering hyperedges capture global semantic similarity but overlook spatial locality, whereas dilated hyperedges capture multi-scale spatial relationships but are limited in range. The two paths are complementary

2. **Dynamic Hypergraph Convolution (DHConv)**:

    - **Function**: Adaptive feature exchange and fusion
    - **Mechanism**: Two-stage message passing. Vertex convolution phase: For clustering hyperedges, aggregate vertex features to hyperedges using sigmoid-weighted cosine similarity ($h_e = \frac{1}{C}(h_c + \sum \text{sig}(\alpha s_i + \beta) x_i)$); for dilated hyperedges, aggregate with learnable sparse weights $w_r$. Hyperedge convolution phase: Distribute hyperedge features back to vertices using cosine similarity or sparse weights, and update using GIN-style formulation: $x'_i = FC(\sigma(\text{Conv}((1+\varepsilon)x_i + z_i)))$
    - **Design Motivation**: Different types of hyperedges require distinct aggregation strategies—semantic hyperedges are suited for similarity-based soft assignment, while spatial hyperedges are suited for weight-based fixed assignment

3. **ConvFFN + Multi-Head Mechanism**:

    - **Function**: Enhancing feature transformation capability and relieving over-smoothing
    - **Mechanism**: A ConvFFN (a feed-forward network with convolution, similar to ViG) is appended after the hypergraph convolution to expand the local receptive feld. A multi-head mechanism is applied by grouping features to perform hypergraph convolutions independently and then concatenating them, enhancing representational diversity
    - **Design Motivation**: Pure hypergraph convolutions can cause node representations to converge (over-smoothing). ConvFFN mitigates this issue by introducing non-linearity and local inductive bias

### Loss & Training
Standard ImageNet classification training (cross-entropy loss + label smoothing + conventional augmentations like Mixup).

## Key Experimental Results

### Main Results

| Model | Type | Params (M) | FLOPs (G) | Top-1 Acc |
|------|------|---------|----------|----------|
| DeiT-S | ViT | 22.1 | 4.6 | 79.8% |
| Swin-S | ViT | 50.0 | 8.7 | 83.0% |
| ViG-S | GNN | 27.3 | 4.6 | 82.1% |
| ViHGNN-S | HGNN | 28.5 | 6.3 | 82.5% |
| **DVHGNN-S** | **HGNN** | **30.2** | **5.2** | **83.1%** |
| **DVHGNN-B** | **HGNN** | **92.8** | **16.8** | **84.2%** |

Downstream tasks: Consistent improvements are also achieved in COCO object detection/segmentation and ADE20K semantic segmentation.

### Ablation Study

| Configuration | Top-1 Acc |
|------|----------|
| Clustering hyperedges only | 82.3% |
| Dilated hyperedges only | 82.5% |
| Clustering + Dilated (Full) | **83.1%** |
| Without dynamic weights (fixed uniform aggregation) | 82.7% |
| Fixed window partition vs No partition | 83.1% vs OOM |

### Key Findings
- The dual-path hypergraph (clustering + dilated) outperforms either single-path variant (83.1% vs 82.3%/82.5%), proving that semantic and spatial information are complementary.
- Dynamic hypergraph convolution (based on cosine similarity) yields a 0.4% gain over fixed aggregation, demonstrating the effectiveness of adaptive weights.
- DVHGNN-S has 3.0M more parameters than ViG-S, but remains highly FLOP-efficient (5.2G vs 4.6G), achieving a 1.0% higher accuracy.
- The region partitioning strategy significantly reduces memory consumption while maintaining performance.

## Highlights & Insights
- **Dilated Hypergraphs Analogous to Dilated Convolutions**: Introducing the multi-scale concept of dilated convolutions to hypergraph construction is an intuitive and effective design, where different dilation rates correspond to spatial relationships at different scales.
- **Dual-Path Complementary Design**: The combination of clustering hyperedges (global semantics) and dilated hyperedges (local space) is conceptually similar to the dual-stream approach in SlowFast networks, but is implemented more naturally within the hypergraph framework.
- **Feasibility of Hypergraph GNNs as General Vision Backbones**: Outperforming Swin-S in a comparable accuracy range on ImageNet demonstrates the strong potential of the hypergraph paradigm.

## Limitations & Future Work
- The parameter count and FLOPs are still higher than those of ViG-S, leaving room for efficiency optimization.
- The number of cluster centers $C$ and the dilation rates $R$ are manually designed hyperparameters.
- Comparisons with modern linear-complexity architectures such as Mamba are missing.
- Hypergraph construction is conducted independently at each layer, lacking cross-layer propagation of hypergraph structures.

## Related Work & Insights
- **vs ViG**: ViG utilizes KNN to build ordinary graphs (with quadratic complexity and pairwise relationships), whereas DVHGNN uses clustering and dilation to construct hypergraphs (with linear complexity and high-order relationships).
- **vs ViHGNN**: ViHGNN relies on fuzzy C-means clustering (global and non-adaptive), whereas DVHGNN applies cosine-similarity clustering and dilated hypergraphs (multi-scale and adaptive).
- **vs Swin Transformer**: Swin performs self-attention within windows, while DVHGNN executes hypergraph convolutions within windows, enabling the latter to capture high-order relationships.

## Rating
- Novelty: ⭐⭐⭐⭐ The dilated hypergraph construction and dual-path design are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation covering ImageNet, COCO, and ADE20K.
- Writing Quality: ⭐⭐⭐ Rich in content but slightly verbose in structure.
- Value: ⭐⭐⭐⭐ Advances the development of hypergraph-based vision architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hypergraph Vision Transformers: Images are More than Nodes, More than Edges](hypergraph_vision_transformers_images_are_more_than_nodes_more_than_edges.md)
- [\[NeurIPS 2025\] The Underappreciated Power of Vision Models for Graph Structural Understanding](../../NeurIPS2025/graph_learning/the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](../../ICML2025/graph_learning/open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [\[ACL 2025\] M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](../../ACL2025/graph_learning/m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)
- [\[ECCV 2024\] GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition](../../ECCV2024/graph_learning/gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)

</div>

<!-- RELATED:END -->
