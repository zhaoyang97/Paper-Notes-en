---
title: >-
  [Paper Note] GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition
description: >-
  [ECCV2024][Graph Learning][multi-label image recognition] Proposes GKGNet, the first fully graph convolutional multi-label recognition model, which dynamically constructs graph structures between labels and image regions utilizing a Group KNN mechanism, achieving SOTA performance on MS-COCO and VOC2007 with lower computational cost.
tags:
  - "ECCV2024"
  - "Graph Learning"
  - "multi-label image recognition"
  - "graph convolutional network"
  - "group KNN"
  - "label-region correlation"
date: 2026-05-08
content_hash: a490484bb076b371
---

# GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition

**Conference**: ECCV2024  
**arXiv**: [2308.14378](https://arxiv.org/abs/2308.14378)  
**Code**: [jin-s13/GKGNet](https://github.com/jin-s13/GKGNet)  
**Area**: Graph Learning  
**Keywords**: multi-label image recognition, graph convolutional network, group KNN, label-region correlation

## TL;DR

Proposes GKGNet, the first fully graph convolutional multi-label recognition model, which dynamically constructs graph structures between labels and image regions utilizing a Group KNN mechanism, achieving SOTA performance on MS-COCO and VOC2007 with lower computational cost.

## Background & Motivation

Multi-label image recognition (MLIR) requires predicting multiple object labels in an image simultaneously and modeling complex relationships between labels and spatial image regions. Existing methods possess distinct limitations:

- **CNN methods** (ResNet, SRN, etc.): Process continuous regions with sliding windows, which struggles to capture irregular and non-contiguous regions of interest (e.g., multiple scattered dogs).
- **Transformer methods** (C-Tran, Q2L, etc.): Capture complex regions through global attention but introduce significant background distraction. Especially when targets are small, the attention scores of background patches are non-negligible, and the computational overhead is high.
- **Existing GCN methods** (ML-GCN, ADD-GCN, etc.): Only utilize GCNs to model relationships between labels, while image features are still extracted via CNNs. The label embeddings and visual features do not share a unified representation space, limiting the effectiveness of message passing.

The authors observe that graph structures are naturally suited for modeling flexible connections between labels and spatially dispersed regions. Thus, they propose to unify image patches and label embeddings in a joint graph representation for processing.

## Core Problem

1. How to represent both visual features and label embeddings in a unified graph structure to explicitly model the relationship between labels and irregular regions of interest?
2. The fixed number of neighbors $K$ in traditional KNN graphs cannot adapt to objects of different scales—a large $K$ leads to over-smoothing and background distraction, while a small $K$ results in insufficient feature extraction.
3. A single distance metric struggles to fully characterize the rich semantic dimensions of "high-level" labels.

## Method

### Overall Architecture

GKGNet divides the input image into $N$ patches, mapping each patch through a fully connected layer into a $C$-dimensional feature vector to serve as a **patch node**; learnable label embeddings with the same dimension $C$ serve as **label nodes**. Both node types are processed in a unified graph structure across a four-stage hierarchical pipeline, with the number of patch nodes decreasing after each stage to extract multi-scale features.

Each stage contains two types of Group KGCN modules:

- **Patch-Level Group KGCN**: Graph convolutions among patch nodes to capture spatial semantic relationships among visual features.
- **Cross-Level Group KGCN**: Information propagation from patch nodes (sources) to label nodes (targets) to model cross-level correlations between labels and regions.

### Group KNN Mechanism

The core innovation lies in dividing the node features into $G$ groups along the channel dimension, with each group performing an independent KNN search (based on cosine similarity). This enables the actual number of source nodes connected to a target node to dynamically vary between $K$ and $K \times G$:

- **Large targets**: The selected neighbors of each group do not overlap, enabling the target node to interact with up to $K \times G$ source nodes to cover a wider area.
- **Small targets**: The neighbors of each group highly overlap, reducing the actual number of interacting nodes and effectively avoiding background distraction.

Key formula—Group max-relative graph convolution updating sub-target nodes:

$$D'_{ig} = \max(\{D_{ig} - \hat{S}_{kg} \mid k \in [1, K]\})$$

The updated sub-nodes from each group are concatenated with the original features, and then passed through a linear layer and a Feed-Forward Network (FFN, with residual connections) to produce the updated target node:

$$\widetilde{D_i} = D_i + \text{FFN}(D_i + \text{Linear}(\text{Concat}(D_i, \{D'_{ig}\})))$$

### Classifier and Loss

The final prediction combines the outputs of both patch nodes and label nodes: $Y = \text{Sigmoid}(Y_{x_p} + Y_{x_l})$. The training loss is the sum of label smoothing loss and asymmetric loss.

### Computational Complexity

The computational complexity of distance estimation in Group KGCN is $O(G \times N_S \times N_D \times C/G) = O(N_S \times N_D \times C)$, which is identical to traditional KNN, introducing no extra computational overhead.

## Key Experimental Results

### MS-COCO Main Results

| Method | Resolution | Params (M) | FLOPs (G) | mAP |
|------|--------|-----------|----------|-----|
| Q2L-R101† | 448 | 193.6 | 51.4 | 84.9 |
| TDRG | 448 | 68.3 | 42.2 | 84.6 |
| **GKGNet** | **448** | **34.0** | **21.9** | **86.7** |
| Q2L-R101† | 576 | 193.6 | 80.8 | 86.5 |
| C-Tran | 576 | 120.4 | 84.2 | 85.1 |
| **GKGNet** | **576** | **34.7** | **40.1** | **87.7** |

GKGNet achieves 86.7 mAP at 448 resolution with 34M parameters and 21.9G FLOPs, requiring only 1/6 of the parameters and less than half the FLOPs of Q2L.

### VOC2007

GKGNet achieves **96.8% mAP**, outperforming the previous SOTA Q2L (96.1%) by 0.7 percentage points, and achieving the best results in 14 out of 20 categories.

### Ablation Study

| Patch-Level | Cross-Level | Group KNN | mAP |
|:-----------:|:-----------:|:---------:|-----|
| | | | 79.9 |
| ✓ | | | 82.5 |
| ✓ | ✓ | | 85.5 |
| ✓ | ✓ | ✓ | **86.7** |

### Performance on Different Object Scales (448 Resolution)

| Method | Small | Medium | Large |
|------|-------|--------|-------|
| Q2L | 30.7 | 70.2 | 85.6 |
| **GKGNet** | **35.6** | **73.6** | **86.6** |

GKGNet outperforms Q2L by **4.9% mAP** on small objects, validating the advantage of Group KNN's adaptive neighbor selection for small objects.

### Improvement of Group KNN on General Classification

Applying Group KNN to Pyramid ViG-Tiny (without adding extra parameters or computational cost) improves ImageNet-1K top-1 accuracy from 78.2% to 79.3%, and Flowers dataset accuracy from 83.6% to 87.2%.

## Highlights & Insights

- **First fully graph convolutional multi-label recognition model**: Unifies visual patches and label embeddings in a single graph structure to achieve genuine end-to-end graph learning.
- **Exquisitely designed Group KNN mechanism**: Achieves dynamic neighbor counts via feature grouping, requiring zero extra computational overhead while accommodating both large and small objects.
- **Outstanding efficiency advantage**: Outperforms Q2L with less than half the FLOPs and 1/6 of the parameters, demonstrating high practical utility.
- **Thorough visualization validation**: The Cross-Level module adaptively focuses on target regions of different scales and can even capture co-occurring classes (e.g., car → traffic light).

## Limitations & Future Work

- Due to resource limitations, validation on large models or large-scale pre-training (e.g., ImageNet-22K) has not been performed, leaving scalability to be explored.
- Experiments are limited to MS-COCO (80 classes) and VOC (20 classes), lacking validation on larger-scale multi-label datasets (e.g., OpenImages).
- Performance saturates at Group number $G=2$, and more groups do not yield additional gains, suggesting that the grouping strategy still has room for optimization (e.g., adaptive grouping).
- Currently, label embeddings are randomly initialized. Integrating pre-trained label embeddings from language models (e.g., CLIP text encoder) could potentially yield further improvements.

## Related Work & Insights

| Category | Method | Differences with GKGNet |
|------|------|------------------|
| CNN-based | ResNet, SRN | Only uses global features for multi-binary classification, without modeling label-region relationships. |
| Transformer-based | C-Tran, Q2L | Global attention introduces background distraction, with high computational cost. |
| GCN + CNN | ML-GCN, ADD-GCN, TDRG | GCN only models relations between labels, with visual features extracted independently by CNN. The two are not in the same representation space. |
| Graph Backbones | ViG | Converts images into graph structures for classification, but fails to introduce label-region interactions. |

The core difference of GKGNet lies in its **unified graph representation**—both patches and labels dynamically interact in the same space via the Group KGCN modules.

## Inspirations & Connections

- The idea of dynamic grouping neighbors in Group KNN can be generalized to scenarios like point cloud processing and 3D object detection for handling multi-scale features.
- The concept of unified patch-label graph representation can be extended to multi-modal tasks (e.g., vision-language alignment) to replace cross-attention with graph structures.
- The cross-level graph interaction design between label embeddings and visual patches can inspire alignment methods between class embeddings and region features in open-vocabulary detection.
- The significant performance boost on small objects (+4.9% mAP) suggests that graph structures hold potential in fine-grained multi-label scenarios (e.g., attribute recognition, multi-label medical image diagnosis).

## Rating

- Novelty: ⭐⭐⭐⭐ — First fully graph convolutional MLIR model; the Group KNN dynamic neighbor mechanism design is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies, clear visualizations, and thorough multi-resolution settings; validation on larger-scale datasets is slightly lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear writing, intuitive illustrations, and thoroughly explained motivations.
- Value: ⭐⭐⭐⭐ — Provides a new paradigm for multi-label recognition; Group KNN is transferable to other graph learning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Confidence Self-Calibration for Multi-Label Class-Incremental Learning](confidence_self-calibration_for_multi-label_class-incremental_learning.md)
- [\[CVPR 2025\] DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition](../../CVPR2025/graph_learning/dvhgnn_multi-scale_dilated_vision_hgnn_for_efficient_vision_recognition.md)
- [\[ICML 2026\] Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions](../../ICML2026/graph_learning/whom_to_query_for_what_adaptive_group_elicitation_via_multi-turn_llm_interaction.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[ACL 2025\] Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning](../../ACL2025/graph_learning/disentangled_multi-span_evolutionary_network_against_temporal_knowledge_graph_re.md)

</div>

<!-- RELATED:END -->
