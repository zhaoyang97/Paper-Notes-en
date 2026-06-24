---
title: >-
  [Paper Note] Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections
description: >-
  [ICCV 2025][Video Understanding][Skeleton-based action recognition] This paper proposes Hyper-GCN, which transcends the limitation of conventional GCNs that model only binary pairwise joint relationships, by introducing adaptive non-uniform hypergraph convolution and virtual hyper joints. The design enables efficient aggregation of multi-joint collaborative semantics, achieving state-of-the-art performance on NTU-60/120 and NW-UCLA benchmarks with the most lightweight GCN arc…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Skeleton-based action recognition"
  - "Hypergraph convolution"
  - "Virtual connections"
  - "Adaptive topology"
  - "Graph convolutional network"
date: 2026-05-08
content_hash: 73efb8e205d10966
---

# Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections

**Conference**: ICCV 2025
**Code**: [https://github.com/6UOOON9/Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN)  
**Area**: Video Understanding
**Keywords**: Skeleton-based action recognition, Hypergraph convolution, Virtual connections, Adaptive topology, Graph convolutional network

## TL;DR

This paper proposes Hyper-GCN, which transcends the limitation of conventional GCNs that model only binary pairwise joint relationships, by introducing adaptive non-uniform hypergraph convolution and virtual hyper joints. The design enables efficient aggregation of multi-joint collaborative semantics, achieving state-of-the-art performance on NTU-60/120 and NW-UCLA benchmarks with the most lightweight GCN architecture to date.

## Background & Motivation

1. **Background**: Skeleton-based action recognition is a core task in video understanding. Existing methods are broadly categorized into two paradigms: GCN-based methods (e.g., CTR-GCN, BlockGCN), which model skeletal joint relationships via graph topology with high parameter efficiency but are constrained to binary connections, and Transformer-based methods (e.g., SkateFormer), which model joint relationships via attention with strong expressiveness at higher computational cost.

2. **Limitations of Prior Work**: Existing GCN methods rely on adjacency matrices of normal graphs to represent joint topology, which can only encode binary pairwise relationships between two joints. However, human actions inherently involve multi-joint coordination (e.g., "starting to run" requires simultaneous lifting of the left hand and stepping forward with the right leg). Binary connections fail to capture such higher-order collaborative relationships. Although a few works have explored hypergraphs, they rely on fixed, manually designed structures that lack adaptability.

3. **Key Challenge**: While hypergraphs can express multi-joint relationships, the key challenge lies in adaptively constructing action-relevant hypergraph structures. Fixed, manually designed hypergraphs cannot accommodate the topological variation across different action categories. Furthermore, since the number of skeletal joints is fixed (e.g., 25 in NTU), each joint must both store local context and transmit global semantics, imposing a limited information capacity per node.

4. **Goal**: To design an adaptive hypergraph convolutional network that dynamically constructs optimal hypergraph topologies conditioned on input data, and to expand the capacity and pathways of information propagation by introducing virtual nodes.

5. **Key Insight**: Inspired by puppetry (shadow puppets and marionettes), where motion is driven by virtual connections (strings) that manipulate real joints, the authors propose introducing "hyper joints" as virtual manipulation points. These are connected to real joints via hyperedges, simultaneously expanding information propagation pathways and relieving individual joints from the burden of storing global semantics.

6. **Core Idea**: The Adaptive Non-uniform Hypergraph (A-NHG) dynamically constructs hyperedges via K-nearest neighbors in feature space; Multi-Head Hypergraph Convolution (M-HGC) independently constructs hypergraph topologies in different channel subspaces; virtual hyper joints supplement global semantic propagation capacity.

## Method

### Overall Architecture

Hyper-GCN consists of an embedding layer followed by 9 spatial-temporal convolutional layers organized into 3 stages with dense connections within each stage. Each layer comprises two modules: Multi-Head Hypergraph Convolution (M-HGC) and Multi-Scale Temporal Convolution (MS-TC). The input is a skeleton sequence feature $F_{in} \in \mathbb{R}^{C \times T \times V}$, and the output is an action classification result.

### Key Designs

1. **Adaptive Non-uniform Hypergraph Construction (A-NHG)**:

    - Function: Dynamically constructs action-relevant hypergraph topology conditioned on input features.
    - Mechanism: Given joint features $X \in \mathbb{R}^{N \times C}$, a projection function $\Phi$ maps them to a subspace $X_H \in \mathbb{R}^{N \times C_h}$, after which the Euclidean distance matrix between joints is computed as $m_{i,j} = \|v_i - v_j\|_2$. For each joint $i$, only the $K$ nearest joints are retained to form a hyperedge, and distances are converted to probabilities via softmax: $h_{i,j} = \exp(-m_{i,j}) / \sum_{k \in set_i} \exp(-m_{i,k})$. The resulting hypergraph is non-uniform — each hyperedge contains a variable number of joints — enabling more differentiated association patterns.
    - Design Motivation: The distinction between uniform hypergraphs (each hyperedge contains exactly $K$ joints) and non-uniform hypergraphs (each joint is constrained to belong to at most $K$ hyperedges) lies in the latter allowing variable hyperedge sizes, which more flexibly accommodates the topological requirements of different actions.

2. **Multi-Head Hypergraph Convolution (M-HGC)**:

    - Function: Independently constructs hypergraphs in different channel subspaces to capture multi-scale semantic relationships.
    - Mechanism: The input features are split into 8 heads along the channel dimension; each head independently constructs a hypergraph and performs hypergraph convolution. A temporal average pooling is first applied to obtain spatial features $\bar{F}_{in}$, after which each head constructs an A-NHG via an independent projection to obtain the incidence matrix $H$, and an MLP is used to learn hyperedge weights $W$. The convolution operation is $F_{out} = \oplus_{k=1}^8 (\hat{A}^k + \alpha \cdot \hat{H}^k) F_{in}^k P^k$, where $\hat{A}$ denotes the physical topology, $\hat{H}$ the hypergraph topology, and $\alpha$ a learnable fusion weight.
    - Design Motivation: Features in different channel subspaces may reflect distinct types of semantic relationships between joints (e.g., positional relationships, motion trends), and the multi-head design allows the model to capture this diversity.

3. **Virtual Hyper Joints**:

    - Function: Expand information propagation pathways and relieve real joints from the burden of storing global semantics.
    - Mechanism: $V_h$ learnable hyper joints $F_h \in \mathbb{R}^{C \times T \times V_h}$ are introduced, sharing the same shape as real joint features and shared across frames. Hyper joints are manually connected to all physical joints and participate in spatial hypergraph convolution but not temporal convolution. To prevent hyper joint homogenization, a diversity loss is designed: $L_h = (\sum_{i,j} \text{ReLU}(c_{i,j}) - V_h) / (V_h(V_h - 1))$, where $c_{i,j}$ denotes the cosine similarity between hyper joints. Independent hyper joints are assigned to each layer to accommodate features at different depths.
    - Design Motivation: By analogy with the CLS token in Transformers, hyper joints serve as relay stations for global information, allowing real joints to focus on storing local neighborhood features while delegating global information propagation to the hyper joints.

### Loss & Training

- Total loss: $L = L_{CE} + \frac{1}{L}\sum_{l=1}^L L_h(C_l)$, comprising cross-entropy and per-layer diversity losses
- Label-smoothed cross-entropy
- SGD optimizer, Nesterov momentum=0.9, weight decay=0.0004
- 140 epochs, 5-epoch warmup, initial lr=0.05, decayed to 0.005 at epoch 110

## Key Experimental Results

### Main Results

| Method | Type | Params (M) | NTU60-XSub (%) | NTU120-XSub (%) | NW-UCLA (%) |
|--------|------|-----------|----------------|-----------------|-------------|
| CTR-GCN | GCN | 1.5 | 92.4 | 88.9 | 96.5 |
| BlockGCN | GCN | 1.3 | 93.1 | 90.3 | 96.9 |
| HD-GCN | GCN | 1.7 | 93.0 | 89.8 | 96.9 |
| SkateFormer | Trans. | 2.0 | 93.5 | 89.8 | 98.3 |
| DST-HCN | HGCN | 3.5 | 92.3 | 88.8 | 96.6 |
| **Ours (Base)** | HGCN | **1.1** | **93.3** | **90.5** | 97.2 |
| **Ours (Large)** | HGCN | **2.3** | **93.7** | **90.9** | 97.6 |

### Ablation Study

| Configuration | NTU120 XSub (%) | Note |
|---------------|-----------------|------|
| Baseline (normal GCN) | 84.7 | No hypergraph |
| + M-HGC (K=9, non-uniform) | 86.7 (+2.0) | Largest contribution from hypergraph convolution |
| + 3 hyper joints w/o $L_h$ | 86.6 (−0.1) | Hyper joints homogenize without diversity loss |
| + 3 hyper joints w/ $L_h$ | **86.9** (+0.2) | Diversity loss promotes differentiation |

### Key Findings

- The non-uniform hypergraph achieves peak performance at K=9 (86.7%), whereas the uniform hypergraph peaks at K=5 (86.5%), indicating greater flexibility of the non-uniform design.
- Virtual hyper joints yield limited gains in the absence of hypergraph convolution (+0.2), but produce substantial improvements when combined with it (+2.2).
- The diversity loss is critical for configurations with 3 or more hyper joints; without it, hyper joints collapse to homogeneous representations and become ineffective.
- Hyper-GCN (Base) surpasses all GCN-based methods with only 1.1M parameters, achieving the highest parameter efficiency.

## Highlights & Insights

- **Hypergraph modeling of multi-joint coordination**: The approach transcends the binary pairwise constraint of normal graphs; hyperedges naturally encode multi-joint collaborative semantics (e.g., "running" involves coordinated movement of both arms and legs), offering a more structured modeling paradigm than attention mechanisms.
- **Elegant design of virtual hyper joints**: By analogy with the strings of a puppet, virtual nodes serve as relay stations for global information and are connected to real joints via hyperedges, achieving a clean decoupling of local and global information. This concept is transferable to graph neural networks in other domains.
- **Extreme parameter efficiency**: SOTA performance is achieved with only 1.1M parameters — smaller than comparable GCN methods (1.3–1.7M) and several times smaller than Transformer-based methods (2.0–3.5M).

## Limitations & Future Work

- The hyperparameter $K$ requires manual selection; although ablation studies provide guidance, different datasets may require different values.
- The number of virtual hyper joints also needs to be manually specified; future work could explore adaptive determination.
- The proposed method does not reach SkateFormer's 98.3% on NW-UCLA, suggesting potentially weaker generalization on smaller datasets.

## Related Work & Insights

- **vs. CTR-GCN**: CTR-GCN learns channel-wise topology but retains binary relationships; this work extends to higher-order multi-joint relationships via hypergraphs.
- **vs. DST-HCN**: DST-HCN employs manually designed fixed hypergraphs, whereas the proposed A-NHG adaptively constructs topology conditioned on input features.
- **vs. SkateFormer**: SkateFormer implicitly models relationships via attention, while this work explicitly models them via hypergraphs, with significantly fewer parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of adaptive hypergraphs and virtual hyper joints is original
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + detailed ablations + visualization analysis
- Writing Quality: ⭐⭐⭐⭐ The puppetry analogy is vivid and intuitive; mathematical derivations are clear
- Value: ⭐⭐⭐⭐ Achieves SOTA with minimal parameters, offering a new direction for skeleton-based action recognition

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ICCV 2025\] DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition](despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)

</div>

<!-- RELATED:END -->
