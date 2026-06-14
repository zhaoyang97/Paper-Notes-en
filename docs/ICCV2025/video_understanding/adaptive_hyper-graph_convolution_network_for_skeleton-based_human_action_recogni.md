---
title: >-
  [Paper Note] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition
description: >-
  [ICCV 2025][Video Understanding][Skeleton-based action recognition] This paper proposes Hyper-GCN, which replaces conventional binary graphs with an **adaptive non-uniform hypergraph** to model skeletal topology…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Skeleton-based action recognition"
  - "hypergraph convolution"
  - "adaptive topology"
  - "virtual connections"
  - "graph convolutional network"
date: 2026-05-08
content_hash: c2a12886f48a9572
---

# Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition

**Conference**: ICCV 2025
**Code**: [https://github.com/6UOOON9/Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN)  
**Area**: Video Understanding
**Keywords**: Skeleton-based action recognition, hypergraph convolution, adaptive topology, virtual connections, graph convolutional network

## TL;DR

This paper proposes Hyper-GCN, which replaces conventional binary graphs with an **adaptive non-uniform hypergraph** to model skeletal topology, and introduces **virtual hyper-joints** to create virtual connections that enable direct modeling of multi-joint cooperative relationships. The approach achieves state-of-the-art performance on NTU-60/120 and NW-UCLA with the most lightweight GCN design (base variant: only 1.1M parameters, 1.63 GFLOPs).

## Background & Motivation

The central challenge in skeleton-based action recognition is how to model topological relationships among joints. Existing methods fall into three categories:

**GCN-based methods** (ST-GCN, CTR-GCN, etc.): represent binary joint connections via adjacency matrices, but **can only model pairwise relationships between two joints**.

**Transformer-based methods** (SkateFormer, etc.): model topology through attention maps with strong performance, but at significantly higher parameter counts and GFLOPs.

**Existing hypergraph methods** (Hyper-GNN, Selective-HCN): attempt to model multi-joint relationships via hypergraphs, but **rely on fixed, hand-crafted designs**.

**Why are binary connections insufficient?** Consider the "start running" action — it is a **coordinated combination** of raising the left hand and stepping forward with the right leg. Such multi-joint cooperative relationships cannot be adequately expressed by pairwise edges. In an ordinary graph, information requires 2 layers of convolution to propagate from one joint to its 2-hop neighbors; in a hypergraph, a single convolution propagates information to all joints connected by a hyperedge, effectively **expanding the receptive field**.

**Why are fixed hypergraphs insufficient?** Existing hypergraph methods (e.g., Hyper-GNN) define hyperedges manually based on human prior knowledge. However, the critical joint combinations vary entirely across actions — "drinking water" focuses on the head and hands, while "kicking something" involves leg-hand coordination. This motivates the need to **adaptively** learn different hypergraph topologies for different actions.

**Inspiration from shadow puppetry**: The skeleton resembles a puppet, and the virtual hyper-joints correspond to the "strings" and "control points" that manipulate it. Real joints bear the dual burden of storing local features and transmitting global semantics; introducing virtual joints offloads the global information propagation task, allowing real joints to focus on local feature representation.

## Method

### Overall Architecture

Hyper-GCN consists of the following components:
- **Embedding layer**: maps raw coordinates into feature space
- **9 spatiotemporal convolution layers**: each containing M-HGC (multi-head hypergraph convolution) + MS-TC (multi-scale temporal convolution)
- **3 stages**: channel dimensions of 128/256/256 (base) or 128/256/512 (large), with dense connections within each stage
- **Classification head**: global average pooling + FC

### Key Designs

#### 1. Adaptive Non-Uniform Hypergraph Construction (A-NHG)

**Core Idea**: Rather than fixing the hyperedge structure, the hypergraph is **adaptively constructed** based on distances between joint features in semantic space.

Given joint features $X \in \mathbb{R}^{N \times C}$, a mapping function $\Phi$ first projects them into a subspace $X_H \in \mathbb{R}^{N \times C_h}$, and then the Euclidean distance matrix between joints is computed:

$$m_{i,j} = m_{j,i} = \|v_i - v_j\|_2$$

**Why Euclidean distance rather than cosine similarity?** Euclidean distance better captures the absolute positional relationships between joints in feature space, and computing distances in the projected subspace preserves the spatial correlations of the original features.

For each joint $i$, only the hyperedges corresponding to the $K$ nearest joints are retained, converted to probabilities via softmax:

$$h_{i,j} = \begin{cases} \frac{\exp(-m_{i,j})}{\sum_{k \in \text{set}_i} \exp(-m_{i,k})}, & j \in \text{set}_i \\ 0, & j \notin \text{set}_i \end{cases}$$

**Why "non-uniform"?** Unlike a uniform hypergraph (where each hyperedge contains a fixed $K$ joints), A-NHG constrains the number of hyperedges each joint **is included in** to at most $K$. Consequently, different hyperedges contain varying numbers of joints, allowing hyperedges to capture more diverse joint combination patterns.

#### 2. Multi-Head Hypergraph Convolution (M-HGC)

Features are divided into 8 heads along the channel dimension; each head independently constructs a hypergraph and performs convolution:

$$F_{out} = \bigoplus_{k=1}^{8} (\hat{A}^k + \alpha \cdot \hat{H}^k) F_{in}^k P^k$$

where $\hat{A}$ is the normalized physical adjacency matrix, $\hat{H}$ is the normalized hypergraph incidence matrix, and $\alpha$ is a learnable fusion parameter.

**Why retain physical topology?** Although the hypergraph can capture high-order relationships, the physical connections of the body (bones) remain the fundamental structural constraint of motion. By fusing physical and learned topologies in a weighted manner, the model can explore implicit relationships while maintaining physical plausibility.

**Why 8 heads?** Different channel groups may attend to different semantics (position, velocity, pose, etc.); the multi-head design gives each channel group its own dedicated hypergraph topology.

Hypergraph weights are learned via an MLP with LeakyReLU + Tanh activations, constraining weights to the range $[-1, 1]$ — thereby permitting **inhibitory connections**, a capability absent in standard GCNs.

#### 3. Virtual Hyper-Joints and Connections

Learnable hyper-joints $F_h \in \mathbb{R}^{C \times T \times V_h}$ (where $V_h$ is the number of hyper-joints) are introduced and participate in hypergraph convolution alongside real joints:

- Hyper-joints are shared across frames along the temporal dimension (aligned to the temporal dimension)
- Each layer maintains independent hyper-joints (to match the feature hierarchy at different depths)
- Hyper-joints participate only in spatial hypergraph convolution, not temporal convolution
- Hyper-joints are manually connected to all physical joints

**Divergence Loss**: To prevent hyper-joint homogenization, the cosine similarity matrix is used to measure diversity among hyper-joints:

$$\mathcal{L}_h(C) = \frac{\sum_{i=1}^{V_h} \sum_{j=1}^{V_h} \text{ReLU}(c_{i,j}) - V_h}{V_h(V_h - 1)}$$

$$\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{L} \sum_{l=1}^{L} \mathcal{L}_h(C^l)$$

### Loss & Training

- **Primary loss**: label-smoothed cross-entropy loss $\mathcal{L}_{CE}$
- **Auxiliary loss**: divergence loss $\mathcal{L}_h$, averaged over layers and added to the primary loss
- **Optimizer**: SGD + Nesterov momentum (0.9), weight decay 0.0004
- **Training schedule**: 140 epochs with 5-epoch warm-up
- **Learning rate**: initial 0.05, decayed to 0.005 at epoch 110 and 0.0005 at epoch 120
- **Multi-stream ensemble**: 4-stream (joint + bone + joint motion + bone motion)
- **Hardware**: single RTX 3090

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods (4-stream ensemble):

| Method | Type | Params (M) | GFLOPs | NTU60-XSub | NTU60-XView | NTU120-XSub | NTU120-XSet | NW-UCLA |
|--------|------|-----------|--------|------------|-------------|-------------|-------------|---------|
| CTR-GCN | GCN | 1.5 | 1.97 | 92.4 | 96.4 | 88.9 | 90.6 | 96.5 |
| InfoGCN | GCN | 1.6 | 1.84 | 92.7 | 96.9 | 89.4 | 90.7 | 96.6 |
| BlockGCN | GCN | 1.3 | 1.63 | 93.1 | 97.0 | 90.3 | 91.5 | 96.9 |
| SkateFormer | Trans | 2.0 | 3.62 | 93.5 | **97.8** | 89.8 | 91.4 | **98.3** |
| DST-HCN | HGCN | 3.5 | 2.93 | 92.3 | 96.8 | 88.8 | 90.7 | 96.6 |
| **Ours (B)** | HGCN | **1.1** | **1.63** | 93.3 | 97.4 | 90.5 | 91.7 | 97.2 |
| **Ours (L)** | HGCN | 2.3 | 2.88 | **93.7** | **97.8** | **90.9** | **92.0** | 97.6 |

Key observations:
- The base variant surpasses all GCN and hypergraph methods with the **fewest parameters** (1.1M) and **lowest GFLOPs** (1.63)
- The large variant outperforms SkateFormer, the most parameter-efficient Transformer method, on NTU120

### Ablation Study

Effect of A-NHG hyperparameter $K$ (NTU120 X-Sub, single-stream joint):

| K | Uniform Hypergraph (%) | Non-Uniform Hypergraph (%) |
|---|----------------------|---------------------------|
| Baseline | 84.7 | 84.7 |
| 3 | 86.4 (+1.7) | 86.0 (+1.3) |
| 5 | **86.5 (+1.9)** | 86.2 (+1.5) |
| 7 | 86.3 (+1.6) | 86.5 (+1.8) |
| 9 | 86.0 (+1.3) | **86.7 (+2.2)** |
| 11 | 85.9 (+1.2) | 86.4 (+1.7) |

Ablation on virtual hyper-joints (NTU120 X-Sub):

| # Hyper-joints | w/o M-HGC, w/o Div. Loss | w/o M-HGC, w/ Div. Loss | w/ M-HGC, w/o Div. Loss | w/ M-HGC, w/ Div. Loss |
|---------------|--------------------------|--------------------------|--------------------------|-------------------------|
| 0 | 84.7 | 84.7 | — | — |
| 1 | 84.9 | 84.9 | 86.7 | 86.7 |
| 3 | 84.9 | **85.2** | 86.6 | **86.9** |
| 5 | 84.7 | 85.0 | 86.6 | 86.8 |

### Key Findings

1. **Non-uniform hypergraph is optimal at $K=9$**: Unlike the uniform hypergraph, which peaks at $K=5$, the non-uniform structure tolerates more connections without introducing excessive noise.
2. **3 hyper-joints is optimal**: Too many hyper-joints introduce redundancy and ambiguous cues, degrading performance.
3. **Divergence loss is critical for multiple hyper-joints**: Visualization shows severe homogenization of hyper-joints without the divergence loss (cosine similarity matrix approaching all ones).
4. **Hypergraph construction aligns with action semantics**: Visualization shows that during "kicking something," hyperedges connect the left leg and right hand (co-moving body parts), while during "standing up," hyperedges connect load-bearing joints.
5. **t-SNE reveals semantic convergence**: Joint features in the final layer of Hyper-GCN form tight clusters, indicating more thorough information exchange.

## Highlights & Insights

1. **Adaptive hypergraph is the core innovation**: All existing hypergraph methods rely on fixed structures; A-NHG dynamically constructs the hypergraph from feature distances, achieving truly data-driven topology learning.
2. **The "puppet string" analogy for virtual joints**: This intuition clearly explains why additional information carriers are needed to relieve the burden on real joints — analogous in spirit to the class token in Transformers.
3. **Exceptional efficiency–performance balance**: The base variant surpasses numerous GCN methods with only 1.1M parameters and 1.63 GFLOPs, demonstrating the superior information aggregation efficiency of hypergraph convolution.
4. **Simplicity and effectiveness of the divergence loss**: The straightforward design of cosine similarity + ReLU clipping effectively resolves the hyper-joint homogenization problem.

## Limitations & Future Work

1. **Validation limited to standard benchmarks**: NTU and NW-UCLA primarily cover indoor single-person or two-person actions; validation in complex multi-person and outdoor scenarios is absent.
2. **Fixed number of hyper-joints**: Currently set manually to 3; whether this count can be determined adaptively based on action complexity remains an open question.
3. **Continued reliance on multi-stream ensemble**: The 4-stream setup increases overall system complexity and inference cost.
4. **No comparison with recent Mamba/state-space models**: More efficient alternatives may exist for temporal modeling of skeleton sequences.
5. **Sensitivity of $K$ across datasets not analyzed**: The ablation on $K$ is conducted solely on NTU120.

## Related Work & Insights

- **CTR-GCN (ICCV 2021)**: Proposes channel-wise topology refinement; this paper extends the idea to multi-joint hyperedges.
- **InfoGCN (CVPR 2022)**: GCN from an information bottleneck perspective; complementary to Hyper-GCN in terms of information aggregation efficiency.
- **BlockGCN**: The current most parameter-efficient GCN state-of-the-art, surpassed by Hyper-GCN Base with even fewer parameters.
- **SkateFormer (ECCV 2024)**: The latest advance in Transformer-based approaches with strong performance, but requiring more than twice the computation of Hyper-GCN.
- **HGNN+ (TPAMI 2023)**: Provides a general framework for hypergraph convolution, inspiring the normalization scheme adopted in this paper.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections](adaptive_hyper_graph_convolution_network_skeleton_action_recognition.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ICCV 2025\] DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition](despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)

</div>

<!-- RELATED:END -->
