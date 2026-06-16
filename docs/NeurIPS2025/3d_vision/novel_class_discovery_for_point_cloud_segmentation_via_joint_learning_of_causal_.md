---
title: >-
  [Paper Note] Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning
description: >-
  [NeurIPS 2025][3D Vision][Novel Class Discovery] This paper is the first to introduce causal learning into 3D point cloud novel class discovery (3D-NCD). By leveraging a Structural Causal Model (SCM) to analyze confounde…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Novel Class Discovery"
  - "Point Cloud Semantic Segmentation"
  - "Causal Learning"
  - "Structural Causal Model"
  - "Graph Convolutional Network"
date: 2026-05-08
content_hash: e0db2f8498ea8cfe
---

# Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2510.13307](https://arxiv.org/abs/2510.13307)  
**Code**: Unavailable  
**Area**: 3D Vision
**Keywords**: Novel Class Discovery, Point Cloud Semantic Segmentation, Causal Learning, Structural Causal Model, Graph Convolutional Network

## TL;DR

This paper is the first to introduce causal learning into 3D point cloud novel class discovery (3D-NCD). By leveraging a Structural Causal Model (SCM) to analyze confounders in base classes and causal relationships between base and novel classes, it proposes Causal Representation Prototype learning (CRP, which eliminates confounders via an adversarial network) and graph-based causal reasoning (GCN-based pseudo-label generation), achieving state-of-the-art results on SemanticKITTI and SemanticPOSS.

## Background & Motivation

Traditional point cloud semantic segmentation adopts a "closed-world" assumption—only categories seen during training can be segmented. **3D Novel Class Discovery (3D-NCD)** aims to learn a model capable of segmenting unlabeled novel classes using only supervision from annotated base classes. This is critical for applications such as autonomous driving and robotics that must handle unknown categories.

The key challenges of this task are:

**Accurately establishing associations between point representations and base class labels**

**Establishing representational associations between base and novel classes**

Existing methods face two core problems:

**Problem 1: Shortcut Features interfere with base class learning.** Base class classifiers are fundamentally statistical models that tend to learn spurious surface-level correlations rather than intrinsic causal mechanisms. For example, a model may associate the visually salient but non-causal feature "circular support structure" with "stool," causing it to misclassify "chair"—which shares a similar structure—as "stool" during novel class inference. A causal learning approach would instead identify "chair legs" as the causally relevant discriminative feature.

**Problem 2: Lack of causal relationship modeling between base and novel classes.** Novel classes are often variants of base classes under some causal mechanism (e.g., "rider" is a causal variation of "person" in the context of "bicycle"; "truck" and "car" share a causal prior of "vehicle"). Existing methods rely solely on statistical similarity without explicitly modeling such causal transmission paths.

The authors reformulate 3D-NCD via SCM using four causal variables: raw point cloud $X$, base class $B$, novel class $N$, and confounder $U$. The path $U \to B$ represents confounders influencing base class learning, and $B \to N$ represents the causal influence of base classes on novel classes. The objective is to eliminate $U$ and establish causal reasoning along $B \to N$.

## Method

### Overall Architecture

The method comprises three core modules:
1. **Causal Representation Prototype learning (CRP)**: Eliminates confounders via an adversarial mechanism to obtain causally grounded prototype representations of base classes.
2. **Causal Reasoning Graph construction (CRG)**: Explicitly models the causal relationship $B \to N$ using a graph structure.
3. **GCN-based Pseudo-Label generation (GCPL)**: Generates pseudo-labels for novel classes via graph convolutional networks.

### Key Designs

1. **Causal Representation Prototype Learning (CRP)**: The core objective is to learn a feature representation $Z$ that is independent of the confounder $U$ (i.e., $Z \perp U$), minimizing the mutual information $I(Z;U)$. This adheres to the Independent Causal Mechanisms (ICM) principle—the true causal mechanism generating base classes should be independent of the mechanism producing confounders. This is achieved via GAN-style adversarial training:

$$\min_\theta \max_\phi \mathcal{L}_{ADV} = \mathcal{L}_{cls}(f_\theta(X_B), Y_B) - \lambda_{adv} \mathcal{L}_{adv}(g_\phi(Z), U)$$

The feature extractor $f_\theta$ attempts to extract causal features from which $U$ has been removed, while the adversarial network $g_\phi$ attempts to recover $U$ from $Z$ (successful recovery indicates $Z$ still contains confounding information). $\mathcal{L}_{cls}$ is a cross-entropy classification loss and $\mathcal{L}_{adv}$ is a binary cross-entropy adversarial loss. After training, prototypes are iteratively updated using soft assignment weights $W_{ij} = \frac{\exp(\text{sim}(Z_j, C_i))}{\sum_k \exp(\text{sim}(Z_j, C_k))}$ as $C_i^{(t+1)} = \frac{\sum_j W_{ij} \cdot Z_j}{\sum_j W_{ij}}$. The prototype matching loss is: $\mathcal{L}_{PRO} = -\sum_i \sum_j W_{ij} \cdot \text{sim}(Z_j, C_i) + \lambda \|C_i\|_2^2$.

2. **Causal Reasoning Graph Construction (CRG)**: Inspired by Causal Bayesian Networks (CBN) and the Causal Markov Condition, a directed graph is constructed to model the causal path $B \to N$. Graph nodes include $M$ base class causal prototypes $C$ and $K$ novel class prototypes $N$. A **causal adaptive adjacency matrix** $A = [A_{ij}]_{M \times K}$ is introduced, with edge weights dynamically adjusted via a self-attention mechanism: $w_{ij} = \text{softmax}(\text{Attention}(c_i, n_j)/\tau)$. Two key constraints enforce causal validity:

    - **Reasoning Direction Consistency Constraint**: Ensures information flows in the causal direction (base → novel): $\mathcal{L}_{direction} = \sum_{(c_i, n_j) \in E} (w_{ij} \cdot (1 - \mathbb{I}(c_i \to n_j)))^2$
    - **Causal Pruning Constraint**: Removes edges whose causal weight falls below a learnable threshold $\theta$: $\mathcal{L}_{pruning}(\theta) = \sum_{(c_i, n_j) \in E} \mathbb{I}(w_{ij} < \theta) \cdot w_{ij}^2$

3. **GCN-based Pseudo-Label Generation (GCPL)**: Existing methods generate pseudo-labels via direct similarity matching, ignoring higher-order inter-class dependencies. This paper aggregates neighborhood information via graph convolutional networks:

$$\mathbf{n}_j^{(t+1)} = \sigma\left(\sum_{i=1}^M \frac{w_{ij}}{\sqrt{d_i d_j}} \cdot \mathbf{c}_i^{(t)} + \sum_{k=1}^K \frac{w_{jk}}{\sqrt{d_j d_k}} \cdot \mathbf{n}_k^{(t)}\right)$$

After multiple GCN layers, pseudo-labels are generated by cosine similarity matching: $\hat{y}_j = \arg\min_i \text{sim}(\mathbf{n}_j^{\text{final}}, \mathbf{c}_i)$.

### Loss & Training

- Backbone: MinkowskiUNet-34C
- Optimizer: AdamW, initial lr = 1e-3, decayed every 5 epochs to a minimum of 1e-5
- $\lambda_{adv}$ and pruning threshold $\theta$ are both initialized at 0.5 and dynamically adjusted during training
- Temperature parameter $\tau = 0.06$, regularization coefficient $\lambda = 0.02$
- Number of GCN layers: 3
- Novel classes undergo only causal representation prototype learning without causal deconfounding (due to the absence of reliable labels)

## Key Experimental Results

### Main Results

**SemanticPOSS dataset (average over 4 splits):**

| Method | Split 0 Novel | Split 0 All | Split 1 Novel | Split 1 All | Split 2 Novel | Split 3 Novel |
|--------|--------------|------------|--------------|------------|--------------|--------------|
| EUMS | 17.4 | 20.3 | 21.0 | 35.6 | 8.3 | 13.0 |
| NOPS | 35.7 | 29.4 | 30.0 | 36.4 | 9.0 | 10.9 |
| SNOPS | 51.2 | 33.6 | 32.1 | 37.2 | 16.9 | 20.1 |
| DASL | 48.4 | 41.2 | 36.2 | 44.0 | 12.6 | 17.7 |
| **Ours** | **51.3** | **41.7** | **37.3** | **45.8** | **13.6** | **27.9** |

**SemanticKITTI dataset (4 splits):**

| Method | Split 0 Novel↑ | Split 0 All↑ | Split 1 Novel↑ | Split 2 Novel↑ | Split 3 Novel↑ |
|--------|--------------|------------|--------------|--------------|--------------|
| NOPS | 37.1 | 29.3 | 25.4 | 16.5 | 12.4 |
| SNOPS | 45.9 | 31.2 | 27.2 | 17.6 | 14.9 |
| DASL | 45.7 | 36.8 | 28.7 | 20.1 | 12.6 |
| **Ours** | **46.9** | **36.9** | **33.6** | **18.5** | **15.1** |

### Ablation Study

**Component analysis (SemanticPOSS Split 0 and average over all splits):**

| Baseline | CRP | CRG | GCPL | Split0 Novel↑ | All-Split Avg↑ | Note |
|----------|-----|-----|------|--------------|----------------|------|
| ✓ | | | | 38.3 | 24.7 | Baseline |
| ✓ | ✓ | | | 41.6 | 25.9 | +CRP |
| ✓ | | ✓ | | 45.5 | 28.8 | +CRG (w/ conventional clustering prototypes) |
| ✓ | ✓ | ✓ | | 47.4 | 30.1 | CRP+CRG |
| ✓ | ✓ | ✓ | ✓ | **51.2** | **32.5** | Full method |

**2D NCD extension (PASCAL-5i / COCO-20i):**

| Method | PASCAL-5i Avg | COCO-20i Avg |
|--------|--------------|-------------|
| EUMS | 59.1 | 26.81 |
| **Ours** | **61.2** | **27.00** |

### Key Findings

- CRG yields the largest performance gain (Novel: 38.3 → 45.5), indicating that modeling the causal relationship between base and novel classes is more important than deconfounding alone.
- Combining CRP and CRG yields complementary gains (47.4), and further adding GCPL improves performance to 51.2, demonstrating that all three components are mutually complementary.
- On the most challenging Split 3, the proposed method achieves 36.2% on the cone-stone class (DASL and NOPS both score 0), highlighting the particular advantage of causal reasoning for difficult novel classes.
- The method generalizes to 2D NCD (validated on PASCAL and COCO), confirming the universality of causal learning.
- Grad-CAM visualizations show that the proposed method focuses on more precise causal regions, whereas EUMS produces diffuse feature maps.

## Highlights & Insights

- **The first work to introduce causal learning into 3D NCD**, providing a novel theoretical perspective for the field.
- Reformulating NCD via SCM is conceptually clean: confounder $U$ corresponds to shortcut features, and $B \to N$ corresponds to inter-class knowledge transfer.
- The design combining causal pruning with a direction consistency constraint is elegant, ensuring that information flow in the graph adheres to the causal direction.
- The adversarial deconfounding approach does not require explicitly defining the form of the confounders, making it more practical than backdoor adjustment.
- The method performs particularly well on difficult categories, suggesting that causal reasoning offers unique advantages for handling highly ambiguous classes.

## Limitations & Future Work

- Novel classes are not subjected to causal deconfounding (due to the lack of labels), which may cause novel class prototypes to retain noise.
- The specific form of the confounder $U$ is not explicitly defined; adversarial training only indirectly approximates $Z \perp U$.
- With only 3 GCN layers, the model may be insufficient for capturing complex multi-hop causal relationships.
- Experiments are conducted only on two relatively small datasets; performance on large-scale point clouds (e.g., the full nuScenes dataset) remains unvalidated.
- The implementation details of the indicator function $\mathbb{I}(c_i \to n_j)$ in the direction consistency constraint are insufficiently specified.

## Related Work & Insights

- **NOPS/SNOPS/DASL** are the direct predecessors of 3D NCD; this paper extends that line of work by incorporating a causal framework.
- CausalPC applies SCM to 3D point cloud robustness against adversarial perturbations; this paper adapts SCM to the NCD setting, drawing some methodological inspiration from that work.
- Causal representation learning (Schölkopf et al.) provides the theoretical foundation, specifically the ICM principle and do-calculus.
- The method is extensible to 2D NCD (validated) and may also apply to other segmentation tasks requiring cross-class knowledge transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to introduce causal learning into 3D NCD, with clear theoretical motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two 3D datasets + two 2D datasets, with ablation studies and visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The causal analysis section is clearly written, though notation could be more consistent in places.
- **Value**: ⭐⭐⭐⭐ Establishes a new causal learning paradigm for NCD with implications for 3D open-world perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation](../../CVPR2026/3d_vision/geometric-aware_hypergraph_reasoning_for_novel_class_discovery_in_point_cloud_se.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[NeurIPS 2025\] Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction](object-centric_representation_learning_for_enhanced_3d_semantic_scene_graph_pred.md)

</div>

<!-- RELATED:END -->
