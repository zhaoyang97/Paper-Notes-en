---
title: >-
  [Paper Note] Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction
description: >-
  [NeurIPS 2025][3D Vision][3D semantic scene graph] Through empirical analysis, this paper identifies object feature discriminability as the critical bottleneck in 3D scene graph predicate prediction (object misclassification accounts for 92%+ of predicate errors). It proposes an independently contrastively pre-trained object encoder (3D-2D-Text tri-modal alignment), a geometry-regularized relation encoder, and a bidirectional edge-gated GNN, achieving new SOTA on 3DSSG with Object R@1 59.53% and Predicate R@50 91.40%.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D semantic scene graph
  - object-centric representation
  - contrastive pre-training
  - GNN
  - relation prediction
date: 2026-05-08
content_hash: 200a4d4bc9829b84
---

# Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2510.04714](https://arxiv.org/abs/2510.04714)
**Code**: [https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes](https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes)
**Area**: 3D Vision / Scene Understanding
**Keywords**: 3D semantic scene graph, object-centric representation, contrastive pre-training, GNN, relation prediction

## TL;DR
Through empirical analysis, this paper identifies object feature discriminability as the critical bottleneck in 3D scene graph predicate prediction (object misclassification accounts for 92%+ of predicate errors). It proposes an independently contrastively pre-trained object encoder (3D-2D-Text tri-modal alignment), a geometry-regularized relation encoder, and a bidirectional edge-gated GNN, achieving new SOTA on 3DSSG with Object R@1 59.53% and Predicate R@50 91.40%.

## Background & Motivation

**Background**: 3D semantic scene graphs (3D-SSG) represent 3D scenes as directed graphs with nodes (objects) and edges (relations), serving as a key representation for robot navigation and AR/VR interaction. Methods such as SGPN, SGFN, and VL-SAT have progressively advanced performance.

**Limitations of Prior Work**: (a) Existing methods over-rely on GNNs for relational reasoning while neglecting insufficient discriminability of object representations — VL-SAT's object embeddings are non-discriminative, leading to low-confidence predictions and frequent misclassifications; (b) Relation feature encoding relies solely on geometric descriptors (centroid differences, bounding box differences, etc.), ignoring the integration of object semantic features; (c) GNN edge processing is symmetric, whereas real-world relations (e.g., "A standing on B") exhibit directional asymmetry.

**Key Challenge**: **Object misclassification → predicate prediction errors**. Analysis of VL-SAT reveals that only 8% of predicate errors occur when both subject and object are correctly classified, while error rates surge under object misclassification. Using ground-truth object labels brings predicate R@50 close to 94%+, confirming that the bottleneck lies in object encoding rather than relational reasoning.

**Goal**: (a) Improve the discriminability of object features to indirectly enhance all downstream metrics; (b) Integrate semantic and geometric information to improve relation encoding; (c) Introduce directionality modeling to capture asymmetric relations.

**Key Insight**: A probabilistic formalization — $P(e_{ij}|z_i, z_j) = \sum P(e_{ij}|o'_i, o'_j) P(o'_i|z_i) P(o'_j|z_j)$ — shows that the sharper the object posterior (i.e., higher discriminability), the more accurate the predicate prediction.

**Core Idea**: Independent contrastive pre-training enables the object encoder to produce highly discriminative embeddings → reduces object classification entropy → automatically improves predicate and triplet prediction through probabilistic propagation.

## Method

### Overall Architecture
The framework proceeds in two stages: (1) **Pre-training stage**: contrastive learning trains the object feature encoder via 3D point cloud ↔ multi-view 2D images ↔ CLIP text description alignment, independently of the scene graph task; (2) **Scene graph prediction stage**: the object encoder is frozen, and the relation feature encoder (object-pair features + geometric descriptors + LSE auxiliary task) and a GNN with GSE and BEG are trained jointly.

### Key Designs

1. **Discriminative Object Feature Encoder (Contrastive Pre-training)**:

    - **Function**: Pre-trains an encoder that produces highly discriminative object embeddings, independent of the downstream task.
    - **Mechanism**: The input is a 3D point cloud of an object instance; after T-Net affine transformation for invariance, features $z^t$ are extracted. The contrastive objectives are: (a) visual contrast $\mathcal{L}^{visual}$ — pulling $z^t$ closer to multi-view 2D CLIP features of same-class objects and pushing apart different-class ones; (b) text contrast $\mathcal{L}^{text}$ — aligning $z^t$ with CLIP text features of "A point cloud of {object}". Supervised contrastive learning is adopted, with same-class objects sharing positive samples.
    - **Design Motivation**: Object encoders in methods such as VL-SAT are jointly trained with scene graph objectives, resulting in insufficiently discriminative object features. Independent pre-training decouples the two objectives and lets the object encoder focus on classification accuracy. Experiments confirm that plugging this pre-trained encoder into existing frameworks (SGFN/VL-SAT) consistently improves all metrics.

2. **Relation Feature Encoder + LSE**:

    - **Function**: Fuses semantic features of object pairs with geometric descriptors to construct relation edge features.
    - **Mechanism**: $z^e_{ij} = f_{\theta_r}(\text{CAT}(g_{obj}(z^t_i), g_{obj}(z^t_j), g_{geo}(g_{ij})))$, where $g_{ij} \in \mathbb{R}^{11}$ includes centroid differences, standard deviation differences, bounding box differences, volume ratio, and longest-edge ratio.
    - **LSE (Local Spatial Enhancement)**: An auxiliary task that reconstructs original geometric descriptors from relation features (L1 loss), enforcing the relation representation to retain geometric information and mitigating the information imbalance between high-dimensional object features and low-dimensional geometric descriptors.
    - **Design Motivation**: Prior methods such as SGFN and VL-SAT use only geometric descriptors as edge features, ignoring object semantics; SGPN uses the entire scene point cloud, introducing excessive background noise.

3. **GNN: GSE + BEG**:

    - **GSE (Global Spatial Enhancement)**: Uses the Euclidean distance matrix $D$ between objects as an attention bias — $\alpha_{ij} = \text{softmax}(\frac{q_i^T k_j}{\sqrt{d_k}} + w_{ij}^{(h)})$, where $w_{ij}^{(h)} = W^{(h)}D$ — so that spatially proximate objects receive stronger attention.
    - **BEG (Bidirectional Edge Gating)**: Separates each node's edges into outgoing edges (as subject) and incoming edges (as object), aggregates them separately, then concatenates and gates the result. When updating edges, the reverse edge $z^e_{ji}$ is modulated by a gate scalar $\beta_{ij} = \text{gate}(z^e_{ij})$, reflecting that information flow for "A standing on B" and "B supporting A" should differ.
    - **Design Motivation**: Standard GNNs treat edges symmetrically, whereas relations in 3D scenes are inherently directional.

### Loss & Training
- Pre-training: $\mathcal{L}_{pretrain} = 0.001 \mathcal{L}_{reg} + \mathcal{L}_{cross}$ (affine regularization + cross-modal contrastive)
- Scene graph: $\mathcal{L}_{sg} = \lambda_{obj} \mathcal{L}_{obj} + \lambda_{rel} \mathcal{L}_{rel} + \lambda_{lse} \mathcal{L}_{lse}$

## Key Experimental Results

### Main Results (3DSSG, 1553 scenes, 160 object classes, 26 predicate classes)

| Method | Object R@1 | Object R@5 | Pred R@1 | Pred R@50 | Triplet R@100 |
|--------|-----------|-----------|---------|----------|--------------|
| SGPN | 49.46 | 73.99 | 86.92 | 85.38 | 88.59 |
| SGFN | 53.36 | 76.88 | 89.00 | 88.59 | 91.14 |
| VL-SAT | 55.93 | 78.06 | 89.81 | 89.35 | 92.20 |
| **Ours** | **59.53** | **81.20** | **91.27** | **91.40** | **93.80** |

### Ablation Study

| Configuration | Obj R@1 | Pred R@50 | Triplet R@100 |
|---------------|---------|----------|--------------|
| Baseline (SGFN-style) | 53.36 | 88.59 | 91.14 |
| + Contrastive pre-trained encoder | 59.53 (+6.17) | — | — |
| + LSE | — | +1–2% | — |
| + GSE + BEG | — | +1–2% | — |
| Full model | **59.53** | **91.40** | **93.80** |

**Plug-in validation**: Inserting the pre-trained encoder into SGFN raises Obj R@1 from 53.36 to ~57%; inserting it into VL-SAT yields a similar gain of 2–3%.

### Key Findings
- **Empirical evidence that object discriminability is the core bottleneck**: Object classification entropy $H(o|z)$ is nearly monotonically correlated with predicate error rate — even when the Top-1 prediction is correct, high-entropy objects still cause more predicate errors.
- **92% of predicate errors are associated with object misclassification**: Only 8% of predicate errors occur when both subject and object are correctly identified.
- **The independently pre-trained object encoder is plug-and-play**: Replacing only the object encoder in existing frameworks consistently improves all metrics without modifying other components.
- **LSE auxiliary task is effective**: Forcing the relation representation to retain geometric information improves predicate R@50 by ~1%.
- **BEG captures directional asymmetry**: Gains are especially pronounced for directional predicates such as "standing on" and "hanging from."

## Highlights & Insights
- **The probabilistic argument linking object discriminability to relation prediction is elegant**: The formulation $P(e_{ij}|z_i,z_j) = \sum P(e_{ij}|o'_i,o'_j) P(o'_i|z_i) P(o'_j|z_j)$ formally characterizes the mechanism by which better object embeddings → sharper posteriors → reduced predicate confusion.
- **Decoupled pre-training design**: Pre-training the object encoder independently of the scene graph objective prevents the two objectives from competing. This principle generalizes — upstream encoders in any pipeline may benefit from independent pre-training followed by freezing.
- **Design philosophy of LSE**: Rather than naively concatenating geometric descriptors (which would be overwhelmed by high-dimensional object features), an auxiliary reconstruction task compels the relation encoder to internalize geometric information.

## Limitations & Future Work
- Requires multi-view RGB data from 3RScan for 2D–3D alignment, imposing a strong data dependency.
- Restricted to a closed vocabulary of 160 object classes — generalization to open-vocabulary 3D scene graphs remains unexplored.
- Contrastive pre-training requires CLIP feature extraction and multi-view image processing, increasing computational overhead.
- The 3DSSG dataset is relatively small (1553 scenes); generalization to larger-scale scenes is unknown.

## Related Work & Insights
- **vs. VL-SAT**: VL-SAT employs vision-language pre-training but jointly optimizes within the scene graph task, yielding less discriminative object features than independent pre-training. The proposed method outperforms VL-SAT by +3.6% on Object R@1.
- **vs. SGFN**: A basic GNN baseline with no contrastive pre-training and no directionality modeling. The proposed method improves Object R@1 by +6.17%.
- The insight that object encoder discriminability may be a latent bottleneck in predicate prediction applies equally to 2D scene graph generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Bottleneck diagnosis + probabilistic formalization + multimodal contrastive pre-training + bidirectional gated GNN; component designs are well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Standard 3DSSG benchmark + multi-component ablation + plug-in validation.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth problem analysis, clear probabilistic formalization, and intuitive comparisons in Figure 1.
- **Value**: ⭐⭐⭐⭐ Effective improvement for 3D scene understanding; the insight that object discriminability drives relation prediction is broadly applicable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[NeurIPS 2025\] MaNGO: Adaptable Graph Network Simulators via Meta-Learning](mango_-_adaptable_graph_network_simulators_via_meta-learning.md)
- [\[AAAI 2026\] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis](../../AAAI2026/3d_vision/graph_smoothing_for_enhanced_local_geometry_learning_in_point_cloud_analysis.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](../../ICCV2025/3d_vision/disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](../../ICCV2025/3d_vision/fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)

<!-- RELATED:END -->
