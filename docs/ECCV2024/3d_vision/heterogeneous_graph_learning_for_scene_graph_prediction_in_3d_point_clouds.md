---
title: >-
  [Paper Note] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds
description: >-
  [ECCV 2024][3D Vision][3D Scene Graph] The 3D-HetSGP framework is proposed to model 3D scene graph prediction as a heterogeneous graph learning problem. By utilizing a two-stage process of Heterogeneous Graph Structure Learning (HGSL) and Heterogeneous Graph Reasoning (HGR), it addresses the suboptimal performance issue caused by indiscriminate message passing in existing homogeneous fully-connected graph methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Scene Graph"
  - "Heterogeneous Graph Learning"
  - "Point Cloud Understanding"
  - "Relation Prediction"
  - "Graph Structure Learning"
date: 2026-05-08
content_hash: bf2c4eef5f6a7c31
---

# Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds

**Conference**: ECCV 2024  
**Code**: None  
**Area**: 3D Vision / Scene Graph Prediction  
**Keywords**: 3D Scene Graph, Heterogeneous Graph Learning, Point Cloud Understanding, Relation Prediction, Graph Structure Learning

## TL;DR

The 3D-HetSGP framework is proposed to model 3D scene graph prediction as a heterogeneous graph learning problem. By utilizing a two-stage process of Heterogeneous Graph Structure Learning (HGSL) and Heterogeneous Graph Reasoning (HGR), it addresses the suboptimal performance issue caused by indiscriminate message passing in existing homogeneous fully-connected graph methods.

## Background & Motivation

**Background**: 3D Scene Graph Prediction (3D SGP) aims to identify objects from 3D scenes (typically point cloud inputs) and predict the semantic and spatial relationships between objects, ultimately generating a structured scene graph representation (nodes = objects, edges = relations). Existing methods such as 3DSSG and SGGPoint typically construct the scene as a fully-connected homogeneous graph, performing message passing across all object nodes to learn relation representations.

**Limitations of Prior Work**: Message passing in fully-connected homogeneous graphs suffers from two critical issues: (1) **Indiscriminate information propagation**—message passing occurs regardless of whether a meaningful relationship actually exists between two objects, leading to the introduction of a large amount of noisy information. For instance, two unrelated objects at opposite ends of a room will exchange features, diluting the signal between truly related object pairs. (2) **Homogenization of edge types**—all edges are treated as a single type, whereas in reality, relationships like "cup on table" (support relation) and "table next to chair" (spatial proximity relation) are semantically distinct and should be processed using different message passing schemes.

**Key Challenge**: Edge relationships in scene graphs naturally possess distinct types and directional properties, which are ignored by prior homogeneous graph modeling. While fully-connected graphs ensure that potential relations are not missed, they introduce significant edge noise and result in a computational complexity that scales quadratically with the number of objects.

**Goal**: (1) Automatically identify different edge types in the graph (relation vs. no relation, and specific relation types); (2) perform differentiated message passing based on edge types; (3) establish effective synergy between graph structure learning and graph reasoning.

**Key Insight**: The authors argue that the graph structure (i.e., which edges exist and their respective types) should be learned first, followed by message passing and relation prediction on the learned heterogeneous graph. This corresponds to two stages: graph structure learning (pruning noisy edges and classifying edge types) and graph reasoning (efficient message passing over the pruned heterogeneous graph).

**Core Idea**: Deconstruct 3D scene graph prediction into a two-stage process of "Heterogeneous Graph Structure Learning + Heterogeneous Graph Reasoning", enhancing scene graph quality via directed edge type prediction and typed message passing.

## Method

### Overall Architecture

Given an input 3D point cloud scene, a 3D object detector (e.g., PointNet++/VoteNet) is first employed to detect all objects and extract their 3D features. An initial fully-connected homogeneous graph is then constructed where each detected object serves as a node. The framework consists of two core stages: (1) The HGSL stage predicts the type of each directed edge (including a "no relation" type) to transform the homogeneous fully-connected graph into a sparse heterogeneous graph; (2) the HGR stage performs typed message passing on the learned heterogeneous graph to ultimately predict the specific relation label of each edge and the object category of each node.

### Key Designs

1. **Heterogeneous Graph Structure Learning (HGSL)**:

    - **Function**: Predict the type of each directed edge and construct a sparse heterogeneous graph structure.
    - **Mechanism**: For a directed edge from node $i$ to node $j$, its edge feature $e_{ij}$ is extracted (composed of the concatenation of the two node features, their difference, and spatial relative position encoding). An edge type classifier $f_{edge}$ is utilized to predict the edge type: $t_{ij} = f_{edge}(e_{ij})$. The edge types are defined within a finite set of categories, such as $\{$support, spatial proximity, functional, no relation$\}$. Edges predicted as "no relation" are pruned, and the remaining edges are organized into a heterogeneous graph according to their types. Note that the edges are directed—relationships like "A is on B" and "B is under A" represent distinct directed edges.
    - **Design Motivation**: Through explicit edge type prediction, two objectives are achieved: (a) pruning irrelevant edges to reduce noise, and (b) grouping the remaining edges by semantic types to lay the foundation for subsequent parameterized message passing. The directed edge design also captures the directionality of relationships.

2. **Heterogeneous Graph Reasoning (HGR)**:

    - **Function**: Perform typed message passing over the heterogeneous graph structure to update node and edge representations for final predictions.
    - **Mechanism**: Different message passing functions are utilized for different edge types. Specifically, each edge type $t$ corresponds to an independent set of attention weights $W_t$: $m_{ij}^{(t)} = \text{Attn}(h_i W_Q^{(t)}, h_j W_K^{(t)}, h_j W_V^{(t)})$. The updated feature of node $i$ aggregates messages received across all edge types: $h_i' = \sum_t \sum_{j \in \mathcal{N}_i^{(t)}} \alpha_{ij}^{(t)} m_{ij}^{(t)}$, where $\mathcal{N}_i^{(t)}$ denotes the set of neighbors connected to node $i$ via edges of type $t$. After multiple rounds of message passing, node features are used for object classification, and edge features are used for relation prediction.
    - **Design Motivation**: Different types of relationships require different semantic interpretations: support relations focus on vertical spatial features, proximity relations focus on horizontal distances, and functional relations focus on the functional properties of objects. Using independent attention parameters to process different types of messages prevents the mixing of heterogeneous information.

3. **Spatial Relative Position Encoding**:

    - **Function**: Encode relative position information in 3D space as part of the edge features.
    - **Mechanism**: For object $i$ and object $j$, the relative position vector in 3D space $\Delta p_{ij} = (x_j - x_i, y_j - y_i, z_j - z_i)$ and the relative scale ratio $\Delta s_{ij} = (w_j/w_i, h_j/h_i, l_j/l_i)$ are computed. These geometric features are encoded using an MLP and concatenated with semantic features to form the complete edge representation. In addition, 3D IoU overlap is included as an indicator of spatial contact.
    - **Design Motivation**: Relationships in 3D scenes are highly dependent on spatial layout—relations like "above", "inside", and "next to" are directly determined by spatial geometry. Explicit spatial encoding makes it easier for the model to capture these spatial relationship patterns.

### Loss & Training

Joint end-to-end training of both stages. The HGSL stage employs a cross-entropy loss to supervise edge type prediction, while the HGR stage utilizes cross-entropy losses to supervise object classification and relation prediction, respectively. Downsampling is applied to the "no relation" category to balance positive and negative samples. The total loss is a weighted sum of the three terms: $L = L_{edge\_type} + \lambda_1 L_{obj} + \lambda_2 L_{rel}$.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | Ours | 3DSSG | SGGPoint | Gain |
|--------|------|------|------|-------|----------|------|
| 3DSSG | Relation Prediction | R@50 | 35.8 | 29.4 | 32.1 | +3.7 vs SGGPoint |
| 3DSSG | Relation Prediction | R@100 | 41.2 | 35.6 | 37.8 | +3.4 vs SGGPoint |
| 3DSSG | Scene Graph Classification | R@50 | 38.5 | 33.2 | 35.7 | +2.8 vs SGGPoint |
| 3DSSG | Scene Graph Generation | R@50 | 27.3 | 22.8 | 24.5 | +2.8 vs SGGPoint |

### Ablation Study

| Configuration | R@50 (Relation Prediction) | Description |
|------|----------------|------|
| Full (3D-HetSGP) | 35.8 | Full model |
| Homogeneous Graph (w/o HGSL) | 31.2 | Degenerates to a homogeneous fully-connected graph, drops by 4.6 |
| Heterogeneous Graph + Homogeneous Reasoning | 33.4 | Graph structure is heterogeneous, but message passing does not differentiate types |
| w/o Spatial Encoding | 33.1 | Removes relative position encoding |
| w/o Edge Pruning | 32.8 | Retains all edges but still groups them by type |

### Key Findings
- The HGSL module contributes the most (4.6 R@50), indicating that removing noisy edges and differentiating edge types are crucial for scene graph quality.
- Performing only heterogeneous graph structure learning with homogeneous message passing (+2.2) is inferior to the full model (+4.6), which demonstrates that typed message passing is also critical.
- Spatial relative position encoding yields the most significant improvement for spatial relationship predictions (such as "above", "below", "next to").
- Edge pruning effectively reduces computation (approximately 60% of edges are pruned) while improving accuracy, proving that noisy edges are indeed detrimental.
- In complex scenes with high object density, the advantages of heterogeneous graphs are more pronounced.

## Highlights & Insights

- **Introducing heterogeneous graph learning into 3D scene graph prediction is a natural and effective choice**: Different kinds of relationships in a scene naturally lend themselves to heterogeneous graph modeling. While this insight is intuitive, it has not been systematically explored previously. The elegance lies in the two-stage design of learning the structure first before performing reasoning, which avoids reasoning on noisy graphs.
- **Dual benefits of edge pruning**: It reduces computational complexity (from $O(N^2)$ to $O(N \cdot k)$) and enhances accuracy by eliminating noisy edges. This "less is more" finding holds broad applicability.
- The design methodology of heterogeneous message passing can be transferred to 2D scene graph generation (such as on the Visual Genome dataset) and knowledge graph reasoning, among other tasks requiring differentiated edge types.

## Limitations & Future Work

- The definition of edge types is restricted to a predefined finite set, which may not cover all relationship types across various scenes. Continuous edge type embeddings could be explored.
- The accuracy of edge type prediction in the HGSL stage directly dictates the quality of subsequent reasoning—errors in edge type prediction will propagate to the HGR stage. Alternating or iterative optimization of structure learning and reasoning could be considered.
- Evaluation was conducted only on the 3DSSG dataset, which primarily contains indoor environments. Performance on larger and more diverse scenes (such as extended versions of ScanNet or outdoor scenarios) remains to be validated.
- The performance of the object detector heavily influences the quality of the scene graph—objects missed during detection cannot be recovered in the scene graph.

## Related Work & Insights

- **vs 3DSSG**: 3DSSG is the first 3D scene graph dataset and method, which uses fully-connected homogeneous graphs and GCNs for reasoning. 3D-HetSGP upgrades both the graph structure and message passing mechanisms.
- **vs SGGPoint**: SGGPoint introduces an edge attention mechanism to weight the importance of different edges, but still relies on homogeneous modeling. 3D-HetSGP achieves a more thorough separation via explicit type differentiation compared to implicit attention weighting.
- **vs VL-SAT**: VL-SAT utilizes vision-language pre-trained models to assist 3D scene graph prediction, approaching the problem from the perspective of external knowledge sources. 3D-HetSGP approaches from the perspective of graph structure, making the two complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The introduction of heterogeneous graph learning is reasonable and effective, with a straightforward two-stage design.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated on only one dataset, lacking validation on more benchmarks.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the methodology is systematically presented.
- Value: ⭐⭐⭐⭐ Provides a heterogeneous graph perspective for 3D scene graph prediction, offering reference value for graph structure learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction](../../NeurIPS2025/3d_vision/object-centric_representation_learning_for_enhanced_3d_semantic_scene_graph_pred.md)
- [\[ECCV 2024\] A Direct Approach to Viewing Graph Solvability](a_direct_approach_to_viewing_graph_solvability.md)
- [\[ECCV 2024\] Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration](equi-gspr_equivariant_se3_graph_network_model_for_sparse_point_cloud_registratio.md)
- [\[ECCV 2024\] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)
- [\[ECCV 2024\] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)

</div>

<!-- RELATED:END -->
