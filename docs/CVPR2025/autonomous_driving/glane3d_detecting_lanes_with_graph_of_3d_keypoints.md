---
title: >-
  [Paper Note] GLane3D: Detecting Lanes with Graph of 3D Keypoints
description: >-
  [CVPR 2025][Autonomous Driving][3D Lane Detection] This paper proposes GLane3D, a keypoint-based 3D lane detection method. It constructs a graph structure by detecting lane keypoints and predicting directed connections between them. After removing redundant keypoint proposals using PointNMS, Dijkstra's shortest path algorithm is employed to extract lane instances. It achieves state-of-the-art (SOTA) F1 scores on OpenLane and Apollo datasets with superior generalization capabi…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "3D Lane Detection"
  - "Keypoint Detection"
  - "Directed Graph"
  - "PointNMS"
  - "Cross-dataset Generalization"
date: 2026-05-08
content_hash: 9673874670ba47b1
---

# GLane3D: Detecting Lanes with Graph of 3D Keypoints

**Conference**: CVPR 2025  
**arXiv**: [2503.23882](https://arxiv.org/abs/2503.23882)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: 3D Lane Detection, Keypoint Detection, Directed Graph, PointNMS, Cross-dataset Generalization

## TL;DR

This paper proposes GLane3D, a keypoint-based 3D lane detection method. It constructs a graph structure by detecting lane keypoints and predicting directed connections between them. After removing redundant keypoint proposals using PointNMS, Dijkstra's shortest path algorithm is employed to extract lane instances. It achieves state-of-the-art (SOTA) F1 scores on OpenLane and Apollo datasets with superior generalization capability.

## Background & Motivation

**Background**: 3D lane detection is mainly categorized into two groups: top-down methods that directly predict entire lane instances (e.g., LATR, PersFormer) and bottom-up methods that first detect keypoints and then group them into lanes. Feature projection from front-view (FV) to bird's-eye-view (BEV) via IPM or LSS is the standard practice.

**Limitations of Prior Work**: Top-down methods rely on lane patterns learned from training data and generalize poorly to unseen lane structures. Although bottom-up methods exhibit better generalization (as they only need to detect local components), the keypoint grouping phase is difficult—existing approaches use either clustering (BEVLaneDet), direction prediction (GANet), or iterative association (CLRerNet), leading to complex and unstable post-processing. Furthermore, a single erroneous keypoint detection can lead to the fracture of the entire lane.

**Key Challenge**: Keypoint detection is flexible but difficult to group, while predicting entire lanes is easy to group but generalizes poorly. How to simplify grouping while maintaining the generalization advantages of keypoint-based methods?

**Goal**: (1) Simplify keypoint grouping by modeling lane extraction as a path-searching problem on a graph; (2) Improve keypoint detection recall by allowing multiple proposals per target point; (3) Reduce the computational overhead of redundant proposals using PointNMS filtering.

**Key Insight**: Treating lane detection as a directed graph problem, where keypoints represent nodes and connections between adjacent keypoints represent edges. Grouping then becomes a shortest-path search (Dijkstra's algorithm) from source nodes to target nodes, which is more elegant than clustering or direction prediction.

**Core Idea**: Modeling lane detection as a three-step pipeline: "keypoint detection + directed connection prediction + graph-search lane extraction", utilizing multiple proposals combined with PointNMS to guarantee high recall and low redundancy.

## Method

### Overall Architecture

Given the input front-view image $\mathbf{I}$, the backbone network extracts the FV features $\mathbf{F}_{FV}$, which are then projected via IPM to the BEV features $\mathbf{F}_{BEV}$ (using customized non-uniform BEV sampling points). Each grid on the BEV corresponds to an anchor point. First, the model predicts the foreground segmentation map $\mathbf{M}_{seg}$ to select the top-$N$ proposal keypoints $\mathbf{K}_P$. Then, a Transformer module predicts the classification, lateral offset $\Delta x$, height $z$, and connection feature $\mathbf{f}_c$ for each proposal. PointNMS filters out the $S$ strongest keypoints $\mathbf{K}_S$. The connection head predicts an $S \times S$ adjacency matrix $\mathbf{A}$. Finally, lane instances are extracted from the source to target keypoints using Dijkstra's algorithm.

### Key Designs

1. **Multiple Proposal Keypoint Detection + PointNMS**:

    - **Function**: Generate multiple proposals for each target keypoint to improve recall, and then remove redundancy using NMS.
    - **Mechanism**: Allow multiple anchor points within a lateral distance $d_x$ of the target lane to serve as proposal keypoints. Each proposal predicts a lateral offset $\Delta x$ to align with the ground-truth lane position. PointNMS retains the proposal with the highest confidence within distance $d_x$ and eliminates duplicates. While multiple proposals alone would cause connection graph ambiguity due to multiple keypoints at the same location (one predecessor corresponding to multiple successors), PointNMS perfectly addresses this issue.
    - **Design Motivation**: In bottom-up methods, missing a single keypoint can cause the entire lane to fracture. Multiple proposals greatly reduce the missed detection rate, while PointNMS ensures graph clarity and computational efficiency. Ablation studies show that the combination of both brings a $+5.0\%$ F1 score improvement.

2. **Customized Non-uniform BEV Geometry**:

    - **Function**: Improve the sampling density distribution of IPM projection.
    - **Mechanism**: Standard uniformly distributed BEV points are sparse in the near range and dense in the far range when projected to the front view (due to perspective effects). The customized scheme of GLane3D reduces the longitudinal and lateral intervals of BEV points as they approach the ego-vehicle. This ensures that the projection to the FV is denser in the near region and not oversaturated in the far region. The number of points per row remains unchanged, with only the intervals being adjusted.
    - **Design Motivation**: Lane lines are more critical near the ego-vehicle (for immediate trajectory planning), yet uniform BEV samples most sparsely in this near range. Customized geometry resolves this spatial resolution mismatch. Ablations show that it contributes $+0.4\%$ F1 score.

3. **Directed Connection Estimation and Graph-Search Lane Extraction**:

    - **Function**: Assemble independent keypoints into complete lane instances.
    - **Mechanism**: The connection feature $\mathbf{f}_c$ is concatenated with positional encoding and then processed by two MLPs to output the origin feature $\mathbf{F}_{orig}$ and the destination feature $\mathbf{F}_{dest}$. After reshaping, element-wise multiplication yields an $S \times S \times d$ tensor, which passes through a linear layer and a sigmoid function to obtain the adjacency matrix $\mathbf{A}$. The connection head is supervised using Focal Loss. For lane extraction, source points (no incoming edges but outgoing edges) and sink points (no outgoing edges but incoming edges) are identified. Dijkstra's algorithm is then applied to search for the shortest path with edge weights set to $(1-\mathbf{A})$.
    - **Design Motivation**: Directed connections impose stronger structural constraints than undirected ones, explicitly determining the direction of travel. Dijkstra's algorithm ensures global optimality of the extracted path. PointNMS reduces the number of nodes from $N$ to $S$, dramatically decreasing the complexity of graph search.

### Loss & Training

The total loss is defined as $L_{total} = w_{kp}L_{kp} + w_r L_r + w_{cn}L_{cn} + w_c L_c$:
- $L_{kp}$: BCE loss for keypoint proposals
- $L_r$: $L_1$ regression loss for lateral offset and height
- $L_{cn}$: Focal Loss for the connection head
- $L_c$: CE loss for classification
- The weights $w_*$ are learnable (refer to uncertainty-based weighting).

Double Hungarian Matching: Matching is first performed on all proposals $\mathbf{K}_P$ (with GT repeated $n$ times), and then matching is conducted on $\mathbf{K}_S$ (with non-repeated GT) after PointNMS. The model is trained using the Adam optimizer with a learning rate of $3\text{e-}4$ for 24 epochs on OpenLane and 300 epochs on Apollo.

## Key Experimental Results

### Main Results

| Dataset | Method | Backbone | F1(%) @1.5m↑ | X-err near(m)↓ | X-err far(m)↓ |
|--------|------|------|-------------|-----------------|----------------|
| OpenLane | PVALane | Swin-B | 63.4 | 0.226 | 0.257 |
| OpenLane | LATR | R50 | 61.9 | 0.219 | 0.259 |
| OpenLane | **GLane3D-Base** | R50 | **63.9** | **0.193** | **0.234** |
| OpenLane | **GLane3D-Large** | Swin-B | **66.0** | **0.170** | **0.203** |
| OpenLane @0.5m | LATR | R50 | 54.0 | 0.171 | 0.201 |
| OpenLane @0.5m | **GLane3D-Base** | R50 | **57.9** | **0.157** | **0.179** |

GLane3D-Large outperforms all previous methods with an F1 score of $66.0\%$ on OpenLane, exhibiting a larger margin under the strict threshold of $0.5\text{m}$.

### Ablation Study

| Configuration | F1(%) | Gain |
|------|-------|------|
| Baseline | 66.6 | - |
| + PointNMS | 69.2 | +2.6 |
| + Multiple Proposal (without PointNMS) | 42.7 | -23.9 |
| + Multiple Proposal + PointNMS | 71.6 | +5.0 |
| + Multiple Proposal + PointNMS + Custom BEV | **72.0** | **+5.4** |

| Number of Keypoints $S$ | F1(%)↑ | FPS↑ |
|-----------|--------|------|
| 128 | 55.5 | 28.5 |
| 256 | 72.0 | 27.8 |
| 512 | 72.4 | 25.9 |
| 1024 | 72.4 | 21.0 |

### Key Findings

- When used alone, Multiple Proposal leads to a sharp F1 drop of $24\%$ due to connectivity-graph ambiguity caused by multiple keypoint proposals at the same position. However, combined with PointNMS, it conversely gains $+5\%$ over the baseline, suggesting that these two components must be used in tandem.
- A keypoint count of 256 is the optimal trade-off; further increasing it to 512/1024 yields almost no quality gain but decreases FPS.
- The method outperforms the previous SOTA across all OpenLane categories, notably improving Curve by $+3.8\%$ and Merge-Split by $+4.8\%$, demonstrating the flexibility of keypoint-based methods in handling complex lane structures.
- GLane3D-Lite (with ResNet-18) achieves $62.2$ FPS, far exceeding other methods, which makes it suitable for real-time onboard deployment.
- Cross-dataset evaluation: The model trained on OpenLane generalizes well when directly evaluated on Apollo, demonstrating the strong generalization ability of the bottom-up paradigm.

## Highlights & Insights

- **Lane Detection as a Graph-Search Problem**: Formulating the grouping problem as a shortest-path search on a directed graph introduces a more elegant and stable alternative to clustering or direction prediction. Dijkstra's algorithm guarantees the global optimality of the path, eliminating the need for complex, heuristic post-processing rules.
- **Complementary Design of Multiple Proposals and PointNMS**: The ablation study clearly reveals that both components are indispensable—multiple proposals improve recall at the cost of ambiguity, whilst PointNMS resolves ambiguity but offers limited standalone improvement. This "add and then prune" strategy is highly instructional.
- **Real-time Feasibility**: Equipped with a ResNet-18 backbone, GLane3D-Lite achieves $62$ FPS and a $61.5\%$ F1 score. It outperforms most existing methods in both speed and accuracy, holding practical significance for onboard deployment.

## Limitations & Future Work

- IPM relies on the flat-ground assumption; although results on the Up&Down category are promising, it is theoretically limited when encountering highly complex outdoor terrains.
- The connections are restricted only to adjacent keypoints, which might be insufficient for exceptionally long lanes or complex topologies.
- Temporal information is currently unused (single-frame detection); introducing video-level consistency could further enhance performance.
- The distance threshold $d_x$ for PointNMS requires manual tuning; an adaptive selection strategy could be more robust.
- This work only validates the camera-only setting and simple camera-lidar fusion; exploring more sophisticated fusion schemes could yield further improvements.

## Related Work & Insights

- **vs LATR**: LATR uses 3D lane queries and Transformers to directly predict the entire lane, representing a top-down paradigm. GLane3D's keypoint-based manner exhibits a distinct advantage in complex categories like Curve and Intersection, as it is free from the constraints of fixed lane query patterns.
- **vs BEVLaneDet**: BEVLaneDet is also a keypoint-based method but uses clustering for grouping. GLane3D's graph-search-based grouping is more stable and does not require learning embedding features.
- **vs PVALane**: PVALane employs DETR-style queries. Under the same Swin-B backbone, GLane3D achieves a $2.6\%$ higher F1 score, indicating that the keypoint-plus-graph paradigm is better suited for lane detection.

## Rating

- Novelty: ⭐⭐⭐⭐ The formulation of modeling lanes with keypoints plus a directed graph is novel in 3D lane detection. The combination of Multiple Proposals and PointNMS is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Broad evaluations covering OpenLane and Apollo datasets, detailed category-wise analysis, inference speed (FPS) benchmarks, cross-dataset generalization, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is described clearly, accompanied by intuitive illustrations.
- Value: ⭐⭐⭐⭐ This work holds practical implementation value for autonomous driving lane detection, especially considering the real-time speed of the Lite version.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rethinking Lanes and Points in Complex Scenarios for Monocular 3D Lane Detection](rethinking_lanes_and_points_in_complex_scenarios_for_monocular_3d_lane_detection.md)
- [\[CVPR 2025\] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)
- [\[ECCV 2024\] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](../../ECCV2024/autonomous_driving/detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)
- [\[CVPR 2025\] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method](towards_satellite_image_road_graph_extraction_a_global-scale_dataset_and_a_novel.md)
- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)

</div>

<!-- RELATED:END -->
