---
title: >-
  [Paper Note] VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection
description: >-
  [CVPR 2026][3D Vision][Multi-view 3D object detection] This paper proposes VGGT-Det, the first multi-view indoor 3D object detection framework under a sensor-geometry-free (SG-Free) setting. By mining semantic priors (via attention-guided query generation, AG) and geometric priors (via query-driven feature aggregation, QD) from the internal representations of the VGGT encoder, VGGT-Det surpasses prior state-of-the-art methods by 4.4 and 8.6 mAP@0.25 on ScanNet and ARKitScenes, respectively.
tags:
  - CVPR 2026
  - 3D Vision
  - Multi-view 3D object detection
  - indoor scene understanding
  - sensor-geometry-free
  - VGGT
  - Transformer
date: 2026-05-08
content_hash: e26c6d5bceb45291
---

# VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.00912](https://arxiv.org/abs/2603.00912)
**Authors**: Yang Cao, Feize Wu, Dave Zhenyu Chen, Yingji Zhong, Lanqing Hong, Dan Xu (HKUST, Huawei, Sun Yat-sen University)
**Code**: GitHub (open-sourced as noted in the paper)
**Area**: 3D Vision
**Keywords**: Multi-view 3D object detection, indoor scene understanding, sensor-geometry-free, VGGT, Transformer

## TL;DR

This paper proposes VGGT-Det, the first multi-view indoor 3D object detection framework under a sensor-geometry-free (SG-Free) setting. By mining semantic priors (via attention-guided query generation, AG) and geometric priors (via query-driven feature aggregation, QD) from the internal representations of the VGGT encoder, VGGT-Det surpasses prior state-of-the-art methods by 4.4 and 8.6 mAP@0.25 on ScanNet and ARKitScenes, respectively.

## Background & Motivation

Existing multi-view indoor 3D object detection methods (e.g., ImVoxelNet, NeRF-Det, MVSDet) rely heavily on **sensor-provided geometric inputs**—precisely calibrated multi-view camera poses and depth maps. However, in practical deployments, indoor cameras are often handheld or frequently repositioned, making accurate pose acquisition costly and often infeasible, which severely limits the scalability of these approaches.

Recent feed-forward 3D reconstruction models (DUSt3R, MASt3R, VGGT, etc.) have demonstrated that strong 3D cues can be inferred directly from pose-free 2D images. In particular, VGGT (Visual Geometry Grounded Transformer) can predict camera poses, point clouds, and other 3D properties from multi-view images in a single forward pass. This opens a new opportunity for sensor-geometry-free indoor 3D detection.

## Core Problem

1. **Setting level**: How to perform multi-view indoor 3D object detection without relying on any sensor-provided poses or depth? (the SG-Free setting)
2. **Method level**: How to go beyond merely "consuming" VGGT's predicted outputs and instead deeply exploit the semantic and geometric priors encoded in its internal representations?
3. **Technical level**: VGGT's predicted point cloud is a dense scene reconstruction that does not distinguish foreground from background; naive farthest point sampling (FPS) results in many queries falling in background regions.

## Method

### Overall Architecture

VGGT-Det adopts an encoder–decoder Transformer architecture:

- **Encoder**: A pretrained, frozen VGGT encoder for extracting 3D-aware features
- **Decoder**: A Transformer decoder that iteratively updates object queries via cross-attention
- **Two core modules**: AG (Attention-Guided Query Generation) and QD (Query-Driven Feature Aggregation)

### Basic Backbone

Given multi-view images $\{I_1, I_2, \dots, I_V\}$, the VGGT encoder outputs a token sequence $\mathbf{T}_i \in \mathbb{R}^{M \times C}$ per view, where $M$ is the number of tokens per view and $C$ is the feature dimension. Tokens from all views are concatenated along the token dimension:

$$\mathbf{T}_{\text{concat}} = [\mathbf{T}_1; \mathbf{T}_2; \dots; \mathbf{T}_V] \in \mathbb{R}^{(V \cdot M) \times C}$$

Initial object queries are obtained by applying FPS to the VGGT-predicted point cloud $\mathbf{P}_{\text{pred}}$ to select $K$ seed points, which are then encoded as initial queries $\mathbf{Q}_0$. The decoder consists of $L$ layers, each containing self-attention and cross-attention, and a detection head outputs class labels $\hat{\mathbf{c}} \in \mathbb{R}^K$ and bounding boxes $\hat{\mathbf{b}} \in \mathbb{R}^{K \times 7}$.

### Attention-Guided Query Generation (AG)

**Key observation**: Although the VGGT encoder's attention maps are not trained on any semantic task, they naturally capture rich semantic information—object regions tend to receive higher attention weights.

**Motivation**: Naive FPS on VGGT's dense predicted point cloud distributes query points indiscriminately across foreground and background regions, wasting many queries on background. AG leverages attention priors to guide sampling toward object regions.

**Procedure**:

1. **Attention normalization**: Min-max normalize the VGGT encoder attention weights $\mathbf{A} \in \mathbb{R}^N$ to obtain $\mathbf{A}_{\text{norm}}$
2. **First point selection**: Select the point with the highest attention score as the first query point: $\mathbf{I}[1] = \arg\max \mathbf{A}_{\text{norm}}$
3. **Iterative sampling**: Subsequent points are selected via a combined priority score that balances semantic attention and spatial diversity:
$$\text{Priority} = \mathbf{A}_{\text{norm}} + \lambda_{\text{dist}} \cdot \mathbf{D}_{\text{norm}}$$
   where $\lambda_{\text{dist}} \in [0,1]$ is a trade-off coefficient and $\mathbf{D}_{\text{norm}}$ is the min-max normalized minimum Euclidean distance from each point to the already-selected set
4. **Distance computation**: $\mathbf{D}_{\min} = \min_{j \in \{1,\dots,k-1\}} \|\mathbf{P} - \mathbf{P}_{\mathbf{I}[j]}\|_2$, followed by min-max normalization to obtain $\mathbf{D}_{\text{norm}}$
5. **Greedy selection**: At each iteration, the unselected point with the highest priority is chosen: $\mathbf{I}[k] = \arg\max_{i \notin \mathcal{S}} \text{Priority}$

**Design advantage**: AG concentrates query points in semantically meaningful object regions while maintaining spatial diversity to cover the entire 3D space, achieving a balance between localization accuracy and global coverage.

### Query-Driven Feature Aggregation (QD)

**Key insight**: The VGGT encoder progressively lifts 2D features into 3D representations across layers, with different layers encoding different levels of geometric abstraction. Naively using features from a fixed layer in a sequential manner cannot adaptively match the needs of the detection task.

**Core design — See-Query mechanism**:

A learnable See-Query token $\mathbf{q}_{\text{see}} \in \mathbb{R}^C$ is introduced. It interacts with object queries to "perceive" their requirements and then dynamically aggregates multi-level geometric features.

**Multi-level feature aggregation**:

1. **Weight generation**: The See-Query passes through an MLP followed by Softmax to generate attention weights over features from $L$ layers:
$$\mathbf{w} = \text{Softmax}(\text{MLP}(\mathbf{q}_{\text{see}})), \quad \mathbf{w}_i \geq 0, \sum_{i=1}^L \mathbf{w}_i = 1$$
2. **Weighted aggregation**: The aggregated feature is a weighted sum of multi-level geometric features:
$$\mathbf{F}_{\text{agg}} = \sum_{i=1}^L \mathbf{w}_i \cdot \mathbf{F}_i$$

**Interaction with object queries**:

1. **Concatenation**: The See-Query is concatenated with $K$ object queries to form a unified query set $\mathbf{Q}_{\text{input}} \in \mathbb{R}^{(K+1) \times C}$
2. **Self-attention**: The unified query set exchanges information via self-attention, through which the See-Query perceives the needs of the object queries
3. **Cross-attention**: All queries attend to the aggregated features $\mathbf{F}_{\text{agg}}$ via cross-attention to retrieve hierarchically encoded encoder features
4. **Iterative update**: The updated See-Query is passed to the next decoder layer, repeating the process from weight generation

**Key point**: The See-Query is iteratively refined across decoder layers, enabling dynamic, context-aware guidance for multi-level geometric feature aggregation.

### Training Details

| Configuration | Setting |
|---|---|
| VGGT encoder | Pretrained and frozen; no gradient updates |
| Number of object queries | 256 |
| Optimizer | AdamW, lr=2.5×10⁻⁴, weight decay=1×10⁻⁴ |
| Gradient clipping | max norm=35, norm type=2 |
| LR schedule | Cosine annealing, decaying to 1×10⁻⁶ |
| Loss function | Following 3DETR settings |
| Encoder layers selected | VGGT layers 4/11/17/23 |
| Training hardware | 8×H800 GPUs, approximately two days |

## Key Experimental Results

### Main Results on ScanNet (mAP@0.25)

| Method | Setting | mAP@0.25 | Key Category Performance |
|---|---|---|---|
| ImVoxelNet | SG-Free (VGGT poses) | 35.2 | bed 76.3, toilet 83.2 |
| FCAF3D | SG-Free (VGGT point cloud) | 40.6 | bed 81.0, toilet 83.6 |
| NeRF-Det | SG-Free (VGGT poses) | 41.2 | bed 85.3, toilet 88.4 |
| MVSDet | SG-Free (VGGT poses) | 42.5 | bed 80.6, toilet 89.7 |
| **VGGT-Det (BB)** | SG-Free | 41.4 | bed 85.5, sofa 78.5, toilet 89.5 |
| **VGGT-Det (BB+AG)** | SG-Free | 44.2 (+2.8) | bath 84.0, shower 45.2 |
| **VGGT-Det (BB+AG+QD)** | SG-Free | **46.9 (+4.4)** | chair 61.9, bookshelf 36.9, curtain 31.4 |

### Results on ARKitScenes (mAP@0.25)

| Method | mAP@0.25 |
|---|---|
| ImVoxelNet | 12.4 |
| NeRF-Det | 18.1 |
| MVSDet | 19.4 |
| **VGGT-Det** | **28.0 (+8.6)** |

VGGT-Det achieves an even larger advantage on ARKitScenes, with notable gains on washer (+16.3), refrigerator (+32.2), and sofa (+13.9).

### Ablation Study

| Experiment | Variable | mAP@0.25 | Finding |
|---|---|---|---|
| Effect of AG | BB → BB+AG | 41.4 → 44.2 (+2.8) | Attention priors effectively guide queries toward objects |
| Effect of QD | BB+AG → BB+AG+QD | 44.2 → 46.9 (+2.7) | See-Query adaptive aggregation outperforms fixed strategies |
| Input frame count | 40/60/80 frames | 45.3/46.2/46.9 | 80 frames approaches saturation |
| Feature aggregation | Last layer only / Fixed 4 layers / QD | 37.5/44.2/46.9 | Multi-level features far outperform single-layer; QD adds +2.7 |
| Time & memory | VGGT-Det vs. MVSDet | 0.23s+3.57GB vs. 0.21s+13.81GB | 74% memory reduction with +5.6 mAP |

## Highlights & Insights

1. **Pioneering setting**: The first work to explicitly define and address the SG-Free multi-view indoor 3D object detection problem, eliminating dependence on sensor-provided poses and depth
2. **Interesting observation**: VGGT attention maps, despite being trained without any semantic supervision, encode rich semantic information that can be effectively exploited
3. **Elegant AG design**: The iterative sampling strategy combining semantic attention and spatial diversity balances object-focused querying with global scene coverage
4. **Novel See-Query mechanism in QD**: A learnable token interacts with object queries to achieve adaptive multi-level feature aggregation, outperforming manually selected layer combinations
5. **Memory efficiency**: Additional memory usage of only 3.57 GB, far below MVSDet's 13.81 GB, yielding strong practical utility
6. **Thorough validation**: Training loss dynamics are analyzed to validate the individual contributions of AG and QD

## Limitations & Future Work

1. **Difficulty with small/thin objects**: Categories such as TV and stove perform poorly across all methods, and this problem is exacerbated under the SG-Free setting
2. **Frozen VGGT encoder**: Fine-tuning or partially unfreezing the VGGT encoder has not been explored and may yield further improvements
3. **Dependence on VGGT quality**: The performance ceiling of the framework is bounded by the 3D reconstruction quality of VGGT, which may degrade in low-texture or repetitive-texture scenes
4. **Limited to indoor scenes**: Generalizability to outdoor or large-scale environments remains unverified
5. **Greedy AG sampling**: The iterative greedy sampling may not be globally optimal; more principled sampling strategies warrant further exploration

## Related Work & Insights

- **vs. MVSDet/NeRF-Det**: These methods depend on sensor poses; substituting VGGT-predicted poses limits their performance. VGGT-Det directly leverages VGGT's internal representations, avoiding error propagation from pose estimation
- **vs. FCAF3D**: Point-cloud-based methods lack effective semantic guidance in the SG-Free setting; VGGT-Det surpasses FCAF3D by 6.3 mAP
- **vs. outdoor 3D detection (DETR3D, BEVFormer, etc.)**: Outdoor cameras are rigidly mounted on vehicles with reliable poses; indoor handheld cameras make the SG-Free setting considerably more challenging
- **vs. DUSt3R/MASt3R**: These reconstruction methods handle only pairwise inputs and require multiple forward passes plus global coordinate alignment; VGGT supports multi-view single forward inference
- **Internal representation mining**: This work demonstrates that rich priors in intermediate layers of pretrained models substantially exceed what their final outputs reveal, a principle worth exploring across a broader range of tasks
- **Learnable aggregation vs. fixed layer selection**: The See-Query design philosophy generalizes to any scenario requiring multi-level feature aggregation
- **Generality of the SG-Free setting**: The setting can be extended to downstream tasks such as 3D semantic segmentation, instance segmentation, and scene graph generation
- **Complementarity with 3DGS-Det**: The same research group previously proposed 3DGS-Det based on Gaussian splatting for 3D detection; VGGT-Det takes the sensor-free route, representing a complementary technical direction

## Rating

- Novelty: ⭐⭐⭐⭐ (First to propose the SG-Free setting + dual-module AG/QD design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets + detailed ablations + time/memory analysis + loss dynamics)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables, naturally motivated contributions)
- Value: ⭐⭐⭐⭐ (Strong practical significance of the SG-Free setting; methodology offers meaningful insights to the community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](vggt-slam.md)
- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](../../AAAI2026/3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[CVPR 2026\] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)
- [\[CVPR 2026\] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration](fast3dcache_training-free_3d_geometry_synthesis_acceleration.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)

</div>

<!-- RELATED:END -->
