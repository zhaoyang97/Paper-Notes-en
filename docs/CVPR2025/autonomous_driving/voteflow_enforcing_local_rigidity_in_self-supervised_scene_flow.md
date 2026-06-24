---
title: >-
  [Paper Note] VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow
description: >-
  [CVPR 2025][Autonomous Driving][Scene Flow Estimation] VoteFlow incorporates local rigid motion constraints as an inductive bias into self-supervised scene flow estimation models by introducing a lightweight module based on differentiable voting within the network architecture. It outperforms previous state-of-the-art self-supervised methods on the Argoverse 2 and Waymo datasets with extremely low computational overhead.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Scene Flow Estimation"
  - "Local Rigidity"
  - "Voting Mechanism"
  - "Self-Supervised Learning"
  - "LiDAR Point Clouds"
date: 2026-05-08
content_hash: 413c411c1be5150e
---

# VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow

**Conference**: CVPR 2025  
**arXiv**: [2503.22328](https://arxiv.org/abs/2503.22328)  
**Code**: [https://github.com/tudelft-iv/VoteFlow](https://github.com/tudelft-iv/VoteFlow)  
**Area**: Autonomous Driving / 3D Vision  
**Keywords**: Scene Flow Estimation, Local Rigidity, Voting Mechanism, Self-Supervised Learning, LiDAR Point Clouds

## TL;DR

VoteFlow incorporates local rigid motion constraints as an inductive bias into self-supervised scene flow estimation models by introducing a lightweight module based on differentiable voting within the network architecture. It outperforms previous state-of-the-art self-supervised methods on the Argoverse 2 and Waymo datasets with extremely low computational overhead.

## Background & Motivation

**Background**: Scene flow estimation aims to recover per-point motion vectors from two consecutive LiDAR scans. In autonomous driving, this serves as a cornerstone for self-supervised scene understanding, which is applicable to downstream tasks such as moving object association and pseudo-label generation. Current-feed forward methods (e.g., ZeroFlow, SeFlow) offer advantages in inference speed and generalization but still face performance bottlenecks.

**Limitations of Prior Work**: In the real world, neighboring points on the same rigid object should share the same motion. Existing methods enforce rigidity constraints through extra loss functions (e.g., the cluster loss in SeFlow) or post-processing (e.g., clustering alignment in ICP-Flow). However, these approaches lack a structural inductive bias—the model itself does not possess the capacity to "encode local rigidity." ICP-Flow uses pre-clustered points for ICP alignment, which incurs significant errors once over-segmentation or under-segmentation occurs in clustering.

**Key Challenge**: Rigidity constraints are either introduced through non-differentiable post-processing (precluding end-to-end optimization) or via additional regularization terms (which may conflict with the primary loss and lower training efficiency). A prior that "neighboring points share motion" has not been directly encoded at the network architecture level.

**Goal**: To design a lightweight, differentiable, and plug-and-play network module that enables the model to identify and exploit locally shared motions during the forward pass.

**Key Insight**: The authors observe that in autonomous driving with short time intervals (~0.1s), object motion is predominantly translation-dominated. Thus, the problem can be simplified to identifying the dominant translation direction in a discretized translation space through voting from neighboring pillars.

**Core Idea**: Build a discretized voting space covering all possible translations, allow neighboring pillars to vote for each direction based on feature similarity, and aggregate the voting information into continuous features using a CNN to achieve end-to-end learning.

## Method

### Overall Architecture

The inputs to VoteFlow are two consecutive LiDAR point clouds $X^t$ and $X^{t+\Delta t}$ (with ego-motion compensated). The workflow is as follows:
1. **Pillarization**: Point clouds are converted into bird's-eye view (BEV) pseudo-images via a Pillar Feature Net, with each grid (pillar) sized at 0.2m × 0.2m, accompanied by embedded features.
2. **U-Net Backbone**: Concatenates the two pseudo-images and extracts fused features $G$ through a U-Net.
3. **Voting Module (Core)**: Constructs a voting space for each non-empty pillar, performs differentiable voting, and outputs voting features $H$.
4. **Decoder**: Concatenates pseudo-image features, fused features, voting features, and the offset of points relative to their pillar centers, then predicts per-point scene flow through 4 fully connected layers.

### Key Designs

1. **Discretized Voting Space**:

    - **Function**: Constructs a discrete grid for each pillar, covering all possible 2D translation directions.
    - **Mechanism**: Given a maximum translation range of $\pm 2$ meters and a pillar size of 0.2m, the voting space $V_k^t$ for each pillar $k$ is a $20 \times 20$ discrete grid. For pillar $k$, $M=8$ nearest neighbor pillars at time $t$ are selected. Each neighboring pillar searches $N=128$ candidate pillars at time $t+\Delta t$ to compute voting scores $s_{k,m,n} \in [-1,+1]$ based on cosine feature similarity. These scores are accumulated into the corresponding translation bins: $V_k^t(\vec{T}_{k,m,n}) \leftarrow V_k^t(\vec{T}_{k,m,n}) + s_{k,m,n}$
    - **Design Motivation**: Direct argmax yields only a single coarse direction and is unstable in early training stages. Thus, a two-layer CNN + ReLU is used to compress the voting space into a continuous feature vector, providing the decoder with complete voting distribution information.

2. **Pillar-level Operations and Sparsity Exploitation**:

    - **Function**: Executes voting at the pillar level rather than the point level, significantly reducing computation.
    - **Mechanism**: In autonomous driving scenarios, over 90% of pillars are empty (especially after ground point removal). The voting module operates only on non-empty pillars, utilizing a ball query function to search for neighboring pillars at time $t+\Delta t$, thereby avoiding exhaustive computation.
    - **Design Motivation**: Per-point operations incur unacceptable computational costs on large-scale point clouds. Pillarization not only reduces computation but also naturally provides spatial aggregation.

3. **Lightweight Decoder Design**:

    - **Function**: Fuses multi-source features into per-point scene flow.
    - **Mechanism**: Unlike SeFlow which uses GRU layers, VoteFlow uses 4 fully connected layers + ReLU for decoding. It retrieves per-point features from pseudo-images $I^t$, $I^{t+\Delta t}$, fused features $G$, and voting features $H$ using point-to-pillar indices, and appends the offset of each point relative to its pillar center.
    - **Design Motivation**: Simplifies the design to lower the computational cost of training and inference, replacing the sequential modeling capability of GRUs with the structural prior from the Voting Module.

### Loss & Training

Following SeFlow's self-supervised loss framework, the total loss is formulated as $\mathcal{L}_{total} = \mathcal{L}_{chamfer} + \mathcal{L}_{dynamic} + \mathcal{L}_{static} + \mathcal{L}_{cluster}$:

- **Bidirectional Chamfer Loss** $\mathcal{L}_{chamfer}$: Minimizes the distance between the warped source point cloud $\hat{X}^t = X^t + F^t$ and the target point cloud $X^{t+\Delta t}$.
- **Dynamic Loss** $\mathcal{L}_{dynamic}$: Applies Chamfer loss only to dynamic points to address class imbalance (dynamic points being the minority). Dynamic points are pre-defined by the offline method DUFOMap.
- **Static Loss** $\mathcal{L}_{static}$: Encourages the flow of static points to be zero.
- **Cluster Loss** $\mathcal{L}_{cluster}$: Enforces flow consistency constraints on points clustered by HDBSCAN.

Training uses the Adam optimizer with an initial learning rate of $2 \times 10^{-4}$ for 12 epochs. The learning rate is reduced by a factor of 10 after the 6th epoch.

## Key Experimental Results

### Main Results

**Argoverse 2 Test Set (Bucketed Normalized EPE):**

| Method | Label | Dynamic Norm. EPE ↓ (avg) | Car | Pedestrian | Wheeled VRU | Static EPE ↓ (avg) |
|------|------|------------------------|-----|------------|-------------|------------------|
| Flow4D | ✓ | 0.174 | 0.096 | 0.278 | 0.155 | 0.012 |
| SeFlow | ✗ | 0.309 | 0.214 | 0.463 | 0.267 | 0.014 |
| ICP-Flow | ✗ | 0.331 | 0.195 | 0.435 | 0.363 | 0.027 |
| **VoteFlow** | **✗** | **0.289** | **0.202** | **0.417** | **0.249** | **0.014** |

**Waymo Open Validation Set (Cross-Dataset Zero-Shot Transfer):**

| Method | In-domain Training | FD EPE ↓ | FS EPE ↓ | BS EPE ↓ |
|------|---------|----------|----------|----------|
| SeFlow | ✓ | 0.151 | 0.018 | 0.011 |
| SeFlow | ✗ | 0.155 | 0.018 | 0.013 |
| **VoteFlow** | **✗** | **0.142** | **0.014** | **0.012** |

### Ablation Study

| Configuration | Dynamic Norm. EPE ↓ | Description |
|------|------------------|------|
| SeFlow baseline | 0.309 | Baseline |
| + Voting Module (VoteFlow) | 0.289 | Adding voting module, +2.0%pt improvement |
| VoteFlow w/ GRU decoder | - | More computationally expensive than FC decoder |
| VoteFlow (FC decoder) | 0.289 | Lighter and faster |

### Key Findings

- VoteFlow achieves the best dynamic average Normalized EPE (0.289) among all self-supervised methods, outperforming SeFlow by 2.0 percentage points.
- The largest improvement is observed in the pedestrian category (+4.6%pt), indicating that the voting mechanism is more effective for modeling the motion of small, non-vehicle objects.
- ICP-Flow performs best on the Car category (0.195) but falls significantly behind on Wheeled VRU (0.363), suggesting that its clustering strategy is unstable.
- Excellent cross-dataset generalization: Trained on Argoverse 2 and zero-shot transferred to Waymo, VoteFlow achieves an FD EPE of 0.142, surpassing SeFlow trained on Waymo (0.151).
- The inference speed is approximately 25.6ms/sample (on an A100 GPU), meeting real-time requirements.

## Highlights & Insights

- **Architecture-level Inductive Bias Replacing Loss-level Constraints**: Moving the rigidity prior from loss functions into the network structure represents a more elegant and efficient approach. As a plug-and-play component, the voting module can be adapted to various baselines.
- The design of **voting $\rightarrow$ CNN $\rightarrow$ continuous features** cleverly avoids the non-differentiable issue of argmax while retaining full voting distribution information for the decoder to utilize flexibly.
- **Sparsity Exploitation**: The high sparsity of autonomous driving LiDAR point clouds (>90% empty pillars) is naturally suited for the voting mechanism, resulting in extremely low computational overhead.

## Limitations & Future Work

- The voting space only models 2D translation, offering limited capacity for modeling 3D rotational motion (e.g., turning vehicles).
- It relies on DUFOMap to pre-define dynamic/static points and HDBSCAN clustering, introducing external dependencies.
- It does not surpass ICP-Flow on the Car category, likely because vehicle motion aligns better with the rigid body assumptions of ICP.
- Future work could consider extending to a 3D voting space or introducing rotational voting dimensions.

## Related Work & Insights

- **vs ICP-Flow**: ICP-Flow utilizes handcrafted features and pre-clustering for ICP alignment, whereas VoteFlow leverages learned features and architecture-level voting. ICP-Flow performs better on Cars, but lacks the generalization and robustness of VoteFlow.
- **vs SeFlow**: SeFlow enforces rigidity constraints through cluster loss regularization, whereas VoteFlow internalizes them as architectural priors. VoteFlow replaces SeFlow's GRU decoder with a much lighter FC decoder.
- **vs Test-Time Optimization Methods (NSFP, FastNSF)**: These methods offer high accuracy but suffer from long inference times (several minutes), whereas VoteFlow achieves real-time inference (~25ms).

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing the voting mechanism as an architectural inductive bias into scene flow is novel, though the overall framework remains an incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across two major datasets, cross-dataset generalization, and qualitative/quantitative analyses.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the visualization in Figure 1 is highly intuitive.
- Value: ⭐⭐⭐⭐ High practical value for self-supervised scene flow; the code has been open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](../../ECCV2024/autonomous_driving/seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [\[CVPR 2025\] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)
- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[CVPR 2025\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)

</div>

<!-- RELATED:END -->
