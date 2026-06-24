---
title: >-
  [Paper Note] T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning
description: >-
  [ECCV 2024][3D Vision][Self-Supervised Learning] T-MAE proposes a temporal masked autoencoder pre-training strategy that takes two temporally adjacent frames as input and learns temporal dependencies by masking the current frame and reconstructing it with the help of historical frame information. Equipped with the proposed SiamWCA (Siamese encoder + Windowed Cross-Attention) architecture, it outperforms SOTA self-supervised methods on the Waymo and ONCE datasets with fewer la…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Self-Supervised Learning"
  - "Point Cloud Representation Learning"
  - "Masked Autoencoders"
  - "Temporal Modeling"
  - "3D Object Detection"
date: 2026-05-08
content_hash: e67129ff2a8f6a39
---

# T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning

**Conference**: ECCV 2024  
**arXiv**: [2312.10217](https://arxiv.org/abs/2312.10217)  
**Code**: [https://github.com/weijie-wei/T-MAE](https://github.com/weijie-wei/T-MAE)  
**Area**: Autonomous Driving  
**Keywords**: Self-Supervised Learning, Point Cloud Representation Learning, Masked Autoencoders, Temporal Modeling, 3D Object Detection

## TL;DR
T-MAE proposes a temporal masked autoencoder pre-training strategy that takes two temporally adjacent frames as input and learns temporal dependencies by masking the current frame and reconstructing it with the help of historical frame information. Equipped with the proposed SiamWCA (Siamese encoder + Windowed Cross-Attention) architecture, it outperforms SOTA self-supervised methods on the Waymo and ONCE datasets with fewer labeled data and fewer training iterations.

## Background & Motivation
The scarcity of labeled data in LiDAR point cloud understanding severely restricts representation learning. For example, nuScenes labels only 10% of the frames, and ONCE labels only 0.8%. Self-supervised pre-training (SSL) is an effective approach to tackle this issue.

**Limitations of Prior Work**:
- **Contrastive learning methods** (e.g., PointContrast, SegContrast): Require meticulously tuned hyperparameters and complex pre-/post-processing to establish correspondences.
- **Masked reconstruction methods** (e.g., GD-MAE, MV-JAR): Although effective, they operate in single-frame scenarios, ignoring the natural temporal continuity of LiDAR data.
- **Methods utilizing temporal information** (e.g., STSSL, TARL): Although they introduce multi-frame inputs, their core remains contrastive learning, treating scans from different timestamps as augmented versions of the same scene without explicitly modeling temporal correspondences.

**Key Insight**: Temporally adjacent frames contain not only redundant information (useful for representation enhancement) but, more importantly, motion information. The movement of the ego-vehicle changes the perspective of the same object, acting as a natural and powerful data augmentation.

**Goal**: Extends the masked autoencoding paradigm from single-frame to dual-frame, taking a complete historical frame and a highly masked current frame as input. The pre-training task is to reconstruct the masked voxels of the current frame, thereby forcing the network to learn temporal correspondence and motion modeling capabilities.

## Method

### Overall Architecture
The complete pipeline of T-MAE:
1. **Temporal Batch Sampling**: A pair of frames $(\mathcal{P}^{t_1}, \mathcal{P}^{t_2})$ is sampled from the point cloud sequence with a proper time interval (too close leads to information redundancy, while too far leads to insufficient overlap).
2. **Alignment and Voxelization**: The historical frame is aligned to the coordinate system of the current frame using the ego-pose, and both frames are voxelized into pillar features.
3. **Masking** (pre-training only): Pillars in the current frame are randomly masked out with a high masking ratio of 75%.
4. **Siamese Encoder**: A weight-sharing encoder (SPT) extracts sparse tokens from both frames.
5. **Windowed Cross-Attention (WCA)**: Tokens of the current frame retrieve information from the tokens of the historical frame.
6. **Dense Feature Recovery**: Hierarchical sparse tokens are mapped back to spatial positions to form a dense feature map, which is then refined by four-layer dense convolutions to fill empty positions.
7. **Task Heads**: During pre-training, a reconstruction head recovers the coordinates of the masked points via Chamfer Distance loss; during fine-tuning, a detection head (CenterPoint-style) is employed.

### Key Designs
1. **SiamWCA Architecture (Siamese Encoder + Windowed Cross-Attention)**:

    - **Function**: Establishes the foundational architecture for dual-frame point cloud processing, enabling the current frame to utilize historical frame information.
    - **Siamese Encoder**: Both branches share the same configuration and weights (Siamese) to encode pillar features of the two frames into sparse tokens. Three schemes are compared:
        - Asymmetric encoder (historical frame encoder with halved channels) $\rightarrow$ suboptimal.
        - SimSiam-style (historical frame encoder receives no gradients) $\rightarrow$ worst.
        - **Siamese encoder (gradient accumulation) $\rightarrow$ optimal.**
    - **Design Motivation**: Shared weights ensure that the features of both frames lie in the same latent space, which benefits cross-attention; the gradient accumulation method allows synchronous optimization of both encoders.

2. **Windowed Cross-Attention (WCA)**:

    - **Function**: Performs cross-attention within local windows, allowing tokens of the current frame to query information from the historical frame.
    - **Mechanism**:
        - **Joint Token Grouping**: Divides the 3D space into non-overlapping windows. Tokens of both frames (with aligned coordinates) are assigned to corresponding windows based on their spatial positions.
        - **Sparse Region Cross-Attention (SRCA)**: $\hat{\mathcal{F}}^{t_2} = \text{MCA}(\mathcal{F}^{t_2} + \text{PE}(\mathcal{I}^{t_2}), \mathcal{F}^{t_1} + \text{PE}(\mathcal{I}^{t_1}), \mathcal{F}^{t_1})$
        - The query comes from the current frame $\mathcal{P}^{t_2}$, and keys/values come from the historical frame $\mathcal{P}^{t_1}$.
        - If a window in the historical frame is empty, the current frame tokens remain unchanged.
        - **Window Shifting + Repetitive Operation**: Shift by half the window size for regrouping and perform a second SRCA to expand the receptive field.
    - **Design Motivation**: Global cross-attention is computationally prohibitive on high-resolution 3D point cloud feature maps ($468 \times 468$ vs. $14 \times 14$ in ViT). The windowed implementation keeps the computational complexity within an acceptable range.

3. **T-MAE Pre-training Strategy**:

    - **Function**: Learns temporal correspondences and motion modeling by reconstructing the masked current frame.
    - **Mechanism**:
        - The historical frame is fed completely into the encoder (providing reference information).
        - The current frame is masked with a high ratio of 75% before being fed into the encoder (creating information deficiency).
        - The WCA module leverages historical tokens to enhance the remaining tokens of the current frame.
        - After dense recovery, the reconstruction head retrieves features of masked pillars from the feature map and reconstructs relative coordinates of a fixed number $K^O$ of points within each pillar.
        - Loss: Chamfer Distance between reconstructed points and ground-truth points.
    - **Key Distinction from Single-Frame MAE**:
        - Single-frame methods (e.g., GD-MAE) only reuse encoder weights.
        - T-MAE retains the weights of the entire SiamWCA (encoder + WCA), thus reserving temporal alignment capabilities for downstream tasks.
    - **Design Motivation**: Masked reconstruction forces the network to simultaneously learn two capabilities: (a) robust representation of sparse point clouds; (b) temporal modeling that reasons about the current frame from historical observations.

### Loss & Training
- **Pre-training Loss**: Chamfer Distance, reconstructing $K^O$ points for each masked pillar.
- **Downstream Detection Loss**: Leverages a CenterPoint-style center-based head with the same target assignment strategy.
- **Data Augmentation**: Random flipping, scaling, and rotation are applied to both frames; copy-and-paste augmentation is added during fine-tuning to handle class imbalance.
- **Temporal Batch Sampling**: Samples continuous $n$ frames from a sequence to form a batch; the two frames are sampled from the first and last third respectively to ensure a proper temporal interval.
- Implemented based on the OpenPCDet framework.

## Key Experimental Results

### Main Results

**Waymo Dataset (val, Level 2, different annotation ratios)**:

| Ratio | Method | Initialization | Overall mAPH | Vehicle APH | Ped APH | Cyclist APH |
|---------|------|--------|-------------|-------------|---------|-------------|
| 5% | Random Init| Scratch | 40.29 | 53.50 | 44.76 | 22.61 |
| 5% | MV-JAR | SSL | 46.68 | 56.01 | 47.69 | 36.33 |
| 5% | GD-MAE | SSL | 44.56 | 55.76 | 46.22 | 31.69 |
| 5% | **T-MAE** | **SSL** | **49.46 (+9.17)** | **56.63** | **55.28** | **36.48** |
| 10% | MV-JAR | SSL | 54.06 | 58.00 | 54.66 | 49.52 |
| 10% | **T-MAE** | **SSL** | **57.99 (+4.86)** | **59.77** | **61.10** | **53.09** |
| 100% | GD-MAE | SSL | 67.64 | 68.29 | 65.47 | 69.16 |
| 100% | MV-JAR | SSL | 66.20 | 65.12 | 65.28 | 68.20 |
| 100% | Random Init| Scratch | 69.13 | 68.62 | 68.80 | 69.97 |
| 100% | **T-MAE** | **SSL** | **70.52 (+1.39)** | **68.89** | **72.01** | **70.65** |

**ONCE Dataset (val)**:

| Method | Pre-training | mAP | Vehicle | Pedestrian | Cyclist |
|------|--------|-----|---------|------------|---------|
| SiamWCA | ✗ | 63.71 | 76.47 | 47.27 | 67.40 |
| GD-MAE | ✓ | 64.92 | 76.79 | 48.84 | 69.14 |
| **T-MAE** | **✓** | **67.00 (+3.29)** | **78.35** | **52.57** | **70.09** |

### Ablation Study

**Comparison of architectural designs (Waymo, 5% data)**:

| Model | Encoder | Fusion Method | Overall mAPH | Description |
|------|--------|---------|-------------|------|
| (a) | Asymmetric | WCA | 44.78 | Encoders do not share weights |
| (b) | SimSiam | WCA | 42.05 | One encoder receives no gradient |
| (c) | Siamese | WCA+WSA | 45.11 | With self-attention layer |
| (d) | Siamese | WSA | 40.90 | Self-attention only (frame concatenation) |
| **(e) Ours** | **Siamese** | **WCA** | **46.78** | **Optimal** |

**Dual-Frame vs. Single-Frame vs. Frame Concatenation (Waymo, 5% data)**:

| Method | Input Method | Overall mAPH | Cyclist APH | Description |
|------|---------|-------------|-------------|------|
| GD-MAE | Single frame | 44.56 | 31.69 | Baseline |
| GD-MAE | Concatenated frames | 43.69 | 26.12 | Cyclist performance drops |
| **T-MAE** | **Dual-frame WCA** | **46.78** | **32.37** | **Consistent improvement** |

### Key Findings
- **T-MAE pre-training gains boost as label ratios decrease**: Improves performance by $+1.39$ mAPH with $100\%$ data, rising to $+9.17$ mAPH with $5\%$ data.
- **Pedestrian detection shows the most significant improvement**: T-MAE with $5\%$ labeled data achieves a pedestrian mAPH of $55.28$, surpassing MV-JAR trained with $10\%$ data, indicating that temporal modeling is highly effective for orientation perception (APH).
- **SiamWCA is a powerful backbone by itself**: Training from scratch with $100\%$ data ($69.13$ mAPH) already outperforms prior SSL methods, though it heavily relies on annotations when data is scarce.
- **Frame concatenation is inferior to learned fusion**: Simply concatenating two frames, although increasing point density, introduces "ghost points" for moving objects, leading to degraded cyclist detection.
- **Fast convergence**: T-MAE outperforms MV-JAR with $1.6\times$ to $2.4\times$ fewer fine-tuning iterations.
- **Strong compatibility**: T-MAE consistently brings significant improvements across different backbones (SST, SpCNN, SPT) and detection heads (CenterPoint, Graph R-CNN).

## Highlights & Insights
1. **Self-Supervised Exploitation of Temporal Information** - Extends the masked autoencoding paradigm from single-frame to dual-frame for LiDAR point clouds for the first time, enabling the network to naturally learn temporal correspondence through reconstruction.
2. **Ego-motion as Natural Data Augmentation** - Ego-vehicle movement alters the viewing perspective of the same object, acting as a robust and free data augmentation that requires no manual design.
3. **Efficiency of WCA Design** - Windowed cross-attention restricts computational complexity within an acceptable range, while expanding the receptive field via window shifting.
4. **Significant Improvement in Pedestrian Orientation Detection** - Temporal modeling enables the network to better understand pedestrian orientations, which has practical value for downstream pedestrian intention prediction.
5. **Discovery on SiamWCA** - Proves that the dual-frame architecture is strong on its own, and the role of SSL lies in reducing its dependency on annotated data.

## Limitations & Future Work
- Currently limited to two frames; extending to multi-frame inputs could potentially bring further improvements.
- The WCA module requires dual-frame inputs during both pre-training and inference, introducing extra computational overhead.
- The masking strategy relies on random pillar masking; adaptive masking based on motion regions has yet to be explored.
- Temporal alignment remains challenging for fast-moving objects (e.g., changes in cyclist orientation).
- Combinations with contrastive learning methods remain unexplored.

## Related Work & Insights
- **GD-MAE**: A SOTA baseline for single-frame masked reconstruction, which T-MAE extends to dual-frame.
- **SiameseMAE**: Dual-frame masked autoencoding in video understanding, which inspired T-MAE to transfer similar paradigms to point clouds.
- **SST**: The proposed Sparse Regional Attention (SRA) serves as the foundation of WCA.
- **TARL/STSSL**: Contrastive learning methods utilizing temporal information but relying on complex pre-processing (HDBSCAN to obtain segments), which T-MAE elegantly avoids.
- **Insight**: The temporal dimension is an underestimated source of information in point cloud SSL; even utilizing just two frames can yield substantial improvements.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](../../ICCV2025/3d_vision/dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](../../ICCV2025/3d_vision/strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[ECCV 2024\] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)
- [\[ECCV 2024\] 3D Single-Object Tracking in Point Clouds with High Temporal Variation](3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)
- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](../../AAAI2026/3d_vision/distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)

</div>

<!-- RELATED:END -->
