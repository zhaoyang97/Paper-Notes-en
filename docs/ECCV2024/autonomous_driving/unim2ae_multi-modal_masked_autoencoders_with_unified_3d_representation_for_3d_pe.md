---
title: >-
  [Paper Note] UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][Multi-modal Masked Autoencoders] This paper proposes UniM2AE, a multi-modal self-supervised pre-training framework. By projecting image and LiDAR point cloud features into a unified 3D voxel space (which retains the height dimension unlike BEV) and designing a Multi-modal 3D Interactive Module (MMIM) for efficient cross-modal interaction, it achieves superior performance improvements in 3D detection (+1.2% NDS) and BEV segmentation (+6.5% mIoU)…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Multi-modal Masked Autoencoders"
  - "Self-Supervised Pre-training"
  - "3D Voxel Space"
  - "LiDAR-Camera Fusion"
  - "BEV Perception"
date: 2026-05-08
content_hash: b84e8a45706b226d
---

# UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving

**Conference**: ECCV 2024  
**arXiv**: [2308.10421](https://arxiv.org/abs/2308.10421)  
**Code**: [https://github.com/hollow-503/UniM2AE](https://github.com/hollow-503/UniM2AE)  
**Area**: Autonomous Driving  
**Keywords**: Multi-modal Masked Autoencoders, Self-Supervised Pre-training, 3D Voxel Space, LiDAR-Camera Fusion, BEV Perception

## TL;DR
This paper proposes UniM2AE, a multi-modal self-supervised pre-training framework. By projecting image and LiDAR point cloud features into a unified 3D voxel space (which retains the height dimension unlike BEV) and designing a Multi-modal 3D Interactive Module (MMIM) for efficient cross-modal interaction, it achieves superior performance improvements in 3D detection (+1.2% NDS) and BEV segmentation (+6.5% mIoU) compared to independent pre-training and simple concatenation baselines.

## Background & Motivation

**Background**: Masked Autoencoders (MAE) have demonstrated powerful self-supervised pre-training capabilities in 2D vision and 3D perception tasks. In autonomous driving scenarios, multi-sensor fusion (LiDAR + Camera) is the standard practice to obtain rich environmental perception. Existing methods like GreenMIM (images) and Voxel-MAE (LiDAR) conduct MAE pre-training on single modalities respectively, but lack an effective solution for joint multi-modal pre-training.

**Limitations of Prior Work**: The core challenge of multi-modal MAE lies in the significant discrepancy between the two modalities: images provide dense 2D semantic information, while LiDAR provides sparse 3D geometric information, and they differ drastically in data density, spatial dimension, and information type. Existing attempts (such as PiMAE) project LiDAR onto the image plane for alignment, but this introduces severe geometric distortion—physically distant points may be adjacent in pixel coordinates, and many LiDAR points outside the camera's field of view cannot be projected. Conversely, projection from camera to LiDAR also leads to a loss of dense camera features due to the sparsity of LiDAR.

**Key Challenge**: Multi-modal feature fusion requires a common representation space, but existing representation spaces (image plane or LiDAR coordinate system) have inherent flaws, either losing 3D geometric information or dense semantic information. The BEV representation is a compromise, but it compresses the height dimension, failing to accurately represent objects with different heights (e.g., traffic lights vs. vehicles).

**Goal**: (1) How to design a unified multi-modal representation space that preserves both image semantics and LiDAR geometry? (2) How to efficiently achieve cross-modal interaction in this space? (3) Can the pre-trained features be effectively transferred to multiple downstream tasks?

**Key Insight**: The authors propose extending the BEV space along the height axis (z-axis) into a 3D voxel space. While seemingly straightforward, this extension offers two key advantages: (1) it retains the height information of objects, avoiding the information compression loss of BEV; (2) the 3D voxel space naturally aligns with both modalities—LiDAR points map directly through coordinate transformation, and image features are projected via spatial cross-attention using known camera intrinsic and extrinsic parameters.

**Core Idea**: Projecting multi-modal features onto a 3D voxel space extended along the height dimension for unified representation and interaction, realizing a more information-preserving multi-modal MAE pre-training.

## Method

### Overall Architecture
UniM2AE employs a symmetrical architecture featuring a dual-branch encoder, a single fusion module, and a dual-branch decoder. The LiDAR branch voxelizes the point cloud and uses an SST encoder to extract features $F_V$; the Camera branch patches multi-view images and uses a Swin-T encoder to extract features $F_I$. Both branches randomly mask their inputs (70% for LiDAR, 75% for Camera). The encoded features enter the unified 3D voxel space via Token-Volume projection, interact and fuse through the MMIM module, and are then inversely projected back to their respective modal spaces via Volume-Token projection. Finally, modality-specific decoders reconstruct the original inputs.

### Key Designs

1. **Unified 3D Volume Space**:

    - **Function**: Provides a unified space for cross-modal feature alignment and fusion while preserving complete spatial information.
    - **Mechanism**: Defines the perception range as x/y axes [-50m, 50m], z-axis [-5m, 3m], discretizing the space into voxel grids. For LiDAR, voxel features are directly mapped to the 3D voxel space according to their position in the ego-vehicle coordinate system, obtaining $F_V^{vol}$. For images, 2D-3D Spatial Cross-Attention is used to project 3D voxel query points onto 2D image views via camera intrinsic and extrinsic parameters, sampling corresponding image features with the formula $F_I^{vol} = \frac{1}{|\mathcal{V}_{hit}|} \sum_{i \in \mathcal{V}_{hit}} \sum_j \text{DeformAttn}(Q_{vol}, \mathcal{P}(p,i,j), F_I^i)$. The key lies in the extension of the z-axis—using 2 layers of height resolution by default to strike a balance between accuracy and efficiency.
    - **Design Motivation**: Compared with BEV, the 3D voxel space preserves height information, allowing objects at different heights, such as traffic lights and pedestrian heads, to be accurately represented. More importantly, 3D voxels can be directly back-projected to the original modalities for reconstruction—a fundamental requirement for the MAE framework.

2. **Multi-modal 3D Interaction Module (MMIM)**:

    - **Function**: Efficiently achieves cross-modal feature interaction within the unified 3D voxel space.
    - **Mechanism**: MMIM consists of $L=3$ stacked 3D deformable self-attention blocks. First, the LiDAR voxel features $F_V^{vol}$ and image voxel features $F_I^{vol}$ are concatenated along the channel dimension and reshaped to $F_c^{vol} \in \mathbb{R}^{HWZ \times 2C}$. They are then fed into the 3D deformable self-attention module for interaction: $F_c' = \sum_m W_m \sum_k A_{mk} \cdot W_m' F_c^{vol}(p_{vol} + \Delta p_k^{vol})$, where $\Delta p_k^{vol}$ is the learned sampling offset and $A_{mk}$ is the attention weight. After the interaction, $F_c'$ is split along the channel dimension to obtain the fused, modality-specific 3D features $(F_V', F_I')$.
    - **Design Motivation**: Deformable self-attention is used instead of standard self-attention because the token sequence length $H \times W \times Z$ in 3D voxel space can be very large, making standard self-attention computationally prohibitive. Deformable attention adaptively focuses on the most prominent spatial locations, and its computational complexity scales only linearly with the number of sampling points $K$. Furthermore, the pre-trained weights of MMIM can be directly transferred to downstream fusion tasks.

3. **Dual-modal Reconstruction Targets**:

    - **Function**: Provides multi-granularity supervision signals for MAE pre-training.
    - **Mechanism**: The fused features are mapped back to their respective modality spaces via Volume-Token inverse projection: the LiDAR branch samples voxel features $F_V^{sp}$ directly in the ego coordinates, while the image branch maps 3D voxel features to 2D pixel coordinates via the camera projection function $T_{proj}$, yielding $F_I^{sp}$. The three reconstruction targets are: (a) LiDAR point count reconstruction—predicting the number of points in each voxel, supervised by Chamfer Distance $\mathcal{L}_c$; (b) voxel occupancy prediction—predicting whether a voxel is empty, using BCE loss $\mathcal{L}_{occ}$; (c) image pixel reconstruction—predicting the original pixels of masked patches, using MSE loss $\mathcal{L}_img$.
    - **Design Motivation**: Multiple reconstruction targets facilitate cross-modal feature learning from different perspectives. LiDAR reconstruction forces image features to incorporate geometric information, while image reconstruction forces LiDAR features to capture semantic information, thereby achieving true cross-modal enhancement.

### Loss & Training
The total pre-training loss is $\mathcal{L} = \mathcal{L}_{voxel} + \mathcal{L}_{img} = (\mathcal{L}_c + \mathcal{L}_{occ}) + \mathcal{L}_{MSE}$. Pre-training runs for 200 epochs on 8 GPUs with a base learning rate of 2.5e-5. During downstream fine-tuning, the decoders are removed, and only the encoders and the optional MMIM module are used.

## Key Experimental Results

### Main Results

| Method | Modality | NDS↑ | mAP↑ | Gain (vs Random) |
|--------|------|------|----------|------|
| BEVFusion-SST (Random) | C+L | 67.4 | 63.6 | - |
| MIM + Voxel-MAE | C+L | 67.7 | 63.7 | +0.3/+0.1 |
| PiMAE | C+L | 67.9 | 63.9 | +0.5/+0.3 |
| **UniM2AE** | C+L | **68.1** | **64.3** | **+0.7/+0.7** |
| BEVFusion-SST + MMIM† | C+L | **72.7** | **69.7** | -- |

### Ablation Study

| Configuration | mAP | NDS | Explanation |
|------|---------|------|------|
| Train from scratch (no pre-training) | 59.0 | 61.8 | Baseline |
| Camera pre-training only | 59.7 | 62.6 | Limited image prior |
| LiDAR pre-training only | 60.1 | 62.6 | Limited geometric prior |
| Camera + LiDAR independent pre-training | 60.7 | 63.1 | Simple merger of two pre-training results |
| UniM2AE (BEV interaction) | 62.0 | 64.3 | BEV compresses height information |
| UniM2AE (3D Volume interaction) | **62.8** | **65.2** | 3D voxel preserves height information |

### Key Findings
- **Joint Pre-training vs. Independent Pre-training**: UniM2AE outperforms the independent pre-training and merging scheme by 2.1 NDS, showing that joint learning in a unified space leverages complementary advantages.
- **3D Voxel vs. BEV**: Replacing the BEV space with the 3D voxel space yields a 0.9 NDS improvement, illustrating that height information is crucial for precise pose and detection.
- **Z-axis Layers**: A 2-layer height resolution achieves the best balance (more layers increase computation but offer marginal gains, as object height distribution in road scenes is limited).
- **Masking Ratio**: The optimal combination is 70% for LiDAR and 75% for Camera; ratios too high or too low degrade performance.
- **Data Efficiency**: With only 20% labeled data, UniM2AE achieves the most significant improvement compared to training from scratch (+4.4 mAP), validating the value of self-supervised pre-training when labels are scarce.
- On the BEV segmentation task, UniM2AE with MMIM outperforms X-Align by 2.1 mIoU, reflecting strong transferability across tasks.

## Highlights & Insights
- **3D voxel space as a unified representation** is a simple yet insightful design: it resolves the fundamental issue of BEV losing height information, while naturally supporting bidirectional projection (necessary for both encoding and decoding), allowing the MAE framework to function seamlessly. This design can be extended to any multi-modal 3D understanding task.
- **The transferability of MMIM** is a practical highlight—the pre-trained MMIM can be directly integrated into downstream fusion models, yielding significant improvements (+5.1 NDS) and eliminating the necessity of training fusion modules from scratch.
- In experiments with 20% data, PiMAE improves by only 1.1 NDS (since image plane projection loses depth details), whereas UniM2AE improves by 4.4 NDS, highlighting the paramount importance of representation space selection.

## Limitations & Future Work
- Using a random masking strategy without considering correlation between the two modalities—complementary masking (masking a region in one modality while keeping the corresponding region in the other) might compel more thorough cross-modal learning.
- Temporal continuity is overlooked—adjacent frames in nuScenes are highly similar, so the current method repeats pre-training on redundant data, which reduces efficiency. Incorporating temporal masking or inter-frame contrastive learning could be a promising direction.
- The voxel resolution is fixed (0.15m), which may provide insufficient representation precision for small objects (e.g., pedestrians, bicycles).
- Evaluation is limited to nuScenes; it has not been tested on larger-scale datasets such as Waymo or Argoverse.
- Pre-training is computationally expensive (200 epochs on 8 GPUs), and its scale remains limited compared to modern large-scale pre-training.

## Related Work & Insights
- **vs. Voxel-MAE**: Pre-trains LiDAR only, neglecting image information; UniM2AE fuses both through a unified space.
- **vs. GreenMIM**: Pre-trains images only within the 2D space; UniM2AE extends this to 3D multi-modality.
- **vs. PiMAE**: Aligns LiDAR and images on the 2D image plane, introducing geometric distortion; UniM2AE aligns them in the 3D voxel space, suffering from less information loss.
- **vs. BEVFusion**: BEVFusion fuses features in the BEV space, whereas UniM2AE utilizes the voxel space to preserve more information; the two are complementary, as pre-trained weights from UniM2AE can initialize BEVFusion.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a unified 3D voxel space and multi-modal MAE is implemented in autonomous driving for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Various data ratios, multiple downstream tasks, in-depth ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic methodology description.
- Value: ⭐⭐⭐⭐ Provides an effective, unified framework for multi-modal self-supervised pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)
- [\[ICLR 2026\] GaussianFusion: Unified 3D Gaussian Representation for Multi-Modal Fusion Perception](../../ICLR2026/autonomous_driving/gaussianfusion_unified_3d_gaussian_representation_for_multi-modal_fusion_percept.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[ECCV 2024\] 4D Contrastive Superflows are Dense 3D Representation Learners](4d_contrastive_superflows_are_dense_3d_representation_learners.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
