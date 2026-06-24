---
title: >-
  [Paper Note] OccMamba: Semantic Occupancy Prediction with State Space Models
description: >-
  [CVPR 2025][Autonomous Driving][Semantic Occupancy] OccMamba introduces SSM/Mamba into outdoor semantic occupancy prediction. It serializes 3D voxels into 1D sequences via a height-prioritized 2D Hilbert flattening strategy, and uses a hierarchical Mamba structure coupled with a local context processor to model both global and local contexts. It achieves state-of-the-art (SOTA) results on OpenOccupancy, SemanticKITTI, and SemanticPOSS, with GPU memory consumption far lower th…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Semantic Occupancy"
  - "Mamba"
  - "Hilbert Curve"
  - "LiDAR-Camera Fusion"
  - "Long-range Modeling"
date: 2026-05-08
content_hash: a00fe366760ef991
---

# OccMamba: Semantic Occupancy Prediction with State Space Models

**Conference**: CVPR 2025  
**arXiv**: [2408.09859](https://arxiv.org/abs/2408.09859)  
**Code**: [https://github.com/USTCLH/OccMamba](https://github.com/USTCLH/OccMamba)  
**Area**: Autonomous Driving / 3D Perception / Semantic Occupancy Prediction  
**Keywords**: Semantic Occupancy, Mamba, Hilbert Curve, LiDAR-Camera Fusion, Long-range Modeling

## TL;DR
OccMamba introduces SSM/Mamba into outdoor semantic occupancy prediction. It serializes 3D voxels into 1D sequences via a height-prioritized 2D Hilbert flattening strategy, and uses a hierarchical Mamba structure coupled with a local context processor to model both global and local contexts. It achieves state-of-the-art (SOTA) results on OpenOccupancy, SemanticKITTI, and SemanticPOSS, with GPU memory consumption far lower than Transformer-based approaches.

## Background & Motivation
1. **Background**: Semantic occupancy prediction, which outputs occupancy states and category labels for large-scale 3D voxels (millions of voxels), is a crucial perception task for autonomous driving, AR, and robotics.
2. **Limitations of Prior Work**: 
    - Single-modal (LiDAR or Camera) methods suffer from insufficient information, whereas multi-modal CNNs (such as M-CONet) fail to capture global contexts.
    - Transformer-based approaches (such as OccFormer and OccNet) scale quadratically $\mathcal{O}(N^2)$ in complexity, causing GPU memory explosion when processing large numbers of voxels. Consequently, they must compromise accuracy through deformable attention or spatial projections.
3. **Key Challenge**: Large-scale semantic occupancy prediction requires both global context modeling and computationally feasible scaling for millions of voxels.
4. **Goal**: Achieve simultaneous global, local, and cross-modal fusion capabilities within linear computational complexity.
5. **Key Insight**: While Mamba (SSM) has been proven to achieve global modeling with linear complexity in NLP and 3D point cloud domains, it acts as a 1D sequence model. Flattening 3D voxel grids directly into 1D sequences disrupts their spatial adjacency.
6. **Core Idea**: Design a serialization strategy tailored for the "horizontally wide and vertically short" structures typical of driving scenes. The method stacks voxels along the $z$-axis (height) first to form vertical columns, and then scans these columns in the $xy$-plane using a 2D Hilbert curve, maximizing the preservation of spatial adjacency.

## Method

### Overall Architecture
Based on the M-CONet topology:
- **Multi-modal Encoding**: LiDAR inputs are processed by sparse convolutions to obtain $\mathbf{V}_\mathcal{L}$. Multi-view images are mapped into voxel space through ResNet, FPN, and View Transformer components to obtain $\mathbf{V}_\mathcal{C}$. Both representations are concatenated along the channel dimension to yield $\mathbf{V}_\mathcal{F}$.
- **OccMamba Encoder**: A hierarchical Mamba module (structured with encoder-decoder layouts and skip connections) followed by a local context processor outputs $\mathbf{V}_\mathcal{P}$.
- **Occupancy Head**: Implements coarse-to-fine upsampling coupled with an MLP to predict semantic categories for each voxel.

### Key Designs

1. **Height-Prioritized 2D Hilbert Flattening**
    - **Function**: Serializes 3D voxels into a 1D sequence with high quality, ensuring spatially adjacent voxels remain as close as possible within the 1D sequence.
    - **Mechanism**: The 3D coordinates $(x,y,z)$ are decoupled into the horizontal $xy$-plane and the vertical $z$-axis. Starting from $z=0$, voxels are first grouped along the $z$-axis to form vertical columns, which are then connected sequentially by traversing the $xy$-plane using a 2D Hilbert curve. Formally, this is expressed as $\mathbf{V} = \mathcal{R}_{1D\to 3D}(\mathcal{R}_{3D\to 1D}(\mathbf{V}))$.
    - **Design Motivation**: Driving scenes feature a flat structure where the $z$-axis dimension is much smaller than the $xy$-plane, and height information acts as a strong category prior (e.g., road surfaces, vegetation, and vehicles correspond to specific height segments). Clustering along the $z$-axis first and then sliding via 2D Hilbert curves on the $xy$-plane yields shorter sequential distances for adjacent voxels than conventional sweeps (like XYZ, ZXY) or standard 3D Hilbert curves, fitting the geometric priors of autonomous driving.

2. **Hierarchical Mamba Module (Encoder-Decoder)**
    - **Function**: Aggregates global contextual information across multiple resolutions.
    - **Mechanism**: Each encoding stage consists of two Mamba blocks followed by downsampling; the decoder symmetrically contains Mamba blocks, upsampling, and skip connections. Each Mamba block contains standard layers: LN $\rightarrow$ Linear $\rightarrow$ Conv1D $\rightarrow$ SiLU $\rightarrow$ Selective SSM $\rightarrow$ Gated Path $\rightarrow$ Linear, possessing a linear complexity of $\mathcal{O}(L)$ for a sequence of length $L$. Voxel reordering/de-reordering is executed before and after each block to maintain spatial topology.
    - **Design Motivation**: Occupancy prediction requires fine-grained accuracy at both large scales (whole scenes) and small scales (individual vehicles). Hierarchical downsampling and skip connections allow the network to handle multi-scale information concurrently.

3. **Local Context Processor**
    - **Function**: Compensates for fine-grained local details that might be overlooked during global Mamba modeling.
    - **Mechanism**: The global Mamba output $\mathbf{V}_\mathcal{M}$ is partitioned into patches over the $xy$-plane using multiple window sizes $\{w_i\}$ and strides $\{s_i\}$. Mamba blocks are applied to each patch individually. Output features from different window scales are concatenated along the channel dimension and compressed via a $1\times 1\times 1$ 3D convolution.
    - **Design Motivation**: Different local window sizes correspond to various object scales (such as pedestrians, vehicles, or trucks), behaving similarly to multi-scale local attention.

### Loss & Training
- Implements standard semantic occupancy cross-entropy, Lovasz-Softmax, and scale-invariant loss (following M-CONet).
- Only the newly introduced modules are trained, while the weights of the LiDAR and camera backbones are frozen or reused.

## Key Experimental Results

### Main Results
**OpenOccupancy Val Set** (Camera+LiDAR):

| Method | IoU | mIoU |
|------|-----|------|
| MonoScene (C) | 18.4 | 6.9 |
| M-CONet (C+L) | ~30 | ~20 |
| Co-Occ (C+L, Prev. SOTA) | 31.2 | 22.5 |
| **OccMamba (C+L)** | **36.3** | **26.8** |

Ours achieves a gain of **+5.1 IoU / +4.3 mIoU**.

**SemanticKITTI** (Single-modal / Multi-modal) and **SemanticPOSS**: New SOTA results across all benchmarks.

**GPU Memory Comparison** (Fig. 1b): As the voxel resolution scales beyond $256^3$, the GPU memory footprint of OccMamba increases linearly, whereas Transformer-based methods exhibit quadratic scaling. Transformer-based models experience Out-Of-Memory (OOM) errors at $512\times 512\times 40$, while OccMamba remains trainable.

### Ablation Study

| Configuration | mIoU (OpenOccupancy) |
|------|-----|
| XYZ Scan | 24.x |
| ZXY Scan | 25.x |
| 3D Hilbert Curve | 25.5 |
| **Height-prioritized 2D Hilbert (Ours)** | **26.8** |
| W/o Hierarchical Mamba | Significant Drop |
| W/o Local Context Processor | -1.x mIoU |

### Key Findings
- The flattening strategy is vital for Mamba performance; suboptimal serialization runs the risk of a 1–2% drop in mIoU.
- Scanning priorities along the spatial axes (z-axis first) are key in driving scenarios, as spatial categories (e.g., roads, sky, vehicle roofs) are heavily correlated with vertical heights ($z$).
- Mamba allows OccMamba to process dense voxels **without compression**, avoiding the need for keypoint selection as in deformable attention, which guarantees robust performance under occlusions.

## Highlights & Insights
- **First Mamba-based network for outdoor semantic occupancy prediction**, proving the feasibility of state space models (SSMs) for large-scale 3D perception tasks.
- **Task-prior-driven flattening strategy**: Instead of directly copying standard 3D Hilbert curves, this work models the geometric flat structures and height-specific category distributions of driving scenes to design a task-specific scan. This paradigm of "tailoring serialization architectures to task-specific geometry" can easily adapt to other 3D tasks such as semantic segmentation, object detection, and completion.
- **Linear complexity enables uncompressed multi-modal fusion**: Previous approaches suffered from GPU memory limits and had to rely on BEV or range-view compression. OccMamba predicts based directly on dense voxels, leading to superior restoration in occluded spaces.
- **Local Context Processor acts as an elegant patch for Mamba**: While global SSMs tend to overlook fine-grained details, patch-wise Mamba blocks elegantly restore local spatial patterns.

## Limitations & Future Work
- The voxel-prior scan is still manually crafted; learning the optimal sequence traversal in an end-to-end manner remains an open question.
- Mamba is highly order-sensitive, and bidirectional scaling comes with heavy computational overhead; unidirectional processing might overlook backward dependencies.
- Temporal 4D occupancy modeling has not yet been integrated.
- Extremely large scale network capacities have not been fully explored, with validation limited to medium model sizes.
- Future Directions: Introducing study-based learnable permutations or random shuffle augmentations alongside the z-prior + xy-Hilbert sequence to improve generalization across unseen urban scenarios.

## Related Work & Insights
- **vs Co-Occ (CVPR 24)**: Co-Occ relies on high-overhead transformer projection layers, whereas OccMamba uses an SSM framework directly on high-density voxels, leading to solid improvements in IoU/mIoU.
- **vs PointMamba**: PointMamba relies on 3D Hilbert curves to serialize unordered point cloud data, while OccMamba designs a specialized elevation-first sequence scanning flow for dense driving voxels, resulting in better alignment with occupancy setups.
- **vs OccFormer/OccNet**: Compared to other Transformer-based models, OccMamba achieves a Pareto improvement in both memory efficiency and performance.
- **Insight**: Any 3D task featuring large but non-uniform volume distributions can benefit from combining "physically prioritized axes + Hilbert serialization".

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Mamba to occupancy modeling is novel, and the task-driven reordering strategy is targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on three key benchmarks, with extensive memory usage comparisons and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Mathematically and visually clear description, with an intuitive explanation of the Hilbert curves.
- Value: ⭐⭐⭐⭐ Significantly lowers the GPU memory requirements of occupancy models, serving as a powerful new baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2025\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](gdfusion_temporal_fusion_occupancy.md)

</div>

<!-- RELATED:END -->
