---
title: >-
  [Paper Note] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][Panoptic Mapping] This paper proposes OnlinePG, the first online open-vocabulary panoptic mapping system built upon 3DGS. It adopts a local-to-global paradigm: within a sliding window, a multi-cue clustering graph (geometric overlap + semantic similarity + view consensus) constructs locally consistent 3D instances, which are then incrementally merged into a global map via bidirectional bipartite matching. OnlinePG achieves state-of-the-art semantic and panoptic segmentation among online methods, surpassing OnlineAnySeg by +17.2 mIoU on ScanNet (48.48), while running at 10–18 FPS.
tags:
  - CVPR 2026
  - 3D Vision
  - Panoptic Mapping
  - Open-Vocabulary
  - 3D Gaussian Splatting
  - Online Reconstruction
  - Instance Segmentation
date: 2026-05-08
content_hash: 7bd7906ee4258262
---

# OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2603.18510](https://arxiv.org/abs/2603.18510)
**Institution**: State Key Lab of CAD&CG, Zhejiang University; VIVO BlueImage Lab; HKUST
**Area**: 3D Vision
**Keywords**: Panoptic Mapping, Open-Vocabulary, 3D Gaussian Splatting, Online Reconstruction, Instance Segmentation

## TL;DR

This paper proposes OnlinePG, the first online open-vocabulary panoptic mapping system built upon 3DGS. It adopts a local-to-global paradigm: within a sliding window, a multi-cue clustering graph (geometric overlap + semantic similarity + view consensus) constructs locally consistent 3D instances, which are then incrementally merged into a global map via bidirectional bipartite matching. OnlinePG achieves state-of-the-art semantic and panoptic segmentation among online methods, surpassing OnlineAnySeg by +17.2 mIoU on ScanNet (48.48), while running at 10–18 FPS.

## Background & Motivation

**Background**: Open-vocabulary 3D scene understanding is foundational for embodied intelligence. Recent works lift 2D VLM features (CLIP, LSeg, SAM) into 3D space via NeRF/3DGS, achieving strong results in offline settings (LangSplat, OpenGaussian, PanoGS, etc.).

**Limitations of Prior Work**:
   - **(a) Offline Constraint**: Most methods (PanoGS, LangSplat, OpenGaussian) require pre-collected complete data and global optimization, making them unsuitable for real-time robotic tasks.
   - **(b) Lack of Instance-Level Understanding**: The online method O2V-Mapping provides only semantic segmentation without distinguishing individual instances of the same class.
   - **(c) 2D Segmentation Noise**: 2D segmentations from VLMs are inconsistent across views (over- and under-segmentation), and direct lifting to 3D leads to noise accumulation.
   - **(d) Slow Contrastive Learning Convergence**: Offline methods (InstanceGaussian, PanoGS) rely on slowly converging contrastive feature learning for instance clustering, which is unsuitable for online systems.

**Key Challenge**: 2D VLM segmentation results are inconsistent across views (over/under-segmentation), and direct 3D lifting produces noisy instances. The core challenge is obtaining 3D-consistent panoptic instances and semantics under online streaming input.

**Goal**: (1) Online panoptic (instance + semantic) mapping; (2) deriving consistent 3D instances from noisy 2D segmentations; (3) open-vocabulary querying.

**Key Insight**: Resolve 2D inconsistencies locally within a sliding window via multi-cue clustering, then incrementally merge into a global map.

**Core Idea**: A local-to-global paradigm — multi-cue segment clustering within a sliding window → locally consistent instances → bidirectional bipartite matching → globally consistent panoptic map.

## Method

### Overall Architecture

Input: RGB-D stream with poses. Output: 3D Gaussian panoptic map with open-vocabulary query capability.

Pipeline overview:

```
RGB-D Stream → Keyframe Selection (every 20 frames) → Sliding Window (size 12)
                                                              ↓
                                                   2D VLM Feature Extraction
                                                   (LSeg + EntitySeg)
                                                              ↓
                                              3D Gaussian Segment Initialization
                                              (grouped by mask)
                                                              ↓ every 7 keyframes
                                        Multi-Cue Clustering Graph (Geometric + Semantic + View Consensus)
                                                              ↓
                                           Locally Consistent 3D Instances + Spatial Attribute Grid
                                                              ↓
                                        Bidirectional Bipartite Matching → Global Map Incremental Fusion
                                                              ↓
                                           Global Panoptic Map + Open-Vocabulary Query
```

### Key Design 1: Scene Representation — 3D Gaussians + Voxelized Spatial Attributes

**Function**: Model geometric and semantic information using two complementary representations.

**3D Gaussian Primitives**: Each Gaussian is defined as $\mathcal{G}_i := \{\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \sigma_i, \boldsymbol{c}_i\}$, comprising position $\boldsymbol{\mu}_i \in \mathbb{R}^3$, covariance $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3 \times 3}$, opacity $\sigma_i$, and color $\boldsymbol{c}_i \in \mathbb{R}^3$, used for geometric reconstruction and differentiable rendering.

**Voxelized Spatial Attributes**: The reconstructed region is voxelized (voxel size 3 cm), and each occupied voxel stores four attributes:

| Attribute | Symbol | Dimension | Purpose |
|-----------|--------|-----------|---------|
| Language Feature | $\mathcal{F}$ | $\mathbb{R}^{512}$ | Stores VLM language features for open-vocabulary querying |
| Feature Confidence | $\mathcal{C}$ | $\mathbb{R}$ | Weight for multi-view feature fusion |
| Instance Label | $\mathcal{T}$ | $\mathbb{R}$ | Panoptic instance ID |
| Instance Weight | $\mathcal{K}$ | $\mathbb{R}$ | Confidence of instance label for global fusion decisions |

**Design Motivation**: 3D Gaussians are a continuous representation suited for geometric optimization and rendering; discrete voxel grids are suited for storing instance labels and performing incremental updates. The two representations are complementary, each leveraged for its strengths.

### Key Design 2: Multi-Cue Segment Clustering — From Noisy 2D to Consistent 3D Instances

**Function**: Merge inconsistent 3D segments from multiple keyframes within the sliding window into consistent 3D instances.

**Step 1 — 3D Gaussian Segment Initialization**: 3D Gaussian primitives are initialized by projecting depth maps from each keyframe, and grouped into 3D segments $\mathcal{S}_i := \{\mathcal{G}_j\}_{j=1}$ according to 2D mask IDs. The full set of segments in the window is $\mathcal{S} := \{\mathcal{S}_1, \cdots, \mathcal{S}_n\}$, where $n = \sum_{i \in \mathcal{W}} |m_i|$.

**Step 2 — Building the Multi-Cue Clustering Graph**: Graph vertices are 3D segments $\mathcal{S}_i$; edges $\mathcal{E}_{ij}$ are determined jointly by three affinity cues:

**(1) Geometric Overlap Cue $\mathcal{O}$**: Segments are voxelized and the bidirectional visible voxel overlap ratio is computed as a symmetric average:

$$\mathcal{O}(\mathcal{S}_i, \mathcal{S}_j) = \frac{1}{2} \cdot \left(\frac{|\mathcal{S}_i \cap \mathcal{S}_j|}{\text{Cont.}(\mathcal{S}_i, \mathcal{S}_j)} + \frac{|\mathcal{S}_i \cap \mathcal{S}_j|}{\text{Cont.}(\mathcal{S}_j, \mathcal{S}_i)}\right)$$

where $\text{Cont.}(\mathcal{S}_i, \mathcal{S}_j)$ denotes the ratio of visible voxels of $\mathcal{S}_j$ contained when projected back to the viewpoint of $\mathcal{S}_i$. The symmetric average mitigates bias between segments of different sizes.

**(2) Semantic Similarity Cue $\mathcal{X}$**: Segment-level language features $z_i = \Phi(\{f(u,v): m(u,v) = i\})$ are obtained by average-pooling LSeg feature maps over 2D masks, and cosine similarity is computed:

$$\mathcal{X}(\mathcal{S}_i, \mathcal{S}_j) = \frac{z_i \cdot z_j}{\|z_i\| \cdot \|z_j\|}$$

**(3) View Consensus Cue $\mathcal{V}$**: The proportion of co-visible keyframes in which two segments are labeled as the same instance:

$$\mathcal{V}(\mathcal{S}_i, \mathcal{S}_j) = \frac{N_{\text{supp}}(\mathcal{S}_i, \mathcal{S}_j)}{N_{\text{vis}}(\mathcal{S}_i, \mathcal{S}_j)}$$

**Step 3 — Clustering Criterion and Connected Component Merging**: A composite criterion determines whether two segments should be merged:

$$\Delta_{ij} = \left((\mathcal{O}_{ij} + \mathcal{X}_{ij}) > \lambda_1\right) \cup \left(\mathcal{V}_{ij} > \lambda_2\right)$$

where $\lambda_1 = 1.5$ and $\lambda_2 = 0.8$. Consistent instances are obtained via connected component analysis: $\mathcal{I} = \text{Cluster}(\{\mathcal{S}_i\}, \{\mathcal{E}_{ij}\})$.

**Design Motivation**: No single cue suffices — geometric overlap fails for co-located segments with different semantics; semantic similarity fails for same-class different instances; view consensus is unreliable under sparse observations. The three cues are complementary, and the OR logic in the merging criterion enables correct associations across diverse scene configurations.

### Key Design 3: Bidirectional Bipartite Matching — Robust Local-to-Global Fusion

**Function**: Incrementally merge locally consistent instances from the sliding window into the global map, ensuring global consistency.

**Step 1 — Constructing Bidirectional Matching Matrices**:

Forward matrix $\mathcal{M}_{l \to g} \in \mathbb{R}^{n_l \times n_g}$ (semantic similarity + geometric containment ratio):

$$\mathcal{M}_{l \to g} = \frac{z_l \cdot z_g}{\|z_l\| \cdot \|z_g\|} + \frac{|\mathcal{I}_l \cap \mathcal{I}_g|}{\text{Cont.}(\mathcal{I}_l, \mathcal{I}_g)}$$

Backward matrix $\mathcal{M}_{g \to l} \in \mathbb{R}^{n_g \times n_l}$: reverses the geometric containment direction.

**Step 2 — Hungarian Algorithm + Intersection Confirmation**:

$$\mathcal{A} = \text{Hung.}(\mathcal{M}_{l \to g}) \cap \text{Hung.}(\mathcal{M}_{g \to l})^T$$

**Step 3 — Global Map Update Rules**:

- **Language feature fusion** (weighted average): $\mathcal{F}_g^t(v) = \frac{\mathcal{C}_l^t \cdot \mathcal{F}_l^t + \mathcal{C}_g^{t-1} \cdot \mathcal{F}_g^{t-1}}{\mathcal{C}_g^t}$
- **Matched instances**: retain global label, accumulate weight $\mathcal{K}_g^t = \mathcal{K}_g^{t-1} + \mathcal{K}_l^t$
- **Unmatched, local weight ≤ global**: retain global label, reduce weight $\mathcal{K}_g^t = \mathcal{K}_g^{t-1} - \mathcal{K}_l^t$
- **Unmatched, local weight > global**: replace with local label $\mathcal{T}_g^t = \mathcal{T}_l^t$, with the weight difference as the new weight

**Design Motivation**: The local map contains newly explored regions while the global map encodes historical ones, making the geometric containment ratio inherently asymmetric. Bidirectional matching requires agreement in both directions before confirming a correspondence, preventing erroneous fusion. The weight competition mechanism allows high-confidence segmentations to progressively supersede low-confidence ones.

### Loss & Training

3DGS optimization uses a weighted combination of appearance and geometric L1 losses:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_c + (1-\alpha) \cdot \mathcal{L}_d, \quad \alpha = 0.9$$

Upon each new keyframe insertion, 5 historical frames are randomly selected for 20 optimization iterations. Semantic and instance information do not participate in gradient optimization; they are maintained via the discrete update mechanism of the voxel grid.

## Key Experimental Results

### Main Results (3D Semantic and Panoptic Segmentation)

| Method | Online | Panoptic | ScanNet mIoU↑ | ScanNet mAcc↑ | ScanNet PRQ(T)↑ | ScanNet PRQ(S)↑ | Replica mIoU↑ | Replica PRQ(T)↑ | Replica PRQ(S)↑ |
|--------|:------:|:--------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LangSplat* | ✗ | ✗ | 29.47 | 45.29 | 22.57 | 28.44 | 4.82 | 8.29 | 1.28 |
| OpenGaussian* | ✗ | ✗ | 24.89 | 37.35 | 22.87 | 19.71 | – | – | – |
| OpenScene* | ✗ | ✗ | 47.63 | 69.74 | 43.53 | 40.43 | 49.03 | 33.04 | 11.84 |
| InstanceGaussian | ✗ | ✓ | 34.14 | 54.95 | 39.04 | 27.41 | – | – | – |
| PanoGS | ✗ | ✓ | 50.72 | 70.20 | 33.84 | 36.22 | 54.98 | 43.04 | 30.60 |
| O2V-Mapping | ✓ | ✗ | 33.74 | 55.52 | – | – | 24.35 | – | – |
| OnlineAnySeg | ✓ | ✓ | 31.28 | 52.20 | 35.98 | 26.27 | 37.48 | 34.19 | 9.13 |
| **OnlinePG (Ours)** | **✓** | **✓** | **48.48** | **66.01** | **37.97** | **41.81** | **47.93** | **41.02** | **12.83** |

\* Methods marked in gray use supervised 3D instance segmentation as auxiliary input for PRQ computation.

### Ablation Study 1: Matching Strategy (ScanNet)

| Matching Strategy | PRQ(T) | PRQ(S) |
|-------------------|:------:|:------:|
| #1 Nearest Neighbor Matching | 24.67 | 22.98 |
| #2 Forward-Only $\mathcal{M}_{l \to g}$ | 35.83 | 38.40 |
| #3 Backward-Only $\mathcal{M}_{g \to l}$ | 33.71 | 42.72 |
| **#4 Bidirectional Bipartite Matching (Full)** | **37.97** | **41.81** |

### Ablation Study 2: System Components (ScanNet)

| Configuration | mIoU | PRQ(T) | PRQ(S) |
|---------------|:----:|:------:|:------:|
| #1 w/o Segment Clustering (direct keyframe segment fusion) | 48.48 | 32.25 | 30.68 |
| #2 w/o Feature Grid (instance-level coarse features only) | 30.40 | 26.71 | 24.92 |
| **#3 Full System** | **48.48** | **37.97** | **41.81** |

### Key Findings

- **Best among online methods**: mIoU exceeds OnlineAnySeg by **+17.2** (ScanNet) and **+10.5** (Replica).
- **Large gains in panoptic segmentation**: PRQ(S) surpasses OnlineAnySeg by **+15.5** (ScanNet), with particularly notable improvements on Stuff categories.
- **Approaches or exceeds offline SOTA**: The mIoU gap with PanoGS is only 2.24; PRQ(S) surpasses PanoGS (41.81 vs. 36.22).
- **Segment clustering is critical**: Removing it causes PRQ(S) to drop by 11.13, especially for large-area Stuff regions.
- **Feature grid is critical**: Removing it causes mIoU to drop by 18.08; instance-level coarse features lead to severe semantic drift.
- **Bidirectional vs. nearest-neighbor matching**: PRQ improves by +13.3/+18.8; backward verification is especially important for Stuff (+4.3).
- **Multi-cue clustering**: Outperforms single-cue baselines by 8–18 PRQ points, with only ~40 ms additional latency.
- **Real-time performance**: 18 FPS for simple scenes, 10 FPS for complex scenes (excluding VLM frontend).
- **Open-vocabulary advantage**: Significantly outperforms OnlineAnySeg on long-tail concepts (e.g., "bag") and multi-instance queries of the same semantic class (e.g., "pillow" ×N).

## Highlights & Insights

1. **First online 3DGS system with panoptic + open-vocabulary capabilities**: Unifies geometric reconstruction, instance segmentation, and open-vocabulary understanding.
2. **Local-to-Global paradigm**: 2D inconsistencies are resolved locally within the sliding window (small-scale, tractable), and only lightweight matching is performed for incremental global fusion — decomposing a hard global problem into two manageable steps.
3. **Multi-cue clustering graph**: Geometric, semantic, and view-consensus cues are complementary; processes a 12-frame window in only ~350 ms.
4. **Bidirectional bipartite matching**: Taking the intersection of forward and backward Hungarian assignments is more robust than unidirectional matching and elegantly handles the asymmetry between local and global maps.
5. **Complementary voxel attributes and Gaussians**: Gaussians handle continuous rendering optimization; voxels handle discrete semantic/instance updates — each exploited for its respective strengths.
6. **Substantial lead among online methods**: Both mIoU and PRQ significantly exceed OnlineAnySeg and O2V-Mapping, and several metrics surpass the offline method PanoGS.

## Limitations & Future Work

1. **No support for dynamic objects**: The system handles only static scenes; dynamic objects lead to erroneous reconstruction.
2. **Dependency on depth and pose inputs**: Requires RGB-D data and known camera poses, limiting applicability to RGB-only settings.
3. **VLM frontend latency excluded**: The reported 10–18 FPS does not include inference time for LSeg and EntitySeg; the full system's real-time capability is bottlenecked by the frontend.
4. **Primarily indoor scenes**: Evaluated only on ScanNetV2 and Replica indoor datasets; large-scale outdoor scenes are not assessed.
5. **Fixed hyperparameters**: $\lambda_1$, $\lambda_2$, voxel size, clustering frequency, etc. are manually set; adaptive tuning could improve efficiency.

## Rating

| Dimension | Score |
|-----------|:-----:|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](../../NeurIPS2025/3d_vision/segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)

</div>

<!-- RELATED:END -->
