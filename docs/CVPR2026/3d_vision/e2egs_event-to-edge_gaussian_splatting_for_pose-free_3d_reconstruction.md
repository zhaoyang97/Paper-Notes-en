---
title: >-
  [Paper Note] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction
description: >-
  [CVPR 2026][3D Vision][Event camera] This paper proposes E2EGS, a fully pose-free 3D reconstruction framework driven entirely by event streams. It extracts noise-robust edge maps from event streams via patch-based tempor…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Event camera"
  - "3D Gaussian splatting"
  - "edge detection"
  - "pose-free reconstruction"
  - "visual odometry"
date: 2026-05-08
content_hash: c3052b55010b76d6
---

# E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.14684](https://arxiv.org/abs/2603.14684)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: Event camera, 3D Gaussian splatting, edge detection, pose-free reconstruction, visual odometry

## TL;DR

This paper proposes E2EGS, a fully pose-free 3D reconstruction framework driven entirely by event streams. It extracts noise-robust edge maps from event streams via patch-based temporal consistency analysis, leverages edge information to guide Gaussian initialization and weighted loss optimization, and achieves high-quality trajectory estimation and 3D reconstruction without any depth model or RGB input.

## Background & Motivation

NeRF and 3D Gaussian Splatting (3DGS) have driven remarkable progress in novel view synthesis, yet both fundamentally rely on high-quality RGB images and accurate camera poses. In real-world scenarios involving rapid motion or adverse lighting, RGB image quality degrades severely, limiting the robustness of these methods.

**Advantages of event cameras**: Event cameras asynchronously capture per-pixel brightness changes with extremely high temporal resolution and wide dynamic range, making them naturally suited for handling motion blur and extreme illumination. Crucially, event cameras produce dense responses at edges and texture boundaries, providing rich structural information about scene geometry.

**Limitations of prior work**:

**Methods requiring known poses** (EventSplat, Ev-GS, Event3DGS, etc.): rely on SfM or ground-truth poses and cannot be applied when poses are unavailable.

**IncEventGS** (the only existing pose-free method): adopts a SLAM framework for joint optimization of poses and 3D Gaussians, but critically depends on the pretrained depth estimation model Marigold for initialization. **Core problem**: the depth model estimates from the initial frame, and as the camera moves into new regions beyond the initial coverage, depth estimation degrades → trajectory drift → reconstruction quality collapses.

**Key Challenge**: Pose-free event-based 3D reconstruction requires reliable geometric priors to guide optimization, yet existing methods either depend on external depth models (poor generalization) or resort to fully random initialization (prone to local optima).

**Key Insight**: Event cameras inherently encode edge information — during camera motion, edge regions produce temporally consistent dense events, whereas non-edge regions yield only sparse noise. This spatiotemporal contrast can be exploited to extract robust edge maps that provide geometric constraints for Gaussian initialization and pose optimization, entirely without depth models or RGB assistance.

**Core Idea**: Replace the depth model with the intrinsic edge information embedded in the event stream, enabling fully autonomous pose-free event-based 3D reconstruction.

## Method

### Overall Architecture

E2EGS adopts 3DGS as the scene representation and follows the tracking–mapping alternating optimization framework of IncEventGS. The input is a raw event stream; the output is a 3D Gaussian scene representation and a camera trajectory. The event stream is processed in temporal chunks, each associated with a segment of the continuous trajectory. Event supervision is achieved by minimizing the discrepancy between the measured event map $E_t(\mathbf{x})$ and the synthesized event map $\hat{E}_t(\mathbf{x}) = \log \hat{I}_{t+\Delta t}(\mathbf{x}) - \log \hat{I}_t(\mathbf{x})$.

The core innovation of E2EGS is a three-stage edge-guided pipeline: (1) extract robust edges from consecutive event maps → (2) initialize 3D Gaussians along detected edges → (3) apply edge-weighted losses throughout tracking and bundle adjustment.

### Key Designs

1. **Patch-based Temporal Consistency Analysis (Edge Detection)**

    - **Function**: Extracts noise-robust edge maps from noisy event streams without any training.
    - **Mechanism**: Given $T$ consecutive event maps $\{E_t\}_{t=1}^T$, they are divided into overlapping $p \times p$ patches. For each patch location $P_{x,y}$, the temporal difference signal between adjacent frames is computed as $D_t(P_{x,y}) = |G_\sigma * E_t(P_{x,y}) - G_\sigma * E_{t-1}(P_{x,y})|$ (Gaussian smoothing suppresses sharp artifacts). The consistency score is then $C(P_{x,y}) = \max_{t} \text{Var}(D_t(P_{x,y}))$; patches with variance exceeding threshold $\tau$ are classified as containing edges.
    - **Design Motivation**: During camera motion, edge regions produce spatially coherent, structured event patterns across consecutive frames (high variance), whereas non-edge regions generate only sparse random noise (low variance). This variance contrast serves as a natural edge/non-edge discriminator. The approach draws inspiration from the contrast maximization framework while avoiding its computationally expensive trajectory estimation step.
    - **Self-correction mechanism**: Temporal differencing simultaneously captures edges from both the previous frame $E_{t-1}$ and the current frame $E_t$; incorrectly estimated edges are automatically suppressed when they are inconsistent with current observations.

2. **Edge-aware Gaussian Initialization**

    - **Function**: Places 3D Gaussians along detected edges, replacing depth-model-based initialization.
    - **Mechanism**: 2D edge points $\mathcal{P}$ are extracted from the edge map; KNN and PCA are applied to obtain the edge normal direction at each point. Recursive grid subdivision (split criterion based on normal direction consistency) generates a 2D edge Gaussian set $\mathcal{G}_\text{edge}$. 3D points are then placed along the viewing ray of each edge Gaussian via inverse depth sampling: $d = \frac{1}{\frac{1}{d_\max} + u(\frac{1}{d_\min} - \frac{1}{d_\max})}$, $u \sim \mathcal{U}(0,1)$.
    - **Complementary depth and surface sampling**: An edge ratio $r_\text{edge} \in [0,1]$ controls the proportion of edge-based and random Gaussians. $N_\text{edge} = \lfloor r_\text{edge} \cdot N_\text{total} \rfloor$ Gaussians are initialized along edges; the remaining $N_\text{random}$ are placed randomly to cover textureless regions.
    - **Design Motivation**: Inverse depth sampling places more samples at greater distances — distant points produce larger pixel displacements under camera rotation, improving the observability of rotational motion and thereby enhancing pose estimation accuracy. This is geometrically more principled than uniform depth sampling.

3. **Edge-weighted Loss Function**

    - **Function**: Applies edge-weighted reconstruction loss throughout initialization, tracking, and bundle adjustment.
    - **Mechanism**: The edge-weighted loss is $\mathcal{L}_\text{edge} = \frac{1}{|\Omega|} \sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \cdot \|\hat{E}(\mathbf{x}) - E(\mathbf{x})\|^2$, where the weight $w(\mathbf{x}) = 1 + \beta \cdot M(\mathbf{x})$ and $\beta$ controls the degree of edge emphasis. The total loss is $\mathcal{L}_\text{total} = (1-\lambda) \mathcal{L}_\text{edge} + \lambda \mathcal{L}_\text{dssim}$.
    - **Design Motivation**: Event cameras produce sparse noise in non-edge regions; uniform pixel-level loss treats these noisy events equally, introducing unreliable gradient signals. Edge weighting focuses optimization on geometrically salient boundary regions, where geometric constraints are more robust and less susceptible to event noise.

### Loss & Training

The total loss combines the edge-weighted reconstruction loss and a structural similarity loss. The system alternates between tracking and mapping: during tracking, Gaussian parameters are fixed and the pose of each new chunk is optimized; during mapping, Gaussian parameters and trajectory are jointly optimized within a sliding window.

## Key Experimental Results

### Main Results (Novel View Synthesis — Replica Dataset)

| Method | Depth Dep. | Pose Dep. | room0 PSNR↑ | office0 PSNR↑ | office3 PSNR↑ |
|--------|-----------|-----------|-------------|---------------|---------------|
| EvGGS | ✗ | Known | 17.57 | 14.34 | 15.51 |
| Event-3DGS* | ✗ | E2VID+COLMAP | 22.27 | 15.97 | 17.82 |
| IncEventGS | Marigold | ✗ | 23.54 | 26.53 | 19.21 |
| IncEventGS† | ✗ | ✗ | 19.81 | 27.72 | 20.04 |
| **E2EGS** | **✗** | **✗** | **23.86** | **28.01** | **20.75** |

### Main Results (Trajectory Accuracy — ATE RMSE cm)

| Method | room0 | room2 | office0 | TUM-VIE 1d | TUM-VIE 3d | TUM-VIE 6dof |
|--------|-------|-------|---------|-----------|-----------|-------------|
| DEVO | 0.271 | 0.381 | 0.287 | 0.23 | 1.00 | 1.82 |
| IncEventGS | 0.051 | 0.071 | 0.085 | 2.19 | 1.62 | 0.70 |
| IncEventGS† | 6.817 | 0.446 | 0.698 | 2.58 | 4.48 | 8.24 |
| **E2EGS** | **0.049** | **0.065** | **0.078** | **1.12** | **0.65** | **0.58** |

Key comparison: IncEventGS† reaches an ATE of 96.59 cm on TUM-VIE desk2 (catastrophic failure), whereas E2EGS achieves only 0.40 cm.

### Ablation Study

| Configuration | Edge loss | Edge init | Depth init | ATE (cm) |
|---------------|-----------|-----------|------------|----------|
| IncEventGS | ✗ | ✗ | ✓ | 0.37 |
| IncEventGS† | ✗ | ✗ | ✗ | 6.62 |
| w/ Edge loss only | ✓ | ✗ | ✗ | 0.50 |
| w/ Edge init only | ✗ | ✓ | ✗ | 0.29 |
| **E2EGS (full)** | **✓** | **✓** | **✗** | **0.28** |

Edge ratio $r_\text{edge}$ ablation: $r_\text{edge} = 0.0$ → ATE 5.68 cm; $r_\text{edge} \in [0.1, 0.3]$ → ATE ~0.40 cm (optimal); $r_\text{edge} = 1.0$ → ATE 11.93 cm (excessive edge emphasis causes insufficient surface coverage).

### Key Findings

- **Edge init contributes more than edge loss**: Adding edge init alone reduces ATE from 6.62 to 0.29 cm, outperforming the depth-based baseline (0.37); adding edge loss alone reduces it to 0.50. The combination yields the best result (0.28).
- **Robustness over long sequences**: As sequence length increases, the ATE of IncEventGS (depth-dependent) rises sharply due to progressively unreliable depth estimates, whereas E2EGS maintains consistently low error — edges are local features unaffected by sequence length.
- **Without vs. with depth**: E2EGS surpasses IncEventGS (with depth) on synthetic data and substantially outperforms it on real data, all without any depth model.
- **Edge ratio requires balance**: Too low (0.0) lacks geometric constraints and causes drift; too high (0.7–1.0) results in insufficient coverage, allowing non-edge region losses to dominate and mislead optimization.

## Highlights & Insights

- The **training-free edge detection** design is elegant: edges are extracted purely from the spatiotemporal statistics of the event stream without any learning, exploiting the most fundamental physical property of event cameras (motion + edges → consistent events) in a simple yet effective manner.
- **Complete elimination of depth models** represents an important paradigm shift: the depth model degradation in unseen regions is a fundamental flaw of IncEventGS; E2EGS replaces depth with edges as geometric priors, unconstrained by observation range.
- The geometric intuition behind **inverse depth sampling** is transferable: distant points are more sensitive to rotation → denser sampling at distance → improved observability of rotational motion. This insight can be adapted to other SLAM systems that require depth initialization.
- The **edge-weighted loss** is essentially an attention mechanism: it focuses optimization on the most information-rich pixels, preventing gradient signals from being overwhelmed by noisy regions.

## Limitations & Future Work

- The edge extraction strategy is conservative (prioritizing reliability over completeness) and may fail to extract sufficient edges in regions with weak geometric structure (e.g., large textureless walls).
- Event cameras inherently produce sparse responses on textureless planar regions, limiting the quality of Gaussians in those areas (acknowledged by the authors).
- The edge ratio $r_\text{edge}$ requires manual tuning; an adaptive strategy warrants exploration.
- Evaluation is conducted only on indoor datasets; performance in outdoor large-scale scenes (e.g., autonomous driving) remains unknown.
- No end-to-end comparison is made with recent learning-based event visual odometry methods (e.g., RAMP-VO, which fuses events and images).

## Related Work & Insights

- **vs. IncEventGS**: Both are pose-free event-based 3DGS methods, but IncEventGS depends on the Marigold depth model for initialization and degrades severely over long sequences or in new regions; E2EGS replaces depth with edges, fundamentally addressing the generalization problem.
- **vs. DEVO**: DEVO is a learning-based event visual odometry method with good trajectory accuracy but does not perform 3D reconstruction; E2EGS unifies trajectory estimation and scene reconstruction.
- **vs. Contrast Maximization (CMax-SLAM)**: CMax estimates motion by maximizing the sharpness of warped event maps, which is computationally expensive; E2EGS extracts edges directly from temporal consistency, achieving greater efficiency.
- Takeaway: The physical properties of event cameras inherently encode rich geometric cues; relying on pretrained vision models is unnecessary, and returning to sensor-level principles may be a more robust path forward.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Using edges in place of a depth model as geometric priors for event-based 3DGS is a clear and effective idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on synthetic and real datasets with component and parameter ablations; outdoor large-scale evaluation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated, method description is well-organized, and figures and tables are intuitive.
- **Value**: ⭐⭐⭐⭐ First fully dependency-free pose-free event-based 3D reconstruction system, expanding the autonomy frontier of event camera applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](../../NeurIPS2025/3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [\[CVPR 2026\] Global-Aware Edge Prioritization for Pose Graph Initialization](global-aware_edge_prioritization_for_pose_graph_initialization.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
