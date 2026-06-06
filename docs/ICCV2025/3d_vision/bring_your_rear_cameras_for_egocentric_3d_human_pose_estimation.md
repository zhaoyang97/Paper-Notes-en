---
title: >-
  [Paper Note] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation
description: >-
  [ICCV 2025][3D Vision][egocentric view] This paper is the first to investigate the value of rear-mounted cameras on HMDs for egocentric 3D whole-body pose estimation. It proposes a Transformer-based multi-view heatmap re…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "egocentric view"
  - "3D human pose estimation"
  - "rear cameras"
  - "multi-view fusion"
  - "head-mounted devices"
date: 2026-05-08
content_hash: ca17267b722d3500
---

# Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation

**Conference**: ICCV 2025
**arXiv**: [2503.11652](https://arxiv.org/abs/2503.11652)  
**Code**: [https://4dqv.mpi-inf.mpg.de/EgoRear/](https://4dqv.mpi-inf.mpg.de/EgoRear/)  
**Area**: 3D Vision / Human Pose Estimation
**Keywords**: egocentric view, 3D human pose estimation, rear cameras, multi-view fusion, head-mounted devices

## TL;DR

This paper is the first to investigate the value of rear-mounted cameras on HMDs for egocentric 3D whole-body pose estimation. It proposes a Transformer-based multi-view heatmap refinement method with an uncertainty-aware masking mechanism, achieving >10% MPJPE improvement on the newly constructed Ego4View dataset.

## Background & Motivation

Egocentric 3D whole-body pose estimation typically relies on cameras mounted at the front of an HMD, a design choice that carries fundamental limitations:

**Severe self-occlusion**: When users look upward — common during physical activity — front-facing cameras can barely observe the body, causing even the state-of-the-art method EgoPoseFormer to fail.

**Limited field of view**: The posterior body region is entirely unobserved, despite containing critical cues for 3D reconstruction.

**Limitations of existing HMD designs**: Apple Vision Pro features eight front-facing sensors yet provides no whole-body tracking, likely due to insufficient accuracy from front-only inputs.

An intuitive remedy is to mount cameras at the rear of the HMD. However, the authors find that **naively incorporating rear views into existing methods does not always help and can even degrade accuracy**. The root cause is that existing methods rely on independent 2D joint detectors without an effective multi-view integration mechanism — self-occlusion and missing body parts in rear views produce inaccurate 2D joint detections, which in turn corrupt the 3D estimation.

## Method

### Overall Architecture

The overall pipeline is: four-view fisheye images → 2D joint heatmap estimation → multi-view heatmap refinement module → refined heatmaps and features → 2D-to-3D lifting module → 3D pose. The core contribution is the intermediate multi-view heatmap refinement module, which can be integrated as a plug-and-play component into existing methods (EgoPoseFormer, EgoTAP).

### Key Designs

1. **2D Joint Heatmap Refinement Module**: Based on a Transformer decoder architecture, this module refines initial heatmap estimates by leveraging multi-view context. The core assumption is that **front and rear views are complementary** — due to human body symmetry, unreliable rear-view heatmaps can be improved by reliable front-view heatmaps, and vice versa.

    - View-specific joint queries $\mathbf{Q}_{\text{front\_left}} \in \mathbb{R}^{15 \times 256}$ are defined per view to encode view-specific 2D skeleton information.
    - 2D joint positions extracted from initial heatmaps serve as anchor points; Deformable Attention enables joint queries to interact with heatmap features from all views in the vicinity of these anchors: $\hat{\mathbf{Q}}^k = \text{DeformAttn}(\mathbf{Q}, \mathbf{T}_k, \mathbf{F}_k)$
    - Updated queries from all views are concatenated, then passed through a fully-connected layer and self-attention to yield multi-view-aware joint queries.
    - An offset regression network generates offset features, which are added to the initial heatmap features to produce refined features.

2. **Initial Heatmap State Propagation**: Directly using view queries for attention is suboptimal due to the absence of context from the current view's initial heatmap predictions. The solution is:

    - Project the initial heatmap $\hat{\mathbf{H}}$ via an MLP to obtain heatmap embeddings $\mathbf{E}$.
    - Project the RGB features $\mathbf{B}$ from the encoder backbone via an MLP to obtain RGB embeddings $\mathbf{G}$.
    - Sum all three and pass through a query projection layer: $\mathbf{Q'} = \mathcal{P}_Q(\mathbf{Q} + \mathbf{E} + \mathbf{G})$

3. **Uncertainty-Aware Masking Mechanism**: Frequent self-occlusion in egocentric images leads to inconsistent reliability in initial heatmaps. Anchor confidence is assessed via heatmap values to construct a binary mask:

    - If the heatmap value $\geq 0.5$, the mask is 1 (reliable); otherwise 0.
    - The mask is applied as an element-wise multiplier to the updated queries: $\hat{\mathbf{Q'}}^k = \hat{\mathbf{Q}}^k \times \mathbf{M}^k$
    - This directs subsequent self-attention to focus more on high-confidence heatmap features.
    - The refinement module is supervised with an MSE loss.

### Loss & Training

Training proceeds in two stages:
1. The 2D joint heatmap estimator and the refinement module are trained separately for 12 epochs each (AdamW, initial lr = 10⁻³).
2. The full architecture (including the 3D module) is jointly fine-tuned for 12 epochs.
- Batch sizes: 64 for 2D heatmap estimation, 32 for 3D pose estimation.
- Learning rate is decayed by ×0.1 at epochs 8 and 10.
- Input resolution: 256×256; heatmap resolution: 64×64.

## Key Experimental Results

### Main Results

3D pose estimation under different camera configurations (MPJPE, mm):

| Setting | Method | Ego4View-Syn | Ego4View-RW |
|---------|--------|-------------|-------------|
| 2 front | EgoPoseFormer | 27.36 | 77.95 |
| 2 front | EgoPoseFormer + Ours | 27.04 | 76.35 |
| 2 front + 2 rear | EgoPoseFormer | 20.20 | 63.38 |
| 2 front + 2 rear | **EgoPoseFormer + Ours** | **19.25** | **56.94** |
| 2 front | EgoTAP | 32.56 | 91.23 |
| 2 front + 2 rear | EgoTAP | 23.88 | 69.78 |
| 2 front + 2 rear | **EgoTAP + Ours** | **22.57** | **62.11** |

On Ego4View-RW, the full proposed method achieves >10% improvement over front-only EgoPoseFormer (63.38 → 56.94 MPJPE).

### Ablation Study

Per-joint evaluation (MPJPE, mm; 2 front + 2 rear; Ego4View-RW):

| Joint | head | neck | arms | forearms | hands | legs | feet | toes | whole body |
|-------|------|------|------|----------|-------|------|------|------|------------|
| EgoPoseFormer | 11.80 | 16.36 | 21.55 | 34.30 | 60.35 | 85.88 | 115.40 | 129.56 | 63.38 |
| + Ours | **11.49** | **15.89** | **21.27** | **30.90** | **48.17** | **79.67** | **103.46** | **116.04** | **56.94** |

The largest improvement is observed at the hands (60.35 → 48.17, −20.2%), with significant gains across upper and lower limbs.

Camera count ablation (Ego4View-RW, EgoPoseFormer + Ours):
- 2 front: 76.35 mm
- 2 front + 1 rear-left: 60.96 mm (−20.2%)
- 2 front + 1 rear-right: 60.17 mm (−21.2%)
- 2 front + 2 rear: **56.94 mm** (−25.4%)

### Key Findings

- Rear cameras are highly valuable for whole-body tracking: adding just one rear camera yields ~20% MPJPE improvement.
- Naively concatenating rear views into existing methods can degrade accuracy, as erroneous 2D detections caused by self-occlusion in rear views propagate into the 3D estimate.
- The uncertainty-aware masking mechanism is critical for handling unreliable detections from rear views.
- Although rear-view hand visibility is only 8–27% (far below front-view rates of 47–66%), multi-view fusion still yields substantial hand estimation improvements.
- A front-to-rear distance of 37 cm represents the optimal balance between visibility and appearance factors.

## Highlights & Insights

- **Pioneering research direction**: This is the first work to challenge the assumption that HMD whole-body tracking requires only front-facing cameras, offering a new perspective for HMD hardware design.
- **Practical motivation**: Apple Vision Pro's failure to support whole-body tracking despite eight front sensors corroborates the fundamental limitations of front-only input.
- **Concise and effective method design**: The heatmap refinement module is a lightweight, plug-and-play component that can be integrated into any existing framework.
- **Significant dataset contribution**: Ego4View-Syn/RW is the first large-scale egocentric dataset featuring rear-mounted cameras.
- **Thorough experimental design**: The dataset includes loose-fitting clothing (long skirts, kimonos, etc.), making it more challenging than existing benchmarks.

## Limitations & Future Work

- The HMD prototype remains bulky (helmet with external cameras), leaving substantial distance from a commercially viable product.
- Rear cameras increase hardware cost and weight, and their feasibility for real-world deployment requires further evaluation.
- Severe distortion in fisheye images hinders the application of traditional stereo methods such as epipolar geometry.
- The potential benefit of temporal information (video-level) for rear-view fusion remains unexplored.
- Pretraining data for 2D detectors does not include rear views, which may introduce a domain gap.
- Fusion of rear cameras with other modalities, such as IMUs, is a promising direction for future work.

## Related Work & Insights

- EgoPoseFormer uses deformable attention to directly update 3D poses; the proposed method instead performs multi-view fusion at the earlier 2D heatmap stage.
- This approach is complementary to body-worn IMU solutions (e.g., Meta Quest): rear cameras require no additional wearable devices.
- The uncertainty-aware design philosophy is generalizable to other multi-view fusion problems.
- The utility of rear cameras extends beyond pose estimation — potential applications include avatar reconstruction, environmental perception, and collision avoidance.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to propose and validate the value of rear-mounted cameras on HMDs, opening an entirely new research direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both synthetic and real datasets, multiple configuration ablations, and per-joint analysis; lacks comparison with IMU-based approaches.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and experiments are thorough; some notation is moderately complex.
- **Value**: ⭐⭐⭐⭐⭐ Provides important insights for HMD hardware design and the egocentric perception community; the open-sourced dataset is of high value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)
- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[ICCV 2025\] BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation](boxdreamer_dreaming_box_corners_for_generalizable_object_pose_estimation.md)
- [\[ICCV 2025\] RePoseD: Efficient Relative Pose Estimation with Known Depth Information](reposed_efficient_relative_pose_estimation_with_known_depth_information.md)

</div>

<!-- RELATED:END -->
