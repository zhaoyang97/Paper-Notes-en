---
title: >-
  [Paper Note] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior
description: >-
  [ICCV 2025][3D Vision][Autonomous driving simulation] DriveX is a driving view synthesis framework that progressively distills generative priors from a video diffusion model into a 3DGS representation. It designs an inpainting-based video restoration task to generate pseudo-labels for novel trajectories and iteratively refines the 3D reconstruction, enabling high-quality real-time rendering on free-form trajectories.
tags:
  - ICCV 2025
  - 3D Vision
  - Autonomous driving simulation
  - free-trajectory view synthesis
  - 3D Gaussian splatting
  - video diffusion
  - generative prior distillation
  - inpainting restoration
date: 2026-05-08
content_hash: 0949e81ab8994c30
---

# DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior

**Conference**: ICCV 2025
**arXiv**: [2412.01717](https://arxiv.org/abs/2412.01717)
**Code**: [https://fudan-zvg.github.io/DriveX](https://fudan-zvg.github.io/DriveX)
**Area**: 3D Vision
**Keywords**: Autonomous driving simulation, free-trajectory view synthesis, 3D Gaussian splatting, video diffusion, generative prior distillation, inpainting restoration

## TL;DR
DriveX is a driving view synthesis framework that progressively distills generative priors from a video diffusion model into a 3DGS representation. It designs an inpainting-based video restoration task to generate pseudo-labels for novel trajectories and iteratively refines the 3D reconstruction, enabling high-quality real-time rendering on free-form trajectories.

## Background & Motivation

### Core Problem
Building a virtual driving world that can interact with driving policies is critical for developing robust autonomous driving systems. Key requirements include efficient training data augmentation, synthesis of safety-critical long-tail scenarios, and closed-loop evaluation of end-to-end driving systems. **Core bottleneck**: existing reconstruction methods perform well on recorded trajectories but degrade severely on free novel trajectories (e.g., lane changes, lateral translations).

### Unique Challenges of Driving Scenes
Compared to general novel view synthesis, driving scenes pose:
- **Extremely sparse viewpoints**: single-trajectory forward-facing videos with minimal inter-frame overlap
- **Large untextured regions**: roads, sky, etc.
- **Limited trajectory diversity**: datasets contain almost exclusively straight-line trajectories, lacking views from deviated paths

### Limitations of Prior Work

**Pure reconstruction methods** (StreetGaussian, PVG, etc.): perform well on recorded trajectories but have poor extrapolation ability, producing artifacts when deviating from the recording path.

**Pure generative methods** (video world models): offer good diversity but lack an underlying 3D representation, unable to guarantee geometric/texture consistency across multi-trajectory videos.

**Challenges of hybrid reconstruction + generation approaches**:
- SGD/VEGS: learning precise pose control is difficult given limited trajectory diversity in driving data.
- DriveDreamer4D: using only a single reference frame to generate an entire novel-trajectory video leads to insufficient constraints, causing hallucinations and inconsistencies.
- FreeSim/Difix3D+: require carefully curated training data to simulate degradation patterns of off-trajectory rendering.

### Core Insight

**Reformulating the problem as an inpainting task**: rather than requiring the generative model to identify degraded regions, an unreliability mask explicitly tells the model which regions need restoration. This yields:
1. No need to simulate specific degradation patterns during training.
2. The generative model can focus on semantic in-filling, at which it excels.
3. Rendered results and restored results can mutually reinforce each other.

## Method

### Overall Architecture (Interleaved Reconstruction–Generation)
During 3DGS optimization, the framework alternates between:
1. Supervising 3DGS optimization with ground-truth recorded-trajectory frames (standard pipeline).
2. Using the video diffusion model to restore novel-trajectory renderings → generating pseudo-labels → using them as additional supervision to optimize 3DGS.
3. Updating pseudo-labels every $K$ steps; the two components co-evolve.

### Key Designs

#### 1. Novel-Trajectory Pseudo-Label Generation (Video Restoration)
Given a novel trajectory $\mathcal{T}'$, a video $\mathcal{V}_t'$ (containing artifacts) is rendered from the current 3DGS $\mathcal{G}_t$ and restored by the video diffusion model:
$$\mathcal{V}'_{t,refine} = \mathcal{D}(\mathcal{V}'_t, \mathcal{M})$$
The restored video serves as pseudo-labels to optimize 3DGS via a photometric loss:
$$\mathcal{L}' = \mathcal{L}_{img}(\mathcal{V}'_t, \mathcal{V}'_{t,refine})$$

#### 2. Unreliability Mask Construction (Core Innovation)
Artifact regions in rendered images are detected via geometric consistency:
- The nearest real recorded image $I_{rec}$ is warped to the novel viewpoint using the rendered depth $D_{ren}$.
- Warping formula: $(x,y,d) = \psi(p|P)$ projects to $I_{rec}$ and samples colors to obtain a pseudo image $\hat{I}_{ren}$.
- SSIM is used to compare the rendered image and warped image:
$$\mathcal{M} = \mathbb{1}(\text{SSIM}(I_{ren}, \hat{I}_{ren}) < \tau)$$
with $\tau = 0.65$. Since warping involves both rendered depth and rendered appearance, inconsistencies indicate geometrically or visually unreliable regions.

#### 3. Progressive Iterative Refinement
Rather than generating pseudo-labels once, they are updated every $K=3000$ steps during optimization:
- Standard reconstruction training runs for $T_0=50000$ steps first.
- Generative priors are introduced in the subsequent 30,000 steps.
- As 3DGS quality improves, the conditioning provided to the diffusion model also improves → better restoration results → positive feedback loop.

#### 4. Novel Trajectory Sampling Strategy
A panning camera trajectory is adopted, starting from the recorded frontal viewpoint $P'_0$ and gradually translating laterally:
$$P'_i = [R_0 | \frac{i}{F} s \mathbf{v} + T_0]$$
where $s$ controls the maximum translation distance and $\mathbf{v}$ is the translation direction. Key design choices:
- The initial frame coincides with the recorded view, providing an accurate reference for the generative model.
- $s$ is progressively increased to avoid excessive deviation during early, low-quality optimization.
- Multi-camera setups are exploited (side cameras assist with mask regions not visible to the front camera).

#### 5. Inpainting-Based Generative Model Training
The training strategy elegantly sidesteps the lack of diverse trajectories in driving data:
- An **edge-aware strategy** generates masks on training images: Sobel edge detection → edge-probability sampling → $3\times3$ patch occlusion.
- Both front-camera and side-camera videos are used for training (the viewpoint of a side camera on a forward-moving trajectory is analogous to the lateral shift of a front camera).
- Initialized from [39] and fine-tuned on WOD for 15,000 steps.

#### 6. LiDAR Assistance
- LiDAR points from adjacent $\pm2$ frames are projected onto the novel viewpoint as colored LiDAR projection conditions.
- LiDAR points of dynamic objects are aligned according to tracked poses.
- LiDAR depth serves as sparse depth supervision for novel viewpoints.

### Total Loss
$$\mathcal{L}_{total} = \mathcal{L}_{img}(\mathcal{V}_t, \mathcal{V}_{gt}) + \mathcal{L}_{img}(\mathcal{V}'_t, \mathcal{V}'_{k,refine})$$

## Key Experimental Results

### Main Results: Novel Trajectory Synthesis on Waymo Dataset

| Method | ±0m PSNR↑ | ±1m IoU↑ | ±2m IoU↑ | ±3m IoU↑ | ±3m FID↓ | FPS |
|------|------|------|------|------|------|------|
| EmerNeRF | 29.60 | 0.1147 | 0.0881 | 0.0735 | 122.83 | 0.12 |
| PVG | 29.98 | 0.1082 | 0.0291 | 0.0170 | 123.53 | 48 |
| StreetGaussian | 29.76 | 0.2671 | 0.2122 | 0.1720 | 93.04 | 34 |
| **DriveX** | 29.74 | **0.2880** | **0.2710** | **0.2699** | **79.78** | 34 |

Key findings:
- At ±3m lateral offset, IoU improves by 56.9% (vs. StreetGaussian) and FID decreases by 14.3%.
- At inference time, only 3DGS rendering is used without running the diffusion model, maintaining real-time speed (34 FPS).
- Quality on recorded trajectories remains on par, confirming that generative priors do not interfere with the original reconstruction.

### Comparison with Generative Methods

| Method | Lane change NTA-IoU↑ | ±3m NTA-IoU↑ | ±3m NTL-IoU↑ |
|------|------|------|------|
| DriveDreamer4D | 0.495 | 0.340 | 51.32 |
| ReconDreamer | 0.554 | 0.539 | 54.58 |
| **DriveX** | **0.620** | **0.567** | **58.29** |

DriveX outperforms DriveDreamer4D's single-reference-frame generation and ReconDreamer.

### Ablation Study

**Effect of the Unreliability Mask**:

| Configuration | IoU↑ | FID↓ |
|------|------|------|
| mask all ($\tau=1.0$) | 0.2153 | 102.69 |
| w/o mask ($\tau=-1.0$) | 0.2193 | 76.26 |
| **w/ mask ($\tau=0.65$)** | **0.2263** | **74.34** |

Masking everything → diffusion model generates from scratch → inconsistency; no mask → model must identify degradation on its own → suboptimal performance.

**Effect of Iterative Refinement Update Frequency**:

| Buffer Interval K | IoU↑ | FID↓ | Time Multiplier |
|------|------|------|------|
| 500 | 0.2271 | 75.33 | 4.8× |
| 3000 | 0.2263 | 74.34 | 1.6× |
| 6000 | 0.2228 | 75.98 | 1.3× |

$K=3000$ achieves a favorable balance between quality and efficiency.

## Highlights & Insights
1. **Elegant inpainting formulation**: decouples "identifying degradation" from "restoring it" — the mask handles the former, the diffusion model the latter, avoiding the need to simulate degradation patterns.
2. **Progressive co-evolution**: a positive feedback loop in which improved 3DGS yields better conditioning → better generation → further improved 3DGS.
3. **Training–inference consistency**: inference relies solely on 3DGS rendering without running the diffusion model, preserving real-time speed.
4. **Geometry-driven masks are more general than semantic masks**: depth warping detects inconsistencies without relying on scene-specific semantics.

## Limitations & Future Work
1. Running the video diffusion model multiple times during training increases total training time to approximately 1.6× that of the base method.
2. At extreme offsets (>6m), the diffusion model may fall outside its generalization range.
3. The method relies on LiDAR data for sparse depth anchors and is not applicable to purely vision-based settings.
4. Novel trajectories are limited to lateral translations; more complex patterns (e.g., rotation, longitudinal offset) have not been validated.

## Related Work & Insights
- **Reconstructive driving simulation**: StreetGaussian/PVG (3DGS-based) → Ours (+generative priors).
- **Generative driving simulation**: GAIA-1/GenAD (video world models) → lack 3D consistency.
- **Sparse-view reconstruction + diffusion priors**: ZeroNVS/ReconFusion → Ours (customized inpainting for driving scenes).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The inpainting formulation combined with progressive distillation is a highly original framework design.
- Technical Depth: ⭐⭐⭐⭐ — Individual components are well-designed, though the diffusion model itself is a fine-tuned existing model.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Thorough validation on Waymo; evaluation on other datasets such as nuScenes is absent.
- Practical Value: ⭐⭐⭐⭐⭐ — Directly improves driving simulation quality with no additional inference overhead.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GAS: Generative Avatar Synthesis from a Single Image](gas_generative_avatar_synthesis_from_a_single_image.md)
- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)

<!-- RELATED:END -->
