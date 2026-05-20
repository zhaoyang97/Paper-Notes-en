---
title: >-
  [Paper Note] HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles
description: >-
  [CVPR2026][Autonomous Driving][Driving Scene Editing] HorizonForge proposes a unified framework that reconstructs driving scenes as editable Gaussian Splats combined with Mesh representations…
tags:
  - "CVPR2026"
  - "Autonomous Driving"
  - "Driving Scene Editing"
  - "3D Gaussian Splatting"
  - "Video Diffusion Models"
  - "Trajectory Control"
  - "Mesh Insertion"
  - "Multi-Agent Simulation"
date: 2026-05-08
content_hash: c65cf7dbcfbcdc09
---

# HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles

**Conference**: CVPR2026
**arXiv**: [2602.21333](https://arxiv.org/abs/2602.21333)  
**Code**: [Project Page](https://horizonforge.github.io/)  
**Area**: Autonomous Driving / Scene Generation & Editing
**Keywords**: Driving Scene Editing, 3D Gaussian Splatting, Video Diffusion Models, Trajectory Control, Mesh Insertion, Multi-Agent Simulation

## TL;DR

HorizonForge proposes a unified framework that reconstructs driving scenes as editable Gaussian Splats combined with Mesh representations, enabling fine-grained 3D manipulation via trajectory control and language-driven vehicle insertion. A video diffusion model then renders spatiotemporally consistent, high-quality driving videos. The method achieves a user preference rate of 91.02%, decisively outperforming all baselines.

## Background & Motivation

**Scarcity of long-tail scenes**: Autonomous driving systems must be evaluated in safety-critical scenarios such as aggressive lane changes and hard braking, which are extremely rare and expensive to collect in practice, necessitating simulation-based generation.

**Poor generalization of reconstruction methods**: Reconstruction approaches based on 3DGS/NeRF (e.g., OmniRe) offer high geometric fidelity but tend to produce artifacts under unseen viewpoints and generalize poorly to novel trajectories.

**Lack of physical constraints in generative methods**: Pure generative methods (diffusion models) can hallucinate new content but lack explicit 3D structural priors, leading to inconsistent scene structure and inadequate control over traffic participants.

**Complexity of hybrid methods**: Existing hybrid approaches (e.g., Difix3D, StreetCrafter) rely on complex architectures or require expensive per-trajectory optimization for each new trajectory, limiting scalability.

**Absence of a unified evaluation benchmark**: Existing evaluation sets (e.g., StreetCrafter's ego lane change) cover only a single operation type and lack a comprehensive benchmark spanning both ego- and agent-level editing tasks.

**Lack of systematic study on 3D representations**: The critical design question of which 3D representation is most suitable as a conditioning signal for diffusion models has not been systematically investigated.

## Method

### Overall Architecture

HorizonForge consists of two stages: **3D Assets Harvesting** and **Video Rendering**.

- **Stage 1**: The input driving video is reconstructed into editable 3D assets — Gaussian Splats (obtained via OmniRe) and 3D Meshes (generated via Hunyuan3D). All vehicles are repositioned in 3D space according to the target trajectories $\mathcal{T}=\{\tau_i\}_{i=1}^N$.
- **Stage 2**: The edited 3D scene is rasterized into a 2D frame sequence, which is then processed by a fine-tuned video diffusion model (based on the CogVideoX backbone) to repair artifacts, inpaint missing regions, and produce high-fidelity, temporally consistent driving videos.

### Key Designs

**1. Gaussian Splats Conditioning**

- OmniRe is used to reconstruct a 3DGS scene representation from the input video, encoding continuous density and color information to provide the diffusion model with rich intermediate-level priors.
- Systematic comparison reveals: Gaussian Splats >> colored point clouds >> 3D bounding boxes — richer representations yield higher generation quality.

**2. Mesh Harvesting and Insertion Pipeline**

- **Training phase**: GT 3D bounding boxes and LiDAR points from the Waymo dataset are used to select the best observation frame; SAM performs segmentation → Pix2Gestalt completes occlusions → Hunyuan3D generates the mesh → GPT infers orientation → depth and IoU jointly optimize scale alignment.
- **Inference phase**: The user provides a text description → GPT generates a reference image → Hunyuan3D generates the mesh → a VLM infers rotation and scale → the mesh is inserted into the scene, enabling **arbitrary text-driven vehicle insertion**.

**3. Cyclic Reconstruction Data Pair Construction**

- For Gaussian Splats: the original trajectory is perturbed as $\tilde{\mathcal{T}} = \mathcal{T} + \Delta\mathcal{T}$; perturbed frames are rendered → a new GS scene is reconstructed → frames with artifacts are rendered under the original trajectory and paired with GT for training.
- For Meshes: GS assets are randomly replaced with harvested meshes at 50% probability to construct mixed mesh-GS data pairs, narrowing the lighting/texture gap between meshes and real images.

### Loss & Training

Standard diffusion denoising loss:

$$\mathcal{L}_{\text{vdm}} = \mathbb{E}_{t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, v_c)\|_2^2\right]$$

where $v_c$ denotes the conditioning video frames rendered from Gaussian Splats/Meshes. The model is fine-tuned for 60k steps on the CogVideoX backbone, initialized from TrajectoryCrafter pretrained weights.

## Key Experimental Results

### Main Results: HorizonSuite Benchmark Comparison

Comparisons against StreetCrafter, Difix3D, and OmniRe are conducted on the authors' proposed **HorizonSuite** benchmark (109 high-quality editing trajectories covering ego/agent-level speed changes, lane changes, turning, insertion, and deletion):

| Method | Overall FID↓ | Overall FVD↓ | Best Metrics |
|--------|-------------|-------------|--------------|
| StreetCrafter | 91.16 | 1245.96 | 0 |
| Difix3D | 80.84 | 991.23 | 0 |
| OmniRe | 44.37 | 546.00 | Few |
| **HorizonForge** | **33.19** | **536.49** | **Majority** |

- HorizonForge achieves state-of-the-art performance on nearly all tasks and metrics, with an Overall FID improvement of **25.19%** over the second-best method OmniRe.
- On the vehicle insertion task, FID decreases from 182.29 to 117.46 (↓35.5%) and OSR improves from 4.23 to 5.86.

### Ablation Study

| Conditioning Representation | Overall FID↓ | Overall FVD↓ |
|----------------------------|-------------|-------------|
| 3D BBox + Mesh | 81.74 | 1521.07 |
| Colored Point Cloud + Mesh | 54.14 | 813.67 |
| Image DM (same conditions) | 75.83 | 837.99 |
| **Gaussian + Mesh (Ours)** | **33.19** | **536.49** |

**Key Findings**:
- **3D representation matters**: The rich appearance prior encoded by Gaussian Splats substantially outperforms sparse representations (bounding boxes, point clouds); 3D BBox performs worst across all metrics.
- **Temporal prior matters**: Video diffusion models significantly outperform image diffusion models; the latter, while producing acceptable fidelity, suffers from severe inter-frame flickering.

### User Study

| Method | Wins / Total | Win Rate |
|--------|-------------|----------|
| StreetCrafter | 2/501 | 0.40% |
| Difix3D | 4/501 | 0.80% |
| OmniRe | 39/501 | 7.78% |
| **HorizonForge** | **456/501** | **91.02%** |

The user preference rate decisively surpasses all baselines, validating the overall superiority of the generated videos in terms of realism, stability, and trajectory consistency.

## Highlights & Insights

- **Simple and unified framework**: No complex conditioning pipelines or per-trajectory optimization are required; once 3D scene reconstruction is complete, variants under arbitrary trajectories can be generated in a feed-forward manner.
- **Arbitrary vehicle insertion**: A purely text-based description suffices to generate and insert a 3D vehicle mesh, establishing an end-to-end pipeline from language → 3D → video.
- **Systematic design insights**: This work is the first to systematically compare the effects of 3D representations and temporal modeling under a unified benchmark, providing clear design guidelines.
- **HorizonSuite benchmark**: Covers 5 editing task categories at both ego and agent levels, evaluated with 5 metrics (including VIMS, BAS, and OSR), addressing the gap in evaluation of multi-agent controllable driving scene editing.
- **Overwhelming user preference**: A 91.02% win rate and a 25.19% FID improvement collectively demonstrate the effectiveness of the proposed method.

## Limitations & Future Work

- **Dependency on high-quality 3DGS reconstruction**: The quality of OmniRe reconstruction directly affects downstream generation; degradation may occur in regions with sparse LiDAR coverage or severe occlusion.
- **Lighting/texture gap in mesh generation**: Although training-time lighting randomization partially alleviates the issue, meshes generated by Hunyuan3D may still exhibit noticeable texture discrepancies relative to real scenes.
- **Validation limited to the Waymo dataset**: Generalization to other datasets such as nuScenes has not been verified.
- **Relatively weaker performance on direction change tasks**: The ego Direction Change task yields FID=144.83, substantially higher than other editing types; large-angle turning remains challenging.
- **Computational overhead**: The pipeline requires OmniRe reconstruction, Hunyuan3D mesh generation, and 60k steps of diffusion model fine-tuning, resulting in a heavy overall workflow.

## Related Work & Insights

- **Reconstruction-based approaches**: OmniRe, 3DGS, and similar methods provide high-fidelity reconstruction but offer limited controllability.
- **Generation-based approaches**: Stable Video Diffusion, VideoCrafter2, and similar methods lack explicit physical constraints, producing unstable outputs.
- **Hybrid approaches**: StreetCrafter (colored point clouds + image diffusion) and Difix3D (reconstruction denoising + image diffusion) are constrained by sparse conditioning or the absence of temporal modeling.
- **Core distinction of HorizonForge**: By pairing the richest 3D representation (GS + Mesh) with the most effective temporal model (video diffusion), the framework achieves state-of-the-art results with a minimal architecture.

## Rating

- Novelty: ⭐⭐⭐⭐ — The framework design is not entirely novel (the reconstruction + diffusion refinement paradigm has prior precedents), but the mesh harvesting/insertion pipeline and systematic representation comparison make original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — The self-constructed HorizonSuite benchmark offers comprehensive coverage; ablations are detailed and the user study is large-scale.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear and figures are informative, though some notation definitions are introduced hastily.
- Value: ⭐⭐⭐⭐⭐ — Provides clear design guidelines for autonomous driving simulation scene editing; both the benchmark and the framework offer substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](terraseg_self-supervised_ground_segmentation_for_any_lidar.md)
- [\[ICLR 2026\] SEAL: Segment Any Events with Language](../../ICLR2026/autonomous_driving/segment_any_events_with_language.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](../../NeurIPS2025/autonomous_driving/towards_predicting_any_human_trajectory_in_context.md)
- [\[NeurIPS 2025\] OpenBox: Annotate Any Bounding Boxes in 3D](../../NeurIPS2025/autonomous_driving/openbox_annotate_any_bounding_boxes_in_3d.md)
- [\[NeurIPS 2025\] LabelAny3D: Label Any Object 3D in the Wild](../../NeurIPS2025/autonomous_driving/labelany3d_label_any_object_3d_in_the_wild.md)

</div>

<!-- RELATED:END -->
