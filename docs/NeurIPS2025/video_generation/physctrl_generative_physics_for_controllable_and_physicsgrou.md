---
title: >-
  [Paper Note] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation
description: >-
  [NeurIPS 2025][Video Generation][Physics-driven video generation] PhysCtrl employs diffusion models to learn the physical dynamics distribution of four material types (elastic, sand, plasticine, and rigid bodies)…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Physics-driven video generation"
  - "diffusion models"
  - "3D point trajectories"
  - "material simulation"
  - "force control"
date: 2026-05-08
content_hash: a1e6bba6d1a76490
---

# PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation

**Conference**: NeurIPS 2025
**arXiv**: [2509.20358](https://arxiv.org/abs/2509.20358)  
**Code**: [Project Page](https://cwchenwang.github.io/physctrl)  
**Area**: Image/Video Generation
**Keywords**: Physics-driven video generation, diffusion models, 3D point trajectories, material simulation, force control

## TL;DR

PhysCtrl employs diffusion models to learn the physical dynamics distribution of four material types (elastic, sand, plasticine, and rigid bodies), representing dynamics as 3D point trajectories. A diffusion model incorporating spatiotemporal attention and physics constraints is trained on 550K synthetic animations; the generated trajectories drive a pretrained video model to achieve high-fidelity physics video generation controllable by force and material parameters.

## Background & Motivation

**Background**: Modern video generation models can produce photorealistic video but lack physical plausibility and 3D controllability.

**Limitations of Prior Work**: Traditional physics simulators (e.g., MPM) are computationally expensive, sensitive to hyperparameters, and numerically unstable; directly coupling simulators with video models requires manual parameter tuning and may necessitate switching between different simulators.

**Key Challenge**: How can physical plausibility be maintained while avoiding the limitations of traditional simulators?

**Goal**: Embed physical priors into a diffusion model to support fast forward/inverse inference, using physical parameters and external forces as control signals.

**Key Insight**: Address two fundamental questions — what representation is suitable for controlling video models? → 3D point trajectories; how to embed multi-material physical priors? → spatiotemporal attention diffusion model + physics constraints.

**Core Idea**: Use a diffusion model to learn the latent distribution of physical dynamics, with 3D point trajectories as a bridge between the physical world and video generation.

## Method

### Overall Architecture

Single image → SAM segmentation + SV3D multi-view + LGM 3D point cloud reconstruction → conditional diffusion model generates 3D point trajectories → projected 2D as tracking video → DaS video model generates the final video.

### Key Designs

1. **Spatiotemporal Attention Diffusion Model**:
    - Function: Learns the physical dynamics distribution $p(\mathcal{P}|c)$ for four material types.
    - Mechanism: 2048 points × 24 frames of trajectories; conditions include initial point cloud, force, application point, Young's modulus, Poisson's ratio, ground height, and material type. Spatial-Temporal Attention Block: spatial attention (intra-frame self-attention among points, physical condition tokens injected via AdaLN) → temporal attention (cross-frame self-attention for each point).
    - Design Motivation: Simulates particle dynamics — integrating neighboring particle information before temporal propagation, reflecting the P2G/G2P cycle of MPM.

2. **Physics-Constrained Training Loss**:
    - Function: Incorporates the deformation gradient update formula from MPM as an explicit physics constraint.
    - Mechanism: $\mathcal{L}_{phys}$ enforces $\mathbf{F}_p^{f+1} \approx g(\hat{\mathbf{x}}_p^f)\mathbf{F}_p^f$ (deformation gradient consistency), complemented by $\mathcal{L}_{floor}$ to prevent penetration.
    - Design Motivation: Physics loss serves as regularization to ensure trajectory physical plausibility.

3. **Large-Scale Synthetic Dataset**:
    - Function: 550K animations covering four material types (150K elastic + 100K each of sand, plasticine, rigid body, and gravity).
    - Mechanism: High-quality 3D objects from ObjaverseXL + MPM/rigid body simulators; 2048 points × 24 frames.
    - Design Motivation: Diverse data is foundational for learning the physical distribution.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{diff} + \lambda_{vel}\mathcal{L}_{vel} + \lambda_{phys}\mathcal{L}_{phys} + \lambda_{floor}\mathcal{L}_{floor}$. Base: 6 layers, 256 dimensions; Large: 12 layers, 512 dimensions. AdamW lr=1e-4. DDIM 25 steps ≈ 1–3 seconds; 4 steps ≈ 0.13–0.48 seconds.

## Key Experimental Results

### Main Results

| Method | SA↑ | PC↑ | VQ↑ |
|--------|-----|-----|-----|
| DragAnything | 2.9 | 2.8 | 2.8 |
| ObjCtrl-2.5D | 1.5 | 1.3 | 1.4 |
| Wan2.1 | 3.8 | 3.7 | 3.6 |
| CogVideoX | 3.2 | 3.2 | 3.1 |
| **PhysCtrl** | **4.5** | **4.5** | **4.3** |

| Method | vIoU↑ | CD↓ | Corr↓ |
|--------|-------|-----|-------|
| Motion2VecSets | 24.92% | 0.2160 | 0.1064 |
| MDM | 53.78% | 0.0159 | 0.0240 |
| **Ours** | **77.59%** | **0.0028** | **0.0015** |

### Ablation Study

| Configuration | vIoU↑ | CD↓ | Note |
|---------------|-------|-----|------|
| w/o spatial attention | 33.76% | 0.2348 | Spatial interaction is critical |
| w/o temporal attention | 53.63% | 0.0480 | Temporal consistency is essential |
| w/o physics loss | 76.30% | 0.0030 | Physics constraints further improve results |
| Full model | **77.59%** | **0.0028** | — |

### Key Findings
- User study: Physical plausibility preference rate of 81%, far exceeding all baselines.
- Removing spatial attention causes a sharp drop in vIoU (77.59% → 33.76%).
- Poisson's ratio $\nu$ has negligible influence on generated trajectories (consistent with PhysDreamer).
- High-quality trajectories can be obtained with as few as 4 diffusion steps.

## Highlights & Insights
- Parameterizes physics simulation as a conditional generation problem, avoiding the limitations of traditional simulators.
- 3D point trajectories serve as an intermediate representation — both flexible and general, while directly controllable for video models.
- The spatiotemporal attention design elegantly mirrors the computational structure of physics simulation.
- Supports inverse problems; Young's modulus estimation requires only ~2 minutes.

## Limitations & Future Work
- Primarily handles single objects; multi-object interaction is only preliminarily explored.
- Limited to four material types; fluids are not included.
- Video model priors may conflict with physics-based trajectories.
- Handling of thin structures remains insufficient.

## Related Work & Insights
- **vs PhysGaussian**: Scene-specific and requires high-quality 3D reconstruction; PhysCtrl learns a general prior.
- **vs PhysGen/PhysMotion**: Relies on simulators to generate dynamics; PhysCtrl embeds the prior directly into a diffusion model.

## Rating
- Novelty: ⭐⭐⭐⭐ First to learn multi-material physical dynamics distributions via a diffusion model.
- Experimental Thoroughness: ⭐⭐⭐⭐ GPT-4o evaluation + user study + complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and well-defined.
- Value: ⭐⭐⭐⭐ An important step toward physically controllable video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)
- [\[CVPR 2026\] PhysVid: Physics Aware Local Conditioning for Generative Video Models](../../CVPR2026/video_generation/physvid_physics_aware_local_conditioning_for_generative_video_models.md)
- [\[ICCV 2025\] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](../../ICCV2025/video_generation/fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)
- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](../../CVPR2026/video_generation/phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](../../ICCV2025/video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)

</div>

<!-- RELATED:END -->
