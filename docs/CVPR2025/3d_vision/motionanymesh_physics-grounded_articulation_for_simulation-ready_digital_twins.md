---
title: >-
  [Paper Note] MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins
description: >-
  [CVPR 2025][3D Vision][Articulated Object Modeling] Proposes MotionAnyMesh, a zero-shot framework that eliminates hallucinations by guiding VLM inference with SP4D kinematic priors and guarantees collision-free execution through physics-constrained trajectory optimization. It automatically transforms static 3D meshes into simulation-ready articulated digital twins, achieving a physics execution success rate of 87%, nearly double that of the best existing methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Articulated Object Modeling"
  - "Digital Twins"
  - "URDF Generation"
  - "VLM Inference"
  - "Physics-Constrained Optimization"
date: 2026-05-08
content_hash: cec33a23755578ef
---

# MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins

**Conference**: CVPR 2025  
**arXiv**: [2603.12936](https://arxiv.org/abs/2603.12936)  
**Code**: Pending  
**Area**: 3D Vision / Robotics  
**Keywords**: Articulated Object Modeling, Digital Twins, URDF Generation, VLM Inference, Physics-Constrained Optimization

## TL;DR
Proposes MotionAnyMesh, a zero-shot framework that eliminates hallucinations by guiding VLM inference with SP4D kinematic priors and guarantees collision-free execution through physics-constrained trajectory optimization. It automatically transforms static 3D meshes into simulation-ready articulated digital twins, achieving a physics execution success rate of 87%, nearly double that of the best existing methods.

## Background & Motivation
**Background**: Embodied AI and robotics simulations require a large number of articulated objects (doors, drawers, etc.), but the vast majority of assets in massive 3D asset libraries (such as Objaverse) are static meshes lacking joint structures and part definitions. Manually modeling URDFs is extremely costly.

**Limitations of Prior Work**: Existing pipelines suffer from two fundamental flaws—(a) "Kinematic hallucinations" driven by pure semantic reasoning of VLMs: when facing complex, unnamed mechanical components, VLMs tend to incorrectly merge or over-segment; (b) Joint parameter estimation lacks physical constraints, where minor deviations accumulate over long-range motions in physics engines like SAPIEN, causing severe penetration or freezing.

**Key Challenge**: There is a huge gap between vision perception pipelines and truly physics-executable digital twins—seemingly reasonable parameters do not equate to physical feasibility.

**Goal**: (1) Eliminate kinematic hallucinations of VLMs to achieve accurate part segmentation; (2) Guarantee that joint parameters are collision-free and executable in physics simulations.

**Key Insight**: Decouple boundary extraction from semantic reasoning—first extract pure geometric primitives in 3D native space, then anchor VLM inference with kinematic priors; use SDF-constrained trajectory optimization to ensure no penetration.

**Core Idea**: SP4D kinematic priors to eliminate VLM hallucinations + physics-constrained trajectory optimization to eliminate penetration.

## Method

### Overall Architecture
The input is a static 3D mesh, and the output is a simulation URDF asset with joint parameters and part segmentation. It consists of three stages: (1) Kinematic-aware part segmentation: P3-SAM extracts 3D primitives $\rightarrow$ SP4D+VLM clusters them into functional parts; (2) Joint estimation and optimization: type-aware initialization $\rightarrow$ physics-constrained trajectory optimization; (3) Simulation asset finalization: motion range estimation + texture preservation $\rightarrow$ URDF output.

### Key Designs

1. **Kinematic-Aware Part Segmentation**:

    - **Function**: Segment the static mesh into kinematically consistent functional parts.
    - **Mechanism**: First, over-segment the mesh into pure geometric primitives $\mathcal{P} = \{p_1,...,p_m\}$ using P3-SAM in the 3D native space to maintain sharp physical boundaries. Then, render the mesh from multiple views, generating kinematic segmentation masks via SP4D as physical priors. Input the primitive map (with unique colors/digital IDs) and SP4D masks into the VLM (GPT-4o), letting the VLM cluster fine-grained primitives into kinematically consistent parts as if reading an "assembly manual".
    - **Design Motivation**: Pure VLM semantic reasoning suffers from severe hallucinations when facing unnamed mechanical parts (mIoU drops from 0.86 to 0.68); 2D-to-3D projection segmentation destroys geometric continuity; SP4D provides explicit kinematic anchors, allowing the VLM to reason based on physics rather than pure semantics.

2. **Type-Aware Kinematic Initialization**:

    - **Function**: Infer initial joint parameters from the geometric features of contact interfaces based on joint types.
    - **Mechanism**: First extract the contact point cloud $S_{contact}$ (vertices within < 0.01m from the parent part). For rotational joints, categorize them into two types: Spin joints (wheels/knobs) use the minimum eigenvalue direction of PCA as the axis + RANSAC 2D circle fitting to locate the rotation center; Hinge joints (hinges) use the maximum eigenvalue direction of PCA as the axis + centroid as the pivot. For translational joints: apply global PCA to obtain 3 candidate directions, and select the optimal sliding direction using a normalized double-penalty function (collision penalty + derailment penalty).
    - **Design Motivation**: Different joint types possess different geometric characteristics—Spin contact surfaces are disk-shaped (normal = axis), and Hinge contact surfaces extend along the axis. Utilizing geometric priors instead of learning is more robust.

3. **Physics-Constrained Trajectory Optimization**:

    - **Function**: Refining the initial joint parameters until they are completely collision-free in simulation.
    - **Mechanism**: Minimize a unified trajectory deviation loss $\mathcal{L}_{opt}(\mathbf{v},\mathbf{q}) = \sum_{\phi} \sum_{\mathbf{x} \in S_{contact}} \|\mathcal{D}_{SDF}(\mathcal{T}(\mathbf{x};\mathbf{v},\mathbf{q},\phi), \mathcal{M}_{static})\|_2^2$ over discrete virtual motion states $\Phi$. This uses SDF to measure the distance from contact points to static surfaces during motion, solved via Levenberg-Marquardt optimization.
    - **Design Motivation**: Minor initialization deviations accumulate into penetrations during long-range motions; SDF penalty ensures the entire trajectory remains collision-free.

### Motion Range Estimation
Rotational joints: incrementally rotate in both directions until collision determines $[\theta_{min}, \theta_{max}]$, with step sizes from coarse to fine (first 5° then 1°), where collisions are detected when the SDF value falls below a threshold. Translational joints: push inward along the sliding axis to collision point $d_{min}$, pull outward until the contact area drops to zero (simulating derailment) to obtain $d_{max}$. The accuracy of the motion range directly affects the usable space for policy learning in simulation, where overestimation leads to penetration and underestimation limits the manipulation range.

Specifically, the collision detection threshold $\tau_{SDF}$ is set to 0.005m. An initial step size of 5° for rotational joints can cover large angular ranges (such as a 120° cabinet door), and refining it to 1° ensures boundary accuracy within 2°. Derailment detection for translational joints is achieved by monitoring the coverage area of the contact point set: derailment is determined when the area falls below 10% of the initial area. The overall motion range estimation takes <2 seconds per joint, which does not constitute a bottleneck in the pipeline.

### Implementation Details
- P3-SAM is used for 3D native geometric primitive extraction, SP4D generates multi-view kinematic masks, and GPT-4o serves as the VLM.
- Trajectory optimization uses Trimesh to calculate SDF, and SciPy's Nelder-Mead algorithm is employed to solve the non-linear objective.
- Hunyuan3D integration is used for optional generative re-texturing.
- Contact point distance threshold $\tau=0.01$m, RANSAC inlier threshold $\delta=0.005$m, collision penetration threshold $\epsilon_c=0.005$m.
- In the double-penalty function for translational joints, $\omega=20$ is used to balance the magnitude difference between the collision penalty and the derailment penalty.
- All experiments were executed on three NVIDIA RTX 4090 GPUs.

## Key Experimental Results

### Main Results

| Method | mIoU↑ | Count Accuracy↑ | Type Error↓ | Axis Error↓ | Pivot Error↓ | Physics Executability↑ |
|------|-------|---------|---------|--------|---------|----------|
| PARIS | 0.17 | 0.23 | 0.67 | 1.56 | 1.14 | 11% |
| Articulate-Anything | 0.47 | 0.61 | 0.21 | 0.86 | 0.64 | 46% |
| Articulate-AnyMesh | 0.59 | 0.74 | 0.35 | 0.64 | 0.44 | 35% |
| **MotionAnyMesh** | **0.86** | **0.92** | **0.08** | **0.12** | **0.10** | **87%** |

### Ablation Study

| Configuration | mIoU / Axis Error / Executability | Description |
|------|-------------|------|
| w/o SP4D | 0.68 / — / — | Severe VLM kinematic hallucination |
| w/ SP4D (Full) | 0.86 / — / — | Eliminates hallucination +0.18 mIoU |
| w/o Trajectory Optimization | — / 0.23 / 65% | Fair initialization but unstable simulation |
| w/ Trajectory Optimization | — / 0.12 / 87% | Executability +22% after optimization |

### Key Findings
- Physics executability is the most persuasive metric—MotionAnyMesh achieves 87% vs. the best baseline's 46%, nearly doubling it. The evaluation standard is strict: executing 100 motion steps in the SAPIEN engine, where any step with penetration is deemed a failure.
- SP4D priors are crucial for eliminating VLM hallucinations: pure semantic VLMs experience a 0.18 mIoU drop on complex machinery. A typical hallucination case: the VLM mistakenly identifies the spray arm of a dishwasher as an independent jointed component; SP4D's motion prior corrects such errors.
- The two-stage strategy of initialization $\rightarrow$ optimization is necessary: even if the static metrics of geometric initialization look acceptable (axis error 0.23), the dynamic executability is only 65%.
- Successfully demonstrates Real-to-Sim-to-Real: single photo $\rightarrow$ Hunyuan3D reconstruction $\rightarrow$ MotionAnyMesh URDF generation $\rightarrow$ SAPIEN simulation policy training $\rightarrow$ deployment onto physical robots for execution. It end-to-end validates the physical accuracy of joint parameters.
- The normalized double-penalty function is crucial for selecting the direction of translational joints—the collision penalty penalizes the penetration direction, while the derailment penalty penalizes directions that are not parallel to the contact surface.
- Maintains stable performance on both PartNet-Mobility and Objaverse data sources, indicating that the method does not rely on specific data formats.
- Dataset construction: In addition to standard PartNet-Mobility (with GT URDF annotations), tests were also conducted on Objaverse open-vocabulary static meshes and Text/Image-to-3D generated assets, where the latter two were manually annotated with GT URDF.
- Strict criteria were used to evaluate physics executability: the URDF is loaded into the SAPIEN engine and driven across its entire valid range; any occurrence of penetration, detachment, or freezing is classified as a failure.

## Highlights & Insights
- The **"Perception-Action" methodology** is visionary: it does not merely pursue visually accurate segmentation/joint parameters, but instead targets physical simulation executability as the ultimate metric. This end-to-end design goal provides valuable insights.
- The **P3-SAM over-segmentation + VLM clustering** two-step segmentation strategy balances precision and semantics: geometric primitives maintain sharp boundaries, while VLM aggregation preserves semantic consistency.
- **SP4D as a physical anchor for VLM** is a key innovation: while VLMs possess powerful semantic reasoning capabilities, they lack physical common sense; supplementing them with kinematic video priors forms an elegant and complementary combination.
- **SDF-constrained trajectory optimization** is simple and effective: embedding collision detection into a continuous optimization objective results in smoother trajectories than discrete collision detection.

## Limitations & Future Work
- Relies on the closed-source reasoning of VLMs (GPT-4o), which is high in cost and hard to replicate—each object requires multiple VLM calls (part clustering + joint-tree inference). The API cost and latency could become bottlenecks for large-scale applications.
- The SP4D kinematic prior originates from video generation models, which may provide incorrect priors for rare mechanical structures outside the training data distribution (such as non-standard hinges or flexible joints).
- Only processes rigid articulated objects (revolute + prismatic joints), failing to support flexible deformations (e.g., soft tubes, ropes) or composite motions (e.g., helical joints).
- SDF trajectory optimization depends on Trimesh's SDF calculation, which may yield incorrect signed distance values for non-watertight meshes.
- The hardware requirement of three RTX 4090 GPUs is high, and the end-to-end processing time per object is not reported, leaving its efficiency for massive asset library conversion unknown.
- The motion range estimation is based on collision-detection heuristics, which may be too conservative or too aggressive for precision machinery with gaps or tolerances (e.g., gearboxes).
- The scale of the evaluation dataset is limited—the total test set size across PartNet-Mobility, Objaverse, and generated assets is not specified, and statistical significance requires more samples to verify.
- No direct comparison was conducted with the latest 3DGS-based articulated object methods (e.g., ArticulatedGS, ReArtGS).

## Related Work & Insights
- **vs Articulate-Anything**: Pure VLM reasoning + CAD retrieval, failing completely on out-of-domain objects; MotionAnyMesh bypasses these issues using 3D native segmentation and physical constraints.
- **vs Articulate-AnyMesh**: Also zero-shot, but relies on 2D-to-3D projection and pure semantic VLMs, leading to geometric fragmentation and kinematic hallucinations.
- **vs PARIS**: Requires multi-state observations (before and after object movement), making it inapplicable to purely static single-state meshes.
- **vs DreamArt/FreeArt3D**: Generative methods use diffusion models to infer motion but are limited by 3D topological constraints, causing penetration.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of using SP4D + VLM to eliminate hallucinations is inspiring, and the physics-constrained trajectory optimization is practical.
- Technical Depth: ⭐⭐⭐⭐ The three-stage pipeline has solid geometric and physical motivations for each step.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strict physics executability evaluation criteria, comprehensive ablations.
- Engineering Value: ⭐⭐⭐⭐⭐ Closed-loop Real-to-Sim-to-Real validation, directly applicable to robotics simulation.
- Depends on the quality of SP4D's kinematic masks—SP4D might generate inaccurate motion priors for atypical objects.
- Dependent on GPT-4o as the VLM, with high costs and latency.
- Only handles single-level articulation (one parent part + one child part); further scaling to complex multi-level linkage systems (like multi-joint robot arms) may require extending to tree-like joint structures.
- Levenberg-Marquardt optimization can get trapped in local optima, maintaining a certain dependency on the quality of initialization.
- Limitations of the URDF format itself—it does not support advanced physical attributes like soft bodies.

## Related Work & Insights
- **vs Articulate-AnyMesh**: Both are zero-shot articulation, but the latter uses pure VLM semantics + 2D-to-3D projection, leading to severe kinematic hallucinations and penetration. This work completely outperforms it via SP4D priors + physical optimization.
- **vs Articulate-Anything**: Based on CAD retrieval assembly, limited by predefined libraries, with poor generalization to novel geometries.
- **vs PARIS**: Requires multi-state observation input, whereas this work starts purely zero-shot from single-state static meshes.
- **vs GAPartNet**: Targeted at semantic part segmentation but does not estimate joint motion parameters, representing a different application scenario.

### Dataset & Evaluation Protocol
Evaluation is performed on PartNet-Mobility (a synthetic articulated dataset covering 46 object categories) and a subset of Objaverse (real-world scanned meshes). PartNet-Mobility contains common furniture and household appliances such as storage cabinets and microwaves, while the Objaverse subset contains atypical objects like industrial parts. Physics executability is evaluated in the SAPIEN engine, with a simulation step size of 0.01s and a friction coefficient of 0.5.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of SP4D-anchored VLM + physics-constrained optimization is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple data sources + 5 baselines + ablation + robot deployment.
- Writing Quality: ⭐⭐⭐⭐ Clearly organized with well-analyzed motivations.
- Value: ⭐⭐⭐⭐⭐ High practical value for generating simulation assets for embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Twinner: Shining Light on Digital Twins in a Few Snaps](twinner_shining_light_on_digital_twins_in_a_few_snaps.md)
- [\[CVPR 2025\] SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing](simavatar_simulation-ready_avatars_with_layered_hair_and_clothing.md)
- [\[CVPR 2025\] Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset](digital_twin_catalog_a_large-scale_photorealistic_3d_object_digital_twin_dataset.md)
- [\[ICML 2026\] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](../../ICML2026/3d_vision/stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)
- [\[CVPR 2025\] VGGT: Visual Geometry Grounded Transformer](vggt_visual_geometry_grounded_transformer.md)

</div>

<!-- RELATED:END -->
