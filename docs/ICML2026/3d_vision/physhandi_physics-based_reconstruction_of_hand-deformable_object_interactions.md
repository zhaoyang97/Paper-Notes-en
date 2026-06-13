---
title: >-
  [Paper Note] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions
description: >-
  [ICML 2026][3D Vision][Hand-Object Interaction] This paper introduces PhysHanDI, which couples the MANO hand model with a Spring-Mass soft object model. Dense hand meshes drive the physical simulation of deformable objec…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Hand-Object Interaction"
  - "Deformable Objects"
  - "Spring-Mass"
  - "MANO"
  - "Inverse Physics"
date: 2026-05-08
content_hash: 5c6a6f24259852bf
---

# PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions

**Conference**: ICML 2026  
**arXiv**: [2605.09538](https://arxiv.org/abs/2605.09538)  
**Code**: Not released  
**Area**: 3D Vision / Hand-Object Interaction Reconstruction / Physics Simulation  
**Keywords**: Hand-Object Interaction, Deformable Objects, Spring-Mass, MANO, Inverse Physics

## TL;DR
This paper introduces PhysHanDI, which couples the MANO hand model with a Spring-Mass soft object model. Dense hand meshes drive the physical simulation of deformable objects, while object simulation refines hand reconstruction. The method achieves SOTA dense 3D reconstruction of both hands and soft objects from sparse-view RGB-D videos.

## Background & Motivation
**Background**: Existing hand-object interaction reconstruction methods often assume objects are rigid or piecewise rigid (e.g., HOnnotate, ARCTIC), where object dynamics can be described using a reference shape and global or partial rigid transformations.

**Limitations of Prior Work**: Many everyday objects, such as clothing, plush toys, and charging cables, exhibit significant, spatially varying non-rigid deformations. Rigid frameworks fail entirely in such cases. Limited work on soft objects (e.g., HMDO, qi2025human) only addresses small-scale local deformations caused by finger pressing, unable to model large-scale global deformations like bending a plush toy’s arm by 180°.

**Key Challenge**: The most relevant work, PhysTwin, uses a Spring-Mass model for large deformations but simplifies the hand to approximately 30 sparse control points sampled from depth maps. This leads to two issues: (1) contact points are unobservable under occlusion, resulting in inaccurate force modeling; (2) the sparse control points produce suboptimal Spring-Mass topologies (with an overly large connection radius $\delta$), compromising simulation stability. Moreover, PhysTwin does not provide complete 3D hand reconstruction.

**Goal**: (1) Achieve dense 3D reconstruction of both hands and soft objects; (2) use dense hand meshes to drive object simulation for accurate force modeling; (3) leverage object physics priors to improve hand reconstruction accuracy.

**Key Insight**: Since Spring-Mass simulation heavily depends on control point geometry, replacing the 30 sparse depth points with 778 dense vertices from the MANO model naturally yields denser, more accurate contact sets and more reasonable connection radii.

**Core Idea**: Use MANO’s dense hand vertices as "virtual nodes" in the Spring-Mass system, coupling hand-object physics through a hand→object force field. Inverse physics (propagating object simulation errors back to MANO parameters) enables complementary hand↔object optimization.

## Method

### Overall Architecture
The input is sparse-view (default three-view) RGB-D video. The system parameterizes each hand as a MANO model $\Theta_h=\{\bm\theta,\bm\beta,\mathbf R,\mathbf t\}$ and each soft object as a Spring-Mass graph $\mathcal O=(\mathcal N,\mathcal E)$ (with node positions $\mathbf x_i$, velocities $\mathbf v_i$, unit mass, spring stiffness $s_{ij}$, damping $\gamma_{ij}$, and connection radius $\delta$). The pipeline consists of three stages: (1) **Hand Reconstruction**: Fit MANO to each frame using 2D keypoints, depth, and temporal smoothing losses; (2) **Object Reconstruction**: Given the hand, use MANO vertices as force sources for forward Spring-Mass simulation, optimizing spring parameters with Chamfer and CoTracker3 trajectory losses; (3) **Hand Refinement**: Freeze the object model and backpropagate object simulation errors to MANO parameters via inverse physics for more accurate hand reconstruction.

### Key Designs

1. **Dense MANO-Driven Spring-Mass Force Modeling**:
    - **Function**: Use all 778 MANO vertices as virtual control nodes $\mathcal V'$, injecting contact forces into the soft object.
    - **Mechanism**: The force on each node $\mathbf F_i$ comprises three components: $\mathbf F_i=\sum_{(i,j)\in\mathcal E}\mathbf F_{i,j}^{\text{spring}}+\mathbf F_{i,j}^{\text{damping}}+\mathbf F_i^{\text{external}}$, where spring forces follow Hooke’s law $\mathbf F_{i,j}^{\text{spring}}=s_{ij}(\|\mathbf x_j-\mathbf x_i\|-r_{ij})\frac{\mathbf x_j-\mathbf x_i}{\|\mathbf x_j-\mathbf x_i\|}$. Virtual springs $\mathcal E^{\text{virtual}}$ are automatically established within the connection radius $\delta$ between hand and object. MANO vertex positions are fixed as boundary conditions, and Newton’s second law is integrated over time.
    - **Design Motivation**: PhysTwin’s sparse depth points result in overly long virtual springs, with a $\delta/\Delta x$ ratio far from the recommended value of 3, causing excessive wave diffusion and distorted contact modeling. Using 778 dense vertices naturally reduces $\delta$ to a reasonable range, concentrating contact forces on actual contact surfaces.

2. **Inverse Physics-Driven Hand Refinement**:
    - **Function**: Backpropagate object simulation losses to MANO parameters, ensuring the reconstructed hand aligns with physically plausible object deformations.
    - **Mechanism**: Define $\mathcal S_t(\Theta_h)$ as the differentiable simulation of object node positions at time $t$ given hand parameters. Solve $\tilde\Theta_h=\arg\min_{\Theta_h}\tfrac{1}{T}\sum_t[\mathcal L_{ch}(\mathcal S_t(\Theta_h),\mathcal P)+\lambda_{tr}\mathcal L_{tr}(\mathcal S_t(\Theta_h),\mathbf T)]$, where $\mathcal P$ is the observed point cloud and $\mathbf T$ is the CoTracker3 trajectory. Gradients are backpropagated through the Spring-Mass differentiable integration to $\Theta_h$.
    - **Design Motivation**: Single-view RGB-D data suffers from severe under-determination (e.g., fingers occluded by objects or self-occlusion). Physical consistency imposes an additional constraint that "the hand must deform the object in this way," eliminating many physically implausible poses and providing an additional form of supervision.

3. **Three-Stage Sequential Optimization with Topology Preservation**:
    - **Function**: Avoid local minima caused by hand-object joint optimization confusion and explicitly ensure Spring-Mass topology approaches the optimal configuration recommended in peridynamics literature.
    - **Mechanism**: Alternate between fixing the hand for object optimization and fixing the object for hand refinement, with unidirectional gradients in each stage. The object stage uses PhysTwin’s differentiable simulation, but the dense MANO control significantly reduces the optimized $\delta$. The authors quantify topology quality using Radius-to-Resolution Deviation $RRD=|(\delta/\Delta x)/r-1|$ ($r=3$).
    - **Design Motivation**: Joint optimization often leads to spurious solutions where object changes compensate for hand errors. Sequential optimization with $RRD$ monitoring ensures progress toward physically interpretable solutions.

### Loss & Training
Hand stage: $\min_{\Theta_h}\mathcal L_{2D}+\lambda_d\mathcal L_d+\lambda_t\mathcal L_t$, where $\mathcal L_{2D}$ is the 2D keypoint reprojection error, $\mathcal L_d$ is the depth rendering difference, and $\mathcal L_t$ is temporal smoothing. Object stage: Chamfer loss $\mathcal L_{ch}$ measures the distance between simulation nodes and the point cloud, while $\mathcal L_{tr}$ measures the $\ell_2$ difference with CoTracker3 pseudo-GT trajectories. Hand refinement stage: Reuse $\mathcal L_{ch}+\lambda_{tr}\mathcal L_{tr}$, but gradients are applied to MANO parameters. All masses are set to unit mass (no GT mass information).

## Key Experimental Results

### Main Results
On the PhysTwin-dense subset (excluding sequences with needle-like contact) and the custom DenseHDI dataset (19 sequences, 10 soft object categories), PhysHanDI is compared with PhysTwin, Spring-Gaus, and GS-Dynamics.

| Dataset         | Task               | Metric         | PhysTwin | PhysHanDI | Gain   |
|------------------|--------------------|----------------|----------|-----------|--------|
| PhysTwin-dense  | Reconstruction+Sim | $CD_{dyn}$ ↓   | 10.78    | 8.32      | -22.8% |
| PhysTwin-dense  | Reconstruction+Sim | Track Err. ↓   | 1.00     | 0.89      | -11%   |
| PhysTwin-dense  | Future Prediction  | $CD_{dyn}$ ↓   | 16.32    | 14.35     | -12%   |
| DenseHDI        | Reconstruction+Sim | CD ↓           | 5.59     | 5.06      | -9.5%  |
| DenseHDI        | Future Prediction  | CD ↓           | 7.98     | 7.54      | -5.5%  |

Spring-Gaus fails under sparse three-view input ($CD_{dyn}=27.79$), and GS-Dynamics degrades to learning only small motions in short sequences ($CD_{dyn}=33.37$).

### Ablation Study
**Single-View Future Prediction (PhysTwin fails due to contact point recognition issues):**

| Configuration    | Hand CD ↓ | CD ↓  | Track Err. ↓ |
|------------------|-----------|-------|--------------|
| Ours w/o Hand Refinement | 7.36      | 42.8  | 7.57         |
| Ours Full        | 7.17      | 33.5  | 6.75         |

**Spring-Mass Topology Quality ($RRD$ closer to 0 indicates better adherence to peridynamics recommendations):**

| Method     | $RRD_{\text{object}}$ ↓ | $RRD_{\text{virtual}}$ ↓ |
|------------|-------------------------|--------------------------|
| PhysTwin   | 0.64                   | 2.63                    |
| PhysHanDI  | 0.32                   | 0.35                    |

The virtual spring $RRD$ drops by 7×, indicating that dense hand control nearly eliminates PhysTwin’s "stretched spring hard contact" topology distortion.

### Key Findings
- Under sufficient multi-view input, initial MANO fitting is already accurate, and inverse physics refinement primarily benefits under under-determined scenarios like "single-view future prediction," where Hand CD drops from 7.36 to 7.17, and object CD decreases by 9.3 mm.
- Robustness tests (adding noise to depth, tracking, and control) show that PhysHanDI’s CD remains nearly unchanged under perturbed tracking (5.30→5.56), while PhysTwin jumps to 9.60, demonstrating that dense hand control absorbs most upstream noise.
- Topology analysis reveals a transferable principle: simulation accuracy depends primarily on the Spring-Mass discretization-to-connection radius ratio $\delta/\Delta x$, and controller density directly determines whether this ratio falls within the recommended range.

## Highlights & Insights
- **Inverse Physics for Hand Supervision**: This is the first instance of using "object physical consistency" to refine hand poses, elevating hand reconstruction from "fitting pixels" to "fitting physics." The approach is elegant and transferable to any hand-object differentiable simulation framework.
- **Controller Density and Topology Quality**: Introduces the peridynamics $\delta/\Delta x\approx 3$ heuristic into hand-object simulation and uses $RRD$ to explain why PhysTwin performs poorly, providing a theoretical basis for dense control.
- **Training-Free Physical Priors**: The entire pipeline avoids training new networks, with all differentiable optimization performed per video, making it naturally suited for small datasets and new objects.

## Limitations & Future Work
- **Unit Mass Assumption**: All mass nodes use $m_i=1$, failing to distinguish between light soft objects (scarves) and heavy ones (wet towels). Incorporating material priors or multimodal tactile data could unlock finer dynamics.
- **Dependency on RGB-D + Multi-View CoTracker3 Pseudo-GT**: Adapting to monocular RGB requires addressing the precision limits of upstream depth/tracking estimation.
- **Limited to Spring-Mass Models**: Only supports "continuous medium + discrete spring" models, with limited capability for higher-order phenomena like self-contact or topological changes (e.g., tearing) in cloth folding.
- **No Real-Time Discussion**: Differentiable simulation and inverse physics optimization over long videos pose engineering bottlenecks.

## Related Work & Insights
- **vs PhysTwin (jiang2025phystwin)**: Both use Spring-Mass for object simulation, but PhysTwin relies on ≈30 sparse depth-sampled points as control, while PhysHanDI uses 778 MANO vertices. PhysHanDI also provides complete hand reconstruction and hand refinement.
- **vs HMDO / qi2025human**: HMDO assumes objects undergo only local deformations under finger contact, while PhysHanDI handles large-scale global deformations.
- **vs Spring-Gaus (zhong2024reconstruction) / GS-Dynamics (zhang2024dynamics)**: These methods require dense views or long sequences to learn dynamics, while PhysHanDI remains stable under sparse three-view short sequences, thanks to explicit MANO + physics priors.
- **Insights**: The hand↔object inverse physics approach can be extended to robot↔object (using robot joint motion to infer contact parameters and refine motion planning) and body↔scene (refining human poses after interacting with a scene, e.g., sitting on a sofa).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First closed-loop inverse physics refinement for hand-object reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three-view, single-view, four perturbations, and topology quantification; lacks long-term evaluation on real-world open objects.
- Writing Quality: ⭐⭐⭐⭐ Clear storyline (hand→object→hand) with well-explained physics-learning interplay.
- Value: ⭐⭐⭐⭐ Provides a new physics-learning baseline for soft object reconstruction in AR/VR and robotic teleoperation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](../../CVPR2026/3d_vision/arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[CVPR 2026\] PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](../../CVPR2026/3d_vision/pad-hand_physics-aware_diffusion_for_hand_motion_recovery.md)
- [\[CVPR 2026\] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves](../../CVPR2026/3d_vision/glove2hand_synthesizing_natural_hand-object_interaction_from_multi-modal_sensing.md)
- [\[CVPR 2026\] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](../../CVPR2026/3d_vision/physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](../../AAAI2026/3d_vision/physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)

</div>

<!-- RELATED:END -->
