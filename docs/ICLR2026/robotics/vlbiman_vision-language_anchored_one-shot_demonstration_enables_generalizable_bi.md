---
title: >-
  [Paper Note] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation
description: >-
  [ICLR 2026][Robotics][bimanual manipulation] VLBiMan is proposed as a framework that decomposes a single demonstration into invariant and adaptable atomic skills via task-aware bimanual decomposition, employs vision-language anchoring with a VLM to adapt to new object positions and instances in novel scenes, and achieves bimanual coordination through kinematics-aware trajectory composition. The framework achieves an 85.3% success rate across 10 complex bimanual tasks with only one demonstration, substantially outperforming imitation learning baselines that require hundreds of demonstrations.
tags:
  - ICLR 2026
  - Robotics
  - bimanual manipulation
  - one-shot demonstration
  - VLM anchoring
  - skill decomposition
  - cross-embodiment transfer
date: 2026-05-08
content_hash: 377e2573c982d2f2
---

# VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation

**Conference**: ICLR 2026
**arXiv**: [2509.21723](https://arxiv.org/abs/2509.21723)
**Code**: [Project Page](https://hnuzhy.github.io/projects/VLBiMan)
**Area**: Robotic Manipulation / Bimanual Manipulation
**Keywords**: bimanual manipulation, one-shot demonstration, VLM anchoring, skill decomposition, cross-embodiment transfer

## TL;DR

VLBiMan is proposed as a framework that decomposes a single demonstration into invariant and adaptable atomic skills via task-aware bimanual decomposition, employs vision-language anchoring with a VLM to adapt to new object positions and instances in novel scenes, and achieves bimanual coordination through kinematics-aware trajectory composition. The framework achieves an 85.3% success rate across 10 complex bimanual tasks with only one demonstration, substantially outperforming imitation learning baselines that require hundreds of demonstrations.

## Background & Motivation

**Background**: Bimanual robotic manipulation is a core challenge in embodied intelligence. Dominant approaches based on VLA models (ALOHA, π0, RDT-1B) train end-to-end policies from large-scale teleoperation demonstrations, demonstrating impressive performance on long-horizon tasks.

**Limitations of Prior Work**:
- VLA models require hundreds to thousands of teleoperation demonstrations; bimanual teleoperation is more difficult than single-arm (14-dimensional action space), making data collection prohibitively expensive.
- Adapting to new objects or tasks typically requires re-collecting demonstrations and retraining, which does not scale to open-world settings.
- Zero-shot methods (e.g., ReKep) rely on LLMs for task decomposition and prompt engineering, which is fragile and unreliable.
- One-shot imitation learning has been explored for single-arm settings, but the synchronous/asynchronous coordination complexity of bimanual tasks is substantially higher.

**Key Challenge**: Achieving broad generalization from minimal demonstrations requires identifying *what is invariant* and *what must be adapted* in manipulation tasks — a separation made harder by the coordination constraints inherent to bimanual tasks.

**Goal**: How can reusable bimanual manipulation skills be extracted from a single human demonstration and generalized to novel scenes (new positions, new object instances, new robot platforms)?

**Key Insight**: **"What matters more than How"** — rather than imitating the precise execution postures, the framework captures and reproduces relative spatial relationships between objects. In a pouring task, for instance, the key factor is the relative pose between the cup and the bottle, not the specific arm trajectory.

**Core Idea**: Decompose a demonstration into *invariant sub-skills* (directly reusable) and *adaptable sub-skills* (re-synthesized after VLM anchoring), enabling generalization from one demonstration to many scenarios.

## Method

### Overall Architecture: Three-Stage Pipeline

Given a task description $\mathcal{T}$ and a single demonstration $\mathcal{D} = \{(\mathcal{O}_t, \mathcal{A}_t)\}_{t=1}^T$, VLBiMan learns the mapping:

$$\mathcal{F}_{\text{VLBiMan}}: (\mathcal{T}, \mathcal{D}, \mathcal{S}_{\text{new}}) \mapsto \{\widetilde{\mathcal{A}}_t^{\text{new}}\}_{t=1}^{T'}$$

where $\mathcal{S}_{\text{new}}$ is the novel scene and $\widetilde{\mathcal{A}}_t^{\text{new}}$ denotes the adapted bimanual trajectory. The three stages are: decomposition → adaptation → composition.

### Key Design 1: Task-Aware Bimanual Decomposition (Invariant/Adaptable Separation)

**Spatiotemporal Segmentation**: Key poses are detected based on motion dynamics (velocity discontinuities, acceleration spikes) and state transitions (gripper open/close), segmenting the demonstration into temporal intervals $\tau_i = [t_i, t_{i+1}]$, each corresponding to a motion primitive $\mathcal{M}_i$.

**Atomic Skill Classification**: Each primitive is classified based on the object-robot coupling state. A binding indicator $\text{bind}(o, r, t)$ is defined, such that:

$$\forall t \in \tau_i, \text{bind}(o_k, r, t) = 1 \text{ and } \text{geometry}(o_k) \approx \text{geometry}(o_k^{\text{demo}})$$

Primitives satisfying this condition are labeled as **invariant skills** $\mathcal{M}_i^{\text{inv}}$ (actions performed after the object is firmly grasped, independent of scene layout); otherwise, they are labeled as **adaptable skills** $\mathcal{M}_j^{\text{var}}$ (pre-contact motions that must be adjusted based on new object positions). The demonstration is decomposed as:

$$\mathcal{D} \Rightarrow \{\mathcal{M}_i^{\text{inv}}\}_{i=1}^{N_{\text{inv}}} \cup \{\mathcal{M}_j^{\text{var}}\}_{j=1}^{N_{\text{var}}}$$

**Design Motivation**: The invariant/adaptable separation is central to generalization. The invariant component encodes the task essence (e.g., "how to pour stably"), while the adaptable component depends solely on the geometry of the new scene. This separation allows the majority of skills to be directly reused.

### Key Design 2: VLM-Anchored Adaptation (Semantics-Aware Geometric Alignment)

**VLM Scene Understanding**: Object category prompts are extracted from $\mathcal{T}$ and fed into a VLM (Florence-2 + SAM2) to obtain high-quality 2D semantic masks $\mathbf{M}_k^{\text{2D}}$, requiring neither CAD models nor 6D pose estimation.

**Geometric Adaptation** proceeds in three steps:
1. **Position Transfer**: The 3D displacement between representative points of the new and demonstration objects is computed as $\Delta\mathbf{x} = \mathbf{p}^{\text{new}} - \mathbf{p}^{\text{demo}}$ (representative points can be mask centroids or planar contact points).
2. **Orientation Adaptation**: For orientation-sensitive objects (e.g., pens, spoons), the principal axis direction is extracted from the second-order image moments of the 2D mask, and the relative rotation $\Delta\theta = \angle(\mathbf{v}^{\text{new}}, \mathbf{v}^{\text{demo}})$ is computed.
3. **Size Compensation**: For category-level variation (e.g., bottles of different sizes), the height difference $\Delta h_k$ is estimated from the z-extent of the point cloud to adjust vertical placement motions.

**Design Motivation**: The VLM is assigned the role of "anchoring" (segmentation and localization) rather than "planning" (task decomposition). VLM segmentation capabilities are strong and robust to lighting and distractors, whereas LLM-based planning remains fragile — a deliberate and appropriate assignment of roles.

### Key Design 3: Autonomous Trajectory Composition (Kinematic Feasibility Guarantee)

**Progressive IK Optimization**: For initial grasp motions, spline interpolation is used to progressively approach the target pose, with iterative inverse kinematics solved at each step:

$$\mathbf{q}^{(n+1)} = \text{IK}(\mathbf{T}_g^{(n)}), \quad \mathbf{T}_g^{(n)} = \text{SplineInterp}(\mathbf{T}_{\text{start}}, \mathbf{T}_{\text{goal}}, n)$$

**Dynamic Collision Compensation**: Safety margins are added along the base and vertical directions during the grasp approach phase:

$$\tilde{\mathbf{x}}^{\text{goal}} = \mathbf{x}^{\text{goal}} + \delta_{\text{base}}\mathbf{u}_\| + \delta_z\mathbf{u}_z$$

This ensures premature collisions are avoided under new object layouts. The composed trajectory is validated through a single physical playback.

## Key Experimental Results

### Main Results: Success Rate on 6 Basic Bimanual Tasks (25 trials/task)

| Method | plugpen | inserting | unscrew | pouring | pressing | reorient | Avg (Same) | Avg (New Instance) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mechanisms | 11/25 | 9/25 | 5/25 | 5/25 | 7/25 | 3/25 | 26.7% | 12.7% |
| MAGIC | 16/25 | 15/25 | 10/25 | 10/25 | 9/25 | 7/25 | 44.7% | 27.3% |
| ReKep | 14/25 | 11/25 | 10/25 | 12/25 | 10/25 | 8/25 | 43.3% | 29.3% |
| ReKep+ | 19/25 | 18/25 | 13/25 | 17/25 | 17/25 | 11/25 | 63.3% | 42.7% |
| **VLBiMan** | **25/25** | **23/25** | **20/25** | **21/25** | **20/25** | **19/25** | **85.3%** | **78.0%** |

### Ablation Study (Average Success Rate under New Instance + Distractor Conditions)

| VLM Type | Initial Grasp | IK Opt. | Collision Avoid. | Avg SR |
|------|:---:|:---:|:---:|:---:|
| SAM+DINOv2 | ours | ✓ | ✓ | 35.8% |
| ours | AnyGrasp | ✓ | ✓ | 31.7% |
| ours | ours | ✗ | ✓ | 29.2% |
| ours | ours | ✓ | ✗ | 34.2% |
| **ours** | **ours** | **✓** | **✓** | **59.2%** |

### Long-Horizon Tasks: 4 Multi-Stage Tasks (No Distractors)

| Method | reorient+unscrew | unscrew+pouring | tool-use scoop | tool-use funnel | Avg (Same) | Avg (New) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| ReKep+ | 11/25 | 10/25 | 7/25 | 6/25 | 34.0% | 19.0% |
| **VLBiMan** | **15/25** | **15/25** | **12/25** | **10/25** | **52.0%** | **41.0%** |

### Key Findings

- VLBiMan achieves 85.3% success rate with one demonstration, substantially outperforming imitation learning methods that require 50–100+ demonstrations.
- The plugpen task achieves a perfect 25/25 success rate, demonstrating the effectiveness of invariant/adaptable separation combined with VLM anchoring for fine-grained coordination tasks.
- New-instance generalization (78.0%) is only 7.3 percentage points below same-object performance (85.3%), indicating strong category-level adaptation via VLM anchoring.
- ReKep+ (with oracle initial grasps injected) achieves 63.3% but still lags significantly behind VLBiMan, indicating that the advantage extends beyond perception to the skill reuse strategy.
- Successful cross-embodiment transfer to a humanoid bimanual robot confirms that the skill representation is sufficiently abstract and not tied to specific hardware.

## Highlights & Insights

- **Efficiency Revolution: 1 vs. 100+ Demonstrations**: Data requirements are reduced by two orders of magnitude — particularly impactful for bimanual manipulation, where teleoperation difficulty and cost are 2–3× that of single-arm setups.
- **VLM as "Anchor" Rather Than "Planner"**: The VLM is assigned segmentation and localization tasks (at which it excels), rather than task decomposition and planning (where it remains unreliable) — a deliberate and well-matched role assignment.
- **Generalizable Invariant/Adaptable Separation Principle**: This separation is not limited to bimanual settings and can be extended to manipulation tasks in general, offering methodological inspiration for universal skill reuse frameworks.
- **"What > How" Manipulation Philosophy**: Capturing the task essence (relative spatial relationships between objects) rather than surface appearance (precise trajectories) means that the low-dimensional essence is sufficient to generalize from a single demonstration.

## Limitations & Future Work

- Only rigid objects are handled; deformable objects (cloth, ropes) require fundamentally different representations and control strategies.
- No runtime anomaly detection or recovery mechanism is present, making the system sensitive to slippage or occlusion.
- Fixed-base bimanual platforms limit reachable workspace; force/tactile sensing is absent. Future work may extend to mobile bases with haptic feedback.
- Skill decomposition and anchor selection still require human-in-the-loop involvement, leaving a gap to fully automated systems.

## Related Work & Insights

- **vs. ALOHA/π0 (End-to-End VLA)**: These methods require large demonstration sets and retraining; VLBiMan requires only one demonstration with no retraining — an efficiency gap exceeding 100×. However, VLA approaches may be more robust under extreme scene diversity.
- **vs. ReKep (Zero-Shot)**: No demonstrations are required, but the approach depends on LLM prompting and VFM keypoints, making it fragile. VLBiMan uses one demonstration to obtain task structure, yielding greater reliability than zero-shot methods.
- **vs. Mechanisms/MAGIC (Single-Arm One-Shot)**: Direct adaptation to bimanual settings yields poor performance (26.7%/44.7%), confirming that bimanual coordination is a unique challenge requiring dedicated decomposition and synchronization mechanisms.
- **Inspiration**: A promising direction is combining VLBiMan's decompose–adapt–compose pipeline with the generalization capability of VLA models — using a small number of demonstrations for structured initialization, supplemented by data-driven fine-tuning for broader coverage.

## Rating

⭐⭐⭐⭐⭐ (5/5)

Overall assessment: Achieving 85.3% success from a single demonstration represents a breakthrough balance between efficiency and generalizability in bimanual manipulation. The invariant/adaptable separation principle carries methodological value beyond this specific setting. Validation across 10 real-robot tasks and cross-embodiment transfer demonstrates practical utility, establishing this work as a benchmark contribution to the bimanual manipulation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICLR 2026\] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)
- [\[ICLR 2026\] When would Vision-Proprioception Policies Fail in Robotic Manipulation?](when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2026/robotics/lada_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
