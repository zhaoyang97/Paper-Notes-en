---
title: >-
  [Paper Note] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation
description: >-
  [AAAI 2026][Robotics][open-vocabulary mobile manipulation] This paper addresses the base placement problem in open-vocabulary mobile manipulation (OVMM) and proposes a zero-shot framework that constructs a cross-modal re…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "open-vocabulary mobile manipulation"
  - "base placement"
  - "affordance reasoning"
  - "VLM visual prompting"
  - "coarse-to-fine optimization"
date: 2026-05-08
content_hash: 58d1a28c19019728
---

# Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation

**Conference**: AAAI 2026
**arXiv**: [2511.06240](https://arxiv.org/abs/2511.06240)  
**Code**: None  
**Area**: Robot Manipulation / Mobile Manipulation / Embodied Intelligence
**Keywords**: open-vocabulary mobile manipulation, base placement, affordance reasoning, VLM visual prompting, coarse-to-fine optimization

## TL;DR

This paper addresses the base placement problem in open-vocabulary mobile manipulation (OVMM) and proposes a zero-shot framework that constructs a cross-modal representation (Affordance RGB + Obstacle Map+) to project semantic affordance cues onto an obstacle map, followed by a coarse-to-fine iterative optimization that balances semantic and geometric constraints. The method achieves an 85% success rate across five manipulation tasks, substantially outperforming both geometric planners and pure VLM-based approaches.

## Background & Motivation

Open-vocabulary mobile manipulation (OVMM) requires robots to locate target objects and execute operations in unseen environments based on natural language instructions. Existing methods typically focus only on "navigating near the object" during the navigation phase, using classical planners such as A* or RRT* to find a collision-free position sufficiently close to the target before attempting manipulation. This paradigm overlooks a critical issue: **proximity does not imply manipulability**. For example, opening a cabinet requires the robot to face the drawer direction; grasping a kettle handle requires alignment with the handle side; placing a cup on a shelf requires facing the opening. An incorrectly chosen base position may render the manipulator unable to complete the task due to orientation or reachability constraints, even when the robot is in close proximity.

VLM-based methods, on the other hand, can understand task semantics (e.g., "which side is the handle"), but rely solely on a single RGB image for inference, limiting the field of view and preventing perception of occluded regions, while also lacking geometric constraints on collision and reachability. Semantic and geometric approaches thus each have distinct shortcomings, motivating a unified solution that accounts for both.

## Core Problem

**How to select a base placement position for a mobile manipulation robot that simultaneously satisfies task semantics (facing the correct affordance direction) and geometric feasibility (collision-free, reachable, appropriate distance)?** The key challenges are: (1) joint reasoning over semantic intent and spatial constraints is required; (2) the robot's perceptual field of view is limited (egocentric perspective), potentially occluding the most suitable placement direction.

## Method

### Overall Architecture

The system receives a natural language instruction (e.g., "place the cup on the shelf"), which GPT-4 parses into a sequence of sub-instructions. Each sub-instruction contains a target object name and an action description. Execution proceeds in three stages:

1. **Coarse Navigation**: An A* planner navigates the robot to within 1.5 m of the target object, facing it.
2. **Base Placement Selection** (the focus of this paper): Near the coarse navigation position, the optimal precise base placement is selected via affordance-guided coarse-to-fine optimization.
3. **Manipulation Execution**: A predefined manipulation primitive (pick/place/open) is executed at the selected position.

The core methodology consists of two major modules: **Affordance Guidance Projection (cross-modal projection)** and **Affordance-Driven Coarse-to-Fine Optimization**.

### Key Designs

1. **Affordance Guidance Projection (Cross-Modal Representation Construction)**: This is the most critical design in the method—it "projects" semantic affordance information from the RGB image onto the 2D obstacle map, overcoming the limitation that VLMs can only reason over RGB inputs. Two complementary representations are constructed:

    - **Affordance RGB ($I_{aff}$)**: Twelve directional arrows at 30° intervals in distinct colors are overlaid on the RGB image, along with an "A" arrow indicating the coarse affordance direction recommended by the VLM.
    - **Obstacle Map+ ($M_{local}^+$)**: The top-down obstacle map is augmented with the target object footprint $\mathcal{R}_t$, the robot's current position, a fan-shaped affordance region $\mathcal{F}_t$ centered on the "A" direction with ±60° extent, and 12 directional arrows color-matched to those in the RGB.

   Cross-modal alignment is achieved via **color consistency**: the VLM can associate directional arrows seen in the RGB with spatial positions on the map, enabling global semantic reasoning beyond the egocentric field of view. The affordance direction is determined by querying the VLM three times and applying majority voting to ensure robustness.

2. **Affordance Point Selection**: DINOv2 is used to extract visual features from the target object region; k-means clustering ($k=20$, cosine similarity) generates candidate keypoints, which are spatially deduplicated, annotated on the RGB, and submitted to the VLM to select the keypoint $\mathbf{g}$ most relevant to the task (e.g., kettle handle, cabinet door handle). This point serves as the center for subsequent Gaussian sampling.

3. **Coarse-to-Fine Iterative Optimization**: Candidate base positions are iteratively sampled around $\mathbf{g}$ and scored via a dynamically weighted composite function that balances semantic and geometric objectives:

    - **Scoring**: Each candidate $x$ receives score $w(x) = w_{geo}(x)^{\alpha_t} \cdot w_{sem}(x)^{1-\alpha_t}$, where the geometric term encourages maintaining the preferred radius $r^*$ from $\mathbf{g}$, and the semantic term encourages proximity to the VLM-updated semantic center $\mu_t$.
    - **Sampling**: $N_{sample}$ candidate points are drawn according to normalized weights, projected onto $M_{local}^+$ with index annotations, and submitted alongside $I_{aff}$ and the sub-instruction to the VLM for semantic ranking.
    - **Refinement**: The VLM returns the top-$k$ semantically preferred points; the semantic center $\mu_t$ is updated and $\sigma_s$ is reduced to promote convergence. In the final iteration, the top-5 candidates are taken, 2 outliers are removed, and the mean of the remaining 3 is used as the final base position.

   The key design is the **sigmoid schedule for $\alpha_t$**: early iterations use small $\alpha_t$ (emphasizing semantic exploration), while later iterations use large $\alpha_t$ (emphasizing geometric precision), achieving a smooth coarse-to-fine transition and avoiding local optima.

### Loss & Training

This method is a **zero-shot inference framework** that requires no training. There are no loss functions or training procedures. All decisions are made at inference time by the VLM (GPT-4o). Key hyperparameters include:

- Sampling standard deviation $\sigma_{sample}$ and truncation radius $r_{max}$
- Preferred distance $r^* = 0.7$ m
- Collision safety margin $\geq 0.4$ m
- Sigmoid schedule parameters $\alpha_{max}$, $\gamma$, and total iteration steps $T$
- VLM query count of 3 (majority voting mechanism)

## Key Experimental Results

| Task | Ours | Obj Center+A* | Obj Center+RRT* | Aff Point+A* | Aff Point+RRT* | Pivot(I) | Pivot(M+,Iaff) |
|------|------|---------------|-----------------|--------------|----------------|----------|----------------|
| Throw can into trash | 17/20 | 20/20 | 19/20 | 16/20 | 18/20 | 0/20 | 2/20 |
| Move kettle near red cup | 18/20 | 9/20 | 8/20 | 10/20 | 10/20 | 2/20 | 3/20 |
| Place cup on shelf | 17/20 | 8/20 | 3/20 | 13/20 | 10/20 | 1/20 | 2/20 |
| Open cabinet | 16/20 | 5/20 | 10/20 | 10/20 | 11/20 | 17/20 | 10/20 |
| Open dishwasher | 17/20 | 5/20 | 10/20 | 9/20 | 12/20 | 6/20 | 6/20 |
| **Overall Success Rate** | **85%** | 47% | 50% | 58% | 61% | 26% | 23% |

Experiments are conducted in NVIDIA Isaac Sim using the TIAGo++ robot platform (7-DOF left arm + differential drive base, head-mounted RGB-D camera at 1280×720), with 20 random initializations per task.

### Ablation Study

- **The $\alpha$ schedule is the core contribution**: Fixed $\alpha=0$ (pure semantic) achieves only 43%; fixed $\alpha=0.5$ (balanced) achieves 76%; fixed $\alpha=1$ (pure geometric) achieves 79%; dynamic sigmoid-scheduled $\alpha_t$ achieves 85%. The coarse-to-fine transition outperforms all fixed weight settings.
- **The cross-modal projection module is indispensable**: Removing all projections (using only raw RGB and map) causes a dramatic drop from 85% to 48%; removing the "A" arrow drops performance from 85% to 62%; removing the 12 auxiliary directional arrows causes only a minor drop to 80%. This indicates that the coarse affordance direction encoded by the "A" arrow is the most critical component.
- Pure VLM methods (Pivot) perform extremely poorly (23–26%), demonstrating that while current VLMs possess strong semantic capabilities, they lack the ability to translate semantic understanding into spatial reasoning without explicit projection mechanisms.

## Highlights & Insights

- **The cross-modal projection design is highly elegant**: Rather than having the VLM reason directly over the map (at which VLMs perform poorly), color-consistent arrows serve as visual anchors bridging the RGB and the map, allowing the VLM to leverage its strength in RGB understanding to indirectly perform spatial reasoning. This approach is transferable to other scenarios requiring VLM-based spatial decision-making.
- **The sigmoid schedule for coarse-to-fine optimization is simple yet effective**: A single parameter $\alpha_t$ achieves a smooth transition from semantic exploration to geometric refinement, avoiding the dilemma of conflicting objectives.
- **Zero-shot, cross-task generalization**: No task-specific training or fine-tuning is required; the framework handles both pick-and-place and articulation tasks purely via zero-shot VLM inference.
- A valuable empirical insight emerges: **VLMs cannot automatically translate semantic understanding into spatial reasoning**, and explicit projection mechanisms are necessary—a finding with broad implications for downstream VLM-robot integration.

## Limitations & Future Work

- **Simulation-only validation**: All experiments are conducted in Isaac Sim; no physical robot experiments are performed, and the sim-to-real gap is not addressed.
- **Assumed known object location**: The system assumes the 2D position of the target object is directly provided by the simulator, bypassing the object detection and localization problem in open-world settings.
- **Limited geometric precision**: Compared to purely geometric methods, VLM-guided placement may be insufficiently accurate for tasks requiring precise distance estimation.
- **Arm motion feasibility not considered**: Only the base position is optimized; the full collision-free motion trajectory of the manipulator arm to the target is not evaluated, which may still result in failure in cluttered environments.
- **VLM inference efficiency**: Each iteration requires a GPT-4o call for ranking; the latency and cost of multi-round iterations are substantial, making the approach unsuitable for real-time deployment.
- **Future directions**: Incorporating arm trajectory feasibility checking, extending to real robots, and combining active perception to reduce VLM query frequency.

## Related Work & Insights

- **vs. OK-Robot / COME-robot**: These OVMM systems address the full pipeline (perception + planning + execution), but still rely on simple distance heuristics for base placement, effectively delegating the final step ("where to stop") to a general-purpose planner. This paper specifically addresses that final step.
- **vs. MoMa-Pos / MoMa-Kitchen**: MoMa-Pos requires per-object-class modeling and generalizes poorly; MoMa-Kitchen uses egocentric affordance prediction, constrained by the field of view. The proposed cross-modal projection overcomes this field-of-view limitation without object-level modeling.
- **vs. PIVOT**: PIVOT uses iterative VLM annotation on RGB for spatial reasoning but lacks geometric constraints, performing poorly in this paper's experiments (26%). Even when augmented with the proposed cross-modal inputs (Pivot(M+, Iaff)), performance remains at only 23%, confirming that VLM-based iterative selection alone is insufficient and must be paired with geometric optimization.

The cross-modal projection paradigm (using color-consistent visual anchors to bridge RGB and spatial maps) can be broadly applied to other scenarios requiring VLM-based spatial decisions, such as indoor navigation, object placement planning, and scene rearrangement. The dynamic weight scheduling strategy (sigmoid schedule for smooth transition between potentially conflicting objectives) is a general optimization paradigm transferable to other multi-objective problems. The paper also highlights an important limitation of current VLMs—**strong semantic understanding but weak spatial reasoning**—which informs future directions for improving VLM spatial capabilities or designing better VLM-robot interfaces.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of cross-modal projection and coarse-to-fine scheduling is novel, though individual components (VLM prompting, Gaussian sampling, CEM-style optimization) are established techniques.
- Experimental Thoroughness: ⭐⭐⭐ The scale of 5 tasks × 20 random initializations is moderate, ablation studies are well-designed, but evaluation is limited to simulation with no real-robot experiments.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with precise problem formulation, detailed method description, and well-coordinated figures and tables.
- Value: ⭐⭐⭐⭐ The paper identifies a critical and overlooked problem in OVMM (base placement selection), provides a practical zero-shot solution, and offers broadly applicable insights for VLM integration in robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations](from_passive_perception_to_active_memory_a_weakly_supervised_image_manipulation_.md)
- [\[ICLR 2026\] Attribution-Guided Decoding](../../ICLR2026/robotics/attribution-guided_decoding.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[AAAI 2026\] Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)

</div>

<!-- RELATED:END -->
