---
title: >-
  [Paper Note] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM] This paper inverts the conventional instruction grounding paradigm — rather than compressing VLM knowledge into intermediate representations (symbolic skills or constraints)…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "VLM"
  - "Anti-Grounding"
  - "Robotic Manipulation"
  - "MPC"
  - "Structured VQA"
  - "Real2Sim2Real"
date: 2026-05-08
content_hash: f7e2cb44b8a58fc1
---

# AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making

**Conference**: NeurIPS 2025
**arXiv**: [2506.12374](https://arxiv.org/abs/2506.12374)  
**Code**: None  
**Area**: Multimodal VLM / Robotic Manipulation
**Keywords**: VLM, Anti-Grounding, Robotic Manipulation, MPC, Structured VQA, Real2Sim2Real

## TL;DR

This paper inverts the conventional instruction grounding paradigm — rather than compressing VLM knowledge into intermediate representations (symbolic skills or constraints), it renders candidate robot trajectories into multi-view scene images and evaluates action proposals directly within the VLM's native high-dimensional representation space, enabling zero-shot closed-loop robotic manipulation control.

## Background & Motivation

VLMs, trained on massive multimodal data, implicitly encode rich spatial understanding, physical intuition, and semantic reasoning capabilities. However, mainstream approaches that apply VLMs to robotic manipulation share a common challenge: the **information bottleneck**.

**Limitations of three dominant paradigms**:

1. **Symbolic skill sequence methods** (e.g., Code as Policies, SayCan): VLM outputs are translated into sequences of predefined skill calls (e.g., `pour(teapot, cup)`). The skill library is finite and discrete, unable to express continuous action details — for instance, pouring tea requires a precise, gradual tilt angle and spout alignment, all of which are lost when compressed into a single `pour` primitive.
2. **3D spatial constraint methods** (e.g., Voxposer): The scene is discretized into a voxel grid and a value function is computed. Voxelization discards fine geometric and texture information, and resolution is memory-bounded.
3. **Keypoint constraint methods** (e.g., Rekep): Spatial relationship constraints are defined via a small set of keypoints. This depends on manual keypoint selection, and constraint templates tend to be overly rigid and difficult to adapt to diverse manipulation scenarios.

The shared problem across all these methods is the compression of the VLM's high-dimensional representation space into low-dimensional intermediate representations, discarding substantial task-relevant information encoded in the VLM.

**Core Problem**: Can the process be reversed — rather than "grounding" VLM knowledge into action space, can actions be "lifted" into the VLM's representation space for evaluation?

## Method

### Overall Architecture

AntiGrounding constructs a Real2Sim2Real closed-loop control system:

1. **Real → Sim**: 3D reconstruction of the real scene into a simulation environment.
2. **MPC loop within Sim**: Generate candidate trajectories → multi-view rendering → VLM structured evaluation → select optimal trajectory.
3. **Sim → Real**: Synchronize simulation-verified actions to the physical robot for execution.

The key advantage of this architecture is safety — all actions are validated in simulation first, and only feasible plans are executed.

### Key Designs

1. **VLM-Driven Trajectory Evaluation (Structured VQA)**: The robot decision problem is reformulated as a standard visual question answering task. Candidate trajectories are rendered from multiple viewpoints in simulation (annotated as colored curves overlaid on scene images), and a set of structured sub-questions is posed to the VLM for scoring. Sub-questions span five dimensions: safety (collision risk), task alignment (semantic conformance to the instruction), efficiency (path length and time), physical feasibility (joint constraints and load), and viewpoint clarity (whether the current viewpoint suffices to judge trajectory quality). Each dimension carries an independent weight $w_k$, and the total score is a weighted sum. A multi-VLM agent ensemble (MoA framework) is adopted, where multiple VLMs score independently and their outputs are averaged to reduce variance from individual models. The final score is:
$$S_{j,t} = \frac{1}{M'}\sum_{m}\sum_{v} C'_{v,t}\!\left(\sum_k w_k \cdot s_{m,v,j,k,t}\right)$$

2. **Adaptive Multi-View Fusion with Confidence Weighting**: Different viewpoints carry different amounts of information at different manipulation stages. Viewpoint confidence is defined as $C_{v,t} = \frac{q_{\text{view}}}{1 + \lambda_C \cdot \sigma_{v,t}}$, where $q_{\text{view}}$ is the VLM's self-assessed clarity score for that viewpoint, and $\sigma_{v,t}$ is the variance of agent scores at that viewpoint (high variance indicates inconsistent judgment and low reliability). High-confidence viewpoints receive greater weight. This mechanism enables the system to automatically focus on information-rich viewpoints while discounting ambiguous ones.

3. **Annealing Trajectory Generation with Experience-Guided Sampling**: At each MPC timestep, candidate goal positions are sampled within a spherical region centered at the current end-effector position. The sampling radius and angular dispersion decay via exponential annealing:
$$R_{t'} = R_{\min} + (R_0 - R_{\min}) \cdot e^{-\lambda_R t'}$$
This enables broad exploration early on and fine-grained search in later stages. Historical VLM score feedback is incorporated to bias sampling toward high-scoring directions. When a sub-task transition signal is detected, annealing resets and exploration resumes.

### Real2Sim2Real Pipeline

Three components are used to construct a high-fidelity simulation environment:

- **SPAR3D**: Reconstructs object 3D meshes from single RGB images (reconstructed once and reusable across scenes).
- **SAM-6D**: Estimates object 6D pose to place meshes accurately within the simulation scene.
- **Scalable Real2Sim**: Estimates physical parameters (mass, inertia, friction) from robot grasp-and-place interaction data.

### Rigid-Coupled Rotation Module

Current VLMs struggle to directly understand and reason about 3D rotations. This module converts rotation matrices estimated by SAM-6D into visualized coordinate axis markers, enabling the VLM to perform axis-alignment reasoning (which axis needs to rotate and in what direction), and then maps the reasoning output back to a concrete rotation matrix — bridging VLM semantic reasoning and precise numerical control.

### Offline Policy Optimization

Accumulated execution data (VLM inputs/outputs, actual execution outcomes) is reviewed globally by a meta-VLM, which identifies prediction bias patterns (e.g., tasks where safety weights should be higher) and iteratively refines the sub-question phrasing and weight assignments in the evaluation template.

## Key Experimental Results

### Main Results (8 Manipulation Task Categories)

| Task Category | Specific Task | Code as Policies | Voxposer | Rekep | **AntiGrounding** |
|---|---|---|---|---|---|
| Precise Placement | Ring tape stacking | 1/10 | 3/10 | 5/10 | **6/10** |
| Precise Placement | Sponge stacking | 2/10 | 2/10 | 6/10 | **7/10** |
| Cluttered Scene | Toy maze navigation | 1/10 | 3/10 | 3/10 | **5/10** |
| Cluttered Scene | Drawer retrieval | 1/10 | 4/10 | 4/10 | **6/10** |
| Multi-Stage | Pouring water | 0/10 | 0/10 | **6/10** | 5/10 |
| Multi-Stage | Slipper arrangement | 0/10 | 0/10 | 5/10 | **7/10** |
| Commonsense Reasoning | Waste sorting | 0/10 | 2/10 | 2/10 | **7/10** |
| Commonsense Reasoning | Shape matching | 0/10 | 1/10 | 2/10 | **4/10** |
| **Total** | | **6.25%** | **18.75%** | **41.25%** | **57.5%** |

AntiGrounding outperforms or matches all baselines on 7 of 8 tasks. The only underperforming task — "pouring water" (5/10 vs. Rekep's 6/10) — relies more heavily on precise predefined constraints than on general reasoning.

### Ablation Study

| Variant | Tape Stacking | Maze Navigation | Pouring Water | Waste Sorting |
|---|---|---|---|---|
| Full AntiGrounding | 56.67% | 53.33% | 46.67% | 73.33% |
| w/o Structured VQA | 43.33% | 36.67% | 20.00% | 43.33% |
| w/o Multi-View Fusion | 26.67% | 23.33% | 23.33% | 36.67% |
| w/o VLM-Guided Trajectory Generation | 3.33% | 6.67% | 0% | 0% |
| w/o Rigid-Coupled Rotation | 13.33% | 16.67% | 0% | 13.33% |

VLM-guided trajectory generation is the most critical component — its removal causes near-complete system failure, indicating that purely random trajectories almost never yield feasible candidates under VLM evaluation.

### Offline Policy Optimization Results

| Task | Without Optimization | With Optimization | Gain |
|---|---|---|---|
| Tape Stacking | 55.24% | 67.02% | +11.78% |
| Maze Navigation | 48.72% | 72.85% | +24.13% |
| Pouring Water | 40.21% | 62.77% | +22.56% |
| Waste Sorting | 68.49% | 75.16% | +6.67% |

Offline optimization yields substantial gains on complex tasks, particularly for maze navigation and pouring water, which require fine-grained manipulation.

### Key Findings

- AntiGrounding exhibits the greatest advantage on tasks requiring precise spatial reasoning and commonsense understanding (e.g., waste sorting, which demands semantic understanding of garbage categories).
- Error analysis reveals that VLM errors are dominant (approximately 40%), followed by trajectory generation errors and R2S2R errors — VLM spatial reasoning remains the primary bottleneck.
- Multi-view fusion and structured VQA each contribute significantly, and their combination yields super-additive gains.
- Annealing-based trajectory search substantially outperforms fixed-radius search in complex scenes.

## Highlights & Insights

- **Paradigm innovation through inversion**: Lifting actions into VLM space rather than grounding VLMs into action space — bypassing the information compression bottleneck and allowing VLMs to operate on their "home turf" (image understanding + QA).
- **Reformulating robot decision-making as VQA**: Candidate trajectories are visually rendered into scene images, and the VLM selects the best plan by answering questions about the image. This reformulation enables any VLM to participate in robot decision-making in a zero-shot manner.
- **Implicit 3D via multi-view**: Rather than constructing explicit 3D representations, the system enables the VLM to implicitly reason about 3D information through multiple 2D viewpoints — "enough eyes see everything."
- **Complete Real2Sim2Real closed loop**: A fully automated pipeline from scene reconstruction through simulation verification to real-world execution.

## Limitations & Future Work

1. **High VLM inference cost**: Each MPC step requires multiple VLM calls (multi-view × multi-agent), severely limiting real-time performance — in practice, each step may take several seconds.
2. **Absolute success rates remain low**: The best result is 7/10, with an overall rate of 57.5%, leaving a significant gap before reliable deployment.
3. **No comparison with end-to-end VLA methods**: The absence of comparisons with end-to-end vision-language-action models such as RT-2 and Octo makes it unclear whether anti-grounding outperforms end-to-end learning.
4. **Dependence on simulation fidelity**: Reconstruction and physical parameter estimation errors in the R2S2R pipeline propagate to final performance.
5. **Rigid coupling assumption**: The method assumes a fixed rotational relationship between the end-effector and the grasped object, precluding handling of flexible objects or scenarios requiring re-grasping.
6. **VLM spatial reasoning ceiling**: Current VLMs exhibit notable deficiencies in precise spatial relationship reasoning, constituting the fundamental bottleneck of the approach.

## Related Work & Insights

- **Code as Policies**: Representative of symbolic skill sequence methods; achieves only 6.25% success — exposing the fundamental limitations of predefined skill libraries.
- **Voxposer**: 3D voxel value map method; suffers from severe information loss, achieving 18.75% success.
- **Rekep**: Keypoint constraint method; effective on tasks where predefined constraints match well (e.g., pouring water), but weak on commonsense reasoning.
- **PIVOT**: Also employs VLMs for trajectory evaluation, but uses a single viewpoint and simple prompts; AntiGrounding substantially improves upon this through multi-view fusion and structured VQA.
- **Inspiration**: The anti-grounding idea generalizes to autonomous driving planning evaluation (rendering candidate paths into driving scene images for VLM evaluation) and architectural/interior design plan assessment.

## Rating

⭐⭐⭐⭐ (3.5/5)

The paradigm innovation is impressive — the inversion of "grounding" opens a new direction for applying VLMs to robot control. The complete R2S2R pipeline and systematic ablation analysis demonstrate solid engineering. However, the low absolute success rate (57.5%), the absence of comparisons with end-to-end VLA methods, and high inference latency limit practical utility. The fundamental bottleneck lies in VLM spatial reasoning precision, which lies beyond the scope of what the proposed method itself can resolve.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)
- [\[ICML 2026\] Visual Persuasion: What Influences the Decision-Making of Vision-Language Models?](../../ICML2026/multimodal_vlm/visual_persuasion_what_influences_decisions_of_vision-language_models.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)
- [\[NeurIPS 2025\] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning](cyin_cyclic_informative_latent_space_for_bridging_complete_and_incomplete_multim.md)

</div>

<!-- RELATED:END -->
