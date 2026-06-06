---
title: >-
  [Paper Note] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action
description: >-
  [ICML 2026][Robotics][VLA] SOMA equips VLAs with persistent spatial-semantic memory—constructed via movable head camera scans, updated incrementally online…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLA"
  - "Spatial Memory"
  - "Out-of-Vision Manipulation"
  - "Movable Head Camera"
  - "Memory-Guided Manipulation"
date: 2026-05-08
content_hash: e69b0800658fbf15
---

# Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action

**Conference**: ICML 2026  
**arXiv**: [2605.22283](https://arxiv.org/abs/2605.22283)  
**Code**: To be released  
**Area**: robotics  
**Keywords**: VLA, Spatial Memory, Out-of-Vision Manipulation, Movable Head Camera, Memory-Guided Manipulation

## TL;DR
SOMA equips VLAs with persistent spatial-semantic memory—constructed via movable head camera scans, updated incrementally online, and retrievable by instructions—enabling robots to stably manipulate objects currently outside their field of view (OOV). Across 5 real-world OOV grasping tasks, it reduces the time to first sight, head search path, and number of grasp attempts by 40-60%.

## Background & Motivation
**Background**: Current mainstream VLAs typically map image-language inputs to actions end-to-end using multimodal large models with an action head, assuming a fixed tabletop or third-person perspective. This setup facilitates calibration and large-scale data collection; thus, most mainstream VLAs (e.g., GR00T-N1.5, π0, OpenVLA-OFT, SpatialVLA) assume task targets are "visible" by default.

**Limitations of Prior Work**: Purely reactive perception fails once a target is temporarily occluded or falls outside the camera's field of view. Models cannot localize targets or recall previously observed position information, leading to blind searching and sharp increases in failure rates for multi-stage and bimanual tasks.

**Key Challenge**: The perception-action loop is strictly sight-bound, yet semantic objects of manipulation often span multiple perspectives. Solutions either rely on MLLM "spatial imagination" (where spatial estimation quickly drifts when targets are invisible) or active head scanning (which lacks spatial memory and leads to forgetting in multi-step tasks). Both lack a unified mechanism for "look-then-remember-then-use."

**Goal**: Enable VLAs to solve OOV at three granularities: (1) scanning the entire workspace into a queryable spatial-semantic memory before the task; (2) incrementally correcting this memory based on new observations during manipulation; (3) accurately retrieving memory regions relevant to the current sub-goal during instruction reasoning.

**Key Insight**: The authors observe that failure modes are not "inaccurate guessing due to invisibility" but "seeing once but failing to retain." By scanning once to solidify scene instances into object-level memory with 3D geometry, the model can still retrieve position, appearance, and category even after targets leave the field of view. The core requirement is a stable memory substrate rather than deeper reasoning.

**Core Idea**: Use a movable head camera for an active scan to build an "object-level spatial-semantic memory bank," refresh it online using similarity-aware EMA, and allow VLM tokens to retrieve semantically relevant entries via cross-attention. This mechanism is seamlessly integrated into a DiT-based action diffusion head.

## Method
SOMA remodels perception as a "memory-centric" process. When the perception module finds the instruction target is not in view, it triggers an active scan for memory initialization. During manipulation, the head view continuously integrates new observations into memory. The DiT action head retrieves global context from memory to predict action chunks. Built on GR00T-N1.5, the framework adds three lightweight memory modules and a head-scanning script.

### Overall Architecture
Inputs include current frames from wrist (left/right) and head cameras, robot state, noisy action sequences, natural language instructions, and a pre-constructed scene memory $\mathcal{M}_0$. The output is an action chunk for the next $H$ steps. The pipeline consists of four stages: (1) Head pre-scan → Memory construction; (2) Real-time head view updates → Dynamic Memory Refinement to obtain $\hat{\mathcal{M}}_t$; (3) VLM-encoded vision-language tokens serve as Queries to perform cross-attention on $\hat{\mathcal{M}}_t$, generating memory-augmented tokens; (4) Augmented tokens + robot state + noisy action tokens enter DiT blocks, where the action decoder outputs the action chunk. The VLM language decoder remains frozen during training; only perception, memory, and action parameters are optimized.

### Key Designs

1. **Spatial Memory Construction (One-time Multi-view Spatial-Semantic Mapping)**:
    - **Function**: Drives the head along a predefined path before the task to fuse multi-view images into an object-level, queryable, 3D-aware initial memory $\mathcal{M}_0$.
    - **Mechanism**: Uniformly samples frames $\tilde{V}$ from scan video $V$. Three parallel pipelines process each frame: VGGT provides camera poses and scene geometry priors, YOLO provides 2D detection boxes and categories, and DINOv3 provides dense features. Instance appearance embeddings $\mathbf{f}_j^{(i)} \in \mathbb{R}^C$ are obtained via 2D boxes and average pooling, while VGGT geometry lifts 2D boxes to global 3D coordinates $\mathbf{b}_j^{(i)} \in \mathbb{R}^{8\times 3}$. Intra-class instance association is performed across views using "DINOv3 cosine similarity + 3D box spatial consistency." Matches exceeding a threshold are merged by averaging appearance and geometry into a global instance set $\{(\mathbf{f}_k, c_k, \mathbf{b}_k)\}_{k=1}^{N_I}$. Finally, 3D boxes are passed through learnable positional embeddings $\mathbf{p}_k = \Phi_{\text{pos}}(\mathbf{b}_k)$, and appearances through mapping $\Phi_{\text{mem}}$, concatenated as memory tokens $\mathbf{m}_k^0 = \Phi_{\text{mem}}(\mathbf{f}_k) + \mathbf{p}_k$.
    - **Design Motivation**: Purely learned spatial priors collapse when targets leave the view, necessitating a "foundation" of real multi-view observations. YOLO (semantics), DINOv3 (fine-grained appearance), and VGGT (geometry) are complementary. Learnable placeholder embeddings are injected for missed detections to prevent memory sparsity.

2. **Dynamic Memory Refinement (Similarity-aware EMA Incremental Update)**:
    - **Function**: Incrementally fuses new head-view observations into $\mathcal{M}_{t-1}$ during manipulation, maintaining global consistency while reflecting scene evolution (moved objects, occlusions, new arrivals).
    - **Mechanism**: The current frame $o_h^t$ generates $\mathcal{M}_t = \{\mathbf{m}_j^t\}$ via the same pipeline. Intra-class instance matching is performed (one-to-one mapping). Two scores are calculated: semantic similarity $s_{kj}^t = \sigma(\Phi_{\text{sim}}([\mathbf{m}_k^{t-1} - \mathbf{m}_j^t]))$ and dynamic fusion score $g_{kj}^t = \sigma(\Phi_{\text{fuse}}([\mathbf{m}_k^{t-1}, \mathbf{m}_j^t]))$. Their product yields an adaptive update coefficient $\alpha_{kj}^t = g_{kj}^t \cdot s_{kj}^t$ for temporal EMA: $\mathbf{m}_k^t = \alpha_{kj}^t \mathbf{m}_j^t + (1 - \alpha_{kj}^t) \mathbf{m}_k^{t-1}$. Unmatched new instances are appended, while unmatched old memories are retained to handle temporary occlusion.
    - **Design Motivation**: Fixed EMA coefficients either cause jitter or react slowly. Learning coefficients adaptively via sigmoid networks allows the model to switch between "small view jitter → stability" and "object movement → fast update." Retaining unmatched memory is key to OOV—targets are out of sight, not gone.

3. **Contextual Memory Retrieval (Instruction-Guided Memory Retrieval)**:
    - **Function**: Allows VLM vision-language tokens to selectively activate memory entries most relevant to the current instruction, injecting global spatial context into the action generation path.
    - **Mechanism**: VLM tokens $\mathbf{X}_{\text{vl}} = \{\mathbf{x}_i\}_{i=1}^{N_q}$ act as Queries, while memory $\hat{\mathcal{M}}_t$ aligned via $\Phi_{\text{align}}$ acts as Keys/Values. Standard scaled dot-product attention $\mathbf{X}_{\text{boost}} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{C}) \mathbf{V}$ produces memory-augmented tokens. These are injected into DiT blocks as global spatial priors, performing joint diffusion with original VL tokens, robot state, and noisy action tokens.
    - **Design Motivation**: Concatenating all memory tokens to VLM input increases context length and dilutes observations. "Retrieval-on-demand" via cross-attention keeps the perception path clean while providing task-relevant spatial evidence to the DiT during conditional generation.

### Loss & Training
Training data: 400 VR teleoperation demonstrations per real-world task, split into scanning phases (offline $\mathcal{M}_0$ construction) and manipulation phases. In simulation, $\mathcal{M}_0$ is built from the first frame. Optimization follows GR00T-N1.5’s diffusion action matching loss. The VLM language decoder is frozen; all other parts are trained jointly. Multi-task batch size 60, trained for 30k steps on 32 H200s. At inference, a lightweight detector triggers head scanning if the instruction target is absent from the current view.

## Key Experimental Results

### Main Results
Behavioral metrics on 5 real-world OOV grasping tasks: SOMA consistently reduces "Time to First Sight," "Head Search Angle," "View Correction Count," "Grasp Attempts," and "First Grasp Time" by 40-60% compared to GR00T-N1.5.

| Metrics (Task 5 Bimanual) | Ours (SOMA) | GR00T-N1.5 | Gain (Relative) |
|--------|------|----------|------|
| Time to First Sight (s) | 4.7 | 11.5 | -59% |
| Head Search Path (deg) | 70.4 | 164.0 | -57% |
| View Corrections | 2.3 | 5.3 | -57% |
| Grasp Attempts | 1.6 | 3.7 | -57% |
| Time to First Grasp (s) | 14.6 | 36.5 | -60% |

SimplerEnv (Visual Matching protocol, OXE pre-training + Fractal fine-tuning):

| Method | Pick Coke Can | Move Near | Open/Close Drawer | Average |
|------|------|------|------|------|
| OpenVLA-OFT | 72.3 | 69.6 | 47.2 | 63.0 |
| GR00T-N1.5 | 47.0 | 70.0 | 18.1 | 45.0 |
| **SOMA** | **85.0** | **73.0** | 31.5 | **63.2** |

RoboCasa Tabletop GR1 (5 task categories, 300 demo setting): SOMA achieves an average success rate of 52.0%, significantly higher than GR00T-N1.5 (44.3) and Diffusion Policy (39.2), maintaining its lead across all data scales (30/100/300/Full).

### Ablation Study
| Configuration | OOV Avg SR (%) | Description |
|------|---------|------|
| Scan + GR00T | 18.5 | Head scan only, no persistent memory |
| No-Scan SOMA | 19.8 | Single-frame initialized memory, no scan |
| Scan-only SOMA | 24.1 | Multi-view scan memory, no online updates |
| Full SOMA | **28.3** | Scan + Persistent Memory + Dynamic Refresh |

### Key Findings
- Adding scanning actions alone (Scan+GR00T) yields negligible gains, proving the OOV bottleneck is "memory loss" rather than "action lack." Persistent memory and online refreshing are the primary sources of gain.
- Even without multi-view scans (No-Scan SOMA), the model outperforms Scan+GR00T, suggesting the explicit memory structure itself is valuable.
- Behaviorally, SOMA exhibits "one-shot" grasping—grasp attempts drop from 3.7 to 1.6, a feat unattainable by reactive strategies.
- Gains scale with task difficulty: ~45% for single-step tasks vs. ~60% for bimanual multi-stage tasks (Task 5), which demand the highest cross-stage spatial consistency.

## Highlights & Insights
- Elevating "spatial memory" from implicit KV cache to an object-level, 3D-aware, language-retrievable explicit structure—this abstraction is applicable to navigation, long-video QA, and state tracking in HRI.
- Replacing fixed smoothing with adaptive EMA coefficients is a universal trick: any online memory requiring "stability under small perturbations and speed under real changes" can benefit.
- The trigger mechanism is clever—scanning is not performed at every step (expensive) but is coupled with task difficulty via a lightweight "target presence" detector.
- Putting memory in the cross-attention KV instead of the prompt context keeps the VLM mainline compact and remains friendly to diffusion action heads—a paradigm adaptable to π0 or OpenVLA.

## Limitations & Future Work
- Relies on the assumption that "pose drift is negligible" during short-range static scans via VGGT; geometry alignment may fail in large-scale or dynamic scenes, requiring SLAM or visual odometry with loop closure.
- Instance association via intra-class DINOv3 + 3D IoU is prone to merging errors with multiple identical objects (e.g., identical cups); a stronger instance ID mechanism is needed.
- Memory focuses on 3D boxes and appearance, failing to track internal states of articulated objects (e.g., drawer openness, bottle cap position), limiting articulated manipulation tasks.
- The scanning phase uses preset trajectories; future work should learn "active vision" policies to plan the most efficient viewpoints.

## Related Work & Insights
- **vs MemoryVLA / ContextVLA**: These store token-level features or keyframes (perceptual memory); SOMA stores object-level 3D instances, offering stronger geometry priors and interpretability.
- **vs SpatialVLA / RoboBrain**: These rely on implicit MLLM spatial priors; SOMA uses real multi-view observations for explicit grounding, showing significantly higher OOV robustness.
- **vs SAM2Act / MemER**: SAM2Act uses SAM2 memory banks + keyframes; MemER uses VLM-generated sub-goals. SOMA builds memory at the object-geometry level and integrates it end-to-end into a DiT head without external planners.

## Rating
- Novelty: ⭐⭐⭐⭐ The thre-stage "scan-memory-retrieve" pipeline for VLA is a clear and practical combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Robust coverage across real-world tasks, behavioral metrics, RoboCasa, and SimplerEnv.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive diagrams, and well-structured module explanations.
- Value: ⭐⭐⭐⭐ OOV is a critical requirement for long-horizon VLA; SOMA provides a reusable memory plugin paradigm for any VLA backbone.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](../../ICLR2026/robotics/memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)
- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)

</div>

<!-- RELATED:END -->
