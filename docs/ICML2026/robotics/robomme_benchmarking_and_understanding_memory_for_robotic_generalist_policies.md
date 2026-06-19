---
title: >-
  [Paper Note] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies
description: >-
  [ICML 2026][Robotics & Embodied AI][VLA] RoboMME systematically maps the four categories of human cognitive memory—"temporal, spatial, object, and procedural"—to 16 long-horizon robotic manipulation tasks (770k high-quality timesteps) for the first time. It performs systematic ablations on 14 "memory representation × integration mechanism" combinations using
tags:
  - ICML 2026
  - Robotics & Embodied AI
  - VLA
  - π0.5
date: 2026-05-08
content_hash: a752d99a18c22917
---
# RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies

**Conference**: ICML 2026  
**arXiv**: [2603.04639](https://arxiv.org/abs/2603.04639)  
**Code**: https://robomme.github.io/  
**Area**: Robotics  
**Keywords**: Memory augmentation, VLA, Robotic benchmark, Long-horizon manipulation, π0.5

## TL;DR
RoboMME systematically maps the four categories of human cognitive memory—"temporal, spatial, object, and procedural"—to 16 long-horizon robotic manipulation tasks (770k high-quality timesteps) for the first time. It performs systematic ablations on 14 "memory representation × integration mechanism" combinations using a π0.5 backbone, concluding that "perceptual memory + AdaLN modulator" offers the current best comprehensive trade-off.

## Background & Motivation
**Background**: While mainstream manipulation benchmarks like LIBERO, CALVIN, RLBench, and SimplerEnv involve temporal sequences, the vast majority of their tasks are effectively "Markovian." The current frame combined with the instruction is sufficient to predict the next action, making past observations disposable. Consequently, almost all VLAs (π0.5, RoboVLM, etc.) achieve high Success Rates (SR) on these benchmarks without truly testing memory capabilities.

**Limitations of Prior Work**: The few works intentionally examining memory employ disparate approaches: MemoryBench only covers three nearly-solved spatial tasks; MIKASA-Robo tasks are too short and lack high-quality demonstrations; memory-augmented models such as HistRISE, MemoryVLA, ContextVLA, and RoboMamba use different backbones and evaluation protocols, making it impossible to benchmark which "memory design" is superior.

**Key Challenge**: (1) The lack of a benchmark where tasks are explicitly non-Markovian and sufficiently large-scale; (2) The absence of an experimental framework to systematically compare all mainstream memory architectures under a fixed backbone and data budget. These gaps leave the "memory augmentation" field in a state where progress seems apparent, but the optimal direction remains unknown.

**Goal**: (i) To design a large-scale manipulation benchmark driven by cognitive theory, characterized by explicit non-Markovian properties across four types of memory requirements; (ii) To conduct a comprehensive orthogonal ablation of three memory representations (symbolic, perceptual, recurrent) and three integration mechanisms under a unified π0.5 backbone and fixed data budget.

**Key Insight**: The classic Atkinson-Shiffrin memory model divides long-term memory into procedural and declarative, with declarative memory further split into episodic (temporal, spatial, and object-based) and semantic. This work uses these four dimensions as the axes for task design, creating a task suite for each to ensure the benchmark covers a cognitively grounded "memory stimulus space."

**Core Idea**: By organizing tasks according to cognitive dimensions (when/where/what/how) and models via a 2D "memory representation × integration mechanism" matrix, the result is not just "another SOTA" but an interpretable conclusion on "which design is effective for which type of task."

## Method
The work produces two independent but coupled outputs: the RoboMME benchmark (16 tasks, 4 suites, 1600 demonstrations, 770k timesteps) and the MME-VLA model suite (14 memory variants built on π0.5). The benchmark provides a strictly non-Markovian evaluation environment, while the model suite provides controlled orthogonal ablations.

### Overall Architecture
The benchmark utilizes ManiSkill simulation with a 7-DoF Franka Panda, dual 256×256 cameras (front/wrist), and a joint-end-effector dual action space. Each task includes 100 episodes generated via trajectory playback; 5% key waypoint noise is injected into trajectories to enhance failure-recovery behaviors. On the model side, a fixed π0.5 backbone, a memory budget of 512 tokens, and 80k training steps are used. Evaluations are standardized across 50 episodes × 3 seeds × last-3 checkpoints to isolate the variables of "memory design."

### Key Designs

**1. Cognitive-oriented 4D Task Classification (Counting / Permanence / Reference / Imitation): Decomposing "memory" into when/where/what/how.**

Previous memory benchmarks either tested a single category or confounded multiple types, making it impossible to pinpoint "what memory the model lacks." RoboMME adopts the Atkinson-Shiffrin model to create four task suites along cognitive dimensions, each with 4 tasks of increasing difficulty, all designed to fail under Markovian policies. The Counting suite tests temporal memory (e.g., PickXTimes requires repeated grasping tasks); the Permanence suite tests spatial memory (VideoUnmask/ButtonUnmask requires remembering targets when objects are occluded); the Reference suite tests object memory (PickHighlight requires picking a briefly highlighted block); the Imitation suite tests procedural memory (RouteStick requires reproducing a trajectory after watching a video demonstration).

**2. MME-VLA Model Suite: An orthogonal ablation matrix of three representations × three integration mechanisms.**

To ensure comparability, all variables are locked to π0.5 with a 512-token memory budget, varying only how memory is stored and used. Representations include: Symbolic memory (compressing history into natural language sub-goals like SimpleSG or GroundSG with $[x,y]$ coordinates); Perceptual memory (retaining raw visual tokens compressed via token dropping or frame sampling); and Recurrent memory (compressing sequences into fixed-length hidden states using TTT or RMT). Integration mechanisms include: Memory-as-Context (concatenation at the input); Memory-as-Modulator (using AdaLN to modulate action features via multi-head attention); and Memory-as-Expert (adding a dedicated memory expert pathway with blockwise causal attention).

**3. Strict Non-Markovian Data Construction and Unified Evaluation Protocol.**

RoboMME ensures that "the same observation may correspond to different histories, leading to different correct actions" (e.g., seeing a red button requires different responses depending on whether it has been pressed 2 or 5 times). Video-conditioned tasks provide history only at initial steps, forcing the model to rely on memory during execution. The protocol fixes the action chunk training length at 20 and execution length at 16, with results averaged across 9 runs of the last 3 checkpoints × 3 seeds.

### Loss & Training
All models utilize the native flow matching action diffusion loss of π0.5 for multi-task joint training. Non-recurrent variants use a batch size of 64, while recurrent variants (TTT/RMT) are reduced to 16 due to VRAM constraints. Symbolic memory using QwenVL is fine-tuned on sub-goal annotations from 1600 demonstrations, while Gemini-based variants utilize prompt engineering.

## Key Experimental Results

### Main Results (Average SR% for representative variants across 16 tasks)

| Model Category | Variant (Memory + Integration) | Avg SR (%) |
|------|------|------|
| Memoryless Baseline | π0.5 | 17.93 |
| Memoryless Baseline | π0.5 + past actions | 19.73 |
| External SOTA | SAM2Act+ | 21.37 |
| External SOTA | MemER | 42.38 |
| Symbolic (Oracle Upper Bound) | GroundSG + Oracle | **84.08** |
| Symbolic (Real VLM) | GroundSG + QwenVL | 32.70 |
| Perceptual (Ours Best) | **FrameSamp + Modul** | **44.51** |
| Perceptual | TokenDrop + Modul | 38.04 |
| Perceptual | FrameSamp + Expert | 36.25 |
| Recurrent | TTT + Context | 22.28 |
| Recurrent | RMT + Context | 19.46 |
| Human Reference | Human | 90.50 |

### Ablation Study (FrameSamp + Modul vs π0.5 Baseline by Suite)

| Suite (Representative Task) | π0.5 Baseline | FrameSamp + Modul | Gain |
|------|---------|---------|---------|
| Counting (StopCube) | 6.67 | 42.00 | +35.3 |
| Permanence (VideoUnmaskSwap) | 18.67 | 24.44 | +5.8 |
| Reference (VideoRepick) | 0.44 | 30.44 | +30.0 |
| Imitation (RouteStick) | 4.67 | 66.67 | +62.0 |

### Key Findings
- **No Silver Bullet**: No single variant leads across all four memory types. Symbolic memory excels in tasks with "high-level discrete logic" (Counting/Permanence) but fails in continuous action imitation (Imitation), where perceptual memory thrives.
- **Perceptual + AdaLN Modulator is Overall Best**: FrameSamp+Modul achieved an average SR of 44.51, outperforming all other trainable variants and improving by 26.6 points over the memoryless baseline, validating that direct conditioning of the action pathway is more efficient than prompt insertion.
- **Recurrent Memory Lags Significantly**: Compressing history into fixed-length hidden states via TTT/RMT loses too much visual detail, resulting in SRs under 23, suggesting current SSM-style recurrent representations are not yet strong enough for this domain.
- **High Upper Bound of GroundSG+Oracle**: If sub-goal information is accurate (including coordinates), simple concatenation can approach human-level performance. The bottleneck lies in the sub-goal predictor, not the VLA itself.
- **MemER vs SAM2Act+**: MemER, which mixes perceptual and symbolic cues, doubled the performance of the purely perceptual SAM2Act+, indicating that "vision-language hybrid" memory is more robust.

## Highlights & Insights
- Deconstructing "memory" from an overused adjective into cognitive dimensions (when/where/what/how) is a rigorous scientific advancement for the field.
- The 2D "Representation × Integration" ablation matrix serves as a methodology for future work integrating new modules into existing models.
- The discovery that AdaLN modulation is superior to prompt concatenation for "non-semantic conditional signals" offers transferrable value for Designing DiT-style diffusion action models.
- Comparison with Oracle bounds (84.08) and Human performance (90.50) reveals that while a 46-point gap remains between current VLAs and humans, improving sub-goal prediction could close that gap to within 6 points.

## Limitations & Future Work
- The study is conducted entirely in ManiSkill simulation and lacks real-robot validation, where Permanence tasks might degrade under sensor noise.
- The 16 tasks are limited to tabletop single-arm scenarios, excluding mobile manipulation or human-robot interaction.
- The fixed 512-token memory budget precludes an analysis of the "budget-performance" curve.
- Implementation of recurrent memory was conservative; more powerful SSM variants like Mamba-2 or Griffin were not tested.
- Sub-goal annotation relies on simulator ground truth; real-world deployment requires a more generalized annotation pipeline.

## Related Work & Insights
- **vs MemoryBench**: RoboMME covers 16 non-Markovian tasks across 4 categories, whereas MemoryBench is limited to 3 space-centric tasks that are near saturation.
- **vs MIKASA-Robo**: RoboMME provides 770k high-quality demonstrations for large-scale imitation learning, contrasting with MIKASA's limited RL-focused data.
- **vs MemoryVLA / SAM2Act / MemER**: RoboMME provides the first unified benchmark to compare these methods under a standardized backbone and protocol.
- **vs ContextVLA / UniVLA**: While these utilize past frames/actions via concatenation, this work specifically identifies the Modulator as a significantly superior integration mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ The cognitive task design and orthogonal ablation matrix offer significant methodological value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive testing of 14 variants across 16 tasks with 9 runs each plus 4 external baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and high information density; could benefit from more visual error analysis.
- Value: ⭐⭐⭐⭐⭐ Provides the first truly usable standardized evaluation platform for memory-augmented robotic policies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[CVPR 2026\] FM-Steer: Enhance Generalist Policies with Value-Guided Cascaded Denoising](../../CVPR2026/robotics/fm-steer_enhance_generalist_policies_with_value-guided_cascaded_denoising.md)
- [\[ICML 2026\] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)

</div>

<!-- RELATED:END -->
