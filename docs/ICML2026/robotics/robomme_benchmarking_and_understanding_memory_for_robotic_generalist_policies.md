---
title: >-
  [Paper Note] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies
description: >-
  [ICML 2026][Robotics][Memory enhancement] RoboMME systematically maps the four categories of human cognition—"temporal, spatial, object…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Memory enhancement"
  - "VLA"
  - "Robot benchmark"
  - "Long-horizon manipulation"
  - "π0.5"
date: 2026-05-08
content_hash: ae4565adc8adf0d0
---

# RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies

**Conference**: ICML 2026  
**arXiv**: [2603.04639](https://arxiv.org/abs/2603.04639)  
**Code**: https://robomme.github.io/  
**Area**: robotics  
**Keywords**: Memory enhancement, VLA, Robot benchmark, Long-horizon manipulation, π0.5

## TL;DR
RoboMME systematically maps the four categories of human cognition—"temporal, spatial, object, and procedural"—to 16 long-horizon robotic manipulation tasks (770k high-quality timesteps) for the first time. Through a systematic ablation of 14 "memory representation × integration" combinations on the π0.5 base, it identifies "perceptual memory + AdaLN modulator" as the current optimal comprehensive trade-off.

## Background & Motivation
**Background**: While mainstream manipulation benchmarks like LIBERO, CALVIN, RLBench, and SimplerEnv involve temporal sequences, the vast majority of tasks are actually "Markovian"—the current frame plus the instruction is sufficient to predict the next action, and past observations can be discarded. Consequently, almost all VLAs (such as π0.5, RoboVLM) achieve high SR on these benchmarks without truly examining memory capabilities.

**Limitations of Prior Work**: The few works intentionally examining memory follow disparate paths: MemoryBench only covers three nearly-resolved spatial tasks; MIKASA-Robo tasks are too short and lack high-quality demonstrations; and memory-enhanced models like HistRISE, MemoryVLA, ContextVLA, and RoboMamba each use different backbones and evaluation protocols, making it impossible to horizontally compare which "memory design" is genuinely superior.

**Key Challenge**: (1) The lack of a benchmark that is truly non-Markovian and sufficiently large-scale; (2) The lack of an experimental framework to systematically compare all mainstream memory architectures under a fixed backbone and fixed data budget. These gaps leave the "memory enhancement" field in a state where progress seems apparent, but it is unclear which approach actually wins.

**Goal**: (i) To design a large-scale manipulation benchmark driven by cognitive theory, explicitly non-Markovian, and covering four types of memory requirements; (ii) To perform a complete orthogonal ablation of three memory representations (symbolic, perceptual, recurrent) and three integration methods on the same π0.5 base under the same data budget.

**Key Insight**: The classic Atkinson-Shiffrin memory model divides long-term memory into procedural and declarative; declarative memory is further divided into episodic (containing temporal, spatial, and object dimensions) and procedural. The authors utilize these four dimensions as the axes for task design, creating a task suite for each dimension to ensure the benchmark covers a cognitively sound "memory stimulus space."

**Core Idea**: By organizing tasks by cognitive dimensions (when/where/what/how) and models by a 2D "memory representation × integration mechanism" matrix, the result is not just "another SOTA," but an interpretable conclusion on "which design is effective for which type of task."

## Method
There are two independent but coupled products: the RoboMME benchmark (16 tasks, 4 suites, 1600 demonstrations, 770k timesteps) and the MME-VLA model suite (14 memory variants built on π0.5). The benchmark provides a strictly non-Markovian evaluation environment, while the model suite provides controlled orthogonal ablations.

### Overall Architecture
The benchmark utilizes ManiSkill simulation with a 7-DoF Franka Panda, dual frontal/wrist cameras at 256×256, and joint-end-effector dual action spaces. Each task includes 100 episodes generated via trajectory replay; trajectories are injected with 5% key-waypoint noise and recovery to enhance failure-recovery behavior. On the model side, the π0.5 backbone is fixed with a 512-token memory budget and 80k-step training, evaluated uniformly under 50 episodes × 3 seeds × last-3 ckpt, converging all variables to the "memory design" itself.

### Key Designs

1.  **Cognitive-Oriented 4-Dimensional Task Taxonomy (Counting / Permanence / Reference / Imitation)**:
    - **Function**: Decouples the abstract concept of "memory" into four independently evaluable cognitive dimensions (when/where/what/how), with 4 tasks of increasing difficulty per dimension, covering scenarios where Markovian policies are guaranteed to fail.
    - **Mechanism**: The **Counting suite** tests temporal memory (e.g., PickXTimes requires repeated grasping as specified; StopCube requires pressing a button at a specific moment); the **Permanence suite** tests spatial memory (VideoUnmask/ButtonUnmask requires identifying targets based on memory while all blocks are occluded; Swap variants dynamically exchange container positions); the **Reference suite** tests object memory (PickHighlight requires picking a block that flashed briefly; VideoPlaceButton/Order requires execution based on temporal/ordinal references in language); the **Imitation suite** tests procedural memory (MoveCube/InsertPeg/PatternLock/RouteStick requires replicating grasp patterns, insertion directions, or linear/circular trajectories after watching a video demonstration).
    - **Design Motivation**: Previous benchmarks either tested one type of memory or mixed them, preventing localization of "what memory the model lacks." By slicing according to cognitive dimensions, the strength/weakness profile of each model can be directly read, facilitating diagnosis and improvement.

2.  **MME-VLA Model Suite: Three Memory Representations × Three Integration Methods**:
    - **Function**: Implements 14 memory architecture variants on a fixed π0.5 backbone, forming an orthogonal ablation design space.
    - **Mechanism**: **Representation Side**—*Symbolic memory* compresses history into natural language sub-goals (SimpleSG for description; GroundSG adds frontal view coordinates $[x, y]$), generated by Gemini-2.5-Pro, fine-tuned Qwen3-VL-4B, or Oracle simulator truths; *Perceptual memory* preserves history as visual tokens, controlled within a 512-token budget via token dropping (redundancy removal by RGB difference) or frame sampling (uniform downsampling); *Recurrent memory* uses TTT (online updating of fast weights) or RMT (learned memory slots) to compress sequences into fixed-length hidden states. **Integration Side**—*Memory-as-Context* concatenates memory tokens directly into the VLM input; *Memory-as-Modulator* uses AdaLN in each layer of the action expert to transform memory via multi-head attention into scale/shift parameters to modulate action features; *Memory-as-Expert* adds a dedicated memory expert path communicating with the VLM/action expert via blockwise causal attention.
    - **Design Motivation**: Previous memory methods reported results using different backbones, making them incomparable. By locking down π0.5 and the 512-token budget, the remaining differences can be purely attributed to "how memory is stored" and "how memory is used," providing a scientific controlled experiment.

3.  **Strictly Non-Markovian Data Construction and Unified Evaluation Protocol**:
    - **Function**: Ensures each task "must use historical information for correct decision-making" and freezes all confounding variables that might introduce variance.
    - **Mechanism**: All tasks are designed such that the same observation might correspond to different histories $\rightarrow$ different correct actions (e.g., seeing a red button may trigger different sub-actions depending on whether it was pressed twice or five times before); video-conditioned tasks provide a piece of historical video frames + paired proprioception at the initial step, but only the current frame during execution; action chunk training length is 20, execution length is 16; evaluation involves 50 episodes per task, max 1300 steps, with results averaged over the last 3 ckpt × 3 seeds (9 runs total); outside baselines like π0.5, π0.5 w/ past actions, SAM2Act+, and MemER are included for comparison.
    - **Design Motivation**: Many tasks in previous "memory benchmarks" could actually be solved with the current frame, allowing baselines to achieve high scores. This benchmark cuts off such shortcuts at the source. The unified evaluation protocol makes the horizontal comparison of 14 variants truly credible.

### Loss & Training
All models use the native flow matching action diffusion loss of π0.5 for multi-task joint training. Non-recurrent memory variants use batch=64, while recurrent variants (TTT/RMT) are reduced to batch=16 due to VRAM constraints. Training is unified at 80k steps. For symbolic memory, QwenVL is supervised fine-tuned on sub-goal annotations of 1600 demonstrations, while Gemini relies solely on prompt engineering.

## Key Experimental Results

### Main Results (Average SR% of representative variants across 16 tasks)
| Model Category | Variant (Memory + Integration) | Avg SR (%) |
|------|------|------|
| No-Memory Baseline | π0.5 | 17.93 |
| No-Memory Baseline | π0.5 + past actions | 19.73 |
| External SOTA | SAM2Act+ | 21.37 |
| External SOTA | MemER | 42.38 |
| Symbolic (Oracle Upper Bound) | GroundSG + Oracle | **84.08** |
| Symbolic (Actual VLM) | GroundSG + QwenVL | 32.70 |
| Perceptual (Ours Best) | **FrameSamp + Modul** | **44.51** |
| Perceptual | TokenDrop + Modul | 38.04 |
| Perceptual | FrameSamp + Expert | 36.25 |
| Recurrent | TTT + Context | 22.28 |
| Recurrent | RMT + Context | 19.46 |
| Human Reference | Human | 90.50 |

### Ablation Study (FrameSamp+Modul vs. π0.5 Baseline by Suite)
| Suite (Representative Task) | π0.5 Baseline | FrameSamp+Modul | Gain |
|------|---------|---------|---------|
| Counting (StopCube) | 6.67 | 42.00 | +35.33 |
| Permanence (VideoUnmaskSwap) | 18.67 | 24.44 | +5.77 |
| Reference (VideoRepick) | 0.44 | 30.44 | +30.00 |
| Imitation (RouteStick) | 4.67 | 66.67 | +62.00 |

### Key Findings
- **No Silver Bullet**: None of the 14 variants lead across all 4 types of memory. Symbolic memory is strong in tasks with "high-level discrete logic" like Counting/Permanence but weak in continuous action imitation like Imitation; perceptual memory shows the opposite.
- **Perceptual + AdaLN Modulator is the best overall**: FrameSamp+Modul (44.51%) outperforms all other trainable variants, gaining 26.6 points over the memory-less π0.5, validating that "directly conditioning the action path on memory" is more efficient than "stuffing it into the prompt."
- **Recurrent Memory Lags Significantly**: Compressing history into fixed-length hidden states via TTT/RMT loses too much visual detail, with an average SR below 23, performing worse than simply retaining visual tokens. This suggests current SSM-style recurrent representations are not yet strong enough.
- **GroundSG+Oracle 84.08 shows a high upper bound**: As long as sub-goal information is accurate (including $[x, y]$ coordinates), simple sub-goal concatenation can approach human levels. The bottleneck lies entirely in the sub-goal predictor, not the VLA itself—using a VLM as a "memory generator" is a promising direction.
- **MemER 42.38 vs. SAM2Act+ 21.37**: MemER, which mixes "perceptual keyframes + symbolic sub-goals," doubles the performance of SAM2Act+ (pure perceptual memory bank), again indicating that "language-visual hybrid" memory designs are more robust than single-modality ones.

## Highlights & Insights
- Returning "memory" from an overused adjective to deconstructible cognitive dimensions (when/where/what/how) is a rare scientific attempt in this field—any subsequent memory method can draw a radar chart on these four dimensions.
- The "Representation × Integration" 2D ablation matrix orthogonalizes the model design space and serves as a general methodology: any work "adding a new module to an existing model" should compare across such a grid rather than just reporting numbers for a single combination.
- The finding that FrameSamp+Modul is superior to Context and Expert suggests that in flow matching action heads, AdaLN modulation is more suitable for "non-semantic conditional signals" than prompt concatenation. This has transferable value for designing DiT-style diffusion action models.
- Plotting both human performance (90.5) and the Oracle upper bound (84.08) reveals a 46-point gap between current best VLAs and humans, while showing that if the sub-goal predictor were perfect, the gap would close to just 6 points. This clearly points subsequent research toward "improving the precision of sub-goal VLMs."

## Limitations & Future Work
- The work is entirely within ManiSkill simulation and lacks real-world robot validation; specifically, performance in Permanence-type tasks may drop significantly under real occlusion and sensor noise.
- All 16 tasks are single-arm tabletop scenarios, lacking more complex memory requirements such as mobile manipulation, dual-arm collaboration, or human-robot interaction.
- The 512-token memory budget is fixed; the "budget-performance" curve was not studied. Different tasks may require different budgets.
- Recurrent memory implementations (TTT, RMT) are relatively conservative; stronger recent SSM variants like Mamba-2 or Griffin were not tested, and their potential may be underestimated.
- Sub-goal labeling relies on simulator ground truths; porting to real-world scenarios would require a more generalized labeling pipeline.

## Related Work & Insights
- **vs. MemoryBench**: The latter has only 3 spatial tasks and is nearly saturated; RoboMME covers 16 non-Markovian tasks across 4 memory types, with a scale an order of magnitude larger.
- **vs. MIKASA-Robo**: MIKASA has short tasks and few demonstrations, mainly targeting RL; RoboMME provides 770k high-quality demonstrations to support large-scale imitation learning.
- **vs. MemoryVLA / SAM2Act / MemER**: These methods each report results on custom tasks; RoboMME is the first to compare them horizontally with 14 new variants under the same benchmark and backbone.
- **vs. ContextVLA / UniVLA**: They use simple concatenation of past frames or actions; this work further refines "integration methods" into three categories (Context/Modulator/Expert) and systematically compares them, finding Modulator to be significantly superior.

## Rating
- Novelty: ⭐⭐⭐⭐ The cognitive orientation of the benchmark design plus the orthogonalization of the ablation matrix are of high methodological value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 variants × 16 tasks × 9 runs + 4 external baselines—essentially everything that should be tested has been tested.
- Writing Quality: ⭐⭐⭐⭐ Task definitions are clear and table information density is high; lacks some visual analysis of failure cases.
- Value: ⭐⭐⭐⭐⭐ Provides the first truly usable standardized evaluation platform for "memory-enhanced robotic policies"; impact is likely to grow over time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](../../ICLR2026/robotics/memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[ICML 2026\] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[ICML 2026\] DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics](dlo-lab_benchmarking_deformable_linear_object_manipulations_with_differentiable_.md)

</div>

<!-- RELATED:END -->
