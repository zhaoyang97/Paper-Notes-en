---
title: >-
  [Paper Note] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies
description: >-
  [ICML 2026][Robotics & Embodied AI][VLA] RoboMME is the first to systematically map "temporal/spatial/object/procedural" memory from human cognition to 16 long-horizon robotic manipulation tasks (770k high-quality timesteps). By performing a systematic ablation of 14 "memory representation × integration method" combinations on a π0.5 base, it concludes that "
tags:
  - ICML 2026
  - Robotics & Embodied AI
  - VLA
  - π0.5
date: 2026-05-08
content_hash: 6dbf6ad3daae6769
---
# RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies

**Conference**: ICML 2026  
**arXiv**: [2603.04639](https://arxiv.org/abs/2603.04639)  
**Code**: https://robomme.github.io/  
**Area**: Robotics  
**Keywords**: Memory augmentation, VLA, Robot benchmark, Long-horizon manipulation, π0.5

## TL;DR
RoboMME is the first to systematically map "temporal/spatial/object/procedural" memory from human cognition to 16 long-horizon robotic manipulation tasks (770k high-quality timesteps). By performing a systematic ablation of 14 "memory representation × integration method" combinations on a π0.5 base, it concludes that "Perceptual Memory + AdaLN Modulator" currently offers the best comprehensive trade-off.

## Background & Motivation
**Background**: While mainstream manipulation benchmarks like LIBERO, CALVIN, RLBench, and SimplerEnv involve temporal sequences, the vast majority of tasks are actually "Markovian"—the current frame and instruction are sufficient to predict the next action, and past observations can be discarded. Consequently, almost all VLAs (π0.5, RoboVLM, etc.) achieve high SR on these benchmarks without truly examining memory capabilities.

**Limitations of Prior Work**: The few works intentionally examining memory take divergent paths: MemoryBench only covers three nearly-solved spatial tasks; MIKASA-Robo tasks are too short and lack high-quality demonstrations; and memory-augmented models like HistRISE / MemoryVLA / ContextVLA / RoboMamba each use different backbones and evaluation protocols, making it impossible to horizontally compare which "memory design" is superior.

**Key Challenge**: (1) The lack of a benchmark where tasks are truly non-Markovian and sufficiently large-scale. (2) The lack of an experimental framework to systematically compare all mainstream memory architectures under a fixed backbone and data budget. These gaps leave the "memory augmentation" field in a state where it "appears to be progressing, but it is unclear which approach actually wins."

**Goal**: (i) Design a large-scale manipulation benchmark driven by cognitive theory that is explicitly non-Markovian and covers four types of memory requirements. (ii) Conduct a complete orthogonal ablation of symbolic/perceptual/recurrent memory representations and three integration methods under the same π0.5 base and data budget.

**Key Insight**: The classic Atkinson-Shiffrin memory model divides long-term memory into procedural and declarative, with declarative memory further split into episodic (temporal, spatial, object) and semantic. The authors use these four dimensions as the axes for task design, creating a task suite for each to ensure the benchmark covers a cognitively sound "memory stimulus space."

**Core Idea**: By organizing tasks by cognitive dimensions (when/where/what/how) and models by a "memory representation × integration mechanism" 2D matrix, the study yields interpretable conclusions on "which design is effective for which type of task" rather than just "yet another SOTA."

## Method
The work presents two independent but coupled products: the RoboMME benchmark (16 tasks, 4 suites, 1600 demonstrations, 770k timesteps) and the MME-VLA model suite (14 memory variants built on π0.5). The benchmark provides a strictly non-Markovian evaluation environment, while the model suite provides controlled orthogonal ablations.

### Overall Architecture
The benchmark uses ManiSkill simulation with a 7-DoF Franka Panda, featuring dual $256 \times 256$ cameras (front/wrist) and a joint-end-effector dual action space. Each task includes 100 episodes generated via trajectory replay. To improve failure-recovery behavior, 5% key waypoint noise is injected into trajectories followed by recovery. The models use a fixed π0.5 backbone, a memory budget of 512 tokens, and 80k training steps. Evaluations are unified under 50 episodes × 3 seeds × last-3 ckpts to converge all variables onto the "memory design" itself.

### Key Designs

**1. Cognitive-oriented 4-dimensional task classification (Counting / Permanence / Reference / Imitation): Decoupling "memory" into when/where/what/how**

Previous memory benchmarks either tested one type or mixed multiple types, making it impossible to pinpoint "where the model lacks memory." RoboMME borrows the Atkinson-Shiffrin model to create four task suites for each cognitive dimension, with 4 tasks of increasing difficulty per dimension. All are designed such that Markovian policies will fail. The Counting suite tests temporal memory (e.g., *PickXTimes* requires repeating a grasp a specific number of times); the Permanence suite tests spatial memory (e.g., *VideoUnmask/ButtonUnmask* requires finding targets using color memory when blocks are occluded); the Reference suite tests object memory (e.g., *PickHighlight* requires picking a briefly highlighted block); and the Imitation suite tests procedural memory (e.g., *MoveCube/InsertPeg* requires reproducing a grasp pattern or trajectory after watching a video demo).

By slicing by cognitive dimension, the strength/weakness profile of each model can be directly observed—any subsequent memory method can draw a radar chart across these four dimensions for diagnosis and improvement.

**2. MME-VLA Model Suite: An orthogonal ablation matrix of three memory representations × three integration methods**

RoboMME locks all variables to π0.5 with a 512-token memory budget, varying only the "how to store" and "how to use" axes to create 14 variants. Representations are categorized into three paths: Symbolic memory compresses history into natural language subgoals (SimpleSG for descriptions; GroundSG adds front-view coordinates $[x, y]$, generated by Gemini-2.5-Pro / fine-tuned Qwen3-VL-4B / Oracle ground truth). Perceptual memory retains history as visual tokens, using token dropping (redundancy removal by RGB difference) or frame sampling. Recurrent memory uses TTT (online weight updates) or RMT (learned memory slots) to compress sequences into fixed-length hidden states. Integration methods include: Memory-as-Context (direct concatenation to VLM input), Memory-as-Modulator (AdaLN scales/shifts action features via multi-head attention), and Memory-as-Expert (a memory expert path with blockwise causal attention).

Fixing the backbone and budget ensures the scientific integrity of the work—performance differences can be cleanly attributed to the "memory design" itself rather than backbone strength or data volume.

**3. Strict non-Markovian data construction and unified evaluation protocol: Severing shortcuts and locking confounding variables**

In many memory benchmarks, tasks can be solved with the current frame, allowing memoryless baselines to score high. RoboMME ensures "same observation + different history $\rightarrow$ different correct actions" (e.g., seeing a red button might trigger different sub-actions depending on whether it was pressed 2 or 5 times previously). Video-conditioned tasks only provide historical frames and proprioception at the initial step; thereafter, only the current frame is provided, forcing the model to use history. The evaluation locks confounding variables: action chunk training length of 20, execution length of 16, 50 episodes per task, max 1300 steps, with results averaged over 9 runs (3 ckpt × 3 seeds), using π0.5, π0.5 w/ past actions, SAM2Act+, and MemER as external baselines.

### Loss & Training
All models use the native π0.5 flow matching action diffusion loss for multi-task joint training. Non-recurrent variants use batch=64, while recurrent variants (TTT/RMT) use batch=16 due to VRAM constraints. Training lasts 80k steps. The QwenVL for symbolic memory is supervised fine-tuned on subgoal annotations from 1600 demonstrations, while Gemini relies solely on prompt engineering.

## Key Experimental Results

### Main Results (Representative variants, average SR% across 16 tasks)

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

### Ablation Study (FrameSamp + Modul vs. π0.5 Baseline by Suite)

| Suite (Example Task) | π0.5 Baseline | FrameSamp + Modul | Gain |
|------|---------|---------|---------|
| Counting (StopCube) | 6.67 | 42.00 | +35.3 |
| Permanence (VideoUnmaskSwap) | 18.67 | 24.44 | +5.8 |
| Reference (VideoRepick) | 0.44 | 30.44 | +30.0 |
| Imitation (RouteStick) | 4.67 | 66.67 | +62.0 |

### Key Findings
- **No Silver Bullet**: None of the 14 variants leads across all 4 memory types. Symbolic memory is strong in "high-level discrete logic" tasks like Counting/Permanence but weak in continuous action imitation; Perceptual memory shows the opposite trend.
- **Perceptual + AdaLN Modulator is Overall Best**: FrameSamp + Modul (44.51 SR) outperforms all other trainable variants, gaining 26.6 points over memoryless π0.5, validating that directly conditioning the action path on memory via AdaLN is more efficient than "prompt stuffing."
- **Recurrent Memory Lags Behind**: Compressing history into fixed-length hidden states (TTT/RMT) loses too much visual detail, resulting in average SR < 23. This suggests current SSM-style recurrent representations are not yet strong enough for this domain.
- **GroundSG + Oracle at 84.08 shows a high ceiling**: Simple subgoal concatenation can approach human levels if subgoal information is precise (including $[x, y]$ coordinates). The bottleneck lies in the subgoal predictor, not the VLA—positioning VLMs as "memory generators" is a promising direction.
- **MemER (42.38) vs. SAM2Act+ (21.37)**: Mixing perceptual keyframes with symbolic subgoals (MemER) doubles the performance of a pure perceptual bank (SAM2Act+), showing that "vision-language hybrid" designs are more robust.

## Highlights & Insights
- Deconstructing "memory" from an overused adjective into cognitive dimensions (when/where/what/how) is a scientific attempt rare in the field—allowing for radar chart diagnostics.
- The "Representation × Integration" ablation matrix orthogonalizes the model design space, serving as a general methodology: any work adding a new module should compare across such a grid.
- The discovery that FrameSamp + Modul outperforms Context and Expert suggests that for flow matching action heads, AdaLN modulation is better suited for "non-semantic conditioning signals" than prompt concatenation—a transferable insight for DiT-style diffusion policy design.
- Benchmarking human performance (90.5) and Oracle upper bounds (84.08) reveals a 46-point gap between current VLAs and humans, but suggests this gap shrinks to 6 points with a perfect subgoal predictor, clearly pointing research toward "subgoal VLM precision."

## Limitations & Future Work
- Entirely simulation-based (ManiSkill), lacking real-world validation; Permanence tasks may drop significantly under real-world occlusion and sensor noise.
- The 16 tasks are single-arm tabletop scenarios; more complex memory needs like mobile manipulation, bimanual coordination, and HRI are missing.
- The 512-token memory budget is fixed; the "budget-performance" curve remains unexplored.
- Recurrent implementations (TTT, RMT) are relatively conservative, not testing stronger recent SSM variants like Mamba-2 or Griffin.
- Subgoal annotation depends on simulator ground truth; porting to real scenarios requires a more generalized annotation pipeline.

## Related Work & Insights
- **vs. MemoryBench**: The latter has only 3 spatial tasks and is close to saturating; RoboMME covers 16 non-Markovian tasks across 4 categories and is an order of magnitude larger.
- **vs. MIKASA-Robo**: MIKASA has shorter tasks and fewer demos, aimed at RL; RoboMME provides 770k high-quality demos for large-scale imitation learning.
- **vs. MemoryVLA / SAM2Act / MemER**: These reported results on custom tasks; RoboMME evaluates them horizontally under a unified benchmark and backbone.
- **vs. ContextVLA / UniVLA**: While they use simple concatenation of past frames/actions, this work refines integration into Context/Modulator/Expert and finds Modulator to be significantly superior.

## Rating
- Novelty: ⭐⭐⭐⭐ The cognitive orientation and orthogonal ablation matrix are methodologically valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 variants × 16 tasks × 9 runs + 4 external baselines—essentially covers everything.
- Writing Quality: ⭐⭐⭐⭐ Definitions are clear and table density is high, though it lacks some visual analysis of failure cases.
- Value: ⭐⭐⭐⭐⭐ Provides the first truly usable standardized evaluation platform for memory-augmented robotic policies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[ICML 2026\] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)
- [\[CVPR 2026\] FM-Steer: Enhance Generalist Policies with Value-Guided Cascaded Denoising](../../CVPR2026/robotics/fm-steer_enhance_generalist_policies_with_value-guided_cascaded_denoising.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[CVPR 2026\] OctoNav: Towards Generalist Embodied Navigation](../../CVPR2026/robotics/octonav_towards_generalist_embodied_navigation.md)

</div>

<!-- RELATED:END -->
