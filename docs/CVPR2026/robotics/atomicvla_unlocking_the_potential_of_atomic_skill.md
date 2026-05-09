---
title: >-
  [Paper Note] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
description: >-
  [CVPR 2026][Robotics][VLA] AtomicVLA proposes a unified planning-execution framework that adaptively switches between Think and Act modes to generate task chains and atomic skill abstractions. Using a Skill-Guided MoE (SG-MoE), it builds a scalable atomic skill expert library, surpassing π₀ by 10% on LIBERO-LONG, exceeding baselines by 21% in real-world continual learning, with forgetting as low as 1.3%.
tags:
  - CVPR 2026
  - Robotics
  - VLA
  - atomic skills
  - mixture of experts
  - continual learning
  - long-horizon task planning
date: 2026-05-08
content_hash: b89d083013097fc3
---

# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots

**Conference**: CVPR 2026
**arXiv**: [2603.07648](https://arxiv.org/abs/2603.07648)
**Code**: [https://zhanglk9.github.io/atomicvla-web/](https://zhanglk9.github.io/atomicvla-web/)
**Area**: Robotics / Embodied Intelligence
**Keywords**: VLA, atomic skills, mixture of experts, continual learning, long-horizon task planning

## TL;DR
AtomicVLA proposes a unified planning-execution framework that adaptively switches between Think and Act modes to generate task chains and atomic skill abstractions. Using a Skill-Guided MoE (SG-MoE), it builds a scalable atomic skill expert library, surpassing π₀ by 10% on LIBERO-LONG, exceeding baselines by 21% in real-world continual learning, with forgetting as low as 1.3%.

## Background & Motivation

### State of the Field

**State of the Field**: Existing VLA models train a single action decoder on mixed data, suffering from two major issues: (1) long-horizon tasks require task decomposition and planning, but modular decoupling (VLM planner + VLA controller) leads to upstream-downstream desynchronization; (2) single-decoder mixed training causes inter-skill interference, and incrementally learning new skills leads to catastrophic forgetting. The core challenge is: how to achieve task planning, precise execution, and scalable continual skill learning within a unified framework?

### Mechanism

**Paper Goals**: How to construct an end-to-end VLA framework that adaptively performs high-level task planning and atomic skill decomposition, while enabling efficient multi-skill learning and forgetting-free continual expansion through a modular expert library?

## Method

### Overall Architecture
Built upon π₀/π₀.5. At each timestep, the model adaptively predicts a [think] or [act] token. In [think] mode, it generates a task chain $C_{0-k}$, current progress $C_t$, and atomic skill abstractions $\sigma$ (e.g., pick/place/open); in [act] mode, it activates the corresponding skill expert based on the latest $\sigma$ to generate an action chunk. The SG-MoE action decoder consists of shared experts (retaining π₀'s general capability) plus multiple atomic skill experts.

### Key Designs
1. **Unified Think-Act Architecture**: Rather than a two-stage external VLM + VLA pipeline, a single VLM adaptively decides whether to plan or execute. Think mode is triggered at task initiation or sub-skill transitions to output task chains and atomic skill abstractions; otherwise, Act mode directly generates actions. Key advantage: planning and execution share the same representation space, avoiding information loss across modules.

2. **Skill-Guided MoE (SG-MoE)**: Each atomic skill $\sigma$ is mapped to a fixed high-dimensional embedding vector $Z_\sigma$ (inspired by noise schedule encodings in diffusion models), and a router selects the top-1 skill expert based on $Z_\sigma$. The output is a weighted combination of the shared expert and the selected expert: $F_{\text{out}} = (1-w_k) \cdot F_{\text{share}}(x_t) + w_k \cdot F_k(x_t)$. Unlike standard MoE, routing is skill-level rather than token-level — all tokens within the same skill phase are routed to the same expert, ensuring intra-skill consistency. Ablations show SG-MoE (95.2%) substantially outperforms standard MoE (88.6%) and timestep-level MoDE (89.5%).

3. **Continual Learning via Skill Expansion**: When a new skill is introduced, only a new expert is added along with an expanded router (initialized from the original router weights, with small random values for new branches). Only the new expert and router parameters are trained; all existing experts remain frozen. Experiments show that after learning new skills, π₀.5 loses an average of 15% on old skills (stack drops by up to 20%), while AtomicVLA* loses only 1.3%.

### Loss & Training
Training follows π₀'s flow matching objective. Atomic skill annotations are generated automatically via principal axis analysis (PCA on end-effector trajectory translation/rotation/gripper state changes), validated by InternVideo2.5. Five experts are used for LIBERO and eight for CALVIN. Training runs for 100K steps (8×H200 GPUs), and 30K steps for real-world experiments.

## Key Experimental Results

| Benchmark | Metric | AtomicVLA(*) | π₀ / π₀.5 | Gain |
|--------|------|------|----------|------|
| LIBERO-LONG | SR(%) | **95.2** | 85.2 (π₀) | +10.0 |
| LIBERO Avg | SR(%) | **96.6/97.8** | 94.2 (π₀) | +2.4 |
| CALVIN ABC-D | Avg Len | **4.09/4.27** | 3.87/4.02 | +0.22/+0.25 |
| Real · Long-Horizon (3 tasks) | Avg SR(%) | **56.7/63.3** | 36.7/45.0 | +20/+18.3 |
| Real · Continual Learning (5 tasks) | Avg SR(%) | **82** | 61 (π₀.5 CL) | +21 |
| Real · Forgetting | ΔAvg(%) | **-1.3** | -15.0 (π₀.5) | Significant improvement |

### Ablation Study
- SG-MoE vs. standard MoE vs. MoDE vs. no MoE: 95.2% vs. 88.6% vs. 89.5% vs. 85.2% (LIBERO-LONG)
- Skill-level routing substantially outperforms token-level/timestep-level routing, ensuring consistency within each skill execution phase
- Continual learning: after learning new skills, π₀.5 drops 20% on stack; AtomicVLA* shows negligible degradation (close even improves from 70% to 80%)
- Mixed-training interference: open drawer does not require gripper closing, interfering with grasping tasks; SG-MoE effectively isolates this
- Error recovery: when a sub-skill fails, the model can automatically re-plan and retry

## Highlights & Insights
- The unified Think-Act framework resolves the upstream-downstream misalignment of modular approaches, with adaptive switching that is both elegant and practical
- MoE is reinterpreted from general-purpose routing to "skill modularity" — each expert corresponds to a semantically well-defined atomic skill, yielding strong interpretability
- The continual learning scheme is remarkably simple (add expert + expand router), achieving only 1.3% forgetting (in stark contrast to π₀.5's -15%)
- The principal axis analysis pipeline for automatic atomic skill annotation reduces reliance on manual labeling
- On real robots, the model jointly trains on three long-horizon tasks and consistently outperforms π₀.5

## Limitations & Future Work
- Depends on the VLM's planning accuracy — incorrect atomic skill abstractions propagate errors to execution
- New skills still require large demonstration datasets for imitation learning; RL and few-shot learning remain unexplored
- Atomic skill granularity is fixed (Pick/Place/Open, etc.); finer-grained or more abstract skill hierarchies are not explored
- Inference latency: Think mode ~104ms + Act mode ~92ms (5 experts), totaling ~200ms

## Related Work & Insights
- **π₀/π₀.5**: Unified VLA but with a single decoder; mixed training causes inter-skill interference. AtomicVLA uses SG-MoE for skill isolation.
- **MoDE**: Token-level MoE denoiser with no semantic correspondence between experts. AtomicVLA's skill-level routing is more effective (+5.7%).
- **Hi-Robot/OneTwoVLA et al.**: Also pursue unified planning-execution, but lack skill modularization, making continual learning difficult.
- **LOTUS et al. (continual learning)**: Unsupervised skill discovery but non-VLA architectures; AtomicVLA realizes a scalable skill library within the VLA framework.

## Related Work & Insights
- "Atomic skills" serve as an intermediate abstraction layer for VLA, complementary to the "latent reasoning" in Fast-ThinkAct — one leverages textual skill abstractions, the other continuous latent representations.
- The skill-level MoE routing design is generalizable to scaling research in multi-task VLA.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of unified Think-Act, skill-level MoE, and continual expansion is highly complete and original
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers LIBERO, CALVIN, real-world Franka long-horizon tasks, continual learning, and ablations comprehensively
- Writing Quality: ⭐⭐⭐⭐ Clear architecture, rich visualizations, and sufficient technical detail
- Value: ⭐⭐⭐⭐⭐ Provides practical solutions for VLA scalability and continual learning

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)
- [\[CVPR 2026\] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures.md)

<!-- RELATED:END -->
