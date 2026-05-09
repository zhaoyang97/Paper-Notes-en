---
title: >-
  [Paper Note] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing
description: >-
  [NeurIPS 2025][Robotics][Surgical Robotics] This paper presents SutureBot — the first precision-oriented benchmark and goal-conditioned framework for end-to-end autonomous suturing on the da Vinci surgical robot. It releases a high-fidelity dataset of 1,890 demonstrations, achieves 59%–74% improvements in needle insertion accuracy via point-label goal conditioning, and systematically evaluates state-of-the-art VLA models including π0, GR00T N1, OpenVLA-OFT, and multi-task ACT.
tags:
  - NeurIPS 2025
  - Robotics
  - Surgical Robotics
  - Suturing Autonomy
  - Imitation Learning
  - Goal-Conditioned Control
  - VLA Benchmark
date: 2026-05-08
content_hash: f7f1cfb65164bc28
---

# SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing

**Conference**: NeurIPS 2025
**arXiv**: [2510.20965](https://arxiv.org/abs/2510.20965)
**Code**: [Hugging Face Dataset](https://huggingface.co/)
**Area**: Robotics
**Keywords**: Surgical Robotics, Suturing Autonomy, Imitation Learning, Goal-Conditioned Control, VLA Benchmark

## TL;DR

This paper presents SutureBot — the first precision-oriented benchmark and goal-conditioned framework for end-to-end autonomous suturing on the da Vinci surgical robot. It releases a high-fidelity dataset of 1,890 demonstrations, achieves 59%–74% improvements in needle insertion accuracy via point-label goal conditioning, and systematically evaluates state-of-the-art VLA models including π0, GR00T N1, OpenVLA-OFT, and multi-task ACT.

## Background & Motivation

Robotic suturing is a prototypical long-horizon dexterous manipulation task requiring coordinated needle grasping, precise tissue puncture, and safe knot tying. Despite progress on individual sub-tasks, complete end-to-end autonomous suturing has not yet been demonstrated on real hardware. Key bottlenecks include:

**Data scarcity**: Existing public dVRK suturing datasets contain fewer than 200 trajectories in total — far insufficient for modern IL architectures.

**No unified benchmark**: The surgical robotics field lacks a reproducible benchmark for tracking progress in autonomous suturing.

**Missing precision metrics**: Traditional evaluation relies on coarse-grained task completion rates, lacking clinically meaningful precision indicators.

**Unevaluated VLA models**: The performance of recent VLA models (π0, GR00T N1, etc.) on high-precision surgical suturing tasks remains unknown.

Prior methods such as the STAR system achieved precise suturing using dedicated tools combined with computer vision, but rely on custom hardware and lack generalization and error recovery. MPC-based approaches succeed at suture placement but lack flexibility. SurgicAI achieved 50% end-to-end success in simulation but has not been demonstrated on real hardware.

## Method

### Overall Architecture

A hierarchical architecture is employed: a high-level policy (based on Swin Transformer) generates language instructions from visual observations, while a low-level policy (VLA model) receives language instructions, camera images, and goal conditions to output robot action chunks. Suturing is decomposed into three sub-tasks: needle pickup, needle throw, and knot tying.

### Key Designs

1. **Large-Scale Dataset Construction**

   Using the dVRK Si platform, 1,890 demonstrations were collected via a standard teleoperation console:
   - Needle pickup: 628 (including 148 recovery demonstrations)
   - Needle throw: 310 (including 96 recovery demonstrations)
   - Knot tying: 952 (including 210 recovery demonstrations)

   Recovery demonstrations are inspired by DAgger: an initial policy is first deployed to identify common failure modes, followed by collection of additional demonstrations recovering from failure states to successful completion. Data includes synchronized visual streams (wrist camera 640×480@30Hz + stereo endoscope 960×540@30Hz) and kinematic data (6-DoF Cartesian poses, jaw angles, etc.). Diversity is introduced by varying robot joint configurations, RCM positions, suture pad placement, needle initial poses, and camera mounting.

   **Design Motivation**: Recovery demonstrations substantially increase training data diversity, teaching policies to recover from suboptimal states and improving deployment robustness.

2. **Goal-Conditioned Representations**

   Three goal conditioning formats are explored for precision-controlled suturing:
   - **Point Labels**: Opaque blue pixels (insertion point) and green pixels (exit point) overlaid on the endoscope image.
   - **Binary Masks**: A three-channel image encoding insertion and exit masks in separate channels.
   - **Distance Maps**: A three-channel image where the first two channels encode normalized pixel offset vectors $(dx, dy)$ pointing toward the insertion point, and the third channel encodes an intensity heatmap.

   **Design Motivation**: Point labels embed the goal directly into the task image, providing the most explicit and intuitive goal representation. Masks and distance maps, as separate inputs, may impose additional cognitive burden on the model for information integration.

3. **Systematic Multi-VLA Model Evaluation**

   Four low-level policies are compared:
   - **π0**: Pre-trained VLM backbone with a flow-matching action head.
   - **GR00T N1**: Similar architecture but pre-trained primarily on humanoid robot data.
   - **OpenVLA-OFT**: Parallel decoding with L1 regression and FiLM conditioning.
   - **Multi-task ACT**: A non-VLA baseline without a pre-trained VLM backbone.

   ACT and OpenVLA are trained with L1 regression for at least 10,000 steps; π0 and GR00T use MSE with fewer steps and require early stopping to prevent overfitting. All training is conducted on a DGX A100 (8×A100 80GB). Each sub-task is assigned a maximum time limit during evaluation (120s each for needle pickup/throw/knot tying; 60s for thread pull-through).

### Loss & Training

- L1 regression training (ACT, OpenVLA): no fewer than 10,000 steps.
- MSE training (π0, GR00T): fewer steps with early stopping to prevent overfitting.
- High-level policy based on Swin Transformer: trained for 2,000 epochs; best validation epoch at 282.
- FiLM conditioning is used to inject language information.
- Maximum time limits are enforced per sub-task during evaluation.

## Key Experimental Results

### Main Results

Low-level policy performance on the suturing benchmark (10 complete suturing trials):

| Policy | Pickup | Throw | Pull-through | Knot | Insertion Error (mm) | Exit Error (mm) | Time (s) | End-to-End |
|---|---|---|---|---|---|---|---|---|
| ACT | 9/10 | 8/10 | 4/10 | 9/10 | 1.5±0.8 | 2.6±1.2 | 182±58 | **3/10** |
| π0 | 7/10 | 7/10 | 3/10 | 4/10 | 1.9±1.0 | 3.2±2.3 | 348±45 | 0/10 |
| GR00T N1 | 1/10 | 2/10 | 1/10 | 1/10 | 2.3±1.2 | 2.9±0.6 | 388±67 | 0/10 |
| OpenVLA | 0/10 | 0/10 | 0/10 | 0/10 | NA | 2.8 | NA | 0/10 |

### Ablation Study

Effect of goal conditioning format on needle throw precision:

| Policy + Goal Condition | Throw Success | Pull-through Success | Insertion Error (mm) | Exit Error (mm) |
|---|---|---|---|---|
| ACT + Point Label | 9/10 | 7/10 | **1.3±0.9** | 2.0±1.3 |
| ACT + Distance Map | 8/10 | 8/10 | 2.6±1.5 | 2.2±1.8 |
| ACT + Mask | 10/10 | 4/10 | 2.9±1.7 | 3.0±1.0 |
| ACT (no goal) | 10/10 | 9/10 | 3.2±2.2 | 3.6±1.8 |
| π0 + Point Label | 6/10 | 2/10 | **1.0±1.3** | 2.4±1.6 |
| π0 (no goal) | 8/10 | 3/10 | 3.9±2.5 | 3.7±2.5 |

Generalization tests:

| Scenario | Pickup | Throw | Pull-through | Knot | Insertion Error |
|---|---|---|---|---|---|
| ACT Training wound (1) | 9/10 | 8/10 | 4/10 | 9/10 | 1.5±0.8 |
| ACT Unseen wounds (2–6) | 5/10 | 6/10 | 2/10 | 5/10 | 1.2±0.8 |
| ACT Different lighting | 3/10 | 5/10 | 7/10 | 4/10 | 2.2±0.9 |
| π0 Training wound (1) | 7/10 | 7/10 | 3/10 | 4/10 | 1.9±1.0 |
| π0 Unseen wounds (2–6) | 5/10 | 6/10 | 0/10 | 8/10 | 2.0±1.5 |

### Key Findings

1. ACT achieves the best overall sub-task completion rates and is the only policy to succeed end-to-end (3/10).
2. Point-label goal conditioning yields 59%–74% precision improvements; the lowest insertion errors are 1.3 mm (ACT) and 1.0 mm (π0).
3. Models conditioned on point labels exhibit deliberate, hesitant adjustment motions when approaching the target, suggesting improved spatial awareness.
4. π0 pre-training provides limited benefit for dVRK — training from scratch yields only slightly lower performance than fine-tuning from a checkpoint.
5. The high-level policy achieves an F1 of 0.92 with 100% accuracy in task-transition detection, and is not the system bottleneck.

## Highlights & Insights

- First end-to-end autonomous suturing benchmark and dataset demonstrated on real hardware.
- UV-marker-based puncture precision measurement goes beyond conventional coarse-grained task completion evaluation.
- The DAgger-inspired recovery demonstration strategy offers a practical engineering insight.
- The fact that ACT — a compact, task-specific model — outperforms large VLAs on small-scale homogeneous data raises important questions about the role of model scale and pre-training in surgical robotics.

## Limitations & Future Work

- Only 10 trials per experiment, limiting statistical power.
- Target puncture points are currently selected manually; full autonomy requires automated goal specification.
- Generalization to different materials (e.g., real tissue, simulated blood) is not evaluated.
- Clinically relevant metrics (bite depth, tissue trauma, suture tension) are not included.
- Only ACT achieves end-to-end success, leaving a substantial gap before clinical deployment.

## Related Work & Insights

- This work extends the hierarchical surgical learning framework of SRT-H to the highly dexterous task of suturing.
- It complements the simulation-based suturing work of SurgicAI by advancing the problem to real hardware.
- A promising direction: VLA pre-training for surgical scenarios may require data specifically tailored to bimanual manipulation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Significant benchmark and dataset contributions; the goal-conditioning approach is relatively straightforward but effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-model comparison, ablations, generalization tests, and statistical analysis are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized with rigorous metric definitions.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a critical gap by providing the first end-to-end autonomous suturing benchmark in surgical robotics.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[NeurIPS 2025\] A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](a_snapshot_of_influence_a_local_data_attribution_framework_f.md)

<!-- RELATED:END -->
