---
title: >-
  [Paper Note] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning
description: >-
  [NeurIPS 2025][Robotics][Vision-Language-Action Model] AutoVLA integrates physical action tokens directly into a pretrained VLM (Qwen2.5-VL-3B), equips the model with fast/slow dual-thinking modes via SFT, and applies GRPO reinforcement fine-tuning to enable adaptive reasoning switching and optimize planning performance. The approach achieves competitive end-to-end driving performance across four major autonomous driving benchmarks: nuPlan, Waymo, nuScenes, and CARLA.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Vision-Language-Action Model"
  - "End-to-End Autonomous Driving"
  - "Action Tokenization"
  - "Reinforcement Fine-Tuning"
  - "Adaptive Reasoning"
date: 2026-05-08
content_hash: 2f4ca0ac00927114
---

# AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2506.13757](https://arxiv.org/abs/2506.13757)  
**Code**: [https://autovla.github.io/](https://autovla.github.io/)  
**Area**: Autonomous Driving / Multimodal VLM
**Keywords**: Vision-Language-Action Model, End-to-End Autonomous Driving, Action Tokenization, Reinforcement Fine-Tuning, Adaptive Reasoning

## TL;DR

AutoVLA integrates physical action tokens directly into a pretrained VLM (Qwen2.5-VL-3B), equips the model with fast/slow dual-thinking modes via SFT, and applies GRPO reinforcement fine-tuning to enable adaptive reasoning switching and optimize planning performance. The approach achieves competitive end-to-end driving performance across four major autonomous driving benchmarks: nuPlan, Waymo, nuScenes, and CARLA.

## Background & Motivation

**Background**: End-to-end autonomous driving uses a unified model to map directly from sensors to actions, avoiding the error accumulation inherent in modular pipelines. The world knowledge and reasoning capabilities of recent VLMs have made VLA models a prominent research direction; however, existing approaches face fundamental challenges in the physical feasibility of action generation and reasoning efficiency.

**Limitations of Prior Work**: (1) **Infeasible action generation**: Some methods (EMMA, OpenEMMA) generate waypoint coordinates in text format directly from VLMs, but language models have inherent limitations in precise numerical reasoning, leading to physically infeasible trajectories or mode collapse. Intermediate representations such as meta-actions (AlphaDrive, SENNA) or latent tokens (CarLLaVa, Orion) have been introduced to address this, but at the cost of disrupting end-to-end optimization or increasing architectural complexity. (2) **Rigid reasoning strategies**: Most methods adopt a fixed reasoning depth—either always generating lengthy CoT (wasteful for straight-road scenarios, requiring 10+ seconds) or skipping reasoning entirely (poor performance in complex scenarios). DriveVLM supports dual modes but relies on two separate models.

**Key Challenge**: How to simultaneously achieve high-quality semantic reasoning and physically feasible trajectory planning within a single autoregressive model, while adaptively adjusting reasoning depth according to scene complexity.

**Goal**: (1) Embed physically constrained trajectory planning directly into the LM token space; (2) Enable the model to automatically switch between fast thinking (~1s) and slow thinking (~10s); (3) Further align planning performance via RL in the post-training stage.

**Key Insight**: A codebook of 2,048 physical action tokens is constructed via K-disk clustering from real Waymo driving data, where each token corresponds to a 0.5-second feasible vehicle motion $(\Delta x, \Delta y, \Delta\theta)$, casting planning as next-token prediction.

**Core Idea**: Introduce a K-disk clustering-based physical action codebook into the VLM to enable token-level generation of feasible trajectories, combined with GRPO and CoT length penalty to achieve adaptive fast/slow reasoning switching.

## Method

### Overall Architecture

AutoVLA adopts Qwen2.5-VL-3B as its backbone, taking as input a 4-frame image sequence (at 2 Hz) from three front-facing cameras, high-level navigation commands, and ego-vehicle state. The output is a mixed sequence of text tokens (reasoning content) and physical action tokens (trajectory). Training proceeds in two stages: **SFT** trains the dual-mode capability using mixed data with and without CoT; **RFT** optimizes performance and efficiency using GRPO with driving rewards and a CoT length penalty.

### Key Designs

1. **Physical Action Tokenization**:

    - Function: Discretize continuous vehicle trajectories into token sequences natively processable by the language model.
    - Mechanism: 0.5-second motion segments are extracted from real Waymo trajectories and clustered using K-disk clustering ($\delta=0.05$m distance threshold) to select $K=2048$ representative tokens, each encoding $(\Delta x, \Delta y, \Delta\theta)$. These are added to the VLM vocabulary as new tokens (`<action_0>` to `<action_2047>`). During inference, 10 tokens are generated and decoded into a 5-second trajectory. Reconstruction accuracy: ADE = 0.018m, motion coverage = 99.42%, substantially outperforming RT-1 (ADE = 0.101m) and FAST DCT (ADE = 0.028m).
    - Design Motivation: Text-based numerical generation suffers from low precision, physical infeasibility, and slow computation. Action tokens inherently satisfy kinematic constraints, and casting planning as next-token prediction aligns perfectly with the LM paradigm.

2. **Dual-Mode Adaptive Reasoning (Fast/Slow Thinking)**:

    - Function: Adaptively switch reasoning depth according to scene complexity.
    - Mechanism: SFT trains on a mixture of two data types—**fast-thinking** data contains only action tokens (with a brief template indicating no reasoning is needed), while **slow-thinking** data contains structured CoT (scene description → key objects → intent prediction → decision) followed by action tokens. CoT data is automatically annotated by Qwen2.5-VL-72B with a human-verified accuracy of 88.8%. The SFT loss assigns higher weight to CoT samples ($\lambda_{cot}=40$).
    - Design Motivation: Fast thinking takes ~1.07s and slow thinking ~10.52s—a 10× gap that makes fixed strategies unacceptable in real deployment. CoT only outperforms action-only training when the training set exceeds 50k samples (CoT is harmful with insufficient data).

3. **GRPO Reinforcement Fine-Tuning**:

    - Function: Further optimize planning performance and enable adaptive fast/slow switching.
    - Mechanism: For each scene, $G$ groups of outputs are sampled and the within-group relative advantage is computed as $A_i = (r_i - \text{mean}) / \text{std}$. The reward is $r = r_{Driving} - \lambda_r r_{CoT}$, where the driving reward uses PDMS (nuPlan) or ADE (Waymo), and the CoT length penalty suppresses redundant reasoning in simple scenes. LoRA adapters are used with learning rate $3\times10^{-5}$, $\beta=0.04$, trained for 6,000 steps.
    - Design Motivation: SFT models may produce suboptimal trajectories due to accumulated generation errors. GRPO's group sampling naturally handles the multimodal nature of planning (multiple valid trajectories per scene), while the CoT penalty teaches the model to "reason when needed, act directly otherwise."

### Loss & Training

SFT stage: $\mathcal{L}^{SFT} = w_i \cdot (\mathcal{L}_{LM} + \lambda_a \mathcal{L}_{action})$, with $\lambda_a=1$ and $\lambda_{cot}=40$. RFT stage uses the GRPO objective (PPO clip + KL divergence regularization). SFT is trained with FSDP on 8×L40S GPUs for 5 epochs with batch size 32. A separate codebook is used for CARLA experiments due to differing simulator dynamics.

## Key Experimental Results

### Main Results

| Benchmark | Metric | AutoVLA (Post-RFT) | AutoVLA (Best-of-N) | TrajHF | Centaur |
|------|------|------|------|------|------|
| NAVSIM (nuPlan) | PDMS↑ | 89.11 | **92.12** | 93.95 | 92.10 |
| NAVSIM | Collision↑ | 98.41 | 99.14 | 99.30 | 99.23 |
| NAVSIM | TTC↑ | **98.04** | 97.12 | 98.02 | 97.17 |
| Bench2Drive (CARLA) | Driving Score↑ | **78.84** | - | - | - |
| Bench2Drive | Success Rate↑ | **57.73%** | - | - | - |

### Ablation Study

| Configuration | PDMS↑ | Runtime | Notes |
|------|-------|---------|------|
| SFT (Action only) | 79.28 | 1.07s | No reasoning; fast but limited performance |
| SFT (CoT) | 80.54 | 10.52s | Always reasons; slow but stronger |
| Post-RFT | **89.11** | **3.49s** | Adaptive switching |
| Gain vs. SFT CoT | **+10.6%** | **−66.8%** | Substantial gains in both performance and efficiency |

| Action Generation | PDMS↑ | Avg. L2↓ | Collision Rate↓ | Runtime |
|-------------|-------|---------|---------|---------|
| Text Waypoint | 71.31 | 0.89m | 0.36% | 7.65s |
| FAST (DCT) | 67.63 | - | - | - |
| K-disk Token (Ours) | **80.54** | **0.70m** | **0.31%** | **3.95s** |

### Key Findings

- **RFT delivers the largest performance leap**: PDMS improves by 10.6% (80.54→89.11) while runtime decreases by 66.8% (10.52s→3.49s). The core mechanism is that the CoT length penalty teaches the model to automatically switch to fast thinking in simple scenes.
- Physical action tokenization vs. text waypoints: PDMS 80.54 vs. 71.31, runtime 3.95s vs. 7.65s—a comprehensive improvement that quantifies the inherent limitations of language models in precise numerical reasoning.
- CoT advantages become more pronounced with larger data: with fewer than 50k training samples, CoT underperforms action-only training, indicating that sufficient data is required to learn structured reasoning.
- Pretraining and cross-dataset training significantly benefit long-tail Waymo scenarios, improving RFS from 56.5 to 64.8.
- In closed-loop testing (CARLA), AutoVLA achieves a 57.73% success rate, surpassing Orion (54.62%), validating the approach in interactive environments.

## Highlights & Insights

- **Action codebook as the missing link for VLA in driving**: K-disk clustering at $K=2048$ achieves 99.42% motion coverage, demonstrating that 2,048 tokens suffice to represent the vast majority of driving behaviors. This "continuous space → discrete token" approach preserves physical feasibility while seamlessly integrating into the LM autoregressive paradigm.
- **GRPO + CoT penalty = "reason only when necessary"**: This adaptive reasoning strategy is highly practical—straight-road driving requires no CoT, while construction zone navigation genuinely does. The 10× runtime gap makes fixed reasoning strategies unacceptable in deployment.
- **72B→3B knowledge distillation pipeline**: The approach of using Qwen2.5-VL-72B to automatically generate reasoning annotations for a 3B student model is broadly reusable; the 88.8% human-verified accuracy confirms its reliability.

## Limitations & Future Work

- Inference speed remains far below real-time requirements (fast mode ~1 Hz, slow mode ~0.1 Hz); the authors identify quantization and model compression as priority directions.
- Only three front-facing cameras are used, leaving rear and side-rear perception uncovered, which limits comprehensive scene understanding (e.g., reversing, blind-spot lane changes).
- The action codebook is clustered from a specific dataset; cross-vehicle-type or cross-domain transfer requires rebuilding the codebook.
- Best-of-N inference requires an oracle scorer, which is unavailable in real deployment—alternative self-evaluation mechanisms are needed.
- CoT reasoning quality is entirely dependent on the distillation source model (72B), preventing the generation of reasoning that exceeds the source model's capability.

## Related Work & Insights

- **vs. EMMA/OpenEMMA**: These methods generate text-format waypoints directly from VLMs, suffering from physical infeasibility and mode collapse. AutoVLA's action tokenization addresses both issues fundamentally.
- **vs. DriveVLM**: Also features fast/slow dual processes, but relies on two separate models without end-to-end optimization. AutoVLA achieves adaptive switching within a single model.
- **vs. AlphaDrive**: Also uses GRPO, but only handles high-level meta-actions. AutoVLA extends RFT to low-level trajectory planning.
- **vs. Orion**: Integrates a generative planner into the VLM at the cost of architectural complexity. AutoVLA achieves a comparable effect through the action codebook with a simpler structure.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of physical action tokens, adaptive reasoning, and GRPO fine-tuning is pioneering in the autonomous driving VLA literature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four major benchmarks (nuPlan/Waymo/nuScenes/CARLA) with both open-loop and closed-loop evaluations and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich illustrations; the paradigm comparison figure (Fig. 2) conveys key distinctions at a glance.
- Value: ⭐⭐⭐⭐ Provides a practical unified framework for VLA in autonomous driving; action tokenization and adaptive reasoning are broadly influential ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[CVPR 2025\] TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](../../CVPR2025/robotics/tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)
- [\[CVPR 2026\] End-to-End Language-Action Model for Humanoid Whole Body Control](../../CVPR2026/robotics/end-to-end_language-action_model_for_humanoid_whole_body_control.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](profit_a_specialized_optimizer_for_deep_fine_tuning.md)
- [\[ICLR 2026\] OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning](../../ICLR2026/robotics/onetwovla_a_unified_vision-language-action_model_with_adaptive_reasoning.md)

</div>

<!-- RELATED:END -->
