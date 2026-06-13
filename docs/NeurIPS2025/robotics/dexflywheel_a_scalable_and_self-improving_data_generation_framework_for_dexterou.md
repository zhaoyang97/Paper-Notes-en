---
title: >-
  [Paper Note] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation
description: >-
  [NeurIPS 2025][Robotics][Dexterous Manipulation] This paper proposes DexFlyWheel, a dexterous manipulation data generation framework that starts from a single human demonstration and progressively scales data diversity t…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Dexterous Manipulation"
  - "Data Flywheel"
  - "Imitation Learning"
  - "Residual Reinforcement Learning"
  - "Sim-to-Real"
date: 2026-05-08
content_hash: 658b18e0c7a2223a
---

# DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2509.23829](https://arxiv.org/abs/2509.23829)  
**Code**: [Project Page](https://DexFlyWheel.github.io)  
**Area**: Robotics
**Keywords**: Dexterous Manipulation, Data Flywheel, Imitation Learning, Residual Reinforcement Learning, Sim-to-Real

## TL;DR

This paper proposes DexFlyWheel, a dexterous manipulation data generation framework that starts from a single human demonstration and progressively scales data diversity through a self-improving loop composed of IL, residual RL, and data augmentation. The framework generates 2,000+ demonstrations across 4 tasks, achieving an average policy success rate of 81.9% and a real-world transfer success rate of 78.3%.

## Background & Motivation

Dexterous manipulation is a core challenge in robotics. The high degrees of freedom of multi-fingered hands and rich contact interactions demand larger, more diverse, and higher-quality datasets. Existing data collection methods suffer from the following bottlenecks:

**Human teleoperation**: Requires substantial human effort, is typically confined to laboratory settings, and scales poorly.

**Motion planning**: Effective for parallel-jaw grippers, but struggles with the high-dimensional action spaces and complex contact dynamics of multi-fingered hands.

**LLM-driven methods**: Capable of generating high-level instructions, but unable to provide finger-level fine-grained control.

**Pure RL methods**: Suffer from exploration difficulty, dependence on reward engineering, and produce non-human-like behaviors that hinder sim-to-real transfer.

**Replay-and-edit methods** (e.g., DexMimicGen): Only apply spatial transformations to existing demonstrations, failing to explore new manipulation strategies and breaking down when object geometry varies significantly.

Core insight: Manipulating different objects typically requires only **minor adjustments** to the manipulation trajectory — human demonstrations should be treated as **behavioral priors** rather than mere replay data.

## Method

### Overall Architecture

DexFlyWheel operates in two phases:

1. **Warm-up Phase**: Starting from a single VR teleoperation demonstration, an initial dataset $\mathcal{D}_1$ is generated via data augmentation.
2. **Self-Improving Data Flywheel Phase**: Multiple iterations $i = \{1, 2, ..., n-1\}$, each executing a closed-loop pipeline.

Each flywheel iteration consists of four steps:
1. Train a base policy $\pi_{\text{base}}^i$ (imitation learning).
2. Train a residual policy $\pi_{\text{res}}^i$ (residual RL).
3. Roll out the combined policy to collect new trajectories $\mathcal{D}_O^i$.
4. Apply data augmentation to expand diversity → $\mathcal{D}_{i+1}$.

### Key Designs

#### 1. VR Teleoperation + Data Augmentation (Warm-up)

Apple Vision Pro is used to track hand/wrist/head poses, collecting **a single** seed demonstration $d_{\text{seed}}$ in simulation.

The data augmentation module $\mathcal{A}_{\text{EP}}$, extended from MimicGen, supports multi-dimensional augmentation:
- **Environment diversity**: Lighting conditions and tabletop appearance variations.
- **Spatial diversity**: Object pose variations.
- Implemented via trajectory editing and simulation domain randomization.

#### 2. Base Policy Training (Imitation Learning)

Diffusion Policy is adopted as the base policy. The input state is $s_t = \{s_t^{\text{vis}}, s_t^{\text{obj}}, s_t^{\text{prop}}\}$:
- $s_t^{\text{vis}}$: Camera visual input.
- $s_t^{\text{obj}}$: Object state (6D pose + velocity).
- $s_t^{\text{prop}}$: Robot proprioception (joint angles/velocities + end-effector pose).

The policy outputs an action sequence $(a_t, a_{t+1}, ..., a_{t+H})$, where $H$ is the prediction horizon.

#### 3. Residual Reinforcement Learning

This is the core innovation of the flywheel mechanism. The base policy $\pi_{\text{base}}$ is frozen, and a residual policy $\pi_{\text{res}}$ is trained to generate corrective actions:

$$\tilde{a}_t = a_t + \alpha \cdot \triangle a_t$$

The combined policy is $\pi_{\text{combined}} = \pi_{\text{base}} + \alpha \cdot \pi_{\text{res}}$.

A progressive scheduling scheme is employed during training:

$$\pi_{\text{combined}}(s) = \begin{cases} \pi_{\text{base}}(s) + \alpha \cdot \pi_{\text{res}}(s) & \text{with probability } \epsilon \\ \pi_{\text{base}}(s) & \text{with probability } 1 - \epsilon \end{cases}$$

$\epsilon$ increases linearly from 0 to 1, gradually transferring control from the base policy to the residual policy. The residual policy receives only $s_t^{\text{obj}}$ and $s_t^{\text{prop}}$, focusing on object adaptation rather than learning manipulation from scratch.

#### 4. Rollout + Augmentation Loop

The combined policy is rolled out under randomized object configurations. High-quality trajectories are filtered by task success and then augmented via $\mathcal{A}_{\text{EP}}$ to cover more environment and spatial configurations, generating training data for the next iteration.

### Loss & Training

- **Base policy**: Standard denoising loss of Diffusion Policy.
- **Residual policy**: Standard RL loss (PPO), with task success as the reward signal.
- **Iteration settings**: $i = \{1, 2, 3\}$, generating 20, 100, and 500 trajectories respectively.
- **Simulation platform**: OmniGibson (photorealistic rendering), 80 object categories, 12 environments.

## Key Experimental Results

### Main Results: Data Flywheel Effectiveness

| Task | i=1 Success | i=2 Success | i=3 Success | # Scenes |
|---|---|---|---|---|
| Grasp | 15.0% | 58.0% | **90.0%** | 3960 |
| Pour | 36.1% | 55.6% | **85.8%** | 1440 |
| Lift | 13.9% | 44.4% | **79.4%** | 1560 |
| Handover | 0.8% | 17.5% | **72.5%** | 1200 |
| **Average** | 16.5% | 43.9% | **81.9%** | 2040 |

From i=1 to i=3: object categories increased by 20×, scene count by 214.7×, and success rate by +396.4%.

### Comparison with Baselines

| Method | Grasp | Pour | Lift | Handover | Average |
|---|---|---|---|---|---|
| Human Demo (Default, 20 demos) | 6.1% | 16.7% | 13.9% | 0.8% | 9.4% |
| Human Demo (Enhanced) | 15.0% | 36.1% | 2.5% | 0.0% | 13.4% |
| DexMimicGen (Default, 1 seed) | 30.3% | 38.9% | 28.2% | 28.3% | 31.4% |
| DexMimicGen (Enhanced, 10 seeds) | 50.3% | 44.4% | 43.7% | 42.5% | 45.2% |
| **DexFlyWheel (1 seed)** | **90.0%** | **85.8%** | **79.4%** | **72.5%** | **81.9%** |

Using only a single seed demonstration, DexFlyWheel achieves 81.9%, substantially outperforming DexMimicGen (45.2%), which enjoys a 10× data advantage.

### Data Generation Efficiency

| Method | Time per Trajectory | Total Time for 500 Successful Trajectories | Generation Success Rate |
|---|---|---|---|
| Human Teleoperation | 60s | 12.5h | - |
| DexMimicGen | 15s | 4.4h | 63.0% |
| **DexFlyWheel** | **15s** | **2.4h** | **89.8%** |

### Ablation Study

| Variant | Impact |
|---|---|
| w/o Residual Policy | **Largest performance drop**; object generalization degrades sharply (20 → 8.25 object categories) |
| w/o Data Augmentation $\mathcal{A}_{\text{EP}}$ | Limited environment and spatial diversity |
| w/o Residual + w/o Augmentation | Base policy only; worst overall performance |

### Real-World Deployment

Transferred to a dual-arm Real-Man robot via a Digital Twin:
- **Dual-arm Lift**: 78.3% success rate
- **Handover**: 63.3% success rate

### Key Findings

1. **Pronounced flywheel effect**: Data diversity and policy performance improve jointly in each iteration, forming a virtuous cycle.
2. **Residual RL is the cornerstone**: On average, it improves object generalization by 32.1% ($\pi_{\text{base}} \rightarrow \pi_{\text{combined}}$).
3. **A single demonstration suffices**: The flywheel can be bootstrapped from one human demonstration without requiring large-scale human data collection.
4. **Successful sim-to-real transfer**: Reliable simulation-to-real transfer is achieved via the Digital Twin approach.

## Highlights & Insights

1. **Transplanting LLM self-improvement to robotics**: Inspired by iterative self-improvement in LLMs (RLHF loops), DexFlyWheel applies this paradigm to robotic data generation — an elegant cross-domain knowledge transfer.
2. **Elegant design of residual RL**: Rather than learning manipulation skills from scratch, the residual policy fine-tunes atop imitation learning, aligning with the physical intuition that manipulating different objects requires only minor adjustments.
3. **Stability through progressive scheduling**: Linear scheduling of $\epsilon$ from 0 to 1 prevents the residual policy's early-stage exploration from disrupting the base policy.
4. **Minimal human effort**: Each task requires only 1 demonstration → 2,000+ diverse demonstrations, achieving extremely high data efficiency.

## Limitations & Future Work

1. **Manually designed reward functions**: The residual RL relies on hand-crafted task rewards; future work could explore LLM-driven reward generation.
2. **Absence of tactile feedback**: The current policy and simulation do not incorporate tactile signals, limiting performance on contact-rich tasks.
3. **Limited number of iterations**: Only 3 iterations are demonstrated; whether additional iterations lead to convergence or degradation remains an open question.
4. **OmniGibson sim-to-real gap**: Despite photorealistic rendering, the sim-to-real gap may still constrain transfer for more complex tasks.

## Related Work & Insights

- **Core distinction from DexMimicGen**: DexMimicGen only performs replay-and-edit (spatial transformations) on trajectories and cannot explore new strategies; DexFlyWheel achieves policy-level adaptation via residual RL.
- **Complementarity with pure RL**: Rather than learning from scratch with RL (which suffers from exploration difficulty), the framework combines IL as a prior with RL for fine-tuning — a mutually complementary design.
- **Generality of the data flywheel**: The closed-loop paradigm of IL → residual RL → rollout → augmentation is generalizable to other robotic manipulation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Data flywheel concept is novel; the residual RL + IL combination is elegantly designed)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 tasks × multiple baselines × ablations × real-world deployment — exceptionally comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Addresses the core bottleneck of data scarcity in dexterous manipulation with practical deployment value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [\[NeurIPS 2025\] A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](a_snapshot_of_influence_a_local_data_attribution_framework_f.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)

</div>

<!-- RELATED:END -->
