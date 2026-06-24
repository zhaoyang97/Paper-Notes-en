---
title: >-
  [Paper Note] Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction
description: >-
  [CVPR 2025][Robotics][Self-Correction] This paper proposes the Phoenix framework, which utilizes motion instructions as a bridge to connect the high-level semantic reflection of MLLMs with low-level robotic action correction. By incorporating a dual-process motion adjustment mechanism and a motion-conditioned diffusion policy, Phoenix achieves fine-grained manipulation failure recovery and supports self-improvement through lifelong learning.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Self-Correction"
  - "Motion Instructions"
  - "Diffusion Policy"
  - "Lifelong Learning"
  - "MLLM"
date: 2026-05-08
content_hash: b7ec32759d7e9113
---

# Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction

**Conference**: CVPR 2025  
**arXiv**: [2504.14588](https://arxiv.org/abs/2504.14588)  
**Code**: [https://github.com/GeWu-Lab/Motion-based-Self-Reflection-Framework](https://github.com/GeWu-Lab/Motion-based-Self-Reflection-Framework)  
**Area**: Robotic Manipulation  
**Keywords**: Self-Correction, Motion Instructions, Diffusion Policy, Lifelong Learning, MLLM

## TL;DR

This paper proposes the Phoenix framework, which utilizes motion instructions as a bridge to connect the high-level semantic reflection of MLLMs with low-level robotic action correction. By incorporating a dual-process motion adjustment mechanism and a motion-conditioned diffusion policy, Phoenix achieves fine-grained manipulation failure recovery and supports self-improvement through lifelong learning.

## Background & Motivation

**Background**: Robotic self-correction systems are crucial for achieving robust manipulation. Existing methods fall into two categories: (1) RL methods use reward functions to guide action correction but suffer from unstable training and require task priors; (2) MLLM methods (such as CaP and SayCan) use semantic reflection to decompose failures into subgoals but rely on predefined skill libraries and cannot provide fine-grained action correction.

**Limitations of Prior Work**: Semantic reflection can tell the robot "what to do" (e.g., "insert the coffee pot into the coffee machine"), but it cannot specify "how to correct the physical actions." There exists a massive abstraction gap between high-level semantics and low-level actions—subgoal-level guidance is too coarse to be directly translated into 20Hz joint control signals.

**Key Challenge**: MLLMs possess powerful perception and reasoning capabilities but cannot directly output high-frequency actions; diffusion policies can generate precise actions but lack high-level understanding and error-correction capabilities. There is a lack of an appropriate intermediate layer between the two.

**Goal**: Design an intermediate representation layer (motion instructions) to translate the semantic reflection capabilities of MLLMs into executable, fine-grained action corrections.

**Key Insight**: Motion instructions (e.g., "move arm right with gripper closed") serve as a natural bridge between MLLMs and robot actions—they are abstract enough for MLLMs to comprehend and generate, yet concrete enough for diffusion policies to follow and execute.

**Core Idea**: Connect MLLM semantic reflection with diffusion policy action generation via motion instructions to achieve fine-grained self-correcting manipulation.

## Method

### Overall Architecture

The framework consists of three modules: (1) A dual-process motion adjustment mechanism—comprising a Motion Prediction Module (MPM, fine-tuned from LLaVA-v1.5) that learns to efficiently predict motion instructions from expert data, and a Motion Correction Module (MCM) that analyzes failures and adjusts instructions using Chain-of-Thought (CoT); (2) A motion-conditioned diffusion policy—which translates 5Hz motion instruction conditions into 20Hz high-frequency joint actions; and (3) Lifelong learning—which iteratively updates the MPM using corrected successful trajectories.

### Key Designs

1. **Dual-Process Motion Adjustment Mechanism (MPM + MCM)**:

    - **Function**: Efficiently predict motion instructions (normal conditions) + comprehensively correct failures (anomaly conditions).
    - **Mechanism**: The MPM is fine-tuned from LLaVA-v1.5 on 160k expert data pairs to efficiently predict 37 types of motion instructions (e.g., "move arm right with gripper closed," "make slight adjustments"). The MCM is fine-tuned on a comprehensive correction dataset (including 3,644 online human interventions + 7,365 offline annotations + 6,378 expert data points) to first identify failures using Chain-of-Thought $\rightarrow$ analyze failure types $\rightarrow$ generate semantic correction targets $\rightarrow$ adjust motion instructions. Decoupling these two modules is critical—ablation shows that jointly training a single model performs 6.5% worse.
    - **Design Motivation**: Separate "efficiency" from "robustness"—MPM optimizes speed for normal scenarios, while MCM optimizes quality for failure scenarios. This is analogous to System 1 (fast intuition) and System 2 (slow reasoning) in human cognition.

2. **Learnable Motion Codebook**:

    - **Function**: Provide discriminative features for motion instructions.
    - **Mechanism**: Pre-trained language models (such as the CLIP text encoder) struggle to distinguish semantically similar motion instructions (e.g., "move arm left" vs. "move arm right"). Therefore, a learnable codebook is introduced to assign independent embedding vectors to each motion instruction, retrieving corresponding features through a lookup mechanism.
    - **Design Motivation**: The semantic variance among motion instructions is much smaller than that of natural language task descriptions (most of the 37 instructions differ by only a single direction word), necessitating specialized discriminative features.

3. **Decoupled Condition Injection Diffusion Policy**:

    - **Function**: Inject motion instructions and visual observations at different stages of the diffusion process.
    - **Mechanism**: Directly concatenating visual features with motion instruction features causes the policy to over-rely on visual information and ignore motion instructions. By treating them as independent conditions at different stages of the diffusion denoising process, the guiding role of motion instructions is preserved.
    - **Design Motivation**: Avoid visual information "overwhelming" the conditioning signals of motion instructions.

### Loss & Training

Diffusion policy loss: $\mathcal{L} = \text{MSE}(\mathcal{E}^k, \pi(\mathcal{O}, \mathcal{M}, \mathcal{A}^0 + \mathcal{E}^k, k))$, where $\mathcal{E}^k$ is the noise at step $k$, and $\mathcal{A}^0$ represents the ground-truth actions. The policy is trained with 500 expert demonstrations per task. During lifelong learning, successful interaction trajectories are mixed with 20 expert demonstrations to fine-tune the MPM, preventing catastrophic forgetting.

## Key Experimental Results

### Main Results

Average success rate across 9 RoboMimic tasks:

| Method | Avg. Success Rate | Features |
|------|-----------|------|
| OpenVLA | 38.0% | End-to-end, no self-correction |
| Task-conditioned | 41.8% | Task description conditioned |
| Subgoal Self-reflection | 48.0% | Semantic-level self-correction |
| **Phoenix (Ours)** | **57.8%** | Motion-level self-correction |
| Human Intervention (Oracle) | 78.9% | Human correction upper bound |

Representative task comparison:

| Method | Coffee_D0 | ThreePieceAssembly_D0 | Threading_D0 |
|------|-----------|----------------------|-------------|
| Motion-conditioned | 68% | 30% | 58% |
| Subgoal Self-reflection | 80% | 34% | 80% |
| **Phoenix** | **94%** | **52%** | 68% |

### Ablation Study

| Configuration | Avg. Success Rate | Description |
|------|-----------|------|
| Motion-conditioned (No self-correction) | 46.9% | Baseline |
| Expert-Correction joint training | 49.6% | Shared model |
| Joint training + Self-reflection | 51.3% | Shared model + reflection |
| **Phoenix (Decoupled MPM+MCM)** | **57.8%** | Decoupled design is optimal |

### Key Findings
- **Motion Instructions > Subgoals**: Motion-conditioned performs better on long-horizon tasks (StackThree: 38% vs. 24%) compared to Subgoal-conditioned, indicating that motion instructions provide more direct execution guidance.
- **Decoupled Modules > Joint Training**: Decoupled training of MPM and MCM outperforms the joint training strategy by 6.5%. When data sizes differ significantly (160k vs. 16k), joint training fails to balance both tasks.
- **Lifelong Learning is Effective**: After 30 interaction rounds, the success rate increases from 60% to 75% (in-distribution) and generalizes well to tasks involving positional perturbations.
- **Real-World Feasibility**: In real-world experiments, Phoenix significantly outperforms Motion-conditioned under positional perturbations (55% vs. 35%) and background changes (45% vs. 30%).

## Highlights & Insights

- **Elegance of Motion Instructions as Intermediate Representation**: 37 instructions cover all combinations of arm directions $\times$ gripper states. This is neither too abstract (e.g., "put something down") nor too detailed (e.g., joint angles), presenting the optimal interface between MLLMs and low-level policies.
- **System 1/2 Analogy**: The dual-process design of MPM (fast and efficient) + MCM (slow but accurate) directly maps to dual-process theory in cognitive science, ensuring low latency in standard situations and high-quality corrections during anomalies.
- **Bootstrapping through Lifelong Learning**: Leveraging self-corrected success trajectories to bolster the prediction module establishes a virtuous cycle—as the expertise from MCM corrections is incorporated into MPM, future predictions become more accurate, reducing the need for corrective interventions.

## Limitations & Future Work

- **High Latency of MCM Chain-of-Thought**: Each correction requires a complete CoT inference, limiting real-time performance (5Hz decision frequency).
- **High Cost of Correction Data Collection**: Collecting 3,644 online human intervention trajectories demands substantial human labor.
- **Only 37 Motion Instructions**: The discretized instruction space might not cover all fine-grained movements in the continuous action space.
- **Multi-task Training without Inter-task Transfer**: 500 demonstrations per task mean that the total data volume grows linearly as the number of tasks increases.
- **Absence of Contact Force Feedback**: Relying purely on vision and motion instructions, lacking force feedback which is crucial for precision manipulation.

## Related Work & Insights

- **vs. RT-H**: RT-H also conditions its policy on motion instructions but lacks a self-correction mechanism. Phoenix integrates the MCM of failure detection and instruction adjustment.
- **vs. SayCan/CaP**: Semantic reflection frameworks rely on predefined skill libraries and cannot offer fine-grained action corrections. Phoenix bridges the entire spectrum from semantics to actions via motion instructions.
- **vs. Diffusion Policy**: The original diffusion policy lacks failure recovery capabilities, whereas the motion condition injection in Phoenix allows it to adapt to dynamically adjusted instructions.

## Rating
- Novelty: ⭐⭐⭐⭐ Using motion instructions as an intermediate layer is simple yet elegant, and the dual-process mechanism is theoretically grounded in cognitive science.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 simulation tasks + real-world experiments + lifelong learning + generalization tests.
- Writing Quality: ⭐⭐⭐⭐ The architecture diagram is clear, but detailing the construction of correction datasets could be more robust.
- Value: ⭐⭐⭐⭐ Provides a practical blueprint for designing intermediate abstraction layers in robotic self-correction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CHEER-Ekman: Fine-grained Embodied Emotion Classification](../../ACL2025/robotics/cheer-ekman_fine-grained_embodied_emotion_classification.md)
- [\[CVPR 2026\] FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in Visual-Language Robotic Manipulation](../../CVPR2026/robotics/flare_a_failure-aware_framework_for_autonomous_correction_and_recovery_in_visual.md)
- [\[ICLR 2026\] Primary-Fine Decoupling for Action Generation in Robotic Imitation](../../ICLR2026/robotics/primary-fine_decoupling_for_action_generation_in_robotic_imitation.md)
- [\[AAAI 2026\] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling](../../AAAI2026/robotics/cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](../../NeurIPS2025/robotics/dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)

</div>

<!-- RELATED:END -->
