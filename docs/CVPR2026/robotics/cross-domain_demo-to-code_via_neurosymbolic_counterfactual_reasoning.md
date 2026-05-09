---
title: >-
  [Paper Note] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning
description: >-
  [CVPR2026][Robotics][video-instructed robotic programming] This paper proposes NeSyCR, a neurosymbolic counterfactual reasoning framework that abstracts video demonstrations into a symbolic world model, detects cross-domain incompatibilities via counterfactual state simulation, and automatically corrects program steps. NeSyCR achieves a 31.14% improvement in success rate over the strongest baseline, Statler, on cross-domain demo-to-code tasks.
tags:
  - CVPR2026
  - Robotics
  - video-instructed robotic programming
  - cross-domain adaptation
  - neurosymbolic reasoning
  - counterfactual reasoning
  - code-as-policies
date: 2026-05-08
content_hash: 9c71c9c0a5e59942
---

# Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning

**Conference**: CVPR2026  
**arXiv**: [2603.18495](https://arxiv.org/abs/2603.18495)  
**Code**: To be confirmed  
**Area**: Robotics  
**Keywords**: video-instructed robotic programming, cross-domain adaptation, neurosymbolic reasoning, counterfactual reasoning, code-as-policies

## TL;DR

This paper proposes NeSyCR, a neurosymbolic counterfactual reasoning framework that abstracts video demonstrations into a symbolic world model, detects cross-domain incompatibilities via counterfactual state simulation, and automatically corrects program steps. NeSyCR achieves a 31.14% improvement in success rate over the strongest baseline, Statler, on cross-domain demo-to-code tasks.

## Background & Motivation

1. **Rise of the Code-as-Policies Paradigm**: LLMs/VLMs possess code generation capabilities, enabling the synthesis of executable robot control code from language instructions or video demonstrations. However, cross-domain adaptation remains a critical challenge.
2. **Inevitable Domain Gaps in Video Demonstrations**: Intrinsic differences in environment layout, object attributes, and robot configuration exist between the demonstration domain and the deployment domain; directly imitating demonstration behavior leads to program failures.
3. **Perceptual Observations Are Insufficient to Explain Procedural Discrepancies**: While observations can reveal physical differences, they cannot explain how structural differences undermine the underlying task program or causal dependencies.
4. **VLMs Lack Procedural Understanding**: Current VLMs struggle to reconstruct causal dependencies and achieve behavioral compatibility under domain shift, and tend to produce actions that are semantically plausible but logically inconsistent.
5. **Cascading Incompatibilities Are Difficult to Handle**: Cross-domain differences affect not only individual steps but may also trigger cascading incompatibilities (e.g., a change in tool position blocking subsequent steps), requiring global program reorganization.
6. **Existing Methods Lack Verifiable Adaptation Mechanisms**: VLM-based reasoning approaches lack symbolic tool verification, while world model approaches rely on constructing complete domain knowledge from a single demonstration and frequently produce invalid plans.

## Method

### Overall Architecture: NeSyCR

NeSyCR (Neurosymbolic Counterfactual Reasoning) formulates cross-domain adaptation as a counterfactual reasoning problem and operates in two phases:

- **Phase 1 — Symbolic World Model Construction**: Abstracts symbolic trajectories from video demonstrations and constructs a verifiable world model.
- **Phase 2 — Neurosymbolic Counterfactual Adaptation**: Compares the world model against target-domain observations, detects incompatible steps, and corrects the program.

### Symbolic World Model Construction

- **Symbolic State Translation**: A VLM extracts each frame observation into a scene graph containing object entities and spatial relations, forming a symbolic state sequence $\{s_1, \dots, s_N\}$.
- **Symbolic Dynamics Reconstruction**: For each pair of consecutive states $(s_t, s_{t+1})$, the VLM predicts the action operator $a_t = \Psi(s_t, s_{t+1})$, defining preconditions and effects.
- **Consistency Verification**: A symbolic tool $\Phi$ (based on VAL) performs forward simulation and verifies $\forall t, \Phi(s_t, a_t) \models s_{t+1}$, ensuring the world model $\mathcal{W} = (\mathcal{Q}, \mathcal{P}, \mathcal{A}, \Phi)$ is logically consistent.
- A STRIPS-style formalism is adopted to support forward execution and logical verification.

### Neurosymbolic Counterfactual Adaptation

- **Counterfactual Identification**: The VLM generates a counterfactual initial state $\hat{s}_1$ from target-domain observations; the symbolic tool performs forward simulation along the demonstration program to detect incompatible actions.
- **Incompatibility Detection**: An action is flagged as incompatible when its preconditions are not satisfied ($\text{pre}(a_t) \nsubseteq \hat{s}_t$) or its effects cannot be reproduced.
- **Counterfactual Exploration**: For each incompatible action, the VLM proposes alternative actions to restore the preconditions of subsequent steps, supporting addition, deletion, and reordering operations.
- **Iterative Verification**: The symbolic tool verifies the causal validity of each alternative action, ensuring that the adapted program $\tilde{\pi}$ satisfies $\hat{s}_{t+1} \models s_N$ (reaching the goal state).
- The adapted program is ultimately compiled into an executable code policy $\pi_\theta = \Psi(\tilde{\pi})$.

### Key Designs

- **VLM–Symbolic Tool Collaboration**: The VLM proposes alternative actions (leveraging commonsense knowledge) while the symbolic tool verifies logical consistency, forming a closed loop.
- **Additive and Subtractive Modifications**: Supports inserting new actions (e.g., introducing auxiliary tools) and removing redundant actions (e.g., steps whose goals are already satisfied).
- **Cascading Incompatibility Resolution**: Automatically discovers and resolves downstream incompatibilities triggered by a single modification through global forward simulation.

## Key Experimental Results

### Experimental Setup

- **Cross-Domain Factors**: 5 categories — Obstruction, Object affordance, Kinematic configuration, Gripper type, and combinations.
- **Benchmark Tasks**: Long-horizon manipulation tasks (up to 116 API calls), covering pick-and-place, sweeping, rotation, sliding, and other subtasks.
- **Three Complexity Levels**: Low / Medium / High, with 440 scenarios in total.
- **6 Baselines**: Demo2Code, GPT4V-Robotics, Critic-V, MoReVQA, Statler, LLM-DM.

### Main Results (Table 1 — Simulation Environment)

| Method | SR (Low) | SR (Med) | SR (High) | Notes |
|--------|---------|---------|----------|-------|
| Demo2Code | 26.67 | 25.00 | 22.50 | No adaptation mechanism |
| GPT4V-Robotics | 71.67 | 41.67 | 20.00 | VLM reasoning |
| Statler | 61.67 | 41.67 | 5.00 | World model without symbolic verification |
| **NeSyCR** | **86.67** | **75.00** | **60.00** | Ours |

- NeSyCR achieves an average SR improvement of **31.14%** over Statler and **27.73%** over GPT4V-Robotics.
- Under combined cross-domain factors, NeSyCR maintains 47.5–80.0% SR, while Statler drops to 32.5–67.5%.

### Real-World Experiments (Table 2)

| Method | SR | GC | PD |
|--------|-----|-----|-----|
| Demo2Code | 0.00 | 25.00 | — |
| GPT4V-Robotics | 50.00 | 75.00 | 0.00 |
| Statler | 50.00 | 67.86 | 42.86 |
| **NeSyCR** | **87.50** | **98.21** | 24.49 |

- Experiments use a Franka Emika Research 3 robotic arm, adapting from human video demonstrations to real-robot deployment.
- A vertically placed drawer scenario requires alternating operations on two drawers to avoid mutual interference.

### Ablation Study (Table 3)

| Variant | SR | Drop |
|---------|-----|------|
| NeSyCR (full) | 68.42 | — |
| w/o alternative action verification (Eq. 8) | 50.00 | -18.42 |
| w/o counterfactual identification (Eq. 6) | 47.37 | -21.05 |
| w/o both | 39.47 | -28.95 |
| w/o symbolic world model (Eq. 4) | 34.21 | -34.21 |

Removing the symbolic world model causes the largest performance drop, confirming that verifiable symbolic reasoning is the core component.

## Highlights & Insights

- **Formulating cross-domain demo-to-code as counterfactual reasoning** provides a clear formal framework (STRIPS + counterfactual state-space exploration).
- **The VLM–symbolic tool co-design is elegant**: the VLM leverages commonsense knowledge to propose candidates, while the symbolic tool guarantees logical correctness, making them highly complementary.
- **Cascading incompatibility handling**: the framework automatically detects how a single modification affects downstream steps and performs global correction.
- **Thorough real-world validation**: large-scale quantitative simulation experiments (440 scenarios) are complemented by end-to-end validation on a physical robot.
- **Refined experimental design**: a systematic matrix of 5 cross-domain factor types × 3 complexity levels facilitates fine-grained performance analysis across different dimensions.

## Limitations & Future Work

- **Significant performance degradation under large task complexity gaps**: when the deployment task substantially exceeds the demonstration in complexity, counterfactual reasoning struggles to compensate for missing information.
- **Dependence on VLM scene graph extraction quality**: the accuracy of symbolic state translation is bounded by the perceptual capability of the VLM.
- **Limited expressiveness of STRIPS-style representation**: it is difficult to model continuous physical quantities, deformable objects, and other complex scenarios.
- **Computational overhead**: iterative verification with VLM and symbolic tools may introduce significant latency for long-horizon tasks.
- **Single-demonstration constraint**: the world model is constructed from a single demonstration video, resulting in insufficient diversity.

## Related Work & Insights

- **vs. Code-as-Policies (SayCan, ProgPrompt)**: This work focuses on cross-domain adaptation rather than single-domain code generation.
- **vs. Demo2Code**: Demo2Code directly imitates demonstrations without any adaptation mechanism; NeSyCR corrects the program through counterfactual reasoning.
- **vs. Statler**: Statler employs symbolic state representations but does not integrate symbolic verification tools; re-planning from scratch causes collapse under high complexity.
- **vs. LLM-DM**: LLM-DM constructs complete domain knowledge from a single demonstration and frequently generates invalid plans; NeSyCR preserves the demonstration structure and applies only local corrections.
- **vs. Behavior Cloning / Inverse Reinforcement Learning**: These methods generalize poorly under perceptual and physical changes; NeSyCR performs adaptation at the symbolic level.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of counterfactual reasoning and neurosymbolic verification constitutes a new paradigm in the demo-to-code domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Systematic experiments across 440 scenarios, real-robot validation, and fine-grained controlled variable analysis.
- Writing Quality: ⭐⭐⭐⭐ — Formalization is clear, symbolic notation is rigorous, and case studies are intuitive.
- Value: ⭐⭐⭐⭐ — Provides a verifiable adaptation framework for cross-domain robotic programming, addressing an important research direction.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](../../NeurIPS2025/robotics/nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[ICLR 2026\] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](../../ICLR2026/robotics/one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)

<!-- RELATED:END -->
