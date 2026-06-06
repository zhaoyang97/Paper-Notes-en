---
title: >-
  [Paper Note] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions
description: >-
  [ACL 2026][Robotics][Vision-and-Language Navigation] This paper proposes the VLN-NF benchmark—the first task requiring VLN agents to identify false-premise instructions and output NOT-FOUND in 3D partially observable env…
tags:
  - "ACL 2026"
  - "Robotics"
  - "Vision-and-Language Navigation"
  - "False Premise"
  - "NOT-FOUND"
  - "Embodied Exploration"
  - "Feasibility Awareness"
date: 2026-05-08
content_hash: 47ec2af3b52e785c
---

# VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions

**Conference**: ACL 2026  
**arXiv**: [2604.10533](https://arxiv.org/abs/2604.10533)  
**Code**: [https://vln-nf.github.io/](https://vln-nf.github.io/)  
**Area**: Robotics/Embodied AI  
**Keywords**: Vision-and-Language Navigation, False Premise, NOT-FOUND, Embodied Exploration, Feasibility Awareness

## TL;DR

This paper proposes the VLN-NF benchmark—the first task requiring VLN agents to identify false-premise instructions and output NOT-FOUND in 3D partially observable environments. It introduces the REV-SPL evaluation metric and the ROAM two-stage hybrid framework. ROAM achieves a 6.1 REV-SPL, representing a 45% improvement over supervised baselines.

## Background & Motivation

**Background**: Vision-and-Language Navigation (VLN) studies how embodied agents navigate in 3D environments based on natural language instructions. Existing benchmarks (R2R, REVERIE, etc.) assume instructions are always feasible and the target object exists in the environment.

**Limitations of Prior Work**: In real-world deployment, human instructions are often incorrect—cognitive science research indicates humans make mistakes in approximately one out of every seven object-location recalls. For example, a user might say "get the plate on the kitchen table," but the plate is actually in the living room. Existing VLN agents cannot handle such situations, either hallucinating similar objects or searching indefinitely.

**Key Challenge**: In partially observable 3D environments, the fact that a target does not exist cannot be confirmed from a single viewpoint; it requires sufficient exploration and evidence collection before a NOT-FOUND judgment can be made. However, existing VLN systems lack this evidence-driven verification capability, and simply adding a NOT-FOUND action leads to premature abandonment.

**Goal**: (1) Construct the VLN-NF benchmark dataset containing false-premise instructions; (2) design the REV-SPL evaluation metric to jointly assess navigation, exploration, and decision-making; (3) propose the ROAM framework for evidence-driven NOT-FOUND judgment.

**Key Insight**: Decompose the problem into room-level navigation (amenable to supervised learning) and intra-room exploration/verification (driven by LLM/VLM), avoiding issues caused by exploration behavior uncertainty in end-to-end training.

**Core Idea**: Automatically construct a false-premise dataset through a scalable pipeline of LLM rewriting + VLM verification, and solve the new task with a two-stage hybrid framework (supervised navigation + LLM/VLM exploration/verification).

## Method

### Overall Architecture

VLN-NF includes three contributions: (1) Dataset construction pipeline—using LLMs to rewrite feasible instructions into false-premise ones and VLMs to verify the target truly does not exist; (2) REV-SPL evaluation metric—jointly assessing reaching the target room, exploration coverage, and correctness of FOUND/NOT-FOUND decisions; (3) ROAM two-stage method—the first stage uses a supervised model to locate the target room, and the second stage uses LLM/VLM to explore within the room and make a judgment.

### Key Designs

1.  **Dataset construction pipeline (Rewrite + Verify)**:

    - **Function**: Automatically convert existing feasible VLN instructions into false-premise instructions.
    - **Mechanism**: Given the original instruction and target object $o$, an LLM Rewriter selects a plausible substitute $o'$ from outside the target room's object list (e.g., "water the plant under the window" → "wipe the sofa under the window"), generating a semantically fluent but factually incorrect new instruction. A VLM Verifier runs open-vocabulary detection on all panoramas of the target room to confirm $o'$ is indeed absent. If detected, it resamples; otherwise, it accepts. Human review of 5% of samples showed an error rate <2%.
    - **Design Motivation**: Manually labeling exploration behavior is extremely costly and uncertain; an automated pipeline achieves low-cost, high-quality dataset construction.

2.  **REV-SPL Evaluation Metric**:

    - **Function**: Jointly evaluate navigation efficiency, exploration sufficiency, and decision correctness.
    - **Mechanism**: Defines a reference exploration path—when instructions contain landmark clues, the reference path covers visible viewpoints of the original target object (using TSP to solve for the shortest covering path); without landmarks, a greedy coverage strategy traverses the room until covering 85%+ of objects. REV-SPL penalizes premature stopping (insufficient coverage) and incorrect decisions (judging FOUND as NOT-FOUND or vice versa) while rewarding exploration efficiency.
    - **Design Motivation**: Standard SPL only evaluates shortest path arrival and cannot measure the sufficiency of evidence collection. Simply reusing SPL would encourage degenerate behavior (outputting NOT-FOUND without exploring).

3.  **ROAM Two-Stage Hybrid Framework**:

    - **Function**: Realize evidence-driven exploration and judgment in the VLN-NF task.
    - **Mechanism**: The first stage uses the DUET supervised model to navigate to the target room (weakly supervised, requiring only room-level labels); the second stage uses an LLM to plan exploration strategies, while a VLM performs open-vocabulary detection, combined with a free-space clearance prior to guide exploration towards uncovered areas. Decides FOUND or NOT-FOUND based on detection results after exploration.
    - **Design Motivation**: Pure supervised methods suffer from premature termination due to covariate shift in imitation learning; pure LLM methods perform poorly in inter-room navigation in partially observable environments. The hybrid framework leverages the strengths of both.

### Loss & Training

The first-stage DUET model uses standard VLN training (cross-entropy loss + navigation reward); the second-stage LLM/VLM exploration module requires no training, directly utilizing the zero-shot reasoning capabilities of pre-trained models.

## Key Experimental Results

### Main Results

| Method | Type | REV-SPL (val-unseen) |
|--------|------|------|
| DUET + VLN-NF | Supervised | 4.2 |
| NaviLLM | LLM-based | 1.0 |
| Gemini 1.5 Pro | LLM-based | 1.5 |
| ROAM | Mixed | 6.1 |

ROAM improves by 45% over the strongest supervised baseline and by 4-6 times over LLM baselines.

### Ablation Study

| Configuration | Key Index | Description |
|------|---------|------|
| ROAM Full | REV-SPL 6.1 | Supervised navigation + LLM/VLM exploration |
| w/o Free-space Prior | REV-SPL decreased | Exploration coverage dropped |
| DUET directly with NF | REV-SPL 4.2 | Prematurely outputted NOT-FOUND |

### Key Findings

- **Existing VLN agents cannot handle false premises**: All baselines have very low REV-SPL on VLN-NF, primarily because they make judgments without sufficient exploration.
- **Premature abandonment is the core problem**: Simply adding a NOT-FOUND action to supervised VLN models actually causes them to learn to "give up early," as covariate shift in imitation learning is particularly severe in exploration tasks.
- **LLMs are good at intra-room planning but poor at inter-room navigation**: Pure LLM methods (NaviLLM, Gemini) perform poorly without step-level navigation guidance, but ROAM achieves good results by using LLMs for intra-room exploration planning.
- **High dataset quality**: The LLM rewrite + VLM verification pipeline has a human audit error rate of <2%, with low construction costs and high scalability.

## Highlights & Insights

- **Fills the gap in VLN reliability**: Systematically studies false-premise navigation in 3D partially observable environments for the first time, filling an important gap in the VLN community regarding instruction unreliability.
- **Ingenious REV-SPL metric design**: Extends from SPL to evidence-driven verification scenarios; the dual-mode design of reference exploration paths (landmark-guided vs. coverage scan) balances the evaluation needs of different scenarios well.
- **Transferable two-stage decomposition strategy**: The idea of decoupling navigation and verification can be transferred to other embodied tasks requiring decision-making under uncertainty.

## Limitations & Future Work

- Currently only focuses on object-level false premises (object non-existence), excluding broader unreliable instruction types like attribute errors or ambiguous instructions.
- Terminates immediately after judging NOT-FOUND, lacking recovery strategies (e.g., requesting clarification, trying alternative paths).
- Absolute REV-SPL values remain low (max 6.1), indicating the task is still very challenging with significant room for improvement.
- Only built on top of REVERIE, without extending to other VLN benchmarks like R2R.

## Related Work & Insights

- **vs MoTIF**: MoTIF studies infeasible instructions in 2D mobile apps, but agents have full observability of the screen. VLN-NF is more difficult as it requires confirming target absence through autonomous exploration in 3D partially observable environments.
- **vs R2R-UNO**: R2R-UNO studies instruction-environment mismatches caused by physical obstacles, focusing on changes in navigability. VLN-NF focuses on semantic-level false premises, where the target itself does not exist.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First false-premise benchmark for 3D partially observable VLN; problem definition is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparison with various baselines, though low absolute performance limits the depth of analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous logic in method and evaluation design.
- Value: ⭐⭐⭐⭐ Opens a new direction for VLN reliability research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](../../CVPR2026/robotics/towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)
- [\[ACL 2026\] Capability-Oriented Failure Attribution for Vision-Language Navigation Agents](where_did_it_go_wrong_capability-oriented_failure_attribution_for_vision-and-lan.md)
- [\[ACL 2026\] Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents](breaking_down_and_building_up_mixture_of_skill-based_vision-and-language_navigat.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](../../CVPR2026/robotics/decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[ACL 2026\] GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap](groke_vision-free_navigation_instruction_evaluation_via_graph_reasoning_on_opens.md)

</div>

<!-- RELATED:END -->
