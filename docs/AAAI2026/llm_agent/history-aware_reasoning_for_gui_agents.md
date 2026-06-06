---
title: >-
  [Paper Note] History-Aware Reasoning for GUI Agents
description: >-
  [AAAI 2026][LLM Agent][GUI Agent] This paper proposes the HAR framework, which transforms the reasoning paradigm of GUI agents from "history-unaware" to "history-aware" by constructing reflective learning scenarios…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Short-term Memory"
  - "Reinforcement Learning"
  - "Reflective Learning"
  - "History-Aware Reasoning"
date: 2026-05-08
content_hash: e78e90807b72035c
---

# History-Aware Reasoning for GUI Agents

**Conference**: AAAI 2026
**arXiv**: [2511.09127](https://arxiv.org/abs/2511.09127)  
**Code**: [https://github.com/BigTaige/HAR-GUI](https://github.com/BigTaige/HAR-GUI)  
**Area**: Agent / GUI Automation
**Keywords**: GUI Agent, Short-term Memory, Reinforcement Learning, Reflective Learning, History-Aware Reasoning

## TL;DR
This paper proposes the HAR framework, which transforms the reasoning paradigm of GUI agents from "history-unaware" to "history-aware" by constructing reflective learning scenarios, synthesizing error-correction guidelines, and designing a hybrid RL reward function incorporating a Memory-Augmented Reward (MAR). A 3B model trained under this framework surpasses larger models on multiple benchmarks including AITW, Mind2Web, and GUI-Odyssey.

## Background & Motivation

### State of the Field

**Background**: Existing GUI agents (e.g., UI-R1, GUI-R1, InfiGUI-R1) employ System-2 CoT combined with RL to enhance reasoning, yet they share a critical overlooked limitation: their reasoning is *history-unaware*—multi-step sequential interactions are reduced to isolated single-screen comprehension, disregarding crucial cues embedded in historical interaction context. For example, in an 11-step long-horizon task, the agent at step 8 reasons entirely without considering what transpired in the preceding 7 steps. This stems from the intrinsic CoT patterns of pretrained MLLMs; existing RL training with reasoning-format instructions optimizes only action prediction without altering the underlying reasoning paradigm.

### Solving the Problem

**Goal**: The paper aims to equip GUI agents with stable short-term memory in long-horizon sequential reasoning—specifically, to explicitly integrate and analyze historical interaction information within System-2 CoT. The central challenge is that the "history-unaware" reasoning pattern is deeply entrenched in the pretraining-phase CoT, and standard RL training cannot alter it (it only narrows the gap between pass@k and pass@1).

## Method

### Overall Architecture
HAR comprises two key training stages: (1) GUI scenario warm-up (SFT for domain knowledge injection); and (2) learning from failures (reflective RL for short-term memory enhancement).

### Key Designs
1. **GUI Scenario Warm-up (SFT)**:

    - Collection of GUI understanding data (captioning, QA, grounding, etc.)
    - Synthesis of Action-to-Summary (Act2Sum) data: a teacher model generates goal-oriented semantic summaries for each action to enhance action semantic understanding.
    - System-2 CoT distillation: Qwen2.5-VL-72B synthesizes System-2 reasoning chains for each sample; correctly predicted samples are filtered and used for training.

2. **Reflective Learning Scenario Construction**:

    - The warmed-up model performs inference; erroneous samples are collected as $\mathbb{D}_{his}$.
    - The teacher model generates up to 3 error-correction guidelines $\mathbb{G}$ per erroneous sample (analyzing the cause of the error and providing hints without revealing the answer).
    - A reflective format instruction is constructed: the erroneous prediction, erroneous CoT, and correction guidelines are provided to the model, which is required to first articulate the error (statement) and then re-reason.

3. **Hybrid RL Reward Function**:

    - **Format Reward** $r^{format}$: whether the output conforms to the reflective format.
    - **Action Reward** $r^{action}$: for coordinate-based actions (CLICK), a multi-scale Euclidean distance reward is used (normalized coordinate distance + absolute coordinate distance); a correct prediction yields an additional precision bonus ($r=1+F_{abs}$), while an incorrect prediction receives partial reward based on absolute distance.
    - **Memory-Augmented Reward (MAR)** $r^{memory}$: Qwen3-235B is used to judge whether the CoT contains analysis of historical interactions. This is the key innovation—explicitly rewarding the model for "considering prior actions during reasoning."
    - Combined reward: $r = r^{format} \times (r^{action} + \gamma \times r^{memory})$, with $\gamma=0.2$.
    - Design rationale: compared to forcing attention to history via instructions (GRPO*), MAR uses RL signals to allow the model to autonomously learn when historical context should be consulted.

4. **Round-2 RL + Task Mixing**: After Round-1 RL training in the reflective scenario, Round-2 RL switches to reasoning-format instructions (aligned with inference-time usage), while grounding tasks are mixed in (TMTS) to prevent degradation of grounding capability.

### Loss & Training
- GRPO algorithm for RL optimization.
- Base model: Qwen2.5-VL-3B-Instruct, LoRA rank=64, alpha=128.
- SFT: 1 epoch, lr=5e-6; RL: 2 epochs, lr=2e-6.

## Key Experimental Results

| Benchmark | Metric | HAR-GUI-3B | InfiGUI-R1-3B | GUI-R1-3B | UI-R1-3B | Qwen2.5-VL-7B |
|-----------|--------|-----------|--------------|----------|---------|--------------|
| AITW | SSR (avg) | **70.2** | 67.7 | 65.6 | 59.9 | - |
| Mind2Web | SSR (Cross-Task) | **42.2** | 37.2 | 38.8 | 36.8 | - |
| GUI-Odyssey | SSR (avg) | **62.31** | 50.62 | 48.35 | 46.71 | 58.39 |
| ScreenSpot | Avg | **83.3** | - | - | - | 79.8 |
| ScreenSpot-V2 | Avg | **86.2** | - | - | - | - |

OOD evaluation (Chinese Alipay mini-program): HAR-GUI-3B achieves a step success rate of 76.5%, substantially outperforming GUI-R1-3B (69.99%), with only a ~10% gap relative to Qwen2.5-VL-72B (86.91%).

### Ablation Study
- RL with reasoning-format instructions only (GRPO): reasoning remains history-unaware, with limited performance gains.
- Forcing historical attention via instructions (GRPO*): leads to performance degradation, indicating that rigid constraints are counterproductive—the model should autonomously acquire this behavior.
- HAR's reflective scenario + error-correction guidelines + MAR: the model autonomously develops history-aware reasoning patterns.
- Training RL exclusively on sequential reasoning data degrades grounding capability; TMTS effectively mitigates this.
- In post-training, HAR-GUI as the initialization checkpoint consistently outperforms GRPO and the base Qwen2.5-VL.

## Highlights & Insights
- **Precise problem identification**: The paper discovers and systematically analyzes the "history-unaware" reasoning deficiency in existing GUI agents—a problem present even in 72B models.
- **Memory-Augmented Reward**: Directly rewards whether the CoT incorporates historical information, using RL signals to guide the shift in reasoning paradigm rather than imposing manual constraints.
- **Reflective learning paradigm**: Constructing "error + correction guideline" reflective scenarios injects external domain reasoning knowledge more effectively than pure RL exploration.
- **Multi-scale coordinate reward**: Dual-scale reward combining normalized and absolute coordinates enables finer-grained optimization of CLICK actions.
- **3B surpassing 7B+**: HAR-GUI-3B outperforms Qwen2.5-VL-7B on GUI-Odyssey (62.31 vs. 58.39).

## Limitations & Future Work
- Synthesis of correction guidelines and CoT relies on a 72B teacher model; distillation quality is bounded by the teacher's capability.
- MAR uses a model to judge whether the CoT contains historical information, which may introduce false positives or negatives.
- OOD generalization is evaluated only in CLICK-only scenarios; complex actions such as TYPE are not thoroughly validated.
- The training pipeline is relatively complex (SFT + Round-1 RL + Round-2 RL + post-training).

## Related Work & Insights
- **vs. UI-R1/GUI-R1/InfiGUI-R1**: These methods apply RL with reasoning-format instructions to enhance reasoning but optimize only action prediction without altering the reasoning paradigm; HAR fundamentally changes the reasoning pattern through reflective scenarios.
- **vs. UI-TARS**: UI-TARS introduces System-2 reasoning but does not specifically address short-term memory; HAR's 3B model outperforms UI-TARS-2B on ScreenSpot.
- **vs. traditional agent frameworks (ReAct/Reflexion)**: Traditional methods rely on hand-crafted prompts for reflection; HAR internalizes reflective capability through training.

## Related Work & Insights
- The "short-term memory deficiency" may be a common weakness of all CoT-based agents—CoT tends to reason independently from the current state while neglecting historical context.
- The design rationale behind MAR (using RL rewards to incentivize specific properties of the reasoning process) is generalizable to other scenarios requiring particular reasoning patterns.
- The reflective learning scenario (providing errors + correction guidelines for RL) constitutes an effective knowledge injection paradigm.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Precisely identifies and addresses the history-unaware problem in GUI agents; the MAR design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three categories of benchmarks (sequential reasoning / grounding / understanding), OOD evaluation, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, analysis is convincing, and case studies are rich.
- **Value**: ⭐⭐⭐⭐ Proposes an effective solution to the short-term memory problem in GUI agents; code is open-sourced.

## Additional Notes
- The methodology and experimental design of this work offer reference value for related research.
- Future work may validate the generalizability and scalability of the approach across broader scenarios and larger scales.
- Integration with recent related work (e.g., intersections with RL/MCTS/multimodal methods) presents potential research opportunities.
- Deployment feasibility and computational efficiency should be assessed against practical application requirements.
- The selection of datasets and evaluation metrics may affect the generalizability of conclusions; cross-validation on additional benchmarks is recommended.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[ICML 2026\] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](../../ICML2026/llm_agent/persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[AAAI 2026\] TongUI: Internet-Scale Trajectories from Multimodal Web Tutorials for Generalized GUI Agents](tongui_internet-scale_trajectories_from_multimodal_web_tutor.md)
- [\[ICCV 2025\] Less is More: Empowering GUI Agent with Context-Aware Simplification](../../ICCV2025/llm_agent/less_is_more_empowering_gui_agent_with_context-aware_simplification.md)

</div>

<!-- RELATED:END -->
