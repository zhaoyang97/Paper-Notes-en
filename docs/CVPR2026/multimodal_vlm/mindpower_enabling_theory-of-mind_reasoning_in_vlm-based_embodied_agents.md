---
title: >-
  [Paper Note] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents
description: >-
  [CVPR 2026][Multimodal VLM][Theory of Mind] MindPower proposes a Robot-Centric Theory-of-Mind reasoning framework that organizes perception → belief → desire → intention → decision → action into a three-level six-layer reasoning hierarchy (MindPower Reasoning Hierarchy), and employs Mind-Reward (GRPO-based reinforcement learning) to optimize reasoning consistency, surpassing GPT-4o by 12.77% on decision-making and 12.49% on action generation.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Theory of Mind
  - BDI Reasoning
  - Embodied Agent
  - Mind-Reward
  - GRPO
  - Robot-Centric
date: 2026-05-08
content_hash: 90acbf4cdcd86333
---

# MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents

**Conference**: CVPR 2026
**arXiv**: [2511.23055](https://arxiv.org/abs/2511.23055)
**Code**: [https://zhangdaxia22.github.io/MindPower/](https://zhangdaxia22.github.io/MindPower/) (Benchmark)
**Area**: Multimodal VLM
**Keywords**: Theory of Mind, BDI Reasoning, Embodied Agent, Mind-Reward, GRPO, Robot-Centric

## TL;DR

MindPower proposes a Robot-Centric Theory-of-Mind reasoning framework that organizes perception → belief → desire → intention → decision → action into a three-level six-layer reasoning hierarchy (MindPower Reasoning Hierarchy), and employs Mind-Reward (GRPO-based reinforcement learning) to optimize reasoning consistency, surpassing GPT-4o by 12.77% on decision-making and 12.49% on action generation.

## Background & Motivation

1. **Embodied agents lack mental reasoning capabilities**: Existing VLM-based embodied agents can only execute explicit instructions and are unable to infer human beliefs, desires, and intentions, let alone proactively provide assistance based on such inferences.
2. **Existing ToM benchmarks focus exclusively on Role-Centric perspectives**: Benchmarks such as MuMA-ToM and MMToM-QA only infer the mental states of characters in videos and do not involve reasoning from the agent's own perspective.
3. **No closed loop from reasoning to action**: Existing benchmarks are predominantly multiple-choice in format and do not require agents to generate executable decisions and action sequences based on mental reasoning.
4. **VLMs perform poorly on high-level reasoning**: Experiments show that closed-source VLMs such as GPT-4o and Gemini perform adequately at the perception level but fall far below human performance on belief reasoning and action generation.
5. **Open-source VLMs are weaker still**: Models such as InternVL3.5 and LLaVA-OV achieve near-zero SR/AC on action generation, typically producing vague, non-executable outputs.
6. **Standard CoT reasoning yields limited gains**: General-purpose `<think>` reasoning chains are less effective than structured BDI reasoning hierarchies on Theory-of-Mind tasks.

## Method

### Overall Architecture

MindPower consists of three components: (1) **MindPower Benchmark**—a dataset of 590 household scenarios covering two tasks; (2) **MindPower Reasoning Hierarchy**—a structured three-level, six-layer reasoning framework; and (3) **Mind-Reward**—GRPO-based reinforcement learning optimization. The backbone model is Qwen2.5-VL-7B-Instruct.

### MindPower Reasoning Hierarchy (Three Levels, Six Layers)

- **Level-1 Perception** `<Perception>`: Observes the environment and comprehends what is currently occurring.
- **Level-2 Mental Reasoning**:
  - `<Belief>`: Infers the beliefs of both the agent itself and the human, including second-order beliefs (e.g., "I believe Alice thinks the apple is on the table").
  - `<Desire>`: Identifies assistive goals (e.g., "Alice wants to drink milk").
  - `<Intention>`: Forms specific action intentions (e.g., "I should help her retrieve the milk from the refrigerator").
- **Level-3 Decision & Action**:
  - `<Decision>`: Selects an action plan.
  - `<Action>`: Outputs atomic operation sequences, e.g., `walk(fridge), open(fridge), pick(milk)`.

### Robot-Centric Perspective (Core Innovation)

In contrast to existing Role-Centric designs, the agent not only infers the mental states of others but also explicitly models its own beliefs, forming a complete second-order reasoning loop. For example: "I know the apple is actually in the refrigerator" + "I infer that Alice believes the apple is on the table" → "Her belief is incorrect; I should help correct it."

### Two Core Tasks

1. **False-Belief Correction**: Detects erroneous human beliefs about the environment (e.g., an object has been moved without the person's knowledge); the agent must identify the inconsistency and proactively correct it.
2. **Implicit Goal Inference & Completion**: Infers latent goals from subtle behavioral cues such as repeated search failures, and provides assistance accordingly. This task covers four scenario types: special populations (wheelchair users, children), object property reasoning, functional composition, and dialogue-based inference.

### Mind-Reward Training Strategy

**Two-stage training**: (1) SFT warm-start (5 epochs) to establish basic reasoning alignment; (2) GRPO reinforcement optimization (400 iterations, 8 samples per iteration).

**Mind-Reward design**: Each layer's reasoning output is converted into atomic action sequences by Qwen3-Max, and three alignment metrics are computed:
- **Atomic accuracy** (ROUGE-1): Proportion of correctly matched atomic actions; perspective annotations ensure Robot-Centric alignment.
- **Local consistency** (ROUGE-2): Coherence of adjacent atomic action pairs.
- **Global consistency** (ROUGE-L): Alignment of the overall reasoning sequence.

$$R_{\text{Mind}} = \alpha_1 R_{\text{atomic}} + \alpha_2 R_{\text{local}} + \alpha_3 R_{\text{global}}$$

where $\alpha_1=0.2, \alpha_2=0.3, \alpha_3=0.5$. A format reward $R_{\text{Format}}$ (verifying that all six layer tags appear in order) is added as an auxiliary signal. The total reward is $R = R_{\text{Mind}} + R_{\text{Format}}$, optimized via GRPO.

## Key Experimental Results

### Main Results: Comparison with Baseline VLMs

| Method | Decision (S↑) | Action SR↑ | Action AC↑ | BPC↑ |
|--------|:---:|:---:|:---:|:---:|
| GPT-4o (image input) | 34.35 | 1.82 | 2.91 | 8.05 |
| Gemini-2.5 Pro (video input) | 33.87 | 2.08 | 2.54 | 8.56 |
| Video-R1-7B | 30.33 | 1.43 | 1.72 | 6.45 |
| Qwen2.5-VL-7B (base) | 26.56 | 0.29 | 0.22 | 6.07 |
| **Ours (SFT+Mind-Reward)** | **47.12** | **11.75** | **15.40** | **8.87** |
| Human Baseline | 56.66 | 19.37 | 26.26 | 8.19 |

Relative to GPT-4o: Decision Sentence Transformer score +12.77 pp; Action accuracy AC +12.49 pp.

### Ablation Study

| Configuration | Decision (S↑) | Action AC↑ | BPC↑ |
|---------------|:---:|:---:|:---:|
| Qwen2.5-VL-7B (baseline) | 26.56 | 0.22 | 6.07 |
| Mind-Reward only (no SFT) | 24.68 | 0.40 | 6.63 |
| SFT only (no RL) | 43.84 | 10.48 | 8.78 |
| **SFT + Mind-Reward** | **47.12** | **15.40** | **8.87** |

- SFT alone yields substantial gains (AC: 0.22→10.48), demonstrating the intrinsic effectiveness of the hierarchical reasoning structure.
- Mind-Reward alone without SFT provides limited improvement (AC: 0.40), confirming the necessity of SFT warm-start.
- The SFT+RL combination is optimal; RL further improves upon SFT by approximately 5 points.
- MindPower Hierarchy vs. standard CoT (evaluated on GPT-4o): structured BDI reasoning outperforms general-purpose `<think>` reasoning on decision-making by 4.89%.

## Highlights & Insights

1. **Systematic integration of cognitive science and embodied AI**: The BDI framework (Belief–Desire–Intention) is systematically incorporated into a VLM agent, forming an interpretable reasoning chain from perception to action.
2. **Robot-Centric perspective as a novel contribution**: This work is the first to require the agent to simultaneously model both its own beliefs and those of others, enabling second-order Theory-of-Mind reasoning—a fundamental departure from existing Role-Centric benchmarks.
3. **Principled Mind-Reward design**: Reasoning quality is decomposed into atomic, local, and global granularities, yielding a more controllable and reproducible reward signal than black-box LLM scoring.
4. **Insightful task design**: Both false-belief correction and implicit goal inference are central scenarios in real-world human–robot collaboration, and the benchmark explicitly covers special populations such as wheelchair users and children.

## Limitations & Future Work

1. **Limited data scale**: The 590-sample dataset is relatively small and covers only two simulators (VirtualHome and ThreeDWorld); generalization to real physical environments remains an open question.
2. **Constrained action space**: Atomic actions are high-level operations (e.g., `pick(apple)`) and do not address low-level motion control (joint angles, force control), leaving a substantial gap with real robot deployment.
3. **LLM-dependent evaluation**: BPC scores are produced by GPT-4o, introducing potential evaluation bias and reproducibility concerns.
4. **Computational cost**: GRPO training requires H800 GPUs, posing a barrier for resource-constrained researchers.
5. **Multi-turn interaction not validated**: All scenarios involve single-turn reasoning; the agent's ability to maintain belief consistency across sustained interactions has not been tested.

## Related Work & Insights

### vs. MuMA-ToM / MMToM-QA (Existing ToM Benchmarks)

MuMA-ToM and MMToM-QA infer the mental states of characters in videos (Role-Centric) and produce multiple-choice answers, without involving decision-making or action generation. MindPower adopts a Robot-Centric perspective, requiring the agent to reason from its own viewpoint and output executable atomic action sequences, while evaluating the complete reasoning chain from perception to action.

### vs. Smart-Help / RoboBench (Embodied Collaborative Agents)

Smart-Help relies on predefined objectives to optimize human–robot ergonomics; RoboBench decomposes high-level goals into sequentially executed subtasks. Neither approach performs mental reasoning—first-order or second-order belief inference is absent. The key distinction in MindPower is that the agent must infer "what the human believes" and "what I myself know," then make decisions based on the discrepancy between these mental states, rather than merely executing given objectives.

### vs. Visual-RFT / LLaVA-CoT (Reasoning-Enhanced VLMs)

Visual-RFT and LLaVA-CoT provide general-purpose visual reasoning enhancement methods. MindPower's Reasoning Hierarchy is specifically designed for ToM tasks with a structured pipeline (Perception→Belief→Desire→Intention→Decision→Action), which experiments demonstrate outperforms general CoT by 4.89% on decision-making tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Robot-Centric ToM, three-level six-layer reasoning hierarchy, and Mind-Reward are all introduced for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparisons against 10+ baseline VLMs with ablation and qualitative analysis, though the dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — The framework is clearly presented with rich illustrations and well-organized structure.
- **Value**: ⭐⭐⭐⭐ — Provides an important benchmark and methodology for social intelligence in embodied AI, though a gap with real robot deployment remains.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[CVPR 2026\] EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)

<!-- RELATED:END -->
