---
title: >-
  [Paper Note] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games
description: >-
  [ICCV 2025][Robotics][Vision-Language-Action] This paper proposes CombatVLA, an efficient 3B-parameter VLA model designed for combat tasks in 3D action role-playing games. Through the Action-of-Thought data format and a truncated inference strategy, CombatVLA achieves inference speeds up to 50× faster than existing VLM-based game frameworks while surpassing human players in combat success rate.
tags:
  - ICCV 2025
  - Robotics
  - Vision-Language-Action
  - 3D Games
  - Real-Time Decision Making
  - Action-of-Thought
  - Efficient Inference
date: 2026-05-08
content_hash: 9b5efddd123639dd
---

# CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games

**Conference**: ICCV 2025
**arXiv**: [2503.09527](https://arxiv.org/abs/2503.09527)
**Code**: [https://combatvla.github.io/](https://combatvla.github.io/)
**Area**: Robotics
**Keywords**: Vision-Language-Action, 3D Games, Real-Time Decision Making, Action-of-Thought, Efficient Inference

## TL;DR

This paper proposes CombatVLA, an efficient 3B-parameter VLA model designed for combat tasks in 3D action role-playing games. Through the Action-of-Thought data format and a truncated inference strategy, CombatVLA achieves inference speeds up to 50× faster than existing VLM-based game frameworks while surpassing human players in combat success rate.

## Background & Motivation

- **State of the Field**: VLA models have made remarkable progress in embodied intelligence, yet real-time decision-making in complex 3D environments remains a significant challenge.
- **Limitations of Prior Work**: Combat tasks in 3D ARPGs such as *Black Myth: Wukong* impose three stringent requirements: (1) real-time processing of high-resolution visual streams; (2) tactical adaptation to dynamically changing enemy behaviors; and (3) action execution at the sub-second level. Existing approaches fall short: API-based methods (e.g., Voyager) cannot simulate human visual interaction; RL-based methods require extensive predefined reward functions and trial-and-error training; large-scale VLM frameworks such as Cradle and VARP exhibit single-inference latencies of 60–90 seconds, rendering them fundamentally incompatible with real-time combat.
- **Root Cause**: The gap between the inference speed of large VLMs and the millisecond-level responsiveness demanded by real-time combat scenarios.
- **Paper Goals**: To develop a lightweight yet capable VLA model that achieves real-time combat performance through data-efficient training and truncated inference.

## Method

### Overall Architecture

The CombatVLA pipeline comprises four core components: (1) an action tracker that collects human player interaction data; (2) a data construction module that formats collected data into the Action-of-Thought (AoT) format for training; (3) a three-stage progressive learning paradigm to train the 3B-parameter model; and (4) an action execution framework that integrates the trained model for real-time inference.

### Key Designs

1. **Action Tracker**: A lightweight Python tool running in the background via two independent threads—one monitoring keyboard and mouse inputs, the other capturing game frames. Frame-action alignment is performed using timestamps: for each action $a_j$, the nearest future frame is identified as $i_j = \arg\min_i (t_{f_i} \geq t_{a_j})$, ensuring each action is correctly paired with its corresponding visual context.

2. **Action-of-Thought (AoT) Data Format**: Inspired by Chain-of-Thought reasoning, the collected frame set $F$ and action set $A$ along with their alignment relations are converted into JSON-formatted AoT data. Each entry contains an `[action]` field (e.g., "press space") and an `[explanation]` field describing the current enemy state and the physical semantics of the action. A special token $\langle\text{TRUNC}\rangle$ is introduced to truncate model output during inference to accelerate generation.

3. **Three-Stage Progressive Learning**:
   - **Stage 1 – Coarse-Grained Video-AoT Fine-Tuning**: Each video segment contains $n=20$ frames at $m=10$ fps. Actions corresponding to each frame are arranged chronologically to generate video-AoT data pairs. The model learns a holistic understanding of the combat environment over 3 epochs.
   - **Stage 2 – Fine-Grained Frames-AoT Fine-Tuning**: Action-frame alignment data is constructed by tracing back $k=4$ frames from the current action timestamp, forming precise causal reasoning sequences. The model learns temporal logic in combat scenarios over 1 epoch.
   - **Stage 3 – Truncated Frames-AoT Fine-Tuning**: The $\langle\text{TRUNC}\rangle$ token is introduced to reorganize AoT data, placing the action field before the explanation. During real-time inference, generation halts upon encountering this token, yielding approximately a 2× speedup. Training is conducted for 3 epochs.

4. **Adaptive Action-Weighted Loss**: The training loss consists of three components: a language modeling loss $\mathcal{L}_{lang}$, an action alignment loss $\mathcal{L}_{align}$, and a modality contrastive loss $\mathcal{L}_{con}$. A priority-aware matching criterion $\mathcal{M}(A_l, A_o)$ determines whether the model's output action matches the ground-truth label. Based on the matching result, the distance between visual EOS embeddings and action EOS embeddings is adjusted accordingly. Weights $\alpha_i = 2^{(k-i-1)}$ decay exponentially by priority and are normalized to $[0.1, 1.0]$, ensuring that high-priority rare critical actions (e.g., dodge, healing) receive greater attention.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{lang} + \alpha \cdot \mathcal{L}_{act}$, where $\mathcal{L}_{act}$ switches between a pull loss ($\mathcal{L}_{con}^{pull}$) and a push-plus-alignment loss ($\mathcal{L}_{con}^{push} + \mathcal{L}_{align}$) depending on the matching result. The backbone is Qwen2.5-VL-3B, trained with a learning rate of 1e-5, batch size of 1, and temperature of 0.7. The visual encoder is frozen during training; only the language model parameters are fine-tuned. At inference time, the truncation strategy is applied and actions are translated into keyboard and mouse operations via the `pyautogui` library.

## Key Experimental Results

### Main Results

| Model | CUBench Avg. | Inference Latency (s) | Model Calls |
|---|---|---|---|
| GPT-4o | 57.29 | 61.68 (Cradle) | 5 |
| Gemini-2.0-flash | 57.90 | — | — |
| Qwen2.5-VL-3B (backbone) | 55.87 | — | — |
| **CombatVLA-3B** | **63.61** | **1.85** | **1** |
| VARP framework | — | 90.23 | 10 |

| Task Type | CombatVLA | Cradle | VARP | Human |
|---|---|---|---|---|
| Easy zero-shot (BMW) | ~90% | ~30% | ~60% | ~80% |
| Hard (BMW) | ~80% | ~10% | ~30% | ~70% |
| Very Hard (BMW) | ~60% | 0% | 0% | ~50% |
| Cross-game zero-shot (SSDT) | ~70% | ~10% | ~20% | ~60% |

### Ablation Study

| Training Stage | Gathering | Comprehension | Reasoning | Avg. | Inference Time (s) |
|---|---|---|---|---|---|
| Stage 1 | 53.89 | 57.35 | 60.57 | 57.27 | 3.73 |
| Stage 2 | 59.17 | 62.25 | 62.86 | 61.43 | 3.73 |
| Stage 3 (full) | **60.83** | 60.29 | **69.71** | **63.61** | **1.85** |

| Loss Configuration | Reasoning | Avg. |
|---|---|---|
| Full model | **69.71** | **63.61** |
| w/o $\mathcal{L}_{con}$ | 63.14 | 61.58 |
| w/o $\mathcal{L}_{align}$ | 63.71 | 61.64 |

### Key Findings

- On high-level reasoning tasks, CombatVLA outperforms the second-best model, Claude 3.5 Sonnet, by 14.28 points, attributable to the reasoning capability enhanced by AoT data.
- The truncation strategy yields approximately 2× speedup in Stage 3 compared to Stage 2 (1.85s vs. 3.73s), while simultaneously improving performance.
- Performance on general benchmarks (MME/VideoMME/OCRBench) remains on par with the backbone model, demonstrating that task-specific training does not degrade general-purpose capabilities.
- High zero-shot cross-game success rates (BMW→SSDT) validate strong generalization.

## Highlights & Insights

- The AoT data format elegantly combines the reasoning enhancement of CoT with the efficiency advantage of the truncation strategy, striking an optimal balance between "reason before acting" and "extract action, discard reasoning."
- Training exclusively on Very Hard tasks enables zero-shot generalization to Easy and Hard tasks and even cross-game generalization, suggesting that tactical logic learned under high-difficulty combat is transferable.
- The modality contrastive loss effectively mitigates action class imbalance by aligning the visual and action semantic spaces.

## Limitations & Future Work

- Inference currently requires pausing the game to await model output, and true real-time control has not yet been achieved.
- The training dataset comprises only approximately 5K high-quality AoT samples, limiting data scale.
- Generalization to more diverse game genres beyond the two evaluated titles remains to be verified.
- The action space is relatively fixed (10 actions); more complex combo systems may require extension.

## Related Work & Insights

- The approach shares conceptual similarities with robot VLA models such as RT-2 but incorporates real-time optimizations tailored to game environments.
- The AoT format is generalizable to other embodied AI scenarios requiring real-time decision-making, such as autonomous driving and robotic manipulation.
- The truncated inference strategy is broadly applicable to any scenario requiring reduced VLM inference latency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First efficient VLA targeting 3D ARPG combat; AoT combined with truncated inference is a notable contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dimensional evaluation (benchmarks, real-world combat, cross-game transfer) with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-designed figures and tables, coherent narrative.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for real-time decision-making in game AI and embodied intelligence.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](../../CVPR2026/robotics/mergevla_crossskill_model_merging_toward_a_general.md)
- [\[ICCV 2025\] DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)
- [\[NeurIPS 2025\] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](../../NeurIPS2025/robotics/cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)
- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](../../CVPR2026/robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)

<!-- RELATED:END -->
