---
title: >-
  [Paper Note] "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents
description: >-
  [AAAI2026][Multimodal VLM][Computer Use Agent] This paper proposes a VLM-based autonomous task completion evaluation framework that judges whether a Computer Use Agent (CUA) has completed a task using only screenshots an…
tags:
  - "AAAI2026"
  - "Multimodal VLM"
  - "Computer Use Agent"
  - "Task Completion Evaluation"
  - "Vision-Language Model"
  - "Autonomous Feedback"
date: 2026-05-08
content_hash: 562a44f62833eca4
---

# "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents

**Conference**: AAAI2026
**arXiv**: [2511.20067](https://arxiv.org/abs/2511.20067)
**Code**: [martasumyk/vision-based-judge](https://github.com/martasumyk/vision-based-judge)
**Area**: Multimodal VLM
**Keywords**: Computer Use Agent, Task Completion Evaluation, Vision-Language Model, Autonomous Feedback

## TL;DR

This paper proposes a VLM-based autonomous task completion evaluation framework that judges whether a Computer Use Agent (CUA) has completed a task using only screenshots and task descriptions. Evaluation feedback is passed back to the agent for self-correction, achieving 73% evaluation accuracy and a 27% relative improvement in task success rate on macOS.

## Background & Motivation

Computer Use Agents (CUAs) are AI systems capable of autonomously operating digital interfaces by perceiving screen states and executing actions such as clicks and keystrokes to accomplish user goals. Unlike API-based agents, CUAs interact directly with GUIs in a service-agnostic manner, offering stronger generalizability and scalability.

However, CUAs face a critical challenge: **the inability to reliably determine whether a task has been completed**. This manifests as two failure modes:

1. **False completion**: The agent declares a task complete when it is not, undermining user trust.
2. **Unrecognized completion**: The task is actually complete but the agent fails to recognize it, leading to redundant actions and wasted computation.

Existing approaches such as OSWorld's script-based verification require hand-written validation scripts for each task, offering poor scalability. Web-based evaluation methods rely on structured representations such as HTML and cannot be directly transferred to desktop environments. Desktop interfaces contain more visual elements and lack unified structured representations, making evaluation substantially more difficult.

## Core Problem

How can one automatically determine whether a CUA has completed a given task using only screenshots and task descriptions—without scripts, logs, or structured representations—and leverage a feedback mechanism to improve the agent's task success rate?

## Method

### Overall Architecture

A three-stage evaluation-feedback pipeline is proposed:

1. **Task Execution**: The CUA executes actions in a macOS environment according to the task description, recording a complete trajectory (screenshots, actions, and reasoning processes).
2. **Outcome Evaluation**: The final screenshot and task description are fed to a VLM, which performs zero-shot binary classification (done/not done) and generates a natural language explanation.
3. **Feedback and Retry**: If the VLM determines the task is incomplete, its reasoning is passed back to the CUA, which continues from the current state rather than restarting from scratch.

### Dataset Construction

- Covers **42** macOS built-in applications (productivity, communication, multimedia, system utilities, developer tools, etc.)
- **30** specific tasks are defined per application, yielding **1,260** tasks in total (compared to 369 in OSWorld).
- Task difficulty spans a wide range: from simple operations (opening the Calendar app) to complex multi-step interactions (filtering free apps in the App Store and opening the first result).
- Tasks requiring private data or external configuration are excluded.

### Model Selection

- **Evaluated CUAs**: Claude Computer Use, OpenAI Operator, UI-TARS (open-source).
- **Evaluator VLMs**:
  - Closed-source: GPT-4o, Claude 3.5 Sonnet
  - Open-source: LLaVA-v1.5-7B, InternVL 2-8B, Qwen2-VL-7B

### Evaluation Protocol

Zero-shot prompting is adopted: the VLM performs binary classification (done/not done) based solely on the final screenshot and task description, while outputting a brief natural language rationale. The evaluator is fully decoupled from the executing agent to avoid evaluation bias.

## Key Experimental Results

### Evaluation Accuracy (vs. Human Annotation)

| Evaluator | OpenAI Operator | Anthropic CU | UI-TARS |
|---|---|---|---|
| GPT-4o | 0.61 | 0.69 | 0.64 |
| Claude 3.5 Sonnet | **0.69** | **0.71** | **0.73** |
| LLaVA-v1.5-7B | 0.56 | 0.61 | 0.52 |
| InternVL 2-8B | 0.62 | 0.67 | 0.61 |
| Qwen2-VL-7B | 0.68 | 0.66 | 0.70 |

- Claude 3.5 Sonnet achieves the best performance among closed-source models, reaching up to 73%.
- Qwen2-VL-7B performs best among open-source models, approaching closed-source levels.

### Success Rate Improvement from Feedback

- All VLM-based feedback mechanisms yield substantial performance gains.
- Closed-source evaluators achieve up to a **61%** relative improvement in task success rate.
- Agents with weaker baselines (e.g., Anthropic CU) benefit most from the feedback mechanism.
- A single retry yields an average relative success rate improvement of **27%**.

## Highlights & Insights

1. **Minimalist design**: Evaluation requires only the final screenshot and task description, with no need for scripts, logs, or structured representations, ensuring strong generalizability.
2. **Closed-loop feedback**: Evaluation results are directly passed back to the agent for self-correction, forming a complete evaluate–feedback–retry loop.
3. **Greater gains for weaker agents**: Agents with lower baselines benefit more from feedback, demonstrating that the mechanism effectively compensates for deficiencies in agents' self-awareness.
4. **Dataset contribution**: 1,260 macOS tasks covering 42 applications, approximately 3.4× the scale of OSWorld.
5. **Open-source viability**: 7B open-source models such as Qwen2-VL-7B perform comparably to closed-source models, lowering the barrier to deployment.

## Limitations & Future Work

1. **macOS only**: Windows, Linux, and other major desktop platforms are not covered; generalizability remains to be validated.
2. **Terminal-state evaluation only**: Only the final screenshot is assessed; correctness of intermediate steps cannot be judged. Future work could extend this to step-wise evaluation.
3. **Accuracy ceiling**: A maximum accuracy of 73% implies nearly 30% misclassification, which is insufficient for safety-critical scenarios.
4. **Single retry**: Only one feedback-retry cycle is evaluated; the effectiveness and convergence of multi-round feedback remain unexplored.
5. **Binary classification limitation**: The framework only determines done/not done, lacking fine-grained assessment of partial completion or completion quality.
6. **Absence of ablation studies**: The effects of screenshot quantity, temporal information, and prompt design on evaluation performance are not analyzed.

## Related Work & Insights

| Method | Environment | Evaluation Approach | Scripts Required | Feedback Loop |
|---|---|---|---|---|
| OSWorld | Multi-platform | Hand-written verification scripts | Yes | No |
| Pan et al. | Web/simulation | Structured representation + text reasoning | No | Yes |
| AutoEval (robotics) | Physical environment | VLM evaluation | No | No |
| **Ours** | macOS | VLM + screenshots | **No** | **Yes** |

The approach shares conceptual similarities with AutoEval from the robotics domain but is adapted to the unique challenges of desktop digital environments (absence of physical signals, diverse and complex interfaces). Compared to web-based evaluation methods, this work addresses desktop environments that lack structured representations such as HTML.

The following broader implications are noted:

- **Agent self-awareness**: The failure of CUAs to determine whether they have "finished" is an underappreciated problem; this paper explicitly models it as an independent evaluation task.
- **Evaluation as reward**: The output of the VLM evaluator can serve directly as a reward signal for RL training, replacing human annotation.
- **Multi-agent collaboration**: The decoupled evaluator–executor architecture is naturally suited to multi-agent systems and could evolve into a real-time monitoring and instant feedback framework.
- **Intersection with GUI agent research**: The evaluation framework generalizes to task completion judgment for any GUI agent, not limited to CUAs.

## Rating

- Novelty: ⭐⭐⭐ — The approach is straightforward yet effective; applying VLM-as-judge to CUA evaluation is a natural extension.
- Experimental Thoroughness: ⭐⭐⭐ — Three CUAs and five evaluators provide reasonable coverage, but ablation studies and in-depth analysis are lacking.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, the problem is well-defined, and technical details are sufficient.
- Value: ⭐⭐⭐ — Identifies an important problem and provides a concise solution, though accuracy and scale warrant further improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](../../CVPR2026/multimodal_vlm/adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](../../ACL2026/multimodal_vlm/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[AAAI 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)

</div>

<!-- RELATED:END -->
