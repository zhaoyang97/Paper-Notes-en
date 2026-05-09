---
title: >-
  [Paper Note] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis
description: >-
  [AAAI 2026][LLM Agent][Multi-Agent Systems] This paper proposes COACH — a reconfigurable multi-agent framework built on a shared backbone model — that achieves role specialization via intent-driven strategy orchestration and structured CoT fine-tuning, significantly outperforming generalist models such as Gemini 2.5 Pro on both QA and summarization tasks in badminton video analysis.
tags:
  - AAAI 2026
  - LLM Agent
  - Multi-Agent Systems
  - Video QA
  - Video Summarization
  - Structured Chain-of-Thought
  - Role Specialization
date: 2026-05-08
content_hash: ed8e1cfff018188d
---

# COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis

**Conference**: AAAI 2026
**arXiv**: [2512.01853](https://arxiv.org/abs/2512.01853)
**Code**: [Project Page](https://aiden1020.github.io/COACH-project-page/)
**Area**: LLM Agents / Sports Video Analysis
**Keywords**: Multi-Agent Systems, Video QA, Video Summarization, Structured Chain-of-Thought, Role Specialization

## TL;DR

This paper proposes COACH — a reconfigurable multi-agent framework built on a shared backbone model — that achieves role specialization via intent-driven strategy orchestration and structured CoT fine-tuning, significantly outperforming generalist models such as Gemini 2.5 Pro on both QA and summarization tasks in badminton video analysis.

## Background & Motivation

**Background**: Sports video analysis requires multi-level temporal understanding spanning from the micro scale (strokes, rallies) to the macro scale (match-level tactics). While video-language models such as BLIP-2 and Video-LLaVA have advanced general video understanding, they perform poorly on long-duration, hierarchically structured content.

**Limitations of Prior Work**: End-to-end single-model approaches suffer from three key problems: (a) *high redundancy cost* — each application requires an independent model and training pipeline with no component reuse; (b) *single temporal scale lock-in* — models trained at the rally level cannot transfer to match-level understanding; (c) *opaque reasoning* — decision processes are hidden, making them difficult to interpret, verify, or extend with new reasoning modules.

**Key Challenge**: Sports video analysis simultaneously demands fine-grained short-term reasoning (per-rally QA) and long-horizon global integration (match summarization), yet existing models can only address one of these dimensions.

**Goal**: To construct a flexible, scalable, and interpretable cross-task intelligent sports video analysis system.

**Key Insight**: Each agent is designed as an independent "cognitive tool"; an adaptive pipeline is built through iterative invocation and flexible composition, rather than training a single end-to-end model.

**Core Idea**: All agents share a single backbone model. Through "structured CoT fine-tuning," distinct "multi-persona reasoning modes" are induced within the same model weights, with behavioral separation achieved via instruction conditioning and role-specific chain-of-thought templates.

## Method

### Overall Architecture

COACH adopts a single shared backbone (Flan-T5-XL) with multi-strategy orchestration. The system comprises three core agents (Orchestrator, Localizer, Critic) and three base tools (Vision Module, Retrieval Tool, Media Synthesis Tool). Predefined collaboration strategies (Analytical QA or Generative Summarization) are selected according to user intent, with agents operating dynamically within each strategy.

### Key Designs

**Module 1: Intent-Driven Strategy Orchestration**

- **Function**: Automatically selects a predefined collaboration workflow based on the semantic intent of the user query.
- **Mechanism**: The Orchestrator's core capability is "intent-to-strategy mapping" (learned via supervised imitation learning). A "strategy" is not a dynamic decision model but a predefined, goal-oriented collaboration plan analogous to an SOP. Two primary strategies: (a) **Analytical Rally QA Strategy** — the Orchestrator invokes the Retriever → the Critic verifies factual consistency → results are synthesized into a conclusive answer; (b) **Generative Video Summarization Strategy** — the Orchestrator plans the structure → batch-invokes the Localizer → the Critic verifies → the Orchestrator synthesizes a narrative script → the Media Synthesis Tool produces the video.
- **Design Motivation**: Compared to online planning paradigms, predefined strategies are more stable and reproducible. Agents within a strategy can still operate dynamically based on context, balancing stability with adaptability.

**Module 2: Structured CoT Fine-Tuning (Multi-Role Specialization)**

- **Function**: Enables functionally distinct role behaviors within a single shared backbone model, avoiding task interference.
- **Mechanism**: Fine-tuning is performed by mixing multiple sets of role-specific structured CoT instruction data, inducing "multi-persona reasoning modes." Role switching relies not on different weights but on different CoT structural templates. The chain-of-thought designs for the three roles are distinctly different:
  - **Orchestrator**: A high-degree-of-freedom "strategist" using conditional routing CoT — analyzes query intent first; single-step reasoning for text tasks; multi-step visual reasoning for video tasks; decomposes summarization tasks into sub-queries.
  - **Localizer**: A high-precision "executor" using a rigid "observe → report" CoT — strictly parses instructions and reports only factual temporal positions; reports an empty set if nothing is found.
  - **Critic**: An adversarial "fact-checking engine" using an "analyze assertion → compare evidence → render verdict" CoT — performs backward verification as opposed to the Orchestrator's forward reasoning.
- **Design Motivation**: The central challenge of a single-backbone multi-role design is role conflict. By constraining reasoning patterns through structured CoT, drastically different behaviors can be realized within the same weights while reducing deployment cost (only one model required).

**Module 3: Instruction Conditioning (Role-Switching Mechanism)**

- **Function**: Guides the model into a specific reasoning subspace at inference time via role-specific instruction prefixes.
- **Mechanism**: A role description is prepended at each invocation (e.g., "You are the Localizer agent responsible for temporal localization"), leveraging the conditional capability established during instruction fine-tuning to trigger the corresponding reasoning mode.
- **Design Motivation**: This is a lightweight mechanism for achieving multi-role behavior from a single backbone, incurring no additional parameter overhead or model-switching cost.

### Loss & Training

- Backbone model: Flan-T5-XL
- Visual encoder: TC-CLIP-B/16 + Q-Former
- Fully supervised training for 2 epochs on the COACH-Data dataset; learning rate 2e-5, batch size 16
- The dataset is built upon ShuttleSet, comprising approximately 19,000 annotated strokes from 22 badminton matches

## Key Experimental Results

### Main Results

Analytical Rally QA Task:

| Model | Action Classification (EM%) | Action Counting (EM%) | Summarization (ROUGE-L) | Temporal Localization (Hit@1%) | Knowledge QA (EM%) | Knowledge QA (F1%) |
|---|---|---|---|---|---|---|
| Gemini 2.5 Pro | 24.20 | 37.60 | 23.55 | 18.12 | 7.04 | 13.73 |
| **COACH (w/ Critic)** | **85.60** | **79.20** | **33.56** | **76.66** | **63.97** | **76.21** |

Action classification +61%, temporal localization +59%, F1 +62% — a decisive advantage of the specialist model over the generalist.

### Ablation Study

| Variant | Action Classification | Action Counting | Temporal Localization (Hit@1) |
|---|---|---|---|
| COACH w/o Critic | 82.20 | 79.60 | 76.94 |
| COACH w/ Critic | 85.60 | 79.20 | 76.66 |

The Critic primarily improves action classification accuracy (+3.4%) and has minimal impact on localization tasks.

### Key Findings

- Gemini 2.5 Pro performs poorly on precision-demanding tasks (temporal localization accuracy of only 17.81%), indicating that generalist models produce unreliable reasoning that likely relies on textual priors rather than visual evidence.
- Structured CoT fine-tuning is the key factor — it grants the specialist model an overwhelming advantage on domain-specific tasks.
- The shared backbone design allows the system to support multiple roles with a single deployed model, offering significant deployment efficiency advantages.
- The generative summarization task cannot be directly compared with Gemini due to its inability to process ultra-long videos, indirectly validating the necessity of COACH's divide-and-conquer strategy.

## Highlights & Insights

- The "single backbone, multiple personas" design is elegant and concise — role separation is achieved through differences in CoT structure, eliminating the need for multi-model deployment.
- The Orchestrator's "intent routing as strategy selection" approach is more stable and reliable than dynamic planning, making it well-suited for production environments.
- The Localizer's "report empty set if not found" design is simple yet important, substantially reducing hallucination rates.
- The Critic's "backward reasoning" and the Orchestrator's "forward reasoning" form an adversarial pair — this opposing design is an effective approach to improving overall system reliability.

## Limitations & Future Work

- Validation is limited to badminton; transferring to other sports (e.g., football, basketball) requires reconstructing datasets and adjusting strategies.
- Flan-T5-XL has limited capacity as a backbone model and may be insufficient for more complex tactical analysis.
- Predefined strategies offer limited flexibility and may fail to handle entirely novel query types.
- GPT is used as a teacher model to generate CoT during data construction, potentially introducing biases inherited from the teacher.

## Related Work & Insights

- **MetaGPT** (Hong et al.): A role-coordinated multi-agent system that manages collaboration using SOPs.
- **AgentOrchestra** (Zhang et al.): Hierarchical orchestration and dynamic workflow composition.
- **HuggingGPT** (Shen et al.): LLM as a central controller that delegates tasks to specialized sub-models.
- Insight: For structured domain analysis tasks, predefined strategies combined with role specialization substantially outperform stepwise planning that relies on general-purpose large models.

## Rating

⭐⭐⭐⭐ (4/5)

The framework design is clear and elegant; the "shared backbone + structured CoT multi-role" approach is a technically insightful contribution. Experimental results are convincing. One point is deducted for the narrow application scope (badminton only) and the limited dataset scale, which leave generalization capability in need of further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)
- [\[AAAI 2026\] CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing](causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)
- [\[AAAI 2026\] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)

</div>

<!-- RELATED:END -->
