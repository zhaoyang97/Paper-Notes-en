---
title: >-
  [Paper Note] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents
description: >-
  [ICLR2026][LLM Agent][Web Agent] Web-CogReasoner draws on Bloom's Taxonomy in educational psychology to decompose Web Agent capabilities into a three-tier hierarchy of factual, conceptual, and procedural knowledge, constructing a structured knowledge-driven Chain-of-Thought reasoning framework that substantially outperforms existing methods on web navigation tasks.
tags:
  - ICLR2026
  - LLM Agent
  - Web Agent
  - Cognitive Reasoning
  - Bloom's Taxonomy
  - Knowledge-Driven CoT
  - Curriculum Learning
date: 2026-05-08
content_hash: 9d7a5421bc93b1b1
---

# Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents

**Conference**: ICLR2026
**arXiv**: [2508.01858](https://arxiv.org/abs/2508.01858)
**Code**: [github.com/Gnonymous/Web-CogReasoner](https://github.com/Gnonymous/Web-CogReasoner)
**Area**: LLM Agent
**Keywords**: Web Agent, Cognitive Reasoning, Bloom's Taxonomy, Knowledge-Driven CoT, Curriculum Learning

## TL;DR
Web-CogReasoner draws on Bloom's Taxonomy in educational psychology to decompose Web Agent capabilities into a three-tier hierarchy of factual, conceptual, and procedural knowledge, constructing a structured knowledge-driven Chain-of-Thought reasoning framework that substantially outperforms existing methods on web navigation tasks.

## Background & Motivation
- Multimodal large language models have advanced Web Agent development, yet general-domain pretraining knowledge remains a performance bottleneck on specialized tasks.
- Existing knowledge augmentation approaches lack systematicity or theoretical grounding.
- This paper takes Bloom's Taxonomy—a foundational educational theory—as its starting point, decomposing learning into two stages:
    - **Phase 1 (Knowledge Content Learning)**: accumulating factual and conceptual knowledge → addressing "what to learn"
    - **Phase 2 (Cognitive Process)**: developing procedural knowledge → addressing "how to act"
- This mirrors the human learning trajectory: first acquiring knowledge through education, then learning to apply, analyze, and create based on that knowledge and experience.
- Core problem: how to systematically inject hierarchical knowledge into Web Agents to endow them with cognitive reasoning capabilities?

## Method

### 1. Web-CogKnowledge Framework
A three-tier knowledge hierarchy grounded in Bloom's Taxonomy:

**Factual Knowledge**:
- Concrete information extracted from webpage content
- e.g., identifying attributes of web elements, predicting the direct outcome of a single interaction

**Conceptual Knowledge**:
- Semantic relationships and abstract patterns within webpage content and structure
- e.g., inferring the function of interface components, understanding the overall purpose and structure of a webpage

**Procedural Knowledge**:
- Actionable knowledge for completing specific tasks through interaction
- e.g., executing goal-directed action sequences, inferring user intent, handling unexpected interruptions

### 2. Web-CogDataset
- Multimodal metadata crawled from 14 mainstream real-world websites
- 12 fine-grained, progressively challenging task types:
    - Factual tier: element attribute recognition, next-page prediction, source element prediction
    - Conceptual tier: element understanding, webpage understanding
    - Procedural tier: user intent prediction, popup dismissal, single-step exploration, etc.
- Data progressively deepens from perception → understanding → action

### 3. Web-CogBench
- 876 test instances spanning 8 task types
- Evaluation across three cognitive dimensions:
    - **Memorizing**: assesses the ability to recall specific information (corresponding to factual knowledge)
    - **Understanding**: assesses semantic interpretation ability (corresponding to conceptual knowledge)
    - **Exploring**: assesses the ability to plan and execute goal-directed actions (corresponding to procedural knowledge)
- Multiple evaluation metrics including ROUGE-L, Accuracy, and LVM Judge

### 4. Web-CogReasoner Reasoning Framework
Modeled as a POMDP: $P = (S, A, O, K, T, R)$

**Knowledge-driven CoT reasoning process**:
$$\textbf{Task Prompt} \rightarrow \textbf{Knowledge-driven CoT} \rightarrow \textbf{Plan} \rightarrow \textbf{Action}$$

Reasoning proceeds through three tiers:
1. **Factual Reasoning**: "What is on the page?" → identifying page elements and states
2. **Conceptual Reasoning**: "What does this mean?" → inferring roles and interaction relationships
3. **Procedural Reasoning**: "How to complete the task?" → planning goal-directed steps

The backbone model is Qwen2.5-VL-7B, fine-tuned via supervised training on Web-CogDataset.

## Key Experimental Results

### Web-CogBench Comprehensive Evaluation

| Model | Elem Attr | Next Page | Source Elem | Elem Und | WebPage Und | User Intent | Popup | Step Exp | Overall |
|------|-----------|-----------|-------------|----------|-------------|-------------|-------|----------|---------|
| Claude Sonnet 4 | 79.7 | 93.5 | 62.5 | 62.8 | 54.3 | 64.7 | 100 | 96.8 | 76.8 |
| Gemini 2.5 Pro | 79.8 | 94.6 | 84.4 | 62.6 | 73.5 | 51.9 | 96.6 | 98.4 | 80.4 |
| UI-TARs-7B-SFT | 63.5 | 88.0 | 31.3 | 48.0 | 48.0 | 32.4 | 25.9 | 33.9 | 46.4 |
| **Web-CogReasoner** | **91.4** | 93.5 | **87.5** | **69.2** | **79.0** | 61.4 | 98.3 | 95.2 | **84.4** |

### VisualWebBench

| Model | Perception Avg | Reasoning Avg | Overall Avg |
|------|----------------|---------------|-------------|
| Claude Sonnet 4 | 80.7 | 91.2 | 85.9 |
| UI-TARs-7B-SFT | 82.4 | 89.7 | 86.0 |
| **Web-CogReasoner** | 79.0 | **93.6** | **86.3** |

### Ablation Study on Curriculum Learning

| Configuration | Memorizing | Understanding | Exploring | Overall |
|------|------------|---------------|-----------|---------|
| Qwen2.5-VL-7B (Base) | 67.6 | 61.0 | 77.9 | 69.8 |
| + Factual (S1) | 85.5 (+17.9) | 64.2 | 60.1 | 72.1 |
| + Conceptual (S2) | 88.1 | 75.5 (+11.3) | 65.8 | 78.3 |
| + Procedural (S3) | 90.8 | 74.1 | 85.0 (+19.2) | **84.4** |

### Online Web Tasks — WebVoyager Success Rate

| Agent | Overall |
|-------|---------|
| Claude Sonnet 4 | 47.7% |
| Gemini 2.5 Pro | 54.9% |
| OpenWebVoyager_Max | 26.2% |
| **Web-CogReasoner** | **30.2%** |

Best among open-source models, with the fewest average steps (4.73 steps/successful task).

## Highlights & Insights
1. **Solid theoretical grounding**: Bloom's Taxonomy provides a principled basis for decomposing agent capabilities into trainable knowledge tiers.
2. **Structured knowledge-driven CoT**: Rather than simple prompt engineering, the knowledge hierarchy is embedded directly into the reasoning process.
3. **Pronounced curriculum learning effect**: Ablation experiments clearly quantify each knowledge tier's contribution, yielding an overall gain of 14.6%.
4. **Comprehensive evaluation framework**: Web-CogBench systematically assesses agents along three cognitive dimensions—memorizing, understanding, and exploring.
5. **Surpasses closed-source models**: The 7B model outperforms Claude Sonnet 4 and Gemini 2.5 Pro on Web-CogBench.
6. **Efficient execution**: The fewest average steps in online tasks indicate that knowledge-driven reasoning reduces redundant exploration.

## Limitations & Future Work
- Performance on online tasks (Mind2Web cross-web) lags behind Claude/Gemini; generalizing to entirely unseen websites remains challenging.
- The 14 training websites may offer insufficient coverage to encompass all webpage types.
- The POMDP formulation lacks an explicit memory mechanism, which may limit long-horizon decision-making.
- Integration with reinforcement learning training has not been explored; the current approach relies solely on SFT.
- The partitioning of knowledge tiers may be overly rigid, as the three knowledge types are often intertwined in real-world tasks.

## Related Work & Insights
- Compared to UI-TARs (strong visual perception but weak cognitive reasoning): Web-CogReasoner leads substantially on cognitive tasks (84.4% vs. 46.4%), demonstrating that visual perception alone is insufficient.
- Compared to OpenWebVoyager (requires retraining on target websites): Web-CogReasoner generalizes through structured knowledge alone.
- Compared to general LVMs (Claude/Gemini): the open-source 7B model is competitive on knowledge-intensive tasks.
- Compared to earlier curriculum learning approaches such as AutoWebGLM: Web-CogReasoner offers a more systematic knowledge hierarchy.

### Broader Implications
- Bloom's Taxonomy could be extended to other agent domains (e.g., GUI agents, robotic manipulation).
- The knowledge-driven CoT paradigm is applicable to other tasks requiring structured reasoning.
- The progressive factual → conceptual → procedural training strategy offers a valuable reference for multi-stage training pipelines.
- The cognitive-dimension evaluation methodology of Web-CogBench can inspire better agent benchmark design.

## Rating
- Novelty: ⭐⭐⭐⭐ (The integration of Bloom's Taxonomy with Web Agents is novel, though knowledge injection itself is not an entirely new idea)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 benchmarks, detailed ablations, component analysis, and online task evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though the paper is lengthy and some sections could be condensed)
- Value: ⭐⭐⭐⭐ (Provides a systematic methodology for knowledge-based Web Agent training; Web-CogBench has practical utility)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](../../ACL2026/llm_agent/synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](../../ACL2026/llm_agent/expseek_self-triggered_experience_seeking_for_web_agents.md)

<!-- RELATED:END -->
