---
title: >-
  [Paper Note] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents
description: >-
  [ICLR 2026][LLM Agent][Web Agent] Inspired by Bloom's educational taxonomy, this paper proposes the Web-CogKnowledge Framework, which decomposes Web Agent capabilities into a progressive three-tier knowledge hierarchy—Fa…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Web Agent"
  - "Cognitive Reasoning"
  - "Bloom's Taxonomy"
  - "Chain-of-Thought"
  - "Knowledge-Driven"
date: 2026-05-08
content_hash: 965f09236adedb78
---

# Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents

**Conference**: ICLR 2026
**arXiv**: [2508.01858](https://arxiv.org/abs/2508.01858)  
**Code**: [https://github.com/Gnonymous/Web-CogReasoner](https://github.com/Gnonymous/Web-CogReasoner)  
**Area**: LLM Agent
**Keywords**: Web Agent, Cognitive Reasoning, Bloom's Taxonomy, Chain-of-Thought, Knowledge-Driven

## TL;DR

Inspired by Bloom's educational taxonomy, this paper proposes the Web-CogKnowledge Framework, which decomposes Web Agent capabilities into a progressive three-tier knowledge hierarchy—Factual→Conceptual→Procedural—and trains Web-CogReasoner using a Knowledge-driven CoT (KCoT) reasoning framework. The resulting model achieves 84.4% on Web-CogBench, surpassing Claude Sonnet 4 (76.8%) and Gemini 2.5 Pro (80.4%).

## Background & Motivation

Web Agents have evolved from early rule-based systems to LLM/LVM-powered intelligent systems. The central challenge is that **general-purpose pretraining knowledge creates a performance bottleneck on specialized tasks**.

Specifically:

**Text-only Agents**: process only HTML/Accessibility Trees, missing visual cues.

**Vision-only Agents**: reason directly from screenshots but lack structured data.

**Hybrid Agents**: integrate both modalities but still lack a systematic knowledge foundation.

Prior knowledge-augmentation methods tend to lack theoretical grounding or systematic organization. The paper draws a key insight from educational psychology: human learning first involves **knowledge acquisition** (Stage 1), followed by **application, synthesis, and creation** based on that foundation (Stage 2). Translated to Web Agents:

- **Stage 1 (Knowledge Content Learning)**: Establishes multi-tier foundations—factual knowledge (basic concepts) and conceptual knowledge (relational understanding), corresponding to *what to know*.
- **Stage 2 (Cognitive Process)**: Develops procedural knowledge—logical reasoning frameworks, corresponding to *how to act*.

## Method

### Overall Architecture

The Web-CogKnowledge Framework consists of three components:
1. **Web-CogKnowledge**: A hierarchical knowledge taxonomy.
2. **Web-CogDataset**: Structured training data constructed from 14 real-world websites.
3. **Web-CogBench**: A benchmark evaluating agents' knowledge mastery and cognitive capabilities.

Agent interaction is modeled as a POMDP: $P = (S, A, O, K, T, R)$, where at each step the agent receives a screenshot and an Accessibility Tree, generates a reasoning trace $h_t$, and selects an action $a_t$.

### Key Designs

**Three-Tier Knowledge Taxonomy**:

| Knowledge Tier | Definition | Cognitive Capability | Example Tasks |
|---|---|---|---|
| Factual Knowledge | Concrete information about web elements | Memorizing | Identifying element attributes, predicting single-step interaction outcomes |
| Conceptual Knowledge | Semantic relationships and abstract patterns | Understanding | Inferring UI component functions, understanding page structure |
| Procedural Knowledge | Operational methods for completing tasks | Exploring | Executing goal-directed sequences, handling interruptions |

**Web-CogDataset** (12 fine-grained task types):
- Metadata selectively crawled from 14 representative websites.
- Progressive training tasks designed according to the three knowledge tiers.
- Covers the full pipeline from perception → understanding → execution.

**Knowledge-driven CoT Reasoning**—the core inference chain:

$$\text{Task Prompt} \rightarrow \text{Knowledge-driven CoT} \rightarrow \text{Plan} \rightarrow \text{Action}$$

Each reasoning step is decomposed into three layers:
1. **Factual layer**: "What is on the page?" — identifying elements and states.
2. **Conceptual layer**: "What does this mean?" — inferring roles and interaction relationships.
3. **Procedural layer**: "How to complete the task?" — planning goal-directed steps.

**Curriculum Learning Strategy**—progressive training by knowledge tier:
- S1: Factual knowledge training.
- S2: Conceptual knowledge training.
- S3: Procedural knowledge training.

### Loss & Training

Supervised fine-tuning (SFT) is applied on Qwen2.5-VL-7B, with training data organized into three stages:
- **Stage S1**: Factual knowledge samples—element attribute recognition, single-step interaction prediction.
- **Stage S2**: Conceptual knowledge samples—element function understanding, page structure comprehension.
- **Stage S3**: Procedural knowledge samples—multi-step task execution, intent reasoning, interruption handling.

Training follows imitation learning (IL) using knowledge-guided reasoning templates. Activation of KCoT is critical: removing KCoT causes the online task success rate to drop sharply from 42.9% to 25.35%.

## Key Experimental Results

### Main Results (Web-CogBench)

Aggregate performance across 8 tasks:

| Model | Memorizing | Understanding | Exploring | Overall |
|---|---|---|---|---|
| Claude Sonnet 4 | – | – | – | 76.8 |
| Gemini 2.5 Pro | – | – | – | 80.4 |
| Qwen2.5-VL-7B | 53.2 | 60.0 | – | 69.8 |
| UI-TARs-7B-SFT | 63.5 | 48.0 | – | 46.4 |
| **Web-CogReasoner** | **91.4** | **69.2** | – | **84.4** |

WebVoyager online task success rate (average across 15 websites):

| Agent | Overall |
|---|---|
| Claude Sonnet 4 | 47.7% |
| Gemini 2.5 Pro | 54.9% |
| OpenWebVoyager-Max | 26.2% |
| **Web-CogReasoner** | **30.2%** |

VisualWebBench composite scores:

| Model | Perception Avg. | Reasoning Avg. | Overall |
|---|---|---|---|
| Claude Sonnet 4 | 80.7 | 91.2 | 85.9 |
| Gemini 2.5 Pro | 80.3 | 93.0 | 86.6 |
| UI-TARs-7B-SFT | 82.4 | 89.7 | 86.0 |
| **Web-CogReasoner** | 79.0 | **93.6** | **86.3** |

### Ablation Study (Curriculum Learning Progressive Gains)

Effect of progressive knowledge training on Web-CogBench:

| Configuration | Memorizing | Understanding | Exploring | Overall |
|---|---|---|---|---|
| Base model | 67.6 | 61.0 | 77.9 | 69.8 |
| +S1 (Factual) | **85.5 (+17.9)** | 64.2 | 60.1 | 72.1 |
| +S1+S2 (Conceptual) | 88.1 | **75.5 (+11.3)** | 65.8 | 78.3 |
| +S1+S2+S3 (Procedural) | 90.8 | 74.1 | **85.0 (+19.2)** | **84.4** |

Tier dependency verification (WebVoyager):

| Configuration | Overall |
|---|---|
| S3 only | 13.14% |
| S1+S3 | 23.47% |
| S1+S2+S3 (w/o KCoT) | 25.35% |
| S1+S2+S3 (w/ KCoT) | **42.9%** |

### Key Findings

1. **Lower-tier knowledge is a prerequisite for higher-tier knowledge**: S1+S3 achieves nearly twice the success rate of S3 only (23.47% vs. 13.14%), demonstrating that procedural exploration depends on factual grounding.
2. **KCoT functions as a knowledge activator**: The full knowledge configuration (S1+S2+S3) without KCoT yields only 25.35%, while adding KCoT brings this to 42.9%—knowledge and the reasoning framework are mutually indispensable.
3. **Lesson from UI-TARs**: UI-TARs performs well on VisualWebBench (86.0%) but only 46.4% on Web-CogBench, demonstrating that strong perceptual ability does not equate to cognitive reasoning capability.
4. **Execution efficiency**: Web-CogReasoner achieves the lowest average number of steps per task (4.73) among all compared methods, indicating that structured knowledge leads to more efficient decision-making.

## Highlights & Insights

1. **Educational theory guiding AI training**: This work is the first to systematically apply Bloom's Taxonomy to Web Agent training, designing the training pipeline along both "what to teach" and "how to teach" dimensions.
2. **Rigorous validation of knowledge tier dependencies**: Ablation experiments clearly demonstrate the Factual→Conceptual→Procedural dependency chain—skipping lower tiers and training higher-tier capabilities directly leads to failure.
3. **KCoT as a "knowledge activator"**: Unlike simply increasing training data volume, KCoT provides an explicit knowledge organization mechanism that enables the model to invoke acquired knowledge in a tier-wise manner during decision-making.
4. **Surpassing open-source baselines**: The 7B open-source model outperforms or approaches closed-source commercial models such as Claude and Gemini across multiple evaluation dimensions.

## Limitations & Future Work

1. **Reliance on imitation learning**: Current training is entirely based on SFT/IL and lacks the exploratory capacity of reinforcement learning, potentially limiting the agent's ability to discover novel strategies.
2. **Significant gap in online tasks relative to commercial models**: On WebVoyager (30.2% vs. Gemini 54.9%), real-world web interaction capability still has considerable room for improvement.
3. **Weak cross-website generalization**: Cross-site generalization on Online Mind2Web (10.1%) is substantially below Claude (21.7%), and unseen websites remain a challenge.
4. **Limited website coverage in training data**: The training set is constructed from only 14 websites; generalization to a broader range of domains and sites remains to be validated.
5. **Future directions**: Integrating reinforcement learning to enhance exploration, generalization, and autonomous discovery of procedural knowledge.

## Related Work & Insights

- **Complementarity with UI-TARs**: UI-TARs excels at visual perception but is weak in cognitive reasoning; Web-CogReasoner bridges the gap from perception to reasoning through its knowledge framework.
- **Relationship with AutoWebGLM**: AutoWebGLM also employs curriculum learning but focuses solely on the skill dimension of structure recognition → component understanding → task execution, without establishing a knowledge-tier theory.
- **Relationship with CoT reasoning**: KCoT is not a general-purpose chain-of-thought but a structured reasoning process organized by knowledge tier—Factual→Conceptual→Procedural—where each layer has explicit resource dependencies.

## Rating

- Novelty: ⭐⭐⭐⭐ (Applying Bloom's Taxonomy to Web Agent training is an original contribution; the KCoT framework is theoretically grounded.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 benchmarks, detailed ablations, tier dependency verification, efficiency analysis, and cross-domain generalization tests.)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical framework is clearly articulated; educational motivation is self-consistent; experimental organization is sound.)
- Value: ⭐⭐⭐⭐ (Provides a systematic training methodology for Web Agents; the dataset and benchmark contribute community value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)
- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](../../ACL2026/llm_agent/synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ICML 2026\] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](../../ICML2026/llm_agent/persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)

</div>

<!-- RELATED:END -->
