---
title: >-
  [Paper Note] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning
description: >-
  [ACL 2026][Multi-Agent][Generative Engine Optimization] This paper reframes Generative Engine Optimization (GEO) from instance-wise heuristic optimization to a strategy learning problem…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Generative Engine Optimization"
  - "Multi-agent framework"
  - "strategy reuse"
  - "citation faithfulness"
  - "visibility optimization"
date: 2026-05-08
content_hash: 52c94e7e1beda72e
---

# MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning

**Conference**: ACL 2026  
**arXiv**: [2604.19516](https://arxiv.org/abs/2604.19516)  
**Code**: [https://github.com/Wu-beining/MAGEO](https://github.com/Wu-beining/MAGEO)  
**Area**: Model Compression  
**Keywords**: Generative Engine Optimization, Multi-agent framework, strategy reuse, citation faithfulness, visibility optimization

## TL;DR

This paper reframes Generative Engine Optimization (GEO) from instance-wise heuristic optimization to a strategy learning problem, proposing the MAGEO multi-agent framework. The execution layer involves collaboration among four agents—Preference, Planning, Editing, and Evaluation—while the learning layer distills validated editing patterns into reusable, engine-specific strategy skills. By introducing the Twin Branch causal evaluation protocol and the DSV-CF dual-axis metrics, the framework significantly outperforms heuristic baselines across three mainstream engines.

## Background & Motivation

**Background**: Generative engines (e.g., ChatGPT, Gemini) are reshaping information acquisition by replacing search link lists with citation-anchored answers. Content creators need to optimize pages to gain citations in generated answers—namely Generative Engine Optimization (GEO).

**Limitations of Prior Work**: (1) Existing GEO methods optimize independently for each instance, failing to accumulate or transfer effective strategies; (2) Evaluation confuses surface visibility with semantic impact, allowing exposure gains to be accompanied by incorrect citations; (3) Engine preference modeling is coarse, lacking engine-specific strategy learning.

**Key Challenge**: Current GEO is trapped in an instance-wise trial-and-error cycle rather than evolving into a cumulative, skill-building process. Each optimization starts from scratch, failing to leverage past successes.

**Goal**: (1) Reframing GEO as a strategy learning problem; (2) Building a multi-agent framework capable of accumulating and reusing strategies; (3) Designing causally attributable evaluation methods.

**Key Insight**: A dual-layer architecture where the execution layer handles collaborative optimization and the learning layer distills reusable strategy skills from successful experiences.

**Core Idea**: Abstracting validated editing patterns into structured strategy skills (comprising applicability conditions, editing operations, and effect evaluations), storing them in a skill bank, and retrieving them for reuse in new tasks.

## Method

### Overall Architecture

MAGEO utilizes a dual-layer architecture. The execution layer consists of an iterative Generate-Evaluate-Select loop involving a Preference Agent (analyzing engine preferences), a Planning Agent (formulating revision strategies), an Editing Agent (executing specific modifications), and an Evaluation Agent (quality checks + faithfulness gating). The learning layer includes step-level memory (intra-session) and creator-level memory (inter-session), forming a strategy skill bank. The Twin Branch evaluation protocol is used for causal attribution.

### Key Designs

1.  **Twin Branch Evaluation Protocol**:
    - **Function**: Causally attributes the effects of content editing by eliminating interference from retrieval ranking fluctuations.
    - **Mechanism**: Freezes the retrieval list and creates two branches—the baseline branch maintains the original document, while the optimization branch replaces the target document with the optimized version. Comparing engine responses from both branches under the same retrieval list isolates the effect of the edit itself.
    - **Design Motivation**: In black-box engines, retrieval and generation are intertwined; without controlling the retrieval list, it is impossible to distinguish whether the document improved or the retrieval ranking changed.

2.  **Strategy Skill Bank (Skill Bank)**:
    - **Function**: Distills optimization experiences into reusable strategy skills.
    - **Mechanism**: A three-stage lifecycle: discovery (step-level memory records positive/negative effects of each edit), consolidation (inter-session extraction of consistently effective patterns into structured skills including engine type, scenario, editing operations, and effect metrics), and retrieval (matching skills by engine and scenario for new tasks). Capacity limits and eviction policies (based on usage frequency or recency) maintain scalability.
    - **Design Motivation**: Instance-wise optimization is wasteful; successful patterns on the same engine are often reusable. The skill bank realizes the leap from "experience to skill."

3.  **DSV-CF Dual-Axis Evaluation Metric**:
    - **Function**: Unifies the evaluation of semantic visibility and citation faithfulness.
    - **Mechanism**: $S_{DSV-CF} = \lambda \cdot \bar{S}_{SSV} + (1-\lambda) \cdot \bar{S}_{ISI} - \gamma(1-AA)$. SSV (Surface Semantic Visibility) aggregates word-level visibility, positional authority, citation prominence, and subjective impression. ISI (Intrinsic Semantic Impact) evaluates attribution accuracy, response faithfulness, key point coverage, and answer dominance. $\gamma$ controls the penalty for incorrect citations.
    - **Design Motivation**: Existing metrics focus either on exposure or quality and fail to penalize mis-citations. DSV-CF ensures that visibility gains must be accompanied by accurate attribution.

### Loss & Training

MAGEO is an LLM-based multi-agent reasoning framework and does not involve neural network training. It utilizes GPT-5.2 and Gemini-3 Pro as underlying and evaluation engines. The MSME-GEO-Bench benchmark covers real queries from 5 major domains and 15 sub-categories.

## Key Experimental Results

### Main Results

**DSV-CF Performance on Three Mainstream Engines**

| Method | GPT 5.2 SSV | GPT 5.2 ISI | Gemini-3 SSV | Gemini-3 ISI |
| :--- | :--- | :--- | :--- | :--- |
| No Optimization | Baseline | Baseline | Baseline | Baseline |
| GEO (Heuristic) | Moderate Gain | Mixed | Moderate Gain | Mixed |
| RAID | Gain | Gain | Gain | Gain |
| **MAGEO** | **Optimal** | **Optimal** | **Optimal** | **Optimal** |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Full MAGEO | Optimal | Complete framework |
| w/o Skill Bank | Decrease | Significant contribution of strategy reuse |
| w/o Preference Agent | Decrease | Importance of engine-specific modeling |
| w/o Evaluation Agent | Decrease + Faithfulness Collapse | Faithfulness gating is indispensable |
| w/o Twin Branch | Attribution Failure | Decrease in evaluation reliability |

### Key Findings

- Engine-specific preference modeling and strategy reuse are the two most critical components.
- The faithfulness gating of the Evaluation Agent is vital; without it, optimization may increase surface exposure through mis-citations.
- Strategy skills show good transferability across scenarios within the same engine, but cross-engine transferability is limited.
- Traditional SEO strategies (e.g., keyword stuffing) are ineffective or even harmful for generative engines.

## Highlights & Insights

- The paradigm shift from "instance-wise trial-and-error" to "strategy learning" is a significant theoretical contribution to the GEO field.
- The Twin Branch causal evaluation protocol addresses the fundamental challenge of evaluating black-box engines.
- The three-stage lifecycle design of the skill bank (discovery $\rightarrow$ consolidation $\rightarrow$ retrieval) is transferable to other agent systems requiring experience accumulation.

## Limitations & Future Work

- The effectiveness of strategy skills may decay as engines are updated.
- Evaluation relies heavily on LLM-as-a-Judge, which may introduce systematic biases.
- The query diversity of MSME-GEO-Bench remains limited.
- Future work could explore automatic skill updates and cross-engine transfer learning.

## Related Work & Insights

- **vs GEO/GEO-Bench**: Quantifies exposure but optimizes per instance without strategy accumulation; MAGEO adds a learning layer.
- **vs RAID**: Intent-aware but lacks strategy reuse; MAGEO achieves experience transfer via the skill bank.
- **vs AutoGEO**: Learns preference rules but does not accumulate cross-instance strategies; MAGEO's skill bank evolves continuously.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reframes GEO as strategy learning; both the skill bank and Twin Branch evaluation are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-engine evaluation, though real-world scenario validation is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear framework design and comprehensive metric definitions.
- Value: ⭐⭐⭐⭐ Provides a scalable, learning-driven paradigm for the GEO field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/multi_agent/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[AAAI 2026\] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](../../AAAI2026/multi_agent/safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](../../ICML2026/multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ICML 2026\] Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM](../../ICML2026/multi_agent/learning_multi-agent_coordination_via_sheaf-admm.md)

</div>

<!-- RELATED:END -->
