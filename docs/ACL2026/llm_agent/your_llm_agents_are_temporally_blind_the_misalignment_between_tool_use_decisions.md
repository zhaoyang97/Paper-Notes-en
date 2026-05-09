---
title: >-
  [Paper Note] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception
description: >-
  [ACL 2026][LLM Agent][temporal blindness] This paper reveals a "Temporal Blindness" phenomenon in LLM Agents during multi-turn interactions — the inability to adjust tool-calling decisions based on the real elapsed time between messages — and constructs the TicToc benchmark to evaluate this problem.
tags:
  - ACL 2026
  - LLM Agent
  - temporal blindness
  - tool use decision
  - human preference alignment
  - multi-turn dialogue
  - time sensitivity
date: 2026-05-08
content_hash: 1c0877d28ea24cf8
---

# Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception

**Conference**: ACL 2026
**arXiv**: [2510.23853](https://arxiv.org/abs/2510.23853)
**Code**: [GitHub](https://github.com/chengez/TicToc)
**Area**: LLM Agent / Tool Use
**Keywords**: temporal blindness, tool use decision, human preference alignment, multi-turn dialogue, time sensitivity

## TL;DR

This paper reveals a "Temporal Blindness" phenomenon in LLM Agents during multi-turn interactions — the inability to adjust tool-calling decisions based on the real elapsed time between messages — and constructs the TicToc benchmark to evaluate this problem.

## Background & Motivation

**Background**: LLM Agents are increasingly deployed in dynamic environments for task execution, leveraging external tools (search engines, databases, etc.) to retrieve real-time information. Existing tool-use evaluations primarily focus on **accuracy** (whether the correct tool and parameters are invoked), while neglecting the question of **when** a tool should be called.

**Limitations of Prior Work**: LLM Agents implicitly assume that context is static, disregarding the real-world time elapsed between messages. This leads to two failure modes: (1) **over-reliance** — excessively trusting outdated context and skipping necessary tool calls, producing erroneous outputs; and (2) **under-reliance** — redundantly invoking tools for stable facts (e.g., Earth's radius), causing unnecessary latency.

**Key Challenge**: Humans naturally integrate temporal information into decision-making — knowing when to re-query and when prior information remains valid. LLM Agents lack this temporal awareness and fail to leverage explicit timestamps even when provided.

**Goal**: (1) Systematically identify and quantify the temporal blindness problem in LLM Agents; (2) construct the TicToc evaluation benchmark; (3) explore mitigation strategies.

**Core Idea**: Temporal blindness is a fundamental limitation of LLM Agents that cannot be resolved through simple prompt engineering — dedicated post-training alignment is required for effective mitigation.

## Method

### Overall Architecture

This paper constructs the TicToc dataset and evaluation framework to systematically study temporal blindness. TicToc contains 1,800+ multi-turn dialogue trajectories across 76 scenarios, covering dynamic environments with high, medium, and low time sensitivity. Each trajectory is expanded into three versions by injecting timestamps with different time intervals, and human preference annotations (invoke tool vs. respond directly) are collected. A total of 18 LLMs are evaluated on this benchmark to analyze the effect of temporal information on tool-calling decisions.

### Key Designs

1. **TicToc Dataset Construction**:

    - Function: Provides a benchmark for systematically evaluating the temporal alignment capability of LLM tool-calling decisions.
    - Mechanism: (a) 76 scenarios spanning low (29), medium (25), and high (22) time sensitivity, covering both read-only and read-write interaction modes; (b) 8 dialogue variant types defined (repeated queries, comparisons, multi-retrieval single-query, simple reasoning, retry after failure, user confirmation, repeated requests, resource exhaustion); (c) three timestamp intervals (small/medium/large) injected per trajectory, generating 5,592 samples; (d) each sample annotated by at least 5 annotators, with Krippendorff's alpha = 0.8574 indicating high inter-annotator agreement.
    - Design Motivation: Existing tool-use evaluations lack a temporal dimension, necessitating purpose-built multi-turn dialogue scenarios that incorporate temporal variation.

2. **Temporal Blindness Diagnostic Analysis**:

    - Function: In-depth analysis of why models fail to utilize temporal information.
    - Mechanism: Analyzes the usage of temporal information in reasoning chains: timestamps appear in fewer than 4% of reasoning traces, the keyword "timestamp" appears in fewer than 1.5%, and time-related vocabulary appears in fewer than 15%. A "think-answer inconsistency" phenomenon is also identified — models decide to call a tool during reasoning but produce a direct answer in the final output.
    - Design Motivation: Understanding the root cause of failure is necessary to guide effective improvements.

3. **Alignment Strategy Exploration**:

    - Function: Explores both prompt engineering and post-training strategies to mitigate temporal blindness.
    - Mechanism: Naïve prompting methods prove ineffective for most models, whereas dedicated post-training alignment (fine-tuning on a subset of TicToc) effectively improves temporal awareness.
    - Design Motivation: Provides a complete pipeline from problem identification to preliminary solutions.

### Loss & Training

The evaluation metric is the Normalized Alignment Rate $NAR = \frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$, where 50% corresponds to random chance. Post-training alignment employs supervised fine-tuning on a subset of TicToc. All models are evaluated at Temperature = 0 (except Qwen3 in reasoning mode).

## Key Experimental Results

### Main Results

| Condition | Best NAR | Note |
|-----------|----------|------|
| Without timestamps | ~55% | Near random chance |
| With timestamps | <65% | Best models still perform poorly |

### Key Findings (Analysis)

| Analysis Dimension | Finding | Note |
|--------------------|---------|------|
| Dialogue length | Positively correlated with tool-call frequency | Models use "turn count" as a heuristic instead of elapsed time |
| Reasoning mode (CoT) | Negligible improvement | Temporal awareness is not a reasoning problem |
| Think-answer inconsistency | Up to 61.26% of false positives | Severe disconnect between reasoning and action |
| Time-sensitivity stratification | Uniform failure across high/medium/low sensitivity | Not a scenario-specific issue |

### Key Findings
- **No model** achieves a NAR above 65% when timestamps are provided, indicating that temporal blindness is pervasive and severe.
- Models use **dialogue turn count** rather than **actual elapsed time** as a heuristic for information "expiration."
- Chain-of-thought reasoning fails to improve temporal alignment, as models do not spontaneously reference temporal information during the reasoning process.
- Post-training alignment demonstrates **strong potential** as a viable path toward resolving temporal blindness.

## Highlights & Insights
- **Forward-looking problem identification**: Temporal blindness is a previously entirely overlooked yet critically important capability gap in LLM Agents.
- **Bidirectional analysis of over- and under-reliance**: The paper analyzes not only cases where tool calls are missed but also unnecessary repeated invocations.
- **In-depth analysis of reasoning traces**: Quantifying temporal keywords in reasoning chains exposes the root cause of the problem.
- **Discovery of think-answer inconsistency**: Reveals a systematic disconnect between LLM reasoning and action.
- **TicToc dataset design quality**: Scenario diversity, timestamp injection methodology, and human annotation quality control are all carefully designed.

## Limitations & Future Work
- Evaluation is primarily conducted in English; cross-lingual differences in temporal awareness remain unexplored.
- Detailed methodology and large-scale validation of post-training alignment warrant further investigation.
- Timestamps are provided in a fixed format (ISO 8601); the impact of alternative temporal representations is not explored.
- Although diverse, the scenario set remains limited (76 scenarios); broader domain coverage is needed.
- Future work should investigate how to systematically incorporate temporal awareness during pre-training or alignment stages.

## Related Work & Insights
- **vs. Existing tool-use evaluations (ToolBench, etc.)**: Prior work focuses on "whether the correct tool is called"; this paper is the first to address "when a tool should be called."
- **vs. Temporal reasoning research**: Prior work studies LLM temporal reasoning in isolated settings, without connecting it to tool-calling decisions in an Agent context.
- **vs. LLM alignment**: Introduces temporally-aware tool-calling decisions as a new dimension of the alignment problem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to identify and systematically study temporal blindness in LLM Agents; the problem is important and had been entirely overlooked.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models, 76 scenarios, 5,592 annotated samples, multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation, thorough dataset construction methodology, and in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Reveals a fundamental capability gap in LLM Agents with direct implications for Agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)
- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)
- [\[ACL 2026\] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)

</div>

<!-- RELATED:END -->
