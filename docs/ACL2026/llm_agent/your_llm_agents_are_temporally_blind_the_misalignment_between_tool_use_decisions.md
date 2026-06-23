---
title: >-
  [Paper Note] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception
description: >-
  [ACL 2026][LLM Agent][Paper Note] This paper reveals the "Temporal Blindness" of LLM Agents in multi-turn interactions—their inability to adjust tool-calling decisions based on the actual time elapsed between messages—and constructs the TicToc benchmark to evaluate this issue.
tags:
  - ACL 2026
  - LLM Agent
date: 2026-05-08
content_hash: 50f4ab874219f984
---
# Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.23853](https://arxiv.org/abs/2510.23853)  
**Code**: [GitHub](https://github.com/chengez/TicToc)  
**Area**: LLM Agent / Tool Use  
**Keywords**: Temporal Blindness, Tool Calling Decisions, Human Preference Alignment, Multi-turn Dialogue, Time Sensitivity

## TL;DR

This paper reveals the "Temporal Blindness" of LLM Agents in multi-turn interactions—their inability to adjust tool-calling decisions based on the actual time elapsed between messages—and constructs the TicToc benchmark to evaluate this issue.

## Background & Motivation

**Background**: LLM Agents are increasingly utilized for task execution in dynamic environments, retrieving real-time information by calling external tools (e.g., search engines, databases). Existing tool-use evaluations primarily focus on the **accuracy** of calls (whether the correct tools and parameters were used) but neglect the question of **when a call should be initiated**.

**Limitations of Prior Work**: LLM Agents default to assuming that the context is static, failing to consider the real-world time elapsed between messages. This leads to two failure modes: (1) **Over-reliance**—excessive trust in outdated context, leading to skipped tool calls and erroneous outputs; (2) **Under-reliance**—repeatedly calling tools for stable information (such as the Earth's radius), resulting in unnecessary latency.

**Key Challenge**: Humans naturally integrate the passage of time into decision-making—knowing when data needs to be re-queried and when previous information remains reliable. However, LLM Agents lack this temporal awareness; even when explicit timestamps are provided, they fail to utilize them effectively.

**Goal**: (1) Systematically identify and quantify the temporal blindness issue in LLM Agents; (2) construct the TicToc evaluation benchmark; (3) explore mitigation strategies.

**Core Idea**: Temporal blindness is a fundamental limitation of LLM Agents that cannot be addressed by simple prompt engineering; specialized post-training alignment is required for effective mitigation.

## Method

### Overall Architecture

This work does not propose a new model but instead constructs the TicToc benchmark and diagnostic framework to systematically answer whether LLM Agents decide to use tools based on elapsed real-world time. The input consists of multi-turn dialogue trajectories with timestamps. Human preference labeling is used to establish the ground truth for "whether to call a tool or respond directly" at each step. The output includes performance evaluations of 18 LLMs and a decomposition of failure root causes. TicToc covers 76 scenarios across high, medium, and low time sensitivity, generating multiple versions by injecting different time intervals to isolate "temporal change" as a variable.

### Key Designs

**1. TicToc Dataset Construction: A tool-use benchmark treating "time" as a controllable variable**

Existing tool-calling evaluations focus solely on accuracy without a temporal dimension. Consequently, the authors designed multi-turn scenarios incorporating temporal changes. The 76 scenarios are categorized into low (29), medium (25), and high (22) time sensitivity, covering both read-only and read-write interaction modes. Eight dialogue variants (repeat query, contrast, multi-retrieval single query, simple reasoning, retry after failure, user confirmation, repeat request, resource exhaustion) were defined to cover typical decision contexts. Crucially, each trajectory is injected with small, medium, and large time intervals, generating 5592 samples. Each sample was voted on by at least five annotators, achieving a Krippendorff's alpha of 0.8574, indicating high consensus.

**2. Temporal Blindness Diagnostic Analysis: Pinpointing why models fail to use temporal information**

A total score like the Normalized Alignment Rate (NAR) is insufficient to explain failures. The authors analyzed the actual use of temporal information in reasoning chains: timestamps appear in less than 4% of reasoning trajectories, the keyword "timestamp" appears in less than 1.5%, and all time-related terms combined account for less than 15%. Models rarely reference time actively during deliberation. Furthermore, they discovered a "thought-response inconsistency": models often decide to call a tool during reasoning but proceed to provide a direct answer in the final output. This analysis attributes "incorrect calling" to the root cause of failing to integrate time into the decision-making process.

**3. Alignment Strategy Exploration: From identifying the problem to preliminary solutions**

After confirming that temporal blindness is a pervasive defect, the authors compared two mitigation paths: naive prompt engineering yielded minimal results for most models, indicating this is not a prompt-level issue. Conversely, post-training alignment (Supervised Fine-Tuning) using a subset of TicToc significantly improved temporal awareness. This comparison concludes that prompts are ineffective while post-training is promising.

### Loss & Training Strategy

The evaluation metric is the Normalized Alignment Rate $NAR = \frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$, where 50% is equivalent to random guessing. Post-training alignment utilizes a subset of TicToc for supervised fine-tuning. During evaluation, all models use a temperature of $0$ (except for Qwen3 in reasoning mode).

## Key Experimental Results

### Main Results

| Condition | Highest NAR | Description |
|-----------|-------------|-------------|
| No Timestamp | ~55% | Nearly random guessing |
| With Timestamp | <65% | Best models still perform poorly |

### Key Analysis

| Analysis Dimension | Finding | Description |
|-----------|---------|-------------|
| Dialogue Length | Positively correlated with tool call frequency | Models use "turn count" heuristics instead of "time" |
| Reasoning Mode (CoT) | Almost no improvement | Temporal awareness is not a reasoning-only problem |
| Thought-Response Inconsistency | Up to 61.26% FP | Severe disconnection between reasoning and action |
| Time Sensitivity Levels | Models fail uniformly across levels | Not a scenario-specific issue |

### Key Findings
- **No model** achieved an NAR exceeding 65% after being provided with timestamps, indicating that temporal blindness is a pervasive and severe issue.
- Models treat **dialogue turns** rather than **actual time** as a heuristic for information "expiration."
- Reasoning (CoT) fails to improve temporal alignment because models do not spontaneously reference temporal information during the reasoning process.
- Post-training alignment demonstrates **significant potential** and represents a viable path to mitigating temporal blindness.

## Highlights & Insights
- **Proactive Problem Identification**: Temporal blindness is a critical deficiency in Agent capability that was previously ignored.
- **Bidirectional Analysis of Over/Under-reliance**: The study analyzes both the failure to call tools when necessary and the execution of unnecessary redundant calls.
- **Deep Analysis of Reasoning Trajectories**: Root causes are revealed by statistically analyzing temporal keywords within reasoning chains.
- **Discovery of "Thought-Response Inconsistency"**: Reveals a systematic fracture between LLM reasoning and action.
- **Design Quality of the TicToc Dataset**: Diversified scenarios, timestamp injection methods, and human annotation quality control are meticulously implemented.

## Limitations & Future Work
- The evaluation primarily focuses on English scenarios; cross-lingual differences in temporal perception remain unknown.
- Detailed methods for post-training alignment and large-scale validation require further disclosure and expansion.
- Timestamps are provided in a fixed format (ISO 8601); the impact of different temporal representations was not explored.
- While diverse, the 76 scenarios are still limited; coverage of more domains needs enlargement.
- Future work should investigate how to systematically inject temporal awareness during the pre-training or alignment phases.

## Related Work & Insights
- **vs. Existing Tool Use Evaluation (e.g., ToolBench)**: Prior work focuses on "whether the correct tool was called," while this paper is the first to focus on "when a tool should be called."
- **vs. Temporal Reasoning Research**: Existing work studies LLM temporal reasoning in isolation, whereas this study links it to Agent tool-calling decisions.
- **vs. LLM Alignment**: Introduces temporal-aware tool-use decisions as a new dimension for alignment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to identify and systematically study temporal blindness in LLM Agents; the problem is significant and previously overlooked.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models, 76 scenarios, 5592 annotated samples, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, detailed dataset construction, and deep analysis.
- Value: ⭐⭐⭐⭐⭐ Reveals fundamental limitations in LLM Agents with direct implications for Agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](../../ICML2026/llm_agent/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ACL 2026\] ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"](toolgrad_efficient_tool-use_dataset_generation_with_textual_gradients.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)

</div>

<!-- RELATED:END -->
