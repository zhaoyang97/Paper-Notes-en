---
title: >-
  [Paper Note] Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception
description: >-
  [ACL 2026][LLM Agent][Paper Note] Reveals the "Temporal Blindness" of LLM Agents in multi-turn interactions—the inability to adjust tool-use decisions based on the real time elapsed between messages—and constructs the TicToc benchmark to evaluate this issue.
tags:
  - ACL 2026
  - LLM Agent
date: 2026-05-08
content_hash: 1aebdb3182dc2e1e
---
# Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.23853](https://arxiv.org/abs/2510.23853)  
**Code**: [GitHub](https://github.com/chengez/TicToc)  
**Area**: LLM Agent / Tool Use  
**Keywords**: Temporal blindness, tool-use decision-making, human preference alignment, multi-turn dialogue, temporal sensitivity

## TL;DR

Reveals the "Temporal Blindness" of LLM Agents in multi-turn interactions—the inability to adjust tool-use decisions based on the real time elapsed between messages—and constructs the TicToc benchmark to evaluate this issue.

## Background & Motivation

**Background**: LLM Agents are increasingly utilized for task execution in dynamic environments, acquiring real-time information by calling external tools (search engines, databases, etc.). Existing tool-use evaluations primarily focus on the **accuracy** of calls (whether the correct tools and parameters were used) but neglect the question of **when they should be called**.

**Limitations of Prior Work**: LLM Agents assume a static context by default, failing to consider the real-world time elapsed between messages. This leads to two failure modes: (1) **Over-reliance**—excessive trust in outdated context, skipping necessary tool calls and producing erroneous outputs; (2) **Under-reliance**—repeatedly calling tools for stable information (e.g., the Earth's radius), causing unnecessary latency.

**Key Challenge**: Humans naturally integrate the passage of time into decision-making—knowing when a re-query is necessary and when previous information remains reliable. However, LLM Agents lack this temporal perception capability and cannot effectively utilize explicit timestamps even when provided.

**Goal**: (1) Systematically identify and quantify the temporal blindness issue in LLM Agents; (2) Construct the TicToc evaluation benchmark; (3) Explore mitigation strategies.

**Core Idea**: Temporal blindness is a fundamental limitation of LLM Agents that cannot be addressed by simple prompt engineering—specialized post-training alignment is required for effective mitigation.

## Method

### Overall Architecture

This work does not propose a new model but rather constructs the TicToc benchmark and diagnostic framework to systematically answer "whether LLM Agents decide whether to use tools based on real elapsed time." The input consists of multi-turn dialogue trajectories with timestamps. Human preference annotations establish the gold standard for "whether to call a tool or answer directly" at each moment. The output includes an assessment of temporal alignment capabilities across 18 LLMs and a breakdown of root causes for failure. TicToc covers 76 scenarios across high, medium, and low temporal sensitivity. Each trajectory is generated in multiple versions by injecting different time intervals to isolate "temporal change" as a controlled variable.

### Key Designs

**1. TicToc Dataset Construction: A Tool-Use Benchmark with Time as a Controlled Variable**

Existing tool-use evaluations only assess "correctness of calls" and completely lack a temporal dimension. Consequently, the authors designed multi-turn scenarios incorporating temporal changes. The 76 scenarios are categorized into low (29), medium (25), and high (22) temporal sensitivity, covering both read-only and read-write interaction modes. On top of this, 8 dialogue variants (repeated inquiry, contrast, multi-retrieval single inquiry, simple reasoning, retry after failure, user confirmation, repeated request, resource exhaustion) are defined to cover typical decision-making contexts. The key operation involves injecting small, medium, and large time intervals into each trajectory, resulting in 5,592 samples. Each sample was voted on by at least 5 annotators, with a Krippendorff's alpha of 0.8574 indicating high annotation consistency.

**2. Temporal Blindness Diagnostic Analysis: Locating Why Models Fail to Use Temporal Information**

Providing only the Normalized Alignment Rate (NAR, defined in the experimental section) as a total score is insufficient to explain failures. The authors further quantified the actual usage of temporal information in reasoning chains: timestamps appeared in reasoning trajectories at a rate of less than 4%, the "timestamp" keyword appeared at a rate below 1.5%, and all time-related vocabulary combined accounted for less than 15%—models rarely cite time actively during deliberation. Furthermore, they discovered a "think-act inconsistency" phenomenon: models might decide to call a tool during reasoning but ultimately output a direct answer. This analysis attributes the "incorrect call" symptom to the root cause of "failing to incorporate time into decision-making."

**3. Alignment Strategy Exploration: Moving from Problem Identification to Initial Solutions**

After confirming that temporal blindness is a widespread defect, the authors compared two mitigation paths: naive prompt engineering yielded minimal results for most models, indicating this is not a problem solvable by merely adjusting prompts. Conversely, post-training alignment (supervised fine-tuning) using a subset of TicToc significantly improved temporal perception. This comparison leads to the clear conclusion that "prompts are ineffective while post-training is promising," providing direction for future work.

### Loss & Training

The evaluation metric is the Normalized Alignment Rate $NAR = \frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$, where 50% is equivalent to random guessing. Post-training alignment utilized a subset of TicToc for supervised fine-tuning. During evaluation, all models used Temperature=0 (except for Qwen3 reasoning mode).

## Key Experimental Results

### Main Results

| Condition | Highest NAR | Description |
|------|---------|------|
| No Timestamps | ~55% | Near random guessing |
| With Timestamps | <65% | Best models still perform poorly |

### Key Analysis

| Analysis Dimension | Finding | Description |
|---------|------|------|
| Dialogue Length | Positively correlated with tool call frequency | Models use "turns" as a heuristic instead of "time" |
| Reasoning Mode (CoT) | Almost no improvement | Temporal perception is not a reasoning issue |
| Think-Act Inconsistency | FP rate as high as 61.26% | Severe disconnection between reasoning and action |
| Temporal Sensitivity Tiers | Models fail uniformly across high/medium/low sensitivity | Not a scenario-specific issue |

### Key Findings
- **No single model** exceeded an NAR of 65% even after being provided with timestamps, indicating that temporal blindness is a universal and serious issue.
- Models utilize **dialogue turns** rather than **actual time** as a heuristic indicator for information "expiration."
- Reasoning (CoT) fails to improve temporal alignment because models do not spontaneously reference temporal information during the reasoning process.
- Post-training alignment demonstrates **strong potential** and represents a viable path toward resolving temporal blindness.

## Highlights & Insights
- **Proactive Problem Identification**: Temporal blindness is an essential Agent capability defect that was previously overlooked.
- **Bi-directional Analysis of Over/Under-reliance**: The study analyzes not only when tools should have been called but were not, but also unnecessary repeated calls.
- **In-depth Analysis of Reasoning Trajectories**: Root causes were revealed by quantifying temporal keywords within reasoning chains.
- **Discovery of "Think-Act Inconsistency"**: Reveals a systematic fracture between LLM reasoning and action.
- **Design Quality of the TicToc Dataset**: Scenario diversity, timestamp injection methods, and human annotation quality control are all meticulously executed.

## Limitations & Future Work
- Primarily evaluates English scenarios; temporal perception differences across languages remain unknown.
- Detailed methods for post-training alignment and large-scale validation still require further disclosure and expansion.
- Timestamps are provided in a fixed format (ISO 8601); the impact of different temporal representations has not been explored.
- While diverse, scenario design remains limited (76 scenarios), and coverage of more domains needs expansion.
- Future research should investigate how to systematically inject temporal perception capabilities during the pre-training or alignment phases.

## Related Work & Insights
- **vs. Existing Tool-Use Evaluations (ToolBench, etc.)**: While prior work focuses on "whether the correct tool was called," this paper is the first to focus on "when a tool should be called."
- **vs. Temporal Reasoning Research**: Existing work studies the temporal reasoning capabilities of LLMs in isolated scenarios but does not link them to the tool-use decisions of Agents.
- **vs. LLM Alignment**: Contextualizes temporal-aware tool-use decision-making as a new dimension of the alignment problem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to identify and systematically study the temporal blindness of LLM Agents; the problem is significant and previously ignored.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models, 76 scenarios, 5,592 annotated samples, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, detailed dataset construction methodology, and deep analysis.
- Value: ⭐⭐⭐⭐⭐ Reveals a fundamental capability defect in LLM Agents with direct guiding significance for Agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](../../ICML2026/llm_agent/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ACL 2026\] ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"](toolgrad_efficient_tool-use_dataset_generation_with_textual_gradients.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)

</div>

<!-- RELATED:END -->
