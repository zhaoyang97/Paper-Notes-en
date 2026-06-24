---
title: >-
  [Paper Note] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs
description: >-
  [ACL 2025][LLM Agent][task-oriented dialog] Proposes a comprehensive evaluation framework to quantify the "behavior gap" between LLM agents and human experts in task-oriented dialogues. It systematically diagnoses behavioral discrepancies across three dimensions: dialog acts, tool usage, and knowledge utilization. It reveals that the behavior gap is highly correlated with task complexity ($r=0.963$), and closing this gap via behavior injection improves performance by an avera…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "task-oriented dialog"
  - "behavior gap"
  - "zero-shot agent"
  - "dialog acts"
  - "tool usage evaluation"
date: 2026-05-08
content_hash: 9f8f6c46b2caa0a7
---

# The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs

**Conference**: ACL 2025  
**arXiv**: [2506.12266](https://arxiv.org/abs/2506.12266)  
**Code**: [GitHub](https://github.com/intuit-ai-research/behavior-gap)  
**Area**: Agent / Dialogue Systems  
**Keywords**: task-oriented dialog, behavior gap, zero-shot agent, dialog acts, tool usage evaluation

## TL;DR

Proposes a comprehensive evaluation framework to quantify the "behavior gap" between LLM agents and human experts in task-oriented dialogues. It systematically diagnoses behavioral discrepancies across three dimensions: dialog acts, tool usage, and knowledge utilization. It reveals that the behavior gap is highly correlated with task complexity ($r=0.963$), and closing this gap via behavior injection improves performance by an average of 24.3%.

## Background & Motivation

**Background**: LLM agents (e.g., AutoTOD, ProTOD, DARD, etc.) have been widely adopted in task-oriented dialogue systems (TODS) to replace traditional modular pipelines in a zero-shot manner. However, a significant gap in performance persists between these agents and human experts in real-world deployments.

**Limitations of Prior Work**: Prior studies (Elizabeth et al., 2024; Heck et al., 2023) have documented the performance degradation of LLM agents in zero-shot TODS, but **almost no work systematically analyzes the underlying behavioral causes**—specifically, how LLMs differ from humans in "how" they perform tasks. The only closely related work, Shaikh et al. (2024), solely examines grounding differences across 3 dialog acts, which is far from comprehensive.

**Key Challenge**: While the poor performance of LLM agents is well-documented, the **root causes remain unclear**—whether it is selecting incorrect dialog act strategies, overusing or inaccurately employing tools, or utilizing retrieved knowledge in a fundamentally different manner. The lack of a behavior-level diagnostic framework leaves the direction for improvement ambiguous.

**Ours Goal**: A three-dimensional (dialog acts + tool usage + knowledge usage) behavior evaluation framework is proposed. Under a teacher-forcing setup, this framework compares LLM agent behaviors against human experts turn-by-turn to quantify the "behavior gap" and validate its causal relationship with performance degradation.

**Core Idea**: The performance issues of LLM agents are inherently rooted in behavioral patterns. Systematic quantification reveals that the behavior gap is highly correlated with performance degradation, and narrowing this gap directly improves performance.

## Method

### Overall Architecture

Across three TOD datasets of increasing complexity (MultiWOZ $\rightarrow$ SpokenWOZ $\rightarrow$ PCS), zero-shot ReAct agents are constructed using GPT-4o, GPT-3.5 Turbo, and LLaMA-3.3-70B. Responses are generated turn-by-turn under a teacher-forcing setup. The agents' behaviors are then compared with those of human experts across three behavioral dimensions. Finally, a GPT-4o evaluator is utilized to assess response quality and analyze the correlation between the behavior gap and performance.

### Key Designs

**1. Three-Dimensional Behavior Gap Quantification Module**

Provides a systematic measurement of behavioral differences between LLM agents and human experts:

- **Dialog Acts Evaluation**: Two sets of GPT-4o few-shot classifiers are designed—the WOZ framework (10 dialog act classes, used for MultiWOZ/SpokenWOZ) and the ISO framework (11 classes, used for open-ended PCS tasks). Dialog act types are annotated for each response turn. Alignment between the agent and humans is measured using micro-F1, where misalignment = $1 - \text{micro-F1}$.
- **Tool Usage Evaluation**: GPT-4o few-shot tool classifiers are trained for each of the three datasets to annotate the tools used by human experts in each turn (the agent's tool invocation logs are already known). The micro-F1 alignment score for tool selection is calculated, and the average number of tool calls per turn is recorded.
- **Knowledge Usage Evaluation**: For turns involving knowledge retrieval tools, ROUGE-1 Precision is used to evaluate the degree of direct copying, and the compression ratio ($1 - \frac{\text{response length}}{\text{knowledge length}}$) is used to assess information summarization efficiency.

**2. Teacher-Forcing Evaluation and Performance Evaluation Module**

During evaluation, ground-truth human dialogue history is provided as context input to the LLM agent to prevent noise and error accumulation from user simulators:

- Given history $\{a_0, u_0, \ldots, u_{t-1}\}$, the agent generates $g_t$, which is compared against the human response $a_t$.
- A GPT-4o evaluator scores the responses across 4 dimensions (Coherence, Specificity, Effectiveness, and Satisfaction) on a 1-5 scale.
- The evaluator is validated on MultiWOZ: scores for successful dialogues (success rate = 1) are significantly higher across all metrics than for failed dialogues ($p < 0.05$).

**3. Task Complexity Measurement and Causal Validation Module**

Quantifies task complexity and validates the causal relationship between the behavior gap and performance:

- **Complexity Metric**: Normalized Turn Count = $\frac{\ln(1+t)}{\ln(1+t+C)}$ + Dialog Act Diversity = $d/d_{\max}$. The two terms are complementary (long dialogues vs. intent diversity), and their average yields the overall complexity.
- **Statistical Correlation**: Response quality is compared by grouping turns based on dialog act/tool usage alignment (F1 $\ge$ 0.5 vs. < 0.5), validating that turns with aligned behavior achieve significantly higher performance.
- **Behavior Injection Experiments**: Ground-truth human dialog acts or tool choices are injected into the system prompt to observe performance changes—validating whether narrowing the behavior gap causally improves performance.

## Key Experimental Results

### Statistics of Three Datasets

| Dataset | Dialogues | Avg. Turns / Dialogue | Avg. Words / Turn | Active Slots | No. of Tools |
|--------|-------|-------------|------------|-----------|-------|
| MultiWOZ | 1,000 | 14.7 | 13.4 | 24 | 8 |
| SpokenWOZ | 987 | 35.6 | 11.0 | 36 | 9 |
| PCS | 53 | **120.2** | 11.8 | $\infty$ | 4 |

### GPT-4o Agent Behavior Gap and Performance

| Dimension | MultiWOZ (Low Complexity) | SpokenWOZ (Medium Complexity) | PCS (High Complexity) |
|------|-------------------|---------------------|----------------|
| Dialog Act F1 | High | Medium | **0.464** |
| Tool Usage F1 | Medium | Low | **0.139** |
| Knowledge Copying (ROUGE-1 Prec) | Slightly higher for agent | Agent higher than human | Agent significantly higher than human |
| Knowledge Compression Ratio | Small difference | Medium difference | Agent significantly lower than human |

### Behavior Gap-Performance Correlation Analysis

| Analysis Dimension | Core Results | Explanation |
|---------|---------|------|
| Behavior Gap vs. Complexity | Correlation coefficient **0.963** | The higher the complexity, the more drastically the behavior gap widens |
| Injecting Human Dialog Acts | Avg. performance **+22.4%** (PCS) | Most significant improvement on complex tasks |
| Injecting Human Tool Selection | Avg. performance **+26.3%** (PCS) | Tool correction yields larger improvements |
| Combined Behavior Injection | Avg. **+24.3%** | Causal validation: Narrowing the behavior gap directly boosts performance |
| Model Comparison | GPT-4o < GPT-3.5 < LLaMA | Larger models exhibit smaller gaps, but GPT-4o still shows a massive gap on PCS |

### Key Findings

- **Behavior gap is the root cause of performance degradation**: It is not that the models are "not smart enough", but rather that their "behavioral patterns are incorrect"—both dialog act selection and tool-calling strategies deviate significantly from human patterns.
- **Tool usage is the most prominent pain point**: On PCS, the tool usage F1 is only 0.139, with the agent excessively calling tools and frequently invoking incorrect ones.
- **Fundamentally different knowledge utilization**: Humans synthesize information to deliver refined responses, whereas agents tend to directly copy content from the knowledge base (evidenced by significantly higher ROUGE-1 Precision and remarkably lower compression ratios).
- **Wider gaps in more complex tasks**: While the gap remains manageable under simple slot-filling scenarios, even GPT-4o struggles in real-world customer service scenarios (~120 turns per dialogue).

## Rating

| Dimension | Score (/10) | Explanation |
|------|-----------|------|
| Novelty | 7 | The "behavior gap" perspective is novel, though the evaluation framework leans more towards engineering. The core idea (comparing agent and human behavioral patterns) is highly intuitive. |
| Practicality | 8 | The three-dimensional behavioral diagnosis provides highly actionable improvement directions (rectifying dialog acts / tool selection), and the 24.3% improvement validates its practical value. |
| Technical Depth | 6 | The methodology primarily relies on GPT-4o classifiers and statistical analysis, with no model training or architectural innovation. The teacher-forcing setup is reasonable but not unprecedented. |
| Writing Quality | 8 | The experimental design is clear, with rich illustrations (7 main figures and multiple appendix tables). The conclusions are well-supported by data, and the limitations are discussed honestly. |

## Highlights & Insights

- **High value of the "diagnosis before treatment" paradigm**: Measuring the "behavior gap" offers superior diagnostic value compared to just measuring the "performance gap"—it not only reveals "how big" the gap is, but also pinpoints "where" the discrepancy lies (dialog acts / tools / knowledge), providing concrete paths for future improvement.
- **Real-world customer service data (PCS)** reveals challenges that traditional academic benchmarks (such as MultiWOZ) fail to capture: dialogues averaging 120 turns, infinite slot spaces, and multi-step reasoning, with a complexity far exceeding existing standards.
- **Compelling causal validation from behavior injection experiments**: Merely directing the agent on "what dialog acts/tools to use" yields a 24.3% performance boost, demonstrating that the deficit lies in strategy rather than inherent capability.

## Limitations & Future Work

- **Proprietary nature of the PCS dataset**: The core contributions rely heavily on the most complex task dataset, which is inaccessible to other researchers for result replication.
- **Classifier error propagation**: The GPT-4o dialog act classifier achieves an F1 of approximately 0.77, which may systematically under- or overestimate certain behavioral gaps.
- **Limited to zero-shot settings**: Behavioral gaps in few-shot or fine-tuned agents may differ qualitatively; thus, the generalizability of the findings remains to be verified.
- **Limitations of turn-level evaluation**: Turn-by-turn comparison under the teacher-forcing setup cannot fully capture dialogue-level strategic variations (e.g., humans intentionally delaying replies to aggregate more information).
- **Directions for improvement**: (1) Utilizing behavioral gap signals in RLHF/DPO training to explicitly align behaviors; (2) Open-sourcing complex TOD benchmarks.

## Related Work & Insights
- **vs. AutoTOD (Xu et al.)**: While AutoTOD focuses on modularity-free architectural design, this work focuses on behavioral diagnostics.
- **vs. Shaikh et al. (grounding analysis)**: They evaluate only a single dimension (grounding), whereas this work provides a comprehensive three-dimensional analysis.
- **vs. FED (Mehri et al.)**: FED targets generation quality, whereas this work evaluates behavioral alignment, representing a distinct perspective.

## Rating
- Novelty: ⭐⭐⭐⭐ The behavior gap evaluation framework is a novel contribution, featuring a systematic three-dimensional analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprises three datasets + three models + causal validation + complexity analysis.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the experimental analysis is well-structured.
- Value: ⭐⭐⭐⭐⭐ Provides actionable diagnostic and analytical tools for improving LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[ACL 2025\] GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](guidebench_guideline_following.md)
- [\[ECCV 2024\] Agent3D-Zero: An Agent for Zero-shot 3D Understanding](../../ECCV2024/llm_agent/agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)
- [\[ACL 2025\] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)

</div>

<!-- RELATED:END -->
