---
title: >-
  [Paper Note] MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation
description: >-
  [ACL 2026][LLM Reasoning][Multi-turn reasoning] MTR-Bench constructs an automated multi-turn reasoning evaluation framework featuring 4 categories, 40 tasks, and 3,600 instances…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Multi-turn reasoning"
  - "automatic evaluation"
  - "interactive environment"
  - "difficulty layering"
  - "reasoning pattern analysis"
date: 2026-05-08
content_hash: 536f3bf2a15e64bc
---

# MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation

**Conference**: ACL 2026  
**arXiv**: [2505.17123](https://arxiv.org/abs/2505.17123)  
**Code**: https://github.com/LittleCirc1e/mtr_bench  
**Area**: LLM Reasoning / Multi-turn Interaction Evaluation  
**Keywords**: Multi-turn reasoning, automatic evaluation, interactive environment, difficulty layering, reasoning pattern analysis  

## TL;DR
MTR-Bench constructs an automated multi-turn reasoning evaluation framework featuring 4 categories, 40 tasks, and 3,600 instances, revealing that current frontier reasoning models remain far from reliable in interactive and dynamic feedback environments.

## Background & Motivation
**Background**: Reasoning-enhanced LLMs such as o1, DeepSeek-R1, and QwQ have shown outstanding performance in mathematics, coding, and logic problems. However, most mainstream evaluations are single-turn, where models read a problem and output an answer once. These evaluations fail to reflect interaction, feedback utilization, and long-term state maintenance required in real-world problem-solving.

**Limitations of Prior Work**: Existing multi-turn benchmarks like MT-Bench focus more on conversational coherence and contextual understanding rather than specialized reasoning. While GameArena addresses reasoning, it offers limited scenarios and relies on human interaction, making large-scale, controlled evaluation difficult. Human involvement also complicates difficulty control and automated experimental replication.

**Key Challenge**: A true reasoning system must actively probe the environment, parse feedback, revise plans, and gradually approach a goal across multiple turns. If the evaluation environment cannot be automated, it is difficult to scale or increase difficulty as models progress.

**Goal**: To construct a multi-turn reasoning benchmark capable of automatic problem generation, automatic environmental feedback simulation, and automatic scoring, covering capabilities such as induction, abduction, deduction, and planning while controlling complexity via difficulty parameters.

**Key Insight**: The authors decompose evaluation tasks into three components: Generator, Monitor, and Evaluator. The Generator produces problems of varying difficulty levels; the Monitor acts as a rule-based environment that processes model queries, returns feedback, and determines termination; the Evaluator calculates accuracy, efficiency, invalid operation rates, and reasoning patterns based on the full interaction history.

**Core Idea**: Isolate "pure reasoning capability" using closed, deterministic, rule-defined interactive environments to avoid interference from tool use, open-world noise, or manual labeling costs.

## Method
The methodology of MTR-Bench focuses on benchmark construction and evaluation protocol. Instead of providing a static prompt, the model acts repeatedly within an environment controlled by a rule-based monitor. Each turn, the model must output a valid query or answer; the monitor returns feedback based on task rules. Interaction ends when the model reaches the target state or exceeds the maximum number of turns. This allows for analysis of feedback utilization, planning, and invalid operations beyond just the final answer.

### Overall Architecture
The process begins with task seed collection. Tasks with high reasoning intensity are collected from public websites, categorized by GPT-4o, and manually verified into four categories: Information Probing, Dynamic Adaptation, State Operation, and Strategic Gaming. Ten tasks are selected per category for a total of 40 tasks. Each task includes easy, medium, and hard difficulty levels, with 30 problems generated per level, totaling 3,600 evaluation instances.

During evaluation, the Generator outputs a specific problem $p$ and a reasoning goal $s$. The model generates a query, and the Monitor checks for valid formatting, returns feedback according to rules, and judges if the goal is met. Finally, the Evaluator calculates metrics based on the full dialogue history. The maximum turn limit for all models is 15.

### Key Designs
1. **Four task categories covering diverse reasoning mechanisms**:
	- Function: Tests reasoning capabilities in interactive environments from multiple perspectives.
	- Mechanism: Information Probing tests gradual induction from fixed hidden information; Dynamic Adaptation tests abduction where answers change following failed attempts; State Operation tests deductive execution by inferring hidden mechanisms from feedback; Strategic Gaming tests multi-step planning within systems involving opponents or dynamic variables.
	- Design Motivation: Using a single game or problem type would lead to benchmark overfitting; these four categories isolate different reasoning deficiencies.

2. **Generator-Monitor-Evaluator automated closed-loop**:
	- Function: Enables multi-turn evaluation without requiring real-time human participation.
	- Mechanism: The Generator creates problems using templates and difficulty parameters; the Monitor serves as a deterministic environment for query format validation, feedback generation, and termination; the Evaluator computes Accuracy (Acc), Efficiency (Eff), Invalid Rate (IR), and Pattern Analysis (PA).
	- Design Motivation: Interaction and scoring are the highest costs in multi-turn evaluation; this three-component split allows for scalability.

3. **Process metrics rather than final answers only**:
	- Function: Analyzes why models fail and whether high accuracy is accompanied by high efficiency.
	- Mechanism: Accuracy measures task completion; Efficiency compares the number of turns taken for correctly answered problems; Invalid Rate measures format and operational validity; Pattern Analysis tracks reasoning modes across four categories: Associate, Verify, Plan, and Feedback.
	- Design Motivation: In multi-turn reasoning, a model might reach the correct answer inefficiently or fail due to invalid formatting; looking only at final accuracy loses critical diagnostic information.

### Loss & Training
As this work presents an evaluation benchmark, no models were trained. Difficulty calibration was performed through iterative testing: for example, if generating 10 problems with parameters $n=6,7,8$ failed to create a reasonable performance gradient, parameters were adjusted to $n=6,9,12$ before large-scale evaluation.

## Key Experimental Results

### Main Results
The experiments cover reasoning-enhanced models and non-reasoning instruction models. The table lists the average accuracy for each model across three difficulty levels, derived from the AVG column of the paper's main table.

| Model | Type | Easy AVG | Medium AVG | Hard AVG |
|------|------|----------|------------|----------|
| o3-mini | Reasoning | 56.07 | 41.80 | 31.19 |
| DeepSeek-R1 | Reasoning | 48.62 | 37.33 | 29.19 |
| QwQ-32B | Reasoning | 49.64 | 33.72 | 25.58 |
| Qwen3-235B-A22B-Thinking | Reasoning | 47.45 | 36.20 | 29.08 |
| GPT-4o | Non-reasoning | 28.50 | 16.94 | 12.06 |
| Qwen-Max | Non-reasoning | 32.66 | 19.13 | 12.18 |
| Qwen2.5-72B-IT | Non-reasoning | 29.43 | 19.06 | 12.94 |

### Ablation Study
| Analysis Item | Numbers / Phenomena | Description |
|--------|-------------|------|
| Data Scale | 4 categories, 40 tasks, 3,600 instances | 3 difficulties per task, 30 instances per level |
| Max Interaction Turns | 15 turns | Controls evaluation budget for all models |
| Seed Sources | 32 Codeforces tasks, 8 NYT logic puzzles | Appendix shows an average Codeforces rating of 2453.13 |
| Difficulty Trend | Accuracy drops for all models from Easy to Hard | Demonstrates effective difficulty layering |
| Efficiency Analysis | o3-mini has highest performance but lowest efficiency; R1 is more efficient | High accuracy does not equate to fewer interaction turns |
| Small Model Performance | Models < 7B have almost no meaningful scores | The benchmark is highly challenging for small models |

### Key Findings
- Reasoning models are significantly stronger than non-reasoning models; QwQ-32B even outperforms the stronger non-reasoning model in the same series, Qwen-Max.
- The advantages of the R1-Distill series in mathematics and coding do not translate well to these OOD multi-turn tasks, suggesting that SFT distillation is insufficient for generalizing interactive reasoning.
- o3-mini shows a prominent advantage in IP and SG, but is closer to QwQ-32B and R1 in DA and SO, indicating that parsing complex environmental feedback remains a bottleneck.
- Pattern Analysis reveals that QwQ-32B and R1 are notably stronger than R1-Distill-Qwen-32B in Associate, Verify, and Feedback patterns, suggesting that feedback utilization and self-checking are core multi-turn reasoning capabilities.

## Highlights & Insights
- The primary strength of this paper is transforming "multi-turn reasoning" into an automatically executable environment rather than a manual conversational evaluation. This makes the benchmark repeatable, scalable, and adjustable.
- The Monitor design offers high diagnostic value. Models fail not only due to reasoning errors but also due to invalid query formats, out-of-bounds operations, or failure to understand feedback.
- The paper notes that o3-mini's strength is not just faster reasoning, but superior long-term planning and utilization of historical feedback. This provides insights for training agents: multi-turn capability is not a simple extension of single-step CoT.
- Using closed, rule-based environments sacrifices natural language realism but allows for cleaner measurement of abstract reasoning, making it suitable for capability diagnostics.

## Limitations & Future Work
- Strategic Gaming currently uses random system strategies; the authors acknowledge the need for stronger adversarial strategies in the future.
- The current interaction format is structured rather than natural language chat, meaning it cannot evaluate a model's ability to reason or clarify within natural dialogue.
- While tasks are adapted from public sources, they remain puzzle/competition-style, which differs from open-ended real-world agent tasks.
- These interactive environments are naturally suited for reinforcement learning; MTR-Bench could be expanded from an evaluation tool to a platform for training and curriculum learning.

## Related Work & Insights
- **vs MT-Bench**: MT-Bench focuses on multi-turn dialogue quality and context understanding, whereas MTR-Bench specifically measures multi-turn reasoning and feedback utilization.
- **vs GameArena**: GameArena is closer to game evaluation but has fewer scenarios and relies on humans; MTR-Bench provides 40 tasks and supports automatic scoring.
- **vs AgentBench / AgentBoard**: These benchmarks include open environments like tools, web pages, and OS; MTR-Bench deliberately uses closed rule-based environments to isolate core logical reasoning.
- **Insight**: When training reasoning agents, feedback parsing, state tracking, valid action generation, and long-term planning should be optimized independently, rather than focusing solely on single-turn final answer accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Automated multi-turn reasoning environment design is comprehensive; task taxonomy is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers many models and metrics; process analysis is more valuable than reporting accuracy alone.
- Writing Quality: ⭐⭐⭐⭐☆ Clearly structured, though tables are large and some appendix information is crucial for understanding task origins.
- Value: ⭐⭐⭐⭐☆ Directly relevant for evaluating reasoning models and training interactive agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](../../ICML2026/llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[CVPR 2026\] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](../../CVPR2026/llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)
- [\[ACL 2026\] Scaling Evaluation-Time Compute with Reasoning Models as Evaluators](scaling_evaluation-time_compute_with_reasoning_models_as_evaluators.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)

</div>

<!-- RELATED:END -->
