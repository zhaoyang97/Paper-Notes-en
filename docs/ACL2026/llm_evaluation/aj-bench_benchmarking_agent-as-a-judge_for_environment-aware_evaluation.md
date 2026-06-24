---
title: >-
  [Paper Note] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation
description: >-
  [ACL 2026 Findings][LLM Evaluation][Agent Evaluation] This paper introduces AJ-Bench, the first benchmark to systematically evaluate the capabilities of Agent-as-a-Judge. It covers three domains—Search, Data Systems, and GUI—with a total of 155 tasks and 516 annotated trajectories. Experiments demonstrate that Agent-as-a-Judge improves the average $F1$ score by approximately 13 percentage points compared to LLM-as-a-Judge.
tags:
  - "ACL 2026 Findings"
  - "LLM Evaluation"
  - "Agent Evaluation"
  - "Agent-as-a-Judge"
  - "Environment Interaction Verification"
  - "Trajectory Evaluation"
  - "Benchmarking"
date: 2026-05-08
content_hash: 5581477a0857f89b
---

# AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.18240](https://arxiv.org/abs/2604.18240)  
**Code**: [https://aj-bench.github.io/](https://aj-bench.github.io/)  
**Area**: Reinforcement Learning  
**Keywords**: Agent Evaluation, Agent-as-a-Judge, Environment Interaction Verification, Trajectory Evaluation, Benchmarking

## TL;DR
This paper introduces AJ-Bench, the first benchmark to systematically evaluate the capabilities of Agent-as-a-Judge. It covers three domains—Search, Data Systems, and GUI—with a total of 155 tasks and 516 annotated trajectories. Experiments demonstrate that Agent-as-a-Judge improves the average $F1$ score by approximately 13 percentage points compared to LLM-as-a-Judge.

## Background & Motivation

**Background**: As Reinforcement Learning (RL) continues to scale up LLM Agent training, reliably verifying Agent behavior in complex environments has become increasingly critical. Current mainstream verification methods include rule-based validators and LLM-as-a-Judge; the former relies on predefined rules, while the latter makes judgments based on surface-level textual signals.

**Limitations of Prior Work**: Rule-based validators struggle to generalize to complex, open-ended scenarios (e.g., scientific hypothesis verification, long-form fact-checking). LLM-as-a-Judge lacks access to environmental state information and can only perform surface-level judgments based on trajectory text, which is prone to evaluation errors. For instance, determining whether an Agent correctly queried a database requires checking the actual database state rather than merely observing the trajectory text.

**Key Challenge**: Verifying Agent behavior requires understanding changes in environmental states, but existing evaluation frameworks restrict the Judge to a "bystander" role, unable to interact with the environment to obtain verification evidence.

**Goal**: To construct the first benchmark for systematically evaluating Agent-as-a-Judge capabilities, quantifying the performance of Agent evaluators in information acquisition, state verification, and process verification.

**Key Insight**: Empowering the evaluator with "agentic" capabilities—allowing the Judge to interact with the environment and use tools to gather evidence beyond the trajectory text to make more reliable judgments.

**Core Idea**: Construct verification tasks that require environmental interaction and systematically compare the differences between Agent-as-a-Judge and LLM-as-a-Judge to reveal the critical role of environmental interaction in evaluation reliability.

## Method

### Overall Architecture

AJ-Bench examines whether an evaluator can actively interact with the environment like an agent to verify if another agent has truly completed a task, rather than just reading trajectory text for surface-level judgment. The benchmark covers three domains—Search, Data Systems (File System + Postgres), and GUI (PPT/Word/Excel)—comprising 155 tasks and 516 manually annotated trajectories, with interactive environment replicas reconstructed for the DS and GUI domains. During evaluation, the Judge Agent receives the task description and a candidate trajectory, then calls 60 types of tools to gather evidence from the environment's final state, ultimately outputting a binary "Success/Failure" classification.

### Key Designs

**1. Three-Dimensional Evaluation Capability Design: Stress-testing Information Acquisition, State Verification, and Process Verification through Three Domains**

The blind spot of LLM-as-a-Judge is that it can only see the trajectory text; it fails on tasks requiring external facts or environmental states. AJ-Bench decomposes this blind spot into three quantifiable capabilities: the Search domain requires the evaluator to verify factual claims in the trajectory through external retrieval (Information Acquisition); the Data Systems domain requires the evaluator to use tools to check if the current environment state meets expectations, such as checking if files exist or database records were correctly written (State Verification); and the GUI domain requires the evaluator to verify key actions and execution steps, such as whether a PPT slide was actually modified correctly (Process Verification). These three categories cover the incremental advantages of Agent-as-a-Judge over LLM-as-a-Judge.

**2. Multi-source Trajectory Collection and Annotation: Preventing Judge Shortcuts via Multi-model Sources and Length Decoupling**

Trajectories in the Search domain are generated by Gemini, Grok, and Peplexity. In the DS domain, multi-model trajectories are collected from MCPMark and unified in format. In the GUI domain, data is collected from OSWorld. Using multiple sources avoids the Judge using the style of a single model as a discriminative feature. More importantly, samples are deliberately selected where "successful trajectories have many steps and failed trajectories have few steps" to break the natural correlation between trajectory length and success rate, preventing the Judge from using length as a shortcut. All labels are cross-verified by rule-based validation and manual review to ensure the quality of positive and negative samples.

**3. Environment Reconstruction and Interactive Evaluation: Providing a Real Environment for Active Evidence Collection**

In the DS domain, the final environment state is obtained by replaying trajectories locally. The GUI domain is reconstructed on isolated AWS instances. The Judge Agent starts from this final state and actively gathers evidence through tool calls such as file operations, database queries, and GUI inspections, instead of guessing from static text. This interactive environment is the core infrastructure of the benchmark—it is the "ability to verify by doing" that distinguishes Agent-as-a-Judge from a text-only LLM-as-a-Judge.

### Loss & Training

AJ-Bench is an evaluation benchmark rather than a training framework; the primary metric is $F1$. The Search domain calculates $F1$ after aggregating at the single-item level, while the DS and GUI domains calculate $F1$ at the trajectory level. All results are reported as the average of three runs.

## Key Experimental Results

### Main Results

| Model | Agentic | Search $F1$ | DS $F1$ | GUI $F1$ | Overall $F1$ |
|------|-----------|----------|---------|---------|--------|
| gemini-3-pro | ✗ | 77.0 | 74.5 | 74.2 | 75.1 |
| gpt-5 | ✗ | 73.4 | 60.9 | 52.8 | 61.0 |
| deepseek-v3.2 | ✗ | 63.3 | 63.3 | 66.1 | 64.5 |
| gpt-5-mini | ✓ | 70.8 | 67.4 | 76.8 | 72.4 |
| deepseek-v3.2 | ✓ | 77.3 | 72.7 | 80.5 | **77.3** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Interaction Turns = 5 vs 20 | $F1$: ~65 vs ~77 | More interaction turns continuously improve performance |
| Accessibility Tree Only | GUI $F1$ varies | Sufficient for PPT tasks, insufficient for Word |
| Screenshot Only | Word $F1$ Best | Optimal modality differs across different tasks |
| Mixed Modality | Excel $F1$ Best | Multi-modality is not always better |

### Key Findings
- Agent-as-a-Judge achieves an average $F1$ improvement of approximately 13 percentage points over LLM-as-a-Judge using the same model, with the largest gain in the GUI domain (up to 31 percentage points).
- Weak Model + Tool Usage > Strong Model without Tools: gpt-5-mini (agentic) achieved an overall $F1$ of 72.4, surpassing gpt-5 (non-agentic) at 61.0.
- Increasing reasoning effort does not necessarily improve Agent-as-a-Judge performance: the "thinking" mode of deepseek-v3.2 was 0.23 $F1$ lower than the mode without "thinking."
- Multimodal input is not always beneficial: mixed inputs may introduce noise, and the optimal modality varies for different subtasks.

## Highlights & Insights
- The finding that "Weak Model + Tools > Strong Model without Tools" is significant, indicating that the value of Agent-as-a-Judge lies not in the model's inherent capability but in the information gain provided by environmental interaction.
- The observation that increasing reasoning effort may decrease performance suggests that Agent-as-a-Judge requires better tool-use capabilities rather than deeper thinking.
- The design of the task domains (Search/DS/GUI) corresponds to three core capabilities—information acquisition, state verification, and process verification—providing a clear capability classification framework for future research.

## Limitations & Future Work
- Most tasks were adapted from existing benchmarks rather than built from scratch, leading to limited coverage.
- The Search domain relies on external network environments; network instability affects evaluation consistency.
- Current absolute performance still has significant room for improvement (best $F1$ around 0.77), indicating that Agent-as-a-Judge is far from saturated.
- Future work could expand to more domains (e.g., scientific verification, code review) and increase data scale for training.

## Related Work & Insights
- **vs RewardBench/RM-Bench**: These benchmarks evaluate LLM-as-a-Judge without environmental interaction; AJ-Bench is the first to systematically evaluate environment-aware Agent-as-a-Judge.
- **vs DevAI (Agent-as-a-Judge)**: DevAI only covers the single domain of code verification; AJ-Bench covers three domains and supports multi-modality.
- **vs AgentRewardBench**: This evaluates the Judge’s judgment of Agent trajectories but does not provide environmental interaction capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ First benchmark to systematically evaluate Agent-as-a-Judge.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient multi-model comparisons and ablations, though the number of agentic models is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-articulated motivation for the benchmark design.
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in the infrastructure for Agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](../../ICLR2026/llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ACL 2026\] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)
- [\[ICLR 2026\] Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation](../../ICLR2026/llm_evaluation/holistic_agent_leaderboard_the_missing_infrastructure_for_ai_agent_evaluation.md)
- [\[ICLR 2026\] Towards Self-Evolving Agent Benchmarks: Validatable Agent Trajectory via Test-Time Exploration](../../ICLR2026/llm_evaluation/towards_self-evolving_agent_benchmarks_validatable_agent_trajectory_via_test-tim.md)
- [\[ACL 2026\] LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control](locar_localization-aware_evaluation_of_in-vehicle_assistants_through_fine-graine.md)

</div>

<!-- RELATED:END -->
