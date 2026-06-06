---
title: >-
  [Paper Note] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Agent evaluation] Proposes AJ-Bench, the first benchmark to systematically evaluate Agent-as-a-Judge capabilities, covering 155 tasks and 516 annotated trajectories across search, data systems…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Agent evaluation"
  - "Agent-as-a-Judge"
  - "Environment interaction verification"
  - "Trajectory evaluation"
  - "Benchmarking"
date: 2026-05-08
content_hash: 45b8ab3435e7b935
---

# AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation

**Conference**: ACL 2026  
**arXiv**: [2604.18240](https://arxiv.org/abs/2604.18240)  
**Code**: [https://aj-bench.github.io/](https://aj-bench.github.io/)  
**Area**: Reinforcement Learning  
**Keywords**: Agent evaluation, Agent-as-a-Judge, Environment interaction verification, Trajectory evaluation, Benchmarking

## TL;DR
Proposes AJ-Bench, the first benchmark to systematically evaluate Agent-as-a-Judge capabilities, covering 155 tasks and 516 annotated trajectories across search, data systems, and GUI domains. Experiments show that Agent-as-a-Judge improves F1 by approximately 13 percentage points on average compared to LLM-as-a-Judge.

## Background & Motivation

**Background**: As RL continues to scale LLM Agent training, reliable verification of Agent behavior in complex environments becomes increasingly critical. Current mainstream verification methods include rule-based evaluators and LLM-as-a-Judge; the former relies on predefined rules, while the latter makes judgments based on surface-level text signals.

**Limitations of Prior Work**: Rule-based evaluators struggle to generalize to complex, open-ended scenarios (e.g., scientific hypothesis verification, long-form fact-checking). LLM-as-a-Judge lacks access to environment state information and can only make surface-level judgments based on trajectory text, leading to erroneous evaluations. For example, judging whether an Agent correctly queried a database requires actually checking the database state rather than just reading the trajectory text.

**Key Challenge**: Verifying Agent behavior requires understanding changes in the environment state, yet existing evaluation frameworks restrict the Judge to an "observer" role, unable to interact with the environment to obtain verification evidence.

**Goal**: Construct the first benchmark to systematically evaluate Agent-as-a-Judge capabilities, quantifying the abilities of Agent evaluators in information acquisition, state verification, and process verification.

**Key Insight**: Empower the evaluator with "agentic" capabilities—allowing the Judge to interact with the environment and use tools to obtain evidence beyond the trajectory text to make more reliable judgments.

**Core Idea**: Build verification tasks that require environment interaction to systematically compare the differences between Agent-as-a-Judge and LLM-as-a-Judge, revealing the critical role of environment interaction for evaluation reliability.

## Method

### Overall Architecture
The construction process for AJ-Bench follows three steps: (1) Designing 155 tasks across three domains: search, data systems (file system + Postgres), and GUI (PPT + Word + Excel); (2) Generating 516 trajectories using multiple models followed by manual annotation; (3) Building interactive environment replicas for the DS and GUI domains. During evaluation, the Judge Agent receives the task description and candidate trajectory, interacts with the environment via 60 tools to gather evidence, and finally outputs a binary success/failure judgment.

### Key Designs

1.  **Three-Dimensional Evaluation Capability Design**:

    - **Function**: Comprehensively covers the core verification capabilities required for Agent-as-a-Judge.
    - **Mechanism**: (a) Information Acquisition—verifying factual claims within trajectories via external search (Search domain); (b) State Verification—checking whether the current environment state meets expectations via tools (DS domain, e.g., checking file existence or database records); (c) Process Verification—checking the correctness of key actions and execution steps (GUI domain, e.g., checking if PPT slides were correctly modified).
    - **Design Motivation**: These three dimensions represent the core advantages of Agent-as-a-Judge compared to LLM-as-a-Judge.

2.  **Multi-source Trajectory Collection and Annotation**:

    - **Function**: Provides high-quality, diverse positive and negative sample trajectories.
    - **Mechanism**: The search domain uses Gemini, Grok, and Perplexity to generate trajectories; the DS domain sources multi-model trajectories from MCPMark and standardizes the format; the GUI domain collects samples from OSWorld. Samples are deliberately selected such that "successful trajectories have many steps while failed ones have few," breaking the correlation between trajectory length and success rate. Labels are ensured through rule verification and manual review.
    - **Design Motivation**: Multi-model sources avoid single-model style bias, and length decoupling prevents the Judge from using trajectory length as a shortcut.

3.  **Environment Reconstruction and Interactive Evaluation**:

    - **Function**: Provides an interactive, real environment for the Judge Agent.
    - **Mechanism**: The DS domain replays the final environment state locally, while the GUI domain is reconstructed on isolated AWS instances. The Judge Agent starts from the final environment state and actively gathers evidence through tool calls (file operations, database queries, GUI checks, etc.).
    - **Design Motivation**: Static trajectory text is insufficient for reliably judging task completion status; an actual interactive environment is the critical infrastructure that distinguishes Agent-as-a-Judge from LLM-as-a-Judge.

### Loss & Training
AJ-Bench is an evaluation benchmark rather than a training framework. It uses the F1 score as the primary evaluation metric. For the search domain, F1 is calculated after aggregating at the single-entry level, while for DS and GUI domains, F1 is calculated at the trajectory level. All results are the average of three runs.

## Key Experimental Results

### Main Results

| Model | Is Agentic | Search F1 | DS F1 | GUI F1 | Overall F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gemini-3-pro | ✗ | 77.0 | 74.5 | 74.2 | 75.1 |
| gpt-5 | ✗ | 73.4 | 60.9 | 52.8 | 61.0 |
| deepseek-v3.2 | ✗ | 63.3 | 63.3 | 66.1 | 64.5 |
| gpt-5-mini | ✓ | 70.8 | 67.4 | 76.8 | 72.4 |
| deepseek-v3.2 | ✓ | 77.3 | 72.7 | 80.5 | **77.3** |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Interaction turns=5 vs 20 | F1: ~65 vs ~77 | More interaction turns continuously improve performance. |
| Accessibility tree only | GUI F1 varies | Sufficient for PPT tasks, insufficient for Word. |
| Screenshot only | Word F1 best | Optimal modality varies by task. |
| Hybrid modality | Excel F1 best | Multimodal input is not always better. |

### Key Findings
- Agent-as-a-Judge improves the F1 score by approximately 13 percentage points on average compared to LLM-as-a-Judge using the same model, with the largest improvement in the GUI domain (up to 31 percentage points).
- Weak model + tool use > strong model without tools: gpt-5-mini (agentic) achieved an overall F1 of 72.4, surpassing gpt-5 (non-agentic) at 61.0.
- Increasing reasoning effort does not necessarily improve Agent-as-a-Judge performance: deepseek-v3.2's thinking mode performed 0.23 F1 lower than the non-thinking mode.
- Multimodal input is not always beneficial: hybrid inputs may introduce noise, and different sub-tasks have different optimal modalities.

## Highlights & Insights
- "Weak model + tools > strong model without tools" is an important finding, suggesting that the value of Agent-as-a-Judge lies not in the model's inherent capability, but in the information gain provided by environment interaction.
- The discovery that increased reasoning effort may decrease performance indicates that Agent-as-a-Judge requires better tool-use capabilities rather than deeper thinking.
- The design of task domains (Search/DS/GUI) corresponds to information acquisition, state verification, and process verification, providing a clear capability framework for future research.

## Limitations & Future Work
- Most tasks are adapted from existing benchmarks rather than built from scratch, resulting in limited coverage.
- The search domain depends on external network environments; network instability affects evaluation consistency.
- Current absolute performance still has significant room for improvement (best F1 ~0.77), indicating that Agent-as-a-Judge is far from saturated.
- Future work can scale to more domains (e.g., scientific verification, code review) and increase data scale for training.

## Related Work & Insights
- **vs RewardBench/RM-Bench**: These benchmarks evaluate LLM-as-a-Judge and do not involve environment interaction; AJ-Bench is the first to systematically evaluate environment-aware Agent-as-a-Judge.
- **vs DevAI (Agent-as-a-Judge)**: DevAI only covers the single domain of code verification; AJ-Bench covers three domains and supports multi-modality.
- **vs AgentRewardBench**: Evaluates a Judge's judgment of Agent trajectories but does not provide environment interaction capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ First benchmark to systematically evaluate Agent-as-a-Judge.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-model comparisons and ablations, though the number of agentic models is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-articulated motivation for benchmark design.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in Agent evaluation infrastructure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](../../ICLR2026/llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](../../ICLR2026/llm_evaluation/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)
- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)
- [\[ACL 2026\] Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](finch_benchmarking_finance_amp_accounting_across_spreadsheet-centric_enterprise_.md)

</div>

<!-- RELATED:END -->
