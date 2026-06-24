---
title: >-
  [Paper Note] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues
description: >-
  [ACL 2025][LLM (Other)][function-calling] This work proposes DICE-Bench, a function-calling benchmark targeting multi-round, multi-party dialogue scenarios. It comprises 1,607 high-quality dialogue instances and the DICE-Score metric, which quantifies information dispersion, revealing the limitations of current LLMs in tool invocation during complex dialogues.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "function-calling"
  - "benchmark"
  - "multi-party dialogue"
  - "multi-round"
  - "tool-use evaluation"
date: 2026-05-08
content_hash: c681c85da7f76bcc
---

# DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues

**Conference**: ACL 2025  
**arXiv**: [2506.22853](https://arxiv.org/abs/2506.22853)  
**Code**: [snuhcc/DICE-Bench](https://github.com/snuhcc/DICE-Bench)  
**Area**: LLM / NLP  
**Keywords**: function-calling, benchmark, multi-party dialogue, multi-round, tool-use evaluation  

## TL;DR

This work proposes DICE-Bench, a function-calling benchmark targeting multi-round, multi-party dialogue scenarios. It comprises 1,607 high-quality dialogue instances and the DICE-Score metric, which quantifies information dispersion, revealing the limitations of current LLMs in tool invocation during complex dialogues.

## Background & Motivation

- **Limitations of Prior Work**: Existing function-calling benchmarks (such as APIBench, ToolLLM, etc.) mostly focus on single-turn interaction scenarios where all API parameters reside in a single user instruction, neglecting the complexity of real-world group chats where information is scattered across multi-turn, multi-party dialogues.
- **Real-world Needs**: In practical applications, virtual assistants need to track multi-turn, multi-party conversations in group chat scenarios, aggregating information from dispersed contexts to complete API calls (e.g., booking hotels and flights based on group chat discussions).
- **Evaluation Gap**: There is a lack of a metric to quantitatively measure the degree of dispersion of tool-related information in a dialogue, making it difficult to systematically evaluate the function-calling capabilities of LLMs under realistic conditions.
- **Ours**: This work constructs the DICE-Bench benchmark and propose the DICE-Score metric, generating multi-turn, multi-party dialogue data via multi-agent simulation to systematically evaluate the tool-calling capabilities of 19 LLMs.

## Method

### Overall Architecture

The data construction of DICE-Bench consists of three stages: (1) **Tool Graph Construction**: Collecting 124 tool nodes and 270 directed edges from TaskBench and ToolEyes to model inter-tool dependencies; (2) **Scenario Configuration**: Sampling tool chains via DFS and configuring dialogue types (persuasive negotiation, advisory information seeking, or eristic debate), the number of participants (2–4), and independent personas; (3) **Dialogue Generation**: Simulating dialogues using a multi-agent system, with an orchestrator controlling the turn-taking order to iteratively generate $N$-turn dialogues.

### Key Designs

- **DICE-Score Metric**: Quantitatively measures the dispersion of tool-related information in a dialogue. The formula is $\text{DICE}(S,T) = \frac{\min(|S_{\neq 0}|, T) \cdot \sqrt{|S| \cdot T}}{\sum_{i \in S} \ln(1 + \alpha \times S_i)}$, where $S$ is the count vector of tool-related information mentioned in each turn, $T$ is the total number of distinct tool items to be identified, and $\alpha = e^2$ controls the repetition penalty. A higher score indicates more dispersed information and a more difficult task.
- **Three-Stage Validation Pipeline**: Stage 1 uses G-Eval (GPT-4o) to automatically filter low-quality dialogues based on six-dimensional criteria; Stage 2 applies rule-based filtering (such as refusal response detection); Stage 3 involves human annotators scoring across three dimensions—dialogue quality, functional integration, and real-world applicability—comprising 15 sub-criteria to eliminate low-scoring instances.
- **Tool Graph Dependency Modeling**: Directed edges between tools explicitly encode the cross-turn dependency of "using the previous turn's tool output as the next turn's parameter," ensuring the authenticity of multi-turn scenarios.

### Loss & Training

- **Exact Match (EM)** is used as the primary evaluation metric, which requires the LLM to accurately predict both the function name and all parameter values simultaneously.

## Experiments

### Main Results

| Model | Round 1 | Round 2 | Round 3 | Round 4 | Average |
|------|---------|---------|---------|---------|------|
| GPT-4o | 74.12 | 61.00 | 61.65 | 59.18 | 63.99 |
| Gemini 2 Flash | 74.47 | 59.45 | 59.40 | 58.73 | 63.01 |
| Phi4-15B | 71.29 | 57.06 | 58.02 | 56.44 | 60.70 |
| GLM4-9B-Chat | 58.24 | 47.55 | 47.24 | 46.03 | 49.76 |
| Qwen2.5-32B | 67.76 | 56.76 | 57.23 | 55.92 | 59.42 |
| ToolAce-8B | 2.47 | 0.66 | 0.33 | 0.51 | 0.99 |

### Ablation Study

| Analysis Dimension | Key Findings |
|----------|----------|
| DICE-Score vs. Performance | Pearson correlation coefficient $r \approx -0.984$; higher DICE-Score correlates with worse performance |
| Alignment with Human Evaluation | Human accuracy decreases from 80.5% in Round 1 to 49.3% in Round 4, showing a strong negative correlation with DICE-Score (increasing from 1.42 to 5.36) |
| Impact of Dialogue Types | Eristic dialogues yield significantly lower EM due to frequent stance switching |
| Tool-Specific Models | Specialized models such as ToolAce-8B and CALM-8B underperform general-purpose dialogue models by a wide margin |

### Key Findings

1. The performance of all models drops significantly as the number of turns increases. Average performance in Round 4 decreases by approximately 15 percentage points compared to Round 1, indicating that multi-turn information aggregation is a primary bottleneck for current LLMs.
2. The open-source 15B-parameter Phi4 performs comparably to the closed-source GPT-4o (average score of 60.7 vs 64.0); the 128K context window of Qwen 2.5 benefits long-dialogue scenarios.
3. Models specifically fine-tuned for single-turn function calling perform poorly in multi-party dialogue scenarios (ToolAce-8B scoring only around 1%), suggesting that single-turn training data does not transfer well to multi-turn, multi-party contexts.

## Highlights & Insights

- The first function-calling benchmark covering both multi-turn and multi-party dialogues, filling a critical gap in existing evaluations.
- The proposed DICE-Score metric is highly negatively correlated with human performance ($r \approx -0.984$), demonstrating solid explainability and effectiveness.
- Rigorous three-stage filtering (automated + rule-based + human), selecting 1,607 high-quality instances from 1,800 candidates.

## Limitations & Future Work

- The dialogue length in Round 4 may exceed the 4K-token limit of some models, limiting the ability to evaluate all systems.
- Some models, despite producing semantically correct content, are penalized due to output formats failing to strictly conform to JSON specifications.
- The multi-agent orchestrator (GPT-4o) exhibits limited capabilities in dynamically assigning turn-taking orders, often defaulting to repetitive, structured turns.
- The benchmark only covers daily-life scenarios, lacking professional domain-specific tools in fields like law, finance, and medicine.

## Related Work & Insights

- **Function-Calling Benchmarks**: APIBench, ToolAlpaca, ToolLLM, API-Bank, MetaTool, TaskBench, etc., all focus on single-turn or single-user scenarios.
- **Interactive Dialogue Systems**: Research on multi-turn, multi-party interactions for LLM-integrated virtual assistants is still in its infancy.
- **Dialogue Type Theory**: Reconstruct scenario diversity based on Walton & Krabbe's seven-type dialogue classification framework.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Practicality | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Overall Rating | 4.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LR²Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems](lr2bench_evaluating_long-chain_reflective_reasoning_capabilities_of_large_langua.md)
- [\[ACL 2025\] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [\[ACL 2025\] Controlling Politeness in Multi-Turn Dialogues Through Pre-Phrase Augmentation](controlling_politeness_in_multi-turn_dialogues_through_pre-phrase_augmentation.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
