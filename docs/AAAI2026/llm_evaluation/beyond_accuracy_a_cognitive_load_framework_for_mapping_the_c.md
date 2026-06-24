---
title: >-
  [Paper Note] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents
description: >-
  [AAAI 2026][LLM Evaluation][Cognitive Load Theory] Drawing on Cognitive Load Theory (CLT) from psychology, this work decomposes the complexity of tool-use tasks into intrinsic load (structural complexity of the solution path) and extraneous load (ambiguity of problem formulation). It constructs ToolLoad-Bench, a benchmark with parametrically adjustable cognitive load, and employs an exponential decay model $\text{Acc} \approx e^{-(k \cdot CL + b)}$ to precisely characterize t…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Cognitive Load Theory"
  - "Tool-use Agent"
  - "Capability Boundaries"
  - "Tool Interaction Graph"
  - "Benchmark"
date: 2026-05-08
content_hash: 98940a8342e2f9cd
---

# Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents

**Conference**: AAAI 2026
**arXiv**: [2601.20412](https://arxiv.org/abs/2601.20412)  
**Code**: None  
**Area**: LLM Agent / Tool-use Evaluation
**Keywords**: Cognitive Load Theory, Tool-use Agent, Capability Boundaries, Tool Interaction Graph, Benchmark

## TL;DR
Drawing on Cognitive Load Theory (CLT) from psychology, this work decomposes the complexity of tool-use tasks into intrinsic load (structural complexity of the solution path) and extraneous load (ambiguity of problem formulation). It constructs ToolLoad-Bench, a benchmark with parametrically adjustable cognitive load, and employs an exponential decay model $\text{Acc} \approx e^{-(k \cdot CL + b)}$ to precisely characterize the capability boundaries of different agents.

## Background & Motivation
**Background**: Tool use is a core capability of LLM agents, and benchmarks such as the Berkeley Function Calling Leaderboard are widely adopted. Existing benchmarks provide final accuracy rankings but do not analyze when or why tasks fail.

**Limitations of Prior Work**: Current evaluations are "black boxes"—they reveal what a model can do, but not where its cognitive bottlenecks lie. The lack of fine-grained decomposition of task difficulty makes it impossible to diagnose specific failure modes (e.g., whether failure stems from structural complexity or problem ambiguity).

**Key Challenge**: The ultimate goal of evaluation should go beyond ranking; it should characterize each model's "capability boundary"—the complexity level at which performance begins to degrade significantly. However, "complexity" is inherently multi-dimensional and cannot be captured by a single metric.

**Goal**: To provide a principled framework that quantifies the complexity of tool-use tasks into measurable components, and uses a statistically validated model to characterize the capability limits of each agent.

**Key Insight**: The paper maps CLT—originally grounded in the limited capacity of human working memory—onto LLM reasoning: the model's computational context and reasoning capacity are analogized to human working memory, with cognitive load quantifying the demand a task places on "reasoning capacity."

**Core Idea**: Leverage the intrinsic/extraneous load decomposition from CLT to quantify the complexity of tool-use tasks, and use an exponential decay model to precisely map the capability boundary of each agent.

## Method

### Overall Architecture
**Input**: Multi-turn tool-use tasks $(Q, T)$ (user query sequences + available tool sets). **Approach**: (1) Formalize the solution path using a Tool Interaction Graph (TIG); (2) Extract intrinsic cognitive load $CL_I$ (structural complexity) from the TIG; (3) Extract extraneous cognitive load $CL_E$ (ambiguity) from task formulation; (4) Compute total load $CL_{\text{Total}} = CL_I + \omega_E \cdot CL_E$; (5) Fit $\text{Acc} \approx e^{-(k \cdot CL_{\text{Total}} + b)}$ to obtain a cognitive profile $(k, b)$ for each model.

### Key Designs

1. **Tool Interaction Graph (TIG)**:

    - **Function**: Models the ground-truth solution process as a directed acyclic graph (DAG), where nodes represent user queries and function calls, and edges represent data or execution dependencies.
    - **Mechanism**: The cognitive load of each dependency edge $e = (v_i, v_j)$ is determined by two factors: (1) **Memory load (attention distance)** $\delta(v_i, v_j)$—the number of dialogue turns between two operations, with greater distance implying higher memory demand; (2) **Selection load (interference)** $I(v_i, v_j)$—the number of same-type but incorrect entities in context (e.g., selecting the correct user ID among multiple candidates). Edge weight: $w(e) = \delta(v_i, v_j) \cdot (1 + \lambda \cdot I(v_i, v_j))$.
    - **Design Motivation**: To operationalize abstract "task complexity" as graph-structural metrics, making complexity both computable and adjustable (adding edges increases intrinsic load).

2. **Cognitive Load Decomposition**:

    - **Function**: Decomposes total load into intrinsic load $CL_I$ (inherent to the solution path) and extraneous load $CL_E$ (induced by task presentation).
    - **Mechanism**: $CL_I = \sum_{v_f} \sum_{e \to v_f} w(e)$, the sum of all dependency edge weights over all function call nodes. $CL_E = \sum_{q_i} CL_E(q_i, T)$, assessed by Gemini-2.5-pro per query on ambiguity and distractor-tool distraction (each normalized to $[0,1]$). Total load is a weighted sum $CL_{\text{Total}} = CL_I + \omega_E \cdot CL_E$, where $\omega_E$ is empirically calibrated as the ratio of the two loads' actual impact on accuracy.
    - **Design Motivation**: Different models exhibit different sensitivities to the two load types (e.g., some excel at structurally complex but semantically clear tasks), enabling targeted diagnosis of specific weaknesses.

3. **Exponential Decay Prediction Model**:

    - **Function**: Models performance degradation as a function of cognitive load via $\text{Acc}(Q,T,G) \approx \exp(-(k \cdot CL_{\text{Total}} + b))$.
    - **Mechanism**: Derived from two postulates—(1) the success probability of an individual operation is an exponential function of its cognitive load; (2) the overall task success probability is the product of per-operation success probabilities (independence assumption). This yields the additivity of total cognitive load. Parameter $b$ reflects baseline capability (accuracy at zero load equals $e^{-b}$), while $k$ reflects load sensitivity (smaller $k$ implies greater robustness).
    - **Design Motivation**: Exponential decay is a classical model in information theory and psychology, and the fit quality can be rigorously validated via the Hosmer–Lemeshow test.

### Loss & Training
This paper presents an evaluation framework and requires no model training. ToolLoad-Bench is constructed by expanding 200 instances from BFCL v3 to 500 via graph generation and edge insertion strategies, covering 10 tool categories and 106 distinct tools.

## Key Experimental Results

### Main Results
Overall accuracy on ToolLoad-Bench:

| Model | Accuracy (%) |
|-------|-------------|
| xLAM2-32B (fine-tuned) | **78.8** |
| GPT-4o | 68.0 |
| Claude 3.7 Sonnet | 64.8 |
| GPT-4o-mini | 62.2 |
| Gemini 2.5 Pro | 60.0 |
| Qwen3-235B | 58.0 |
| Qwen3-32B | 55.2 |
| Qwen3-8B | 38.6 |
| Llama3.3-70B | 17.0 |

Cognitive profile parameters:

| Model | $k$ (load sensitivity) ↓ | $b$ (baseline load) ↓ |
|-------|--------------------------|------------------------|
| xLAM2-32B | **0.034** | 1.22 |
| GPT-4o | 0.067 | 1.71 |
| Claude 3.7 | 0.073 | 1.57 |
| Qwen3-32B | 0.075 | 1.60 |
| Qwen3-8B | 0.085 | 1.12 |
| Gemini 2.5 Pro | 0.088 | 1.22 |

### Ablation Study
Hosmer–Lemeshow goodness-of-fit test (p-value > 0.05 indicates no statistically significant difference between model predictions and observed outcomes):

| Model | H-L Statistic | p-value | Note |
|-------|--------------|---------|------|
| xLAM2-32B | 3.59 | **0.89** | Best fit |
| GPT-4o | 4.87 | 0.77 | Good fit |
| Qwen3-235B | 5.19 | 0.74 | Good fit |
| Llama3.3-70B | 13.21 | 0.10 | Still passes |

### Key Findings
- xLAM2-32B achieves the highest accuracy (78.8%) and the lowest load sensitivity ($k=0.034$) at only 32B parameters, demonstrating that targeted fine-tuning is a more efficient strategy for improving tool-use capability than simply scaling model size.
- All models exhibit a "performance cliff": they perform well under low cognitive load but degrade sharply beyond a certain threshold. Llama3.3-70B achieves only 23% even under low load, exposing fundamental capability deficiencies.
- Extraneous load (problem ambiguity) and intrinsic load (structural complexity) exhibit highly similar impact patterns on performance, confirming that clarity of problem formulation is equally as important as structural complexity.
- The framework's predictive validity is statistically confirmed via the Hosmer–Lemeshow test: all models yield p-values well above 0.05, indicating that the exponential decay model is highly consistent with empirical observations.

## Highlights & Insights
- The interdisciplinary analogy of applying CLT from psychology to quantify the "cognitive limits" of AI agents is particularly elegant. Framing a model's context window and reasoning capacity as human working memory is both intuitive and theoretically grounded.
- The two-parameter cognitive profile $(k, b)$ conveys far more information than a single accuracy figure—the same accuracy can correspond to distinct $(k, b)$ configurations (e.g., high baseline + high sensitivity vs. low baseline + low sensitivity), which has direct practical value for model selection and task routing.
- The parametric design of ToolLoad-Bench (adjustable cognitive load) represents a methodological contribution to evaluation: unlike traditional benchmarks with fixed difficulty, it can continuously challenge increasingly capable models.

## Limitations & Future Work
- The measurement of extraneous load $CL_E$ relies on subjective assessments by Gemini-2.5-pro, lacking objectivity and reproducibility—a limitation the authors acknowledge.
- ToolLoad-Bench contains only 500 instances with limited domain coverage (expanded from 8 to 10 tool categories based on BFCL v3).
- The independence assumption (each function call's success probability is independent) does not fully hold in practice, as errors in earlier steps propagate to subsequent ones.
- No code is provided, raising concerns about reproducibility.

## Related Work & Insights
- **vs. BFCL**: BFCL provides accuracy rankings but does not analyze complexity. The cognitive load framework can serve as a diagnostic enhancement layer on top of BFCL, enabling deeper analysis of existing benchmark results.
- **vs. ToolLLM/API-Bank**: These are larger-scale tool-use benchmarks but similarly lack systematic analysis of task complexity. The cognitive load framework is transferable to these benchmarks.
- **Inspiration**: The cognitive profiling approach can be generalized to evaluation of other agent capabilities—such as conversational load in multi-turn dialogue, structural complexity in code generation, or perceptual load in multimodal reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first work to formally introduce CLT into AI agent evaluation; the TIG formalization and statistical validation of the exponential decay model are both compelling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive analysis across 9 models, intrinsic/extraneous load decomposition, goodness-of-fit validation, calibration plots, and capability box plots.
- **Writing Quality**: ⭐⭐⭐⭐⭐ A complete logical chain from theoretical postulates to empirical validation, with excellent visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Proposes a novel and verifiable evaluation methodology with direct implications for understanding and improving tool-use agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](../../ACL2026/llm_evaluation/exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ICLR 2026\] HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities](../../ICLR2026/llm_evaluation/hackworld_evaluating_computer-use_agents_on_exploiting_web_application_vulnerabi.md)
- [\[ICLR 2026\] Computer Agent Arena: Toward Human-Centric Evaluation and Analysis of Computer-Use Agents](../../ICLR2026/llm_evaluation/computer_agent_arena_toward_human-centric_evaluation_and_analysis_of_computer-us.md)
- [\[AAAI 2026\] Towards a Common Framework for Autoformalization](towards_a_common_framework_for_autoformalization.md)
- [\[ACL 2026\] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall](../../ACL2026/llm_evaluation/stratmem-bench_evaluating_strategic_memory_use_in_virtual_character_conversation.md)

</div>

<!-- RELATED:END -->
