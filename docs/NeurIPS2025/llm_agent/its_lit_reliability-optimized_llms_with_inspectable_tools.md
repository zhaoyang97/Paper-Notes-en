---
title: >-
  [Paper Note] It's LIT! Reliability-Optimized LLMs with Inspectable Tools
description: >-
  [NeurIPS 2025 (Workshop on Multi-Turn Interactions in LLMs)][LLM Agent][tool reliability] By defining reliability/inspectability cost functions for each external tool, LIT guides LLMs to select the lowest-cost (most transparent and auditable) tool-calling path among multiple candidates, improving interpretability while maintaining or enhancing task accuracy in 61 out of 65 test scenarios.
tags:
  - NeurIPS 2025 (Workshop on Multi-Turn Interactions in LLMs)
  - LLM Agent
  - tool reliability
  - inspectability
  - cost function
  - tool selection
  - trustworthy AI
  - multi-step tool calling
date: 2026-05-08
content_hash: fc29a7d411c4d308
---

# It's LIT! Reliability-Optimized LLMs with Inspectable Tools

**Conference**: NeurIPS 2025 (Workshop on Multi-Turn Interactions in LLMs)  
**arXiv**: [2511.14903](https://arxiv.org/abs/2511.14903)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: tool reliability, inspectability, cost function, tool selection, trustworthy AI, multi-step tool calling

## TL;DR

By defining reliability/inspectability cost functions for each external tool, LIT guides LLMs to select the lowest-cost (most transparent and auditable) tool-calling path among multiple candidates, improving interpretability while maintaining or enhancing task accuracy in 61 out of 65 test scenarios.

## Background & Motivation

**State of the Field**: LLM tool calling has become the dominant paradigm for extending model capabilities, enabling LLMs to invoke calculators, code executors, database queries, and predictive models to complete complex multi-step tasks. Prior works such as Toolformer, HuggingGPT, and CRAFT primarily focus on the correctness of tool calls and task completion rates.

**Limitations of Prior Work**: When selecting tools, LLMs are entirely task-success-oriented and neglect the substantial differences in **reliability and inspectability** across tools. For example, a calculator is fully reliable and auditable, whereas a BERT classifier or an ARIMA forecaster is difficult to debug and understand. LLMs often blindly select black-box tools even when equally accurate but more transparent alternatives exist.

**Root Cause**: In high-stakes scenarios, users require not only correct answers but also **trustworthy, auditable reasoning paths that facilitate rapid fault localization**. However, the current tool-calling paradigm lacks quantitative measures and optimization mechanisms for tool reliability—there are no standard frameworks, evaluation benchmarks, or baseline methods.

**Paper Goals**: To systematically bias LLM tool selection toward more reliable and inspectable tools without sacrificing task performance, making the final solutions more transparent and controllable for human users.

**Starting Point**: Drawing on HCI literature concerning system reliability, debuggability, and simplicity, the paper defines a three-dimensional cost for each tool—performance robustness (P), debugging difficulty (D), and parameter complexity (C)—and at inference time prompts the LLM to generate multiple candidate solutions and select the one with the lowest total cost.

**Core Idea**: Introduce the LIT (LLMs with Inspectable Tools) framework—without training the model, use carefully designed few-shot prompts combined with tool cost functions to automatically select the most inspectable tool-calling sequence from multiple candidates at inference time.

## Method

### Overall Architecture

The LIT framework consists of two core components: **(1) a tool cost function system** that quantifies the reliability and inspectability of each tool, and **(2) a reliability-guided prompt** that instructs the LLM to generate multiple candidate solutions, compute the total cost of each, select the lowest-cost solution, and execute it. The entire process requires no model training and is purely a prompting-based inference-time method.

### Key Design 1: Three-Dimensional Tool Cost Function

- **Function**: Compute Cost = P + D + C for each tool.
- **Mechanism**:
    - **P (Performance Robustness)**: The reliability of the tool's output across varying inputs. Calculator P=0 (fully reliable); ARIMA Forecaster P=4 (relies on stationarity assumptions).
    - **D (Debugging Difficulty)**: The difficulty of diagnosing errors. PandasInterpreter D=$\sqrt{\text{lines}} \times \max(\text{packages},1) \times 0.5$ (code is readable and modifiable); BERT TextualClassifier D=10 (enormous parameter count, not auditable).
    - **C (Parameter Complexity)**: The complexity of the tool's input parameters. Calculator C=1; TextualClassifier (BERT) C=8.
- **Design Motivation**: Simple tools (Calculator cost=2) substantially outperform black-box tools (BERT classifier cost=20; LLMInferencer cost=30). Cost values can be customized by users according to domain requirements.

### Key Design 2: Multi-Candidate Solution Generation and Comparison

- **Function**: At inference time, the LLM generates up to four candidate solutions, each consisting of a sequence of ordered tool calls.
- **Mechanism**: The LLM independently computes the sum of tool costs for each candidate, selects the solution with the lowest total cost that satisfies correctness constraints, and then executes the chosen tool-calling sequence.
- **Design Motivation**: This prevents the LLM from defaulting to the first solution it generates (typically a black-box one) and, through explicit comparison, makes the model aware of more transparent alternatives. For example, when predicting whether a paper is an oral presentation, the LLM can choose BERT (cost=20) or Logistic Regression (cost=7); the latter's coefficients are directly interpretable.

### Key Design 3: Customizable Few-Shot Prompt

- **Function**: A detailed prompt incorporating cost formula descriptions and five example solutions.
- **Mechanism**: The prompt includes: (a) a table of all tools and their cost formulas; (b) five example problems not in the test set along with their low-cost solutions; (c) explicit instructions requiring the generation of multiple candidates and selection of the lowest-cost solution.
- **Design Motivation**: A pure prompting approach requires no fine-tuning, is applicable to any LLM backbone, and allows users to flexibly adjust cost definitions to suit different domain requirements.

### Key Design 4: Challenging Benchmark Dataset

- **Function**: A benchmark of 1,300 questions spanning 13 question templates, 3 difficulty levels, and 2 external datasets.
- **Mechanism**: Using the Harvard USPTO patent dataset and the NeurIPS 2023 paper dataset, questions cover numerical computation, coding, and classification prediction. Easy questions (Q1–Q6) are optimally solved by transparent tools; Medium questions (Q7–Q10) are solvable by either class of tools; Hard questions (Q11–Q13) are better suited to black-box tools.
- **Design Motivation**: Existing tool-calling benchmarks evaluate only task success rates and lack an evaluation dimension for the reliability of tool selection.

## Loss & Training

This paper involves **no model training**. LIT is entirely an inference-time prompting framework. The core optimization objective is to minimize the total tool cost of a solution:

$$\text{Cost}(S) = \sum_{t \in S} \text{Cost}(t) = \sum_{t \in S} (P_t + D_t + C_t)$$

where $S$ is the tool-calling sequence of a candidate solution, and the solution with the lowest cost subject to correctness constraints is selected. 50% of the data is used for validation and 50% for testing.

## Key Experimental Results

### Main Results: Reliability/Inspectability Cost Comparison (Table 1)

| LLM | Easy Baseline → LIT | Medium Baseline → LIT | Hard Baseline → LIT |
|-----|---------------------|----------------------|---------------------|
| GPT-3.5 | 5.81 → **4.74** | 17.32 → **10.46** | 28.29 → **16.50** |
| GPT-4 | 5.70 → **5.17** | 25.06 → **13.52** | 30.00 → **30.86** |
| Gemini | 4.59 → **4.40** | 23.67 → **15.76** | 30.00 → 30.00 |
| Claude | 5.52 → **5.06** | 20.29 → **17.62** | 30.06 → **29.91** |
| Llama-3.1 | 5.66 → **5.19** | 13.21 → **12.66** | 17.57 → **12.24** |

**Key Findings**: LIT achieves equal or superior reliability/inspectability cost in **61 out of 65** test scenarios, with the most significant improvements on Medium-difficulty questions (GPT-3.5 medium cost reduced by 39.6%).

### Ablation Study / Performance Comparison (Table 2)

| LLM | Easy Accuracy Baseline → LIT | Medium Baseline → LIT | Hard Baseline → LIT |
|-----|------------------------------|----------------------|---------------------|
| GPT-3.5 | 0.60 → **0.62** | 0.33 → 0.25 | 0.23 → 0.19 |
| GPT-4 | 0.81 → **0.90** | 0.35 → 0.31 | 0.56 → **0.58** |
| Gemini | 0.64 → **0.71** | 0.33 → 0.31 | 0.56 → **0.65** |
| Claude | 0.86 → **0.95** | 0.37 → 0.35 | 0.73 → **0.72** |
| Llama-3.1 | 0.64 → 0.58 | 0.28 → 0.27 | 0.22 → **0.28** |

**Key Findings**: LIT maintains or improves performance in **48 out of 65** scenarios. Performance generally improves on Easy questions (using transparent tools is itself more accurate); slight performance degradation occurs on some Medium questions where black-box tools are genuinely stronger; results on Hard questions are mixed.

### Tool Cost Definitions (Table 3 Summary)

| Tool | P | D | C | Total Cost |
|------|---|---|---|------------|
| Calculator | 0 | 1 | 1 | **2** |
| DBLoader | 0 | 2 | 1 | **3** |
| PandasInterpreter | 0 | dynamic | dynamic | $\sqrt{\text{lines}} \times \max(\text{pkg},1)$ |
| Forecaster (ARIMA) | 4 | 3 | 1 | **8** |
| TextualClassifier (LR) | 3 | 2 | 2 | **7** |
| TextualClassifier (BERT) | 2 | 10 | 8 | **20** |
| LLMInferencer | 1 | 15 | 14 | **30** |

## Highlights & Insights

1. **First formal treatment of tool reliability/inspectability**: The paper quantifies qualitative reliability principles from HCI into a three-dimensional cost function (P+D+C), filling a gap in trustworthiness evaluation for the tool-calling paradigm.

2. **Reliability and performance are not in conflict**: Experiments show that selecting transparent tools (particularly for easy and medium questions) often simultaneously yields better accuracy—simple, readable code is less error-prone than complex black-box models.

3. **Zero training overhead**: The framework is entirely prompting-based and can be applied to any LLM—GPT, Claude, Gemini, Llama—without fine-tuning, resulting in minimal deployment cost.

4. **Compelling case study**: In the NeurIPS oral prediction example in Figure 3, LIT selects LR (cost=7) over BERT (cost=20); the two achieve comparable accuracy, but LR coefficients are fully interpretable, vividly demonstrating the value of the framework.

5. **Flexible cost customization**: Tool costs are not fixed; users can adjust them according to specific domain requirements and safety considerations, giving the framework strong generalizability.

## Limitations & Future Work

1. **High token overhead**: LIT requires the LLM to simultaneously generate and evaluate multiple candidate solutions (up to four), substantially increasing input and output token counts, which may be problematic under limited context window constraints.

2. **Subjectivity in cost function design**: The specific values of P, D, and C are manually assigned; differences in how "reliability" is understood across annotators or domains may lead to inconsistent cost definitions, and no automated calibration method is provided.

3. **Limited improvement on Hard questions**: For Hard questions that genuinely require black-box tools (e.g., topic classification of NeurIPS papers), LIT can hardly reduce costs because no transparent alternative tools exist.

4. **Small, closed tool set**: Experiments use only 8 predefined tools; scalability to open-ended tool sets (e.g., hundreds of APIs) is not validated, and cost computation over larger tool combinations may become considerably more complex.

5. **Absence of user studies**: While the paper claims improved debuggability, no human user experiments are conducted to verify whether users can actually debug LIT-selected solutions more easily.

## Related Work & Insights

| Dimension | LIT (Ours) | Toolformer | ToolLLM/RestGPT |
|-----------|-----------|------------|-----------------|
| **Optimization Objective** | Joint optimization of reliability + performance | Task performance only | Task success rate only |
| **Tool Selection** | Cost-function-guided multi-candidate comparison | Model learns autonomously when to call tools | Fixed invocation strategy |
| **Training Requirement** | None (pure prompting) | Requires self-supervised fine-tuning | Requires demonstration data |
| **Interpretability Metric** | Three-dimensional quantitative cost | None | None |
| **Insight** | Tool calling should not focus solely on "getting the right answer," but also on "whether users can understand and debug the answer afterward"—reliability is a second optimization dimension for tool calling. |

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | First to formalize tool reliability/inspectability as a cost function integrated into the tool selection pipeline; a genuinely fresh perspective. |
| Experimental Thoroughness | ⭐⭐⭐ | Coverage across 5 LLMs × 13 question templates is reasonably broad, but lacks user studies, cost sensitivity analysis, and open tool set experiments. |
| Writing Quality | ⭐⭐⭐⭐ | Problem definition is clear, framework diagrams are intuitive, and case analyses are persuasive. |
| Overall Value | ⭐⭐⭐⭐ | Raises an important but overlooked dimension in tool calling (reliability); the direction is sound and the framework is practical, though depth can be further developed beyond a workshop paper. |

## Potential Research Directions

1. **Automatic cost calibration**: Current cost functions are entirely manually specified; future work could explore automatically learning per-tool reliability and inspectability scores for specific domains from historical call logs, reducing subjective bias.
2. **Open tool set scaling**: Extending the framework from 8 closed tools to hundreds of open API scenarios, studying the scalability of cost comparison and candidate generation as the number of tools grows.
3. **Human studies on debugging efficiency**: Conducting rigorous user studies to quantify whether transparent solutions selected by LIT genuinely help human users locate and fix errors more quickly.
4. **Integration with reinforcement learning**: Incorporating the cost function as a reward signal in RLHF/DPO training to internalize the model's preference for reliable tools rather than relying on prompts.
5. **Dynamic cost adjustment**: Dynamically adjusting tool costs based on context (e.g., whether data distributions are stationary) rather than using static preset values.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](../../ICLR2026/llm_agent/judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[NeurIPS 2025\] Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools](attractive_metadata_attack_inducing_llm_agents_to_invoke_malicious_tools.md)
- [\[NeurIPS 2025\] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)
- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](../../ACL2026/llm_agent/creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)

<!-- RELATED:END -->
