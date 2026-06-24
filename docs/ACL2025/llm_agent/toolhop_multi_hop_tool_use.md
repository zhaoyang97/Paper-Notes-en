---
title: >-
  [Paper Note] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use
description: >-
  [ACL 2025][LLM Agent][Tool Use] This paper proposes ToolHop, a benchmark dataset containing 995 multi-hop queries and 3,912 locally executable tools. By adopting a "query-driven" data construction paradigm (generating tools based on queries), the benchmark ensures genuine dependency relationships among tools and verifiable answers. Evaluation of 14 LLMs reveals that the strongest model, GPT-4o, achieves an accuracy of only 49%, highlighting significant strategy differences ac…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Tool Use"
  - "Multi-Hop Reasoning"
  - "Benchmark"
  - "Function Calling"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 88d301f4d3d616c2
---

# ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use

**Conference**: ACL 2025  
**arXiv**: [2501.02506](https://arxiv.org/abs/2501.02506)  
**Code**: [https://huggingface.co/datasets/bytedance-research/ToolHop](https://huggingface.co/datasets/bytedance-research/ToolHop)  
**Area**: LLM Agent  
**Keywords**: Tool Use, Multi-Hop Reasoning, Benchmark, Function Calling, LLM Evaluation

## TL;DR
This paper proposes ToolHop, a benchmark dataset containing 995 multi-hop queries and 3,912 locally executable tools. By adopting a "query-driven" data construction paradigm (generating tools based on queries), the benchmark ensures genuine dependency relationships among tools and verifiable answers. Evaluation of 14 LLMs reveals that the strongest model, GPT-4o, achieves an accuracy of only 49%, highlighting significant strategy differences across different model families in tool use.

## Background & Motivation

**Background**: Tool use (or function calling) of LLMs is a critical capability towards general intelligence. Most existing evaluation datasets adopt a "tool-driven" approach, where tools are collected first, and queries are subsequently generated for these tools.

**Limitations of Prior Work**: (a) Tool-driven methods cannot guarantee genuine dependency relationships between tools, and the generated queries often lack real multi-hop reasoning; (b) Existing datasets lack verifiable ground truth answers, relying on models like GPT-4 for process evaluation which introduces model bias; (c) Many tools depend on external APIs, preventing free and stable execution; (d) Tool feedback is insufficiently detailed, lacking effective correction feedback when models make mistakes.

**Key Challenge**: Multi-hop tool use requires collaborative capabilities in query decomposition, tool selection, parameter passing, and managing chain-of-dependency results, but existing evaluations fail to reliably isolate and measure these capabilities.

**Goal**: Build a high-quality evaluation benchmark for multi-hop tool use that ensures query diversity, genuine tool dependencies, local executability, detailed feedback, and verifiable answers.

**Key Insight**: Reverse the data construction pipeline by starting from multi-hop queries and customizing tools (including documentation and code) for each atomic sub-query, thereby naturally ensuring dependencies between tools.

**Core Idea**: Replace the "tool-driven" data construction paradigm with a "query-driven" approach, reverse-generating tool documentation and executable code starting from multi-hop questions to ensure evaluation authenticity and verifiability.

## Method

### Overall Architecture
Input: A multi-hop query $q$ (from the MoreHopQA dataset) that can be decomposed into atomic sub-queries $q_1, q_2, ..., q_l$. Output: A set of tool documentations $\text{doc}_{1..l}$ + code implementations $\text{fun}_{1..l}$ + ground truth answer $a$. The construction pipeline consists of three phases: tool creation, document refinement, and code generation.

### Key Designs

1. **Query-Driven Tool Creation**:

    - **Function**: Decompose the multi-hop query into a sequence of atomic sub-queries, and generate preliminary tool documentation for each sub-query.
    - **Mechanism**: Each sub-query $q_i$ depends on the preceding result $a_{i-1}$, naturally forming a chain of dependencies. The tool created for each $q_i$ not only solves the specific query but is also designed to generalize to similar queries.
    - **Design Motivation**: Existing tool-driven methods start with independently collected tools where no natural dependency exists, resorting to artificial dependencies. The query-driven approach starts from the question structure, making dependencies inherent.

2. **Document Refinement**:

    - **Function**: Expand preliminary tool documentation into complete documentation that is more complex and closer to real-world scenarios.
    - **Mechanism**: Expand in two directions: (a) functional expansion (adding features like result filtering, customizable formats); (b) parameter complexation (replacing simple string parameters with structured types like array, object). After refinement, the average number of parameters increases from 3.49 to 5.91, and the proportion of string parameters decreases by 12%.
    - **Design Motivation**: Simple tools fail to test a model's ability to comprehend complex parameter structures. Real-world APIs typically contain optional parameters and nested structures; refined tools are closer to practical application scenarios.

3. **Code Generation**:

    - **Function**: Generate locally executable Python functions for each tool documentation.
    - **Mechanism**: Function names are mapped from tool names, with parameter signatures derived from documentation specifications. A key step is utilizing the atomic query $q_i$ and answer $a_i$ as input constraints to ensure the function returns the correct answer when properly called. Meanwhile, exception handling mechanisms are incorporated to return detailed error messages for incorrect inputs.
    - **Design Motivation**: (a) Local execution runs with zero cost and avoids reliance on external APIs; (b) Detailed error feedback helps assess the error-correction capability of models; (c) Predefined answers enable automated, verifiable evaluations.

4. **Evaluation Dimension Design**:

    - Answer Correctness: Evaluated in three scenarios—Direct Answer (no tools), Forced Tool Use, and Free Choice.
    - Invocation Error Analysis: Tool hallucination (invoking non-existent tools), parameter hallucination (using undefined parameters), and parameter omission (missing required parameters).

### Loss & Training
The data construction employs GPT-4o for auxiliary processing. Dataset statistics: 995 multi-hop queries, 3,912 tools, covering 47 domains. Each query requires 3-7 tools (i.e., 3-7 hops).

## Key Experimental Results

### Main Results
Fourteen LLMs from five families (LLaMA-3.1, Qwen-2.5, Gemini-1.5, Claude-3.5, and GPT) were evaluated.

| Model | Direct Answer↑ | Forced Tool↑ | Free Choice↑ | Query Error Rate↓ | Instance Error Rate↓ |
|------|-----------|-----------|-----------|-------------|-------------|
| GPT-4o | 23.12 | **49.04** | 47.74 | 9.45 | 4.31 |
| GPT-4-Turbo | 18.59 | 47.94 | 46.83 | 10.95 | 4.97 |
| Claude3.5-Haiku | **36.08** | 38.09 | 44.72 | 23.48 | 15.81 |
| Qwen2.5-72B | 17.89 | 45.43 | 38.29 | 13.27 | 4.93 |
| LLaMA3.1-70B | 18.79 | 19.10 | 12.76 | 35.08 | 14.24 |
| 14-Model Average | 19.83 | 32.12 | 32.84 | 18.72 | 8.68 |

### Ablation Study (Feedback Analysis of the GPT Family)

| GPT Version | With Detailed Feedback↑ | No Detailed Feedback↑ | Correct→Incorrect | Incorrect→Correct |
|----------|-------------|-------------|-----------|-----------|
| GPT-4o | 47.87 | 24.47 | 25.53 | 2.13 |
| GPT-4o-mini | 38.53 | 11.93 | 29.36 | 2.75 |
| GPT-3.5-Turbo | 36.75 | 21.37 | 20.51 | 5.13 |

### Key Findings
- **Tool use provides significant gains but is far from saturated**: Implementing tool use yields an average gain of 12.29% in accuracy, and 23.59% for the GPT family. However, the best-performing GPT-4o only achieves 49%, indicating that multi-hop tool use remains highly challenging.
- **Distinct strategic differences exist among model families**: Qwen-2.5 heavily over-relies on parallel execution (the 14B version uses parallel tool calls in 70.1% of queries), leading to parameter hallucination; Claude-3.5 achieves outstanding performance even without tool usage due to its CoT reasoning strengths; LLaMA-3.1 does not support simultaneous text output and tool calls, which impairs CoT reasoning.
- **Detailed feedback is crucial**: GPT-4o scores 47.87% accuracy with detailed feedback, but drops to 24.47% (a 23.4% decline) when only simple error prompts are provided, underscoring that detailed tool feedback is vital for the error correction capability of models.
- **Parallel execution is a double-edged sword**: Qwen-2.5-72B showed significant performance improvement after reducing its parallel execution rate to 3.82%, demonstrating that sequential calls are more reliable in multi-hop scenarios.
- **Scaling law holds**: Within the same family, larger models generally perform better and exhibit fewer invocation errors.

## Highlights & Insights
- **Query-Driven Reverse Construction**: Designing tools from questions rather than questions from tools represents a reverse thinking paradigm that guarantees execution-driven correctness. This paradigm can be generalized to other scenarios requiring data construction with dependency relationships.
- **Locally Executable + Verifiable**: All 3,912 tools are locally executable without depending on external APIs, and feature predefined ground-truths for automated validation. This establishes a gold standard for tool-use evaluation benchmarks.
- **Insightful Cross-Family Model Comparison**: Disclosing distinct "dispositions" in tool-usage patterns among different model families (e.g., Qwen prefers parallel calls, Claude relies on self-thought, GPT relies on feedback). These insights are highly valuable for LLM developers.

## Limitations & Future Work
- **Lack of Improvement Solutions**: The paper acknowledges a lack of direct strategies to boost multi-hop tool use capability, although the dataset can be utilized for training but is left unverified.
- **Tool Capabilities Restricted to Computational Types**: Because tools are local code implementations, the benchmark cannot cover tool-use scenarios that require real network interactions (e.g., search engines, live databases).
- **Queries Derived from MoreHopQA May Lack Diversity**: The query distribution from the source dataset might bias toward specific knowledge domains.
- **Documentation Refinement Performed by GPT-4o**: This introduces model bias, potentially causing tool documentations to be excessively uniform in style.

## Related Work & Insights
- **vs ToolBench/ToolACE**: These tool-driven datasets construct queries from pre-collected API candidate sets, lacking genuine tool dependencies. ToolHop's query-driven approach fundamentally overcomes this limitation.
- **vs T-Eval**: While T-Eval evaluates single-step tool execution, ToolHop focuses on multi-hop scenarios, aligning better with realistic and complex challenges.
- **vs NestTools**: NestTools handles nested tool calls but still relies on tool-driven data construction; ToolHop's query-driven methodology is more systematic.

## Rating
- Novelty: ⭐⭐⭐⭐ The query-driven reverse construction methodology is highly novel, and the dataset design is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 14 models from 5 families in multiple dimensions with deep analysis of family-specific strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich charts, and in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Fills a gap in multi-hop tool-use evaluation, offering direct insights for model development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](meco_metacognition_tool_use.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2025\] Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model](can_a_single_model_master_both_multi-turn_conversations_and_tool_use_coalm_a_uni.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)

</div>

<!-- RELATED:END -->
