---
title: >-
  [Paper Note] LegalAgentBench: Evaluating LLM Agents in Legal Domain
description: >-
  [ACL2025][LLM Agent][legal domain] Proposes LegalAgentBench, a comprehensive evaluation benchmark for LLM Agents in the Chinese legal domain. It consists of 17 real-world corpora, 37 tools, and 300 tasks covering multi-hop reasoning and writing, achieving fine-grained evaluation through keyword matching and process rate.
tags:
  - "ACL2025"
  - "LLM Agent"
  - "legal domain"
  - "benchmark"
  - "tool-use"
  - "multi-hop reasoning"
date: 2026-05-08
content_hash: a1985a5f3bf801f5
---

# LegalAgentBench: Evaluating LLM Agents in Legal Domain

**Conference**: ACL2025  
**arXiv**: [2412.17259](https://arxiv.org/abs/2412.17259)  
**Code**: [CSHaitao/LegalAgentBench](https://github.com/CSHaitao/LegalAgentBench)  
**Area**: LLM Agent  
**Keywords**: LLM Agent, legal domain, benchmark, tool-use, multi-hop reasoning

## TL;DR
Proposes LegalAgentBench, a comprehensive evaluation benchmark for LLM Agents in the Chinese legal domain. It consists of 17 real-world corpora, 37 tools, and 300 tasks covering multi-hop reasoning and writing, achieving fine-grained evaluation through keyword matching and process rate.

## Background & Motivation
- LLM Agents demonstrate immense potential in the legal domain, yet existing general-domain benchmarks (such as AgentBench and ToolBench) fail to capture the complexity and nuances of real judicial cognition and decision-making.
- Existing legal datasets mostly focus on relatively basic tasks (e.g., case retrieval, judgment prediction), whereas actual legal practice involves deep case analysis, legal reasoning, and comprehensive judgment based on a wealth of legal precedents.
- **Core Problem**: Deficiency of a standardized benchmark specifically tailored for evaluating LLM Agents' tool-use, multi-step reasoning, and domain knowledge application capabilities in legal scenarios.
- **Goal**: Constructing a comprehensive evaluation framework based on real-world legal data, containing rich tools and multi-level tasks.

## Method

### 1. Environment Design: Corpora and Tools
- **17 Real-world Corpora**: 14 structured tabular databases (company basic information, registration details, subsidiary information, legal documents, court profiles, law firm records, address logs, spending restriction cases, terminated executive cases, dishonest debtors list, administrative penalty records, etc.) + 3 document retrieval libraries (legal knowledge, statutory laws, guiding cases).
- **37 Professional Tools**, categorized into four types:
    - **Text Retrievers** (3 tools): Retrieve content relevant to queries from the document libraries, utilizing Embedding-3 as the default retriever.
    - **Mathematical Tools** (5 tools): Execute operations such as basic arithmetic (addition, subtraction, multiplication, division), sorting, and finding maximum/minimum values.
    - **Database Tools** (28 tools): Extract column content from specific databases based on query conditions.
    - **System Tools** (1 tool): The Finish tool, which parses execution feedback and returns the final answer.

### 2. Scalable Task Formulation Framework (6-Step Procedure)
1. **Planning Tree Construction**: Constructs a planning tree based on instantiation relations between tools. The root node represents the unknown entity (the starting point of the task), branches correspond to available tools, and child nodes contain the information obtained after tool execution.
2. **Path Selection**: Employs stratified sampling and a maximum coverage strategy to select paths with varying depths ($1$-hop to $5$-hop) and breadths from the planning tree, ensuring the diversity of task types and difficulty levels.
3. **Entity Selection**: Traverses all potential entities and chooses those that can successfully complete the pre-designed execution path.
4. **Question Rewriting**: Leverages GPT-4 to rewrite template-style questions into more natural expression styles closer to realistic user habits while concealing the solving paths.
5. **Answer Generation**: Programmatically extracts ground-truth answers from the corpora using the given starting entities and tool chains.
6. **Human Verification**: Manually verifies the correctness of all questions, solving paths, and answers.

### 3. Task Formalization
- At each time step $t$, the Agent executes action $a_t$, receives observation $o_t$, and updates state $s_{t+1} = u(s_t, a_t, o_t)$.
- The action is determined by a decision policy: $a_t = \pi(s_t, o_1, o_2, \dots, o_{t-1})$.
- The interaction iterates until the task is completed or the maximum iteration limit $T=10$ is reached.

### 4. Fine-Grained Evaluation Metrics
- **Success Rate**: Extracts the key answer keywords (`key_answer`) from the tool call paths, and calculates the overlap ratio between the Agent's final output and these keywords.
- **Process Rate**: Extra key intermediate step keywords (`key_middle`) are annotated to evaluate the completion status of each execution stage.
- **BERTScore**: Computes the text similarity between the generated answer and the ground-truth reference.

## Key Experimental Results

### Table 1: Task Statistics

| Attribute | 1-hop | 2-hop | 3-hop | 4-hop | 5-hop | Writing | ALL |
|---|---|---|---|---|---|---|---|
| Number of Tasks | 80 | 80 | 60 | 40 | 20 | 20 | 300 |
| Average Query Length | 88.29 | 87.90 | 99.37 | 118.33 | 110.25 | 1059.95 | 160.65 |
| Average Answer Length | 74.20 | 40.84 | 45.53 | 63.48 | 86.20 | 678.75 | 99.24 |
| Average Number of key_answer | 1.88 | 1.44 | 1.20 | 1.40 | 2.25 | 10.25 | 2.14 |

### Table 2: Success Rates of LLMs on LegalAgentBench (ReAct Method)

| Model | 1-hop | 2-hop | 3-hop | 4-hop | 5-hop | Writing | ALL |
|---|---|---|---|---|---|---|---|
| GPT-4o | **0.926** | **0.840** | **0.750** | **0.642** | **0.612** | 0.654 | **0.791** |
| Qwen-max | 0.906 | 0.792 | 0.633 | 0.583 | 0.608 | 0.666 | 0.742 |
| GLM-4-Plus | 0.913 | 0.810 | 0.642 | 0.617 | 0.430 | 0.766 | 0.750 |
| Claude-sonnet | 0.895 | 0.698 | 0.475 | 0.479 | 0.457 | 0.657 | 0.658 |
| GPT-4o-mini | 0.933 | 0.650 | 0.400 | 0.421 | 0.258 | 0.609 | 0.616 |
| GLM-4 | 0.879 | 0.677 | 0.417 | 0.388 | 0.243 | 0.594 | 0.606 |
| GPT-3.5 | 0.642 | 0.285 | 0.117 | 0.100 | 0.133 | 0.085 | 0.299 |
| LLaMA3.1-8B | 0.602 | 0.154 | 0.075 | 0.071 | 0.060 | 0.087 | 0.236 |

**Key Findings**:
- GPT-4o achieves the best overall success rate of 79.08% under the ReAct method, with relatively lower token usage.
- As the number of hops increases, the performance of all models drops significantly (from a peak of 93% in 1-hop to 61% in 5-hop), verifying the effectiveness of the task difficulty gradient.
- The ReAct method generally outperforms Plan-and-Solve and Plan-and-Execute on multi-hop questions, but at the cost of higher token consumption.
- On Writing tasks, ReAct performs poorly instead, because its step-by-step resolution mechanism is ill-suited for writing tasks that necessitate parallel processing.
- GPT-3.5 and LLaMA3.1-8B have success rates below 30%, showing severely inadequate tool-use capabilities.

## Highlights & Insights
- **First LLM Agent Evaluation Benchmark in the Legal Domain**: Fills the gap in Agent benchmarks for specialized vertical domains.
- **Scalable Task Construction Framework**: The 6-step planning tree-based process can be easily extended to incorporate new knowledge bases and tools.
- **Fine-Grained Evaluation**: Process Rate evaluates not only the final outcome but also measures intermediate step completion, providing deeper diagnostic insights.
- **Real-world Data**: The 17 corpora are entirely sourced from real-world legal scenarios, allowing updates over time to prevent model overfitting.

## Limitations & Future Work
- Currently covers only the Chinese legal system; future work needs to extend to multilingual environments and diverse legal jurisdictions.
- The scale of 300 tasks is relatively limited, which might not be sufficient to comprehensively evaluate all legal scenarios.
- Evaluation relies heavily on keyword matching, which may lead to false negatives for answers that are semantically equivalent but phrased differently.
- Task formulation relies on GPT-4 to rewrite queries, which may introduce specific LLM preferences.
- Detailed prompt templates used during evaluation are not fully open-sourced (only partially disclosed in the appendix).
- The Writing split contains only 20 tasks, which may not be representative due to the small sample size.

## Related Work & Insights
- **vs AgentBench**: AgentBench is a general multi-environment evaluation platform, while LegalAgentBench focuses on the legal vertical domain, offering domain-specific corpora and tools.
- **vs ToolBench/ToolQA**: ToolBench covers general API calling and ToolQA spans 8 generic fields, whereas LegalAgentBench penetrates deeply into the legal domain with highly specialized tools and corpora.
- **vs AgentBoard**: AgentBoard focuses on fine-grained progress rate evaluation in multi-turn interactions; LegalAgentBench adopts this methodology and applies it to the legal domain.
- **vs Existing Legal NLP Datasets**: Prior legal datasets focus on single tasks (retrieval/judgment prediction), while LegalAgentBench demands comprehensive capabilities in multi-hop reasoning and tool use.

## Rating
- Novelty: ⭐⭐⭐⭐ (First LLM Agent benchmark in the legal domain, filling an important gap)
- Experimental Thoroughness: ⭐⭐⭐⭐ (8 models x 3 methods, multi-dimensional metric analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, detailed task construction process)
- Value: ⭐⭐⭐⭐ (Significant reference value for the legal AI community, with methodology transferable to other domains)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](guidebench_guideline_following.md)
- [\[ACL 2025\] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2026\] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](../../ACL2026/llm_agent/mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)
- [\[ACL 2025\] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)

</div>

<!-- RELATED:END -->
