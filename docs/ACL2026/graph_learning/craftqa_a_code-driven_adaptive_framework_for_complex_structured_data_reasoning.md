---
title: >-
  [Paper Note] CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning
description: >-
  [ACL2026][Graph Learning][Structured Data QA] CRAFTQA utilizes CodeSTEP to generate executable step-by-step Python reasoning code. When predefined operations are insufficient, CRAFT dynamically generates custom functions…
tags:
  - "ACL2026"
  - "Graph Learning"
  - "Structured Data QA"
  - "Code Reasoning"
  - "Dynamic Function Generation"
  - "Table QA"
  - "KG Reasoning"
date: 2026-05-08
content_hash: 0b14c1fd1096a307
---

# CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning

**Conference**: ACL2026  
**arXiv**: [2606.02170](https://arxiv.org/abs/2606.02170)  
**Code**: Not specified  
**Area**: Structured Data Reasoning / Knowledge Graph Question Answering  
**Keywords**: Structured Data QA, Code Reasoning, Dynamic Function Generation, Table QA, KG Reasoning  

## TL;DR
CRAFTQA utilizes CodeSTEP to generate executable step-by-step Python reasoning code. When predefined operations are insufficient, CRAFT dynamically generates custom functions, significantly enhancing complex structured data QA capabilities across tables, Knowledge Graphs (KGs), and Temporal KGs. The GPT-4o version achieves an overall score of 76.6% on complex reasoning tasks.

## Background & Motivation
**Background**: Structured data QA requires models to access tables, KGs, or Temporal KGs based on natural language questions and output verifiable answers. Recently, unified structured data QA methods, such as StructGPT, Readi, and TrustUQA, have attempted to handle multiple data types within a single framework.

**Limitations of Prior Work**: Unified frameworks typically rely on a fixed set of functions, such as query, set operations, counting, summation, and min/max. Once a question requires complex operations beyond predefined functions—such as multi-step numerical reasoning, complex conditional combinations, or customized calculations—the system becomes restricted.

**Key Challenge**: Fixed function sets ensure controllability and interpretability but limit reasoning expressiveness. Conversely, allowing LLMs to generate free-text answers is flexible but difficult to verify and execute. Complex structured data QA requires both "executability" and "dynamic expansion capability."

**Goal**: The authors aim to build a unified framework capable of performing reasoning via code execution across different structured data types, while automatically generating specialized functions when encountering out-of-predefined operations.

**Key Insight**: The paper decomposes reasoning into two layers: CodeSTEP manages the backbone of step-by-step code reasoning, while CRAFT dynamically generates custom functions for steps that exceed the library's capabilities during execution, returning results to the main execution flow.

**Core Idea**: Instead of merely calling fixed tools, the LLM writes an executable, verifiable function on the fly when a new tool is needed, then continues the structured reasoning.

## Method
CRAFTQA represents different structured data using a unified Condition Graph and converts natural language questions into executable code sequences. Standard operations in the code sequence call predefined functions; if a step cannot be expressed by a predefined function, the $f_{craft}$ interface is invoked. CRAFT then generates a custom function based on the current question, full code, task description, historical intermediate results, and the expected function signature.

### Overall Architecture
The input consists of a data source $\mathcal{D}$ and a question $\mathcal{Q}$. The system first converts the data source into a schema $\mathcal{D}_{schema}$ and a Condition Graph $\mathcal{D}_{cg}$. Under a few-shot prompt, the LLM generates a code sequence $\mathcal{C}=\{c_i\}_{i=1}^n$. The executor runs these codes sequentially to obtain the final answer $\mathcal{A}$. The overall process is summarized as $\mathcal{M}_{\theta}(\mathcal{D}_{schema}, \mathcal{Q}, \mathcal{P}) \rightarrow \mathcal{C}$, followed by $\textsc{Exec}(\mathcal{C}, \mathcal{D}_{cg}) \rightarrow \mathcal{A}$.

### Key Designs
1. **CodeSTEP Step-by-step Code Reasoning**:
    - **Function**: Explicitly transforms the natural language reasoning process into a sequence of Python executable code.
    - **Mechanism**: CodeSTEP first generates a natural language reasoning path and then generates the corresponding code for each step. Basic query functions $get$ retrieve entities from the Condition Graph based on relations, head/tail entities, comparators, and attribute conditions; set and numerical operations handle union, intersection, difference, min, max, mean, count, and sum.
    - **Design Motivation**: Code decouples reasoning from calculation, making answers executable and checkable while reducing LLM hallucinations in complex numerical and multi-hop operations.

2. **CRAFT Dynamic Function Generation**:
    - **Function**: Handles out-of-predefined operations that cannot be expressed by the predefined function set.
    - **Mechanism**: When a code step cannot be implemented using $\mathcal{F}_{pred}$, CodeSTEP generates an $f_{craft}(\mathcal{T}_i, W_i, F_{exp,i})$ call. CRAFT then generates a self-contained function $\hat{f}_i$ based on the original question, code sequence, current task instructions, prior results, and expected signature, returning result $w_i$ after execution.
    - **Design Motivation**: Complex questions often require ad-hoc functions, such as custom sorting, combined conditional logic, or data format conversion. Dynamically generating functions is more flexible than expanding a fixed library.

3. **Code Verification and Sequential Execution**:
    - **Function**: Increases the executability and stability of the generated code.
    - **Mechanism**: Both the CodeSTEP main code and CRAFT-generated custom functions undergo executability verification via a Python interpreter. If verification fails, the system retries with the same input up to a maximum of $T=3$ times. Verified code is executed sequentially, with each step's result added to the intermediate result set $W_{i+1}=W_i \cup \{w_i\}$.
    - **Design Motivation**: The core risk of code-driven methods is syntax or runtime errors. Pre-execution verification and limited retries can filter out many errors before answer generation.

### Loss & Training
CRAFTQA is a reasoning framework rather than a trained model. Various LLMs are used as reasoning engines, including GPT, LLaMA, DeepSeek, Gemini, and Qwen series. Reasoning strategies include 5-sample self-consistency, a maximum retry of $T=3$, and Sentence-BERT semantic entity alignment to map entity names in code to candidate entities in the Condition Graph.

## Key Experimental Results

### Main Results
| Method | Backbone | TableBench-FC DA | TableBench-NR DA | WikiSQL-E DA | Overall |
|------|----------|------------------|------------------|--------------|---------|
| PoT | GPT-4o | 51.0 | 43.7 | 27.4 | 32.6 |
| StructGPT | GPT-4o | 63.5 | 42.2 | 43.5 | 44.3 |
| Readi | GPT-4o | 62.5 | 49.5 | 51.3 | 51.5 |
| TrustUQA | GPT-4o | 62.5 | 29.6 | 80.1 | 67.2 |
| CRAFTQA | Qwen2.5-7B | 57.9 | 35.4 | 65.9 | 58.3 |
| CRAFTQA | LLaMA-3.1-8B | 57.3 | 31.1 | 76.0 | 64.4 |
| CRAFTQA | GPT-4o-mini | 64.6 | 40.7 | 79.0 | 69.2 |
| CRAFTQA | GPT-4o | 68.8 | 51.3 | 85.6 | 76.6 |

Using the same GPT-4o backbone, CRAFTQA's Overall score of 76.6 is significantly higher than TrustUQA (67.2), Readi (51.5), and StructGPT (44.3). Notably, CRAFTQA-LLaMA-3.1-8B (64.4) outperforms several traditional baselines using GPT-4o.

### Ablation Study
| Configuration | FC DA/F1 | NR DA/F1 | WikiSQL-E DA/F1 | Description |
|------|----------|----------|-----------------|------|
| CRAFTQA | 68.8 / 71.6 | 51.3 / 51.9 | 87.3 / 87.6 | Full Framework |
| w/o CRAFT | 65.3 / 67.7 | 45.9 / 46.3 | 84.5 / 84.6 | Without dynamic custom functions |
| w/o CRAFT & CodeSTEP | 59.4 / 63.2 | 18.4 / 19.7 | 79.8 / 80.0 | Without code reasoning backbone |

### Key Findings
- CRAFT contributes significantly to complex numerical reasoning. Numerical Reasoning (NR) DA drops from 51.3 (full model) to 45.9 (w/o CRAFT), and further to 18.4 without the code backbone.
- Out-of-predefined scenarios are the primary source of CRAFTQA's advantage. On WikiSQL-E, GPT-4o's Calling Denotation Accuracy is 53.57%, compared to only 6.98% for TrustUQA.
- In Numerical Reasoning, CRAFTQA (GPT-4o) achieves a DA of 56.06%, while TrustUQA achieves 29.29%, a gain of 26.77 percentage points.
- CRAFTQA does not sacrifice basic performance on standard reasoning tasks: WebQSP Hit@1 is 85.20, WikiSQL DA is 86.10, and CronQuestions Hit@1 is 97.10. The first two are slightly higher than TrustUQA, while the last is comparable to TrustUQA's 97.20.
- The framework scales with backbone capability. Fact Checking (FC) DA increases from 61.5 (GPT-3.5) to 68.8 (GPT-4o), and reaches 83.3 with o4-mini.

## Highlights & Insights
- A key insight of this paper is that "toolsets do not need to be predefined and fixed." In structured data QA, fixed function sets are controllable but rigid; CRAFT opens the tool space through dynamic code generation.
- There is a clear division of labor between CodeSTEP and CRAFT. The backbone steps maintain unified execution logic, while special steps are delegated to dynamic functions, preventing the uncontrolled generation of entire code blocks for every question.
- Calling Denotation Accuracy is a highly useful diagnostic metric. By evaluating questions that specifically require out-of-predefined functions, it directly verifies whether the CRAFT module addresses the target pain point.
- Results for small models are enlightening. A robust reasoning framework can allow 7B/8B models to approach or even surpass larger closed-source models using older frameworks in complex structured reasoning.

## Limitations & Future Work
- The gains are smaller on simple standard reasoning tasks because these tasks rarely require out-of-predefined operations, leaving little room for CRAFT's dynamic capabilities.
- The framework depends heavily on the backbone's code generation ability. For LLMs with weaker programming skills, both dynamic function generation and the main code sequence may fail or produce low-quality results.
- Code execution introduces safety and sandboxing requirements. While the paper discusses executability verification, practical deployment requires restricting access to the file system, networks, and resources.
- Future work could extend to more data formats, such as semi-structured documents, graph databases, and multi-table relational databases, while incorporating stronger static analysis or unit-test-based code verification.

## Related Work & Insights
- **vs. PoT / PAL**: PoT/PAL demonstrated that code improves numerical reasoning, but they were primarily geared toward general math or text problems; CRAFTQA applies code reasoning to unified structured data access.
- **vs. StructGPT / Readi**: These methods access structured data via iterative reading or path editing, where expressiveness is limited by operation design; CRAFTQA can generate functions when new operations are required.
- **vs. TrustUQA**: TrustUQA features strong Condition Graphs and unified queries but relies on fixed functions; CRAFTQA builds upon unified representation while adding dynamic operation capabilities.
- **Inspiration for Future Systems**: Structured data agents do not necessarily need to predefine all tools; they can provide a controlled dynamic tool generation interface and use execution verification to ensure basic reliability.

## Rating
- Novelty: ⭐⭐⭐⭐ Dynamic custom functions are naturally integrated with structured data QA, addressing a precise problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers complex, standard, out-of-predefined, cross-backbone, and ablation experiments with comprehensive evidence.
- Writing Quality: ⭐⭐⭐⭐ Formalization of methods is clear, and experiments are organized by Research Questions (RQs); discussion on code safety and runtime environments is slightly sparse.
- Value: ⭐⭐⭐⭐ Highly insightful for table/KG QA, enterprise data agents, and complex structured reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[ACL 2026\] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)
- [\[ICML 2026\] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models](../../ICML2026/graph_learning/finding_the_minimal_parameter_budget_for_implicit_reasoning_a_data_complexity_dr.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)

</div>

<!-- RELATED:END -->
