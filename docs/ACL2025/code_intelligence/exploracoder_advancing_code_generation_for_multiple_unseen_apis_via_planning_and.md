---
title: >-
  [Paper Note] ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration
description: >-
  [ACL 2025][Code Intelligence][Unseen API Code Generation] This work proposes the training-free ExploraCoder framework. It decomposes complex multi-API programming problems into subtasks through task planning, and progressively conducts experiments to accumulate experiences on correct API usages via Chained API Exploration (CoAE). It achieves up to 17.28% absolute improvement in pass@10 on multi-API unseen library benchmarks.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Unseen API Code Generation"
  - "Exploratory Programming"
  - "Chained API Exploration"
  - "Task Planning"
  - "RAG-Enhanced Code Generation"
  - "Torchdata"
date: 2026-05-08
content_hash: 00a86ae5cf3e9103
---

# ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration

**Conference**: ACL 2025  
**arXiv**: [2412.05366](https://arxiv.org/abs/2412.05366)  
**Code**: [https://github.com/greenlight2000/ExploraCoder](https://github.com/greenlight2000/ExploraCoder)  
**Authors**: Yunkun Wang, Yue Zhang, Zhen Qin, Chen Zhi, Binhua Li, Fei Huang, Yongbin Li, Shuiguang Deng  
**Affiliation**: Zhejiang University, Alibaba Group  
**Area**: Code Generation / Model Compression (API calls)  
**Keywords**: Unseen API Code Generation, Exploratory Programming, Chained API Exploration, Task Planning, RAG-Enhanced Code Generation, Torchdata  

## TL;DR

This work proposes the training-free ExploraCoder framework. It decomposes complex multi-API programming problems into subtasks through task planning, and progressively conducts experiments to accumulate experiences on correct API usages via Chained API Exploration (CoAE). It achieves up to 17.28% absolute improvement in pass@10 on multi-API unseen library benchmarks.

## Background & Motivation

**Background**: LLMs have demonstrated strong capability in code generation, but their performance drops significantly when facing library APIs unseen in their training data. Continuous library updates and the existence of private libraries make exhaustive retraining impractical.

**Limitations of Prior Work**:
   - **Continuous Pre-training**: Scarcity of training data for new libraries and high retraining cost.
   - **Standard RAG** (e.g., DocPrompting): Code is generated at once after retrieving API documentation, which performs poorly in complex scenarios involving multiple API interactions—the retriever struggles to find all relevant APIs for composite requirements, and the LLM easily hallucinates when coordinating interactions among multiple unfamiliar APIs at once.
   - **Improved Retrieval Methods** (e.g., CAPIR, EpiGen): These improve API document retrieval and pre-planning but ignore the limitations of LLMs/blockages in multi-API interaction reasoning and the ambiguity of API documentation.
   - **Reactive Agent (ReAct)**: End-to-end code construction still exposes the LLM's weaknesses in multi-API coordination.

**Core Motivation**: To simulate the **human exploratory programming paradigm**—when facing unfamiliar libraries, developers first read documentation to understand the overall capabilities of the library, then accumulate practical experience by progressively experimenting with individual API calls, and finally combine them into a complete solution.

## Method

### Overall Architecture (Figure 1)

ExploraCoder consists of three modules:

1. **Task Planning**: Decomposes complex programming problems into multiple API call subtasks.
2. **API Recommendation**: Retrieves and ranks relevant API documentation for each subtask.
3. **Chained API Exploration (CoAE)**: Progressively experiments with API calls for each subtask and accumulates experiences for the final code generation.

### 1. Task Planning

Utilizes LLM's ICL capability for planning:
- Provides high-level overview text $s$ of the library (automatically summarized from library documentation by GPT-3.5).
- Automatically extracts few-shot planning examples $D$ from library code examples.
- LLM generates $n$ API-related subtasks based on the query $\psi$ and context $(D, s)$.

**Design Motivation**: The planning granularity targets a level where each subtask requires only 1-2 unseen API calls.

### 2. API Recommendation

Two-stage retrieval pipeline:
1. **Dense Retrieval**: Process API documentation into tabular formats (path, signature, description), then use a dense retriever to retrieve top-$k$ APIs for each subtask by semantic similarity.
2. **LLM Reranking**: Prompt the LLM to rerank and denoise retrieved APIs at the subtask level $\to$ obtain subtask-level API subset $\tilde{\mathcal{A}}_i$.
3. **Global Reranking**: Simultaneously perform cross-task global API recommendation $\to$ obtain global API set $\tilde{\mathcal{A}}_G$.
4. The final API set provided to the generator is the union of both sets.

### 3. Chained API Exploration (CoAE) — Core Idea

Step-by-step exploration through the subtask sequence (indicated by grey blocks):

**Step 1: Experimental Code Generation**
- For each subtask $t_i$, the LLM generates $m=5$ diverse experimental code snippets based on the subtask description, library overview, recommended API docs, and **exploration experiences from previous subtasks** $\mathcal{E}_{1:i-1}$.

**Step 2: Code Execution & Observation**
- Execute each experimental code snippet directly in a sandbox environment.
- Collect executability $\delta$, error messages $\epsilon$, and program output $\gamma$.
- Encourage the LLM to print key information (e.g., formats of API returned objects) in the experimental code to expand its API usage knowledge.

**Step 3: Experience Selection Strategy**
- Prioritizes randomly selecting a successful execution candidate with valid output.
- If all fail, randomly select one failed candidate.
- The selected experience is passed to the next subtask and accumulated.

**Step 4 (Optional): Intermediate Self-Debugging**
- When all candidate codes for a subtask fail, prompt the LLM to debug.
- Resolve common simple errors (e.g., missing imports) and complex cross-subtask API interaction issues.

The final complete API exploration trajectory $\hat{\mathcal{E}} = \{\hat{\mathcal{E}}_i\}_{i=1}^n$ is provided to the final code generator along with recommended API docs.

### Benchmark Construction

**Torchdata-Manual** (Newly constructed):
- 100 hand-crafted programming problems, each involving **8-14** Torchdata APIs.
- Professional programmers design problems and screen reasonable combinations after randomly combining APIs.
- Currently the longest API call sequence among publicly available executable library code benchmarks.

**Torchdata-Github** (Existing):
- 50 problems from GitHub client code, involving 3-8 API calls.
- This paper supplements missing external resources in the original work.

## Key Experimental Results

### Main Results: GPT-3.5 (API Unseen Model) on Torchdata-Manual

| Method | pass@1 | pass@5 | pass@10 | pass@20 |
|------|--------|--------|---------|---------|
| Direct Generation | 0% | 0% | 0% | 0% |
| DocPrompting (naive RAG) | 0.19% | 0.89% | 1.66% | 2.81% |
| CAPIR | 3.01% | 6.75% | 8.21% | 9.66% |
| EpiGen | 2.16% | 4.40% | 5.23% | 5.86% |
| **ExploraCoder** | **7.00%** | **11.54%** | **13.84%** | **15.67%** |
| ExploraCoder + Self-Debug | **11.5%** | **18.32%** | **20.87%** | **23.51%** |

- ExploraCoder outperforms naive RAG by **12.18%** absolutely on pass@10, and CAPIR by **5.63%**.
- Further utility is demonstrated with self-debugging, rising to 20.87% (pass@10).

### API Unseen vs. Pre-trained Model Comparison

On Torchdata-Github:
- GPT-3.5 + ExploraCoder (pass@10 = 21.67%) outperforms direct generation with GPT-4-1106-preview (21.34%).
- GPT-4-0613 + ExploraCoder reaches 28.11% (pass@10), which is 4.04% higher than naive RAG.

### API Pre-trained Models Also Benefit

GPT-4-1106-preview on Torchdata-Manual:
- Direct Generation: pass@1 is only 0.16%
- + naive RAG: pass@1 = 3.19%
- + ExploraCoder: pass@1 = **14.62%** (absolute gain of 11.43%)

### Comparison with Agent Methods

| Method | pass@10 | success@10 |
|------|---------|-----------|
| ReAct | 2.95% | 12.45% |
| KnowAgent | 11.01% | 23.29% |
| ExploraCoder | 13.84% | 25.40% |
| ExploraCoder* (+ debug) | **20.87%** | **36.81%** |

ExploraCoder's chained exploration is more effective than end-to-end Agent methods.

## Highlights & Insights

1. **Elegant Emulation of the Exploratory Programming Paradigm**: Formalizes the "read documentation -> attempt code -> accumulate experiences -> assemble solution" workflow of human programmers facing unfamiliar libraries into an executable CoAE chain, which is natural and effective.
2. **Divide-and-Conquer Strategy to Resolve Retrieval Bottlenecks**: Decomposes composite requirements into simpler subtasks before API retrieval, naturally avoiding the retrieval degradation problem associated with complex queries.
3. **Experience Transmission Mechanism**: The execution results of each subtask (including output format information) are passed to subsequent subtasks, forming progressive knowledge accumulation.
4. **Torchdata-Manual Benchmark**: The complexity of 8-14 APIs per task significantly exceeds existing benchmarks (3-8), making it closer to real-world programming needs.
5. **No Training Required**: The entire framework is based on prompts and code execution, requiring no extra training or fine-tuning.

## Limitations & Future Work

1. **Evaluation on a Single Library**: Validated only on the Torchdata library; generalization to other programming languages/libraries requires further verification.
2. **Planning Quality Depends on LLMs**: The granularity and accuracy of task planning heavily depend on the capabilities of the LLM, which may not be suitable for smaller models.
3. **Computational Overhead**: CoAE requires generating and executing multiple experimental codes for each subtask, causing higher API call costs and latency.
4. **Simple Experience Selection Strategy**: Randomly selecting successful candidates may not be the optimal strategy. Smarter selection (e.g., based on code coverage) may bring further improvements.
5. **Inaccurate Area Label in Title/Metadata**: The "Model Compression" area label seems somewhat inaccurate, as this work is closer to code generation/tool utilization.

## Related Work & Insights

- **Library-oriented Code Generation**: Zan et al. (2022, 2023, 2024) systematically studied code generation tasks using external library APIs.
- **RAG Code Generation**: DocPrompting (Zhou et al. 2023), CAPIR (Ma et al. 2024) improved API retrieval.
- **CoT Code Generation**: EpiGen (Li et al. 2024) aided code generation using natural language pre-planning.
- **Agent Methods**: ReAct (Yao et al. 2022), KnowAgent employed reactive planning and debugging.
- **Exploratory Programming**: Sheil (1986), Beth Kery & Myers (2017) investigated exploratory programming behaviors of human programmers.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: CoAE's design emulates the human exploratory programming paradigm, cleverly passing API usage experience across subtasks.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Two benchmarks $\times$ multiple models $\times$ comparisons against multiple baselines + ablation studies.
- **Value** ⭐⭐⭐⭐⭐: Direct utility for practical scenarios where LLMs use unseen API libraries for programming, and the framework is easy to integrate.
- **Benchmark Contribution** ⭐⭐⭐⭐: Torchdata-Manual fills the gap for complex multi-API benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](../../ACL2026/code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)
- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](../../ACL2026/code_intelligence/pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ICLR 2026\] RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation](../../ICLR2026/code_intelligence/rpg_a_repository_planning_graph_for_unified_and_scalable_codebase_generation.md)
- [\[ACL 2026\] Bootstrapping Code Translation with Weighted Multilanguage Exploration](../../ACL2026/code_intelligence/bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)

</div>

<!-- RELATED:END -->
