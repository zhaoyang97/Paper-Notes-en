---
title: >-
  [Paper Note] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization
description: >-
  [ACL 2026][Code Intelligence][Factual Consistency] This paper proposes ReFEree, a reference-free and fine-grained factual consistency evaluation method for real-world code summarization. ReFEree defines four categories o…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Factual Consistency"
  - "Code Summarization"
  - "Reference-Free Evaluation"
  - "Fine-Grained Evaluation"
  - "Dependency Analysis"
date: 2026-05-08
content_hash: be811623baf2bfa7
---

# ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization

**Conference**: ACL 2026  
**arXiv**: [2604.10520](https://arxiv.org/abs/2604.10520)  
**Code**: [GitHub](https://github.com/bsy99615/ReFEree)  
**Area**: Code Intelligence / Code Summarization Evaluation  
**Keywords**: Factual Consistency, Code Summarization, Reference-Free Evaluation, Fine-Grained Evaluation, Dependency Analysis

## TL;DR

This paper proposes ReFEree, a reference-free and fine-grained factual consistency evaluation method for real-world code summarization. ReFEree defines four categories of inconsistency criteria and performs evaluation at the sentence-segment level. By incorporating a dependency information search mechanism, it achieves a 15-18% improvement in correlation with human judgment on Python and Java compared to the previous SOTA.

## Background & Motivation

**Background**: LLMs (such as GPT-4, Codex, GitHub Copilot, Claude Code) are being widely integrated into real-world development workflows to automatically generate long and descriptive code summaries. However, when summaries inaccurately reflect the actual implementation of the code, they can lead to developer misunderstandings, delayed debugging, and increased maintenance costs.

**Limitations of Prior Work**: (1) Reference-based methods (ROUGE, BLEU, METEOR) rely on human-written reference summaries, but code summarization is a one-to-many task—semantically correct summaries may use entirely different wording. (2) LLM-as-judge methods treat the summary as a whole and use a single criterion to produce binary or coarse 5-point scores, failing to provide fine-grained evaluation or locate specific inconsistent sentences and their causes. (3) Existing methods evaluate based only on the input code, ignoring external dependency definitions of functions/classes in real code—summaries often describe elements defined externally, but evaluation lacks this context.

**Key Challenge**: Real-world code summaries are increasingly long and descriptive, containing multiple sentences covering multiple functionalities and frequently involving external dependency elements, yet existing evaluation methods are neither fine-grained nor account for dependency context.

**Goal**: Design a reference-free, fine-grained factual consistency evaluation method that considers code dependencies, capable of locating inconsistencies and explaining the reasons behind them.

**Key Insight**: Starting from the actual error patterns of LLM-generated summaries, the authors empirically analyze and induce four typical inconsistency criteria, and then inspect them one by one at the sentence-segment level.

**Core Idea**: Segregate the summary into sentence segments, evaluate each segment against four criteria, and simultaneously search for relevant dependency information through a project context graph as objective evidence, finally aggregating these into an overall score.

## Method

### Overall Architecture

Construct project context graph (AST parsing) → Search for code-related dependency information (DFS + 1-hop strategy) → Segment the summary into sentence segments → Evaluate each segment using LLMs based on four criteria → Aggregate segment scores into an overall consistency score.

### Key Designs

1.  **Four Categories of Factual Inconsistency Criteria**:
    - **Function**: Provides systematic and actionable evaluation dimensions for code summarization assessment.
    - **Mechanism**: Through empirical analysis of 300 LLM-generated summaries (3 models × 100 functions), error patterns were manually annotated to induce four criteria: [C1] Name Inconsistency (14%)—incorrect identifier names; [C2] Type Inconsistency (15%)—wrong return types or variable types; [C3] Functional Inconsistency (35%)—the described functionality does not match the actual implementation, often due to ignoring dependencies; [C4] Context Irrelevance (33%)—contains unnecessary or irrelevant content (hallucinations).
    - **Design Motivation**: A single "factual consistency" criterion is too general; different types of errors impact code understanding differently. C3 and C4 together account for 68%, indicating that functional errors and hallucinations are the primary issues.

2.  **Code-Related Information Search Mechanism**:
    - **Function**: Provides objective external dependency information as evidence for evaluation.
    - **Mechanism**: A two-step approach—(1) Build a project context graph: Traverse the AST to represent code entities as nodes and dependencies as directed edges; (2) Select key information: Traverse the graph using DFS, searching only for the 1-hop dependency context of three core entity types: functions, classes, and variables. External dependencies retrieve pre-defined API documentation.
    - **Design Motivation**: Including all project context introduces noise. Research indicates that noise increases with the number of hops in multi-hop searches, so it is limited to 1-hop. This allows the evaluation to accurately judge whether descriptions involving externally defined elements are consistent.

3.  **Sentence-Level Fine-Grained Scoring**:
    - **Function**: Locates inconsistencies and provides interpretable evaluations.
    - **Mechanism**: Use NLTK to segment the summary into sentence segments $\mathcal{D} = \{S_1, ..., S_n\}$. Each segment is evaluated against the four criteria: $f(S, C)$ outputs 0 (inconsistency detected) or 1 (consistent). The overall score is:
      $$\text{SCORE} = \frac{1}{|\mathcal{D}| \times |Criteria|} \sum_{S \in \mathcal{D}} \sum_{C \in Criteria} f(S, C)$$
    - **Design Motivation**: The decompose-aggregate approach supports both fine-grained inconsistency localization and type identification while providing an interpretable derivation process for the overall score.

### Loss & Training

ReFEree is a training-free evaluation method. Main experiments use GPT-4.1-mini as the segment-level criteria evaluator with temperature 0.1, top-p 0.9, and top-k 50. The evaluation cost per sample is only $0.004. It supports various LLMs (open-source/closed-source) as evaluators.

## Key Experimental Results

### Main Results

| Method | Python Avg(rp/rs/τ) | Java Avg(rp/rs/τ) |
| :--- | :--- | :--- |
| ROUGE-L | 0.037 | 0.172 |
| BERTScore | 0.005 | 0.150 |
| G-Eval (Prev. SOTA) | 0.400 | 0.406 |
| CODERPE | 0.392 | 0.401 |
| ReFEree (w/o info) | 0.404 | 0.438 |
| ReFEree (w/ info) | **0.459** (+15%) | **0.480** (+18%) |

### Ablation Study

| Configuration | Python | Java | Description |
| :--- | :--- | :--- | :--- |
| C1 Only (Name) | 0.394 | 0.318 | Weakest single criterion, lower than G-Eval |
| C3 Only (Function) | 0.419 | 0.391 | Strongest single criterion, exceeds G-Eval |
| All Four Criteria | **0.459** | **0.480** | Multi-criteria synergy is optimal |
| No Dependency Info | 0.404 | 0.438 | Search module contributes ~0.05 gain |

### Key Findings
- ReFEree significantly outperforms all 13 baselines on both Python and Java, with a 15-18% improvement over the previous SOTA (G-Eval).
- Sentence-level evaluation accuracy reaches 93.4% (Python) and 93.0% (Java), indicating that LLMs can reliably perform fine-grained criterion judgments.
- Reference-based methods (BLEU/ROUGE) show extremely low correlation with human judgment (<0.05), making them nearly ineffective in code summarization scenarios.
- Functional inconsistency (C3) has the greatest impact on human judgment, while name inconsistency (C1) has the least.
- The method performs robustly across different LLM evaluators (Llama-8B, Mistral-7B, GPT-4.1-mini, etc.).

## Highlights & Insights
- Refining factual inconsistency criteria from "overall consistency" into four orthogonal dimensions is a core methodological contribution, making the evaluation interpretable and actionable.
- The design of building a project context graph based on AST and limiting it to 1-hop search balances information completeness and noise control.
- The construction process of the evaluation benchmark (Human-AI collaborative annotation, with Krippendorff's α reaching 0.74-0.84) ensures reliability.
- The cost of $0.004 per sample gives the method strong practical deployability.

## Limitations & Future Work
- Primarily validated on Python and Java; generalizability to other programming languages remains unverified.
- Reliance on LLMs as evaluators is limited by the code understanding capabilities of the LLM.
- The four criteria were empirically induced from 300 samples and may not cover all types of inconsistencies.
- Currently, dependency information is obtained only through static code analysis; the ability to handle dynamic analysis or complex cross-file dependencies is limited.

## Related Work & Insights
- **vs G-Eval**: G-Eval uses a single "factual consistency" criterion to score the overall summary; ReFEree improves by 15-18% through multi-criteria segment-level evaluation.
- **vs FactScore**: FactScore decomposes into atomic facts but uses only a single consistency criterion for each; ReFEree’s multi-criteria design is more comprehensive.
- **vs SIDE**: SIDE uses contrastive learning to evaluate semantic alignment but performs poorly on long descriptive summaries; ReFEree is specifically designed for long summaries.
- **vs Maharaj et al.**: That work performs binary detection at the entity level without explaining reasons and relies on the LLM's internal knowledge; ReFEree provides explanations for reasons and explicitly models dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a four-criteria fine-grained evaluation framework and dependency information search is novel and practical, though the core ideas (LLM-as-judge + fine-grained decomposition) have prior foundations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive with 13 baseline comparisons, multi-language validation, segment-level/summary-level evaluation, multi-LLM robustness, and thorough ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivation, intuitive method flowcharts, rigorous experimental organization, and a balance of quantitative and qualitative analysis.
- Value: ⭐⭐⭐⭐ Direct practical value for code summarization quality evaluation, low-cost deployability, and evaluation criteria that can be extended to other code understanding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2026\] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)
- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)
- [\[ACL 2026\] AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor](automonitor-bench_evaluating_the_reliability_of_llm-based_misbehavior_monitor.md)

</div>

<!-- RELATED:END -->
