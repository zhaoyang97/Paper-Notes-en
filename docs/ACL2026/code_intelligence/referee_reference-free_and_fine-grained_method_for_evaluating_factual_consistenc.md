---
title: >-
  [Paper Note] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization
description: >-
  [ACL 2026][Code Intelligence][Factual Consistency] This paper proposes ReFEree, a reference-free and fine-grained factual consistency evaluation method for real-world code summarization. It defines four categories of inc…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Factual Consistency"
  - "Code Summarization"
  - "Reference-Free Evaluation"
  - "Fine-Grained Evaluation"
  - "Dependency Analysis"
date: 2026-05-08
content_hash: f1bad693c266fbde
---

# ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization

**Conference**: ACL 2026
**arXiv**: [2604.10520](https://arxiv.org/abs/2604.10520)
**Code**: [GitHub](https://github.com/bsy99615/ReFEree)
**Area**: Code Intelligence / Code Summarization Evaluation
**Keywords**: Factual Consistency, Code Summarization, Reference-Free Evaluation, Fine-Grained Evaluation, Dependency Analysis

## TL;DR

This paper proposes ReFEree, a reference-free and fine-grained factual consistency evaluation method for real-world code summarization. It defines four categories of inconsistency criteria and evaluates at the sentence-segment level. Combined with a dependency information retrieval mechanism, ReFEree achieves 15–18% improvement in human judgment correlation over the previous state of the art on Python and Java.

## Background & Motivation

**Background**: LLMs (GPT-4, Codex, GitHub Copilot, Claude Code, etc.) are being increasingly integrated into real-world development workflows to automatically generate long, descriptive code summaries. However, when summaries inaccurately reflect the actual implementation, they can mislead developers, delay debugging, and increase maintenance costs.

**Limitations of Prior Work**: (1) Reference-based methods (ROUGE, BLEU, METEOR) rely on human-written reference summaries, but code summarization is a one-to-many task—semantically correct summaries may use entirely different wording. (2) LLM-as-judge approaches treat the summary as a whole, producing binary or coarse-grained 5-point scores under a single criterion, without the ability to localize which sentences are inconsistent or explain why. (3) Existing methods evaluate based only on the input code, ignoring external dependency definitions that real-world functions/classes rely on—summaries often describe externally defined elements, yet this context is absent during evaluation.

**Key Challenge**: Real-world code summaries are increasingly long and descriptive, covering multiple functional points across multiple sentences and frequently involving external dependencies, yet existing evaluation methods are neither fine-grained nor dependency-aware.

**Goal**: Design a reference-free, fine-grained, dependency-aware factual consistency evaluation method that can localize inconsistencies and explain their causes.

**Key Insight**: Starting from empirical analysis of actual error patterns in LLM-generated summaries, the paper inductively identifies four representative inconsistency criteria and evaluates each at the sentence-segment level.

**Core Idea**: Segment summaries into sentence units, evaluate each segment against four criteria, retrieve relevant dependency information from a project context graph as objective evidence, and aggregate segment scores into an overall consistency score.

## Method

### Overall Architecture

Build a project context graph (via AST parsing) → Retrieve code-relevant dependency information (DFS + 1-hop strategy) → Segment summaries into sentence units → Evaluate each segment against four criteria using an LLM → Aggregate segment scores into an overall consistency score.

### Key Designs

1. **Four Factual Inconsistency Criteria**:

    - Function: Provides systematic and actionable evaluation dimensions for code summarization assessment.
    - Mechanism: Through empirical analysis of 300 LLM-generated summaries (3 models × 100 functions) with manual error annotation, four criteria are inductively derived: [C1] Name Inconsistency (14%)—incorrect identifier names; [C2] Type Inconsistency (15%)—incorrect return types or variable types; [C3] Functionality Inconsistency (35%)—described functionality does not match actual implementation, often due to ignored dependencies; [C4] Context Irrelevance (33%)—contains unnecessary or unrelated content (hallucination).
    - Design Motivation: A single "factual consistency" criterion is too coarse; different error types have different impacts on code comprehension. C3 and C4 together account for 68%, indicating that functional errors and hallucinations are the dominant issues.

2. **Code-Relevant Information Retrieval Mechanism**:

    - Function: Provides objective external dependency information as evaluation evidence.
    - Mechanism: A two-step approach—(1) Build a project context graph by traversing the AST, representing code entities as nodes and dependency relations as directed edges; (2) Select key information via DFS traversal of the graph, retrieving only 1-hop dependency context for three core entity types (functions, classes, variables), with external dependencies resolved through predefined API documentation.
    - Design Motivation: Including all project context introduces noise. Studies show that multi-hop search accumulates noise with each additional hop, so the search is restricted to 1-hop. This enables accurate assessment of whether descriptions involving externally defined elements are consistent.

3. **Sentence-Segment-Level Fine-Grained Scoring**:

    - Function: Localizes inconsistencies and provides interpretable evaluation.
    - Mechanism: NLTK is used to segment summaries into units $\mathcal{D} = \{S_1, ..., S_n\}$. Each segment is evaluated against each criterion: $f(S, C)$ outputs 0 (inconsistency detected) or 1 (consistent). The overall score is: $\text{SCORE} = \frac{1}{|\mathcal{D}| \times |Criteria|} \sum_{S} \sum_{C} f(S, C)$
    - Design Motivation: The decompose-then-aggregate approach supports fine-grained inconsistency localization and type identification while providing an interpretable derivation of the overall score.

### Loss & Training

ReFEree is a training-free evaluation method. The main experiments use GPT-4.1-mini as the segment-level criterion evaluator, with temperature 0.1, top-p 0.9, and top-k 50. The per-sample evaluation cost is only \$0.004. Multiple LLMs (open-source and proprietary) are supported as evaluators.

## Key Experimental Results

### Main Results

| Method | Python Avg(rp/rs/τ) | Java Avg(rp/rs/τ) |
|--------|------|------|
| ROUGE-L | 0.037 | 0.172 |
| BERTScore | 0.005 | 0.150 |
| G-Eval (Prev. SOTA) | 0.400 | 0.406 |
| CODERPE | 0.392 | 0.401 |
| ReFEree (w/o info) | 0.404 | 0.438 |
| ReFEree (w/ info) | **0.459** (+15%) | **0.480** (+18%) |

### Ablation Study

| Configuration | Python | Java | Note |
|------|---------|------|----|
| C1 only (Name) | 0.394 | 0.318 | Weakest single criterion, below G-Eval |
| C3 only (Functionality) | 0.419 | 0.391 | Strongest single criterion, surpasses G-Eval |
| All four criteria | **0.459** | **0.480** | Multi-criteria synergy is optimal |
| Without dependency info | 0.404 | 0.438 | Retrieval module contributes ~0.05 gain |

### Key Findings
- ReFEree substantially outperforms all 13 baselines on both Python and Java, achieving 15–18% improvement over the previous SOTA (G-Eval).
- Sentence-segment-level evaluation accuracy reaches 93.4% (Python) and 93.0% (Java), demonstrating that LLMs can reliably perform fine-grained criterion-based judgments.
- Reference-based methods (BLEU/ROUGE) exhibit extremely low correlation with human judgments (<0.05), rendering them nearly ineffective for code summarization.
- Functionality inconsistency (C3) has the greatest impact on human judgment, while name inconsistency (C1) has the least.
- The method demonstrates robustness across different LLM evaluators (Llama-8B, Mistral-7B, GPT-4.1-mini, etc.).

## Highlights & Insights
- Decomposing factual inconsistency criteria from "overall consistency" into four orthogonal dimensions is the methodological core contribution, making evaluation both interpretable and actionable.
- Building a project context graph from ASTs and restricting retrieval to 1-hop strikes a balance between information completeness and noise control.
- The benchmark construction process (Human-AI collaborative annotation, Krippendorff's α of 0.74–0.84) ensures reliability.
- A per-sample cost of \$0.004 makes the method highly deployable in practice.

## Limitations & Future Work
- Validation is primarily conducted on Python and Java; generalizability to other programming languages has not been verified.
- The method relies on LLMs as evaluators, and is thus constrained by their code comprehension capabilities.
- The four criteria are inductively derived from 300 samples and may not cover all inconsistency types.
- The current approach supports only static code analysis for dependency retrieval; handling dynamic analysis or complex cross-file dependencies remains limited.

## Related Work & Insights
- **vs G-Eval**: G-Eval scores the overall summary under a single "factual consistency" criterion; ReFEree achieves 15–18% improvement through multi-criteria sentence-segment-level evaluation.
- **vs FactScore**: FactScore decomposes summaries into atomic facts but applies only a single consistency criterion to each; ReFEree's multi-criteria design is more comprehensive.
- **vs SIDE**: SIDE uses contrastive learning to assess semantic adequacy but performs poorly on long, descriptive summaries; ReFEree is specifically designed for long summaries.
- **vs Maharaj et al.**: That work performs binary entity-level detection without explaining causes and relies on the LLM's internal knowledge; ReFEree provides causal explanations and explicitly models dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a four-criteria fine-grained evaluation framework with dependency information retrieval is novel and practical, though the core approach (LLM-as-judge + fine-grained decomposition) builds on prior foundations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparisons against 13 baselines, multi-language validation, segment-level and summary-level evaluation, multi-LLM robustness, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clearly articulated, method diagrams are intuitive, experimental organization is rigorous, and both quantitative and qualitative analyses are provided.
- Value: ⭐⭐⭐⭐ Directly applicable to code summarization quality assessment, low-cost and deployable, with evaluation criteria extensible to other code comprehension tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2026\] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)

</div>

<!-- RELATED:END -->
