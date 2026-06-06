---
title: >-
  [Paper Note] BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback
description: >-
  [ICML2026][LLM Evaluation][Personalization Evaluation] The authors propose Bespoke, a benchmark comprising 2,870 sessions collected from 30 annotators over 3 weeks of real-world chat and search history. By constructing a…
tags:
  - "ICML2026"
  - "LLM Evaluation"
  - "Personalization Evaluation"
  - "Search-Augmented LLM"
  - "User Preferences"
  - "Diagnostic Feedback"
  - "Benchmark"
date: 2026-05-08
content_hash: e7204b9403e021fc
---

# BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback

**Conference**: ICML2026  
**arXiv**: [2509.21106](https://arxiv.org/abs/2509.21106)  
**Code**: https://github.com/augustinLib/BESPOKE  
**Area**: LLM Evaluation  
**Keywords**: Personalization Evaluation, Search-Augmented LLM, User Preferences, Diagnostic Feedback, Benchmark  

## TL;DR
The authors propose Bespoke, a benchmark comprising 2,870 sessions collected from 30 annotators over 3 weeks of real-world chat and search history. By constructing an evaluation framework with fine-grained preference scoring and diagnostic feedback, the study systematically evaluates the personalization capabilities of search-augmented LLMs. It finds that current models do not exceed an average score of 60 across all configurations, suggesting that the bottleneck in personalization lies in reasoning over history rather than generation.

## Background & Motivation

**Background**: Search-augmented LLMs (e.g., ChatGPT, Gemini) integrate retrieved information via RAG to answer user queries, significantly reducing cognitive load. Recent systems have begun utilizing users' chat and search histories to personalize responses.

**Limitations of Prior Work**: Despite increasing personalization capabilities, systematic evaluation remains insufficient. Existing benchmarks like LaMP-QA are limited to domain-specific QA interactions (e.g., StackExchange) and fail to cover realistic open-web scenarios. RAG-QA Arena and Search Arena provide only binary preference judgments, lacking fine-grained diagnostics for personalization quality.

**Key Challenge**: The same query may correspond to entirely different information needs and presentation preferences depending on the user background (e.g., one user may focus on environmental impact and prefer narrative explanations, while another focuses on performance metrics and prefers concise lists). There is a lack of a benchmark that possesses both "real user history" and "diagnostic feedback" for a comprehensive assessment.

**Goal**: To build a personalization benchmark for search-augmented LLMs that is both realistic (real user history) and diagnostic (fine-grained preference scores + feedback).

**Key Insight**: Effective personalization evaluation requires two key elements: realistic user interaction history to characterize preferences, and reasoning over that history to infer information needs. These elements are addressed through long-term, deep-engagement human annotation.

**Core Idea**: Recruit 30 annotators from diverse backgrounds to use dedicated Google accounts for 3 weeks of actual daily searching and chatting. After collecting complete user histories, annotators write queries and provide four-dimensional scores plus diagnostic feedback for model responses, creating a closed loop for training personalized evaluators.

## Method

### Overall Architecture
Given a query $q$ from user $u$, the user history is defined as $\mathcal{H}_u = \{\mathcal{S}_u, \mathcal{C}_u\}$ (search history + chat history). The search-augmented LLM first infers the information need $n_q$ from the history, retrieves relevant information accordingly, and finally generates a personalized response $r$. Bespoke's construction involves three phases: history collection → multi-stage human annotation → evaluation framework design.

### Key Designs

1.  **Realistic User History Collection and Multi-stage Annotation**:
    *   **Function**: Constructing authentic and diverse evaluation data.
    *   **Mechanism**: 30 annotators with diverse backgrounds (Shannon equitability index of 0.91) used dedicated Google accounts for 3 weeks, collecting 2,870 sessions (2,153 search + 717 chat, averaging 95.67 per person). Annotation followed three stages: (1) Writing a simple query $q$ and its gold information need $n_q^+$; (2) Scoring $k$ sampled responses across four dimensions and writing diagnostic feedback to form Response-Judgment (R-J) pairs; (3) Generating a gold response $r^+$ through iterative refinement.
    *   **Design Motivation**: Existing benchmarks rely on synthetic personas or domain-limited QA data, which fail to reflect the complexity and diversity of real-world user behavior.

2.  **Four-Dimensional Diagnostic Evaluation Framework**:
    *   **Function**: Providing fine-grained diagnostics of personalization quality.
    *   **Mechanism**: Defines four evaluation dimensions: Need Alignment, Content Depth, Tone, and Explanation Style. The evaluator $\mathcal{E}_p$ is based on GPT-5 using a few-shot setup: it first generates a query-specific gold rubric $\mathcal{R}_q^+$ from R-J pairs $\mathcal{D}_q$, then combines examples and gold information needs to score new responses and generate feedback: $(s, f) = \mathcal{E}(\mathcal{D}_q, \mathcal{R}_q^+, n_q^+, q, \hat{r})$.
    *   **Design Motivation**: Traditional binary preference judgments (chosen/rejected) cannot pinpoint specific dimensions of personalization failure; diagnostic feedback not only judges quality but also suggests improvements, serving as a supervisory signal for system optimization.

3.  **Gold Information Coverage Evaluation**:
    *   **Function**: Measuring the extent to which a response covers key information.
    *   **Mechanism**: Atomic claims are extracted from the gold response $r^+$ using GPT-5. Human-screened verifiable claims constitute the gold information set $\mathcal{I}_q^+ = \{i_{q,1}^+, \dots, i_{q,n}^+\}$. For a model response $\hat{r}$, each atomic claim is checked for correct expression to calculate recall: $\text{Recall}(\hat{r}) = |\mathcal{I}_{\hat{r}}| / |\mathcal{I}_q^+|$.
    *   **Design Motivation**: In open-web scenarios, information can be redundant or irrelevant; recall based on atomic claims allows for more precise evaluation of information delivery quality.

## Key Experimental Results

### Main Results: Search-Augmented LLM Personalization Evaluation

Evaluating 6 models across different user context configurations (Best config: query-aware + history selection + profile):

| Model | Need Align. | Content Depth | Tone | Style | Recall | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| o3-search (Best) | 59.07 | 63.73 | 85.20 | 73.87 | 30.53 | **62.48** |
| Gemini-2.5-Pro | 56.40 | 60.27 | 84.40 | 72.40 | 25.32 | 59.76 |
| Gemini-2.5-Flash | 55.73 | 61.03 | 82.83 | 71.73 | 28.09 | 59.88 |
| pplx-sonar | 55.80 | 59.90 | 85.13 | 72.37 | 25.50 | 59.74 |
| pplx-sonar-reasoning | 54.27 | 57.47 | 83.33 | 70.67 | 23.93 | 57.93 |
| GPT-4o-search | 53.80 | 57.20 | 84.83 | 69.93 | 19.23 | 57.00 |
| o3-search (No Personalization) | 51.60 | 57.47 | 78.53 | 70.00 | 22.05 | 55.93 |

### Meta-evaluation: Alignment between Evaluator and Human Judgment

| Evaluator Config | Pearson Corr. (Avg.) | Spearman Corr. (Avg.) | Feedback Acc. (Avg.) |
| :--- | :--- | :--- | :--- |
| w/o Personalization | 0.470 | 0.477 | 0.360 |
| w/o Feedback | 0.809 | 0.814 | 0.801 |
| **w/ Feedback (Bespoke)** | **0.847** | **0.853** | **0.881** |

### Key Findings

*   **User Context Significantly Improves Personalization**: All models showed improvements across metrics when user history was introduced, though Recall remained the lowest metric (max 30.53%), indicating that precise information delivery remains challenging.
*   **Query-aware Profile > Static Profile > Raw History**: Dynamically constructing a query-related user profile is more effective than using the full history or a fixed profile.
*   **Bottleneck is Reasoning, Not Generation**: In Oracle experiments where the gold information need was provided directly, o3-search's Need Alignment surged to 83.47 and Tone reached 88.13. This suggests models possess the capacity to generate personalized responses, but inferring preferences from history is the primary bottleneck.
*   **Reasoning Models are More Sensitive to Search Quality**: When 70% noise was injected, Sonar-Reasoning's performance dropped by 23.13%, significantly exceeding Sonar's 16.78% drop.

## Highlights & Insights
*   The first benchmark for personalized search LLMs featuring both real user history and diagnostic feedback, with data collected over 3 weeks from 30 annotators across 2,870 real sessions.
*   Diagnostic feedback serves not only as an evaluation metric but also as a supervisory signal for improving personalization systems, creating a "Evaluate→Diagnose→Improve" loop.
*   Query expansion (CoT/Pseudo-history) can improve historical retrieval nDCG@10 from 0.082 to over 0.38, providing a practical solution for efficient user history retrieval.
*   The open-source evaluator can utilize open-source models (e.g., GPT-oss-120B, Qwen3-235B) instead of GPT-5 while maintaining high consistency.

## Limitations & Future Work
*   The annotator pool is limited to 30 people; while diverse, the scale may not cover all real user types.
*   The framework relies on LLM-as-judge; although meta-evaluations show high consistency, inherent bias risks remain.
*   History collection was limited to 3 weeks; long-term preference drift was not considered.
*   Recall metrics for atomic claim extraction and judgment rely on GPT-5, which may introduce cascading errors.

## Related Work & Insights
*   **LaMP Series** (Salemi et al.): Early personalization benchmarks based on synthetic personas, limited to specific domains like StackExchange.
*   **Search Arena** (Miroyan et al.): Evaluation of search LLMs in open-web settings, but restricted to binary preference judgments.
*   **RAG-QA Arena** (Han et al.): Long-context QA evaluation, but limited to professional domains without personalization dimensions.
*   Bespoke’s "query expansion + history retrieval" paradigm can inspire future designs for personalized RAG systems.

## Rating
*   Novelty: 9/10 — First benchmark to combine real user history with four-dimensional diagnostic feedback for personalized search LLMs.
*   Experimental Thoroughness: 9/10 — Inclusion of 6 models, multiple configuration ablations, meta-evaluations, Oracle experiments, and noise robustness analysis.
*   Writing Quality: 8/10 — Clear structure, though density of mathematical symbols and tables is high in some sections.
*   Value: 8/10 — Fills an important gap in the evaluation of personalized search LLMs; the diagnostic feedback design is of practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)
- [\[AAAI 2026\] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](../../AAAI2026/llm_evaluation/lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/llm_evaluation/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](../../ACL2026/llm_evaluation/engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)

</div>

<!-- RELATED:END -->
