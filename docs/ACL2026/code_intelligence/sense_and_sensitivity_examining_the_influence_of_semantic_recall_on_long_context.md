---
title: >-
  [Paper Note] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding
description: >-
  [ACL 2026][Semantic Recall] This paper distinguishes between lexical recall (verbatim code retrieval) and semantic recall (understanding runtime code semantics), demonstrating that frontier LLMs achieve near-perfect lexical recall yet exhibit severe semantic recall degradation in long contexts. The paper introduces the SemTrace benchmark, revealing that existing evaluations substantially underestimate the extent of semantic understanding failures.
tags:
  - ACL 2026
  - Semantic Recall
  - Lexical Recall
  - Long Context
  - Code Understanding
  - Lost-in-the-Middle
date: 2026-05-08
content_hash: ec10385910288a05
---

# Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding

**Conference**: ACL 2026
**arXiv**: [2505.13353](https://arxiv.org/abs/2505.13353)
**Code**: [GitHub](https://github.com/adamstorek/long-context-code-understanding)
**Area**: Long-Context Understanding / Code Understanding
**Keywords**: Semantic Recall, Lexical Recall, Long Context, Code Understanding, Lost-in-the-Middle

## TL;DR

This paper distinguishes between lexical recall (verbatim code retrieval) and semantic recall (understanding runtime code semantics), demonstrating that frontier LLMs achieve near-perfect lexical recall yet exhibit severe semantic recall degradation in long contexts. The paper introduces the SemTrace benchmark, revealing that existing evaluations substantially underestimate the extent of semantic understanding failures.

## Background & Motivation

**Background**: LLMs are increasingly deployed for tasks requiring comprehension of large codebases, and recent long-context techniques (FlashAttention, RoPE, etc.) enable models to process inputs of millions of tokens. However, a fundamental question remains unanswered: when models solve code understanding tasks, are they processing the concrete code in context or applying patterns memorized during pretraining?

**Limitations of Prior Work**: Existing Needle-in-a-Haystack (NIAH) benchmarks measure only lexical recall, while code understanding tasks such as output prediction exhibit low semantic recall sensitivity, allowing models to obtain correct answers via pattern-matching shortcuts and thereby concealing genuine semantic understanding failures. For instance, on the CRUXEval benchmark, removing 50% of code lines reduces model accuracy by only 44–60%, far less than the exponential degradation observed in a Python interpreter.

**Key Challenge**: Models can perfectly locate and verbatim reproduce code (lexical recall) yet fail to understand its runtime semantics (semantic recall). These two capabilities are decoupled, but existing benchmarks fail to distinguish between them effectively.

**Goal**: To systematically investigate the differential behavior of lexical and semantic recall in long contexts, quantify the degree to which existing benchmarks underestimate semantic recall failures, and provide more sensitive evaluation tools.

**Key Insight**: The positional variation of code within long contexts is used as a diagnostic probe to systematically measure the degradation patterns of both recall capabilities as a function of position.

**Core Idea**: The paper introduces the concept of *semantic recall sensitivity* to measure whether a task genuinely requires understanding of code semantics, and designs the SemTrace task to isolate semantic recall via unpredictable arithmetic operations, eliminating pattern-matching shortcuts.

## Method

### Overall Architecture

Code understanding is decomposed into lexical recall ($R^L$) and semantic recall ($R^S$) → A semantic recall sensitivity metric and counterfactual measurement methodology are proposed → The SemTrace high-sensitivity task is designed → Systematic evaluation of positional effects is conducted across 10 state-of-the-art LLMs.

### Key Designs

1. **Semantic Recall Sensitivity**:

    - Function: Quantifies the degree to which a code understanding task depends on semantic comprehension.
    - Mechanism: Measured via counterfactual analysis — code lines are systematically removed one by one, and the resulting performance degradation curve is observed. If a model heavily relies on semantic recall, removing critical lines should cause sharp performance drops (analogous to a Python interpreter); if it relies on pattern matching, degradation is gradual.
    - Design Motivation: Existing benchmarks allow models to "guess" outputs by recognizing common algorithmic patterns (e.g., sorting, string operations) rather than genuinely understanding the code. A metric is needed to distinguish these two cases.

2. **SemTrace Task**:

    - Function: Provides an output prediction benchmark with high semantic recall sensitivity.
    - Mechanism: Python functions are generated containing simple but unpredictable arithmetic operations, where each assignment statement independently modifies a different element of a list ($x + y$, with $y$ uniformly sampled from $[-100, 99]$), and assignment order is randomized. The probability of guessing the full output is extremely low (at most $(1/200)^4$), requiring accurate semantic recall of all assignment lines.
    - Design Motivation: Simple two-digit arithmetic minimizes confounding factors from reasoning difficulty while preventing pattern matching. Partial-match analysis is supported to distinguish between progressive semantic recall degradation and complete collapse.

3. **Position-Controlled Experimental Design**:

    - Function: Isolates the differential effects of position on lexical versus semantic recall.
    - Mechanism: Target code is embedded within irrelevant distractor code contexts (20–80 distractor functions, approximately 4k–16k tokens), and the position of the target code is systematically varied across 11 equidistant positions. Lexical recall (function-level retrieval) and semantic recall (input/output prediction) are evaluated separately.
    - Design Motivation: Positional variation serves as a diagnostic lens to probe how models integrate information, rather than treating positional effects as an end goal.

### Loss & Training

This paper is an evaluation study and does not involve model training. Evaluation uses zero-shot exact-match accuracy with greedy decoding to ensure reproducibility. Query-aware contextualization is employed, placing the query both before and after the code so that decoder models can attend to the query while processing the code.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Median Accuracy Drop | Notes |
|-----------|--------|----------------------|-------|
| Lexical Recall | Function Retrieval | 2.39% | Near-perfect, position-invariant |
| CRUXEval-O | Output Prediction | 53.36% | Moderate positional degradation |
| SemTrace | Output Prediction | 92.73% | Severe positional degradation |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| CRUXEval with 50% lines removed | Only 44–60% accuracy loss | Demonstrates low semantic recall sensitivity |
| Python interpreter with 20% lines removed | Near 0% accuracy | Reference baseline showing exponential decay |
| GPT-4.1 on 2-digit SemTrace | 100% accuracy | Arithmetic memorized |
| GPT-4.1 on 4+ digit SemTrace | 31–43% accuracy drop | Semantic recall fragility exposed beyond memorization range |

### Key Findings
- Frontier models achieve near-perfect, position-invariant lexical recall (>95%), but semantic recall degrades severely when code is positioned in the middle of the context.
- The low semantic recall sensitivity of CRUXEval conceals genuine semantic understanding failures — pattern matching compensates for positional degradation.
- GPT-4.1's perfect performance on SemTrace stems from memorized two-digit arithmetic; extending to higher digit counts likewise exposes position-dependent degradation.
- Findings generalize across languages (Python/JS/PHP), ruling out language-specific artifacts.

## Highlights & Insights
- The lexical–semantic recall decoupling is a profound insight: models can "see" code without "understanding" it, which is particularly dangerous in code security auditing scenarios.
- The counterfactual measurement methodology is novel and intuitively clear: progressively removing information quantifies the extent to which a task depends on concrete code.
- The analysis of GPT-4.1's "anomalous" behavior elegantly reveals that even high performance may reflect superior memorization rather than superior understanding.
- Generalizing the findings to domains such as legal and policy analysis broadens the impact of the work.

## Limitations & Future Work
- Distractor code consists of semantically unrelated functions; semantically related distractors (which may induce more severe degradation) are not evaluated.
- Context length is limited to approximately 16k tokens; extreme scenarios at the million-token scale are not explored.
- SemTrace employs simple arithmetic operations and may not capture semantic understanding challenges in more complex algorithmic contexts.
- Each task uses 800 samples; larger-scale evaluation may reveal more fine-grained failure patterns.

## Related Work & Insights
- **vs. NIAH (Needle-in-a-Haystack)**: NIAH tests only lexical recall; this paper demonstrates that lexical recall success does not imply semantic understanding success.
- **vs. CRUXEval**: CRUXEval exhibits insufficient sensitivity as a code reasoning benchmark, permitting pattern matching to bypass genuine semantic understanding.
- **vs. Lost-in-the-Middle (Liu et al., 2024b)**: Prior work identified the loss of middle-positioned information in natural language; this paper is the first to extend this finding to code understanding while distinguishing between lexical and semantic mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the lexical–semantic recall distinction framework and the semantic recall sensitivity concept are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models, 3 languages, multiple context lengths, comprehensive ablation and counterfactual analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logic, concepts introduced in a well-structured progression, clear and informative figures.
- Value: ⭐⭐⭐⭐⭐ Provides fundamental guidance for evaluating long-context code understanding and exposes systematic blind spots in existing evaluation frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Limits of Long-Context Reasoning in Automated Bug Fixing](../../ICLR2026/code_intelligence/the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[AAAI 2026\] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning](../../AAAI2026/code_intelligence/towards_better_code_understanding_in_decoder-only_large_language_models_via_hie.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)

</div>

<!-- RELATED:END -->
