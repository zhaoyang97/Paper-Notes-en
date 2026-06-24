---
title: >-
  [Paper Note] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding
description: >-
  [ACL 2026][Code Intelligence][Semantic Recall] This paper proposes a distinction between lexical recall (verbatim retrieval of code) and semantic recall (understanding code execution semantics). It finds that frontier LLMs achieve near-perfect lexical recall in long contexts but suffer from severe degradation in semantic recall. The introduced SemTrace benchmark reveals that existing evaluations significantly underestimate the extent of semantic understanding failures.
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Semantic Recall"
  - "Lexical Recall"
  - "Long Context"
  - "Code Understanding"
  - "Lost-in-the-Middle"
date: 2026-05-08
content_hash: 5c9b9bacbe367da5
---

# Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding

**Conference**: ACL 2026  
**arXiv**: [2505.13353](https://arxiv.org/abs/2505.13353)  
**Code**: [GitHub](https://github.com/adamstorek/long-context-code-understanding)  
**Area**: Long Context Understanding / Code Understanding  
**Keywords**: Semantic Recall, Lexical Recall, Long Context, Code Understanding, Lost-in-the-Middle

## TL;DR

This paper proposes a distinction between lexical recall (verbatim retrieval of code) and semantic recall (understanding code execution semantics). It finds that frontier LLMs achieve near-perfect lexical recall in long contexts but suffer from severe degradation in semantic recall. The introduced SemTrace benchmark reveals that existing evaluations significantly underestimate the extent of semantic understanding failures.

## Background & Motivation

**Background**: LLMs are increasingly deployed for tasks requiring the understanding of large codebases. Recent long-context techniques (FlashAttention, RoPE, etc.) enable models to process inputs of millions of tokens. However, the fundamental question remains unanswered: when models solve code understanding tasks, are they processing specific code within the context or applying patterns memorized during pre-training?

**Limitations of Prior Work**: Existing Needle-in-a-Haystack (NIAH) benchmarks only measure lexical recall. Furthermore, code understanding tasks like output prediction often have low semantic recall sensitivity, allowing models to achieve correct answers via pattern-matching shortcuts, thereby masking true failures in semantic understanding. For instance, in the CRUXEval benchmark, even after deleting 50% of the code lines, model accuracy only drops by 44-60%, which is far less than the exponential decay observed in a Python interpreter.

**Key Challenge**: Models can perfectly locate and reproduce code verbatim (lexical recall) yet fail to understand the runtime semantics of that code (semantic recall). These two capabilities are decoupled, but existing benchmarks fail to distinguish between them effectively.

**Goal**: To systematically investigate the divergent performance of lexical and semantic recall in long contexts, quantify the degree to which existing benchmarks underestimate semantic recall failures, and provide more sensitive evaluation tools.

**Key Insight**: Utilize positional variations of code within long contexts as a probe to systematically measure the degradation patterns of both recall types.

**Core Idea**: Propose the concept of "semantic recall sensitivity" to measure whether a task truly requires understanding code semantics. Design the SemTrace task to isolate semantic recall through unpredictable operations, eliminating pattern-matching shortcuts.

## Method

### Overall Architecture

The paper decomposes "code understanding" into two capabilities often conflated in existing evaluations: lexical recall $R^L$ (locating and reproducing specific code verbatim in long contexts) and semantic recall $R^S$ (understanding the actual runtime semantics of that code). Centered on this distinction, "semantic recall sensitivity" is defined via counterfactual deletion to measure task dependence on semantic understanding. Based on this, the high-sensitivity SemTrace output prediction task is designed to block pattern-matching shortcuts. Finally, target code is embedded into a distracting context and systematically shifted across 11 equidistant positions to measure the degradation curves of both recall types across 10 SOTA LLMs. Evaluations use zero-shot exact match and greedy decoding for reproducibility, with query-aware contextualization (placing the query both before and after the code) to ensure the decoder maintains focus on the query while processing the code.

### Key Designs

**1. Semantic Recall Sensitivity: Quantifying Task Dependence on Semantic Understanding**

Existing benchmarks allow models to "guess" outputs by identifying common algorithmic patterns (sorting, string manipulation, etc.) rather than actually executing code. A metric is needed to distinguish these cases. This is achieved through counterfactual measurement—deleting lines of code one by one and observing the performance degradation curve. If a model truly relies on semantic recall, accuracy should drop sharply like a Python interpreter after critical lines are removed; if it relies on pattern matching, the curve remains flat. The fact that CRUXEval only loses 44–60% accuracy when 50% of lines are deleted is clear evidence of low sensitivity.

**2. SemTrace Task: Isolating Pure Semantic Recall with Unpredictable Arithmetic**

To create a high-sensitivity benchmark, SemTrace generates a set of simple yet unpredictable Python functions: each assignment statement independently modifies different elements in a list (e.g., $x + y$, where $y$ is sampled uniformly from $[-100, 99]$), and assignment orders are randomized. This makes the probability of guessing the entire output extremely low (at most $(1/200)^4$). Models must accurately perform semantic recall for every assignment to succeed. Two-digit arithmetic is intentionally used to minimize reasoning difficulty and isolate semantic recall as the sole variable, while also supporting partial match analysis to distinguish "progressive semantic degradation" from "total collapse."

**3. Position Control Experimental Design: Position as a Probe**

To clearly observe the distinct effects of position on both recall types, target code is buried within contexts composed of 20-80 irrelevant distractor functions (approx. 4k-16k tokens). The target code's position is shifted across 11 equidistant points. Lexical recall is measured via function-level retrieval, while semantic recall is measured via input/output prediction. Position change serves here as a diagnostic lens to reveal how models integrate dispersed information, rather than being the end goal of the research.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Median Accuracy Drop | Description |
|-----------|--------|----------------------|-------------|
| Lexical Recall | Function Retrieval | 2.39% | Near-perfect, position-independent |
| CRUXEval-O | Output Prediction | 53.36% | Moderate positional degradation |
| SemTrace | Output Prediction | 92.73% | Severe positional degradation |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| CRUXEval (50% lines deleted) | Only 44-60% accuracy loss | Proves low semantic recall sensitivity |
| Python Interpreter (20% lines deleted) | Near 0% accuracy | Exponential decay reference baseline |
| GPT-4.1 (2-digit SemTrace) | 100% Accuracy | Memorization of simple arithmetic |
| GPT-4.1 (4+ digit SemTrace) | 31-43% Accuracy drop | Semantic recall fragility exposed beyond memory |

### Key Findings
- Frontier models achieve near-perfect, position-independent lexical recall (>95%), but semantic recall degrades severely when code is located in the middle of the context.
- The low semantic recall sensitivity of CRUXEval masks true failures in semantic understanding—pattern matching compensates for positional degradation.
- The perfect performance of GPT-4.1 on SemTrace stems from memorization of two-digit arithmetic; scaling to higher digits exposes the same position-dependent degradation.
- Cross-language generalization (Python/JS/PHP) was observed, ruling out language-specific artifacts.

## Highlights & Insights
- The decoupling of lexical and semantic recall is a profound insight: models can "see" code but cannot "understand" it, which is particularly dangerous for code security auditing scenarios.
- The counterfactual measurement method is novel and intuitive: quantifying task dependence on specific code by incrementally removing information.
- The "anomaly" analysis of GPT-4.1 elegantly reveals that even high performance can stem from better memorization rather than superior understanding.
- Generalizing findings to fields such as legal and policy analysis enhances the impact of the work.

## Limitations & Future Work
- Distractor code uses semantically unrelated functions; semantically related distractors (which might cause more severe degradation) were not tested.
- Context length is limited to approximately 16k tokens; extreme scenarios of millions of tokens were not explored.
- SemTrace utilizes simple arithmetic operations, which may not capture semantic understanding challenges in more complex algorithmic contexts.
- With 800 samples per task, larger-scale evaluations might reveal more fine-grained failure modes.

## Related Work & Insights
- **vs NIAH (Needle-in-a-Haystack)**: NIAH only tests lexical recall; this paper proves that successful lexical recall does not equate to successful semantic understanding.
- **vs CRUXEval**: CRUXEval lacks sufficient sensitivity as a code reasoning benchmark, allowing pattern matching to bypass true semantic understanding.
- **vs Lost-in-the-Middle (Liu et al., 2024b)**: Prior work identified information loss in the middle of natural language contexts; this work extends this to code understanding and distinguishes between lexical and semantic mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The lexical-semantic recall decoupling framework and the concept of semantic recall sensitivity are entirely new contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models, 3 languages, multiple context lengths, and exhaustive ablation/counterfactual analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically rigorous, progressive concept development, and clear, powerful visualizations.
- Value: ⭐⭐⭐⭐⭐ Offers fundamental guidance for evaluating long-context code understanding, revealing systematic blind spots in existing evaluation frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](../../ACL2025/code_intelligence/benchmarking_long-context_language_models_on_long_code_understanding.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](cubridge_an_llm-based_framework_for_understanding_and_reconstructing_high-perfor.md)
- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges](creativebench_benchmarking_and_enhancing_machine_creativity_via_self-evolving_ch.md)

</div>

<!-- RELATED:END -->
