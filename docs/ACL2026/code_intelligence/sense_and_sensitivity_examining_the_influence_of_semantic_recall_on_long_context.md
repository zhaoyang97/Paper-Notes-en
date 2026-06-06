---
title: >-
  [Paper Note] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding
description: >-
  [ACL 2026][Code Intelligence][Semantic Recall] This paper proposes to distinguish between lexical recall (retrieving code verbatim) and semantic recall (understanding the runtime semantics of code). It discovers that whi…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Semantic Recall"
  - "Lexical Recall"
  - "Long Context"
  - "Code Understanding"
  - "Lost-in-the-Middle"
date: 2026-05-08
content_hash: 078a619d3b34c987
---

# Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding

**Conference**: ACL 2026  
**arXiv**: [2505.13353](https://arxiv.org/abs/2505.13353)  
**Code**: [GitHub](https://github.com/adamstorek/long-context-code-understanding)  
**Area**: Long-Context Understanding / Code Understanding  
**Keywords**: Semantic Recall, Lexical Recall, Long Context, Code Understanding, Lost-in-the-Middle

## TL;DR

This paper proposes to distinguish between lexical recall (retrieving code verbatim) and semantic recall (understanding the runtime semantics of code). It discovers that while SOTA LLMs exhibit near-perfect lexical recall in long contexts, their semantic recall degrades severely. The introduced SemTrace benchmark reveals that existing evaluations significantly underestimate the extent of semantic understanding failures.

## Background & Motivation

**Background**: LLMs are increasingly deployed for tasks involving the understanding of large codebases. Recent long-context techniques (e.g., FlashAttention, RoPE) enable models to process inputs with millions of tokens. However, the fundamental question remains: when a model solves a code understanding task, is it processing the specific code in the context, or is it merely applying patterns memorized during pre-training?

**Limitations of Prior Work**: Existing Needle-in-a-Haystack (NIAH) benchmarks only measure lexical recall. Meanwhile, code understanding tasks (such as output prediction) often have low semantic recall sensitivity, allowing models to achieve correct answers via pattern-matching shortcuts, thereby masking true semantic understanding failures. For instance, in the CRUXEval benchmark, even after deleting 50% of the code lines, model accuracy only drops by 44-60%, which is far less than the exponential decay seen in a Python interpreter.

**Key Challenge**: Models can perfectly locate and reproduce code verbatim (lexical recall) yet fail to understand the runtime semantics of that code (semantic recall). These two capabilities are decoupled, but existing benchmarks fail to differentiate them effectively.

**Goal**: To systematically investigate the discrepancy between lexical and semantic recall in long contexts, quantify the underestimation of semantic recall failures in existing benchmarks, and provide more sensitive evaluation tools.

**Key Insight**: Utilize positional changes of code within a long context as a probe to systematically measure the degradation patterns of both recall capabilities.

**Core Idea**: Propose the concept of "Semantic Recall Sensitivity" to measure whether a task truly requires understanding code semantics. Design the SemTrace task to isolate semantic recall through unpredictable operations, eliminating pattern-matching shortcuts.

## Method

### Overall Architecture

The study divides code understanding into lexical recall ($R^L$) and semantic recall ($R^S$) → Proposes Semantic Recall Sensitivity metrics and counterfactual measurement methods → Designs the SemTrace high-sensitivity task → Systematically evaluates positional effects across 10 SOTA LLMs.

### Key Designs

1.  **Semantic Recall Sensitivity**:
    - **Function**: Quantifies the degree to which a code understanding task depends on semantic understanding.
    - **Mechanism**: Conducts counterfactual measurements by systematically deleting code lines and observing the performance degradation curve. If a model relies heavily on semantic recall, performance should drop sharply (similar to a Python interpreter); if it relies on pattern matching, degradation is gradual.
    - **Design Motivation**: Existing benchmarks allow models to "guess" outputs by identifying common algorithmic patterns (e.g., sorting, string manipulation) rather than truly understanding the code. A metric is needed to distinguish these scenarios.

2.  **SemTrace Task**:
    - **Function**: Provides an output prediction benchmark with high semantic recall sensitivity.
    - **Mechanism**: Generates Python functions containing simple but unpredictable arithmetic operations. Each assignment statement independently modifies different elements in a list ($x + y$, where $y$ is sampled uniformly from $[-100, 99]$), with the order of assignments randomized. The probability of guessing the full output is extremely low (at most $(1/200)^4$), necessitating accurate semantic recall of all assignment lines.
    - **Design Motivation**: Uses simple two-digit arithmetic to minimize reasoning confounders while preventing pattern matching. It supports partial match analysis to distinguish between progressive semantic recall failure and total collapse.

3.  **Position Control Experimental Design**:
    - **Function**: Isolates the differential impact of positional effects on lexical vs. semantic recall.
    - **Mechanism**: Embeds target code within a context of irrelevant distractor code (20-80 functions, approx. 4k-16k tokens). The position of the target code is systematically varied across 11 equidistant locations to test lexical recall (function-level retrieval) and semantic recall (input/output prediction) separately.
    - **Design Motivation**: Leverages positional variation as a diagnostic lens to detect how models integrate information, rather than treating the positional effect as an end in itself.

### Loss & Training

This work is an evaluation study and does not involve model training. Evaluation uses zero-shot exact match accuracy with greedy decoding for reproducibility. It employs query-aware contextualization, placing the query both before and after the code to ensure the decoder model attends to the query while processing the code.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Median Accuracy Drop | Description |
| :--- | :--- | :--- | :--- |
| Lexical Recall | Function Retrieval | 2.39% | Near-perfect, position-independent |
| CRUXEval-O | Output Prediction | 53.36% | Moderate positional degradation |
| SemTrace | Output Prediction | 92.73% | Severe positional degradation |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| CRUXEval (50% lines deleted) | Only 44-60% accuracy loss | Proves low semantic recall sensitivity |
| Python Interpreter (20% lines deleted) | Near 0% accuracy | Reference baseline for exponential decay |
| GPT-4o (2-digit SemTrace) | 100% accuracy | Memorized simple arithmetic |
| GPT-4o (4+ digit SemTrace) | 31-43% accuracy drop | Semantic recall fragility exposed beyond memory range |

### Key Findings
- SOTA models achieve near-perfect, position-independent lexical recall (>95%), but semantic recall degrades severely when code is located in the middle of the context.
- The low semantic recall sensitivity of CRUXEval masks true semantic understanding failures; pattern matching compensates for positional degradation.
- The "perfect" performance of GPT-4o on SemTrace stems from memorizing two-digit arithmetic; once extended to higher digits, it exhibits the same position-dependent degradation.
- Findings generalize across languages (Python/JS/PHP), ruling out language-specific artifacts.

## Highlights & Insights
- The decoupling of lexical and semantic recall is a profound insight: models can "see" the code but cannot "understand" it, which is particularly dangerous for code security auditing scenarios.
- The counterfactual measurement method is novel and intuitive: quantifying task dependence on specific code by progressively removing information.
- The "anomaly" analysis of GPT-4o elegantly reveals that high performance might stem from superior memorization rather than superior understanding.
- Generalizing these findings to fields like legal/policy analysis enhances the impact of the work.

## Limitations & Future Work
- Distractor code uses semantically irrelevant functions; semantically relevant noise was not tested (which might cause more severe degradation).
- Context length was limited to approximately 16k tokens; extreme scenarios with millions of tokens were not explored.
- SemTrace uses simple arithmetic, which might not capture semantic understanding challenges in more complex algorithmic contexts.
- With 800 samples per task, larger-scale evaluations might reveal more fine-grained failure modes.

## Related Work & Insights
- **vs NIAH (Needle-in-a-Haystack)**: NIAH only tests lexical recall; this work proves that successful lexical recall does not equate to successful semantic understanding.
- **vs CRUXEval**: CRUXEval lacks sufficient sensitivity as a code reasoning benchmark, allowing pattern matching to bypass true semantic understanding.
- **vs Lost-in-the-Middle (Liu et al., 2024b)**: Previous work identified information loss in the middle for natural language; this work extends it to code understanding for the first time and distinguishes between lexical and semantic mechanisms.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The framework distinguishing lexical/semantic recall and the concept of semantic recall sensitivity are entirely new contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 10 models, 3 languages, multiple context lengths, and detailed ablation/counterfactual analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logically rigorous, progressive concept development, and clear, powerful visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Provides fundamental guidance for long-context code understanding evaluation, revealing systematic blind spots in existing frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Limits of Long-Context Reasoning in Automated Bug Fixing](../../ICLR2026/code_intelligence/the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)
- [\[ACL 2026\] CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](cubridge_an_llm-based_framework_for_understanding_and_reconstructing_high-perfor.md)

</div>

<!-- RELATED:END -->
