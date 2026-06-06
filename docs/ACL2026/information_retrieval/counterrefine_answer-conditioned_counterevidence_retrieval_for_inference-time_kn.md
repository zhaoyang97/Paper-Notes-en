---
title: >-
  [Paper Note] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering
description: >-
  [ACL 2026][Information Retrieval & RAG][Inference-time repair] This paper proposes CounterRefine, a lightweight inference-time repair layer: it first generates a preliminary answer using standard RAG…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Inference-time repair"
  - "counterevidence retrieval"
  - "answer conditioning"
  - "factual QA"
  - "RAG enhancement"
date: 2026-05-08
content_hash: 26717a369ad4ad3b
---

# CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering

**Conference**: ACL 2026  
**arXiv**: [2603.16091](https://arxiv.org/abs/2603.16091)  
**Code**: None  
**Area**: Information Retrieval / Question Answering  
**Keywords**: Inference-time repair, counterevidence retrieval, answer conditioning, factual QA, RAG enhancement

## TL;DR
This paper proposes CounterRefine, a lightweight inference-time repair layer: it first generates a preliminary answer using standard RAG, then collects supporting or refuting evidence through answer-conditioned counterevidence retrieval, and finally corrects erroneous answers via a constrained KEEP/REVISE decision and deterministic verification. It improves the accuracy of GPT-5 on SimpleQA from 67.3% to 73.1%.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has become the standard method for knowledge-intensive NLP by grounding language model generation on external evidence. Variants such as multi-round retrieval and query rewriting have further improved retrieval quality.

**Limitations of Prior Work**: Many factual errors are not "access failures" but "commitment failures"—the system retrieves relevant evidence but remains locked onto an incorrect answer. In short-answer factual QA, these errors are inexcusable: incorrect years, neighboring entities, or "almost correct" titles are treated as completely wrong. The first-round retriever is optimized for topical relevance rather than the discriminability of candidate answers.

**Key Challenge**: Once a preliminary answer is generated, the most useful next query is often not the original question, but a question conditioned on that candidate answer. If the preliminary year is wrong, including that year in the query can locate evidence snippets that directly refute it.

**Goal**: To design a simple inference-time repair layer that can be layered onto existing retrieval pipelines to correct factual errors via answer-conditioned counterevidence retrieval.

**Key Insight**: Shift the role of retrieval from "collecting more context" to "testing a tentative answer." Instead of undirected expansion of the search, the preliminary answer is used to guide a targeted second retrieval.

**Core Idea**: Generate a preliminary answer first, then perform answer-conditioned retrieval for counterevidence, and finally decide whether to correct the answer through a constrained KEEP/REVISE gate and deterministic verification.

## Method

### Overall Architecture
A three-stage pipeline: Stage 1 (Baseline Drafting) → Stage 2 (Answer-Conditioned Counterevidence Retrieval) → Stage 3 (Constrained Refinement + Deterministic Verification). The input is a factual question, and the output is a repaired short answer.

### Key Designs

1.  **Answer-Conditioned Counterevidence Retrieval**:
    - **Function**: Constructs new queries based on the preliminary answer to collect evidence that supports or refutes that answer.
    - **Mechanism**: Based on the question type $t(q)$, a query set is constructed as $Q(q, a_0) = \{q, q \| a_0\} \cup \mathbb{I}[t(q) \in \mathcal{T}]\{a_0\}$, where $\mathcal{T} = \{\text{who, where, when, year, number}\}$. For each query, $k_r=5$ evidence items are retrieved, which are merged and de-duplicated with baseline evidence to form $R_1$. Key intuition: The second round of retrieval does not ask "which documents are relevant to this question," but "what evidence most directly supports or refutes this candidate answer."
    - **Design Motivation**: The original query optimizes for topical relevance, while the answer-conditioned query optimizes for candidate discriminability. When the preliminary answer is an incorrect neighboring entity or year, adding it to the query often retrieves snippets that precisely negate it.

2.  **Constrained Refinement Gating**:
    - **Function**: Decides whether to keep or modify the preliminary answer based on expanded evidence.
    - **Mechanism**: The refiner receives the question, baseline answer, and merged evidence set, and must output three fields: DECISION (KEEP/REVISE), ANSWER (short answer), and EVIDENCE (evidence snippet or NONE). The prompt instructs to REVISE only when additional evidence strongly supports a different answer. The output format is highly constrained, not open-ended rewriting.
    - **Design Motivation**: Restricting the refinement range to a binary decision (keep/modify) rather than letting the model re-solve the problem from scratch significantly reduces the risk of introducing new errors.

3.  **Deterministic Verification & Normalization**:
    - **Function**: Prevents unsupported, type-mismatched, or improperly formatted modifications.
    - **Mechanism**: Proposed modifications are rejected if: the answer is empty or identical to the preliminary answer; the answer for a yes/no question is incorrect; the answer for an entity-type question is too long or contains descriptive phrases; the answer for a time/number question lacks clear markers; there is no supporting evidence snippet; or the lexical overlap between the modified answer and the evidence snippet is too weak. Validated modifications undergo question-type-specific normalization (e.g., extracting 4-digit years, compressing numerical ranges).
    - **Design Motivation**: Model decisions cannot be fully trusted; deterministic rule verification provides a hard quality guarantee. This ensures modifications are only accepted with sufficient evidence support, while KEEP decisions retain the original answer unaffected by verification.

### Loss & Training
Training-free. CounterRefine is a pure inference-time pipeline using off-the-shelf LLMs (Claude Sonnet 3.5 or GPT-5) and Web search APIs.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Claude Base-RAG | Claude +CounterRefine | GPT-5 Base-RAG | GPT-5 +CounterRefine |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SimpleQA (4326) | Correct↑ | 63.7 | 67.7 (+4.0) | 67.3 | **73.1 (+5.8)** |
| SimpleQA (4326) | F1↑ | 64.1 | 68.1 (+4.0) | 58.6 | **72.1 (+13.5)** |
| HotpotQA (300) | EM↑ | 70.0 | 74.0 (+4.0) | 68.0 | 71.0 (+3.0) |

### Intervention Analysis (Claude SimpleQA Full)

| Metric | Value |
| :--- | :--- |
| Modification Rate | 5.6% |
| Beneficial Modifications | 180 |
| Harmful Modifications | 8 |
| Benefit-to-Harm Ratio | 22.5:1 |

### Key Findings
- CounterRefine consistently improves exact match metrics across backbone models, datasets, and evaluation scales.
- Interventions are highly precise: only 5.6% of samples are modified, with a benefit-to-harm ratio of 22.5:1, indicating that deterministic verification effectively filters out erroneous modifications.
- The F1 improvement on GPT-5 reaches 13.5 points, far exceeding the 5.8-point gain in accuracy, suggesting significant improvements in lexical precision of the repaired answers.
- Success patterns mainly involve entity confusion, date errors, and numerical imprecision; failure patterns involve relation confusion and event mismatch.

## Highlights & Insights
- **From "Collecting Evidence" to "Testing Hypotheses"**: The role of retrieval is shifted from passive context collection to active hypothesis testing. This conceptual shift is more significant than any technical detail—once a candidate answer exists, the most valuable retrieval is targeted toward that answer.
- **Deterministic Verification as an Indispensable Safety Net**: The 22.5:1 benefit-to-harm ratio proves the value of hard-rule verification. Pure model-based refinement is likely to introduce more errors; deterministic verification limits modifications to high-confidence cases.
- **Minimalist Design Philosophy**: The entire method adds only one additional retrieval, one model call, and rule verification, without modifying model parameters or the retrieval pipeline. This "thin repair layer" design allows it to be layered onto any RAG system.

## Limitations & Future Work
- Only applicable to short-answer factual QA; repairing long-form text generation requires different mechanisms.
- Failure patterns (relation confusion, event mismatch) are difficult to resolve through simple answer-conditioned retrieval.
- Deterministic verification rules are manually designed and may not cover new question types.
- Multi-round iterative refinement has not been explored (currently only one round), which might miss errors requiring multi-step reasoning.

## Related Work & Insights
- **vs Chain-of-Verification (CoVe)**: CoVe generates verification questions and then answers them, but with higher computational costs. CounterRefine uses only one extra retrieval and one model call.
- **vs CRITIC**: CRITIC uses interactive tool usage for verification, making it more general but more complex. CounterRefine focuses on short-answer repair, being simpler and more efficient.
- **vs ROME/MEMIT**: Model editing modifies factual associations in parameters. CounterRefine is a complementary inference-time repair that does not change model parameters.

## Rating
- Novelty: ⭐⭐⭐ Answer-conditioned retrieval is an intuitive yet effective idea; deterministic verification is key.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full SimpleQA official evaluation + cross-model/dataset + intervention analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear writing; the logic chain from motivation to method to analysis is complete.
- Value: ⭐⭐⭐⭐ Highly practical; can be directly layered onto existing RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[AAAI 2026\] Towards Inference-Time Scaling for Continuous Space Reasoning](../../AAAI2026/information_retrieval/towards_inference-time_scaling_for_continuous_space_reasoning.md)
- [\[ICML 2026\] REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](../../ICML2026/information_retrieval/real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)
- [\[ACL 2026\] FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](finrag-12b_a_production-validated_recipe_for_grounded_question_answering_in_bank.md)

</div>

<!-- RELATED:END -->
