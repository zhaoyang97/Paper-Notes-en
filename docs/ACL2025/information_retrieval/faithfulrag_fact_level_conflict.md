---
title: >-
  [Paper Note] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This study reveals that existing context-faithful RAG methods achieve faithfulness by forcibly suppressing parametric knowledge, which increases the risk of misunderstanding the context (unfaithful errors decrease by 6.65% while mismatch errors increase by 6.42%). We propose FaithfulRAG, which resolves knowledge conflicts through fact-level conflict detection (self-fact mining) and conflict reasoning (self-think module)…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "knowledge conflict"
  - "fact-level conflict modeling"
  - "context faithfulness"
  - "self-reasoning"
date: 2026-05-08
content_hash: d7dec3665d022373
---

# FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2506.08938](https://arxiv.org/abs/2506.08938)  
**Code**: [https://github.com/DeepLearnXMU/Faithful-RAG](https://github.com/DeepLearnXMU/Faithful-RAG)  
**Area**: Information Retrieval  
**Keywords**: RAG, knowledge conflict, fact-level conflict modeling, context faithfulness, self-reasoning  

## TL;DR

This study reveals that existing context-faithful RAG methods achieve faithfulness by forcibly suppressing parametric knowledge, which increases the risk of misunderstanding the context (unfaithful errors decrease by 6.65% while mismatch errors increase by 6.42%). We propose FaithfulRAG, which resolves knowledge conflicts through fact-level conflict detection (self-fact mining) and conflict reasoning (self-think module), outperforming the strongest baselines by 8-9 percentage points on FaithEval/SQuAD/MuSiQue/RealtimeQA.

## Background & Motivation

**Background**: RAG systems are prone to generating unfaithful outputs when retrieved contexts conflict with the model's parametric knowledge. Existing methods rely on prompting strategies or decoding interventions to force the model to prioritize context.

**Limitations of Prior Work**: Forcibly suppressing parametric knowledge is dual-edged—while it reduces errors caused by ignoring the context, it simultaneously increases errors stemming from context misunderstanding. This occurs because the model loses the ability to utilize its parametric knowledge to assist in understanding the context.

**Key Challenge**: Faithfulness and correctness have been framed as conflicting goals by existing methods, whereas the ideal state is to achieve both.

**Goal**: To achieve context faithfulness without sacrificing accuracy.

**Key Insight**: Rather than forcibly suppressing parametric knowledge, it is better to explicitly identify the points of conflict between parametric knowledge and context, allowing the model to first "understand where the conflict lies, then decide who to believe."

**Core Idea**: Instead of suppressing parametric knowledge, mine the conflicting facts and guide the model through reasoning before generation.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Self-Fact Mining**—LLMs generate a pre-answer based on parametric knowledge and decompose it into claims; (2) **Conflict Detection**—align self-facts with the retrieved context to pinpoint conflicts; (3) **Self-Think Generation**—organize conflicting information into reasoning prompts to guide the LLM to resolve conflicts before generation.

### Key Designs

1. **Self-Fact Mining**:

    - **Function**: Externalize the LLM's parametric understanding of the question.
    - **Mechanism**: Two steps—(a) Self-Knowledge Extraction: Let the LLM answer the question first without looking at the context to extract its parametric pre-answer; (b) Self-Context Generation: Decompose the pre-answer into fine-grained factual claims.
    - **Design Motivation**: To identify conflicts with retrieved context, the system must first know what the model itself "believes to be true."

2. **Fact-Level Conflict Detection**:

    - **Function**: Pinpoint exact contradictions between parametric knowledge and retrieved contexts.
    - **Mechanism**: Align self-facts with the context one-by-one, using an NLI model or an LLM to classify each pair as supported, contradicting, or neutral.
    - **Design Motivation**: Coarse-grained assessment (whether the entire document is contradictory) is insufficient, as a document may be partially supportive and partially contradictory.

3. **Self-Think Module**:

    - **Function**: Guide the LLM to reason about conflict resolution before generating the final answer.
    - **Mechanism**: Construct reasoning prompts that include the conflicting fact pairs and instructions such as "contextual information should override your parametric knowledge." The model is prompted to first generate a Chain-of-Thought (CoT) explanation of why the context is correct, followed by the final answer.
    - **Design Motivation**: Outperforms direct suppression of parametric knowledge—the model understands *why* it should trust the context, rather than being forced to do so.

### Loss & Training
- **No additional training required**—a training-free method.
- Confirmed across multiple backbone models: Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B.

## Key Experimental Results

### Main Results

| Dataset | Strongest Baseline | FaithfulRAG (Llama3.1) | Gain |
|--------|---------|----------------------|------|
| FaithEval | KRE 73.2% | **81.7%** (Mistral) | +8.5% |
| SQuAD | ChatQA-2.0 77.0% | **86.3%** | +9.3% |
| MuSiQue | - | **79.9%** | Best |
| RealtimeQA | - | **84.1%** | Best |
| SQuAD-golden (non-conflict) | COIECD 95.1% | **96.6%** | +1.5% |

### Ablation Study

| Configuration | Average Accuracy Change | Description |
|------|-------------|------|
| w/o Self-Knowledge Extraction | -1.1% | Cannot identify parametric knowledge |
| w/o Self-Context Generation | -1.9% | Original knowledge claims unavailable |
| w/o Self-Think | -3.2% | Lack of conflict reasoning |
| w/o CoT in Self-Think | -1.5% | CoT-assisted reasoning |
| Remove All | Degrades to standard RAG | Validates component synergy |

### Key Findings
- FaithfulRAG achieves optimal performance in both conflict and non-conflict scenarios—unlike prior methods that degrade in non-conflict settings.
- Stable performance across backbones—the variation in SQuAD accuracy across Llama, Qwen, and Mistral is less than 1%.
- Self-Think is the most critical component (-3.2%), but it cannot function without Self-Fact providing conflict information.
- "Diagnostic reconciliation" is more effective than "forcible suppression"—understanding conflicts is superior to merely ignoring parametric knowledge.

## Highlights & Insights

- **Reveals the "side effects" of existing faithful RAG methods**: while suppressing parametric knowledge reduces unfaithful outputs, it increases context misunderstanding. This is a crucial, previously overlooked finding.
- **The "know yourself and know your enemy" strategy is intuitive**: excavating parametric knowledge first ("know yourself"), comparing it with the context to detect conflicts ("know your enemy"), and finally reasoning to resolve the conflict ("decision making").
- **Fact-level granularity is precise**: it is common for a document to contain both partially supportive and partially contradictory statements.
- The method is training-free, generalizable across models, and highly practical.

## Limitations & Future Work

- The three-stage pipeline increases inference overhead, requiring multiple LLM calls.
- Conflict detection depends heavily on the classification capability of the NLI model or LLM.
- Evaluated only on QA tasks; other RAG scenarios remain untested.

## Related Work & Insights

- **vs ATTR/KRE**: These methods enforce context-priority, whereas FaithfulRAG enables the model to "understand before choosing."
- **vs CAD (Contrastive Decoding)**: CAD modifies probabilities at the decoding level, without solving core comprehension-level issues.
- **vs GainRAG**: GainRAG selects paragraphs with performance gains, whereas FaithfulRAG resolves conflicts between selected paragraphs and parametric knowledge—making them complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ Fact-level conflict modeling and self-think reasoning represent significant innovations. It exposes the "side effect" of existing methods (suppressing parametric knowledge $\propto$ misunderstanding context).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four datasets and three models, with separate assessments for conflict and non-conflict scenarios, detailed ablation studies, and error type analyses.
- Writing Quality: ⭐⭐⭐⭐ Data-driven problem analysis (the error distribution analysis in Figure 2 is highly convincing). The logical flow of the method (knowing oneself and the context) is very natural.
- Value: ⭐⭐⭐⭐⭐ A significant breakthrough for RAG faithfulness. The core insight that "diagnostic reconciliation" outperforms "forcible suppression" can be directly applied to real-world RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)

</div>

<!-- RELATED:END -->
