---
title: >-
  [Paper Note] Inter-Passage Verification for Multi-evidence Multi-answer QA
description: >-
  [ACL 2025 Findings][Multi-answer QA] This paper proposes the RI²VER framework to address the multi-answer QA problem. It first generates a high-recall but noisy candidate answer set by independently reading a large number of retrieved passages, and then filters out incorrect answers through inter-passage verification (generating verification questions $\rightarrow$ collecting additional evidence $\rightarrow$ cross-passage synthesis verification)…
tags:
  - "ACL 2025 Findings"
  - "Multi-answer QA"
  - "Inter-passage verification"
  - "Retrieval-augmented generation"
  - "Evidence synthesis"
  - "Fact-checking"
date: 2026-05-08
content_hash: 9d2ed15dda84b571
---

# Inter-Passage Verification for Multi-evidence Multi-answer QA

**Conference**: ACL 2025 Findings  
**arXiv**: [2506.00425](https://arxiv.org/abs/2506.00425)  
**Code**: None  
**Area**: Others  
**Keywords**: Multi-answer QA, Inter-passage verification, Retrieval-augmented generation, Evidence synthesis, Fact-checking  

## TL;DR

This paper proposes the RI²VER framework to address the multi-answer QA problem. It first generates a high-recall but noisy candidate answer set by independently reading a large number of retrieved passages, and then filters out incorrect answers through inter-passage verification (generating verification questions $\rightarrow$ collecting additional evidence $\rightarrow$ cross-passage synthesis verification), improving average F1 by 11.17% on QAMPARI and RoMQA.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) is the dominant paradigm in current QA systems, where LLMs generate answers based on retrieved relevant passages. While this pipeline works well for single-answer questions, multi-answer questions (e.g., "Which countries had a population over 100 million in 2020?") pose a severe challenge to existing RAG systems.

**Limitations of Prior Work**: Multi-answer QA faces two unique difficulties: (1) Answers are scattered across numerous distinct passages, making it hard to recall all relevant evidence in a single retrieval step. (2) Feeding a large volume of passages simultaneously into an LLM leads to information overload and context length limitations, causing the LLM to easily miss critical information in the passages or generate hallucinated answers. Existing methods typically either sacrifice recall (retrieving less) or sacrifice precision (retrieving more but introducing significant noise).

**Key Challenge**: High recall requires retrieving more passages, but more passages introduce more noise and greater difficulty in synthesis. This trade-off between recall and precision is the core bottleneck of multi-answer QA.

**Goal**: To design a two-stage framework: the first stage aims for high recall (independently reading each passage), while the second stage pursues high precision (verifying the correctness of each candidate answer).

**Key Insight**: The authors observe that if a candidate answer is correct, supporting evidence should ideally be found in other retrieved passages. Conversely, if an answer is extracted from only a single passage and cannot be verified by other information sources, it is highly likely to be noise. This concept of "cross-passage cross-verification" can effectively filter out incorrect answers.

**Core Idea**: Solve the recall-precision trade-off in multi-answer QA using a two-stage strategy of "independent reading + inter-passage verification." The first stage maximizes recall, while the second stage filters candidates precisely through a pipeline of generating verification questions, collecting additional evidence, and performing cross-passage synthesis.

## Method

### Overall Architecture

The complete pipeline of RI²VER (Retrieval-augmented Independent Reading with Inter-passage Verification) is divided into two stages. **Stage 1 (Independent Reading)**: Retrieve the top-$K$ passages (where $K$ can be large, e.g., 100), independently prompt the LLM to extract potential answers from each passage, and then merge and deduplicate them to obtain a high-recall candidate answer set $\mathcal{A}$. **Stage 2 (Inter-Passage Verification)**: For each candidate answer $a \in \mathcal{A}$, perform a three-step verification: (1) generate a verification question regarding $a$; (2) retrieve additional evidence passages related to the verification question; (3) make a cross-passage synthesis judgement based on both the original passages and the additional evidence to decide whether to keep $a$.

### Key Designs

1. **Independent Reading Strategy**:

    - **Function**: Extract candidate answers with high recall from a large volume of retrieved passages.
    - **Mechanism**: Unlike traditional RAG, which concatenates all passages to generate answers in a single run, RI²VER extracts answers from each passage individually. Given question $q$ and passage $p_i$, the LLM extracts all potential answers $\{a_1^i, a_2^i, ...\}$ from $p_i$. The answers from all passages are merged and deduplicated to construct the candidate set $\mathcal{A} = \bigcup_i \text{Extract}(q, p_i)$.
    - **Design Motivation**: Independent reading sidesteps the context length limit and information overload. Even if each individual passage contains only one answer, high recall can be achieved by iterating through all passages. The cost is the introduction of noise (irrelevant passages or hallucinated answers from the LLM), which is precisely resolved by the verification in the second stage.

2. **Verification Question Generation**:

    - **Function**: Convert the abstract question "is the answer correct?" into more specific and verifiable sub-questions.
    - **Mechanism**: For each candidate answer $a$ and the original question $q$, the LLM is prompted to generate a specific verification question $q_v$. For instance, given the original question "Which countries have a population of over 100 million?" and candidate answer "Brazil" $\rightarrow$ verification question: "Does the population of Brazil exceed 100 million?". The verification question is more focused than the original question, making it easier to find explicit corroborating or contradicting evidence through retrieval.
    - **Design Motivation**: Directly asking "Is Brazil the correct answer?" is too abstract for the LLM. Decomposing it into specific factual questions allows the retrieval system to accurately acquire relevant evidence.

3. **Inter-Passage Synthesis Verification**:

    - **Function**: Contextually judge the correctness of candidate answers based on both the original passages and newly retrieved evidence.
    - **Mechanism**: The verification step synthesizes three types of information as input to the LLM: (1) the original passage $p_i$ from which the answer was extracted; (2) target evidence passages $\{e_1, e_2, ...\}$ freshly retrieved (Top-$K$) for the verification question $q_v$; (3) the original question $q$ and the candidate answer $a$. Based on all this information, the LLM outputs a Yes/No judgment. Only answers passing this verification are retained in the final answer set.
    - **Design Motivation**: The core concept is "multi-source evidence cross-verification." If an answer is factually correct, passages from different sources should support each other. This kind of cross-passage synthesis is impossible for conventional single-passage RAG systems.

### Loss & Training

RI²VER is a training-free pipeline framework where all steps are implemented via LLM prompting. It can be applied plug-and-play to LLMs of various scales.

## Key Experimental Results

### Main Results

| Method | QAMPARI (F1) | RoMQA (F1) | Average F1 | Notes |
|------|-------------|-----------|---------|------|
| RAG (Top-10) Baseline | 34.2 | 28.7 | 31.5 | Standard RAG |
| RAG (Top-50) Baseline | 37.8 | 31.4 | 34.6 | More passages but increased noise |
| Self-RAG | 38.1 | 33.2 | 35.7 | Adaptive retrieval |
| RI²VER (Llama-7B) | 42.6 | 38.5 | 40.6 | Significant improvement even with small models |
| RI²VER (GPT-3.5) | 46.3 | 42.1 | 44.2 | Medium-sized model |
| RI²VER (GPT-4) | **49.8** | **45.7** | **47.8** | Best, +11.17% average |

### Ablation Study

| Configuration | QAMPARI (F1) | RoMQA (F1) | Notes |
|------|-------------|-----------|------|
| Full RI²VER | 49.8 | 45.7| Full framework |
| W/o verification (Independent reading only) | 42.1 | 37.3 | Removing verification significantly degrades precision |
| W/o additional evidence retrieval | 45.2 | 41.0 | Verifying with only original passages yields suboptimal results |
| Concatenated reading instead of independent reading | 44.5 | 40.8 | Concatenation leads to information overload |
| W/o verification question generation (Direct verification) | 46.8 | 42.9 | Performance drops slightly; verification questions provide auxiliary help |

### Key Findings

- **Inter-passage verification is the most contributing component**—removing the verification step leads to an F1 drop of 7.7%/8.4%, showing that the noise introduced during the high-recall stage must be eliminated through verification.
- **The contribution of additional evidence retrieval is significant**—verifying only with original passages is insufficient, and newly retrieved evidence provides crucial cross-verification signals.
- **Independent reading outperforms concatenated reading**—even when the context window is sufficiently large, concatenating many passages still results in performance degradation.
- **RI²VER shows the most pronounced advantage on questions requiring multi-evidence synthesis**—gains are smaller for questions answerable with a single source, but exceed 15% for questions requiring the synthesis of multiple passages.
- **The framework generalizes well across model sizes**—even with small models like Llama-7B, it achieves significant F1 improvements.

## Highlights & Insights

- **The two-stage decoupled design of "independent reading + cross-verification"** elegantly resolves the recall-precision trade-off. This "coarse-to-fine" strategy can be generalized to other tasks requiring multi-source information synthesis (such as multi-document summarization, fact-checking).
- **Verification question generation is an ingenious intermediate step**—it converts the vague question "is this answer correct?" into specific, retrievable, and verifiable factual questions, dramatically improving verification efficiency.
- **The training-free pipeline design** allows the framework to be applied plug-and-play to any LLM, lowering deployment barriers.

## Limitations & Future Work

- The computational overhead of the verification step is high—each candidate answer requires verification question generation, additional retrieval, and synthesis judgment, leading to significant latency when candidates are numerous.
- The quality of generated verification questions depends heavily on LLM capability—unrelated verification questions will degrade the verification effectiveness.
- The framework is evaluated only on factual QA datasets; its effectiveness on subjective or reasoning-heavy multi-answer questions (e.g., "What are the ways to lose weight?") remains unknown.
- Retriever quality acts as a bottleneck for the entire framework—if the quality of additionally retrieved passages is poor, the verification process might instead introduce false negatives or positives.

## Related Work & Insights

- **vs Self-RAG (Asai et al. 2024)**: Self-RAG allows LLMs to adaptively decide whether to retrieve, but does not explicitly handle multi-answer scenarios. RI²VER's independent reading + verification strategy is better suited for questions with numerous answers.
- **vs Chain-of-Verification (Dhuliawala et al. 2023)**: CoVe also uses verification questions to mitigate hallucinations, but is designed for single-answer scenarios. RI²VER extends this approach to multi-answer scenarios and introduces cross-passage cross-verification.
- **vs Verify-and-Edit (Zhao et al. 2023)**: Verify-and-Edit edits answers post-generation, whereas RI²VER filters the candidate answer set—the latter is more suitable for multi-answer scenarios where retaining or discarding candidates is preferred over direct editing.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of inter-passage cross-verification is novel and intuitive, though individual components (retrieval, verification questions, synthesis) have counterparts in prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage with two datasets, multiple model sizes, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the framework and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐ Multi-answer QA is a significant problem in practical scenarios, and RI²VER provides a practical and effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification](clac_at_semeval-2025_task_6_a_multi-architecture_approach_for_corporate_environm.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](../../ICCV2025/others/intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ACL 2025\] Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)

</div>

<!-- RELATED:END -->
