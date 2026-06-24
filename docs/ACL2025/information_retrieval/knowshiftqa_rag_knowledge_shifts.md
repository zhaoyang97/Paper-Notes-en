---
title: >-
  [Paper Note] KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG Robustness] The authors construct the KnowShiftQA dataset (3,005 questions across 5 subjects) to simulate differences between textbooks and the parametric knowledge of LLMs through hypothetical knowledge updates. They systematically evaluate the robustness of RAG systems under knowledge shifts, finding that the performance of existing RAG systems drops by 22-27% under knowledge shifts.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG Robustness"
  - "Knowledge Shift"
  - "K-12 Education"
  - "Hypothetical Knowledge Update"
  - "Contextual Knowledge Integration"
date: 2026-05-08
content_hash: a839d7289947858c
---

# KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?

**Conference**: ACL 2025  
**arXiv**: [2412.08985](https://arxiv.org/abs/2412.08985)  
**Code**: [GitHub](https://github.com/HKUST-KnowComp/KnowShiftQA)  
**Area**: Retrieval-Augmented Generation / Educational Question Answering / Knowledge Conflict  
**Keywords**: RAG Robustness, Knowledge Shift, K-12 Education, Hypothetical Knowledge Update, Contextual Knowledge Integration

## TL;DR

The authors construct the KnowShiftQA dataset (3,005 questions across 5 subjects) to simulate differences between textbooks and the parametric knowledge of LLMs through hypothetical knowledge updates. They systematically evaluate the robustness of RAG systems under knowledge shifts, finding that the performance of existing RAG systems drops by 22-27% under knowledge shifts.

## Background & Motivation

**Background**: RAG systems perform exceptionally well in knowledge-intensive tasks and hold great potential as question-answering assistant tools in the K-12 education domain.

**Limitations of Prior Work**: Significant differences may exist between textbook knowledge and LLMs' internal parametric knowledge due to factual evolution, curriculum updates, and regional cultural differences. However, the impact of such knowledge shifts on RAG systems has not been systematically studied.

**Key Challenge**: LLMs must correctly utilize externally retrieved textbook information to answer questions. However, when textbook knowledge conflicts with their internal knowledge, models may tend to rely on their own parametric knowledge, leading to inconsistent answers.

**Goal**: To systematically evaluate the robustness of RAG systems when facing knowledge shifts in K-12 educational scenarios.

**Key Insight**: A hypothetical knowledge update method is designed to replace correct facts in textbooks with plausible but incorrect alternative answers, while modifying relevant contexts to maintain consistency.

**Core Idea**: Simulate real-world knowledge shifts using hypothetical knowledge updates, and design five question types to stress-test LLMs' context utilization and knowledge integration capabilities.

## Method

### Overall Architecture

The construction pipeline of KnowShiftQA consists of: (1) curating factual questions from open-source textbooks; (2) selecting plausible but incorrect answers as the updated correct answers; (3) replacing all relevant content in textbook passages and adjusting for consistency; (4) validating via human annotation. The dataset ultimately covers 5 subjects: Physics, Chemistry, Biology, Geography, and History.

### Key Designs

1. **Hypothetical Knowledge Update**: Replaces original facts in textbooks with plausible alternative answers (e.g., "night-vision goggles detect infrared light" $\rightarrow$ "ultraviolet light"), while modifying relevant passages to maintain semantic consistency. Compared to collecting real-world knowledge conflicts, this approach is more controllable and scalable.
2. **Five Question Types**: 
    - Simple Direct: Single-step direct factual query.
    - Multi-hop Direct: Multi-hop direct reasoning.
    - Multi-hop Distant: Tests long-distance context utilization ability.
    - Multi-hop Implicit: Requires integrating contextual knowledge with parametric knowledge.
    - Distant Implicit: Requires both long-distance utilization and knowledge integration (the most challenging).
3. **Disentangling Two Capabilities**: Context Utilization: whether the model can locate and use the corresponding facts from the context; Knowledge Integration: whether the model can combine contextual facts with its own parametric knowledge for reasoning.

### Loss & Training

Since this paper introduces an evaluation dataset, it does not involve training. Experiments are conducted in a zero-shot setting, utilizing the "Locate-and-Answer" prompting strategy to guide LLMs to actively retrieve information from the context.

## Key Experimental Results

### Main Results

LLM Question Answering Accuracy (%, with ground-truth documents provided):

| Model | Simple Direct | Multi-hop Implicit | Distant Implicit | Average |
|------|-------------|-------------------|-----------------|------|
| Mistral-7b | 77.70 | 45.32 | 33.98 | 61.26 |
| Llama3-8b | 90.33 | 63.55 | 49.43 | 77.77 |
| GPT-4-turbo | 95.74 | 81.06 | 71.71 | 88.99 |
| Claude-3.5-sonnet | 97.54 | 83.69 | 73.82 | 90.08 |
| o1-preview | 95.08 | 86.33 | 78.86 | 91.68 |

Overall performance degradation of RAG systems:

| RAG System | Before Update | After Update | Drop |
|--------|-------|-------|------|
| Llama3-8b + Ada-002 | 87.49 | 62.60 | 24.89 |
| GPT-4o + Rerank | 97.10 | 73.71 | 23.39 |

### Ablation Study

Comparison of retrieval methods (Recall@1 / @5):

| Retrieval Method | R@1 | R@5 |
|---------|-----|-----|
| BM25 | 82.73 | 95.27 |
| Ada-002 | 79.23 | 95.44 |
| Hybrid Rerank | 84.43 | 96.04 |
| Contriever (Finetuned) | 84.19 | 98.96 |
| Con.-msmarco (Finetuned) | **87.95** | **99.50** |

### Key Findings

- **Knowledge integration is an emergent capability**: There is a huge performance gap on "Implicit" type questions (which require integrating context and parametric knowledge) between small and large models (Mistral-7b 34% vs o1-preview 79%), indicating that this is an emergent capability.
- **Multi-hop reasoning and long-distance utilization are not the primary bottlenecks**: Most LLMs exhibit comparable performance across three categories: Simple Direct, Multi-hop Direct, and Multi-hop Distant.
- **Lexical retrieval has advantages in academic domains**: Leveraging exact matches of academic terminology, BM25 performs comparably to or even better than dense retrieval methods in educational document retrieval.
- **RAG systems are universally fragile**: Knowledge shifts lead to a 22-27% accuracy drop.

## Highlights & Insights

- **Exquisite Design of Question Types**: The difficulty gradients across the five question types clearly disentangle two separate capabilities: "context utilization" and "knowledge integration".
- **Addressing Data Scarcity**: The hypothetical knowledge update method successfully addresses the difficulty of systematically collecting real-world knowledge conflict data due to their sparsity.
- **Identifying Core Bottlenecks**: The study identifies that "knowledge integration" (combining retrieved knowledge with parametric knowledge) is the core bottleneck for LLMs when encountering knowledge shifts.
- **Significance of Domain Adaptation**: The finetuned Contriever significantly outperforms general-purpose models in educational document retrieval, emphasizing the critical importance of domain adaptation.

## Limitations & Future Work

- The knowledge shifts simulated by hypothetical updates may not fully reflect real-world knowledge updating patterns.
- Structured retrieval methods (e.g., GraphRAG, HippoRAG) are not included in the evaluation.
- The experiments only evaluate robustness without proposing targeted mitigation strategies.
- The evaluation is limited to multiple-choice questions, failing to assess LLM performance in open-ended question answering.
- The dataset only covers English textbooks, leaving multilingual scenarios unexplored.

## Related Work & Insights

- Complementary to Parenting (also ACL 2025) in terms of perspectives on knowledge conflicts—KnowShiftQA focuses on evaluation, while Parenting focuses on solutions.
- Knowledge conflict training methods such as KAFT and KnowPO can be adapted as mitigation solutions for KnowShiftQA scenarios.
- The hypothetical knowledge update method can be generalized to other scenarios requiring simulated knowledge discrepancies (e.g., legal or medical domains).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The hypothetical knowledge update method is novel, and the question types are ingeniously designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage with 16 LLMs and 10 retrieval methods.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and rich tables.
- **Value**: ⭐⭐⭐⭐ Exposes key vulnerabilities of RAG in educational scenarios, offering practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Investigating Language Preference of Multilingual RAG Systems](investigating_language_preference_of_multilingual_rag_systems.md)
- [\[ACL 2025\] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[ACL 2025\] Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](astute_rag_knowledge_conflicts.md)
- [\[ICLR 2026\] Robust Test-Time Video-Text Retrieval: Benchmarking and Adapting for Query Shifts](../../ICLR2026/information_retrieval/robust_test-time_video-text_retrieval_benchmarking_and_adapting_for_query_shifts.md)

</div>

<!-- RELATED:END -->
