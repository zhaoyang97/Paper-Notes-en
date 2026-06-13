---
title: >-
  [Paper Note] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation
description: >-
  [ACL 2026][Hallucination Detection][Factual Hallucination] This paper systematically analyzes the phenomenon of factual hallucinations caused by learning new knowledge during the SFT stage using a controlled synthetic da…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "Factual Hallucination"
  - "New Knowledge Learning"
  - "Attention Mechanism"
  - "SFT"
  - "KnownPatch"
date: 2026-05-08
content_hash: 622210144bc20b3f
---

# Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation

**Conference**: ACL 2026 Findings  
**arXiv**: [2511.02626](https://arxiv.org/abs/2511.02626)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Factual Hallucination, New Knowledge Learning, Attention Mechanism, SFT, KnownPatch

## TL;DR

This paper systematically analyzes the phenomenon of factual hallucinations caused by learning new knowledge during the SFT stage using a controlled synthetic dataset, Biography-Reasoning. It discovers that the fundamental mechanism of hallucination is the weakening of the model's attention toward key entities. The authors propose KnownPatch—injecting a small amount of known knowledge at the end of training to restore attention patterns—effectively mitigating hallucinations.

## Background & Motivation

**Background**: LLMs acquire rich world knowledge during pre-training and learn to follow instructions during the SFT stage. Existing research indicates that introducing new knowledge not covered in pre-training during SFT increases the risk of factual hallucinations—models incorrectly generate newly learned information in irrelevant contexts.

**Limitations of Prior Work**: Previous work mainly focused on closed-domain QA scenarios with mixed knowledge types, leading to insufficient understanding of specific hallucination manifestations and underlying mechanisms. Specifically: (1) the propagation patterns of hallucinations across different knowledge and task types are unclear; (2) the causes at the attention mechanism level have not been revealed; (3) lightweight mitigation methods are lacking.

**Key Challenge**: When a specific category of knowledge consists entirely of new knowledge, severe hallucinations occur even if the total amount of new knowledge is small. This differs from the simple understanding that "higher proportions of new knowledge lead to more severe hallucinations"—the critical factor is the degree of unfamiliarity *within* a specific knowledge type rather than the global ratio of new knowledge.

**Goal**: (1) Construct a controlled dataset for fine-grained analysis of hallucination performance; (2) reveal the attention mechanism behind hallucinations; (3) propose a lightweight mitigation method.

**Key Insight**: Build a synthetic biography dataset to precisely control the ratio and type of known/unknown knowledge, and use attention analysis to track the generation and propagation mechanisms of hallucinations.

**Core Idea**: Learning new knowledge weakens the model's attention to key entities in the question, leading to over-reliance on other tokens in the context, which results in hallucinations. Injecting known knowledge at the end of training can restore these attention patterns.

## Method

### Overall Architecture

The authors construct the synthetic dataset Biography-Reasoning (Person Entities × 4 Attributes × 4 QA + 12 Reasoning tasks) to analyze hallucinations by controlling the ratio of known/unknown knowledge. The analysis is conducted at three levels: (1) fine-grained performance of hallucination phenomena; (2) interpretability analysis of attention mechanisms; (3) the KnownPatch mitigation method.

### Key Designs

1.  **Controlled Synthetic Dataset Biography-Reasoning**:
    *   **Function**: Precisely control the types and ratios of known/unknown knowledge to isolate causal factors of hallucination.
    *   **Mechanism**: Define four attributes for fictional characters (birth year, death year, profession, university), each representing a knowledge type. Construct four QA tasks and twelve reasoning tasks (single-step, comparative, and novel reasoning). Use continue pre-training to make some knowledge "known" while keeping the rest "unknown," then mix them in different proportions during SFT.
    *   **Design Motivation**: Real-world datasets cannot precisely control what knowledge is already known by the model; synthetic datasets eliminate this confounding factor.

2.  **Attention Analysis and KnownPatch**:
    *   **Function**: Reveal hallucination mechanisms and provide lightweight mitigation.
    *   **Mechanism**: Analyze changes in attention toward key entities (person name tokens) in the middle-to-late layers (layers 12-24). Findings: Learning new knowledge significantly reduces attention to key entities (attention value drop is highly correlated with hallucination severity); learning known knowledge enhances this attention. Based on this, KnownPatch is proposed: inject a small amount of known knowledge samples (5-20%) at the end of training to repair attention patterns damaged by new knowledge.
    *   **Design Motivation**: If hallucinations stem from the disruption of attention patterns, restoring the correct patterns should mitigate hallucinations without filtering all new knowledge from the training data.

3.  **Hallucination Propagation Mechanism Analysis**:
    *   **Function**: Reveal how hallucinations propagate from training tasks to test tasks.
    *   **Mechanism**: Construct task variants that are lexically similar but semantically different, and vice-versa. Findings show that hallucination propagation is primarily driven by lexical similarity (token overlap) rather than semantic similarity. When key entity attention drops, excess attention flows to surrounding context tokens; test samples sharing vocabulary with unknown knowledge samples in training are more easily affected.
    *   **Design Motivation**: Understanding propagation mechanisms helps predict which tasks are most susceptible to hallucinations, allowing for targeted defense.

### Loss & Training

Standard SFT uses cross-entropy loss. KnownPatch injects known knowledge samples in the final stage of training (not shuffled, but placed at the end), utilizing the training order effect to repair attention. Control experiments also tested adding a KL divergence constraint ($\alpha=25$) to maintain consistency in the attention module's output directly.

## Key Experimental Results

### Main Results

| Condition | STQA Accuracy Drop | Wiki Accuracy Drop | Notes |
| :--- | :--- | :--- | :--- |
| All Known (Baseline) | 0% | 0% | No hallucination |
| One Type All Unknown | >50% | Significant drop | Severe hallucination |
| KeepKnown 50% | Moderate drop | Moderate drop | Retaining known knowledge mitigates hallucination |
| RemoveKnown 5% | Severe drop | Severe drop | Entirely unknown types are extremely harmful |

### Ablation Study

| Configuration | STQA | Wiki | Notes |
| :--- | :--- | :--- | :--- |
| KnownPatch 5% | Significant recovery | Significant recovery | Effective with only 5% known injection |
| KnownPatch 20% | Near baseline | Slightly above baseline | Approaches upper bound |
| Shuffled 20% | Moderate recovery | Moderate recovery | Shuffling is less effective than end injection |
| KL Constraint | Partial mitigation | Partial mitigation | Direct attention constraint works but has side effects |

### Key Findings

*   **Type-specific unfamiliarity is more important than global ratio**: Even if the total amount of new knowledge is small, if a specific knowledge type consists entirely of unknown knowledge (RemoveKnown), it leads to extremely severe hallucinations. KeepKnown (replacing 50%) is much better than RemoveKnown (replacing 5%).
*   **Cross-type hallucination propagation**: Learning new knowledge for one type leads to hallucinations in same-type QA (STQA drop >50%) and propagates to different-type QA (DTQA drop ~5%) and OOD Wiki test sets.
*   **Reverse propagation from reasoning tasks to QA**: Learning reasoning tasks with unknown knowledge actually causes more severe hallucinations in the QA test set than in other reasoning test sets, because the QA context has higher lexical overlap with reasoning trajectories.
*   **High correlation between attention and hallucination**: The higher the ratio of unknown knowledge, the lower the attention to key entities and the more severe the hallucination. The correlation curves match almost perfectly.
*   **Non-replay nature of KnownPatch**: Even if the injected known knowledge does not cover all unknown types, it still mitigates hallucinations in those uncovered types, suggesting KnownPatch works by restoring attention patterns rather than through knowledge replay.

## Highlights & Insights

*   **"Type-specific total unfamiliarity" is more dangerous than "high global ratio"**: This finding overturns the simple understanding that "higher ratio of new knowledge is more dangerous" and provides direct guidance for SFT data construction—ensure that every knowledge type retains samples the model already knows.
*   **Lexical similarity drives propagation**: This explains why seemingly unrelated tasks are affected by hallucinations—as long as they share enough token overlap with training samples containing new knowledge.
*   **Lightweight nature of KnownPatch**: Injecting only 5% known knowledge at the end of training significantly mitigates hallucinations, avoiding the need for expensive categorization of the entire training set into known/unknown.

## Limitations & Future Work

*   Experiments were mainly conducted on Qwen2.5-1.5B, though consistency was verified on Llama3.2-1B, Qwen3-8B, and Qwen2.5-32B in the appendix.
*   The use of synthetic datasets means the complexity and distribution of real-world knowledge might differ.
*   KnownPatch requires access to known knowledge samples; determining whether knowledge is "known" remains an open challenge in practice.
*   The mechanism of non-factual hallucinations (e.g., logical or formatting errors) was not explored.

## Related Work & Insights

*   **vs Gekhman et al. (2024)**: They found that higher new knowledge ratios lead to more hallucinations but used mixed knowledge types. This paper reveals a more fine-grained rule—the internal unfamiliarity of a type is key.
*   **vs Sun et al. (2025)**: They analyzed the over-generalization of new knowledge from a token probability perspective; this paper provides a complementary explanation from the perspective of attention mechanisms.

## Rating

*   Novelty: ⭐⭐⭐⭐ The controlled experiment design is ingenious; the discovery of "intra-type unfamiliarity" is novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional ablation, multi-model validation, attention analysis, and propagation mechanism analysis are extremely thorough.
*   Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from phenomenon to mechanism to mitigation is very clear.
*   Value: ⭐⭐⭐⭐⭐ Significant practical guidance for understanding and mitigating hallucinations during the SFT stage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why LLMs Hallucinate on Structured Knowledge: A Mechanistic Analysis of the Reasoning Process](why_llms_hallucinate_on_structured_knowledge_a_mechanistic_analysis_of_reasoning.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)
- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)

</div>

<!-- RELATED:END -->
