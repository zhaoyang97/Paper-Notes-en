---
title: >-
  [Paper Note] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models
description: >-
  [ACL 2025][Information Retrieval & RAG][Retrieval-Augmented Reasoning] Proposes RARE, which introduces two retrieval-augmented actions into the MCTS reasoning framework of rStar (A6: generate search queries based on the original question and retrieve; A7: retrieve and re-answer sub-questions) and replaces the original discriminator with a Retrieval-Augmented Factuality Scorer (RAFS), enabling LLaMA 3.1 to match or exceed GPT-4o performance on medical and commonsense reasoning…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Reasoning"
  - "MCTS"
  - "Factuality Scoring"
  - "Medical Question Answering"
  - "Commonsense Reasoning"
date: 2026-05-08
content_hash: 88255dde6c529cfb
---

# RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2412.02830](https://arxiv.org/abs/2412.02830)  
**Code**: [https://github.com/fatebreaker/RARE](https://github.com/fatebreaker/RARE)  
**Area**: Information Retrieval  
**Keywords**: Retrieval-Augmented Reasoning, MCTS, Factuality Scoring, Medical Question Answering, Commonsense Reasoning

## TL;DR
Proposes RARE, which introduces two retrieval-augmented actions into the MCTS reasoning framework of rStar (A6: generate search queries based on the original question and retrieve; A7: retrieve and re-answer sub-questions) and replaces the original discriminator with a Retrieval-Augmented Factuality Scorer (RAFS), enabling LLaMA 3.1 to match or exceed GPT-4o performance on medical and commonsense reasoning tasks.

## Background & Motivation

**Background**: LLM reasoning enhancement methods include CoT, Self-Consistency, and MCTS search (rStar). For knowledge-intensive tasks (such as medical QA), RAG methods (such as MedRAG, i-MedRAG) are also needed to introduce external knowledge. However, reasoning enhancement and retrieval augmentation are usually designed separately.

**Limitations of Prior Work**:
   - Pure reasoning enhancement (rStar/MCTS) lacks external knowledge, limited by pre-training knowledge on knowledge-intensive questions.
   - Pure RAG methods (MedRAG) perform only a single retrieval, failing to dynamically acquire information at each step of multi-step reasoning.
   - Although i-MedRAG supports iterative retrieval, it lacks a systematic reasoning framework and path selection mechanism.

**Key Challenge**: Knowledge-intensive tasks require a combination of **structured multi-step reasoning + dynamic retrieval**, whereas existing methods only achieve one of the two.

**Key Insight**: Introduce two new retrieval-augmented actions into the MCTS action space of rStar, and replace the answer verification with a retrieval-augmented factuality scorer, achieving deep integration of reasoning and retrieval.

**Core Idea**: Integrate retrieval augmentation into the MCTS action space and scoring mechanism, allowing each branch of the reasoning tree to dynamically acquire external knowledge and verify factuality.

## Method

### Overall Architecture
**Phase 1 - Retrieval-Augmented Generator**: Explores reasoning paths based on MCTS, adding A6 (question-level retrieval) and A7 (sub-question-level retrieval) to the original 5 actions (A1-A5) of rStar, to generate multiple candidate reasoning paths.  
→ **Phase 2 - Retrieval-Augmented Factuality Scorer (RAFS)**: Performs factuality verification on each reasoning path and selects the path with the highest score as the final answer.

### Key Designs

1. **A6: Search Query Generation + Information Retrieval**:

    - The LLM generates multiple search queries based on the original question.
    - ColBERT is used to retrieve relevant documents from the corpora (PubMed, StatPearls, medical textbooks, Wikipedia).
    - The retrieved information is integrated with the original question to generate the final answer.
    - Applicable to single-step knowledge-based questions requiring external knowledge supplementation.

2. **A7: Sub-question Retrieval + Re-answering**:

    - Independent retrieval is performed for the sub-questions generated in A3.
    - Each sub-question is re-answered based on the retrieved contextual information.
    - The final sub-question is a reformulation of the original question, whose answer serves as the answer to the original question.
    - Applicable to complex questions requiring iterative retrieval.

3. **Retrieval-Augmented Factuality Scorer (RAFS)**:

    - **Step 1**: Deconstruct the reasoning path into independent claims.
    - **Step 2**: The LLM generates retrieval queries for each claim.
    - **Step 3**: Retrieve relevant documents.
    - **Step 4**: Compare each claim with the retrieved evidence, labeling it as Supported/Not Supported.
    - **Factuality Score**: The proportion of Supported claims -> select the path with the highest score.

## Key Experimental Results

### Main Results (Medical Reasoning Accuracy)

| Model | Method | MedQA | MedMCQA | MMLU-Med | Avg |
|------|------|-------|---------|----------|-----|
| LLaMA3.1 8B | CoT | 61.51 | 55.15 | 71.63 | 62.76 |
| LLaMA3.1 8B | rStar | 70.40 | 62.13 | 79.16 | 70.56 |
| LLaMA3.1 8B | **RARE** | **75.57** | **64.32** | **81.63** | **73.84** |
| LLaMA3.1 70B | **RARE** | **87.43** | **75.18** | **90.91** | **84.51** |
| GPT-4o | CoT | 85.55 | 74.70 | 90.45 | 83.57 |

RARE + LLaMA3.1-70B **outperforms GPT-4o** on MedQA (87.43) and MMLU-Med (90.91).

### Commonsense Reasoning Comparison

| Model | Method | StrategyQA | CommonsenseQA | PIQA | Avg |
|------|------|-----------|---------------|------|-----|
| LLaMA3.1 8B | rStar | 71.57 | 76.58 | 79.65 | 75.93 |
| LLaMA3.1 8B | **RARE** | **78.02** | **80.84** | **82.52** | **80.46** |

### Key Findings
- **RARE yields the greatest gains on multi-step reasoning tasks**: On StrategyQA, it improves by 10.19% over CoT, far exceeding the 7.22% on CommonsenseQA, indicating that retrieval augmentation is more valuable in multi-hop reasoning.
- RAFS is more effective than the original rStar discriminator—factuality scoring is more reliable than pure self-consistency voting.
- A6 and A7 are complementary: A6 is suitable for single-step knowledge queries, while A7 is suitable for complex compound reasoning.

## Highlights & Insights
- **Deep integration of retrieval and reasoning**: Instead of simple "retrieve first, then reason," retrieval is treated as an atomic action in MCTS, allowing the search tree to naturally explore "when to retrieve and what to retrieve."
- **The factuality verification method of RAFS** can be used independently—the process of decomposing reasoning paths into claims -> verifying claim-by-claim -> percentage scoring is applicable to any scenario requiring factuality evaluation.
- The results of open-source LLaMA outperforming GPT-4o demonstrate the importance of reasoning framework design.

## Limitations & Future Work
- The computational overhead of MCTS is still relatively large (multiple rollouts + retrieval).
- The retrieval corpus needs to be pre-constructed, requiring adaptation for new domains.
- Using LLMs to judge Supported/Not Supported in RAFS may introduce judgment bias.
- Only evaluated on multiple-choice questions; effectiveness on open-ended generation tasks remains unknown.

## Related Work & Insights
- **vs rStar**: rStar provides a reasoning framework but lacks external knowledge; RARE outperforms rStar by 3.28% on average (medical).
- **vs i-MedRAG**: i-MedRAG features iterative retrieval but lacks a systematic reasoning framework; RARE outperforms i-MedRAG by 2.63% on average.
- **vs MedRAG**: Single retrieval in MedRAG is insufficient for complex questions; RARE leads by a large margin.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovatively integrates RAG into the MCTS action space; RAFS factuality scoring has a clear idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 medical + 3 commonsense benchmarks, 3 scales of LLaMA, compared with GPT-4o, and detailed ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear flowcharts, detailed case analyses, and strong motivation.
- Value: ⭐⭐⭐⭐⭐ A practical solution for open-source models outperforming GPT-4o, with direct application value for medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](astute_rag_knowledge_conflicts.md)
- [\[ACL 2025\] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [\[ACL 2025\] SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)

</div>

<!-- RELATED:END -->
