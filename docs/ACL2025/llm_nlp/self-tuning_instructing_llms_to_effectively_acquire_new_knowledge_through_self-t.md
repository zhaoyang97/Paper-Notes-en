---
title: >-
  [Paper Note] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching
description: >-
  [ACL 2025][LLM (Other)][Knowledge Injection] Inspired by the Feynman technique, a Self-Tuning framework is proposed. Through a three-layer self-teaching strategy of memorization, comprehension, and self-reflection, it significantly enhances the ability of LLMs to effectively acquire and recall knowledge from new documents.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Knowledge Injection"
  - "Self-Teaching"
  - "Feynman Technique"
  - "continual learning"
  - "Knowledge Acquisition"
date: 2026-05-08
content_hash: cb6a742f2454c788
---

# Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching

**Conference**: ACL 2025  
**arXiv**: [2406.06326](https://arxiv.org/abs/2406.06326)  
**Code**: [https://github.com/zhangxy-2019/Effective-Knowledge-Injection](https://github.com/zhangxy-2019/Effective-Knowledge-Injection)  
**Area**: LLM NLP / Knowledge Injection  
**Keywords**: Knowledge Injection, Self-Teaching, Feynman Technique, continual learning, Knowledge Acquisition

## TL;DR
Inspired by the Feynman technique, a Self-Tuning framework is proposed. Through a three-layer self-teaching strategy of memorization, comprehension, and self-reflection, it significantly enhances the ability of LLMs to effectively acquire and recall knowledge from new documents.

## Background & Motivation
**Background**: The knowledge of LLMs becomes outdated due to one-time training and a constantly changing world, necessitating the continuous injection of new knowledge.

**Limitations of Prior Work**: Standard continual pre-training struggles to extract stored knowledge; even after instruction fine-tuning, knowledge extraction remains limited.

**Key Challenge**: Existing methods overemphasize "memorization" while neglecting "comprehension"—even if PPL is reduced, knowledge cannot be effectively extracted in QA tasks.

**Goal**: To enable LLMs to efficiently absorb, comprehend, and recall new knowledge from raw documents.

**Key Insight**: Design self-supervised learning tasks by drawing inspiration from the core concept of "comprehension + self-reflection" in the Feynman technique.

**Core Idea**: First teach the model "how to learn" (Stage 1), and then let it learn new documents autonomously (Stage 2-3).

## Method

### Overall Architecture
Three-stage training: Stage 1 learns the ability to absorb knowledge on training documents $\rightarrow$ Stage 2 applies learning strategies to test documents $\rightarrow$ Stage 3 continually learns test documents.

### Key Designs
1. **Memorization Task**:

    - **Function**: Performs next-token prediction on the raw text.
    - **Mechanism**: Standard language modeling to embed factual information into the parameters.
    - **Design Motivation**: The first step of the Feynman technique—memorizing basic facts.

2. **Comprehension Task**:

    - **Function**: Summarization, key information identification, and natural language inference.
    - **Mechanism**: (i) Uses titles as gold standards for summarization, (ii) uses SpaCy to identify entities, and (iii) generates NLI samples from the documents.
    - **Design Motivation**: The "explaining in one's own words" aspect of the Feynman technique.

3. **Self-Reflection Task**:

    - **Function**: "Teaching", "flashcards", cloze tests, multiple choice, and sentence completion.
    - **Mechanism**: All tasks are self-supervisedly generated based on document content, facilitating recall in a closed-book manner.
    - **Design Motivation**: The "finding and filling knowledge gaps" aspect of the Feynman technique.

### Loss & Training
- Stage 1: $L^{Stage1}_\theta = L_\theta(D^{Doc}_{train}) + L_\theta(D^{Self}_{train}) + L_\theta(D^{QA}_{train})$
- Stage 2: $L^{Stage2}_\theta = L_\theta(D^{Doc}_{test}) + L_\theta(D^{QA}_{train})$
- Stage 3: $L^{Stage3}_\theta = L_\theta(D^{Doc}_{test})$

## Key Experimental Results

### Main Results (Llama2-7B, Wiki-Bio single-domain scenario)

| Method | PPL↓ | EM↑ | F1↑ | Reasoning Acc↑ | NQ F1↑ | CSQA Acc↑ |
|------|------|-----|-----|---------|--------|----------|
| Closed-book | 8.41 | 2.87 | 14.63 | 7.96 | 24.67 | 53.40 |
| Cont. Pre-train | 7.28 | 3.62 | 15.96 | 15.09 | 24.11 | 53.40 |
| Std. Ins.-tuning | 6.83 | 5.13 | 19.15 | 39.09 | 23.67 | 51.84 |
| PIT | 2.08 | 11.61 | 27.15 | 11.93 | 26.31 | 57.58 |
| **Self-Tuning** | **1.11** | **31.52** | **50.83** | **44.31** | **25.67** | **66.01** |

### Ablation Study

| Variant | EM | F1 | Reasoning Acc |
|------|-----|-----|---------|
| Self-Tuning (Full) | 31.52 | 50.83| 44.31 |
| w/o Review (remove Stage 2 QA) | EM drops | F1 drops | - |
| via Reading Comp. (replaced with reading comprehension) | Lower than full version | - | - |

### Key Findings
- Self-Tuning improves the knowledge extraction EM from 2.87% to 31.52%, approaching the open-book level (31.83%).
- The PPL drops almost to 1, demonstrating that new documents are effectively memorized.
- Excellent knowledge retention: NQ F1 and CSQA Acc increase rather than decrease.
- Significant advantages are also maintained in the cross-domain scenario (Wiki-Film).

## Highlights & Insights
- The analogy to the Feynman technique is highly intuitive, and the three-layer task design is backed by solid learning theory.
- All self-teaching tasks are generated in a self-supervised manner, requiring no additional annotations or special templates.
- The results of knowledge retention provide confidence—learning new knowledge does not necessarily imply forgetting old knowledge.
- The Wiki-Newpages-2023-QA dataset itself is a highly valuable contribution.

## Limitations & Future Work
- The three-stage training increases computational costs.
- Validated only on Wikipedia-like knowledge documents; the performance on long technical documents remains unknown.
- The training documents for Stage 1 require related QA data, so generalizing to completely new domains requires additional effort.

## Related Work & Insights
- **vs PIT (Jiang et al. 2024)**: PIT focuses only on memorization rather than comprehension, whereas Self-Tuning demonstrates that comprehension + self-reflection is far superior to pure memorization.
- **vs ReadComprehension (Cheng et al. 2024)**: The reading comprehension framework relies on mining patterns, while Self-Tuning's self-supervised generation is more flexible.

## Supplementary Details
- Dataset source: Wikipedia NewPages from September to October 2023.
- Three datasets: Wiki-Bio (single-domain), Wiki-Multi (multi-domain), and Wiki-Film (cross-domain).
- Evaluation dimensions: Memorization (PPL), Extraction (EM/F1), and Reasoning (NLI Accuracy).
- Knowledge retention evaluation: Natural Questions and CommonsenseQA.
- Self-teaching tasks are generated through self-supervision using SpaCy and NLTK.
- Consistent advantages are also verified on Qwen2-7B and Mistral-7B.
- The cross-domain scenario uses Wiki-Bio training data to test generalization capability.
- Self-Tuning's knowledge extraction EM approaches the open-book level.
- Self-reflection tasks encompass five formats: teaching, flashcards, cloze tests, multiple choice, and sentence completion.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of introducing the Feynman technique to LLM knowledge injection is novel, and the self-teaching task design is systematic.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 scenarios $\times$ 3 models $\times$ multiple metrics + knowledge retention evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with a smooth transition between method and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a practical training framework for updating LLM knowledge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ACL 2025\] SDD: Self-Degraded Defense against Malicious Fine-tuning](sdd_self-degraded_defense_against_malicious_fine-tuning.md)
- [\[ACL 2025\] Understanding the Dark Side of LLMs' Intrinsic Self-Correction](understanding_the_dark_side_of_llms_intrinsic_self-correction.md)
- [\[ACL 2025\] Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement](just_a_scratch_enhancing_llm_capabilities_for_self-harm_detection_through_intent.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)

</div>

<!-- RELATED:END -->
