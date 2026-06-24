---
title: >-
  [Paper Note] Beyond Facts: Evaluating Intent Hallucination in Large Language Models
description: >-
  [ACL 2025][Hallucination Detection][Intent Hallucination] This paper proposes the concept of "Intent Hallucination"—generations that deviate from the user's intent due to LLMs omitting or misinterpreting certain intent constraints when handling complex multi-condition queries. It constructs the FaithQA benchmark (20,068 questions) and the Constraint Score evaluation metric. Experiments demonstrate that intent hallucination is prevalent in SOTA models and intensifies as query…
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Intent Hallucination"
  - "FaithQA"
  - "Constraint Score"
  - "Omission"
  - "Misinterpretation"
date: 2026-05-08
content_hash: 9bb307c44cfdc1b1
---

# Beyond Facts: Evaluating Intent Hallucination in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.06539](https://arxiv.org/abs/2506.06539)  
**Area**: Hallucination Detection  
**Keywords**: Intent Hallucination, FaithQA, Constraint Score, Omission, Misinterpretation

## TL;DR

This paper proposes the concept of "Intent Hallucination"—generations that deviate from the user's intent due to LLMs omitting or misinterpreting certain intent constraints when handling complex multi-condition queries. It constructs the FaithQA benchmark (20,068 questions) and the Constraint Score evaluation metric. Experiments demonstrate that intent hallucination is prevalent in SOTA models and intensifies as query complexity increases.

## Background & Motivation

- Existing research on hallucination primarily focuses on factual hallucination, where generated content does not align with real-world facts.
- In practical scenarios, users frequently present complex queries with multiple conditions to LLMs, which often satisfy only a subset of the conditions while ignoring the rest.
- Such "intent hallucinations" might be factually correct but still fail to meet the user's true intent—which existing detection methods (e.g., FActScore, SelfCheckGPT) cannot capture.
- There is a lack of benchmarks and metrics dedicated to evaluating intent hallucination: existing tools either perform only fact-checking or treat the query as a whole for coarse-grained evaluation.

## Method

### Overall Architecture

1. Define **Intent Constraint** as the basic evaluation unit—decomposing the query into a series of short statements, each representing a requirement that must be met.
2. Define two manifestations of intent hallucination: **Omission** (ignoring some parts of the query) and **Misinterpretation** (responding to unmentioned parts of the query).
3. Construct the FaithQA benchmark, covering two scenarios and multiple task formats.
4. Propose the Constraint Score metric for fine-grained evaluation.

### Key Designs

**Intent Constraint Mapping Function $C(q)$**:
- Maps a query $q$ to a three-tier constraint set:
    - $C_m(q)$: Mandatory constraints (location, time, subject, action)
    - $C_i(q)$: Important constraints (qualifiers, quantity)
    - $C_o(q)$: Optional constraints (exclusion criteria, domain-specific requirements, etc.)
- Three-step extraction: Initial evaluation $\rightarrow$ Semantic Role Labeling (SRL) $\rightarrow$ Constraint set extraction

**Constraint Score Calculation**:
- For each constraint $c$ and response $y$, whether it is satisfied is determined by a binary satisfaction function $S_\phi(c, y) \in \{0, 1\}$ parameterized by an LLM.
- Weighted Total Score:
$$\text{Weighted Total Score} = \frac{\sum (\alpha_g \times \text{number of satisfied constraints in each category})}{\sum (\alpha_g \times \text{total constraints in each category})} \times 10$$
- Score $\ge 9$ indicates strong alignment, 7-8 indicates partial satisfaction, and $\le 7$ indicates severe intent hallucination.

**FaithQA Benchmark (20,068 questions)**:
- **Omission Tasks** (query-only):
    - Fact QA (3,000 questions): List subjects that satisfy all constraints, covering technology/culture/history.
    - Creative Writing (2,000 questions): Write stories/poems according to constraints.
    - Difficulty Classification: Easy ($\le 4$ constraints) vs. Hard ($> 4$ constraints).
- **Misinterpretation Tasks** (RAG setup):
    - Response Evaluation (3,210 questions): Evaluate response alignment, randomly removing one input.
    - Content Analysis (11,858 questions): Analyze relations/summaries of articles, randomly removing one article.
    - Ideal behavior: Detect the missing content and refuse to answer.

## Key Experimental Results

### Main Results

**Main Results of FaithQA (Perfect = hallucination-free rate, CS = Constraint Score)**:

| Model | Fact QA Perfect/CS | Story Perfect/CS | Poem Perfect/CS | Response Eval Perfect/CS |
|------|-------------------|-----------------|-----------------|------------------------|
| GPT-4o | 0.49 / 8.62 | 0.38 / 7.99 | 0.40 / 8.29 | 0.09 / 5.73 |
| Claude-3.5 | 0.37 / 6.73 | 0.34 / 7.64 | 0.60 / 9.02 | 0.29 / 5.92 |
| LLaMA3-70B | 0.57 / 8.93 | 0.29 / 7.55 | 0.51 / 8.64 | 0.07 / 4.78 |
| LLaMA3-8B | 0.46 / 8.52 | 0.25 / 7.21 | 0.27 / 7.71 | 0.11 / 5.58 |
| Mistral-7B | 0.20 / 7.15 | 0.08 / 5.92 | 0.07 / 5.49 | 0.23 / 4.46 |

**Fact QA Factually-Verifiable Hallucination Rate** (Partial Results):

| Model/Difficulty | Culture-Easy Fact | Culture-Hard Fact | Tech-Easy Fact | Tech-Hard Fact |
|-----------|-------------------|-------------------|----------------|----------------|
| GPT-4o | 54.9% | 36.1% | 63.5% | 56.6% |
| LLaMA3-8B | 83.8% | 89.5% | 90.9% | 97.6% |

### Key Findings

1. **Intent hallucination is prevalent**: Even the strongest model, GPT-4o, achieves a Perfect rate of only 49% on Fact QA, meaning over half of the responses exhibit intent hallucination.
2. **Number of constraints has a significant impact**: From Easy to Hard, the Perfect rates of all models consistently decrease.
3. **Factual accuracy does not equal intent alignment**: Up to 97.6% of LLaMA3-8B's hallucinated responses are factually correct—rendering traditional fact-checking completely ineffective.
4. **Misinterpretation is harder to address than omission**: The Perfect rate drops sharply in RAG scenarios (e.g., GPT-4o achieves only 0.09 on Response Eval).
5. **LLMs know they are omitting constraints**: Qualitative analysis reveals that models often first acknowledge they might not fully satisfy the query, yet proceed to give an incomplete answer anyway.
6. **LLMs prefer well-known topics**: Even when constraints are not met, models tend to select well-known entities common in their training data.
7. **Constraint Score vs. Human Evaluation**: MSE of 0.50 (vs. Baseline 4.72), with 66.3% of the scores within one standard deviation.
8. **Subjects and actions are most heavily violated**: Compared to details like location and time, LLMs are more likely to omit or misinterpret core semantic elements.

## Highlights & Insights

- The concept of "intent hallucination" fills an important gap in hallucination research—generations that are factually correct but not aligned with user intent are equally critical issues.
- The design of FaithQA is clever: the omission tasks control difficulty via the number of constraints, while the misinterpretation tasks test understanding by removing RAG inputs.
- The three-tier weighted design (mandatory/important/optional) of the Constraint Score is highly reasonable and aligns closer to human judgment than a direct LLM-as-a-judge approach.
- The discovery that "LLMs know they are omitting constraints" is profound—suggesting that instruction tuning might encourage a bias of "providing any answer is better than refusing to answer."

## Limitations & Future Work

- The Constraint Score relies on GPT-4o as the evaluator model, presenting issues of circular dependency and high cost.
- The FaithQA test set is only randomly sampled at 150 questions per category (due to cost constraints), limiting statistical power.
- The boundary between omission and misinterpretation is not always distinct—in some cases, both coexist.
- No mitigation methods for intent hallucination are proposed, as the study focuses solely on detection and evaluation.
- The evaluation of Creative Writing is highly subjective, and disagreements may arise in determining constraint satisfaction.

## Related Work & Insights

- **Factual Hallucination**: FActScore (Min et al. 2023), SelfCheckGPT (Manakul et al. 2023)—focusing solely on fact-checking.
- **Hallucination Benchmarks**: HaluEval (Li et al. 2023), FELM, RAGTruth—focusing on factual hallucination.
- **Instruction Following**: InfoBench (Qin et al. 2024)—evaluating instruction following via query decomposition but not specifically oriented toward hallucination.
- **RAG Faithfulness**: FaithEval (Ming et al. 2025)—focusing on context alignment rather than query alignment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The concept of intent hallucination is original, with a complete formal definition)
- Value: ⭐⭐⭐⭐ (Constraint Score can be directly used for evaluation, and FaithQA is open-source)
- Experimental Thoroughness: ⭐⭐⭐⭐ (7 SOTA models, multi-level/multi-task difficulties, verified by human evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear definitions, rich examples, but math notations are somewhat heavy)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EmotionHallucer: Evaluating Emotion Hallucinations in Multimodal Large Language Models](../../ICLR2026/hallucination/emotionhallucer_evaluating_emotion_hallucinations_in_multimodal_large_language_m.md)
- [\[ACL 2025\] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)
- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)
- [\[AAAI 2026\] Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](../../AAAI2026/hallucination/beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)

</div>

<!-- RELATED:END -->
