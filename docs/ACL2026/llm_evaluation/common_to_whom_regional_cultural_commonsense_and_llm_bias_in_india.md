---
title: >-
  [Paper Note] Common to Whom? Regional Cultural Commonsense and LLM Bias in India
description: >-
  [ACL 2026][LLM Evaluation][Cultural Commonsense] This paper introduces Indica, the first benchmark for evaluating sub-national cultural commonsense in LLMs. Focusing on cultural variations across five major regions of In…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Cultural Commonsense"
  - "Regional Bias"
  - "Indian Cultural Diversity"
  - "Benchmark Construction"
  - "LLM Bias"
date: 2026-05-08
content_hash: 09886484cab4f012
---

# Common to Whom? Regional Cultural Commonsense and LLM Bias in India

**Conference**: ACL 2026  
**arXiv**: [2601.15550](https://arxiv.org/abs/2601.15550)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Cultural Commonsense, Regional Bias, Indian Cultural Diversity, Benchmark Construction, LLM Bias

## TL;DR

This paper introduces Indica, the first benchmark for evaluating sub-national cultural commonsense in LLMs. Focusing on cultural variations across five major regions of India across eight daily life domains, the study finds that only 39.4% of questions reach consensus across all five regions. Furthermore, all evaluated LLMs exhibit geographic bias, frequently selecting Central and North India as the "default" cultural representatives.

## Background & Motivation

**Background**: Cultural commonsense benchmarks (e.g., CultureBank, CulturalBench) have begun to address cross-cultural differences, but they often treat countries as cultural monoliths, assuming uniform cultural practices within national borders.

**Limitations of Prior Work**: (1) Existing benchmarks evaluate cultural commonsense at the national level, ignoring sub-national cultural diversity; (2) Current Indian NLP benchmarks focus on factual knowledge from textbooks and exams, treating Indian culture as a singular entity; (3) LLMs may harbor systematic biases toward specific regions in culturally diverse countries, yet tools to detect such biases are lacking.

**Key Challenge**: In a country like India, with 28 states, 8 union territories, and 22 official languages, "cultural commonsense" cannot be nationally uniform. However, LLMs must make regional choices when describing a cultural practice; these implicit choices may reflect geographic biases present in training data.

**Goal**: (1) Quantify the extent of regional variation in Indian cultural commonsense; (2) Evaluate LLM accuracy on region-specific cultural knowledge; (3) Detect implicit regional biases in LLMs when geographic context is absent.

**Key Insight**: Design eight everyday cultural domains based on the Outline of Cultural Materials (OCM) and collect human-annotated responses from five Indian regions to build a region-specific cultural commonsense benchmark.

**Core Idea**: Cultural commonsense in multicultural nations is primarily regional rather than national; LLMs exhibit systematic geographic bias when processing such knowledge.

## Method

### Overall Architecture

The construction process of Indica: (1) Select 8 cultural domains → 39 themes → 515 questions based on OCM; (2) Recruit 5 participants from each of the five regions (North, South, East, West, Central) to answer all questions (totaling 15,275 responses); (3) Establish gold standards through three-tier consensus: intra-region, inter-region, and all-India consensus.

### Key Designs

1.  **Anthropology-based Question Design**:
    *   **Function**: Ensure questions cover key dimensions of daily cultural practices.
    *   **Mechanism**: Select 8 domains related to everyday cultural knowledge (Interpersonal Relations, Education, Clothing, Food/Drink, Communication, Finance, Festivals/Rituals, Transportation) from over 90 OCM categories. Select 2-4 non-overlapping sub-themes per domain, generate questions using GPT-4, and perform manual review.
    *   **Design Motivation**: Ensure focus on everyday practices rather than institutional knowledge, with sufficient diversity to reveal regional differences.

2.  **Dual-Task Evaluation (RASA + RA-MCQ)**:
    *   **Function**: Separately evaluate regional knowledge accuracy and implicit geographic bias.
    *   **Mechanism**: RASA (Region-Anchored Short Answer) — provides regional context (e.g., "In South India...") to test the ability to generate accurate regional knowledge. RA-MCQ (Region-Agnostic Multiple Choice Question) — removes geographic context to observe which regional practice the model selects by default.
    *   **Design Motivation**: RASA tests knowledge, while RA-MCQ tests bias, providing two complementary perspectives for a comprehensive cultural representation evaluation.

3.  **Three-Tier Consensus Gold Standard**:
    *   **Function**: Establish reliable regional cultural commonsense annotations.
    *   **Mechanism**: Intra-region consensus ($\geq 4/5$ participants' answers are semantically equivalent), Inter-region consensus (two regional answers match perfectly), and All-India consensus (answers from all five regions match). After initial GPT-4o classification, two human annotators perform a full review.
    *   **Design Motivation**: Strict consensus standards ensure the gold standard reflects actual regional practices rather than individual preferences.

### Loss & Training

Indica is an evaluation benchmark and does not involve model training. Evaluation utilizes Gemini 1.5 Flash as an LLM judge. Each question is run 30 times to eliminate randomness, and Chi-square Goodness-of-Fit tests are used to evaluate the statistical significance of bias.

## Key Experimental Results

### Main Results

**RASA Regional Knowledge Accuracy (%)**

| Model | North | South | East | West | Central | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | ~20 | ~19 | ~15 | ~18 | ~20 | 20.9 |
| Claude 3.5 | ~19 | ~18 | ~14 | ~17 | ~19 | 19.3 |
| Lowest Model | - | - | - | - | - | 13.4 |

### Ablation Study

| Analysis Dimension | Finding |
| :--- | :--- |
| All-India Consensus Rate | Only 39.4% of questions reach consensus across all regions. |
| Domain Differences | Transportation highest (22.6%), Festivals/Rituals lowest (1.8%). |
| Regional Pair Bias | North-Central highest (68.3%), South-East lowest (60.1%). |

### Key Findings

*   Only 39.4% of questions have consensus answers across all five regions — cultural commonsense in India is predominantly regional.
*   All 8 LLMs show low accuracy (13.4%-20.9%) on region-specific questions, far below practical utility levels.
*   RA-MCQ reveals systematic bias in all models: Central and North Indian responses are over-selected (30-40% higher than expected), while East and West are underrepresented.
*   Even in domains like Education with national curricula, regional practice differences remain significant (only 13.8% all-India consensus).
*   The Festivals/Rituals domain shows the greatest variation (1.8% all-India consensus), reflecting strong regional traditions.

## Highlights & Insights

*   Systematically challenges the "nation = cultural monolith" assumption, opening a sub-national dimension for cultural NLP research.
*   The dual-task evaluation design (accuracy + implicit bias) provides a comprehensive framework for cultural representation assessment.
*   The question design methodology based on anthropology (OCM) is generalizable and can be transferred to any culturally diverse nation.

## Limitations & Future Work

*   Partitioning into five regions may be too coarse; significant diversity still exists within each region.
*   The sample size of 5 participants per region is relatively small.
*   The gold standard relies on subjective semantic equivalence judgments.
*   The study focuses only on India; the cross-national transferability of the methodology needs verification.

## Related Work & Insights

*   **vs CultureBank/CulturalBench**: These benchmarks evaluate cultural commonsense at the national level, whereas Indica delves into the sub-national level.
*   **vs Indian NLP Benchmarks**: Existing Indian benchmarks focus on textbook knowledge, while Indica focuses on daily cultural practices.
*   **vs CANDLE**: While CANDLE evaluates national cultural norms, Indica reveals internal cultural cleavages within a nation.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First sub-national cultural commonsense benchmark with a unique and important perspective.
*   Experimental Thoroughness: ⭐⭐⭐⭐ 8 models, dual-task evaluation, strict gold standards, though the human sample size is small.
*   Writing Quality: ⭐⭐⭐⭐⭐ Thought-provoking motivation with detailed data analysis.
*   Value: ⭐⭐⭐⭐⭐ Significant implications for research on cultural AI and LLM fairness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[AAAI 2026\] Towards a Common Framework for Autoformalization](../../AAAI2026/llm_evaluation/towards_a_common_framework_for_autoformalization.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)

</div>

<!-- RELATED:END -->
