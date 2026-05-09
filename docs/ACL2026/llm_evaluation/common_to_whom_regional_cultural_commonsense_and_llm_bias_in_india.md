---
title: >-
  [Paper Note] Common to Whom? Regional Cultural Commonsense and LLM Bias in India
description: >-
  [ACL 2026][LLM Evaluation][Cultural commonsense] This paper introduces Indica, the first benchmark for evaluating LLM performance on sub-national cultural commonsense, focusing on cultural differences across five regions of India in eight domains of everyday life. Only 39.4% of questions reach consensus across all five regions, and all evaluated LLMs exhibit geographic bias—systematically over-selecting Central and North India as the "default" cultural representative.
tags:
  - ACL 2026
  - LLM Evaluation
  - Cultural commonsense
  - regional bias
  - Indian cultural diversity
  - benchmark construction
  - LLM bias
date: 2026-05-08
content_hash: 52022e004d6f3adf
---

# Common to Whom? Regional Cultural Commonsense and LLM Bias in India

**Conference**: ACL 2026
**arXiv**: [2601.15550](https://arxiv.org/abs/2601.15550)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: Cultural commonsense, regional bias, Indian cultural diversity, benchmark construction, LLM bias

## TL;DR

This paper introduces Indica, the first benchmark for evaluating LLM performance on sub-national cultural commonsense, focusing on cultural differences across five regions of India in eight domains of everyday life. Only 39.4% of questions reach consensus across all five regions, and all evaluated LLMs exhibit geographic bias—systematically over-selecting Central and North India as the "default" cultural representative.

## Background & Motivation

**Background**: Cultural commonsense benchmarks (e.g., CultureBank, CulturalBench) have begun addressing cross-cultural variation, but treat nations as culturally monolithic entities, assuming uniform cultural practices within a country.

**Limitations of Prior Work**: (1) Existing benchmarks evaluate cultural commonsense at the national level, ignoring sub-national cultural diversity. (2) Existing Indian NLP benchmarks focus on factual knowledge from textbooks and examinations, treating Indian culture as a single homogeneous entity. (3) LLMs may exhibit systematic bias toward certain regions within culturally diverse countries, yet no tool exists to detect this.

**Key Challenge**: In a country like India—with 28 states, 8 union territories, and 22 official languages—"cultural commonsense" cannot be nationally uniform. Yet LLMs must implicitly make regional choices when generating cultural information, and these implicit choices may reflect geographic biases in training data.

**Goal**: (1) Quantify the degree of regional variation in Indian cultural commonsense. (2) Evaluate LLM accuracy on region-specific cultural knowledge. (3) Detect implicit regional bias in LLMs when geographic context is absent.

**Key Insight**: Eight everyday cultural domains are designed based on the Outline of Cultural Materials (OCM) anthropological taxonomy. Human-annotated answers are collected from five Indian regions to construct a region-specific cultural commonsense benchmark.

**Core Idea**: Cultural commonsense in a multicultural country is primarily regional rather than national; LLMs exhibit systematic geographic bias when handling such knowledge.

## Method

### Overall Architecture

The Indica construction pipeline proceeds as follows: (1) Eight cultural domains are selected based on OCM → 39 topics → 515 questions. (2) Five participants per region are recruited from five Indian regions (North, South, East, West, Central) to answer all questions (15,275 responses total). (3) A gold standard is established through three levels of consensus: intra-regional, inter-regional, and pan-regional.

### Key Designs

1. **OCM-Based Question Design**:

    - Function: Ensures questions cover key dimensions of everyday cultural practice.
    - Mechanism: Eight domains relevant to everyday cultural knowledge are selected from OCM's 90+ top-level categories (interpersonal relations, education, clothing, food, communication, finance, festivals and rituals, transportation behavior). For each domain, 2–4 non-overlapping sub-topics are chosen; GPT-4 assists in question generation, followed by human review.
    - Design Motivation: Ensures questions target everyday practices rather than institutional knowledge, with sufficient diversity to surface regional differences.

2. **Dual-Task Evaluation Design (RASA + RA-MCQ)**:

    - Function: Separately evaluates regional knowledge accuracy and implicit geographic bias.
    - Mechanism: RASA (Region-Anchored Short Answer)—given regional context (e.g., "In South India…"), tests the model's ability to generate accurate region-specific cultural knowledge. RA-MCQ (Region-Agnostic Multiple Choice)—removes geographic context and observes which region's cultural practices the model defaults to, revealing implicit bias.
    - Design Motivation: RASA tests knowledge; RA-MCQ tests bias—two complementary perspectives for a comprehensive evaluation of LLM cultural representation.

3. **Three-Level Consensus Gold Standard**:

    - Function: Establishes reliable region-specific cultural commonsense annotations.
    - Mechanism: Intra-regional consensus (≥4/5 participants' answers are semantically equivalent); inter-regional consensus (two regions' answers are fully consistent); pan-regional consensus (all five regions' answers agree). GPT-4o performs initial classification, followed by complete review by two human annotators.
    - Design Motivation: Rigorous consensus criteria ensure the gold standard reflects genuine regional cultural practices rather than individual preferences.

### Loss & Training

Indica is an evaluation benchmark and does not involve model training. Evaluation uses Gemini 3.0 Flash as the LLM judge; each question is run 30 times to mitigate randomness, and a chi-square goodness-of-fit test is applied to assess the statistical significance of bias.

## Key Experimental Results

### Main Results

**RASA Region-Specific Knowledge Accuracy (%)**

| Model | North | South | East | West | Central | Avg. |
|-------|-------|-------|------|------|---------|------|
| GPT-4o | ~20 | ~19 | ~15 | ~18 | ~20 | 20.9 |
| Claude 3.5 | ~19 | ~18 | ~14 | ~17 | ~19 | 19.3 |
| Lowest model | - | - | - | - | - | 13.4 |

### Ablation Study

| Analysis Dimension | Finding |
|--------------------|---------|
| Pan-regional consensus rate | Only 39.4% of questions reach consensus across all regions |
| Domain-level variation | Transportation behavior is highest (22.6%); festivals and rituals is lowest (1.8%) |
| Regional pair bias | North–Central is highest (68.3%); South–East is lowest (60.1%) |

### Key Findings

- Only 39.4% of questions have a consensus answer across all five regions—cultural commonsense in India is primarily regional.
- All 8 LLMs achieve only 13.4%–20.9% accuracy on region-specific questions, far below a usable level.
- RA-MCQ reveals systematic bias in all models: responses associated with Central and North India are over-selected (30–40% above expectation), while East and West are underrepresented.
- Even in domains such as education, which has a nationally unified curriculum, regional practice variation remains significant (only 13.8% pan-regional consensus).
- The festivals and rituals domain exhibits the greatest divergence (1.8% pan-regional consensus), reflecting strong regional traditions.

## Highlights & Insights

- This work is the first to systematically challenge the "nation = cultural monolith" assumption, opening a sub-national dimension for cultural NLP research.
- The dual-task evaluation design (knowledge accuracy + implicit bias) provides a comprehensive framework for assessing cultural representation in LLMs.
- The OCM-based question design methodology is generalizable and transferable to any culturally diverse country.

## Limitations & Future Work

- The five-region partition may be too coarse; significant diversity remains within each region.
- Only five participants per region represents a relatively small sample size.
- Gold standard construction relies on subjective judgments of semantic equivalence.
- The study focuses exclusively on India; cross-national transferability of the methodology requires validation.

## Related Work & Insights

- **vs. CultureBank/CulturalBench**: These benchmarks evaluate cultural commonsense at the national level; Indica is the first to descend to the sub-national level.
- **vs. Indian NLP benchmarks**: Existing Indian benchmarks focus on textbook knowledge; Indica focuses on everyday cultural practices.
- **vs. CANDLE**: CANDLE evaluates national-level cultural norms; Indica reveals cultural fragmentation within a nation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first sub-national cultural commonsense benchmark, with a distinctive and important perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Eight models, dual-task evaluation, and rigorous gold standard construction, though sample size is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Compelling motivation and thorough data analysis.
- Value: ⭐⭐⭐⭐⭐ Significant implications for cultural AI and LLM fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[AAAI 2026\] Towards a Common Framework for Autoformalization](../../AAAI2026/llm_evaluation/towards_a_common_framework_for_autoformalization.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Revisiting the Uniform Information Density Hypothesis in LLM Reasoning](revisiting_the_uniform_information_density_hypothesis_in_llm_reasoning.md)

</div>

<!-- RELATED:END -->
