---
title: >-
  [Paper Note] AfroBench: How Good are Large Language Models on African Languages?
description: >-
  [ACL 2025][LLM (Other)][African languages] Ours proposes AfroBench—a comprehensive evaluation benchmark covering 64 African languages, 15 NLP tasks, and 22 datasets. Evaluating 12 LLMs reveals that the closed-source model (GPT-4o) outperforms the best open-source model (Gemma 2 27B) by approximately 12 points, yet all LLMs still lag behind fine-tuned baselines, and the performance gap with English exceeds 40 points on open-source models.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "African languages"
  - "low-resource languages"
  - "multilingual benchmarks"
  - "LLM evaluation"
  - "fairness"
date: 2026-05-08
content_hash: 3325751efbbc8b10
---

# AfroBench: How Good are Large Language Models on African Languages?

**Conference**: ACL 2025  
**arXiv**: [2311.07978](https://arxiv.org/abs/2311.07978)  
**Code**: [GitHub](https://mcgill-nlp.github.io/AfroBench/)  
**Area**: Multilingual NLP / LLM Evaluation  
**Keywords**: African languages, low-resource languages, multilingual benchmarks, LLM evaluation, fairness

## TL;DR

Ours proposes AfroBench—a comprehensive evaluation benchmark covering 64 African languages, 15 NLP tasks, and 22 datasets. Evaluating 12 LLMs reveals that the closed-source model (GPT-4o) outperforms the best open-source model (Gemma 2 27B) by approximately 12 points, yet all LLMs still lag behind fine-tuned baselines, and the performance gap with English exceeds 40 points on open-source models.

## Background & Motivation

**Background**: LLMs exhibit exceptional performance on high-resource languages, but their capabilities in low-resource languages like African languages are severely lacking, and evaluation is scarce. Existing multilingual benchmarks (e.g., MEGA, Megaverse) only cover 11–16 African languages and limited tasks.

**Limitations of Prior Work**: (a) African language datasets are scattered and difficult to discover; (b) evaluations cover too few languages with a narrow task focus (predominantly NER/POS); (c) high evaluation costs lead to incomplete model coverage; (d) LLMs iterate rapidly, yet there is a lack of platforms to continually track progress on African languages.

**Key Challenge**: Over 90% of the world's 7000+ languages are ignored by the NLP community, and there is an urgent need to quantify and narrow the NLP technology gap for African languages.

**Goal**: To construct the most comprehensive LLM evaluation benchmark for African languages and systematically reveal the capability boundaries of current LLMs on these languages.

## Method

### Overall Architecture

AfroBench aggregates 22 datasets covering 15 tasks (9 NLU + 6 NLG + 6 Knowledge/QA + 1 Mathematical Reasoning) across 64 African languages from 7 language families. All tasks are formulated as text generation problems and evaluated using multiple prompt templates.

### Key Designs

1. **Comprehensive Task Coverage**:

    - NLU: POS, NER, sentiment analysis, topic classification, intent classification, hate speech detection, NLI
    - NLG: Machine translation (4 datasets), summarization, automatic diacritic restoration (AfriADR, a new dataset)
    - Knowledge/QA: Cross-lingual QA, reading comprehension, MMLU, science QA
    - Reasoning: Mathematical reasoning (AfriMGSM)

2. **AfroBench-Lite**:

    - **Function**: Provides a lightweight version containing 14 representative languages and 7 tasks.
    - **Mechanism**: Language selection covers various resource levels and typological diversity (e.g., Swahili, Hausa, Amharic, Igbo, Yorùbá, etc.).
    - **Design Motivation**: Lowers evaluation costs and facilitates rapid leaderboard boarding for new models.

3. **New AfriADR Dataset**:

    - **Function**: Automatic diacritic restoration task covering 5 languages (Ghomálá', Fon, Igbo, Wolof, Yorùbá).
    - **Mechanism**: Strips all diacritics from sentences to use as input, requiring the model to restore the correct diacritics.
    - **Design Motivation**: Diacritics are crucial to the semantics of African languages, and this task is unfamiliar to LLMs.

## Key Experimental Results

### Main Results

Average scores of 12 LLMs across 15 tasks:

| Model | Overall Average | Gap vs. English |
|------|---------|-----------|
| GPT-4o | 59.6 | -25.5 |
| Gemini 1.5 pro | 58.5 | -24.1 |
| Gemma 2 27B | 47.7 | -32.9 |
| LLaMa 3.1 70B | 43.3 | -36.7 |
| Aya-101 13B | 40.1 | (N/A) |
| LLaMa 2 7B | 22.5 | (N/A) |
| Fine-tuned Baselines (AfroXLMR, etc.) | (Task-dependent) | (N/A) |

Performance of English vs. African languages on AfroBench-Lite:

| Model | English | African Languages | Gap |
|------|------|---------|------|
| GPT-4o | 85.1 | 66.0 | -19.1 |
| Gemma 2 27B | 80.6 | 43.5 | -37.1 |
| LLaMa 3.1 70B | 80.0 | 39.9 | -40.1 |

### Ablation Study

Few-shot performance (GPT-4o, 0-shot vs. 5-shot):

| Task | 0-shot | 5-shot | Gain |
|------|--------|--------|------|
| ADR (Diacritic Restoration) | 54.9 | 62.7 | +7.8 |
| Hate Speech | 63.5 | 69.3 | +5.8 |
| Math Reasoning | 49.8 | 54.7 | +4.9 |
| Summarization | 66.5 | 67.9 | +1.4 |

### Key Findings

1. **Closed-source vs. open-source gap is much wider than in English**: While the gap is only 2–5 points in English, it exceeds 12 points in African languages.
2. **Knowledge-intensive tasks show the largest gap**: Arc-Easy (+29.4), Math (+22.6), MMLU (+19.9).
3. **Performance is positively correlated with monolingual data size**: Swahili (with 2.4GB of monolingual data) performs best, while Wolof (5MB) performs worst.
4. **All LLMs still lag behind fine-tuned baselines by ~11.5 points**: This indicates that collecting annotated data for low-resource languages remains highly valuable.
5. **Prompt Sensitivity**: Gemini 1.5 Pro is the least sensitive to prompt variations, while smaller models (Gemma 2 9B) are the most sensitive.
6. **Few-shot prompting helps NLG and novel tasks (ADR) most**, while yielding minimal benefit for translation.

## Highlights & Insights

- **Unprecedented Scale**: Covers 64 African languages, 15 tasks, and 22 datasets, vastly exceeding prior evaluations on African languages.
- **Innovative Contribution of AfriADR**: Automatic diacritic restoration is an important task specific to African languages, where few-shot prompting offers substantial improvements.
- **Compelling Qualitative Analysis**: Demonstrates a massive disparity between 0-shot and 5-shot on Ghomálá' diacritic restoration (ChrF score jumped from 21.4 to 81.6) and shows how few-shot prompting helps models reason correctly in the target language for math.
- **Practical Value**: Establishes a continually updated leaderboard, with newly added models such as GPT-4.1, Gemini 2.0 Flash, and LLaMa 4.

## Limitations & Future Work

- Insufficient transparency in training data, making it impossible to evaluate data contamination.
- High evaluation costs (~$2500 each for GPT-4o and Gemini 1.5) limit model coverage.
- 60% of the languages appear in fewer than 5 datasets; the long-tail distribution restricts reliable evaluation for certain languages.
- Translation evaluation is limited by metrics like chrF, lacking high-quality COMET/AfriCOMET evaluations.

## Related Work & Insights

- **IrokoBench**: A concurrent work in ACL 2025 focusing on 16 African languages and 3 tasks; Ours offers broader coverage.
- **Belebele**: Covers 28 African languages but only supports QA tasks.
- **Insight**: The primary bottleneck for LLM capabilities in African languages lies in the scale of monolingual data rather than model architecture; investing in language resource construction is more critical than optimizing model designs.

## Rating

- **Novelty**: ⭐⭐⭐ Mainly a resource and evaluation contribution with limited methodological innovation (except AfriADR).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 64 languages, 12 models, 15 tasks, with detailed multidimensional analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and comprehensive analysis, though the abundance of tables increases the reading cognitive load.
- **Value**: ⭐⭐⭐⭐⭐ Fills a major gap in LLM evaluation for African languages; the continually updated leaderboard offers long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages](injongo_a_multicultural_intent_detection_and_slot-filling_dataset_for_16_african.md)
- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [\[ACL 2025\] Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)

</div>

<!-- RELATED:END -->
