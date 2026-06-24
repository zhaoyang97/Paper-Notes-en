---
title: >-
  [Paper Note] BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English
description: >-
  [ACL 2025][LLM Evaluation][Varieties of English] BESSTIE is constructed as the first annotated benchmark for sentiment analysis and sarcasm detection tailored to varieties of English (Australian, Indian, and British English). Evaluations using nine fine-tuned LLMs reveal that performance on Indian English (an outer-circle variety) is significantly worse than on inner-circle varieties, and cross-variety generalization remains limited.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Varieties of English"
  - "sentiment analysis"
  - "sarcasm detection"
  - "linguistic fairness"
  - "benchmark"
date: 2026-05-08
content_hash: 500c7a74296d2b6f
---

# BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English

**Conference**: ACL 2025  
**arXiv**: [2412.04726](https://arxiv.org/abs/2412.04726)  
**Code**: [https://huggingface.co/datasets/unswnlporg/BESSTIE](https://huggingface.co/datasets/unswnlporg/BESSTIE)  
**Area**: LLM Evaluation  
**Keywords**: Varieties of English, sentiment analysis, sarcasm detection, linguistic fairness, benchmark

## TL;DR
BESSTIE is constructed as the first annotated benchmark for sentiment analysis and sarcasm detection tailored to varieties of English (Australian, Indian, and British English). Evaluations using nine fine-tuned LLMs reveal that performance on Indian English (an outer-circle variety) is significantly worse than on inner-circle varieties, and cross-variety generalization remains limited.

## Background & Motivation
**Background**: NLP benchmarks are predominantly based on Standard American English, leaving the performance of LLMs on other varieties of English under-evaluated. Existing research indicates that LLMs possess biases against non-standard English varieties.

**Limitations of Prior Work**: There is a lack of annotated datasets for sentiment and sarcasm classification targeted at English varieties. Previous variety benchmarks (such as Multi-VALUE) relied on synthetic data, which fails to realistically reflect the lexical, spelling, and cultural-pragmatic nuances of different varieties.

**Key Challenge**: While LLMs excel in Standard English, is their capability comparable across English varieties? There is a lack of natural-text datasets for such evaluation.

**Goal**: To build a real-text sentiment and sarcasm annotation benchmark for three English varieties (en-AU, en-IN, and en-UK), and evaluate the fairness of LLMs across these varieties.

**Key Insight**: To collect data from two distinct sources—Google Places reviews (filtered by location) and Reddit comments (filtered by topic)—and annotate sentiment and sarcasm using native annotators.

**Core Idea**: To construct an English variety benchmark using natural variety text rather than synthetic data, thereby revealing the systemic unfairness of LLMs toward outer-circle English.

## Method

### Overall Architecture
Data Collection → Quality Assessment (Human + Automatic Variety Verification) → Native Annotation → Evaluation of 9 Fine-Tuned LLMs

### Key Designs

1. **Dual-source Data Collection**:

    - Google Places reviews: Filtered by location (cities in the three countries) and tourism reviews are excluded; 2-star and 4-star reviews (non-extreme) are selected to increase complexity.
    - Reddit comments: Filtered by topic (selecting 3-4 local subreddits for each variety).
    - FastText language detection + human variety annotation + automatic variety classifier verification.

2. **Data Quality Verification**:

    - Human annotation variety identification: en-IN was determined to be the easiest to identify (46 agreements), indicating that this variety has the most distinctive features.
    - Fine-grained ICE-Corpora automatic classifier: DistilBERT fine-tuned on the ICE corpus is used to verify the variety attribution of the samples.

3. **Dual-task Annotation**:

    - One native annotator for each variety annotates sentiment (positive/negative) and sarcasm (yes/no).
    - Verification by an additional independent annotator: sentiment $\kappa=0.61-0.79$, sarcasm $\kappa=0.47-0.63$.

### Evaluation Setup
- 6 encoder models (BERT-Large, RoBERTa, ALBERT, mBERT, mDistilBERT, XLM-R) + 3 decoder models (Gemma2-27B, Mistral-Small, Qwen2.5-72B).
- Full-precision fine-tuning for encoders and QLoRA fine-tuning for decoders.
- 30 epochs, batch size of 8, Adam optimizer, learning rate grid search.

## Key Experimental Results

### Main Results (Macro-averaged F-Score Averaged Across All Models)

| Task-Domain | en-AU | en-IN | en-UK |
|----------|-------|-------|-------|
| Google-Sentiment | **0.94** | 0.64 | 0.86 |
| Reddit-Sentiment | **0.78** | 0.69 | 0.78 |
| Reddit-Sarcasm | **0.62** | 0.56 | 0.58 |
| Overall Average | **0.78** | 0.63 | 0.74 |

### Ablation Study: Comparison of Model Types

| Attribute | Average Sentiment | Average Sarcasm |
|------|---------|---------|
| Encoder Models | Higher | Higher |
| Decoder Models | Lower | Lower |
| Monolingual Models | Slightly Better | Slightly Better (Google) |
| Multilingual Models | Slightly Worse | Slightly Better (Reddit) |

### Key Findings
- **Systemic Underperformance on en-IN**: Across all tasks and models, performance on Indian English is consistently the lowest, with an overall average of 0.63 compared to 0.78 for en-AU.
- Sarcasm detection is universally challenging (ranging from 0.56 to 0.62 across all varieties) because sarcasm heavily relies on localized, contemporary cultural contexts.
- Mistral performs best on sentiment classification (0.91 Google / 0.84 Reddit), while Gemma performs the worst.
- Encoder models outperform decoder models, which aligns with the natural advantage of encoders in sequence classification.
- Cross-variety generalization remains limited; training on one variety and transferring to another leads to a decline in performance.

## Highlights & Insights
- **Natural Text > Synthetic Text**: Language varieties are characterized not only by grammatical differences but also by spelling, lexicon, and cultural pragmatics, which synthetic generation methods fail to capture.
- **Increasing Difficulty via Moderate Ratings (2/4 Stars)**: This strategy avoids the artificially inflated performance associated with extreme labels, providing a more rigorous test of the models' true capabilities.
- **Systemic Disadvantage of en-IN**: Outer-circle English has been systematically neglected in NLP research, and BESSTIE provides quantitative evidence of this gap.

## Limitations & Future Work
- Only three English varieties are covered, excluding African Englishes, Singaporean English, etc.
- There is only one annotator per variety, which may introduce individual biases despite the validation process.
- The Reddit subset may not fully represent daily language use.
- Bias mitigation methods were not explored; the study diagnoses the issue without providing a solution.
- The work can be extended to more NLP tasks (e.g., NLI, QA).

## Related Work & Insights
- **vs Multi-VALUE**: Multi-VALUE generates synthetic variety data via syntactic transformations, whereas BESSTIE utilizes real-world text, which better reflects actual biases.
- **vs DialectBench**: DialectBench covers dialects across multiple languages but excludes English varieties; BESSTIE fills this gap.
- **vs Standard Benchmarks (e.g., SST-2)**: High performance on standard benchmarks does not guarantee capability on other English varieties, highlighting the need for variety-specific evaluations.
- Insight: Fairness evaluations should not only focus on demographic biases; linguistic variety bias is equally crucial.

## Rating
- Novelty: ⭐⭐⭐⭐ First dual-task annotated benchmark for sentiment and sarcasm across English varieties, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ In-depth evaluation with nine models across multiple sources, tasks, and varieties, backed by robust data quality verification.
- Writing Quality: ⭐⭐⭐⭐ Rigorous data construction process and sound experimental design.
- Value: ⭐⭐⭐⭐ Provides a foundational evaluation tool for research on linguistic fairness across English varieties.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](la_leaderboard_spanish.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](../../ACL2026/llm_evaluation/are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [\[ACL 2025\] MisMatched: A Benchmark for Scientific Natural Language Inference](a_mismatched_benchmark_for_scientific_natural_language_inference.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
