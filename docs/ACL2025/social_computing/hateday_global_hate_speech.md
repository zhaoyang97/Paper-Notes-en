---
title: >-
  [Paper Note] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter
description: >-
  [ACL 2025][Social Computing][hate speech] HateDay constructs the first globally representative hate speech dataset—240k randomly sampled tweets covering 8 languages and 4 English-speaking countries. It reveals that academic datasets substantially overestimate the performance of detection models in real-world scenarios, particularly showing extremely poor detection capabilities for non-European languages.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "hate speech"
  - "dataset"
  - "Twitter"
  - "multilingual"
  - "content moderation"
date: 2026-05-08
content_hash: ded6fa6de8e2cbce
---

# HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter

**Conference**: ACL 2025  
**arXiv**: [2411.15462](https://arxiv.org/abs/2411.15462)  
**Code**: None (Dataset: [https://huggingface.co/datasets/manueltonneau/hateday](https://huggingface.co/datasets/manueltonneau/hateday))  
**Area**: NLP Understanding / Hate Speech Detection  
**Keywords**: hate speech, dataset, Twitter, multilingual, content moderation

## TL;DR
HateDay constructs the first globally representative hate speech dataset—240k randomly sampled tweets covering 8 languages and 4 English-speaking countries. It reveals that academic datasets substantially overestimate the performance of detection models in real-world scenarios, particularly showing extremely poor detection capabilities for non-European languages.

## Background & Motivation

**Background**: Hate speech detection is a critical content moderation task. Although academia has developed numerous detection models and datasets, they mainly focus on English, and existing datasets suffer from systematic biases (where class distributions and topic diversity mismatch real-world social media).

**Limitations of Prior Work**: (1) Significant distribution discrepancy between academic evaluation datasets and real-world social media—hate speech is extremely low-frequency in real scenarios (<2%), but artificially enriched in academic datasets; (2) Difficulties in cross-lingual/cross-national comparison—different datasets are constructed using inconsistent methodologies; (3) Existing work focuses on languages while neglecting regional variations within the same language (e.g., Indian vs. Nigerian vs. US English).

**Key Challenge**: Models performing well on biased datasets might be completely unusable when deployed in real-world scenarios.

**Goal**: To build the first globally representative dataset to realistically evaluate the actual performance of detection models on social media.

**Key Insight**: Utilizing the TwitterDay dataset (all 375 million tweets from 2022.09.21), performing random sampling by language/country coupled with human annotation.

**Core Idea**: For the first time, quantify the performance gap between academic evaluations and real-world scenarios using realistic and representative data, revealing that detection performance has been severely bloated.

## Method

### Overall Architecture
Data construction (TwitterDay 375M tweets → random sampling of 20k tweets for each of the 8 languages + 4 countries → 36 annotators labeling hateful/offensive/neutral) → Analysis of hate speech distribution → Evaluating the performance gap of SOTA detection models on HateDay vs. academic datasets → Analyzing reasons for model failures → Assessing the feasibility of content moderation.

### Key Designs

1. **Representative Sampling**: Random sampling from all tweets of a complete day ensures distributional authenticity—avoiding keyword filtering or event-driven collection.
2. **Multilingual + Cross-national**: 8 languages (Arabic, English, French, German, Indonesian, Portuguese, Spanish, Turkish) + 4 English-speaking countries (India, Kenya, Nigeria, USA).
3. **Three-level Annotation**: hateful / offensive / neutral, with targeted groups additionally annotated for hateful tweets.
4. **36 Annotators**, with 3 annotators recruited per language/country to maximize background diversity.

## Key Experimental Results

### Hate Speech Distribution

| Dimension | Hate Rate | Description |
|------|-------|------|
| Global Average | ~1-2% | Hate speech is extremely rare in real-world scenarios |
| Turkish | Highest (~3%) | Large variance across different languages |
| Indonesian | Lowest (<1%) | |
| Academic Datasets | 20-50% | Artificially enriched |

### Model Evaluation: Academic Datasets vs. HateDay

| Metric | Academic Datasets | HateDay (Real) | Gap |
|------|----------|-------------|------|
| F1 (English) | ~80%+ | Drastic decrease | Severely bloated |
| F1 (Non-European) | - | Extremely low | Virtually unusable |
| Precision | High | Low | Extremely high false positive rate |

### Key Findings
- **Academic evaluations drastically overestimate real-world performance**: Models performing "well" on academic datasets perform extremely poorly on HateDay.
- **Extremely poor detection capabilities for non-European languages**: Detection F1 scores for Arabic, Indonesian, and Turkish are far lower than those for English.
- **Models struggle to distinguish hate from offense**: A large volume of offensive but non-hateful content is mistakenly flagged as hateful.
- **Mismatch in target groups between academic datasets and real-world distributions**: Academic datasets over-focus on specific target groups while neglecting groups commonly targeted in real scenarios.
- **Fully automated moderation is unfeasible**: The false positive rate is too high; human-in-the-loop moderation is feasible but requires substantial labor.

## Highlights & Insights
- **Methodological value of "representative data"**: Randomly sampling from all tweets instead of relying on keyword searches represents the gold standard for constructing realistic evaluation benchmarks.
- **Cross-national analysis**: This work provides the first systematic comparison of hate speech variations within the same language (English) across different countries.
- **Practical deployment recommendations**: It explicitly concludes that "public models are unsuitable for automated moderation," directly impacting platform moderation strategies.

## Limitations & Future Work
- Only covers data from a single day, failing to capture event-driven surges in hate speech (e.g., post-political event escalations).
- Twitter-specific, which may not generalize to other platforms (Reddit, Facebook, etc.).
- Country inference relies on self-reported user locations, introducing potential errors.
- 240k instances remain limited for training scale.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first globally representative hate speech dataset, filling a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 240k instances across 8 languages and 4 countries, with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with impactful findings.
- Value: ⭐⭐⭐⭐⭐ Paradigm-shifting impact on the field of hate speech detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)
- [\[ACL 2025\] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)
- [\[ACL 2025\] STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection](state_toxicn_a_benchmark_for_span-level_target-aware_toxicity_extraction_in_chin.md)
- [\[ACL 2025\] Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](foreplay_polish_erotic_detection.md)
- [\[ACL 2026\] RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection](../../ACL2026/social_computing/rv-hate_reinforced_multi-module_voting_for_implicit_hate_speech_detection.md)

</div>

<!-- RELATED:END -->
