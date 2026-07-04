---
title: >-
  [Paper Note] STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection
description: >-
  [ACL 2025][Social Computing][Hate speech] The paper constructs the first Chinese span-level hate speech detection dataset, STATE ToxiCN (comprising 8,029 posts and 9,533 quadruple annotations), introduces a Target-Argument-Hateful-Group quadruple annotation framework, and establishes the first Chinese hateful slang annotation dictionary (830 items). It systematically evaluates the performance of various LLMs on span-level Chinese hate speech detection.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Hate speech"
  - "fine-grained annotation"
  - "Chinese"
  - "Target-Argument extraction"
  - "hateful slang"
date: 2026-05-08
content_hash: 7960136fb1a3bf3b
---

# STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection

**Conference**: ACL 2025  
**arXiv**: [2501.15451](https://arxiv.org/abs/2501.15451)  
**Code**: [Yes](https://github.com/shenmeyemeifashengguo/STATE-ToxiCN)  
**Area**: NLP / Hate Speech Detection / Chinese NLP  
**Keywords**: Hate speech, fine-grained annotation, Chinese, Target-Argument extraction, hateful slang

## TL;DR

The paper constructs the first Chinese span-level hate speech detection dataset, STATE ToxiCN (comprising 8,029 posts and 9,533 quadruple annotations), introduces a Target-Argument-Hateful-Group quadruple annotation framework, and establishes the first Chinese hateful slang annotation dictionary (830 items). It systematically evaluates the performance of various LLMs on span-level Chinese hate speech detection.

## Background & Motivation

There are two critical gaps in the field of hate speech detection:

**Lack of Chinese span-level detection resources**: Existing Chinese hate speech datasets (such as COLD, SWSR, ToxiCN) remain at the post level, lacking fine-grained span-level annotations. However, the intensity and toxicity direction of hate speech are closely tied to its Target and Argument.

**Research gap in Chinese hateful slang**: As an ideographic language, Chinese features rich synonyms and near-synonyms, giving rise to diverse forms of hateful slang—including homophonic substitutions, character splitting/merging, and historical allusions—which pose severe challenges to model detection.

Specific challenges for Chinese span-level detection:
- Flexible word order (e.g., inversion changing the subject-verb-object structure)
- Lack of spaces as word delimiters
- Disguised nature of hateful slang

## Method

### Overall Architecture

The contributions of this paper include two main components:
1. STATE ToxiCN Dataset: Span-level quadruple annotations
2. Chinese Hateful Slang Annotation Dictionary

### Key Designs

1. **Quadruple Annotation Scheme (Target-Argument-Hateful-Group)**

   | Component | Meaning | Example |
   |------|------|------|
   | Target | Target of hate speech | "Gay people" |
   | Argument | Arguments supporting the hatred | "High-incidence group of AIDS" |
   | Hateful | Whether it constitutes hate | hate / non-hate |
   | Group | Target group classification | LGBTQ, others |

   Design Motivation: Compared to the triplets in the TBO dataset, a Group dimension is added to support group-level analysis.

2. **Data Sourcing and Filtering**

    - Perform span-level annotation based on the ToxiCN post-level dataset.
    - Filter low-quality content: Remove advertisements, random characters, and excessively short ($<5$ characters) or excessively long ($>500$ characters) texts.
    - Exclude posts lacking a clear Target-Argument structure.

3. **Annotation Process and Quality Control**

    - Diverse annotation team: Various genders, ages, ethnicities, regions, and educational backgrounds.
    - Independent annotation: Each text is annotated independently by at least two annotators.
    - Regular cross-validation: 20% of samples are re-annotated by other annotators.
    - Expert arbitration: Disagreements are decided by a team of domain experts.
    - IAA (Inter-Annotator Agreement): Target span $\kappa=0.65$, Argument span $\kappa=0.61$, Hateful $\kappa=0.68$, Group $\kappa=0.75$.

4. **Construction of the Chinese Hateful Slang Dictionary**

    - Collect commonly used hateful slang from real online forums (Zhihu, Baidu Tieba).
    - Annotate each slang term with: Frequently targeted group (Group) + Explanation.
    - Explanation details: Literal meaning, derived meaning, reasons for hatred towards the target group, and common usage patterns.
    - Contains a total of 830 annotated slang terms.
    - Dynamic maintenance via shared online documents.

5. **Disguise Manifestations of Hateful Slang**

    - Homophonic substitution
    - Character splitting and merging: e.g., "mo" (black dog) $\rightarrow$ racial discrimination
    - References to historical allusions
    - These techniques make traditional keyword-based methods difficult to detect.

### Loss & Training

In the evaluation experiments, open-source models are fine-tuned:
- A basic prompt is used (containing task definition, output format, and prediction requirements).
- Closed-source models are called via APIs and provided with 2 additional exemplars (1 hate + 1 non-hate).
- Macro-F1 is adopted as the primary evaluation metric.

## Key Experimental Results

### Dataset Statistics

| Metric | Value |
|------|------|
| Total Posts | 8,029 |
| Posts with Hateful Content | 4,942 (61.55%) |
| Total Quadruples | 9,533 |
| Hateful Quadruples | 6,063 (63.60%) |
| Sexism | 1,663 (17.44%) |
| Racism | 1,232 (12.92%) |
| Regional Discrimination | 1,323 (13.88%) |
| LGBTQ | 628 (6.59%) |
| Multi-group | 866 (9.08%) |
| Hateful Slang Dictionary | 830 items |

### Main Results (Comprehensive Evaluation of Span-level Detection, Macro-F1)

| Model | Target (Hard/Soft) | Argument (Hard/Soft) | Quadruple (Hard/Soft) |
|------|-------------------|---------------------|-------------------|
| mT5-base | 59.15/70.55 | 28.63/67.03 | 16.60/38.61 |
| LLaMA3-8B | 64.07/73.74 | 36.72/70.82 | 24.27/46.08 |
| Qwen2.5-7B | 63.96/74.64 | 35.42/70.36 | 23.70/47.03 |
| ShieldGemma-9B | 63.40/74.31 | 34.40/71.11 | 23.49/47.14 |
| GPT-4o | — | — | — |
| Claude-3.5-Sonnet | — | — | — |

### Closed-source LLM Performance (API, 2-shot)

| Model | Target (Hard) | Quadruple (Hard) |
|------|--------------|---------------|
| LLaMA3-70B | 30.54 | 3.69 |
| Qwen2.5-72B | 40.94 | 8.74 |
| Gemini-1.5-Pro | 29.80 | — |

### Key Findings

1. **Hard matching is far lower than Soft matching**: The ambiguity of Chinese span boundaries makes exact matching extremely difficult.
2. **Fine-tuned 7B models significantly outperform 70B+ closed-source LLMs**: This indicates that span-level extraction requires task-specific training.
3. **Argument extraction is much more difficult than Target extraction**: This is because arguments are expressed much more flexibly and diversely.
4. **The F1 score for full quadruple extraction is extremely low (Hard < 25%)**: This suggests that span-level Chinese hate speech detection remains highly challenging.
5. **Safety-aligned models (ShieldLM, ShieldGemma) show no advantages**: They are designed for post-level detection and are not directly applicable to span-level tasks.

## Highlights & Insights

1. **Filling a critical gap**: This work introduces the first Chinese span-level hate speech dataset and the first Chinese hateful slang dictionary with explainable annotations.
2. **Pragmatic quadruple design**: The Target-Argument-Hateful-Group structure captures both the directionality of hatred and enables group-level analysis.
3. **Rigorous annotation quality assurance**: The combination of diverse annotators, cross-validation, and expert arbitration guarantees long-term data quality.
4. **Valuable analysis of hateful slang**: The study reveals unique linguistic phenomena in Chinese hate speech (homophones, character splitting, etc.), guiding future detection system designs.

## Limitations & Future Work

1. The data scale is relatively small (8K posts), which might not cover all forms of Chinese hate speech.
2. Data is collected from only two platforms (Zhihu and Baidu Tieba), potentially introducing platform-specific biases.
3. Multimodal hate speech (combining text and images) is not yet explored.
4. Current evaluations mainly employ basic prompts, which may not fully exploit the capacity of LLMs.
5. The hateful slang dictionary requires continuous updates to incorporate emerging expressions.

## Related Work & Insights

- TBO (Zampieri et al., 2023): An English span-level hate speech dataset, pioneering the Target-Argument-Harmful triplet.
- ToxiCN (Lu et al., 2023): A Chinese post-level hate speech dataset, which STATE ToxiCN builds upon to scale down to the span level.
- HateXplain (Mathew et al., 2021): An English span-level dataset, but without Argument annotations.
- ToxiCloakCN (Xiao et al., 2024): Evaluates LLM robustness against disguised toxicity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First Chinese span-level dataset + first annotated Chinese hateful slang dictionary
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation of 12 models, multi-granularity metrics, and dual Hard/Soft matching
- **Writing Quality**: ⭐⭐⭐⭐ Detailed description of the dataset construction process and a clear annotation framework
- **Value**: ⭐⭐⭐⭐⭐ Provides a crucial foundational resource for the field of Chinese hate speech detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)
- [\[ACL 2025\] Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings](exploring_multimodal_challenges_in_toxic_chinese_detection_taxonomy_benchmark_an.md)
- [\[ACL 2025\] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](hateday_global_hate_speech.md)
- [\[ACL 2025\] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](../../ACL2026/social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)

</div>

<!-- RELATED:END -->
