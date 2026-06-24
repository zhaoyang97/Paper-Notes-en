---
title: >-
  [Paper Note] Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse
description: >-
  [ACL 2025][Social Computing][content moderation] This work constructs forePLay (24,768 sentences, 5 categories), the first Polish erotic content detection dataset, and proposes a multidimensional annotation framework covering ambiguity, violence, and socially unacceptable behaviors. Evaluation results show that language-specific Polish models significantly outperform multilingual models, with Transformer encoder models demonstrating the strongest performance in handling unbal…
tags:
  - "ACL 2025"
  - "Social Computing"
  - "content moderation"
  - "erotic content detection"
  - "Polish NLP"
  - "annotation"
  - "dataset"
date: 2026-05-08
content_hash: 117d2a41ef9e3124
---

# Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse

**Conference**: ACL 2025  
**arXiv**: [2412.17533](https://arxiv.org/abs/2412.17533)  
**Code**: [GitHub](https://github.com/ZILiAT-NASK/ForePLay)  
**Area**: Other  
**Keywords**: content moderation, erotic content detection, Polish NLP, annotation, dataset

## TL;DR
This work constructs forePLay (24,768 sentences, 5 categories), the first Polish erotic content detection dataset, and proposes a multidimensional annotation framework covering ambiguity, violence, and socially unacceptable behaviors. Evaluation results show that language-specific Polish models significantly outperform multilingual models, with Transformer encoder models demonstrating the strongest performance in handling unbalanced categories.

## Background & Motivation
**Background**: The demand for online content moderation has surged, but existing tools are primarily designed for English and show limited effectiveness for morphologically rich languages like Polish.

**Limitations of Prior Work**: (a) Erotic content detection datasets are scarce and mostly in English; (b) Existing datasets mostly rely on simple binary classification, failing to capture fine-grained categories such as eroticism, violence, and socially unacceptable behaviors; (c) Safety filters of LLMs perform poorly on non-English erotic content.

**Key Challenge**: The expression of erotic content in morphologically complex languages like Polish is diverse, making simple binary classification schemes and English-centric tools ineffective for detection.

**Goal**: To provide a high-quality annotated dataset and model benchmarks for automatic erotic content detection in Polish.

**Key Insight**: Sampling from online fiction and Polish literary works, the authors design a 5-class mutually exclusive annotation scheme: erotic/ambiguous/violence/unacceptable/neutral.

**Core Idea**: Utilizing a multidocument-annotated Polish erotic content dataset to reveal the advantages of language-specific models in content moderation.

## Method

### Overall Architecture
Data sources: 69% online amateur fiction + 31% Polish literary works (including LGBTQ+ narratives), totaling 905 text units. Annotation: 6 annotators (3 male, 3 female), with independent annotation of 3 annotators per sentence + majority voting + super-annotator arbitration. Model evaluation: fine-tuning of encoder models (HerBERT, RoBERTa) + Polish LLMs (PLLuM series, Bielik) + general LLMs (GPT-4o, Llama-3.1, etc.) zero/few-shot.

### Key Designs
1. **Five-class Annotation Scheme**:

    - erotic (25.68%): descriptions of sexual activity, explicit erotic suggestions
    - ambiguous (5.43%): content that may trigger sexual associations in specific contexts but is neutral in itself
    - violence (0.28%): sexual harassment/rape/non-consensual violence
    - unacceptable (0.47%): illegal/taboo behaviors such as pedophilia, bestiality, incest, etc.
    - neutral (68.14%): other content
    - Hierarchy rule: unacceptable > violence > erotic > ambiguous > neutral

2. **Annotation Quality Control**:

    - Cohen's Kappa was mostly in the range of 0.66–0.72 (Krippendorff's Alpha = 0.716 after excluding the outlier annotator Fem1)
    - A maximum of 2 stories per author to avoid overfitting to specific writing styles

### Four Dataset Configurations
- Basic: binary classification (neutral/erotic)
- Core: 3-class classification (+ambiguous)
- Extended: 4-class classification (+merged violence+unacceptable)
- Full: complete 5-class classification

## Key Experimental Results

### Encoder Models (Fine-tuning)

| Model | Basic F1 | Core F1 | Full F1 |
|------|----------|---------|---------|
| HerBERT-Large | 0.939 | 0.738 | 0.648 |
| RoBERTa-Base | 0.944 | 0.738 | 0.707 |
| RoBERTa-Large | 0.943 | 0.748 | 0.664 |

### LLM Zero-shot Comparison

| Model | Basic F1 | Core F1 | Full F1 |
|------|----------|---------|---------|
| GPT-4o (0-shot) | 0.888 | 0.640 | 0.340 |
| PLLuM-Mistral-12B | 0.894 | 0.656 | 0.401 |
| PLLuM-Mixtral-8x7B | 0.874 | 0.647 | - |
| Bielik-11B (5-shot) | 0.868 | 0.607 | 0.480 |

### Key Findings
- Language-specific Polish encoder models consistently outperform general multilingual LLMs (RoBERTa-Base Full F1=0.707 vs GPT-4o=0.340).
- As classification granularity increases (Basic $\rightarrow$ Full), the performance of all models drops significantly.
- The ambiguous category has the lowest annotation consistency and remains the primary detection bottleneck.
- Polish-specific LLMs (PLLuM) outperform general LLMs with the same architecture in zero-shot settings.

## Highlights & Insights
- The multidimensional annotation scheme (particularly the distinction of ambiguous and violence/unacceptable) is closer to real-world content moderation requirements than simple binary classification.
- It reveals that fine-tuning small models in language-specific content moderation can significantly outperform large-scale general LLMs.
- The dataset intentionally covers LGBTQ+ content, avoiding systematic omission of such narratives.

## Limitations & Future Work
- Samples in the violence and socially unacceptable categories are extremely sparse ($< 1\%$), making it difficult for models to learn.
- It features sentence-level annotation only, lacking document-level context.
- It is limited to Polish; while the methodology is transferable, the data itself is not directly applicable to other languages.
- Inter-annotator variation (HLV) remains high on the ambiguous category.

## Related Work & Insights
- **vs Jigsaw/BeaverTails**: English-centric + binary classification, while forePLay provides Polish + 5-class classification.
- **vs CENSORCHAT**: Designed for dialog system monitoring, whereas forePLay is tailored for text content detection.
- **vs Llama Guard**: General safety classifiers, whereas this work proves that language-specific models are superior in non-English scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The first Polish erotic content detection dataset with a distinctive annotation scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison among encoders, LLMs, open-source, and closed-source models.
- Writing Quality: ⭐⭐⭐⭐ Detailed annotation pipeline description and thorough ethical considerations.
- Value: ⭐⭐⭐ Directly useful for the Polish NLP community, though the cross-lingual transferability of the method is general.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](hateday_global_hate_speech.md)
- [\[ACL 2025\] BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[NeurIPS 2025\] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](../../NeurIPS2025/social_computing/averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)
- [\[ACL 2026\] Understanding the Sociocultural Dimensions of Mental Health Discourse in Arabic-Language X Communities](../../ACL2026/social_computing/understanding_the_sociocultural_dimensions_of_mental_health_discourse_in_arabic-.md)

</div>

<!-- RELATED:END -->
