---
title: >-
  [Paper Note] BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages
description: >-
  [ACL 2025 (Best Resource Paper)][LLM (Other)][Emotion Recognition] This paper constructs BRIGHTER, a multi-label emotion annotation dataset covering 28 languages, with a focus on low-resource languages in Africa, Asia, Eastern Europe, and Latin America. Annotated by native speakers, it establishes benchmark experimental results on both monolingual and cross-lingual emotion recognition tasks.
tags:
  - "ACL 2025 (Best Resource Paper)"
  - "LLM (Other)"
  - "Emotion Recognition"
  - "Multilingual Dataset"
  - "Low-resource Languages"
  - "Multi-label Annotation"
  - "Cross-lingual Transfer"
date: 2026-05-08
content_hash: 019c261dad1a68de
---

# BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages

**Conference**: ACL 2025 (Best Resource Paper)  
**Link**: [ACL Anthology](https://aclanthology.org/2025.acl-long.436/)
**Code**: Yes (dataset publicly released)  
**Area**: NLP Understanding / Emotion Recognition / Multilingual NLP  
**Keywords**: Emotion Recognition, Multilingual Dataset, Low-resource Languages, Multi-label Annotation, Cross-lingual Transfer

## TL;DR

This paper constructs BRIGHTER, a multi-label emotion annotation dataset covering 28 languages, with a focus on low-resource languages in Africa, Asia, Eastern Europe, and Latin America. Annotated by native speakers, it establishes benchmark experimental results on both monolingual and cross-lingual emotion recognition tasks.

## Background & Motivation

**Background**: Textual emotion recognition is an important task in NLP, widely applied in dialogue systems, social media analysis, and mental health monitoring. Current emotion recognition research and datasets primarily focus on high-resource languages such as English, leading to a severe resource imbalance.

**Limitations of Prior Work**: Low-resource languages (e.g., Hausa and Yoruba in Africa, Urdu and Sinhalese in Asia, Romanian in Eastern Europe) suffer from a severe lack of high-quality emotion annotation datasets. Existing multilingual emotion datasets either have limited language coverage or rely on machine translation instead of native speaker annotations, which causes cultural nuances in emotional expression to be overlooked.

**Key Challenge**: Emotional expressions are strongly culture- and language-dependent—the way, intensity, and social norms of expressing emotions such as "anger" or "sadness" vary significantly across different languages. Relying on translations or proxy models from high-resource languages cannot truly solve the emotion understanding problem in low-resource languages.

**Goal**: (1) Create a multi-label emotion dataset annotated by native speakers for 28 languages; (2) Cover diverse text domains (social media, news, etc.); (3) Establish monolingual and cross-lingual emotion recognition benchmarks.

**Key Insight**: The authors formed a large-scale international collaborative team (40+ co-authors), where native researchers for each language took charge of data collection and annotation, ensuring annotation quality and cultural appropriateness.

**Core Idea**: Build a large-scale, multi-domain, multilingual emotion annotation dataset through international collaboration to fill the data gap in low-resource language emotion recognition, and evaluate the performance gaps of current models (including LLMs) in cross-lingual emotion recognition tasks on this basis.

## Method

### Overall Architecture

The construction of the BRIGHTER dataset follows the pipeline of "data collection $\rightarrow$ annotation scheme design $\rightarrow$ native speaker annotation $\rightarrow$ quality control $\rightarrow$ benchmark experiments". The inputs are raw texts from various text domains, and the outputs are data instances with multi-label emotion annotations. The emotional labels adopt Ekman's six basic emotions (anger, disgust, fear, joy, sadness, surprise) plus a "neutral" category.

### Key Designs

1. **Multilingual Data Collection and Domain Coverage**:

    - **Function**: Collect samples from multiple text domains for each language
    - **Mechanism**: Based on the actual digital content landscape of each language, collect text from sources such as Twitter/X, news comments, Reddit, and local forums. Instead of forcing all languages to use the same source, the selection is flexible based on the online ecosystem of each language.
    - **Design Motivation**: The distribution of digital content varies significantly across different languages, and a flexible collection strategy yields more natural and representative corpora.

2. **Multi-label Emotion Annotation Scheme**:

    - **Function**: Support annotating multiple emotions simultaneously for a single text instance
    - **Mechanism**: Each annotator labels all applicable emotion categories (selected from the seven classes) and can designate emotion intensity for each text. A multi-annotator mechanism is employed (at least 2-3 annotators per text), measuring annotation quality via inter-annotator agreement.
    - **Design Motivation**: Real-world texts often express multiple emotions simultaneously (e.g., "surprised and happy"). Single-label schemes would lose rich emotional information.

3. **Cross-lingual Experimental Framework**:

    - **Function**: Evaluate the effects of monolingual training, cross-lingual zero-shot transfer, and multilingual joint training
    - **Mechanism**: Use multilingual pre-trained models like XLM-RoBERTa for fine-tuning on monolingual datasets, zero-shot transferring to target languages after fine-tuning on English, and joint training on mixed multilingual data. Meanwhile, evaluate the performance of LLMs such as GPT-4 under zero-shot/few-shot settings.
    - **Design Motivation**: Comprehensively evaluate the actual capability of the current technical stack in low-resource language emotion recognition, revealing the performance gap caused by resource imbalance.

### Loss & Training

The benchmark experiments utilize a standard multi-label classification framework with a binary cross-entropy loss. Language-specific fine-tuning and multilingual joint fine-tuning are conducted on XLM-RoBERTa. LLM experiments use zero-shot and few-shot prompting.

## Key Experimental Results

### Main Results

| Language Group | Model | Avg F1 (Weighted) | Gap to English |
|---|---|---|---|
| High-resource (en/fr/zh) | XLM-R fine-tuned | ~72-78% | Baseline |
| Medium-resource (hi/ar) | XLM-R fine-tuned | ~62-70% | -8~15% |
| Low-resource (ha/yo/si) | XLM-R fine-tuned | ~48-58% | -20~30% |
| Cross-lingual zero-shot | XLM-R (en→target) | ~45-65% | Large variation |
| Zero-shot | GPT-4 | ~50-65% | Significant variation across languages |

### Ablation Study

| Configuration | Avg F1 | Note |
|---|---|---|
| Monolingual Fine-tuning | Optimal | Language-specific data performs best |
| Multilingual Joint Training | Close to Monolingual | Slight improvement for low-resource languages |
| Zero-shot Cross-lingual Transfer | Significant drop | Worse performance as linguistic distance increases |
| GPT-4 Zero-shot | Moderate | Unexpectedly poor performance on some low-resource languages |

### Key Findings
- There is a 20-30% F1 gap in emotion recognition performance between low-resource and high-resource languages, which is difficult to completely bridge even with state-of-the-art multilingual models.
- The zero-shot performance of LLMs (such as GPT-4) on low-resource languages is lower than expected, particularly on languages with non-Latin scripts.
- Emotion intensity prediction is more challenging than category recognition, with larger cross-lingual variations.
- Inter-annotator agreement varies significantly across languages, reflecting cultural differences in emotional expression.

## Highlights & Insights
- **The large-scale international collaborative data construction paradigm** is worth emulating. Local researchers taking responsibility for distributed annotation for each language ensures both scale and quality, and this model can be generalized to other multilingual NLP tasks.
- **The multi-label + intensity annotation scheme** is closer to real-world emotional expressions than traditional single-label schemes, providing richer annotation signals for subsequent research.
- **Winning the Best Resource Paper award** demonstrates that data infrastructure construction for low-resource languages is one of the core concerns of the current NLP community.

## Limitations & Future Work
- The data scale for some languages remains small (hundreds of instances), which might be insufficient for training highly robust models.
- The annotation scheme is based on Ekman's basic emotion theory, which may face applicability issues across different cultures (e.g., culture-specific emotion categories are not covered).
- Future work can explore utilizing the BRIGHTER dataset for more in-depth cross-cultural comparative studies on emotion.
- The dataset can be extended to conversational scenarios and code-switched texts.

## Related Work & Insights
- **vs GoEmotions**: GoEmotions provides fine-grained emotion annotations in English, but only covers English; BRIGHTER trades off fine-grained labels for linguistic breadth.
- **vs SemEval Emotion Tasks**: SemEval emotion tasks over the years have progressively increased language coverage but lack a unified annotation framework; BRIGHTER provides a unified approach.
- **vs AfriSenti**: AfriSenti focuses on sentiment analysis for African languages, while BRIGHTER can be viewed as its globalized extension.

## Rating
- Novelty: ⭐⭐⭐⭐ The dataset construction methodology itself is not highly novel, but the scale of covering 28 languages and the native-speaker annotation strategy holds significant value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Monolingual, cross-lingual, and various LLM settings are all covered, with quite comprehensive analyses.
- Writing Quality: ⭐⭐⭐⭐ Transparent structure; the discussion on data collection challenges is highly valuable.
- Value: ⭐⭐⭐⭐⭐ Well-deserved Best Resource Paper, filling an important data gap in multilingual emotion recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA](team_anotheroption_at_semeval-2025_task_8_bridging_the_gap_between_open-source_a.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [\[ACL 2025\] INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages](injongo_a_multicultural_intent_detection_and_slot-filling_dataset_for_16_african.md)

</div>

<!-- RELATED:END -->
