---
title: >-
  [Paper Note] AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues
description: >-
  [ACL 2025][Multimodal VLM][Akan] This work constructs AkaCE—the first multimodal conversational emotion recognition dataset for an African language, covering Akan (the primary language of Ghana, with approximately 20 million speakers). It contains 385 dialogues with 6,162 utterances (spanning audio, visual, and text modalities), 308 speakers (gender-balanced with 155 males and 153 females), and provides the first word-level prosodic prominence annotations for an African langu…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Akan"
  - "Emotion Recognition in Conversation"
  - "Multimodal Dataset"
  - "African Languages"
  - "Prosodic Annotation"
  - "Tonal Languages"
date: 2026-05-08
content_hash: 6cdaff96dcd5796f
---

# AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues

**Conference**: ACL 2025  
**arXiv**: [2502.10973](https://arxiv.org/abs/2502.10973)  
**Code**: [GitHub](https://github.com/dsasu1/AkaCE)  
**Area**: Multimodal / Emotion Recognition / Low-Resource Languages  
**Keywords**: Akan, Emotion Recognition in Conversation, Multimodal Dataset, African Languages, Prosodic Annotation, Tonal Languages

## TL;DR
This work constructs AkaCE—the first multimodal conversational emotion recognition dataset for an African language, covering Akan (the primary language of Ghana, with approximately 20 million speakers). It contains 385 dialogues with 6,162 utterances (spanning audio, visual, and text modalities), 308 speakers (gender-balanced with 155 males and 153 females), and provides the first word-level prosodic prominence annotations for an African language.

## Background & Motivation
**Background**: Emotion Recognition in Conversation (ERC) relies heavily on multimodal datasets, but existing benchmarks are concentrated on high-resource languages—IEMOCAP/MSP-IMPROV/MELD cover English, while M³ED covers Chinese. There is currently no multimodal ERC dataset for any African language.

**Limitations of Prior Work**: Africa possesses approximately 3,000 languages and accounts for 18.3% of the global population, yet it is completely absent from the ERC domain. Furthermore, existing dataset emotion classification systems are based on Western cultural perspectives, and their cross-cultural applicability remains unverified.

**Key Challenge**: How to construct a high-quality multimodal conversational emotion recognition dataset for a low-resource African language? Akan is a tonal language where prosody is crucial for emotional expression, but it lacks both ASR systems and acoustic alignment models.

**Goal**: Construct the first multimodal ERC dataset for an African language, establish baselines, and validate dataset quality.

**Key Insight**: Dialogues were extracted from 21 Akan movies and manually transcribed (due to the lack of available ASR). Word-level prosodic prominence annotations were also incorporated, leveraging the unique characteristics of tonal languages.

**Core Idea**: AkaCE = the first multimodal ERC for an African language + the first prosodic annotation for an African language + a gender-balanced dataset.

## Method

### Dataset Construction

1. **Data Source Selection**:
    - 21 Akan movies (downloaded from the Internet Archive) underwent quality filtering based on: full-length movie, intelligible speech, and clear facial expressions.
    - 385 dialogues, 4,477 turns, and 6,162 utterances.

2. **Text and Speaker Annotation**:
    - **Manual Transcription**: Because existing Akan ASR systems exhibit high error rates, all speech was manually transcribed and timestamped by 7 annotators.
    - All transcriptions were proofread by professional Akan linguists.
    - Speakers were labeled with unique identifiers (based on order of appearance and gender).

3. **Emotion Annotation**:
    - 7 emotion categories: Sadness, Fear, Anger, Surprise, Disgust, Joy, Neutral (Ekman's 6 basic emotions + Neutral).
    - Standards were unified through a pre-training session, referencing the annotation guidelines from Gong et al. (2024).
    - Double annotation and majority voting were used to determine final labels, with disputes arbitrated by an external Akan emotion analysis expert.
    - Inter-annotator agreement yielded a Fleiss' $\kappa = 0.488$, which is comparable to MELD ($0.43$), IEMOCAP ($0.48$), and MSP-IMPROV ($0.49$).

4. **Prosodic Prominence Annotation**:
    - Two annotators listened to the audio and annotated whether each word carried prosodic prominence (1/0).
    - Inter-annotator agreement yielded a Fleiss' $\kappa = 1.0$ (perfect agreement)—likely because Akan, as a tonal language, features highly distinct prosodic cues (confirmed by Kügler & Genzel 2012, which established that Akan speakers mark prominence through consistent pitch patterns).
    - A total of 37,314 prominent words vs. 79,991 non-prominent words.

### Dataset Statistics

| Metric | Value |
|--------|------|
| Number of movies | 21 |
| Number of dialogues | 385 |
| Number of utterances | 6,162 |
| Speakers | 308 (155 male / 153 female) |
| Total words | 117,305 |
| Avg. words per utterance | 19 |
| Avg. seconds per utterance | 6.7s |
| Total duration | ~24.3 hours |

### Baseline Experimental Setup
- Data Split: 7:1.5:1.5 (train/val/test), consisting of 3,888 training, 816 validation, and 834 test instances.
- Unimodal: Text (BERT/XLM-R), Audio (wav2vec 2.0), Video (ResNet-50 for facial feature extraction).
- Multimodal Fusion: Feature concatenation + Transformer-based fusion.

## Key Experimental Results

### Emotion Distribution

| Emotion | Count | Percentage |
|------|------|------|
| Neutral | 2,941 | 47.7% |
| Anger | 1,107 | 18.0% |
| Sadness | 806 | 13.1% |
| Joy | 568 | 9.2% |
| Surprise | 364 | 5.9% |
| Disgust | 162 | 2.6% |
| Fear | 134 | 2.2% |

### Key Findings
- Multimodal fusion > unimodal: performance improves consistently across all classes after integrating the three modalities.
- **Audio is particularly important for tonal language ERC**: In Akan, the unimodal audio performance is close to that of the text modality, unlike English datasets where the text modality typically far outperforms audio.
- Prosodic annotations further improve emotion recognition performance.
- The severe imbalance in emotion distribution (47.7% Neutral, only 2.2% Fear) hurts minority class recognition.

## Highlights & Insights
- **Three "Firsts"**: The first multimodal ERC dataset for an African language + the first prosodic database for an African language + the first ERC dataset featuring tonal language characteristics.
- **Gender-Balanced Design (155 male / 153 female)** avoids gender bias and provides a solid foundation for fairness studies.
- **Perfect agreement in prosodic annotation** is a compelling finding—revealing a high level of consensus on prosodic prominence in tonal languages, in contrast to the moderate agreement typically found in English datasets.
- **Movies are a practical source of multimodal data for low-resource languages**—bypassing technical bottlenecks like ASR and acoustic alignment models.

## Limitations & Future Work
- **Limited to Akan as a single language** with a relatively small scale (6,162 instances)—could be scaled to other African languages.
- **Movie dialogues might not reflect daily conversation**: Played emotional expressions are often more exaggerated.
- **Cultural applicability of the Western Ekman emotion framework is not fully discussed**: Different cultures may interpret emotion categories differently. Although a 15-African-language study by Ahmad et al. (2025) used a similar categorization, an in-depth analysis of cultural variations is still missing.
- **Severe data imbalance**: The Fear and Disgust categories contain very few samples, which might require data augmentation or specific sampling strategies.

## Related Work & Insights
- **vs. IEMOCAP/MELD (English) / M³ED (Chinese)**: AkaCE fills the gap in African languages and uniquely provides prosodic annotations.
- **vs. Unimodal Datasets (EmoryNLP/DailyDialog)**: AkaCE provides audio+visual+text modalities, which is highly valuable for multimodal fusion research.
- Prosodic annotation of tonal languages offers unique research value for speech emotion analysis—and can be extended to other tonal languages such as Chinese.

## Rating
- Novelty: ⭐⭐⭐⭐ The first multimodal ERC + prosodic annotation for an African language, filling a significant gap.
- Experimental Thoroughness: ⭐⭐⭐ Baselines validate dataset usability, but the depth of analysis is limited.
- Writing Quality: ⭐⭐⭐⭐ Sufficient linguistic background with a transparent dataset construction process.
- Value: ⭐⭐⭐⭐⭐ High resource value for low-resource language NLP, promoting inclusive AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset](../../ACL2026/multimodal_vlm/draddp_a_multimodal_multi-party_dialogue_discourse_parsing_dataset.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)
- [\[ACL 2025\] iNews: A Multimodal Dataset for Modeling Personalized Affective Responses to News](inews_a_multimodal_dataset_for_modeling_personalized_affective_responses_to_news.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
