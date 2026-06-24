---
title: >-
  [Paper Note] Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts
description: >-
  [ACL 2025][Multimodal VLM][emotion detection] This paper systematically compares three strategies for collecting author-annotated emotion data (creation, donation, recent posts). It reveals that research-created data exhibits significant differences from real-world data in text length, emotional prototypicality, and image-text relations. However, created data remains effective for training generalizable models, whereas real-world data is indispensable for accurate model evalu…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "emotion detection"
  - "data collection"
  - "multimodal"
  - "social media"
  - "author annotation"
date: 2026-05-08
content_hash: 8bd1e00100095757
---

# Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts

**Conference**: ACL 2025  
**arXiv**: [2505.24427](https://arxiv.org/abs/2505.24427)  
**Code**: [https://www.uni-bamberg.de/en/nlproc/projects/item/](https://www.uni-bamberg.de/en/nlproc/projects/item/)  
**Area**: Multimodal VLM  
**Keywords**: emotion detection, data collection, multimodal, social media, author annotation

## TL;DR
This paper systematically compares three strategies for collecting author-annotated emotion data (creation, donation, recent posts). It reveals that research-created data exhibits significant differences from real-world data in text length, emotional prototypicality, and image-text relations. However, created data remains effective for training generalizable models, whereas real-world data is indispensable for accurate model evaluation.

## Background & Motivation
1. **Background**: Emotion analysis requires author-annotated data to accurately model subjective emotional expressions. Current mainstream practices follow two approaches: asking participants to "create" (Creation) content matching specific emotional labels, or asking participants to "donate" (Donation) and annotate real social media posts.
2. **Limitations of Prior Work**: Data creation is simple to implement and poses low privacy risks, but the generated content may have systematic differences from real social media posts. Although donated data is more realistic, it suffers from privacy concerns and self-filtering by participants. There is currently a lack of systematic analysis on the differences between these two collection strategies.
3. **Key Challenge**: There is a fundamental trade-off between the convenience of data collection and data authenticity—researchers remain unclear about how much created data differs from real data, and how these differences affect downstream models.
4. **Goal**: (1) What specific differences exist between created and real data in terms of content, annotation, and sample characteristics? (2) How do these differences impact emotion classification models? (3) Do different collection strategies lead to demographic differences among participants?
5. **Key Insight**: The authors design three collection strategies (Creation, Donation, Recent) to collect multimodal social media posts and compare them across multiple dimensions (textual features, image styles, cognitive appraisals, participant demographics, and model performance).
6. **Core Idea**: To reveal the impacts of data collection strategies on emotion data quality and model generalization through rigorous comparative experiments.

## Method

### Overall Architecture
The authors design a large-scale comparative experiment: recruiting 522 participants to collect a total of 2,507 multimodal social media posts (text + image) via three different data collection strategies. Each post features author-annotated emotional labels and detailed metadata (cognitive appraisal, image-text relations, etc.). The differences among strategies and their impact on models are then systematically analyzed across five dimensions.

### Key Designs

1. **Three Data Collection Strategies**:

    - **Function**: To acquire author-annotated emotion data in different manners.
    - **Mechanism**: Creation requires participants to recall a certain emotional event and create a social media post (with images selected from a Flickr database); Donation requires participants to retrieve and submit real posts from their own social media accounts that match a specified emotion; Recent requires participants to submit their 5 most recent posts and freely annotate their emotions (without pre-specified emotion categories).
    - **Design Motivation**: The three strategies highlight different trade-offs regarding data authenticity, privacy protection, and class balance. A comparative study can reveal their respective advantages and disadvantages.

2. **Multi-dimensional Metadata Annotation System**:

    - **Function**: To collect rich contextual information of posts to support in-depth analysis.
    - **Mechanism**: Annotation for each post includes emotional intensity (1-5 Likert scale), image-text relation (5-point scale), cognitive appraisal of the event (15 psychological dimensions such as predictability, self-control, etc.), event duration, and emotion duration. Demographic information of participants is also collected.
    - **Design Motivation**: Simply comparing textual features is insufficient to uncover the root causes of the differences. It is necessary to understand how participants select and express emotional events from psychological dimensions.

3. **Cross-Strategy Model Train-Test Matrix**:

    - **Function**: To quantify the actual impact of data differences on model performance.
    - **Mechanism**: Creation and Donation are each split into training sets (800 posts) and test sets (300 posts). Single-modal models (RoBERTa for text, ViT for images), multimodal models (CLIP dual-encoders + classification head), and zero-shot VLMs (llama3.2-vision, llava-llama3, etc.) are used to construct a Train $\times$ Test cross-evaluation matrix.
    - **Design Motivation**: To directly assess cross-strategy generalization capability, verifying whether created data is sufficient to train models that perform well on real-world data.

### Loss & Training
Models are fine-tuned separately on Creation and Donation using cross-entropy loss for 6-class emotion classification. Each configuration is repeated 5 times to compute the average. Zero-shot models are prompted 5 times per prediction and averaged.

## Key Experimental Results

### Main Results

| Train Data | Test Data | Visual (V) F1 | Textual (T) F1 | Multimodal (T+V) F1 | Zero-shot F1 |
|---------|---------|-----------|-----------|--------------|---------|
| Donation | Creation | .16 | .49 | .60 | .24/.61/.56 |
| Creation | Creation | .18 | .58 | .62 | - |
| Donation | Donation | .19 | .41 | .50 | .19/.45/.46 |
| Creation | Donation | .18 | .42 | .50 | - |

### Data Difference Analysis

| Dimension | Creation vs Donation/Recent | Statistical Significance |
|-----------|---------------------------|--------------------------|
| Text Length | Creation is 51% longer than Recent, and 26% longer than Donation | p < 0.01 |
| Event Intensity | Creation is 0.34 points higher than Donation (5-point scale) | p < 0.001 |
| Emotional Response Intensity | Creation is 0.36 points higher than Donation | p < 0.001 |
| Participation Refusal Rate | Donation/Recent is far higher than Creation | p < 0.001 |
| Screenshot-type Images | Creation has almost none, Donation/Recent has more | χ² p < 0.001 |

### Key Findings
- Emotional events in the Creation data are more "prototypical"—participants tend to select typical events with high emotional intensity and long durations, whereas emotional triggers in real-world posts are more diverse.
- The three strategies result in participant samples that differ significantly in age, student ratio, and racial composition, and the refusal rate for Donation/Recent is much higher than for Creation.
- Models trained on Creation data can generalize to the Donation test set (with comparable F1 scores), but **when testing on Donation, model F1 scores are generally lower than when testing on Creation** (.50 vs .62), showing that real-world data is inherently more challenging.
- Multi-emotion posts are highly prevalent: most anger, fear, and disgust posts contain other emotions simultaneously, implying that annotations in Donation might be biased by target emotion prompting.
- Zero-shot multimodal models perform extremely poorly on the image modality (F1 only .19-.24), indicating that current VLMs have limited capacity in emotional visual understanding.

## Highlights & Insights
- The cross-design of the three strategies is highly ingenious: Creation controls privacy + balances emotion, Donation provides authenticity + balances emotion, and Recent provides authenticity + eliminates prompting bias. Their complementarity reveals issues across different dimensions.
- The conclusion "training on created data generalizes, but evaluation must use real data" has broad practical guidance, applicable to any NLP task involving subjective annotations.
- The introduction of cognitive appraisal dimensions from psychology ensures that the difference analysis goes beyond surface-level features, probing deep into the levels of cognitive processing.

## Limitations & Future Work
- The study is limited to English and UK/Ireland cultural backgrounds; cross-cultural generalizability remains unknown.
- The Recent dataset is small and heavily biased towards joy (79%), making it unsuitable for training and testing.
- Images were only selected from Flickr (Creation), which fails to reflect all types of real social media images (e.g., screenshots, memes).
- The study does not explore how to mix data collected from different strategies to train more robust models.
- In Creation, participants selected images from Flickr instead of uploading their own, which may introduce additional bias (as database image styles are uniform).
- Emotion classification only covers Ekman's 6 basic emotions, leaving more fine-grained emotion taxonomies unaddressed.

## Related Work & Insights
- **vs Troiano et al. (2023)**: While the former only collected textual created data, this work extends to multimodal context and adds comparisons with real-world data, revealing key differences in image-text relations.
- **vs Oprea & Magdy (2020)**: They explored data donation methods for irony detection, whereas this work systematically compares the advantages and disadvantages of donation versus creation, finding significant differences in participant demographics as well.
- **vs Kajiwara et al. (2021)**: Also utilizing author annotation, but this work introduces the recent posts strategy to eliminate emotion prompting bias.
- The methodological framework of this paper (collection + multi-dimensional analysis + cross train-test evaluation) can be directly transferred to other subjective annotation tasks, such as stance detection and hate speech annotation.

## Rating
- Overall Evaluation: A solid yet important empirical study with direct guiding significance for future emotion corpus construction.
- **Novelty**: ⭐⭐⭐⭐ The experimental design systematically comparing three strategies is novel and comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The analysis across five dimensions is thorough, and the statistical testing is rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and rich with tables and statistics.
- **Value**: ⭐⭐⭐⭐ High practical guidance value for emotion data collection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[ACL 2025\] AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)
- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ACL 2025\] Harnessing PDF Data for Improving Japanese Large Multimodal Models](harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](error-driven_data-efficient_large_multimodal_model_tuning.md)

</div>

<!-- RELATED:END -->
