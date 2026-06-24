---
title: >-
  [Paper Note] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach
description: >-
  [Multimodal VLM] TikTalkCoref is proposed, the first multimodal coreference resolution dataset for Chinese social media dialogues (based on Douyin short videos), along with a pipeline benchmark containing three modules: textual coreference resolution, visual character tracking, and cross-modal alignment.
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: a4c66532976d025f
---

# Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach

## Basic Information

- **Conference**: ACL2025
- **arXiv**: [2504.14321](https://arxiv.org/abs/2504.14321)
- **Code**: GitHub (To be released)
- **Area**: Multimodal Vision-Language Model (Multimodal VLM)
- **Keywords**: Multimodal coreference resolution, Chinese social media, Douyin, video-text alignment, dataset construction

## TL;DR

TikTalkCoref is proposed, the first multimodal coreference resolution dataset for Chinese social media dialogues (based on Douyin short videos), along with a pipeline benchmark containing three modules: textual coreference resolution, visual character tracking, and cross-modal alignment.

## Background & Motivation

### Problem Definition
Multimodal Coreference Resolution (MCR) aims to identify mentions that refer to the same entity across textual and visual modalities, representing a key task for understanding multimodal content. For instance, in a dialogue, "Gillian Chung", "she", and "Ah Gil" point to the same person, and these textual mentions need to be grounded with the head regions of the corresponding character in the video frames.

### Limitations of Prior Work

**Scarcity of Data**: Existing MCR datasets are mainly focused on human-machine dialogues (such as J-CRe3), movie descriptions (such as MPII-MD), or image captions (such as CIN), which fail to adequately reflect the complexity and diversity of multimodal interactions in real social media.

**Language Coverage Biased towards English/Japanese**: Most studies focus on English, leaving Chinese MCR research virtually blank.

**Implicit Visual Cue Challenges**: In real social media dialogues, speakers often omit descriptions of the appearance or location of visible objects, making it difficult for models to extract visual cues from text to locate the mentioned objects.

### Goal
To construct a Chinese multimodal coreference resolution dataset based on real social media, filling the data gap in this field, and providing an effective benchmark.

## Method

### Overall Architecture
The system is a three-stage pipeline:
1. **Textual Coreference Resolution Module**: Extracts mentions from dialogue text and clusters them.
2. **Visual Character Tracking Module**: Detects character head regions in videos and tracks/clusters them.
3. **Text-Visual Coreference Alignment Module**: Associates textual clusters with visual clusters using contrastive learning.

### Dataset Construction (TikTalkCoref)

**Data Source**: Based on the TikTalk dataset (sourced from the Douyin platform), 4,000 samples were randomly selected from 367k dialogues, yielding 1,012 high-quality dialogues after manual filtering.

**Filtering Criteria**:
- Exclude samples containing personally identifiable information or sensitive content
- Exclude samples with severe video blur/noise or unrecognizable faces
- Exclude dialogues that do not involve character entity coreference

**Annotation Content**:
- Textual mention annotation (proper nouns 33.51%, pronouns 44.41%, common nouns 22.08%)
- Textual cluster annotation (including singletons)
- Annotation of character head bounding boxes (bboxes) in key video frames
- Classification annotation of celebrity/non-celebrity

**Annotation Process**: Independent double annotation by three annotators, followed by one expert arbitration, resulting in an inter-annotator agreement MUC of 78.19.

**Data Scale**:

| Metric | TikTalkCoref | TikTalkCoref-celeb |
|---|---|---|
| Number of Dialogues | 1,012 | 338 |
| Total Duration (min) | 519.65 | 158.33 |
| Number of Mentions | 2,179 | 731 |
| Number of Clusters | 1,435 | 488 |
| Number of Bboxes | 958 | 426 |

### Key Designs

**Textual Coreference Resolution**: Maverick model (a SOTA pipeline method) is adopted, based on the DeBERTa-Chinese-Large encoder.
- Mention Detection: Predicts start/end probabilities for each token, and candidate mentions are determined with a threshold > 0.5.
- Mention Clustering: A coarse-to-fine approach—first roughly filtering the top-$K$ antecedents with a bilinear scoring function, and then scoring precisely with a fully connected layer.

**Visual Character Tracking**:
- YOLOv5 head detection + DeepSORT cross-frame tracking
- MTCNN + MobileFaceNet facial feature extraction $\rightarrow$ Cross-segment clustering based on cosine similarity > 0.6.

**Cross-Modal Alignment**:
- Uses Chinese CLIP (ViT-B/16 image encoder + RoBERTa-wwm-Base text encoder)
- Contrastive Learning: Maximizes the similarity between the matched textual cluster and its corresponding character head image while minimizing the similarity of mismatched pairs.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{coref} + \mathcal{L}_{align}$$

- $\mathcal{L}_{coref} = \mathcal{L}_{start} + \mathcal{L}_{end} + \mathcal{L}_{clust}$ (all of which are binary cross-entropy losses)
- $\mathcal{L}_{align}$: Normalized temperature-scaled cross-entropy loss (from CLIP)

## Experiments

### Main Results

**Textual Coreference Resolution** (TikTalkCoref-celeb test set):

| Model | MUC F1 | B³ F1 | CEAF_ϕ4 F1 | Avg F1 |
|---|---|---|---|---|
| e2e-coref | 66.02 | 30.61 | 20.59 | 39.07 |
| **Maverick** | **51.92** | **68.14** | **76.30** | **65.46** |

Maverick significantly outperforms e2e-coref ($p < 0.001$), especially in handling singletons (B³ 75.30 vs 17.47).

**Cross-Modal Alignment**:

| Model | R@1 | R@2 | R@3 | Mean |
|---|---|---|---|---|
| R2D2 (zero-shot) | 52.50 | 71.50 | 76.25 | 66.67 |
| CN-Clip (zero-shot) | 45.00 | 65.00 | 68.75 | 59.58 |
| R2D2 (fine-tuning) | 56.25 | 73.75 | 80.00 | 70.00 |
| **CN-Clip (fine-tuning)** | **60.83** | **75.83** | **78.75** | **71.81** |

Under the zero-shot setting, R2D2 is stronger; after fine-tuning, CN-Clip surpasses it on R@1 and Mean.

### Ablation Study

**Effect of Data Augmentation**: Training the textual coreference module using non-celebrity data (Train-all vs Train-celeb):
- Maverick w/ DA: Avg F1 65.46 vs w/o DA: 52.67 (+12.79)
- The introduction of non-celebrity data brings more diverse linguistic contexts and mention types, improving model robustness.

**Retrieval Performance across Different Mention Types**:
- Name-central clusters: CN-Clip fine-tuning 70.52%  
- Pronoun-central clusters: CN-Clip fine-tuning 82.22%
- Noun-central clusters: R2D2 fine-tuning 60.61% (R2D2 pre-training data matches common noun-image pairs better)

### Key Findings
1. Maverick's pipeline approach performs exceptionally well on datasets containing a large number of singletons.
2. Data augmentation (adding non-celebrity data) can significantly improve celebrity coreference resolution performance.
3. Implicit visual cues in social media dialogues leave considerable room for improvement in cross-modal alignment.

## Highlights & Insights

1. **Novelty**: The first Chinese social media multimodal coreference resolution dataset, filling an important gap in this field.
2. **Realistic Scenarios**: Based on real Douyin short videos and user comment dialogues, it is closer to real-world applications than existing databases (such as human-machine dialogues and movie narratives).
3. **Meticulous Annotation Design**: Nested mentions and their sub-mentions are annotated independently (as opposed to OntoNotes which takes the longest span), reflecting co-referential relations more accurately.
4. **Full Coverage of Three Mention Types**: Proper nouns + common nouns + pronouns, providing a more comprehensive coverage than existing datasets.
5. **Negative Sample Design in Cross-Modal Alignment**: Using non-matching images from the same video as negative samples, which fits CN-Clip better under this setup.

## Limitations & Future Work

1. **Limited Data Scale**: Only 1,012 dialogues, and all of them are from the Douyin platform, limiting diversity.
2. **Limitations of Supervised Learning**: Model potential might not be fully exploited in low-resource scenarios; semi-supervised or unsupervised methods can be explored in the future.
3. **Benchmark Focused on Celebrity Domain**: Only evaluated fully on the celebrity subset, restricting generalization analysis.

## Related Work & Insights

- **Textual Coreference Resolution**: OntoNotes, LitBank, e2e-coref (Lee et al., 2017), Maverick (Martinelli et al., 2024)
- **Multimodal Coreference Resolution**: MPII-MD (Rohrbach et al., 2017), VisPro (Yu et al., 2019), CIN (Goel et al., 2023), J-CRe3 (Ueda et al., 2024)
- **Vision-Language Alignment**: CLIP (Radford et al., 2021), Chinese CLIP (Yang et al., 2022)

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐ — The first Chinese social media MCR dataset, filling an important gap.
- Practicality: ⭐⭐⭐⭐ — Directly valuable for understanding social media.
- Method Novelty: ⭐⭐⭐ — The pipeline is relatively straightforward, with each module built on existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive multi-dimensional analysis, including mention types, data augmentation, and zero-shot vs. fine-tuning comparison.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)

</div>

<!-- RELATED:END -->
