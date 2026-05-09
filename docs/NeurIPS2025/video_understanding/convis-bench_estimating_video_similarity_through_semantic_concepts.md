---
title: >-
  [Paper Note] ConViS-Bench: Estimating Video Similarity Through Semantic Concepts
description: >-
  [NeurIPS 2025][Video Understanding][video similarity] This paper introduces ConViS, a concept-based video similarity estimation task, along with its accompanying benchmark ConViS-Bench (610 video pairs, 16 domains, 5 concepts). It systematically evaluates 10+ mainstream models on concept-conditioned video comparison, revealing significant deficiencies in current models' understanding of temporal structure and spatial context.
tags:
  - NeurIPS 2025
  - Video Understanding
  - video similarity
  - benchmark
  - semantic concepts
  - Large Multimodal Models
  - video retrieval
date: 2026-05-08
content_hash: cda0c24551916480
---

# ConViS-Bench: Estimating Video Similarity Through Semantic Concepts

**Conference**: NeurIPS 2025
**arXiv**: [2509.19245](https://arxiv.org/abs/2509.19245)
**Code**: [GitHub](https://github.com/benedettaliberatori/convisbench)
**Area**: Video Understanding
**Keywords**: video similarity, benchmark, semantic concepts, Large Multimodal Models, video retrieval

## TL;DR

This paper introduces ConViS, a concept-based video similarity estimation task, along with its accompanying benchmark ConViS-Bench (610 video pairs, 16 domains, 5 concepts). It systematically evaluates 10+ mainstream models on concept-conditioned video comparison, revealing significant deficiencies in current models' understanding of temporal structure and spatial context.

## Background & Motivation

- **Background**: Traditional video similarity methods compute a single global score (e.g., cosine similarity in embedding space), which cannot explain "which aspects are similar and which differ," offering limited interpretability for downstream applications such as retrieval and anomaly detection.
- **Limitations of Prior Work**: Cognitive science research shows that humans naturally compare events along semantic dimensions (actions, subjects, locations, etc.), meaning video similarity depends on which concept is attended to rather than a fixed quantity. Existing video difference description methods (e.g., VidDiffBench) are confined to a single concept (action differences) and only cover 5 domains; StepDiff is limited to cooking videos; both provide only free-text descriptions without quantitative scores. Existing benchmarks (Video-MME, MVBench, etc.) evaluate primarily through QA and lack systematic testing of concept-level comparative reasoning.
- **Key Challenge**: There exists a gap between global similarity methods (quantitative but not interpretable) and video differencing methods (interpretable but not quantitative). Real-world applications often require concept-conditioned retrieval (e.g., "same action, different subject"), which existing methods cannot support.
- **Goal**: ConViS aims to simultaneously provide structured quantitative scores and interpretability by conditioning similarity estimation on user-specified semantic concepts.

## Method

### Overall Architecture: Concept-based Video Similarity (ConViS)

Given a video pair $(V_1, V_2)$ and a predefined concept set $\mathcal{C} = \{C_1, \ldots, C_K\}$ (expressed in natural language), ConViS computes a similarity score for each concept:

$$s(V_1, V_2 \mid C_i) \in \mathbb{R}$$

Individual concept scores can be aggregated into an overall score via weighted combination: $s(V_1, V_2) = \sum_{i=1}^{K} \lambda_i \cdot s(V_1, V_2 \mid C_i)$, where $\sum \lambda_i = 1$. This design offers both **flexibility** (users may introduce arbitrary concepts) and **composability** (scores can be weighted on demand).

### Key Design 1: ConViS-Bench Dataset Construction

- **Concept Selection**: Grounded in cognitive science (humans organize event memory along semantic and temporal features), five general concepts are defined: **main action**, **main subjects**, **main objects**, **location**, and **order of actions**.
- **Video Source**: Videos are drawn from the FineVideo dataset, selecting 16 visually diverse domains (excluding static talking-head content), and clipped into independent segments based on event timestamps.
- **Pairing Strategy**: DINOv2 (visual embeddings) and Sentence-BERT (textual embeddings) are used to compute cosine similarities separately; pairs exhibiting high similarity in only one modality are selected (ensuring both commonalities and differences), followed by manual filtering to yield 610 pairs.
- **Annotation Protocol**: 150 annotators were recruited via Prolific to rate each video pair on a 1–5 Likert scale for each of the five concepts, along with free-text similarity/difference labels. Each pair received an average of 6.2 annotations; 7.75% of low-quality annotations were discarded.

### Key Design 2: LMM Concept-Conditioned Evaluation

Frames from two videos are concatenated and fed into an LMM, with a prompt instructing the model to "output a similarity score from 1 to 5 based solely on \<concept\>." Evaluation uses Spearman's $\rho$ and Kendall's $\tau$ to measure agreement with human judgments. Nine open-source models (mPLUG-Owl3, LLaVA series, Qwen-VL series, InternVL series) and Gemini 2.0-Flash are covered.

### Key Design 3: Probing Implicit Concept Preferences of Global Representations

Three families of methods are designed to probe which concepts global similarity representations implicitly favor: ① Video-to-video (cosine similarity of VideoMAE/DINOv2 embeddings); ② Text-to-text (LMM-generated descriptions compared via Sentence-BERT); ③ Cross-modal (CLIPScore/VQAScore cross-modal alignment scores).

### Key Design 4: Concept-Conditioned Retrieval Task

Given an anchor video and a target concept, the model retrieves the most similar video from four candidates. A total of 532 concept-level partial rankings are constructed, evaluated with R@1/P@1/F1@1.

## Loss & Training

This is a **benchmark paper** that does not involve model training. All evaluations rely on zero-shot inference of existing models, with concept-conditioned predictions obtained via prompt engineering.

## Key Experimental Results

### Table 1: LMM Concept-Conditioned Similarity Estimation (Spearman's ρ × 100)

| Model | Main Action | Main Subjects | Main Objects | Location | Order of Actions |
|:---|:---:|:---:|:---:|:---:|:---:|
| mPLUG-Owl3-7B | 30.64 | 20.59 | 28.53 | 21.00 | 23.11 |
| LLaVA-OV-0.5B | 1.95 | -5.05 | -4.00 | 5.66 | 1.30 |
| **LLaVA-OV-7B** | **51.76** | **48.43** | **58.64** | **58.94** | **41.02** |
| LLaVA-Video-7B | 44.17 | 39.81 | 45.85 | 55.96 | 41.25 |
| Qwen2.5-VL-7B | 37.88 | 17.53 | 26.97 | 23.63 | 23.85 |
| InternVL2.5-8B | 28.70 | 28.60 | 25.06 | 19.64 | 18.15 |
| InternVL3-8B | 40.69 | 36.54 | 42.50 | 45.47 | 32.74 |

**LLaVA-OV-7B consistently achieves the best performance across all concepts**, yet even the top model scores only 41.02 on order of actions, markedly lower than on other concepts. The InternVL series, despite its pretraining data including FineVideo, does not perform prominently, suggesting that data inclusion does not imply genuine conceptual understanding.

### Table 2: Implicit Concept Preferences of Global Representations (Spearman's ρ × 100)

| Model | Method | Main Action | Main Subjects | Main Objects | Location | Order of Actions |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| VideoMAE | Cosine | 13.0 | 23.1 | 13.2 | 37.8 | 15.1 |
| DINOv2 | Cosine | 33.3 | 40.9 | 37.4 | **57.4** | 34.6 |
| mPLUG-Owl3 | SBERT | **52.1** | 45.5 | **55.1** | 28.4 | **49.9** |
| LLaVA-OV | VQAScore | 51.1 | **55.8** | 58.3 | 46.5 | 48.1 |

**Key Findings**: Visual encoders (DINOv2) are biased toward location; text-based methods favor action/objects; all methods consistently score low on order of actions. VQAScore achieves the best overall balance.

### Highlights of Concept-Conditioned Retrieval Results

LLaVA-OV-7B achieves P@1 of 66.4% on main subjects and 68.7% on location, substantially above the random baseline (~35–50%), while P@1 on main action is only 54.8%, indicating that action-based retrieval remains a weak point for current models.

## Highlights & Insights

- **Novel Task Definition with Cognitive Science Grounding**: ConViS precisely fills the gap between global similarity (quantitative but not interpretable) and video difference description (interpretable but not quantitative), with concept selection theoretically grounded in cognitive science.
- **High-Quality Dataset**: 150 annotators, 610 video pairs, 16 domains, 5 concepts, an average of 6.2 annotations per pair, with annotation quality control and IRB approval.
- **Comprehensive and In-Depth Evaluation**: Three evaluation dimensions are covered—LMM concept scoring, global representation preference probing, and concept-conditioned retrieval—along with frame-count ablations that reveal temporal dependency and recency effects.
- **Insightful Findings**: Order of actions is the Achilles' heel of all models; InternVL pretraining on test data does not aid concept-level understanding; visual and textual representations each exhibit distinct concept biases.

## Limitations & Future Work

- **Limited Concept Set**: Only five general concepts are defined, potentially missing domain-specific important dimensions (e.g., "skill level," "camera angle," "lighting conditions"—frequently appearing in annotators' custom concepts).
- **Small Dataset Scale**: 610 video pairs is insufficient for training concept-aware models; the dataset primarily serves as a test set.
- **No Training Method Proposed**: As a pure benchmark paper, no method for learning concept-level similarity is introduced; only the zero-shot capabilities of existing models are evaluated.
- **Moderate Annotator Agreement**: Krippendorff's α reaches a maximum of only 0.361 (location) and as low as 0.244 (main subjects), reflecting the inherent subjectivity of concept-level similarity.
- **Opaque Pretraining Data for Gemini**: The pretraining data of this proprietary model may already include FineVideo, potentially compromising evaluation fairness.

## Related Work & Insights

- **Global Video Similarity** (DNS, SSVL, etc.): Compute a single global score but lack interpretability; ConViS represents a structured upgrade.
- **Video Difference Description** (VidDiffBench [Burgess+ ICLR'25], StepDiff [Nagarajan+ CVPR'24]): Provide textual difference descriptions but without quantification, and are domain-restricted (actions/cooking); ConViS covers 16 domains with quantitative scores.
- **Image Concept Similarity** (Achille+ CVPR'24): Defines concept-level similarity between images; ConViS extends this to the more complex video domain.
- **LMM Video Benchmarks** (Video-MME, MVBench, TempCompass, etc.): Primarily test QA capabilities; ConViS uniquely tests concept-level comparative reasoning.
- **Compositional Video Retrieval** (CoVR, etc.): Queries consist of a reference video plus text modification; ConViS supports multi-concept dimensional exploration with explicit quantitative scores.

## Rating

- Novelty: ⭐⭐⭐⭐ — Concept-level video comparison is a well-defined new task with solid cognitive science foundations
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three-dimensional evaluation with frame-count ablations and recency effect analysis, broad coverage
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, persuasive motivation
- Value: ⭐⭐⭐⭐ — Provides an important new evaluation dimension for the video understanding community with practically instructive findings

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)
- [\[NeurIPS 2025\] Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition](disentangled_concepts_speak_louder_than_words_explainable_video_action_recogniti.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](../../CVPR2026/video_understanding/wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)

</div>

<!-- RELATED:END -->
