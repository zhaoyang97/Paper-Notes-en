---
title: >-
  [Paper Note] FineVQ: Fine-Grained User Generated Content Video Quality Assessment
description: >-
  [CVPR 2025][Recommender Systems][Video Quality Assessment] This work constructs the first large-scale, fine-grained UGC video quality assessment database, FineVD (6,104 videos, 800k+ ratings, 6 dimensions), and proposes FineVQ, an LMM-based approach. FineVQ enables a single model to simultaneously perform quality rating, scoring, and attribution, achieving state-of-the-art performance on FineVD and other UGC-VQA datasets.
tags:
  - "CVPR 2025"
  - "Recommender Systems"
  - "Video Quality Assessment"
  - "User Generated Content"
  - "Fine-Grained Assessment"
  - "Large Multimodal Models"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: 86ad78898e5cfb0e
---

# FineVQ: Fine-Grained User Generated Content Video Quality Assessment

**Conference**: CVPR 2025  
**arXiv**: [2412.19238](https://arxiv.org/abs/2412.19238)  
**Code**: [https://github.com/IntMeGroup/FineVQ](https://github.com/IntMeGroup/FineVQ)  
**Area**: Recommender Systems  
**Keywords**: Video Quality Assessment, User Generated Content, Fine-Grained Assessment, Large Multimodal Models, Instruction Tuning

## TL;DR
This work constructs the first large-scale, fine-grained UGC video quality assessment database, FineVD (6,104 videos, 800k+ ratings, 6 dimensions), and proposes FineVQ, an LMM-based approach. FineVQ enables a single model to simultaneously perform quality rating, scoring, and attribution, achieving state-of-the-art performance on FineVD and other UGC-VQA datasets.

## Background & Motivation

**Background**: With the explosive growth of UGC videos, video quality assessment (VQA) has become crucial for content monitoring, optimization, and recommendation on platforms. However, existing UGC-VQA methods (such as VSFA, SimpleVQA, and DOVER) typically output only a single overall quality score.

**Limitations of Prior Work**: A single overall quality score cannot satisfy the diverse needs of downstream applications. Video processing pipelines need to identify which specific dimension (e.g., distortion, blur, color) has issues; recommender systems require multi-dimensional quality signals; and content creators need to understand specific quality defects. Existing databases also lack multi-dimensional, fine-grained annotations.

**Key Challenge**: Application scenarios demand fine-grained, multi-dimensional quality information, whereas existing databases only provide coarse-grained overall scores, which limits models to outputting a single score. Furthermore, using separate models to evaluate each dimension individually is highly inefficient and suffers from poor consistency.

**Goal**: (1) To establish the first large-scale UGC-VQA database with multi-dimensional, fine-grained quality annotations; (2) to design a "one-for-all" fine-grained quality assessment method.

**Key Insight**: Utilizing the powerful visual understanding and text generation capabilities of Large Multimodal Models (LMMs), instruction tuning is leveraged to empower a single model with multi-task capabilities, including quality scoring (regression), quality rating (classification), and quality description (generation).

**Core Idea**: To construct a database with fine-grained annotations across 6 dimensions (color, noise, artifacts, blur, temporal, and overall quality), and to train a "one-for-all" video quality assessment model based on the InternVL framework, utilizing dual spatial-motion visual encoders and LoRA fine-tuning.

## Method

### Overall Architecture
Given an input UGC video and user prompt, encoding transitions through three parallel pathways: (1) a spatial image encoder (InternViT) extracts spatial features from 8 uniformly sampled frames; (2) a motion encoder (SlowFast) extracts temporal motion features from the entire video; and (3) a text tokenizer encodes the user prompt. The tokens from these three pathways are concatenated and fed into a pre-trained Large Language Model (InternLM) to generate quality-related responses, which can be quality ratings (classification), quality scores (regression), or quality descriptions (attribution).

### Key Designs

1. **FineVD Database Construction**:

    - **Function**: Provides the first large-scale, multi-dimensional, fine-grained UGC-VQA annotated dataset.
    - **Mechanism**: Collects 6,104 UGC videos (including professional and user-shot live broadcasts/VODs) from Bilibili. Twenty-two professional annotators rated the videos in a laboratory environment across six dimensions—color, noise, artifacts, blur, temporal, and overall—using a 5-level scale, yielding over 800k total ratings. Distortion types were annotated, and GPT-4 was employed to generate quality-related QA pairs, which were manually verified. These steps established the training data for three tasks: quality rating, scoring, and description.
    - **Design Motivation**: Existing databases only contain overall ratings, failing to support fine-grained quality assessment research. Laboratory annotations are more controllable in terms of quality compared to crowd-sourced annotations.

2. **Dual Visual Encoders + LoRA Fine-Tuning**:

    - **Function**: Empowers the pre-trained LMM with quality-aware capabilities without excessively increasing parameter scale.
    - **Mechanism**: Spatial features are extracted by InternViT (from 8 sampled frames), while motion features are derived from the entire video using a SlowFast network. Both feature pathways are projected into the language space via 2-layer MLPs. LoRA weights are applied to both the image encoder and the LLM for low-rank adaptation, infusing quality assessment domain knowledge while preserving the general capabilities of the foundation model.
    - **Design Motivation**: Sparse frame sampling is insufficient for capturing temporal quality issues such as jitter and lagging; thus, an auxiliary motion feature extractor is critical. LoRA avoids the high computational costs of full parameter fine-tuning while retaining flexibility across different quality dimensions.

3. **Multi-Task Unification via Instruction Tuning**:

    - **Function**: Enables a single model to simultaneously handle quality rating (classification), quality scoring (regression), and quality description (text generation).
    - **Mechanism**: Diverse types of instruction-answer pairs (QA pairs) are designed. The rating task requires the model to output classes like "good/fair/poor"; the scoring task requires numerical outputs ranging from 0 to 100; and the description task requires text generation of natural language describing the quality degradation. Mixing these three types of QA pairs during training achieves multi-task integration.
    - **Design Motivation**: Different applications require different granularities of quality feedback. Unifying them into a single LMM framework allows them to share low-level visual representation power while distinguishing task types via natural instructions.

### Loss & Training
The scoring task utilizes a regression loss (MSE), while the rating and description tasks employ the cross-entropy loss of language modeling. Training follows a two-stage strategy: the first stage freezes the visual encoders and the LLM, training only the projection layers; the second stage unfrees the LoRA weights for end-to-end fine-tuning.

## Key Experimental Results

### Main Results (FineVD Scoring Task)

| Method | Overall SRCC | Overall PLCC | Noise SRCC | Blur SRCC |
|------|-------------|-------------|-----------|----------|
| **Ours** (FineVQ) | **0.8834** | **0.8891** | **0.8444** | **0.8711** |
| DOVER | 0.8422 | 0.8393 | 0.8018 | 0.8404 |
| SimpleVQA | 0.8311 | 0.8358 | 0.8070 | 0.8466 |
| FAST-VQA | 0.8348 | 0.8474 | 0.8093 | 0.8352 |
| VIDEVAL (Traditional) | 0.7310 | 0.7307 | 0.6912 | 0.7610 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| DNN Method (per-dim) | Each dimension trained individually | Separate model for each dimension |
| FineVQ (one-for-all) | Unified across six dimensions | Single model with multiple dimensions, achieving superior performance |
| General LMM (Zero-Shot) | Lower quality attribution accuracy | InternVL2 zero-shot ~50-70% |
| FineVQ Quality Attribution | 87-95% across all dimensions | Significant improvement after instruction tuning |

### Key Findings
- FineVQ, using a one-for-all strategy, outperforms individually trained DNN-based methods (e.g., DOVER, SimpleVQA) across all six dimensions, illustrating the benefit of a unified model.
- The integration of the motion feature encoder yields notable performance gains in temporal quality assessment.
- General-purpose LMMs (such as InternVL2 and Qwen2-VL) show poor zero-shot performance in quality attribution tasks (~50-70%), whereas fine-tuning via FineVQ boosts accuracy to 87-95%.
- FineVQ displays competitive cross-dataset generalization capabilities on external benchmarks (e.g., LSVQ, KoNViD-1k).

## Highlights & Insights
- **First Multi-Dimensional, Fine-Grained VQA Database**: FineVD fills the vacancy of fine-grained annotations in the UGC-VQA field. The scale of 6 dimensions × 6,104 videos × 22 annotators is sufficient to support modern deep learning research.
- **One-for-All Design Philosophy**: Unifying scoring, rating, and description into a single model via instruction tuning eliminates the complexity of maintaining multiple dedicated models. Notably, shared representations in turn boost the performance of each sub-task.
- **Emergent Application of LMMs to Low-Level Vision**: The work showcases the untapped capability of LMMs in traditional low-level visual quality evaluation, demonstrating that low-rank adaptation with a fraction of parameters is sufficient to inject strong quality perception.

## Limitations & Future Work
- All source videos in the database are collected from a single platform (Bilibili), which may introduce platform-specific content and quality distribution biases.
- The six quality dimensions are pre-defined, limiting direct exploration of finer grains (e.g., specific severity levels of individual distortion types) or adaptive dimension discovery.
- Model inference demands LMM-scale computational resources, making it less suitable for edge devices or real-time application scenarios.
- The PLCC of the scoring task on the noise dimension (0.7986) is lower compared to other dimensions, indicating room for improvement in noise quality assessment.

## Related Work & Insights
- **vs DOVER**: DOVER covers technical and aesthetic qualities but only across two dimensions without rating or description capabilities. FineVQ expands this to six dimensions and supports multiple tasks.
- **vs Q-Align**: Q-Align also uses LMMs for quality assessment but primarily targets images and yields only overall scores. FineVQ focuses on videos, acts multi-dimensionally, and integrates a motion encoder.
- **vs SimpleVQA**: SimpleVQA simplifies quality prediction via pre-trained features and linear regression layers, but fails to provide text descriptions or quality attribution. FineVQ also demonstrates superior scoring performance over SimpleVQA.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The first fine-grained VQA database combined with an LMM-based multi-task unified framework fills a critical gap in the field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple benchmarks, comparative studies against diverse baselines, and rigorous evaluation of all three tasks (rating, scoring, and description).
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, logical descriptions of database construction steps, and rich visualization.
- **Value**: ⭐⭐⭐⭐ Both the dataset and models are open-sourced, delivering high practical value to the UGC video quality assessment community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](../../AAAI2026/recommender/semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[ACL 2025\] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences](../../ACL2025/recommender/lotus_a_leaderboard_for_detailed_image_captioning_from_quality_to_societal_bias_.md)
- [\[ECCV 2024\] AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling](../../ECCV2024/recommender/aid-appeal_automatic_image_dataset_and_algorithm_for_content_appeal_enhancement_.md)
- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](../../NeurIPS2025/recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](../../ACL2026/recommender/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)

</div>

<!-- RELATED:END -->
