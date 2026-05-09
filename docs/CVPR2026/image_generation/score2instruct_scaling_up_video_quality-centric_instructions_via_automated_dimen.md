---
title: >-
  [Paper Note] Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring
description: >-
  [CVPR 2026][Image Generation][Video Quality Assessment] Score2Instruct proposes SIG, a fully automated video quality instruction generation pipeline that requires neither human annotation nor closed-source APIs. By automatically evaluating 14 quality dimensions and aggregating them into comprehensive quality reasoning texts via hierarchical CoT, SIG constructs the S2I dataset (320K+ instruction samples). Combined with a two-stage progressive fine-tuning strategy, multiple video LMMs simultaneously acquire quality scoring and quality reasoning capabilities, achieving an average SRCC improvement of 26–31% across 5 VQA benchmarks.
tags:
  - CVPR 2026
  - Image Generation
  - Video Quality Assessment
  - Instruction Tuning
  - Automated Scoring
  - Quality Reasoning
  - Large Multimodal Models
date: 2026-05-08
content_hash: 38f46e56aa0aba86
---

# Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring

**Conference**: CVPR 2026
**arXiv**: [2506.21011](https://arxiv.org/abs/2506.21011)
**Code**: [https://github.com/KeiChiTse/S2I](https://github.com/KeiChiTse/S2I)
**Area**: Image Generation
**Keywords**: Video Quality Assessment, Instruction Tuning, Automated Scoring, Quality Reasoning, Large Multimodal Models

## TL;DR
Score2Instruct proposes SIG, a fully automated video quality instruction generation pipeline that requires neither human annotation nor closed-source APIs. By automatically evaluating 14 quality dimensions and aggregating them into comprehensive quality reasoning texts via hierarchical CoT, SIG constructs the S2I dataset (320K+ instruction samples). Combined with a two-stage progressive fine-tuning strategy, multiple video LMMs simultaneously acquire quality scoring and quality reasoning capabilities, achieving an average SRCC improvement of 26–31% across 5 VQA benchmarks.

## Background & Motivation

**Background**: Traditional Video Quality Assessment (VQA) methods use deep learning to regress a single overall quality score (MOS). However, such a scalar cannot capture complex multi-dimensional quality issues in video (e.g., noise, motion blur, flicker). The emergence of LMMs has made it possible to produce quality justifications in natural language.

**Limitations of Prior Work**: (1) Existing quality instruction data generation relies heavily on human subjective evaluation — Q-Instruct, for instance, required 39 experts to write 58K descriptions for 18,973 images, which is time-consuming and costly. (2) Data generation depends on closed-source APIs (e.g., GPT-4), limiting scalability and reproducibility. (3) Most prior work focuses on Image Quality Assessment (IQA), lacking deep understanding of video-specific temporal factors (e.g., flicker, motion blur). (4) Existing methods cannot endow models with both precise scoring and quality reasoning capabilities simultaneously.

**Key Challenge**: High-quality video quality instruction data requires both rich dimensional coverage (beyond a single overall score) and a scalable generation approach (without reliance on human labor or closed-source systems). These two requirements were previously at odds — the richer the dimensions, the more expert annotation is needed.

**Goal**: (1) Design a fully automated video quality instruction generation pipeline; (2) Cover 14 quality dimensions and aggregate them via cognitive reasoning; (3) Simultaneously improve model scoring accuracy and reasoning/explanation capability.

**Key Insight**: The paper leverages existing professional video quality assessment models (deployed on practical video processing platforms) to automatically score quality dimensions, maps continuous scores to ITU-standard 5-level textual ratings, and then uses hierarchical CoT to simulate the reasoning process of the Human Visual System (HVS), generating complete quality descriptions.

**Core Idea**: Replace human annotation with automated quality dimension scoring, and replace closed-source LLM generation with hierarchical CoT for quality reasoning, thereby enabling large-scale, low-cost construction of video quality instruction data.

## Method

### Overall Architecture
The SIG (Score-based Instruction Generation) pipeline operates in three steps: (1) **Video source collection** — 104K quality-balanced videos are collected from labeled VQA datasets (Maxwell, 4,543 videos) and an unlabeled general video library (100K videos); (2) **Automated dimension scoring** — professional models evaluate each video across 14 quality dimensions and map scores to textual ratings; (3) **Hierarchical CoT aggregation** — the HVS reasoning process is simulated to aggregate dimension ratings into comprehensive quality reasoning texts, which are then expanded into diverse QA pairs via an LLM. The final S2I dataset contains 104K reasoning samples and 216K QA pairs (320K instructions total).

### Key Designs

1. **Dual-Source Video Collection**:

    - Function: Balances annotation richness with data scale.
    - Mechanism: For labeled datasets, the number of annotated dimensions serves as the selection criterion — Maxwell annotates 13 dimensions per video (far exceeding datasets such as KoNViD that only provide MOS), so 4,543 videos are drawn exclusively from Maxwell. For unlabeled datasets, a lightweight quality estimator computes proxy quality scores for candidate videos, uniform sampling is applied to ensure balanced quality distribution, and videos with inaccurate proxy labels are filtered out, yielding 100K videos.
    - Design Motivation: VQA datasets are typically 1/10 to 1/100 the size of other vision task datasets due to the high cost of MOS collection, making purely VQA-sourced data insufficient. Direct use of unlabeled videos, however, risks severe quality distribution skew.

2. **Automated 14-Dimension Scoring**:

    - Function: Generates a comprehensive multi-dimensional quality profile for each video.
    - Mechanism: Fourteen quality dimensions are systematically enumerated across four stages of video processing (capture, editing, compression, transmission), including lens sharpness, noise, flicker, motion blur, inter-frame smoothness, etc. Professional models deployed on well-known video processing platforms perform automated scoring (continuous scores in [0,1]), avoiding the subjective bias of crowdsourced ratings. Scores are then mapped to ITU-standard 5-level textual ratings (bad/poor/fair/good/excellent), with mapping accuracy verified to achieve SRCC/PLCC > 0.95. To eliminate ambiguity, dimension names are replaced with concise definitions (e.g., "flicker" → "the variation smoothness between adjacent frames").
    - Design Motivation: (1) The 14 dimensions represent the most comprehensive coverage to date; (2) A gap exists between continuous scores and discrete tokens, and the 5-level mapping follows ITU standards with human subjective validation; (3) Ambiguous dimension names (e.g., "flicker is good" is easily misinterpreted) necessitate definition-based descriptions.

3. **Hierarchical Chain-of-Thought Aggregation**:

    - Function: Simulates the quality reasoning process of the Human Visual System to generate complete quality descriptions.
    - Mechanism: The 14 dimensions are grouped into distortion-related and aesthetic-related categories according to HVS preferences. The CoT proceeds bottom-up in three steps: (a) assess the impact of each individual dimension → (b) derive intermediate ratings within each group → (c) synthesize the two intermediate ratings into a final quality assessment. The output is then linguistically diversified by an open-source LLM (Vicuna-7B) and enriched with high-level content descriptions from ShareCaptioner-Video. A final LLM summarization step uses prompt design to ensure the original rating levels are not altered during rewriting.
    - Design Motivation: Merely enumerating dimension scores lacks cognitive reasoning — human assessors do not evaluate each dimension independently but weigh distortion and aesthetic factors holistically. The hierarchical CoT simulates this local-to-global cognitive process.

### Loss & Training
A two-stage progressive fine-tuning strategy is adopted. **Stage I (Pre-training)**: The LLM is frozen; only the visual encoder and projector are trained on 100K simple "dimension name → rating" mapping tasks to establish preliminary quality perception. **Stage II (Instruction Tuning)**: The LLM is unfrozen (using LoRA, $r=16$) and trained on 220K reasoning and QA samples to enhance quality reasoning and scoring. Both stages use cross-entropy loss supervised only on the Assistant response tokens.

## Key Experimental Results

### Main Results (Quality Scoring — SRCC / PLCC)

| Model | S2I | Maxwell | LSVQtest | LSVQ1080p | KoNViD-1k | LIVE-VQC |
|------|-----|---------|----------|-----------|-----------|----------|
| LLaVA-OV-7B | No | 0.474/0.428 | 0.449/0.438 | 0.337/0.311 | 0.392/0.394 | 0.397/0.410 |
| LLaVA-OV-7B | **Yes** | **0.795/0.812** | **0.751/0.730** | **0.671/0.634** | **0.726/0.689** | **0.738/0.752** |
| LLaVA-Video-7B | No | 0.564/0.557 | 0.494/0.446 | 0.422/0.380 | 0.535/0.488 | 0.572/0.530 |
| LLaVA-Video-7B | **Yes** | **0.826/0.774** | **0.760/0.734** | **0.667/0.652** | **0.773/0.769** | **0.730/0.765** |

### Ablation Study

| Configuration | CI↑ | CU↑ | DO↑ | TU↑ | Maxwell SRCC/PLCC | Notes |
|---------|-----|-----|-----|-----|-------------------|------|
| Full (S2I) | 3.02 | 2.49 | 2.13 | 2.24 | 0.795/0.812 | Complete model |
| w/o labeled videos | 2.44 | 1.76 | 1.68 | 1.59 | 0.738/0.702 | Removing Maxwell labeled data degrades reasoning |
| w/o unlabeled videos | 2.57 | 2.08 | 2.10 | 2.14 | 0.506/0.553 | Removing unlabeled data degrades scoring |
| w/o hierarchical CoT | 2.96 | 2.41 | 2.08 | 2.16 | 0.786/0.810 | Reasoning capability declines |
| w/o high-level caption | 2.54 | 2.35 | 1.98 | 2.09 | 0.750/0.723 | Noticeable drop in CU/DO/TU |
| w/o rating-preserving prompt | 3.02 | 2.49 | 2.13 | 2.24 | 0.604/0.658 | Scoring accuracy degrades significantly |
| w/o Stage I pre-training | 2.86 | 2.49 | 2.10 | 2.19 | 0.638/0.592 | Scoring drops substantially (−15.7% SRCC) |

### Key Findings
- **Labeled data primarily contributes to reasoning; unlabeled data primarily contributes to scoring**: The 13-dimensional annotations in Maxwell provide "quality knowledge," while large-scale unlabeled data with automated scoring provides "quantitative calibration."
- **Stage I pre-training is critical for scoring**: Removing it reduces Maxwell SRCC from 0.795 to 0.638, demonstrating that simple dimension-scoring tasks help the model establish foundational quality dimension understanding.
- **The rating-preserving prompt is essential for scoring accuracy**: Without controlling rating consistency during LLM rewriting, SRCC drops sharply from 0.795 to 0.604, revealing that LLMs tend to inadvertently modify rating information during paraphrasing.
- **Data scaling effects are not yet saturated**: Performance improves consistently when training with 20%/50%/100% of the data, suggesting that further scaling the SIG pipeline could yield additional gains.

## Highlights & Insights
- The **"automated scoring → text mapping → CoT aggregation" data generation paradigm** is the paper's most significant contribution: it transforms "quality assessment requiring human subjectivity" into "an automatable engineering workflow," bridging the last mile of automated annotation. This paradigm is transferable to any domain requiring multi-dimensional subjective evaluation (e.g., audio quality, typographic aesthetics).
- The **two-stage progressive strategy** validates the effectiveness of a curriculum learning approach — "learn dimensions first, then learn reasoning" — in quality assessment. Stage I functions as a warm-up, enabling the model to first build a quality vocabulary before tackling complex reasoning.
- **The interplay between quality and content**: Incorporating high-level captions improves CU/DO/TU scores across the board, confirming that human quality perception cannot be decoupled from content understanding — the assessment mechanisms for a blurry but content-rich video and a sharp but unengaging video are fundamentally different.

## Limitations & Future Work
- **Accuracy of the 14-dimension scoring models is not fully validated**: Although the score-to-text mapping achieves high accuracy (SRCC > 0.95), the impact of errors in the underlying scoring models on final instruction quality has not been quantified.
- **S2I-Bench is relatively small** (only 400 questions), and its ground truth is generated via the automated pipeline (albeit with human review), falling short of purely human-annotated ground truth.
- **Only visual quality dimensions are considered**: Audio quality, audio-visual synchronization, and other multimodal quality dimensions are absent, despite being equally important in UGC video.
- **Vicuna-7B is used for rewriting and expansion**: Its limited capacity may result in insufficient linguistic diversity; substituting a stronger open-source LLM may yield better results.

## Related Work & Insights
- **vs. Q-Instruct**: Q-Instruct requires 39 experts to write descriptions, supplemented by ChatGPT expansion; SIG is fully automated and covers more dimensions. The trade-off is that SIG's reasoning texts may be less natural than human-written descriptions.
- **vs. Chat-UniVi-VQA**: A similar video VQA instruction tuning work, but data generation still relies on VQA datasets and manual annotation, offering less scalability than SIG.
- **Insights**: The three-step "automated scoring + mapping + CoT" approach of the SIG pipeline can be generalized to image aesthetic assessment, audio quality evaluation, and similar domains; the key is identifying domain-specific automated scoring tools.

## Rating
- Novelty: ⭐⭐⭐⭐ — The pipeline design is elegant, organically combining automated scoring with CoT reasoning to address data scalability.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six open-source and three closed-source models, five VQA benchmarks, and comprehensive ablation analysis with insightful findings.
- Writing Quality: ⭐⭐⭐⭐ — The pipeline description is clear, though the Related Work section is overly lengthy and the content distribution between the main paper and appendix could be optimized.
- Value: ⭐⭐⭐⭐ — A reproducible automated data pipeline is proposed with practical significance for improving quality understanding in video LMMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization](diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)
- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[ICCV 2025\] Video Color Grading via Look-Up Table Generation](../../ICCV2025/image_generation/video_color_grading_via_look-up_table_generation.md)
- [\[CVPR 2026\] Tiny Inference-Time Scaling with Latent Verifiers](tiny_inference-time_scaling_with_latent_verifiers.md)

</div>

<!-- RELATED:END -->
