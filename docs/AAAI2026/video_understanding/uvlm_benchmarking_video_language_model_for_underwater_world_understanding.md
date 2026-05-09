---
title: >-
  [Paper Note] UVLM: Benchmarking Video Language Model for Underwater World Understanding
description: >-
  [AAAI 2026][Video Understanding][underwater video] This paper constructs the first benchmark for underwater video-language understanding, UVLM, comprising 2,109 video clips, 419 marine species categories, 20 sub-tasks, and approximately 40K video-text pairs. Through a human-AI collaborative annotation pipeline that injects marine domain knowledge, a 7B VidLM fine-tuned on UVLM achieves performance approaching GPT-4o (73.04 vs. 77.95 Overall).
tags:
  - AAAI 2026
  - Video Understanding
  - underwater video
  - VidLM
  - benchmark
  - marine biology
  - fine-grained recognition
  - human-AI annotation
date: 2026-05-08
content_hash: 648770ac782f2df1
---

# UVLM: Benchmarking Video Language Model for Underwater World Understanding

**Conference**: AAAI 2026
**arXiv**: [2507.02373](https://arxiv.org/abs/2507.02373)
**Code**: [GitHub](https://github.com/Cecilia-xue/UVLM-Benchmark)
**Area**: Video-Language Understanding / Underwater Vision
**Keywords**: underwater video, VidLM, benchmark, marine biology, fine-grained recognition, human-AI annotation

## TL;DR

This paper constructs the first benchmark for underwater video-language understanding, UVLM, comprising 2,109 video clips, 419 marine species categories, 20 sub-tasks, and approximately 40K video-text pairs. Through a human-AI collaborative annotation pipeline that injects marine domain knowledge, a 7B VidLM fine-tuned on UVLM achieves performance approaching GPT-4o (73.04 vs. 77.95 Overall).

## Background & Motivation

**Background**: Video language models (VidLMs) have achieved remarkable progress on tasks such as video captioning, temporal grounding, and visual question answering, yet existing work predominantly focuses on terrestrial scenes (human activities, sports, daily life), leaving the underwater domain largely unexplored.

**Limitations of Prior Work**: (1) **Visual feature degradation** — underwater environments suffer from light attenuation, wavelength-dependent color distortion, and turbidity fluctuations, making it difficult for standard VidLMs to effectively process degraded visual cues; (2) **Lack of domain knowledge** — underwater content requires specialized ecological expertise including taxonomic identification, morphological attributes, behavioral states, and ecological relationships, whereas model training is dominated by common objects and human activities, creating a substantial knowledge gap; (3) **Scarcity of data resources** — existing underwater datasets primarily target single vision tasks such as object tracking (WebUOT) and instance segmentation (UIIS), with no multimodal video-language dataset available.

**Key Challenge**: The underwater environment holds immense scientific value (marine biodiversity monitoring, ecological health assessment) and engineering application potential (AUVs, offshore infrastructure inspection), yet no existing VidLM benchmark addresses underwater scenarios.

**Goal**: To construct the first specialized benchmark dataset for underwater video-language understanding, inject marine domain knowledge, and validate the performance of fine-tuned VidLMs in underwater scenes.

**Key Insight**: A human-AI collaborative annotation pipeline (manual frame-level annotation + GPT-4o-assisted generation + expert correction) is employed to construct a benchmark spanning 20 sub-tasks across two dimensions: biological and environmental.

**Core Idea**: Fine-grained taxonomic information from manual frame-level annotation is used to guide GPT-4o in generating diverse video-text pairs, which then undergo three rounds of expert review to ensure quality, resulting in the first large-scale underwater video-language benchmark.

## Method

### Overall Architecture

Video collection (web crawling + WebUOT re-annotation) → Manual frame-level annotation (bounding boxes + Whittaker five-kingdom taxonomic classification) → GPT-4o-assisted Q&A generation → Two rounds of human correction → Final dataset (2,109 videos, 0.86M frames, 419 categories, ~40K video-text pairs) → Evaluation across 8 metrics.

### Key Designs

1. **Underwater-specific Video Collection and Quality Control**
   - Dual-path collection strategy: (a) approximately 400 videos crawled from YouTube/Bilibili, covering marine, lake, river, and aquarium environments with typical degradations such as surface ripples, turbid water, and light scattering; (b) re-annotation of videos from WebUOT, including data cleaning (removal of subtitle watermarks), scene filtering (exclusion of non-natural underwater environments), and SAM+LaMa-based removal of small-area watermarks.
   - **Design Motivation**: To ensure the data captures the unique challenges of the underwater environment, rather than merely collecting clear aquarium footage.

2. **Three-stage Fine-grained Taxonomic Annotation**
   - Stage 1: Four annotators with marine biology expertise independently perform species-level identification and detailed taxonomic classification (kingdom, phylum, class, order, etc., following the Whittaker five-kingdom system).
   - Stage 2: Annotators are paired for cross-validation, with disagreements resolved by majority consensus of a third annotator.
   - Stage 3: Senior marine biology experts review and verify the annotations; disputed cases are resolved through collective discussion among five experts.
   - **Design Motivation**: Fine-grained taxonomic information is incorporated as prior knowledge into the GPT-4o text generation process to ensure ecological accuracy.

3. **Structured Human-AI Collaborative Annotation Pipeline**
   - Prompts are designed based on key marine biology research themes: biological dimension (static: species identification, morphological attributes; dynamic: behavioral analysis, locomotion patterns) + environmental dimension (static: substrate type, coral structure; dynamic: illumination variation, visibility fluctuation).
   - Each video generates 16–20 video-text pairs, including both multiple-choice and open-ended questions.
   - Two rounds of human correction: the first round by general reviewers to detect information conflicts; the second round by senior experts for in-depth editing to ensure factual precision.
   - **Design Motivation**: To maintain data volume while ensuring the reliability of ecological knowledge through multi-round human review.

### Evaluation Metric System

2 objective metrics + 5 LLM-based judgment metrics:
- Objective metrics: Multiple Choice Accuracy (MCA), Fine-grained Taxonomic Classification (FGC)
- LLM-based judgment metrics (GPT-4o as evaluation backbone): Semantic Accuracy (SA), Detail Completeness (DC), Visual Perception Accuracy (VPA), Environmental Description Accuracy (EDA), Species Behavior Matching (SBM)

## Key Experimental Results

### Closed-source Model and Large Model Baselines

| Model | MCA | FGC | SA | DC | VPA | EDA | SBM | Overall |
|-------|-----|-----|-----|-----|-----|-----|-----|---------|
| GPT-4o | 77.72 | 81.47 | 77.67 | 73.40 | 78.23 | 80.07 | 79.73 | 77.95 |
| Claude3.7-Sonnet | 76.61 | 82.64 | 73.35 | 73.58 | 74.10 | 79.71 | 76.35 | 76.09 |
| Gemini2.5-Flash | 78.22 | 86.27 | 72.43 | 73.34 | 70.53 | 78.32 | 74.92 | 75.00 |
| Qwen2.5VL-72B | 75.97 | 80.57 | 74.22 | 71.94 | 74.85 | 78.45 | 77.40 | 75.49 |

### Open-source Model Performance Before and After Fine-tuning

| Model | Before Fine-tuning Overall | After Fine-tuning Overall | Gain |
|-------|---------------------------|--------------------------|------|
| InternVL2.5-1B | 46.73 | 59.14 | +12.41 |
| VideoLLaMA3-2B | 58.39 | 66.67 | +8.28 |
| Qwen2.5VL-2B | 52.97 | 58.44 | +5.47 |
| InternVL2.5-8B | 60.15 | 69.45 | +9.30 |
| VideoLLaMA3-7B | 62.70 | 73.04 | +10.34 |
| Qwen2.5VL-7B | 63.57 | 68.08 | +4.51 |

### Key Findings

- VideoLLaMA3-7B achieves an Overall score of 73.04 after fine-tuning, trailing Qwen2.5VL-72B by only 2.45 points and approaching the closed-source Gemini at 75.00.
- FGC (fine-grained taxonomic classification) is the only task where the performance gap between large and small models cannot be readily bridged through fine-tuning, owing to the complexity of biological domain knowledge that exceeds the capacity of smaller models.
- Fine-tuning on UVLM also yields improvements on general benchmarks (VideoMME, Perception Test).

## Highlights & Insights

- UVLM is the first benchmark for underwater video-language understanding, filling a critical gap in the VidLM literature.
- The three-stage taxonomic annotation pipeline demonstrates exceptional annotation rigor — four independent annotators, paired cross-validation, expert final review, and collective discussion among five specialists.
- The paradigm of distilling knowledge from GPT-4o into smaller models proves effective: a 7B model fine-tuned on UVLM approaches GPT-4o performance while substantially reducing deployment cost.
- The FGC task reveals an important finding: specialized domain knowledge (taxonomic identification) cannot be fully transferred to small models through simple fine-tuning, pointing to a fundamental tension between model capacity and domain knowledge.

## Limitations & Future Work

- The dataset scale (2,109 video clips) is relatively limited compared to terrestrial video benchmarks (HowTo100M, Kinetics).
- The biological category distribution exhibits a long-tail pattern; insufficient samples for rare species may impair fine-tuning effectiveness.
- Evaluation relies on GPT-4o as a judge (5 LLM-based metrics), introducing known biases associated with LLM-as-judge approaches.
- Only three open-source VidLM families are evaluated, limiting diversity.
- Cross-task transfer from UVLM to underwater vision tasks (e.g., tracking, segmentation) remains unexplored.

## Related Work & Insights

- **vs. WebUOT**: WebUOT is the closest existing underwater video dataset (1,500 sequences, 1M frames, 408 categories), but supports only single-object tracking with no language understanding dimension. UVLM builds upon it through re-annotation and extension.
- **vs. MarineInst**: MarineInst supports image segmentation and captioning but lacks video temporal understanding capability.
- **vs. Video-MME / Perception Test**: These are general-purpose video benchmarks that do not address underwater scenarios at all. UVLM fine-tuning jointly improves both underwater and general performance.
- Methodologically, the human-AI collaborative annotation paradigm combined with domain expert final review offers broadly applicable insights for constructing specialized domain datasets.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first benchmark for underwater video-language understanding, with a clearly differentiated research focus.
- **Experimental Thoroughness**: ⭐⭐⭐ Multiple closed-source and open-source models are evaluated, but more ablation studies (e.g., annotation strategy, effect of data volume) are lacking.
- **Writing Quality**: ⭐⭐⭐ Structure is clear and the annotation pipeline is described in detail, though the overall writing tends toward verbosity.
- **Value**: ⭐⭐⭐⭐ Fills an important gap; the dataset and code are open-sourced, providing direct impetus to the underwater vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](../../NeurIPS2025/video_understanding/lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)
- [\[AAAI 2026\] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos](uncovering_zero-shot_generalization_gaps_in_time-series_foundation_models_using_.md)

</div>

<!-- RELATED:END -->
