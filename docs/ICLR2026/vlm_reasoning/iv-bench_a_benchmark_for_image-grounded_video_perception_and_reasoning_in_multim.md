---
title: >-
  [Paper Note] IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs
description: >-
  [ICLR 2026][VLM Reasoning][Image-Grounded Video Understanding] IV-Bench is the first benchmark for "Image-Grounded Video Perception and Reasoning"—using an **externally sourced** reference image as visual context to query video content. With 966 videos, 2,560 image-text queries, and 13 task categories, it reveals that the strongest MLLMs achieve only 28.9% accuracy (compared to 88.8% for humans), exposing the fragility of current models' ability to understand videos via visua…
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "Image-Grounded Video Understanding"
  - "Multimodal Benchmark"
  - "Video Perception"
  - "Video Reasoning"
  - "MLLM Evaluation"
date: 2026-05-08
content_hash: c745f7821f8e925b
---

# IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3di7ct0iOJ](https://openreview.net/forum?id=3di7ct0iOJ)  
**Code**: [https://github.com/multimodal-art-projection/IV-Bench](https://github.com/multimodal-art-projection/IV-Bench)  
**Area**: Multimodal Evaluation / Video Understanding / VLM Reasoning  
**Keywords**: Image-Grounded Video Understanding, Multimodal Benchmark, Video Perception, Video Reasoning, MLLM Evaluation  

## TL;DR
IV-Bench is the first benchmark for "Image-Grounded Video Perception and Reasoning"—using an **externally sourced** reference image as visual context to query video content. With 966 videos, 2,560 image-text queries, and 13 task categories, it reveals that the strongest MLLMs achieve only 28.9% accuracy (compared to 88.8% for humans), exposing the fragility of current models' ability to understand videos via visual anchors.

## Background & Motivation
**Background**: MLLMs have progressed rapidly in image and video understanding, supported by numerous benchmarks (Video-MME, MVBench, LongVideoBench, etc.). However, these benchmarks almost exclusively use **text-only queries** to probe video content.

**Limitations of Prior Work**: Language is inherently abstract and struggles to precisely refer to fine-grained visual attributes (e.g., subtle design differences) or specific instances (e.g., a specific person's face). In reality, demand for "visual search" is massive—Pinterest Lens processed 600 million monthly visual searches as early as 2018, and 62% of young consumers prefer visual search capabilities because text is insufficient in these scenarios. Yet, the task of "using an image as a visual anchor to understand video" has been systematically ignored in current evaluation frameworks.

**Key Challenge**: A few benchmarks attempting image-text queries (V2P-Bench, VideoRefer-Bench) suffer from a fatal flaw—reference images are **extracted directly from video frames**. This causes tasks to degenerate into "frame matching," where models can succeed through simple perceptual similarity (information leakage) without generalized understanding. Meanwhile, Video-MMMU uses external images, but the video content is often auxiliary.

**Goal**: To construct the first benchmark that truly tests "image-grounded video perception and reasoning," requiring **both the image and video to be indispensable** for answering correctly.

**Core Idea**: **[External Images as Mandatory Visual Anchors]** Reference images are sourced entirely from outside the videos. Combined with two rounds of rigorous quality control (QC) to enforce "Image Necessary + Video Necessary + At least two valid distractors," this approach eliminates frame-matching shortcuts and unimodal leakage, forcing models to ground conceptual knowledge from the image into the video content.

## Method

### Overall Architecture
IV-Bench is not a model, but a data construction and evaluation protocol centered on "Image-Text Query $\rightarrow$ Video" pairs. It includes 966 videos (each >5 minutes) covering Knowledge, Film/TV, Sports, Arts, and Life Logs. Each video features multiple queries (one external reference image + text question + multiple choices). The 13 tasks are categorized into 7 Perception tasks (visible/recognizable) and 6 Reasoning tasks (calculable/inferable). Data is produced via a pipeline: "Annotation $\rightarrow$ QC Round 1 (clarity/correctness/classification) $\rightarrow$ QC Round 2 (indispensability/distractor enhancement)." Models are evaluated using a unified format: 32 uniformly sampled video frames + reference image + question.

```mermaid
flowchart LR
    A[Select 966 Videos<br/>5 Categories · Length>5min] --> B[Manual Annotation<br/>External Image+Text Question+Distractors]
    B --> C[QC Round 1<br/>Clarity/Correctness/Classification Check]
    C --> D[QC Round 2<br/>Remove Unimodal Solvable Samples<br/>Add ≥2 Valid Distractors]
    D --> E[2560 Image-Text Queries<br/>13 Tasks=7 Perception+6 Reasoning]
    E --> F[Unified Evaluation<br/>32 Frames+Image+Question → MC Accuracy]
```

### Key Designs

**1. Externally Sourced Reference Images: Blocking "Frame Matching."** Unlike similar benchmarks, IV-Bench prohibits reference images from being extracted from the video. Annotators must find external images related to keywords, characters, or themes within the video. This forces models to understand the concept in the image (e.g., "this is a specific player's jersey") and generalize that knowledge to the video, rather than relying on visual pixel alignment.

**2. Two-Round QC for "Dual-Modality Indispensability."** Samples solvable by common sense or video alone are removed. Any text query that leaks image content is rewritten. Crucially, **valid distractors** are designed such that at least two options are "incorrect for the current image but would be correct if a different image were provided for the same question." This counterfactual design ensures the specific image is essential for every sample.

**3. Hierarchical Perception-Reasoning Tasks.** The benchmark covers a spectrum from basic "seeing" to complex "inferring." The 7 Perception tasks include Existence, Reverse Existence, NLI (scene similarity), Spatial Relationship, Keyframe Extraction, Constrained OCR, and Detailed Events. The 6 Reasoning tasks include Counting, Space-Time Computing (duration/distance), Summary, Instruction Understanding (function/process), Attribute Change, and Temporal Reasoning.

**4. Probing Input Order and Token Allocation.** Evaluations use 32 frames + image + question. The study found that **placing the image after video frames performs best**, **performance is more sensitive to frame count than resolution**, and **only large models benefit from the image context**, while small models show negligible improvement.

## Key Experimental Results

### Main Results
28 MLLMs (5 proprietary, 23 open-source) were evaluated with 32 frames using the `Video frames + Image + Question` format. Accuracy (%):

| Model | Overall | P-Avg (Perception) | R-Avg (Reasoning) |
|------|---------|-------------|-------------|
| Human | **88.8** | 91.5 | 86.9 |
| Qwen2.5-VL-72B | **28.9** | — | — |
| InternVL2.5-78B | 28.6 | 33.4 | 21.9 |
| Gemini-2.0-Pro | 27.7 | — | 24.9 |
| Qwen2.5-VL-7B (Best <10B) | 18.5 | — | — |
| Random Guess | 11.11 | — | — |
| Llama-vid | 10.5 | 11.2 | 9.5 |

Note: With an average of 9 options, the random baseline is approximately 11.11%.

### Ablation Study

| Dimension | Key Finding |
|----------|----------|
| Image Context Inclusion | Adding images significantly improves video understanding (validates necessity). |
| Model Scale | Large models utilize image context; small models gain almost nothing (emergent capability). |
| Frames vs. Resolution | Performance is more sensitive to **increasing frames** than **higher resolution**. |
| Image Position | Placing the image **after video frames** yields the best results. |

### Key Findings
- **Huge Human-AI Gap**: Humans (88.8%) vs. the strongest model (28.9%). This ~60% gap indicates image-grounded video understanding is a major blind spot for current MLLMs.
- **Perception > Reasoning**: Models generally perform better on perception than reasoning (e.g., InternVL2.5-78B: 33.4% vs 21.9%), with Temporal Reasoning being particularly challenging.
- **Small Models Fail to Utilize Context**: Qwen2.5-VL-7B (best <10B) reached only 18.5%, suggesting visual context utilization is highly dependent on scale.

## Highlights & Insights
- **Methodological Innovation**: The "External Image + Dual-Modality Necessity" approach elevates video benchmarks from "frame matching" to "conceptual generalization." The counterfactual distractor mechanism effectively prevents shortcut learning.
- **High Diagnostic Value**: The hierarchical tasks combined with systematic ablations on input order and token allocation provide an engineering guide for feeding image context to MLLMs.
- **Statistical Significance**: Evaluation across 28 models with a high number of distractors (9) makes the human-machine gap highly credible.

## Limitations & Future Work
- **Multiple-Choice Constraint**: All tasks are multiple-choice, which facilitates automated scoring but does not test open-ended generation or bounding box regression.
- **Static Single-Image Constraint**: Each query uses only one external image, not covering complex scenarios like multi-image sequences or mixed image-video retrieval.
- **Diagnostic, Not Prescriptive**: While the paper provides empirical rules (e.g., input order), it does not propose a new architecture to bridge the human-AI gap.
- **Sparse Sampling Bias**: Since videos are >5 minutes, sampling only 32 frames may naturally miss key information, partially affecting the sensitivity conclusions regarding frame counts.

## Related Work & Insights
- **Text-only Video Benchmarks** (Video-MME, MVBench, etc.): Established the paradigm for video evaluation but lack image-grounded capabilities—IV-Bench fills this gap.
- **Image-Text Video Benchmarks** (V2P-Bench, VideoRefer-Bench): Introduced image queries but relied on in-video frames (frame matching); IV-Bench resolves this with external sourcing.
- **Insight**: For researchers in multimodal evaluation, the "Counterfactual Distractors + Mandatory Dual-Modality" framework is a reusable paradigm for preventing data leakage. For practitioners, the finding that "image after video frames" and "prioritizing frames over resolution" serves as an immediate engineering default.

## Rating
- Novelty: ⭐⭐⭐⭐ First external image-grounded benchmark; counterfactual distractors provide genuine methodological advancement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across 28 models and multi-dimensional ablations (scale, order, frames).
- Writing Quality: ⭐⭐⭐⭐ Clear motivation supported by real-world visual search data; well-defined tasks and systematic comparisons.
- Value: ⭐⭐⭐⭐ Exposes a massive blind spot (28.9% vs 88.8%) and provides actionable engineering guidelines for both evaluation and modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](mmr-v_whats_left_unsaid_a_benchmark_for_multimodal_deep_reasoning_in_videos.md)
- [\[CVPR 2026\] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](../../CVPR2026/vlm_reasoning/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)
- [\[ICLR 2026\] GIR-Bench: Versatile Benchmark for Generating Images with Reasoning](gir-bench_versatile_benchmark_for_generating_images_with_reasoning.md)
- [\[ICLR 2026\] VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?](videoreasonbench_can_mllms_perform_vision-centric_complex_video_reasoning.md)
- [\[ICLR 2026\] ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](expvid_a_benchmark_for_experiment_video_understanding_reasoning.md)

</div>

<!-- RELATED:END -->
