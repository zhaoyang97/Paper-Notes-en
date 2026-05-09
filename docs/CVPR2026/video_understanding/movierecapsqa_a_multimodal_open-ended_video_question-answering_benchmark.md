---
title: >-
  [Paper Note] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark
description: >-
  [CVPR 2026][Video Understanding][Video Question Answering] This paper introduces MovieRecapsQA, a multimodal open-ended video QA benchmark constructed from movie recap videos, comprising approximately 8.2K questions across 60 movies. It proposes a reference-free evaluation metric based on atomic facts and reveals that the critical bottleneck of current MLLMs lies in visual perception rather than reasoning.
tags:
  - CVPR 2026
  - Video Understanding
  - Video Question Answering
  - Multimodal Understanding
  - Open-Ended Evaluation
  - Movie Understanding
  - Reference-Free Evaluation
date: 2026-05-08
content_hash: b28d6b1f957afa3e
---

# MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark

**Conference**: CVPR 2026
**arXiv**: [2601.02536](https://arxiv.org/abs/2601.02536)
**Code**: [MovieRecapsQA](https://github.com/MovieRecapsQA) (publicly available)
**Area**: Video Understanding
**Keywords**: Video Question Answering, Multimodal Understanding, Open-Ended Evaluation, Movie Understanding, Reference-Free Evaluation

## TL;DR

This paper introduces MovieRecapsQA, a multimodal open-ended video QA benchmark constructed from movie recap videos, comprising approximately 8.2K questions across 60 movies. It proposes a reference-free evaluation metric based on atomic facts and reveals that the critical bottleneck of current MLLMs lies in visual perception rather than reasoning.

## Background & Motivation

1. **Background**: Video Question Answering (VideoQA) is a core proxy task for evaluating models' video understanding capabilities. Existing benchmarks primarily focus on single-modality or short videos and rely heavily on multiple-choice formats to simplify evaluation complexity. Multimodal long-video QA benchmarks that genuinely require integrating both visual and dialogue cues remain scarce.

2. **Limitations of Prior Work**: (a) Multiple-choice questions provide a "shortcut" — models can answer by elimination without understanding the video; (b) Open-ended QA is extremely difficult to evaluate due to non-fixed answer formats; (c) Reference-based metrics (e.g., ROUGE, BERTScore) exhibit low correlation with human judgments; (d) Using LLMs as judges for VideoQA evaluation with full videos as context is both expensive and unreliable.

3. **Key Challenge**: The fundamental tension between open-ended evaluation and measurability — multiple-choice questions are easy to evaluate but lack authenticity, while open-ended questions are authentic but cannot be reliably assessed.

4. **Goal**: (a) How to construct a high-quality multimodal long-video open-ended QA dataset? (b) How to reliably evaluate open-ended responses without relying on reference answers?

5. **Key Insight**: Leveraging movie recap videos as the data source — narrations in recap videos naturally provide textual summaries of video content, enabling automatic extraction of atomic facts to support reference-free evaluation.

6. **Core Idea**: Atomic facts extracted from recap video narrations serve as an intermediate annotation layer, simultaneously supporting the generation of questions requiring multimodal reasoning and enabling reference-free evaluation of response factuality and relevance.

## Method

### Overall Architecture

The system consists of two major components: (1) **Dataset Construction** — collecting recap videos for 60 movies from YouTube, performing scene alignment, and automatically generating QA pairs and atomic facts; (2) **Evaluation Framework** — designing a reference-free LLM judge based on atomic facts that assesses open-ended responses along the dimensions of factuality and relevance.

### Key Designs

1. **Recap-Movie Alignment**:

    - **Function**: Precisely aligning each segment of a recap video to the corresponding scene and dialogue in the original movie.
    - **Mechanism**: Movie and recap videos are first segmented into scenes using SceneDetect; visual embeddings of the first and last frames of each scene are extracted via SlowFast and matched by cosine similarity; a statistical alignment step then enforces a semi-temporal ordering. This yields not only video-to-video alignment but also a three-way alignment among narration, movie footage, and dialogue.
    - **Design Motivation**: Recap videos naturally couple narration with visual clips at high density at the scene level — denser than Wikipedia synopses or IMDb plot summaries — enabling questions to be precisely grounded to specific temporal segments of the film.

2. **Fact-Based QA Generation**:

    - **Function**: Automatically generating high-quality QA pairs from recap narrations.
    - **Mechanism**: Each recap segment is fed to GPT-4.1, which first extracts all atomic facts (concise, verifiable propositions) and then generates QA pairs dependent on these facts. To prevent overly detailed answers from making questions trivial, simplified QA pairs are additionally generated using simplified questions paired with detailed answers. Each question is annotated with the required modality type (visual, dialogue, or both).
    - **Design Motivation**: Atomic facts as an intermediate layer serve three purposes: (a) ensuring generated questions genuinely require multimodal reasoning; (b) providing precise textual representations of video content for evaluation; (c) eliminating the need for manually written reference answers.

3. **Reference-Free Evaluation**:

    - **Function**: Evaluating the factuality and relevance of model responses without relying on reference answers.
    - **Mechanism**: For each question $q$, the associated atomic fact set $\mathcal{F}_q$ is collected, and claims $\mathcal{C}_r$ are extracted from the model response. GPT-4.1-mini serves as the LLM judge, scoring responses on factuality (0–5) and relevance (0–5) based on the question, atomic facts, and subtitles. This avoids the high computational cost and unreliability of feeding full videos to the judge.
    - **Design Motivation**: LLM judges from text QA cannot be directly transferred to VideoQA, as using full videos as verification context is both expensive and imprecise. Atomic facts provide compact, verifiable textual proxies, making reference-free evaluation feasible.

### Loss & Training

This paper presents a dataset and evaluation framework; no model training is involved. All components — fact extraction, QA generation, and evaluation judging — are constructed using GPT-4.1.

## Key Experimental Results

### Main Results

| Model | ROUGE-L | BERTScore | HELMET Correct. | Factuality (Ours) | Relevance (Ours) |
|---|---|---|---|---|---|
| GPT-4o | 0.28 | 0.68 | 1.43 | **3.99** | **3.97** |
| Gemini-2.5-Flash | 0.22 | 0.63 | 1.82 | 3.26 | 3.70 |
| Claude 3.5 Sonnet | 0.22 | 0.63 | 1.35 | 3.76 | 3.92 |
| Amazon Nova Lite | 0.28 | 0.69 | 1.29 | 3.53 | 3.93 |
| Qwen2.5VL | 0.26 | 0.67 | 1.23 | 3.47 | 3.83 |
| MiniCPM-o | 0.24 | 0.65 | 1.27 | 3.21 | 3.61 |
| LLaVA-NeXT-Video | 0.23 | 0.65 | 0.98 | 2.96 | 3.35 |
| Human (Average) | 0.16 | 0.88 | 0.98 | 4.01 | 4.01 |
| Human (Best) | 0.19 | 0.87 | 1.26 | 4.59 | 4.53 |

### Ablation Study (Breakdown by Modality Type)

| Modality Type | Closed-Source Factuality | Open-Source Factuality | Human Factuality |
|---|---|---|---|
| Dialogue | 3.63 | 3.21 | 4.17 |
| Visual | 3.15 | 3.05 | 3.84 |
| Multimodal | 3.55 | 3.11 | 3.84 |

### Key Findings

- **Semantic metrics fail entirely**: ROUGE-L ranges only from 0.22 to 0.28 and BERTScore from 0.63 to 0.69, providing almost no discrimination among models and even ranking humans below models.
- **Reference-based metrics are counter-intuitive**: HELMET Correctness rates MiniCPM-o (1.27) higher than the best human (1.26), which is entirely unintuitive.
- **The proposed reference-free metrics are most discriminative**: Factuality spans from 2.96 to 3.99, forming a reasonable gap relative to human scores (4.01/4.59).
- **Visual perception is the primary bottleneck**: All models achieve the lowest factuality scores on visual questions, and removing visual input actually improves factuality for closed-source models, indicating that visual inputs introduce erroneous information.
- **Models know where to look but not what to say**: Relevance scores remain stable across modality types while factuality fluctuates substantially, suggesting adequate localization ability but insufficient fine-grained visual information extraction.

## Highlights & Insights

- The design of **atomic facts as an intermediate annotation layer** is elegant: it simultaneously addresses both "how to generate good questions" and "how to evaluate answers." Atomic facts are more flexible than reference answers — the same fact can be expressed in multiple ways, avoiding the rigidity of reference-based evaluation.
- The finding that **"removing visual input actually improves factuality"** is particularly insightful: it reveals that current MLLMs fail not at "reasoning" about visual information, but at "perceiving" it — the perceived information is erroneous, so subsequent reasoning is inevitably flawed.
- The use of **movie recap videos as a data source** has strong scalability: YouTube hosts a large volume of such content with natural video-text alignment, and the approach is transferable to other narrated video types such as educational videos and sports commentary.

## Limitations & Future Work

- Data sourced from YouTube recap videos may contain subjective biases and omissions introduced by the narrators.
- The benchmark covers only 60 movies, limiting scale, and the distribution of film genres is not reported in detail.
- Atomic fact extraction and QA generation rely entirely on GPT-4.1, potentially introducing biases inherent to large language models.
- The evaluation judge uses GPT-4.1-mini to reduce costs, but its judging capability may be inferior to larger models.
- Systematic experiments on longer input settings (e.g., full movies) are absent.

## Related Work & Insights

- **vs. MovieQA / TVQA**: These classic benchmarks employ multiple-choice questions and rely on manual annotation, limiting scale. This paper adopts open-ended QA with automatic construction and introduces modality annotation and reference-free evaluation.
- **vs. CinePile**: CinePile is also a large-scale automatically generated benchmark (303K QA pairs) but still uses multiple-choice questions and lacks modality-level breakdown. Although smaller in scale (8.2K), the proposed benchmark is more advanced in evaluation design.
- **vs. FactScore / VeriScore**: These factuality evaluation works in text QA inspired the present design, but this paper is the first to extend atomic-fact-based evaluation to the VideoQA domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of recap-video-based benchmark construction and reference-free evaluation is novel, though the core techniques (LLM-based fact extraction and judging) are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation covers 7 models plus human assessment, multiple metric comparisons, and detailed breakdown analyses by modality and reasoning type.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is logically clear with naturally motivated contributions and precise, insightful presentation of experimental findings.
- **Value**: ⭐⭐⭐⭐ Provides an important evaluation tool for long-video multimodal understanding; the finding that visual perception is the bottleneck offers meaningful guidance for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[CVPR 2026\] EgoPointVQA: Gesture-Based Egocentric Video Question Answering](egopointvqa_gesture_based_egocentric_video_qa.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/video_understanding/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)

</div>

<!-- RELATED:END -->
