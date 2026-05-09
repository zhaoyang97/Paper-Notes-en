---
title: >-
  [Paper Note] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence
description: >-
  [NeurIPS 2025][Video Understanding][Video Retrieval] This paper introduces MUVR, a benchmark for multi-modal untrimmed video retrieval targeting real-world long-video platforms. It proposes a video-centric multi-modal query format (video + text + tag + mask) and a six-level visual correspondence matching criterion, comprising 53K videos and 1,050 queries, and systematically evaluates the limitations of retrieval models and MLLMs.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Video Retrieval
  - Multi-Modal Query
  - Untrimmed Video
  - Multi-Level Visual Correspondence
  - Benchmark Dataset
date: 2026-05-08
content_hash: 0ba1dd70b2ef019f
---

# MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence

**Conference**: NeurIPS 2025
**arXiv**: [2510.21406](https://arxiv.org/abs/2510.21406)
**Code**: [GitHub](https://github.com/debby-0527/MUVR)
**Area**: Video Understanding
**Keywords**: Video Retrieval, Multi-Modal Query, Untrimmed Video, Multi-Level Visual Correspondence, Benchmark Dataset

## TL;DR

This paper introduces MUVR, a benchmark for multi-modal untrimmed video retrieval targeting real-world long-video platforms. It proposes a video-centric multi-modal query format (video + text + tag + mask) and a six-level visual correspondence matching criterion, comprising 53K videos and 1,050 queries, and systematically evaluates the limitations of retrieval models and MLLMs.

## Background & Motivation

**State of the Field**: Video retrieval is a core technology in recommendation systems and content search. Existing tasks include text-to-video retrieval (TVR), composed video retrieval (CVR), and fine-grained video retrieval (FVR), each with distinct limitations.

**Limitations of Prior Work**: (1) Pure text queries fail to capture fine-grained visual details; (2) pure video queries introduce irrelevant visual information; (3) existing benchmarks support only a single matching criterion (e.g., near-duplicate or event-level), failing to cover diverse video categories; (4) most benchmarks are built on trimmed videos with one-to-one retrieval, which does not reflect real-world platform scenarios.

**Root Cause**: Real-world video platforms require one-to-many retrieval over untrimmed videos with multi-modal queries and broad category coverage, yet no existing benchmark satisfies all these requirements simultaneously.

**Paper Goals**: To construct a comprehensive video retrieval benchmark aligned with the demands of real-world video platforms.

**Starting Point**: The proposed approach uses video as the primary query modality supplemented by textual descriptions, combined with tag and mask prompts, for one-to-many retrieval over an untrimmed video corpus.

**Core Idea**: Multi-level visual correspondence (copy, event, scene, instance, action, others) is proposed as a general retrieval matching criterion, and five partitions are designed to cover diverse video categories.

## Method

### Overall Architecture

The MUVR benchmark comprises three versions:
- **MUVR-Base**: 53K videos, 1,050 multi-modal queries, and 84K annotated matches.
- **MUVR-Filter**: Built upon MUVR-Base with 74K multi-label annotations to support tag-filtered retrieval (9,979 queries).
- **MUVR-QA**: 200 discriminative questions for evaluating MLLM reranking capability.

### Key Designs

1. **Multi-Modal Query Design**: Each query consists of four components:

    - **Video Query**: Carries primary visual information, suited for expressing details that are difficult to describe in text.
    - **Text Description**: Averages 20 words, specifying key visual content and retrieval intent.
    - **Tag Prompt**: User-specified desired/undesired video attributes (e.g., "animated style," "first-person view") to enable fine-grained filtering.
    - **Mask Prompt**: Key regions annotated using SAM2 to guide retrieval models toward specific parts of the query video.

2. **Multi-Level Visual Correspondence**: Six correspondence levels are defined according to the granularity of visual content the user seeks:

    - **Copy**: The target is a copy or edited version of the query.
    - **Event**: Shares the same event (spatiotemporal overlap).
    - **Scene**: Shares the same scene, background, or location.
    - **Instance**: Shares the same instance or object.
    - **Action**: Shares the same human action.
    - **Others**: Subjectively perceived relevance.

3. **Five-Partition Design**: News (news events), Region (travel scenes), Instance (products and pets), Dance (dance actions), and Others (memes/films), each emphasizing different visual correspondence levels and video categories.

4. **Reranking Score**: An evaluation metric designed for MLLM reranking. Each query contains one true positive and one false positive; an MLLM can produce four outcomes:

    - 10 (correctly retains true positive and removes false positive): +1
    - 11 (retains both; no action): 0
    - 00 (removes both): −1
    - 01 (incorrect inversion): −2

### Loss & Training

MUVR is an evaluation benchmark rather than a training method. Retrieval scores are computed as:

$$S_v = \text{Score}(V_{\text{query}}, V_{\text{target}}), \quad S_t = \text{Score}(T_{\text{description}}, V_{\text{target}})$$
$$S_{tv} = (S_t + S_v)/2, \quad S_{tag} = S_{tv} + p \times \text{Score}(T_{\text{tag}}, V_{\text{target}})$$

where $p = \pm 0.3$ is determined by the sign of the tag prompt (positive or negative).

## Key Experimental Results

### Main Results

**Multi-Modal Query Retrieval Performance on MUVR-Base (mAP %):**

| Method | Avg. mAP | News | Others | Instance | Region | Dance |
|--------|----------|------|--------|----------|--------|-------|
| CLIP (RN50x4) | 42.9 | 49.4 | 53.6 | 46.5 | 43.8 | 21.2 |
| EVA-CLIP | **58.0** | **63.1** | 66.1 | **68.2** | **63.8** | 28.7 |
| InternVideo2 | 52.1 | 57.3 | **66.3** | 55.3 | 52.5 | 28.9 |
| S2VS | 47.2 | 51.3 | 63.7 | 49.5 | 49.1 | 22.5 |
| CoVR | 43.3 | 50.5 | 54.3 | 46.9 | 44.0 | 20.8 |

**MLLM Reranking Evaluation on MUVR-QA:**

| Method | Params | Frames | Accuracy (All) | Reranking Score (All) |
|--------|--------|--------|---------------|----------------------|
| InternVL2 (multi-image) | 8B | 6 | 58.5 | −0.23 |
| GPT-4o (multi-image) | N/A | 6 | **65.0** | **0.19** |
| Gemini-2.0-Flash | N/A | 12 | 63.5 | 0.07 |
| InternVL2.5 (text) | 8B | 12 | 58.5 | −0.37 |

### Ablation Study

**Effect of Query Format on Retrieval Performance (EVA-CLIP, mAP %):**

| Query Format | Avg. mAP | Best Partition |
|--------------|----------|----------------|
| Text only | 43.0 | Instance (59.7) |
| Video only | 50.7 | Others (59.2) |
| Multi-modal | **58.0** | Instance (68.2) |
| Multi-modal + Tag | 34.0 (Filter) | News (38.3) |

### Key Findings

- **Video queries outperform text queries**: Video-only retrieval consistently surpasses text-only retrieval (50.7 vs. 43.0 mAP), as videos more precisely convey details that are difficult to express in words.
- **Multi-modal queries yield substantial gains**: Combining video and text achieves a 7.3% mAP improvement (58.0 vs. 50.7), demonstrating the complementarity of the two modalities.
- **Different partitions require different capabilities**: VLMs perform best on the Instance/Region/News partitions (static spatial understanding), while video models are stronger on Dance/Others (dynamic temporal understanding).
- **All methods perform worst on the Dance partition**: The highest mAP reaches only 28.9%, indicating that action-level visual correspondence remains a significant challenge.
- **Tag prompts pose substantial challenges for MLLMs**: Some models achieve 70%+ accuracy on tag-free questions but degrade noticeably when tags are introduced.
- **MLLM reranking is unreliable**: Most models achieve negative reranking scores, indicating that current MLLMs are not yet dependable for video reranking.

## Highlights & Insights

- **Well-designed and practically motivated benchmark**: The five-partition design covers core categories on video platforms, avoiding the bias of single-category benchmarks.
- **Multi-level visual correspondence is a valuable formalization**: Abstracting user retrieval intent into six granularity levels provides more precision than simple semantic matching.
- **Tag prompts are a novel contribution**: Enabling users to filter results with simple tags closely mirrors real-world product experiences.
- **Reranking score design is well-justified**: Distinguishing "no action" from "incorrect inversion" with different penalties reflects the practical requirements of reranking scenarios.
- The benchmark is large in scale (53K videos) and high in annotation quality (professional annotators with two-round verification).

## Limitations & Future Work

- Data is sourced exclusively from Bilibili, potentially introducing a Chinese-video bias; generalizability to English-language platforms remains to be validated.
- Videos are clipped to a maximum of 2 minutes, providing insufficient coverage of retrieval scenarios involving longer content.
- The tag prompt weighting scheme ($p = \pm 0.3$ fixed) is relatively simple; adaptive weighting strategies could be explored.
- MUVR-QA contains only 200 questions, limiting its scale.
- End-to-end multi-modal query understanding models are not explored; only existing models are evaluated.

## Related Work & Insights

- Compared to FIVR (event-level retrieval), MUVR extends the framework to six levels of visual correspondence across multiple video categories.
- Compared to CoVR (composed retrieval), MUVR supports one-to-many retrieval over untrimmed videos.
- The extremely low performance on the Dance partition motivates future work on action-level video understanding.
- The unreliability of MLLM reranking motivates further research into two-stage retrieval + MLLM reranking pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-level visual correspondence and multi-modal query paradigm are novel, though the primary contribution is a dataset rather than a methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic evaluation across 3 retrieval models, 6 VLMs, and 10 MLLMs, with comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Benchmark design is described in detail, findings are clearly summarized, and readability is excellent.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in benchmarks for multi-modal untrimmed video retrieval, making a significant contribution to the community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](../../ICCV2025/video_understanding/multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](../../ICCV2025/video_understanding/dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)

<!-- RELATED:END -->
